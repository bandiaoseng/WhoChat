from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from whochat.config import HistoricalSchedulerConfig
from whochat.services.historical_eligibility import (
    HistoricalActivityMetrics,
    HistoricalBackfillEligibility,
    HistoricalBackfillPriorityScorer,
)
from whochat.storage.database import Database
from whochat.storage.historical import HistoricalArchiveRepository
from whochat.storage.historical_scheduler import HistoricalSchedulerRepository


TERMINAL_JOB_STATES = frozenset({"completed", "no_memory"})
TRANSPORT_OUTCOMES = frozenset({"timeout", "transport_error"})
SYSTEMIC_TECHNICAL_REASONS = frozenset({
    "systemic_invalid_output", "systemic_source_reference_error", "mapping_error",
    "cross_account_risk", "source_evidence_incomplete", "database_integrity_error",
})


@dataclass(frozen=True)
class IdentityDecision:
    status: str
    reason: str
    mapping_id: str
    account_scope_id: str
    contact_id: str | None
    conversation_id: str | None
    basis: str
    evidence: dict[str, Any]

    @property
    def safe(self) -> bool:
        return self.status in {"resolved_existing", "auto_resolvable"}


class DeterministicHistoricalIdentityMapper:
    """Stable account/external/conversation identity; nickname never selects."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.archive = HistoricalArchiveRepository(db)

    def assess(self, *, mapping_id: str, safe_contact_reference: str) -> IdentityDecision:
        with self.db.connect() as conn:
            mapping = conn.execute(
                "SELECT * FROM historical_contact_mappings WHERE id=?", (mapping_id,)
            ).fetchone()
            if mapping is None:
                return self._blocked(mapping_id, "", "mapping_missing")
            scope = str(mapping["account_scope_id"])
            ext_contact = str(mapping["external_contact_id"] or "")
            ext_conversation = str(mapping["external_conversation_id"] or "")
            if not ext_contact or not ext_conversation:
                return self._blocked(mapping_id, scope, "stable_external_identity_missing")
            pair_rows = int(conn.execute(
                """SELECT COUNT(*) FROM historical_contact_mappings WHERE account_scope_id=?
                   AND external_contact_id=? AND external_conversation_id=?""",
                (scope, ext_contact, ext_conversation),
            ).fetchone()[0])
            ev = conn.execute(
                """SELECT COUNT(*) total,
                   SUM(CASE WHEN account_scope_id=? AND external_contact_id=?
                             AND external_conversation_id=? THEN 1 ELSE 0 END) exact_rows,
                   COUNT(DISTINCT account_scope_id) scopes,
                   COUNT(DISTINCT external_contact_id) contacts,
                   COUNT(DISTINCT external_conversation_id) conversations
                   FROM historical_chat_messages WHERE mapping_id=?""",
                (scope, ext_contact, ext_conversation, mapping_id),
            ).fetchone()
            evidence = {
                "stable_pair_rows": pair_rows,
                "historical_rows": int(ev["total"] or 0),
                "exact_evidence_rows": int(ev["exact_rows"] or 0),
                "distinct_scopes": int(ev["scopes"] or 0),
                "distinct_external_contacts": int(ev["contacts"] or 0),
                "distinct_external_conversations": int(ev["conversations"] or 0),
                "nickname_used_for_resolution": False,
            }
            if pair_rows != 1:
                return self._blocked(mapping_id, scope, "stable_identity_not_unique", evidence)
            if (evidence["historical_rows"] <= 0
                    or evidence["exact_evidence_rows"] != evidence["historical_rows"]
                    or evidence["distinct_scopes"] != 1
                    or evidence["distinct_external_contacts"] != 1
                    or evidence["distinct_external_conversations"] != 1):
                return self._blocked(mapping_id, scope, "conversation_evidence_not_exact", evidence)

            if mapping["status"] == "resolved":
                contact_id, conversation_id = str(mapping["contact_id"] or ""), str(mapping["conversation_id"] or "")
                local = conn.execute(
                    """SELECT c.account_scope_id contact_scope,c.merged_into,
                              v.account_scope_id conversation_scope,v.contact_id conversation_contact,
                              v.identity_key FROM contacts c JOIN conversations v ON v.id=? WHERE c.id=?""",
                    (conversation_id, contact_id),
                ).fetchone()
                expected_key = f"historical:{safe_contact_reference}"
                local_scope_exact = bool(
                    local and local["contact_scope"] == scope
                    and local["conversation_scope"] == scope
                    and local["conversation_contact"] == contact_id
                    and local["merged_into"] is None
                )
                basis = str(mapping["match_basis"] or "")
                # The two Pilot identities predate the historical:<safe-ref>
                # conversation key.  They were explicitly approved by the user
                # and are already terminal.  Replaying those immutable mappings
                # may trust that explicit approval only after the stable external
                # pair and every historical row have passed the exact checks
                # above.  This branch never resolves a new/unresolved mapping and
                # never uses the nickname as an identity selector.
                legacy_explicit = basis.startswith("explicit_user_named_pilot")
                exact = bool(local_scope_exact and (
                    local["identity_key"] == expected_key
                    or (legacy_explicit and bool(local["identity_key"]))
                ))
                evidence["local_identity_exact"] = exact
                evidence["legacy_explicit_replay"] = legacy_explicit
                evidence["nickname_used_for_resolution"] = False
                if not exact:
                    return self._blocked(mapping_id, scope, "resolved_local_identity_mismatch", evidence)
                return IdentityDecision(
                    "resolved_existing",
                    ("legacy_explicit_mapping_with_exact_scope_and_conversation_evidence"
                     if legacy_explicit else "exact_stable_identity_and_conversation_evidence"),
                    mapping_id, scope, contact_id, conversation_id,
                    str(mapping["match_basis"] or ""), evidence,
                )
            if mapping["status"] != "unresolved":
                return self._blocked(mapping_id, scope, f"mapping_status_{mapping['status']}", evidence)
            display_name = str(mapping["source_display_name"] or "").strip()
            if not display_name:
                return self._blocked(mapping_id, scope, "display_name_missing", evidence)
            collision = int(conn.execute(
                """SELECT COUNT(DISTINCT c.id) FROM contacts c
                   LEFT JOIN contact_aliases a ON a.contact_id=c.id
                   WHERE c.account_scope_id=? AND c.platform='wechat' AND c.merged_into IS NULL
                     AND (c.display_name=? OR a.alias=?)""",
                (scope, display_name, display_name),
            ).fetchone()[0])
            evidence["same_scope_display_collision"] = collision
            if collision:
                return self._blocked(mapping_id, scope, "same_scope_display_collision", evidence)
            return IdentityDecision(
                "auto_resolvable", "unique_stable_external_identity_with_exact_conversation_evidence",
                mapping_id, scope, None, None, "deterministic_external_identity_long_running_auto", evidence,
            )

    def resolve(self, *, mapping_id: str, safe_contact_reference: str) -> IdentityDecision:
        decision = self.assess(mapping_id=mapping_id, safe_contact_reference=safe_contact_reference)
        if not decision.safe or decision.status == "resolved_existing":
            return decision
        with self.db.connect() as conn:
            mapping = conn.execute("SELECT * FROM historical_contact_mappings WHERE id=?", (mapping_id,)).fetchone()
        if mapping is None:
            return self._blocked(mapping_id, "", "mapping_disappeared")
        self.archive.enroll_explicit_local_identity(
            mapping_id=mapping_id,
            expected_account_scope_id=str(mapping["account_scope_id"]),
            expected_external_contact_id=str(mapping["external_contact_id"]),
            expected_external_conversation_id=str(mapping["external_conversation_id"]),
            safe_contact_reference=safe_contact_reference,
            authorization_batch="long_running_auto",
            match_basis_prefix="deterministic_external_identity",
        )
        refreshed = self.assess(mapping_id=mapping_id, safe_contact_reference=safe_contact_reference)
        if refreshed.status != "resolved_existing":
            raise RuntimeError("historical_auto_identity_postcheck_failed")
        return refreshed

    @staticmethod
    def _blocked(mapping_id: str, scope: str, reason: str,
                 evidence: dict[str, Any] | None = None) -> IdentityDecision:
        return IdentityDecision("needs_review", reason, mapping_id, scope, None, None, "",
                                evidence or {"nickname_used_for_resolution": False})


@dataclass(frozen=True)
class SchedulerCandidate:
    safe_contact_reference: str
    mapping_id: str
    account_scope_id: str
    display_name: str
    value_tier: str
    expected_chunks: int
    priority_score: int
    priority_components: tuple[str, ...]
    identity: IdentityDecision
    history_message_count: int
    effective_message_count: int


def select_bounded_batch(candidates: Iterable[SchedulerCandidate], *, max_customers: int = 20,
                         target_chunks_min: int = 70, target_chunks_max: int = 80,
                         search_pool: int = 240) -> list[SchedulerCandidate]:
    if not 1 <= max_customers <= 20:
        raise ValueError("historical_scheduler_max_customers_invalid")
    if not 1 <= target_chunks_min <= target_chunks_max <= 80:
        raise ValueError("historical_scheduler_chunk_target_invalid")
    ranked = sorted((item for item in candidates if item.identity.safe
                     and item.value_tier in {"A", "B"}
                     and 0 < item.expected_chunks <= target_chunks_max),
                    key=lambda item: (item.priority_score, item.value_tier == "A",
                                      item.effective_message_count, item.safe_contact_reference),
                    reverse=True)[:max(search_pool, max_customers)]
    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {(0, 0): (0, ())}
    for index, item in enumerate(ranked):
        updated = dict(states)
        for (count, chunks), (score, indexes) in states.items():
            if count >= max_customers or chunks + item.expected_chunks > target_chunks_max:
                continue
            key = count + 1, chunks + item.expected_chunks
            value = score + item.priority_score, indexes + (index,)
            if key not in updated or value[0] > updated[key][0]:
                updated[key] = value
        states = updated
    valid = [(count, chunks, score, indexes)
             for (count, chunks), (score, indexes) in states.items()
             if target_chunks_min <= chunks <= target_chunks_max]
    if not valid:
        raise RuntimeError("historical_scheduler_cannot_reach_chunk_target")
    count, chunks, _, indexes = max(valid, key=lambda item: (item[2], item[1], -item[0]))
    selected = [ranked[index] for index in indexes]
    if len(selected) != count or sum(item.expected_chunks for item in selected) != chunks:
        raise AssertionError("historical_scheduler_selection_corrupt")
    return selected


class RealtimePriorityGate:
    def __init__(self, db: Database) -> None:
        self.db = db

    def snapshot(self) -> dict[str, int]:
        with self.db.connect() as conn:
            reply = int(conn.execute(
                "SELECT COUNT(*) FROM ai_analysis_events WHERE status IN ('pending','running')"
            ).fetchone()[0])
            memory = int(conn.execute(
                "SELECT COUNT(*) FROM memory_extraction_jobs WHERE status IN ('pending','running')"
            ).fetchone()[0])
        return {"reply": reply, "memory": memory, "historical": 0}

    def may_claim_historical_chunk(self) -> tuple[bool, str]:
        state = self.snapshot()
        if state["reply"]:
            return False, "yield_to_reply"
        if state["memory"]:
            return False, "yield_to_memory"
        return True, "historical_lane_available"


class AllowedWindow:
    def __init__(self, start_time: str, end_time: str) -> None:
        self.start, self.end = self._parse(start_time), self._parse(end_time)

    @staticmethod
    def _parse(value: str) -> time:
        try:
            return time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"historical_scheduler_time_invalid:{value}") from exc

    def contains(self, local_now: datetime) -> bool:
        current = local_now.time().replace(tzinfo=None)
        if self.start == self.end:
            return True
        if self.start < self.end:
            return self.start <= current < self.end
        return current >= self.start or current < self.end


@dataclass(frozen=True)
class QualityBaseline:
    rejection_rates: tuple[float, ...]
    context_rejection_rates: tuple[float, ...]
    rejection_upper_fence: float
    context_upper_fence: float
    source_evidence_required: float = 1.0

    @classmethod
    def from_observations(cls, *, rejection_rates: Iterable[float],
                          context_rejection_rates: Iterable[float]) -> "QualityBaseline":
        rejection, context = tuple(map(float, rejection_rates)), tuple(map(float, context_rejection_rates))
        if len(rejection) < 3 or len(context) < 3:
            raise ValueError("historical_quality_baseline_requires_three_batches")
        return cls(rejection, context, _upper_fence(rejection), _upper_fence(context))


def _upper_fence(values: Iterable[float]) -> float:
    ordered = sorted(max(0.0, min(1.0, float(value))) for value in values)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return min(1.0, q3 + 1.5 * (q3 - q1))


@dataclass(frozen=True)
class BreakerDecision:
    tripped: bool
    kind: str
    reason: str
    evidence: dict[str, Any]


class TechnicalCircuitBreaker:
    def __init__(self, consecutive_transport_limit: int = 2) -> None:
        self.limit, self.consecutive_transport = max(1, consecutive_transport_limit), 0

    def observe(self, outcome: str, *, reason: str = "") -> BreakerDecision:
        if reason in SYSTEMIC_TECHNICAL_REASONS:
            return BreakerDecision(True, "technical", reason, {"outcome": outcome})
        self.consecutive_transport = self.consecutive_transport + 1 if outcome in TRANSPORT_OUTCOMES else 0
        tripped = self.consecutive_transport >= self.limit
        return BreakerDecision(tripped, "technical",
                               "consecutive_provider_transport_errors" if tripped else "",
                               {"consecutive_transport_errors": self.consecutive_transport,
                                "outcome": outcome})


class QualityCircuitBreaker:
    def __init__(self, baseline: QualityBaseline) -> None:
        self.baseline = baseline

    def evaluate(self, *, accepted: int, rejected: int, context_rejected: int,
                 source_evidence_ratio: float) -> BreakerDecision:
        total = max(0, accepted) + max(0, rejected)
        rejection_rate = rejected / total if total else 0.0
        context_rate = context_rejected / total if total else 0.0
        evidence = {"accepted": accepted, "rejected": rejected,
                    "rejection_rate": rejection_rate,
                    "rejection_upper_fence": self.baseline.rejection_upper_fence,
                    "context_rejection_rate": context_rate,
                    "context_upper_fence": self.baseline.context_upper_fence,
                    "source_evidence_ratio": source_evidence_ratio,
                    "source_evidence_required": self.baseline.source_evidence_required}
        if source_evidence_ratio < self.baseline.source_evidence_required:
            return BreakerDecision(True, "quality", "source_evidence_below_100_percent", evidence)
        if total and rejection_rate > self.baseline.rejection_upper_fence:
            return BreakerDecision(True, "quality", "rejection_rate_above_stable_upper_fence", evidence)
        if total and context_rate > self.baseline.context_upper_fence:
            return BreakerDecision(True, "quality", "context_rejection_above_stable_upper_fence", evidence)
        return BreakerDecision(False, "quality", "", evidence)


@dataclass(frozen=True)
class SchedulerGateResult:
    allowed: bool
    reason: str
    details: dict[str, Any]


class SchedulerRuntimeGate:
    def __init__(self, *, repository: HistoricalSchedulerRepository,
                 realtime: RealtimePriorityGate, config: HistoricalSchedulerConfig) -> None:
        self.repository, self.realtime, self.config = repository, realtime, config
        self.window = AllowedWindow(config.start_time, config.end_time)

    def check_before_claim(self, *, local_now: datetime, calls_today: int,
                           batches_today: int, elapsed_seconds: float,
                           batch_calls_used: int, batch_call_budget: int) -> SchedulerGateResult:
        control = self.repository.control()
        details = {"control": control, "calls_today": calls_today,
                   "daily_cap": self.config.daily_historical_call_cap,
                   "batches_today": batches_today, "batch_calls_used": batch_calls_used,
                   "batch_call_budget": batch_call_budget, "elapsed_seconds": elapsed_seconds}
        checks = [
            (not self.config.enabled or not int(control["enabled"]), "disabled"),
            (bool(control["manual_paused"]), "manual_pause"),
            (bool(control["automatic_paused"]), "automatic_pause"),
            (not self.window.contains(local_now), "outside_allowed_window"),
            (batches_today >= self.config.max_batches_per_day, "daily_batch_cap"),
            (calls_today >= self.config.daily_historical_call_cap, "daily_provider_call_cap"),
            (batch_calls_used >= batch_call_budget, "batch_provider_call_cap"),
            (elapsed_seconds >= self.config.max_runtime_seconds, "batch_runtime_cap"),
        ]
        for blocked, reason in checks:
            if blocked:
                return SchedulerGateResult(False, reason, details)
        allowed, reason = self.realtime.may_claim_historical_chunk()
        details["realtime"] = self.realtime.snapshot()
        return SchedulerGateResult(allowed, reason if not allowed else "allowed", details)


class LongRunningHistoricalBackfillScheduler:
    """Small production facade over the persistent v33 scheduling boundary.

    Historical extraction remains in ``HistoricalMemoryBackfillService``.  The
    facade deliberately owns only admission, priority, pause and accounting so
    daily Reply/Memory paths never import or depend on the migration runner.
    """

    def __init__(self, database: Database, config: HistoricalSchedulerConfig) -> None:
        self.database = database
        self.config = config
        self.repository = HistoricalSchedulerRepository(database)
        self.realtime = RealtimePriorityGate(database)
        self.runtime_gate = SchedulerRuntimeGate(
            repository=self.repository, realtime=self.realtime, config=config
        )

    def check_before_claim(self, *, local_now: datetime, calls_today: int,
                           batches_today: int, elapsed_seconds: float,
                           batch_calls_used: int,
                           batch_call_budget: int) -> SchedulerGateResult:
        return self.runtime_gate.check_before_claim(
            local_now=local_now, calls_today=calls_today,
            batches_today=batches_today, elapsed_seconds=elapsed_seconds,
            batch_calls_used=batch_calls_used,
            batch_call_budget=batch_call_budget,
        )

    def pause(self, reason: str, *, automatic: bool = False) -> dict[str, Any]:
        if automatic:
            return self.repository.set_automatic_pause(True, reason)
        return self.repository.set_manual_pause(True, reason)

    def resume_manual(self, reason: str = "manual_resume") -> dict[str, Any]:
        control = self.repository.control()
        if control.get("automatic_paused"):
            raise ValueError("historical_scheduler_automatic_pause_requires_incident_resolution")
        return self.repository.set_manual_pause(False, reason)

    def snapshot(self) -> dict[str, Any]:
        return {
            "control": self.repository.control(),
            "open_batch": self.repository.latest_open_batch(),
            "realtime": self.realtime.snapshot(),
            "priority": ("reply", "memory", "historical"),
        }


def daily_utc_bounds(local_now: datetime) -> tuple[str, str]:
    if local_now.tzinfo is None:
        raise ValueError("local_now_timezone_required")
    start = datetime.combine(local_now.date(), time.min, tzinfo=local_now.tzinfo)
    return start.astimezone(timezone.utc).isoformat(), (start + timedelta(days=1)).astimezone(timezone.utc).isoformat()


def weekly_utc_bounds(local_now: datetime) -> tuple[str, str]:
    if local_now.tzinfo is None:
        raise ValueError("local_now_timezone_required")
    monday = local_now.date() - timedelta(days=local_now.weekday())
    start = datetime.combine(monday, time.min, tzinfo=local_now.tzinfo)
    return start.astimezone(timezone.utc).isoformat(), (start + timedelta(days=7)).astimezone(timezone.utc).isoformat()


def scheduler_cost_projection(*, total_customers: int, total_chunks: int,
                              completed_customers: int, completed_chunks: int) -> dict[str, Any]:
    customers = max(0, total_customers - completed_customers)
    chunks = max(0, total_chunks - completed_chunks)
    batches = math.ceil(customers / 20) if customers else 0
    return {"total_customers": total_customers, "completed_customers": completed_customers,
            "remaining_customers": customers, "total_estimated_chunks": total_chunks,
            "completed_estimated_chunks": completed_chunks, "remaining_estimated_chunks": chunks,
            "minimum_remaining_calls": chunks,
            "expected_remaining_calls": chunks + math.ceil(chunks * 3 / 75),
            "batches_at_20_customers": batches, "days_at_one_batch": batches,
            "days_at_two_batches": math.ceil(batches / 2) if batches else 0}


def render_batch_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    return (f"# Historical Long-running Batch {report.get('batch_id','')}\n\n"
            f"- 状态：{report.get('status','')}\n- 客户：{summary.get('customers',0)}\n"
            f"- chunks：{summary.get('chunks',0)}\n"
            f"- Provider calls：{summary.get('provider_calls',0)} / "
            f"{summary.get('provider_call_budget',0)}\n"
            f"- source evidence：{summary.get('source_evidence_percent',0)}%\n")


def write_batch_report(directory: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    jp, mp = directory / "long_running_batch_report.json", directory / "long_running_batch_report.md"
    tmp = jp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(jp); mp.write_text(render_batch_markdown(report), encoding="utf-8")
    return jp, mp


def eligibility_from_plan_dict(item: dict[str, Any]) -> HistoricalBackfillEligibility:
    return HistoricalBackfillEligibility(
        eligible_for_backfill=not bool(item.get("is_worthless")), skip_reason=item.get("skip_reason"),
        value_tier=item.get("value_tier"),
        has_transaction_history=bool(item.get("has_transaction_history")),
        info_unknown=bool(item.get("info_unknown")), is_worthless=bool(item.get("is_worthless")),
        worthless_reason=item.get("worthless_reason"),
        tag_parse_status=str(item.get("tag_parse_status") or "none"),
        raw_remark_hash=str(item.get("raw_remark_hash") or ""),
        parsed_tags=tuple(item.get("parsed_tags") or ()),
        history_message_count=int(item.get("history_message_count") or 0),
        history_start_at=item.get("history_start_at"), history_end_at=item.get("history_end_at"),
        estimated_business_messages=int(item.get("estimated_business_messages") or 0),
        estimated_characters=int(item.get("estimated_characters") or 0),
        estimated_tokens=int(item.get("estimated_tokens") or 0),
        estimated_chunks=int(item.get("estimated_chunks") or 0),
        priority_features=tuple(item.get("priority_features") or ()))


def build_candidate(*, plan: dict[str, Any], mapping: dict[str, Any],
                    activity: HistoricalActivityMetrics,
                    identity: IdentityDecision) -> SchedulerCandidate | None:
    eligibility = eligibility_from_plan_dict(plan)
    if eligibility.is_worthless or eligibility.value_tier not in {"A", "B"}:
        return None
    priority = HistoricalBackfillPriorityScorer().score(
        eligibility=eligibility, activity=activity,
        historical_backfill_completed=str(mapping.get("job_status") or "") in TERMINAL_JOB_STATES)
    if not priority.eligible or eligibility.history_message_count <= 0:
        return None
    return SchedulerCandidate(
        str(plan["safe_contact_reference"]), str(mapping["id"]), str(plan["account_scope_id"]),
        str(plan.get("display_name") or ""), str(eligibility.value_tier),
        int(eligibility.estimated_chunks), int(priority.score), tuple(priority.components),
        identity, int(eligibility.history_message_count), int(eligibility.estimated_business_messages))
