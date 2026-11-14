from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from threading import Lock, RLock
import time
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from whochat.ai.models import ReplyGenerationResult, ReplySuggestion
from whochat.config import AppConfig
from whochat.core.models import Contact, ContactStatus, ConversationType, utc_now_iso
from whochat.diagnostics import append_diagnostics_log
from whochat.services.conversation_style import CONVERSATION_STYLE_VERSION
from whochat.services.historical_eligibility import CustomerRemarkTagParser
from whochat.services.proactive_contact import (
    ACTIVATION_PREWARM_CLOUD_AUTH_SCOPE,
    PROACTIVE_POLICY_VERSION,
    ProactiveContactOutcome,
    ProactiveContactReason,
    ProactiveContactRequest,
    ProactiveContactService,
    ProactiveContactReasonEngine,
    _local_proactive_preview,
    _outcome_from_json,
    _outcome_to_json,
    _proactive_context_hash,
    _with_activation_assessments,
    evaluate_proactive_eligibility,
)
from whochat.services.tracking_reasoning import TRACKING_REASONING_VERSION
from whochat.storage.database import ActivationInvalidationNotice, Database
from whochat.storage.repositories import ContactRepository, new_id


_MIN_FAST_IDENTITY_CONFIDENCE = 0.95
_LOCAL_SNAPSHOT_TTL_DAYS = 7


def normalize_activation_title(value: str) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


@dataclass(frozen=True)
class CompactCustomer:
    account_scope_id: str
    platform: str
    normalized_title: str
    contact_id: str
    conversation_id: str
    canonical_remark: str
    value_tier: str
    is_worthless: bool
    confidence: float
    verified: bool


@dataclass(frozen=True)
class FastCustomerResolution:
    resolved: bool
    reason: str
    contact_id: str = ""
    conversation_id: str = ""
    canonical_remark: str = ""
    value_tier: str = ""
    confidence: float = 0.0
    account_scope_id: str = ""
    target_instance_id: str = ""
    normalized_title: str = ""
    elapsed_ms: float = 0.0

    @property
    def activation_safe(self) -> bool:
        return self.resolved and bool(self.contact_id)


@dataclass(frozen=True)
class FastTitleHint:
    normalized_title: str
    contact_id: str
    confidence: float


class CompactActivationIndex:
    """Small, immutable-at-read index for exact account-scoped DM titles."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = Lock()
        self._titles: dict[tuple[str, str, str], tuple[CompactCustomer, ...]] = {}
        self._fingerprints: dict[tuple[str, str, str], FastTitleHint] = {}
        self._visual_fingerprints: dict[tuple[str, str, str], FastTitleHint] = {}
        self._contacts: dict[str, Contact] = {}
        self.refresh()

    def refresh(self) -> None:
        parser = CustomerRemarkTagParser()
        titles: dict[tuple[str, str, str], list[CompactCustomer]] = {}
        fingerprints: dict[tuple[str, str, str], FastTitleHint] = {}
        visual_candidates: dict[tuple[str, str, str], list[FastTitleHint]] = {}
        contacts: dict[str, Contact] = {}
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT b.account_scope_id,b.platform,b.normalized_title,b.contact_id,
                       b.confidence,b.verified,c.display_name,c.remark,c.status,c.merged_into,
                       c.strategy_id,c.allow_cloud_ai,c.avatar_fingerprint,c.created_at,c.updated_at,
                       COALESCE((
                         SELECT v.id FROM conversations v
                         WHERE v.account_scope_id=b.account_scope_id
                           AND v.contact_id=b.contact_id
                           AND v.platform=b.platform
                           AND v.conversation_type='dm'
                         ORDER BY v.updated_at DESC LIMIT 1
                       ),'') AS conversation_id
                FROM contact_identity_bindings b
                JOIN contacts c ON c.id=b.contact_id
                WHERE b.conversation_type='dm'
                  AND c.account_scope_id=b.account_scope_id
                  AND c.merged_into IS NULL
                """
            ).fetchall()
            for row in rows:
                contact = Contact(
                    id=str(row["contact_id"]),
                    platform=str(row["platform"]),
                    display_name=str(row["display_name"]),
                    conversation_type=ConversationType.DM,
                    status=ContactStatus(str(row["status"])),
                    strategy_id=str(row["strategy_id"] or "default"),
                    allow_cloud_ai=bool(row["allow_cloud_ai"]),
                    remark=str(row["remark"] or ""),
                    avatar_fingerprint=str(row["avatar_fingerprint"] or ""),
                    merged_into=row["merged_into"],
                    created_at=str(row["created_at"] or ""),
                    updated_at=str(row["updated_at"] or ""),
                    account_scope_id=str(row["account_scope_id"]),
                )
                eligibility = evaluate_proactive_eligibility(contact, parser)
                contacts[contact.id] = contact
                normalized = normalize_activation_title(str(row["normalized_title"]))
                item = CompactCustomer(
                    account_scope_id=str(row["account_scope_id"]),
                    platform=str(row["platform"]),
                    normalized_title=normalized,
                    contact_id=str(row["contact_id"]),
                    conversation_id=str(row["conversation_id"] or ""),
                    canonical_remark=str(row["remark"] or row["display_name"]),
                    value_tier=str(eligibility.value_tier or ""),
                    is_worthless=bool(eligibility.is_worthless),
                    confidence=float(row["confidence"] or 0.0),
                    verified=bool(row["verified"]),
                )
                titles.setdefault((item.account_scope_id, item.platform, normalized), []).append(item)
            fingerprint_rows = conn.execute(
                """
                SELECT account_scope_id,target_instance_id,exact_title_hash,visual_title_hash,
                       normalized_title,contact_id,confidence
                FROM activation_title_fingerprints
                WHERE conflicted=0 AND confidence>=?
                """,
                (_MIN_FAST_IDENTITY_CONFIDENCE,),
            ).fetchall()
            for row in fingerprint_rows:
                hint = FastTitleHint(
                    normalize_activation_title(str(row["normalized_title"])),
                    str(row["contact_id"]),
                    float(row["confidence"]),
                )
                fingerprints[
                    (
                        str(row["account_scope_id"]),
                        str(row["target_instance_id"]),
                        str(row["exact_title_hash"]),
                    )
                ] = hint
                visual_hash = str(row["visual_title_hash"] or "")
                if visual_hash:
                    visual_candidates.setdefault(
                        (
                            str(row["account_scope_id"]),
                            str(row["target_instance_id"]),
                            visual_hash,
                        ),
                        [],
                    ).append(hint)
        visual_fingerprints: dict[tuple[str, str, str], FastTitleHint] = {}
        for key, hints in visual_candidates.items():
            unique = {(item.contact_id, item.normalized_title): item for item in hints}
            if len(unique) == 1:
                visual_fingerprints[key] = next(iter(unique.values()))
        with self._lock:
            self._titles = {key: tuple(value) for key, value in titles.items()}
            self._fingerprints = fingerprints
            self._visual_fingerprints = visual_fingerprints
            self._contacts = contacts

    def resolve(
        self,
        *,
        account_scope_id: str,
        platform: str,
        target_instance_id: str,
        normalized_title: str,
        title_confidence: float,
        retained_contact_id: str = "",
    ) -> FastCustomerResolution:
        started = time.perf_counter()
        title = normalize_activation_title(normalized_title)
        base = {
            "account_scope_id": account_scope_id,
            "target_instance_id": target_instance_id,
            "normalized_title": title,
        }
        if not account_scope_id or not target_instance_id:
            return FastCustomerResolution(False, "account_or_target_unstable", **base, elapsed_ms=_ms(started))
        if not title or title_confidence < _MIN_FAST_IDENTITY_CONFIDENCE:
            return FastCustomerResolution(False, "title_confidence_too_low", **base, elapsed_ms=_ms(started))
        with self._lock:
            candidates = self._titles.get((account_scope_id, platform, title), ())
        unique = {item.contact_id: item for item in candidates}
        variant = False
        variant_conflict = False
        if not unique and retained_contact_id:
            with self._lock:
                variant_candidates = [
                    item
                    for (scope, item_platform, indexed_title), items in self._titles.items()
                    if scope == account_scope_id
                    and item_platform == platform
                    and _safe_title_variant(title, indexed_title)
                    for item in items
                    if item.contact_id == retained_contact_id
                    and item.confidence >= _MIN_FAST_IDENTITY_CONFIDENCE
                ]
                all_variant_contacts = {
                    item.contact_id
                    for (scope, item_platform, indexed_title), items in self._titles.items()
                    if scope == account_scope_id
                    and item_platform == platform
                    and _safe_title_variant(title, indexed_title)
                    for item in items
                    if item.confidence >= _MIN_FAST_IDENTITY_CONFIDENCE
                }
            if len(all_variant_contacts) == 1:
                unique = {item.contact_id: item for item in variant_candidates}
                variant = len(unique) == 1
            elif len(all_variant_contacts) > 1:
                variant_conflict = True
        if len(unique) != 1:
            reason = (
                "ambiguous_title_variant"
                if variant_conflict
                else ("title_not_indexed" if not unique else "duplicate_title_conflict")
            )
            return FastCustomerResolution(False, reason, **base, elapsed_ms=_ms(started))
        item = next(iter(unique.values()))
        if item.confidence < _MIN_FAST_IDENTITY_CONFIDENCE:
            return FastCustomerResolution(False, "binding_confidence_too_low", **base, elapsed_ms=_ms(started))
        if item.is_worthless:
            return FastCustomerResolution(False, "worthless_contact", **base, elapsed_ms=_ms(started))
        return FastCustomerResolution(
            True,
            "unique_safe_title_variant" if variant else "unique_exact_account_title",
            item.contact_id,
            item.conversation_id,
            item.canonical_remark,
            item.value_tier,
            min(float(title_confidence), item.confidence),
            account_scope_id,
            target_instance_id,
            title,
            _ms(started),
        )

    def lookup_fingerprint(
        self,
        account_scope_id: str,
        target_instance_id: str,
        exact_title_hash: str,
        visual_title_hash: str = "",
    ) -> FastTitleHint | None:
        if not account_scope_id or not target_instance_id or (not exact_title_hash and not visual_title_hash):
            return None
        with self._lock:
            exact = (
                self._fingerprints.get((account_scope_id, target_instance_id, exact_title_hash))
                if exact_title_hash
                else None
            )
            if exact is not None:
                return exact
            # A perceptual/average hash is useful for repaint suppression,
            # never positive customer identity. Different title pixels can
            # produce the same 8x8 hash; only OCR may resolve an exact miss.
            return None

    def contact(self, contact_id: str) -> Contact | None:
        with self._lock:
            return self._contacts.get(contact_id)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "titles": sum(len(items) for items in self._titles.values()),
                "exact_fingerprints": len(self._fingerprints),
                "visual_fingerprints": len(self._visual_fingerprints),
                "contacts": len(self._contacts),
            }

    def confirm_fingerprint(
        self,
        *,
        account_scope_id: str,
        target_instance_id: str,
        exact_title_hash: str,
        visual_title_hash: str,
        normalized_title: str,
        contact_id: str,
        confidence: float,
    ) -> None:
        if (
            not account_scope_id
            or not target_instance_id
            or not exact_title_hash
            or not normalized_title
            or not contact_id
            or confidence < _MIN_FAST_IDENTITY_CONFIDENCE
        ):
            return
        now = utc_now_iso()
        title = normalize_activation_title(normalized_title)
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                """
                SELECT id,contact_id,normalized_title,conflicted
                FROM activation_title_fingerprints
                WHERE account_scope_id=? AND target_instance_id=? AND exact_title_hash=?
                """,
                (account_scope_id, target_instance_id, exact_title_hash),
            ).fetchone()
            if current is None:
                conn.execute(
                    """
                    INSERT INTO activation_title_fingerprints(
                        id,account_scope_id,target_instance_id,exact_title_hash,visual_title_hash,
                        normalized_title,contact_id,confidence,confirmed_count,conflicted,
                        first_confirmed_at,last_confirmed_at
                    ) VALUES(?,?,?,?,?,?,?,?,1,0,?,?)
                    """,
                    (
                        new_id("activation_title"), account_scope_id, target_instance_id,
                        exact_title_hash, visual_title_hash, title, contact_id,
                        min(1.0, max(0.0, confidence)), now, now,
                    ),
                )
            elif str(current["contact_id"]) != contact_id or str(current["normalized_title"]) != title:
                conn.execute(
                    """
                    UPDATE activation_title_fingerprints
                    SET conflicted=1,last_confirmed_at=? WHERE id=?
                    """,
                    (now, str(current["id"])),
                )
            elif not bool(current["conflicted"]):
                conn.execute(
                    """
                    UPDATE activation_title_fingerprints
                    SET visual_title_hash=?,confidence=MAX(confidence,?),
                        confirmed_count=confirmed_count+1,last_confirmed_at=?
                    WHERE id=?
                    """,
                    (visual_title_hash, confidence, now, str(current["id"])),
                )
        self.refresh()


class FastCustomerResolver:
    def __init__(self, index: CompactActivationIndex) -> None:
        self.index = index

    def resolve(self, **kwargs) -> FastCustomerResolution:
        return self.index.resolve(**kwargs)


@dataclass(frozen=True)
class ActivationSnapshotRecord:
    account_scope_id: str
    contact_id: str
    conversation_id: str
    value_tier: str
    stage_json: str
    stage_hash: str
    need_json: str
    need_hash: str
    talking_point: str
    talking_point_hash: str
    local_text: str
    local_outcome_json: str
    local_context_hash: str
    local_generated_at: str
    local_valid_until: str
    ai_text: str
    ai_outcome_json: str
    ai_context_hash: str
    ai_source: str
    ai_provider: str
    ai_model: str
    ai_generated_at: str | None
    ai_valid_until: str | None
    last_customer_message_at: str | None
    last_our_message_at: str | None
    last_contact_at: str | None
    last_message_fingerprint: str
    known_reply_pending: bool
    confidence: str
    source_revision: str
    invalidated_at: str | None
    invalidation_reason: str
    preferred_salutation: str
    salutation_confidence: str
    conversation_style_version: str
    salutation_evidence_hash: str
    updated_at: str


@dataclass(frozen=True)
class ActivationInvalidationEvent:
    account_scope_id: str
    contact_id: str
    reason: str
    revision: int
    invalidated_at: str


@dataclass(frozen=True)
class InstantActivationResult:
    available: bool
    reason: str
    contact_id: str = ""
    conversation_id: str = ""
    source: str = ""
    text_source: str = ""
    outcome: ProactiveContactOutcome | None = None
    context_hash: str = ""
    lookup_ms: float = 0.0
    activation_revision: int = 0


class ActivationSnapshotStaleError(RuntimeError):
    pass


class ActivationSnapshotRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, account_scope_id: str, contact_id: str) -> ActivationSnapshotRecord | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM activation_snapshots
                WHERE account_scope_id=? AND contact_id=? AND status='ready'
                  AND invalidated_at IS NULL
                """,
                (account_scope_id, contact_id),
            ).fetchone()
        return _snapshot_from_row(row) if row else None

    def invalidate(self, account_scope_id: str, contact_id: str, reason: str) -> bool:
        now = utc_now_iso()
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE activation_snapshots
                SET invalidated_at=?,invalidation_reason=?,updated_at=?
                WHERE account_scope_id=? AND contact_id=? AND invalidated_at IS NULL
                """,
                (now, reason, now, account_scope_id, contact_id),
            )
        return cursor.rowcount == 1

    def list_ready(self, *, limit: int = 2000) -> list[ActivationSnapshotRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM activation_snapshots
                WHERE status='ready' AND invalidated_at IS NULL
                ORDER BY COALESCE(last_accessed_at,updated_at) DESC LIMIT ?
                """,
                (max(1, min(limit, 5000)),),
            ).fetchall()
        return [_snapshot_from_row(row) for row in rows]

    def save_local(
        self,
        *,
        outcome: ProactiveContactOutcome,
        snapshot,
        value_tier: str,
        local_text: str,
        valid_until: str,
        accepted_edit_context_hash: str = "",
    ) -> None:
        now = utc_now_iso()
        recency = snapshot.contact_recency
        stage = snapshot.stage_assessment
        need = snapshot.need_assessment
        latest = snapshot.messages[-1] if snapshot.messages else None
        known_reply_pending = bool(outcome.decision.context_payload.get("reply_mode_required"))
        source_revision = hashlib.sha256(
            json.dumps(snapshot.component_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        stage_payload = asdict(stage) if stage else {}
        stage_payload["proactive_policy_version"] = PROACTIVE_POLICY_VERSION
        if accepted_edit_context_hash:
            stage_payload["accepted_edit_context_hash"] = accepted_edit_context_hash
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO activation_snapshots(
                    id,account_scope_id,contact_id,conversation_id,value_tier,status,
                    stage_json,stage_hash,need_json,need_hash,talking_point,talking_point_hash,
                    local_text,local_outcome_json,
                    local_context_hash,local_generated_at,local_valid_until,
                    last_customer_message_at,last_our_message_at,last_contact_at,
                    last_message_fingerprint,known_reply_pending,confidence,source_revision,
                    invalidated_at,invalidation_reason,
                    preferred_salutation,salutation_confidence,
                    conversation_style_version,salutation_evidence_hash,
                    created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(account_scope_id,contact_id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,value_tier=excluded.value_tier,
                    status='ready',stage_json=excluded.stage_json,stage_hash=excluded.stage_hash,
                    need_json=excluded.need_json,need_hash=excluded.need_hash,
                    talking_point=excluded.talking_point,talking_point_hash=excluded.talking_point_hash,
                    local_text=excluded.local_text,
                    local_outcome_json=excluded.local_outcome_json,
                    local_context_hash=excluded.local_context_hash,
                    local_generated_at=excluded.local_generated_at,
                    local_valid_until=excluded.local_valid_until,
                    last_customer_message_at=excluded.last_customer_message_at,
                    last_our_message_at=excluded.last_our_message_at,
                    last_contact_at=excluded.last_contact_at,
                    last_message_fingerprint=excluded.last_message_fingerprint,
                    known_reply_pending=excluded.known_reply_pending,
                    confidence=excluded.confidence,source_revision=excluded.source_revision,
                    preferred_salutation=excluded.preferred_salutation,
                    salutation_confidence=excluded.salutation_confidence,
                    conversation_style_version=excluded.conversation_style_version,
                    salutation_evidence_hash=excluded.salutation_evidence_hash,
                    ai_text='',ai_outcome_json='',ai_context_hash='',ai_source='',
                    ai_provider='',ai_model='',ai_generated_at=NULL,ai_valid_until=NULL,
                    invalidated_at=NULL,invalidation_reason='',updated_at=excluded.updated_at
                """,
                (
                    new_id("activation_snapshot"), outcome.request.account_scope_id,
                    outcome.request.contact_id, outcome.request.conversation_id or None, value_tier, "ready",
                    json.dumps(stage_payload, ensure_ascii=False, sort_keys=True),
                    snapshot.component_hashes.get("stage_inference", ""),
                    json.dumps(asdict(need) if need else {}, ensure_ascii=False, sort_keys=True),
                    snapshot.component_hashes.get("need_inference", ""),
                    need.best_talking_point if need else "",
                    snapshot.component_hashes.get("talking_point", ""),
                    local_text, _outcome_to_json(outcome), outcome.context_hash, now, valid_until,
                    recency.last_customer_message_at if recency else None,
                    recency.last_our_message_at if recency else None,
                    recency.last_any_contact_at if recency else None,
                    latest.fingerprint if latest else "", int(known_reply_pending),
                    stage.confidence if stage and stage.confidence in {"low", "medium", "high"} else "low",
                    source_revision, None, "",
                    snapshot.conversation_style.usable_salutation if snapshot.conversation_style else "",
                    snapshot.conversation_style.salutation_confidence if snapshot.conversation_style else "low",
                    snapshot.conversation_style.style_version if snapshot.conversation_style else CONVERSATION_STYLE_VERSION,
                    snapshot.conversation_style.evidence_hash if snapshot.conversation_style else "",
                    now, now,
                ),
            )

    def save_ai(
        self,
        outcome: ProactiveContactOutcome,
        *,
        provider: str,
        model: str,
        valid_until: str,
        source: str,
    ) -> bool:
        if outcome.result is None or not outcome.result.allowed or len(outcome.result.suggestions) != 1:
            raise ValueError("activation_ai_outcome_not_displayable")
        if source not in {"prewarm", "on_demand"}:
            raise ValueError("activation_ai_source_invalid")
        now = utc_now_iso()
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE activation_snapshots SET
                    ai_text=?,ai_outcome_json=?,ai_context_hash=?,ai_source=?,ai_provider=?,ai_model=?,
                    ai_generated_at=?,ai_valid_until=?,updated_at=?
                WHERE account_scope_id=? AND contact_id=? AND status='ready'
                  AND invalidated_at IS NULL
                  AND (?='prewarm' OR ai_source<>'prewarm' OR ai_valid_until IS NULL OR ai_valid_until<=?)
                """,
                (
                    outcome.result.suggestions[0].text, _outcome_to_json(outcome), outcome.context_hash,
                    source, provider, model, now, valid_until, now,
                    outcome.request.account_scope_id, outcome.request.contact_id,
                    source, now,
                ),
            )
        if cursor.rowcount != 1 and source == "prewarm":
            raise ValueError("activation_snapshot_changed_during_prewarm")
        return cursor.rowcount == 1

    def touch(self, account_scope_id: str, contact_id: str) -> None:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE activation_snapshots SET last_accessed_at=?
                   WHERE account_scope_id=? AND contact_id=?""",
                (now, account_scope_id, contact_id),
            )

    def next_ai_candidate(self, now_iso: str) -> ActivationSnapshotRecord | None:
        local_date = datetime.now().astimezone().date().isoformat()
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM activation_snapshots
                WHERE status='ready' AND invalidated_at IS NULL AND value_tier='A'
                  AND known_reply_pending=0
                  AND (ai_valid_until IS NULL OR ai_valid_until<=?)
                  AND NOT EXISTS (
                    SELECT 1 FROM activation_prewarm_attempts p
                    WHERE p.account_scope_id=activation_snapshots.account_scope_id
                      AND p.contact_id=activation_snapshots.contact_id
                      AND p.local_date=? AND p.status='failed'
                    GROUP BY p.account_scope_id,p.contact_id
                    HAVING COUNT(*)>=2
                  )
                ORDER BY
                  CASE WHEN last_accessed_at IS NULL THEN 1 ELSE 0 END,
                  last_accessed_at DESC,
                  CASE WHEN last_contact_at IS NULL THEN 0 ELSE 1 END,
                  last_contact_at ASC,
                  updated_at ASC
                LIMIT 1
                """,
                (now_iso, local_date),
            ).fetchone()
        return _snapshot_from_row(row) if row else None

    def stale_local_contacts(self, *, limit: int = 20) -> list[tuple[str, str, str]]:
        """Return safe A/B contacts whose local snapshot is missing or stale."""

        parser = CustomerRemarkTagParser()
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*,
                       COALESCE((SELECT v.id FROM conversations v
                         WHERE v.account_scope_id=c.account_scope_id AND v.contact_id=c.id
                           AND v.conversation_type='dm'
                         ORDER BY v.updated_at DESC LIMIT 1),'') AS conversation_id,
                       s.contact_id AS snapshot_contact_id,s.invalidated_at,s.local_valid_until,
                       s.conversation_style_version,s.stage_json,s.updated_at AS snapshot_updated_at
                FROM contacts c
                LEFT JOIN activation_snapshots s
                  ON s.account_scope_id=c.account_scope_id AND s.contact_id=c.id
                WHERE c.merged_into IS NULL
                ORDER BY CASE WHEN s.contact_id IS NULL THEN 0 ELSE 1 END,
                         COALESCE(s.updated_at,c.updated_at)
                """,
            ).fetchall()
        result: list[tuple[str, str, str]] = []
        now = datetime.now(timezone.utc)
        for row in rows:
            contact = Contact(
                id=str(row["id"]),
                platform=str(row["platform"]),
                display_name=str(row["display_name"]),
                conversation_type=ConversationType(str(row["conversation_type"])),
                status=ContactStatus(str(row["status"])),
                strategy_id=str(row["strategy_id"] or "default"),
                allow_cloud_ai=bool(row["allow_cloud_ai"]),
                remark=str(row["remark"] or ""),
                avatar_fingerprint=str(row["avatar_fingerprint"] or ""),
                merged_into=row["merged_into"],
                created_at=str(row["created_at"] or ""),
                updated_at=str(row["updated_at"] or ""),
                account_scope_id=str(row["account_scope_id"] or ""),
            )
            eligibility = evaluate_proactive_eligibility(contact, parser)
            if (
                not eligibility.eligible
                or eligibility.is_worthless
                or eligibility.value_tier not in {"A", "B"}
                or not contact.account_scope_id
            ):
                continue
            missing = row["snapshot_contact_id"] is None
            expired = _parse_utc(str(row["local_valid_until"] or "")) <= now
            stale = bool(row["invalidated_at"]) or expired
            stale = stale or str(row["conversation_style_version"] or "") != CONVERSATION_STYLE_VERSION
            stale = stale or not _activation_snapshot_is_current(str(row["stage_json"] or "{}"))
            if missing or stale:
                result.append(
                    (contact.account_scope_id, contact.id, str(row["conversation_id"] or ""))
                )
            if len(result) >= max(1, min(limit, 100)):
                break
        return result


class ActivationSnapshotInvalidator(QObject):
    """One process-local authority for L2/L1/revision invalidation."""

    invalidated = Signal(object)

    def __init__(
        self,
        db: Database,
        repository: ActivationSnapshotRepository,
        *,
        state_lock: RLock,
        evict_l1: Callable[[tuple[str, str]], None],
    ) -> None:
        super().__init__()
        self.db = db
        self.repository = repository
        self._state_lock = state_lock
        self._evict_l1 = evict_l1
        self._revisions: dict[tuple[str, str], int] = {}
        self._last_tokens: dict[tuple[str, str], str] = {}
        self._last_business_tokens: dict[tuple[str, str], str] = {}
        self._invalidated_keys: set[tuple[str, str]] = set()
        self._sequence = 0
        self.db.register_activation_invalidation_listener(self._on_database_invalidation)

    def revision(self, account_scope_id: str, contact_id: str) -> int:
        with self._state_lock:
            return self._revisions.get((account_scope_id, contact_id), 0)

    def sequence(self) -> int:
        with self._state_lock:
            return self._sequence

    def is_current(self, account_scope_id: str, contact_id: str, revision: int) -> bool:
        with self._state_lock:
            return self._revisions.get((account_scope_id, contact_id), 0) == revision

    def invalidate(
        self,
        account_scope_id: str,
        contact_id: str,
        reason: str,
        *,
        mutation_token: str = "",
    ) -> int:
        """Invalidate L2 and synchronously make its L1 copy unusable.

        A trigger may already have invalidated L2 before this explicit business
        notification arrives.  In that case the committed-database listener has
        already evicted L1.  If no L2 row exists (for example while its first
        build is running), the synthetic token still advances the revision and
        makes that build stale.
        """

        key = (account_scope_id, contact_id)
        token = f"business:{mutation_token or reason}"
        with self._state_lock:
            if self._last_business_tokens.get(key) == token:
                return self._revisions.get(key, 0)
        changed = self.repository.invalidate(account_scope_id, contact_id, reason)
        with self._state_lock:
            already_invalidated = key in self._invalidated_keys
            if changed:
                self._last_business_tokens[key] = token
        if changed:
            return self.revision(*key)
        if already_invalidated and not mutation_token:
            return self.revision(*key)
        revision = self._apply(key, reason, utc_now_iso(), token)
        with self._state_lock:
            self._last_business_tokens[key] = token
        return revision

    def mark_fresh(self, account_scope_id: str, contact_id: str) -> None:
        key = (account_scope_id, contact_id)
        with self._state_lock:
            self._invalidated_keys.discard(key)
            self._last_tokens.pop(key, None)
            self._last_business_tokens.pop(key, None)

    def _on_database_invalidation(self, notice: ActivationInvalidationNotice) -> None:
        self._apply(
            (notice.account_scope_id, notice.contact_id),
            notice.reason,
            notice.invalidated_at,
            notice.token,
        )

    def _apply(
        self,
        key: tuple[str, str],
        reason: str,
        invalidated_at: str,
        token: str,
    ) -> int:
        with self._state_lock:
            if self._last_tokens.get(key) == token:
                return self._revisions.get(key, 0)
            revision = self._revisions.get(key, 0) + 1
            self._revisions[key] = revision
            self._last_tokens[key] = token
            self._invalidated_keys.add(key)
            self._sequence += 1
            self._evict_l1(key)
        self.invalidated.emit(
            ActivationInvalidationEvent(
                key[0], key[1], reason, revision, invalidated_at
            )
        )
        return revision


class ActivationSnapshotService:
    """L1/L2 read path plus provider-free local snapshot construction."""

    def __init__(
        self,
        db: Database,
        contacts: ContactRepository,
        proactive: ProactiveContactService,
        config_provider: Callable[[], AppConfig],
        *,
        l1_capacity: int = 512,
    ) -> None:
        self.db = db
        self.contacts = contacts
        self.proactive = proactive
        self.config_provider = config_provider
        self.repository = ActivationSnapshotRepository(db)
        self._l1_capacity = max(8, int(l1_capacity))
        self._l1: OrderedDict[tuple[str, str], ActivationSnapshotRecord] = OrderedDict()
        self._feedback_cache: dict[tuple[str, str, str], tuple[str, str]] = {}
        self._lock = RLock()
        self._build_state_lock = Lock()
        self._build_futures: dict[tuple[str, str], Future] = {}
        self.invalidator = ActivationSnapshotInvalidator(
            db,
            self.repository,
            state_lock=self._lock,
            evict_l1=self._evict_l1_locked,
        )
        self.warm_l1()

    def warm_l1(self) -> int:
        for _attempt in range(2):
            sequence = self.invalidator.sequence()
            rows = self.repository.list_ready(limit=self._l1_capacity)
            feedback_cache = self._load_feedback_cache(rows)
            with self._lock:
                if self.invalidator.sequence() != sequence:
                    continue
                self._l1.clear()
                self._feedback_cache = feedback_cache
                for row in reversed(rows):
                    self._put_l1(row)
                return len(rows)
        return 0

    def revision(self, account_scope_id: str, contact_id: str) -> int:
        return self.invalidator.revision(account_scope_id, contact_id)

    def revision_is_current(self, account_scope_id: str, contact_id: str, revision: int) -> bool:
        return self.invalidator.is_current(account_scope_id, contact_id, revision)

    def invalidate(
        self,
        account_scope_id: str,
        contact_id: str,
        reason: str,
        *,
        mutation_token: str = "",
    ) -> int:
        return self.invalidator.invalidate(
            account_scope_id,
            contact_id,
            reason,
            mutation_token=mutation_token,
        )

    def clear_l1(self) -> None:
        with self._lock:
            self._l1.clear()
            self._feedback_cache.clear()

    def l1_stats(self) -> dict[str, int]:
        with self._lock:
            rows = tuple(self._l1.values())
            feedback = tuple(self._feedback_cache.items())
        text_bytes = sum(
            len(value.encode("utf-8"))
            for row in rows
            for value in (
                row.stage_json,
                row.need_json,
                row.talking_point,
                row.local_text,
                row.local_outcome_json,
                row.ai_text,
                row.ai_outcome_json,
                row.preferred_salutation,
            )
        )
        feedback_bytes = sum(
            len(text.encode("utf-8")) + len(kind.encode("utf-8"))
            for _key, (text, kind) in feedback
        )
        return {
            "rows": len(rows),
            "capacity": self._l1_capacity,
            "feedback_rows": len(feedback),
            "serialized_payload_bytes": text_bytes + feedback_bytes,
        }

    def lookup(
        self,
        resolution: FastCustomerResolution,
        *,
        allow_l2: bool = True,
    ) -> InstantActivationResult:
        started = time.perf_counter()
        if not resolution.resolved:
            return InstantActivationResult(False, resolution.reason, lookup_ms=_ms(started))
        key = (resolution.account_scope_id, resolution.contact_id)
        with self._lock:
            activation_revision = self.invalidator.revision(*key)
            row = self._l1.get(key)
            if row is not None:
                self._l1.move_to_end(key)
        source = "l1"
        if row is None and allow_l2:
            row = self.repository.get(*key)
            source = "l2"
            if row is not None:
                with self._lock:
                    if self.invalidator.revision(*key) != activation_revision:
                        row = None
                    else:
                        self._put_l1(row)
        if row is None:
            reason = (
                "snapshot_changed_during_lookup"
                if self.invalidator.revision(*key) != activation_revision
                else "snapshot_missing"
            )
            return InstantActivationResult(
                False,
                reason,
                resolution.contact_id,
                resolution.conversation_id,
                lookup_ms=_ms(started),
                activation_revision=self.invalidator.revision(*key),
            )
        if row.invalidated_at:
            return InstantActivationResult(False, f"snapshot_invalidated:{row.invalidation_reason}", row.contact_id, row.conversation_id, lookup_ms=_ms(started))
        if not _tracking_snapshot_is_current(row.stage_json):
            return InstantActivationResult(False, "snapshot_tracking_version_stale", row.contact_id, row.conversation_id, lookup_ms=_ms(started))
        if not _activation_snapshot_is_current(row.stage_json):
            return InstantActivationResult(False, "snapshot_policy_version_stale", row.contact_id, row.conversation_id, lookup_ms=_ms(started))
        if row.conversation_style_version != CONVERSATION_STYLE_VERSION:
            return InstantActivationResult(False, "snapshot_conversation_style_stale", row.contact_id, row.conversation_id, lookup_ms=_ms(started))
        if row.known_reply_pending:
            return InstantActivationResult(False, "known_reply_pending", row.contact_id, row.conversation_id, lookup_ms=_ms(started))
        now = datetime.now(timezone.utc)
        if _parse_utc(row.local_valid_until) <= now:
            return InstantActivationResult(False, "local_snapshot_expired", row.contact_id, row.conversation_id, lookup_ms=_ms(started))
        outcome_payload = row.local_outcome_json
        context_hash = row.local_context_hash
        text_source = "local_safe"
        if row.ai_outcome_json and row.ai_valid_until and _parse_utc(row.ai_valid_until) > now:
            outcome_payload = row.ai_outcome_json
            context_hash = row.ai_context_hash
            text_source = "ai_prewarm" if row.ai_source == "prewarm" else "ai_on_demand"
        request = ProactiveContactRequest(
            row.account_scope_id, row.contact_id, row.conversation_id,
            resolution.target_instance_id, 0, "", 0,
            datetime.now().astimezone().date().isoformat(), "",
        )
        outcome = _outcome_from_json(outcome_payload, request, context_hash, 0.0)
        if outcome is None:
            return InstantActivationResult(False, "snapshot_payload_invalid", row.contact_id, row.conversation_id, lookup_ms=_ms(started))
        selected_text, feedback_kind = self._feedback_text(
            row,
            context_hash,
            allow_db=(source == "l2"),
        )
        if (
            not selected_text
            and context_hash == row.ai_context_hash
            and row.local_context_hash
        ):
            local_feedback = self._feedback_text(
                row,
                row.local_context_hash,
                allow_db=(source == "l2"),
            )
            if local_feedback[1] == "edited_and_accepted":
                selected_text, feedback_kind = local_feedback
        # A policy-only rebuild may change the generator context hash while
        # customer inputs remain identical. Keep the original human feedback
        # row intact and refer to it only through a verified same-input alias.
        edit_context_hash = _accepted_edit_context_hash(row.stage_json)
        if edit_context_hash and feedback_kind != "edited_and_accepted":
            preserved_edit = self._feedback_text(row, edit_context_hash, allow_db=(source == "l2"))
            if preserved_edit[1] == "edited_and_accepted":
                selected_text, feedback_kind = preserved_edit
        if selected_text:
            outcome = _replace_outcome_text(outcome, selected_text, f"feedback_{feedback_kind}")
            text_source = feedback_kind
        with self._lock:
            if self.invalidator.revision(*key) != activation_revision:
                return InstantActivationResult(
                    False,
                    "snapshot_changed_during_lookup",
                    row.contact_id,
                    row.conversation_id,
                    lookup_ms=_ms(started),
                    activation_revision=self.invalidator.revision(*key),
                )
        return InstantActivationResult(
            True, "ready", row.contact_id, row.conversation_id, source, text_source,
            outcome, context_hash, _ms(started), activation_revision,
        )

    def safe_fallback(self, resolution: FastCustomerResolution) -> InstantActivationResult:
        """Report missing context without inventing a sendable opening.

        Identity alone cannot establish a useful contact purpose. A pending
        local snapshot is therefore a loading state, never an activation that
        can hide the subsequent context-based result.
        """

        started = time.perf_counter()
        if not resolution.activation_safe:
            return InstantActivationResult(False, resolution.reason, lookup_ms=_ms(started))
        return InstantActivationResult(
            available=False,
            reason="snapshot_pending",
            contact_id=resolution.contact_id,
            conversation_id=resolution.conversation_id,
            lookup_ms=_ms(started),
            activation_revision=self.invalidator.revision(resolution.account_scope_id, resolution.contact_id),
        )

    def build_local(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
    ) -> ActivationSnapshotRecord:
        key = (account_scope_id, contact_id)
        with self._build_state_lock:
            shared = self._build_futures.get(key)
            if shared is None:
                shared = Future()
                self._build_futures[key] = shared
                owner = True
            else:
                owner = False
        if not owner:
            return shared.result()
        try:
            for attempt in range(2):
                try:
                    row = self._build_local_once(
                        account_scope_id=account_scope_id,
                        contact_id=contact_id,
                        conversation_id=conversation_id,
                    )
                    break
                except ActivationSnapshotStaleError:
                    if attempt >= 1:
                        raise
                    append_diagnostics_log(
                        "instant_activation",
                        f"local_snapshot_stale_retry contact={_safe_ref(contact_id)}",
                    )
            shared.set_result(row)
            return row
        except BaseException as exc:
            shared.set_exception(exc)
            raise
        finally:
            with self._build_state_lock:
                self._build_futures.pop(key, None)

    def _build_local_once(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
    ) -> ActivationSnapshotRecord:
        expected_revision = self.invalidator.revision(account_scope_id, contact_id)
        config = self.config_provider()
        local_config = replace(config, ai=replace(config.ai, provider="Local Model", api_key=""))
        request = ProactiveContactRequest(
            account_scope_id, contact_id, conversation_id, "activation-local", 0, "", 0,
            datetime.now().astimezone().date().isoformat(), "",
        )
        snapshot = self.proactive.builder.build(request)
        snapshot = _with_activation_assessments(
            snapshot,
            now=datetime.now().astimezone(),
            silence_days=local_config.proactive.silence_days,
            force=True,
        )
        decision = self.proactive.engine.evaluate(snapshot)
        context_hash = _proactive_context_hash(snapshot, decision, local_config)
        result = _local_proactive_preview(snapshot, decision)
        outcome = ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)
        if not self.invalidator.is_current(account_scope_id, contact_id, expected_revision):
            raise ActivationSnapshotStaleError("activation_snapshot_build_stale_before_save")
        eligibility = evaluate_proactive_eligibility(snapshot.contact)
        if not eligibility.eligible or eligibility.is_worthless or eligibility.value_tier not in {"A", "B"}:
            raise ValueError(f"activation_ineligible:{eligibility.reason}")
        local_text = result.suggestions[0].text
        accepted_edit_context_hash = self._preserve_policy_upgrade_edit(snapshot, decision)
        valid_until = (datetime.now(timezone.utc) + timedelta(days=_LOCAL_SNAPSHOT_TTL_DAYS)).isoformat()
        self.repository.save_local(
            outcome=outcome,
            snapshot=snapshot,
            value_tier=str(eligibility.value_tier),
            local_text=local_text,
            valid_until=valid_until,
            accepted_edit_context_hash=accepted_edit_context_hash,
        )
        if not self.invalidator.is_current(account_scope_id, contact_id, expected_revision):
            self.repository.invalidate(
                account_scope_id,
                contact_id,
                "activation_snapshot_build_stale_after_save",
            )
            raise ActivationSnapshotStaleError("activation_snapshot_build_stale_after_save")
        row = self.repository.get(account_scope_id, contact_id)
        if row is None:
            raise RuntimeError("activation_snapshot_save_failed")
        with self._lock:
            if not self.invalidator.is_current(account_scope_id, contact_id, expected_revision):
                stale = True
            else:
                stale = False
                self.invalidator.mark_fresh(account_scope_id, contact_id)
                self._put_l1(row)
        if stale:
            self.repository.invalidate(
                account_scope_id,
                contact_id,
                "activation_snapshot_build_stale_before_l1",
            )
            raise ActivationSnapshotStaleError("activation_snapshot_build_stale_before_l1")
        return row

    def initialize_safe_a(self, *, include_b: bool = True) -> dict[str, int]:
        parser = CustomerRemarkTagParser()
        result = {
            "a_total": 0,
            "safe_a": 0,
            "generated": 0,
            "confirmation_fallback": 0,
            "worthless_skipped": 0,
            "missing_conversation": 0,
            "failed": 0,
            "b_total": 0,
            "safe_b": 0,
            "b_generated": 0,
            "b_failed": 0,
            "b_confirmation_fallback": 0,
            "already_ready": 0,
            "missing_snapshot": 0,
            "stale_snapshot": 0,
        }
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*,
                       COALESCE((SELECT v.id FROM conversations v
                         WHERE v.account_scope_id=c.account_scope_id AND v.contact_id=c.id
                           AND v.conversation_type='dm'
                         ORDER BY v.updated_at DESC LIMIT 1),'') AS conversation_id
                FROM contacts c WHERE c.merged_into IS NULL
                """
            ).fetchall()
        for row in rows:
            contact = self.contacts.get(str(row["id"]))
            if contact is None:
                continue
            eligibility = evaluate_proactive_eligibility(contact, parser)
            tier = eligibility.value_tier
            if tier not in ({"A", "B"} if include_b else {"A"}):
                continue
            if tier == "A":
                result["a_total"] += 1
            else:
                result["b_total"] += 1
            if eligibility.is_worthless:
                result["worthless_skipped"] += 1
                continue
            if tier == "A":
                result["safe_a"] += 1
            else:
                result["safe_b"] += 1
            conversation_id = str(row["conversation_id"] or "")
            if not conversation_id:
                result["missing_conversation"] += 1
            existing = self.repository.get(contact.account_scope_id, contact.id)
            if existing is None:
                result["missing_snapshot"] += 1
            else:
                fresh = bool(
                    existing.invalidated_at is None
                    and _parse_utc(existing.local_valid_until) > datetime.now(timezone.utc)
                    and _activation_snapshot_is_current(existing.stage_json)
                    and existing.conversation_style_version == CONVERSATION_STYLE_VERSION
                    and existing.local_text
                )
                if fresh:
                    result["already_ready"] += 1
                    continue
                result["stale_snapshot"] += 1
            try:
                snapshot = self.build_local(
                    account_scope_id=contact.account_scope_id,
                    contact_id=contact.id,
                    conversation_id=conversation_id,
                )
                result["generated" if tier == "A" else "b_generated"] += 1
                stage = json.loads(snapshot.stage_json or "{}")
                if str(stage.get("status") or "unknown") == "unknown":
                    result[
                        "confirmation_fallback" if tier == "A" else "b_confirmation_fallback"
                    ] += 1
            except Exception as exc:
                result["failed" if tier == "A" else "b_failed"] += 1
                append_diagnostics_log("instant_activation", f"local_snapshot_failed contact={_safe_ref(contact.id)} error={type(exc).__name__}:{exc}")
        self.warm_l1()
        return result

    def refresh_ai(self, row: ActivationSnapshotRecord) -> ProactiveContactOutcome:
        expected_revision = self.invalidator.revision(row.account_scope_id, row.contact_id)
        if not _activation_snapshot_is_current(row.stage_json):
            raise RuntimeError("activation_tracking_snapshot_stale")
        config = self.config_provider()
        request = ProactiveContactRequest(
            row.account_scope_id, row.contact_id, row.conversation_id,
            "activation-prewarm", 0, "", 0,
            datetime.now().astimezone().date().isoformat(),
            ACTIVATION_PREWARM_CLOUD_AUTH_SCOPE,
        )
        outcome = self.proactive.evaluate(request, config)
        if not self.invalidator.is_current(row.account_scope_id, row.contact_id, expected_revision):
            raise ActivationSnapshotStaleError("activation_prewarm_stale_before_save")
        if (
            outcome.result is None
            or not outcome.result.allowed
            or len(outcome.result.suggestions) != 1
            or outcome.result.provider not in {"OpenAI", "OpenAI Compatible"}
        ):
            raise RuntimeError(outcome.result.status if outcome.result else outcome.display_mode)
        ttl = max(1, int(config.proactive.activation_prewarm_ttl_days))
        valid_until = (datetime.now(timezone.utc) + timedelta(days=ttl)).isoformat()
        saved = self.repository.save_ai(
            outcome,
            provider=outcome.result.provider,
            model=config.ai.model,
            valid_until=valid_until,
            source="prewarm",
        )
        if not saved:
            raise ActivationSnapshotStaleError("activation_prewarm_snapshot_changed")
        if not self.invalidator.is_current(row.account_scope_id, row.contact_id, expected_revision):
            raise ActivationSnapshotStaleError("activation_prewarm_stale_after_save")
        refreshed = self.repository.get(row.account_scope_id, row.contact_id)
        if refreshed is not None:
            with self._lock:
                if self.invalidator.is_current(row.account_scope_id, row.contact_id, expected_revision):
                    self._put_l1(refreshed)
        return outcome

    def cache_on_demand(
        self,
        outcome: ProactiveContactOutcome,
        config: AppConfig,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        if (
            outcome.result is None
            or not outcome.result.allowed
            or len(outcome.result.suggestions) != 1
            or outcome.result.provider not in {"OpenAI", "OpenAI Compatible"}
        ):
            return False
        account_scope_id = outcome.request.account_scope_id
        contact_id = outcome.request.contact_id
        expected_revision = (
            self.invalidator.revision(account_scope_id, contact_id)
            if expected_revision is None
            else expected_revision
        )
        if not self.invalidator.is_current(account_scope_id, contact_id, expected_revision):
            return False
        ttl = max(1, int(config.proactive.activation_prewarm_ttl_days))
        valid_until = (datetime.now(timezone.utc) + timedelta(days=ttl)).isoformat()
        saved = self.repository.save_ai(
            outcome,
            provider=outcome.result.provider,
            model=config.ai.model,
            valid_until=valid_until,
            source="on_demand",
        )
        if not saved:
            return False
        if not self.invalidator.is_current(account_scope_id, contact_id, expected_revision):
            self.repository.invalidate(
                account_scope_id,
                contact_id,
                "activation_on_demand_stale_after_save",
            )
            return False
        refreshed = self.repository.get(outcome.request.account_scope_id, outcome.request.contact_id)
        if refreshed is not None:
            with self._lock:
                if not self.invalidator.is_current(account_scope_id, contact_id, expected_revision):
                    return False
                self._put_l1(refreshed)
        return True

    def _feedback_text(
        self,
        row: ActivationSnapshotRecord,
        context_hash: str,
        *,
        allow_db: bool,
    ) -> tuple[str, str]:
        key = (row.account_scope_id, row.contact_id, context_hash)
        with self._lock:
            cached = self._feedback_cache.get(key)
        if cached is not None or not allow_db:
            return cached or ("", "")
        result = self._query_feedback_text(row, context_hash)
        if result[0]:
            with self._lock:
                self._feedback_cache[key] = result
        return result

    def _preserve_policy_upgrade_edit(self, snapshot, decision) -> str:
        """Carry human text only when every underlying customer input agrees.

        Derived stage/need/talking-point hashes can change with this policy;
        all other component hashes must still match the old source revision.
        Source invalidation, a new conversation or Reply/wait priority wins.
        """
        request = snapshot.request
        old = self.repository.get(request.account_scope_id, request.contact_id)
        if (
            old is None or old.invalidated_at
            or old.conversation_id != request.conversation_id
            or not _tracking_snapshot_is_current(old.stage_json)
            or _parse_utc(old.local_valid_until) <= datetime.now(timezone.utc)
            or old.known_reply_pending or not decision.eligible
            or decision.reason_type == "do_not_contact_now"
            or decision.context_payload.get("reply_mode_required")
        ):
            return ""
        components = dict(snapshot.component_hashes)
        components.update({
            "stage_inference": old.stage_hash,
            "need_inference": old.need_hash,
            "talking_point": old.talking_point_hash,
        })
        previous_source_revision = hashlib.sha256(
            json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if previous_source_revision != old.source_revision:
            return ""
        candidates = (
            old.ai_context_hash, old.local_context_hash, _accepted_edit_context_hash(old.stage_json),
        )
        for context_hash in candidates:
            if context_hash:
                text, kind = self._feedback_text(old, context_hash, allow_db=True)
                if text and kind == "edited_and_accepted":
                    return context_hash
        return ""

    def _query_feedback_text(self, row: ActivationSnapshotRecord, context_hash: str) -> tuple[str, str]:
        with self.db.connect() as conn:
            feedback = conn.execute(
                """
                SELECT feedback_type,original_text,final_text
                FROM suggestion_feedback
                WHERE account_scope_id=? AND contact_id=? AND context_hash=?
                  AND feedback_type IN ('edited_and_accepted','accepted')
                ORDER BY CASE feedback_type WHEN 'edited_and_accepted' THEN 0 ELSE 1 END,
                         updated_at DESC LIMIT 1
                """,
                (row.account_scope_id, row.contact_id, context_hash),
            ).fetchone()
        if feedback is None:
            return "", ""
        kind = str(feedback["feedback_type"])
        text = str(feedback["final_text"] or feedback["original_text"] or "").strip()
        return text, kind

    def _load_feedback_cache(
        self,
        rows: list[ActivationSnapshotRecord],
    ) -> dict[tuple[str, str, str], tuple[str, str]]:
        """Warm accepted feedback in one query; the click path remains SQLite-free."""

        wanted = {
            (row.account_scope_id, row.contact_id, context_hash)
            for row in rows
            for context_hash in (
                row.local_context_hash, row.ai_context_hash,
                _accepted_edit_context_hash(row.stage_json),
            )
            if context_hash
        }
        if not wanted:
            return {}
        with self.db.connect() as conn:
            feedback_rows = conn.execute(
                """
                SELECT account_scope_id,contact_id,context_hash,feedback_type,
                       original_text,final_text,updated_at
                FROM suggestion_feedback
                WHERE feedback_type IN ('edited_and_accepted','accepted')
                ORDER BY CASE feedback_type WHEN 'edited_and_accepted' THEN 0 ELSE 1 END,
                         updated_at DESC
                """
            ).fetchall()
        result: dict[tuple[str, str, str], tuple[str, str]] = {}
        for feedback in feedback_rows:
            key = (
                str(feedback["account_scope_id"]),
                str(feedback["contact_id"]),
                str(feedback["context_hash"]),
            )
            if key not in wanted or key in result:
                continue
            text = str(feedback["final_text"] or feedback["original_text"] or "").strip()
            if text:
                result[key] = (text, str(feedback["feedback_type"]))
        return result

    def _put_l1(self, row: ActivationSnapshotRecord) -> None:
        if row.invalidated_at:
            return
        key = (row.account_scope_id, row.contact_id)
        self._l1[key] = row
        self._l1.move_to_end(key)
        while len(self._l1) > self._l1_capacity:
            self._l1.popitem(last=False)

    def _evict_l1_locked(self, key: tuple[str, str]) -> None:
        self._l1.pop(key, None)
        stale_feedback = [
            item for item in self._feedback_cache if item[:2] == key
        ]
        for item in stale_feedback:
            self._feedback_cache.pop(item, None)


class ActivationPrewarmScheduler(QObject):
    """Single-worker local A/B maintenance plus optional A-only cloud prewarm."""

    local_refresh_completed = Signal()
    snapshot_ready = Signal(str, str, int)

    def __init__(
        self,
        snapshots: ActivationSnapshotService,
        config_provider: Callable[[], AppConfig],
        *,
        busy_provider: Callable[[], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.snapshots = snapshots
        self.config_provider = config_provider
        self.busy_provider = busy_provider
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.local_refresh_completed.connect(self._tick)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whochat-activation-prewarm")
        self._running: Future | None = None
        self._local_requests: deque[tuple[str, str, str]] = deque()
        self._queued_local_keys: set[tuple[str, str]] = set()
        self._local_running_key: tuple[str, str] | None = None
        self._last_user_activity = time.monotonic()
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._running is not None and not self._running.done()

    def start(self) -> None:
        config = self.config_provider().proactive
        self._started = True
        initial_delay = (
            int(config.activation_prewarm_initial_delay_seconds)
            if config.activation_prewarm_enabled
            else 30
        )
        QTimer.singleShot(
            max(30, initial_delay) * 1000,
            self._start_after_delay,
        )

    def _start_after_delay(self) -> None:
        if not self._started:
            return
        config = self.config_provider().proactive
        interval = (
            int(config.activation_prewarm_interval_seconds)
            if config.activation_prewarm_enabled
            else 30
        )
        self._timer.start(max(30, interval) * 1000)
        self._tick()

    def stop(self) -> None:
        self._started = False
        self._timer.stop()
        if self._running is not None and not self._running.done():
            self._running.cancel()
        self._local_requests.clear()
        self._queued_local_keys.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def note_user_activity(self) -> None:
        self._last_user_activity = time.monotonic()

    def request_local_build(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str = "",
    ) -> bool:
        """Queue a provider-free snapshot build without blocking first paint."""

        key = (account_scope_id, contact_id)
        if not all(key) or key in self._queued_local_keys or key == self._local_running_key:
            return False
        self._queued_local_keys.add(key)
        self._local_requests.append((account_scope_id, contact_id, conversation_id))
        QTimer.singleShot(0, self._tick)
        return True

    def _tick(self) -> None:
        if not self._started or self.is_running:
            return
        if self._local_requests:
            scope_id, contact_id, conversation_id = self._local_requests.popleft()
            key = (scope_id, contact_id)
            self._queued_local_keys.discard(key)
            self._local_running_key = key
            started = time.perf_counter()
            self._running = self._executor.submit(
                self.snapshots.build_local,
                account_scope_id=scope_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
            )
            self._running.add_done_callback(
                lambda future: self._complete_local_refresh(scope_id, contact_id, started, future)
            )
            return
        config = self.config_provider()
        policy = config.proactive
        if time.monotonic() - self._last_user_activity < max(30, policy.activation_prewarm_idle_seconds):
            return
        if self.busy_provider():
            return
        stale_local = self.snapshots.repository.stale_local_contacts(limit=1)
        if stale_local:
            scope_id, contact_id, conversation_id = stale_local[0]
            self._local_running_key = (scope_id, contact_id)
            started = time.perf_counter()
            self._running = self._executor.submit(
                self.snapshots.build_local,
                account_scope_id=scope_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
            )
            self._running.add_done_callback(
                lambda future: self._complete_local_refresh(scope_id, contact_id, started, future)
            )
            return
        if not policy.activation_prewarm_enabled:
            return
        if self._daily_attempts() >= max(0, policy.activation_prewarm_daily_provider_cap):
            return
        candidate = self.snapshots.repository.next_ai_candidate(utc_now_iso())
        if candidate is None:
            return
        attempt_id = new_id("activation_prewarm")
        started_at = utc_now_iso()
        with self.snapshots.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO activation_prewarm_attempts(
                    id,local_date,account_scope_id,contact_id,context_hash,provider,model,
                    status,started_at,detail
                ) VALUES(?,?,?,?,?,?,?,'started',?,'idle_scheduler')
                """,
                (
                    attempt_id, datetime.now().astimezone().date().isoformat(),
                    candidate.account_scope_id, candidate.contact_id, candidate.local_context_hash,
                    config.ai.provider, config.ai.model, started_at,
                ),
            )
        started = time.perf_counter()
        self._running = self._executor.submit(self.snapshots.refresh_ai, candidate)
        self._running.add_done_callback(
            lambda future: self._complete_attempt(attempt_id, started, future)
        )

    def _complete_attempt(self, attempt_id: str, started: float, future: Future) -> None:
        try:
            outcome = future.result()
            status = "completed"
            detail = outcome.context_hash[:16]
        except Exception as exc:
            status = "failed"
            detail = f"{type(exc).__name__}:{exc}"[:300]
        with self.snapshots.db.connect() as conn:
            conn.execute(
                """
                UPDATE activation_prewarm_attempts
                SET status=?,completed_at=?,elapsed_ms=?,detail=? WHERE id=?
                """,
                (status, utc_now_iso(), int(_ms(started)), detail, attempt_id),
            )
        append_diagnostics_log("activation_prewarm", f"attempt={attempt_id} status={status} elapsed_ms={int(_ms(started))}")

    def _complete_local_refresh(self, account_scope_id: str, contact_id: str, started: float, future: Future) -> None:
        try:
            future.result()
            status = "completed"
        except Exception as exc:
            status = f"failed:{type(exc).__name__}"
        append_diagnostics_log(
            "activation_prewarm",
            f"local_refresh contact={_safe_ref(contact_id)} status={status} elapsed_ms={int(_ms(started))}",
        )
        self._local_running_key = None
        if status == "completed":
            self.snapshot_ready.emit(
                account_scope_id, contact_id, self.snapshots.revision(account_scope_id, contact_id)
            )
        self.local_refresh_completed.emit()

    def _daily_attempts(self) -> int:
        local_date = datetime.now().astimezone().date().isoformat()
        with self.snapshots.db.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM activation_prewarm_attempts
                WHERE local_date=? AND status IN ('started','completed','failed')
                """,
                (local_date,),
            ).fetchone()
        return int(row[0]) if row else 0


def _replace_outcome_text(outcome: ProactiveContactOutcome, text: str, provider: str) -> ProactiveContactOutcome:
    original = outcome.result
    if original is None:
        return outcome
    result = ReplyGenerationResult(
        allowed=True,
        status=original.status,
        suggestions=[ReplySuggestion("激活建议", text, "low", "使用用户已确认的同上下文话术")],
        provider=provider,
        customer_summary=original.customer_summary,
        intent=original.intent,
        relevant_history=original.relevant_history,
        ask_first=original.ask_first,
        sales_judgement=original.sales_judgement,
        next_action=original.next_action,
        performance_trace=original.performance_trace,
    )
    return replace(outcome, result=result, cache_hit=True)


def _snapshot_from_row(row) -> ActivationSnapshotRecord:
    return ActivationSnapshotRecord(
        account_scope_id=str(row["account_scope_id"]),
        contact_id=str(row["contact_id"]),
        conversation_id=str(row["conversation_id"] or ""),
        value_tier=str(row["value_tier"]),
        stage_json=str(row["stage_json"]),
        stage_hash=str(row["stage_hash"]),
        need_json=str(row["need_json"]),
        need_hash=str(row["need_hash"]),
        talking_point=str(row["talking_point"]),
        talking_point_hash=str(row["talking_point_hash"]),
        local_text=str(row["local_text"]),
        local_outcome_json=str(row["local_outcome_json"]),
        local_context_hash=str(row["local_context_hash"]),
        local_generated_at=str(row["local_generated_at"]),
        local_valid_until=str(row["local_valid_until"]),
        ai_text=str(row["ai_text"]),
        ai_outcome_json=str(row["ai_outcome_json"]),
        ai_context_hash=str(row["ai_context_hash"]),
        ai_source=str(row["ai_source"]),
        ai_provider=str(row["ai_provider"]),
        ai_model=str(row["ai_model"]),
        ai_generated_at=row["ai_generated_at"],
        ai_valid_until=row["ai_valid_until"],
        last_customer_message_at=row["last_customer_message_at"],
        last_our_message_at=row["last_our_message_at"],
        last_contact_at=row["last_contact_at"],
        last_message_fingerprint=str(row["last_message_fingerprint"]),
        known_reply_pending=bool(row["known_reply_pending"]),
        confidence=str(row["confidence"]),
        source_revision=str(row["source_revision"]),
        invalidated_at=row["invalidated_at"],
        invalidation_reason=str(row["invalidation_reason"]),
        preferred_salutation=str(row["preferred_salutation"] or ""),
        salutation_confidence=str(row["salutation_confidence"] or "low"),
        conversation_style_version=str(row["conversation_style_version"] or ""),
        salutation_evidence_hash=str(row["salutation_evidence_hash"] or ""),
        updated_at=str(row["updated_at"]),
    )


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_title_variant(observed: str, indexed: str) -> bool:
    """Very conservative account-local prefix/suffix truncation match."""

    observed = normalize_activation_title(observed)
    indexed = normalize_activation_title(indexed)
    if not observed or not indexed or observed == indexed:
        return False
    shorter, longer = sorted((observed, indexed), key=len)
    if len(shorter) < 8 or len(longer) - len(shorter) > 2:
        return False
    if len(shorter) / len(longer) < 0.90:
        return False
    return longer.startswith(shorter) or longer.endswith(shorter)


def _tracking_snapshot_is_current(stage_json: str) -> bool:
    return str(_snapshot_metadata(stage_json).get("tracking_reasoning_version") or "") == TRACKING_REASONING_VERSION


def _activation_snapshot_is_current(stage_json: str) -> bool:
    if not _tracking_snapshot_is_current(stage_json):
        return False
    return _snapshot_metadata(stage_json).get("proactive_policy_version") == PROACTIVE_POLICY_VERSION


def _accepted_edit_context_hash(stage_json: str) -> str:
    return str(_snapshot_metadata(stage_json).get("accepted_edit_context_hash") or "")


def _snapshot_metadata(stage_json: str) -> dict:
    try:
        payload = json.loads(stage_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_ref(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)
