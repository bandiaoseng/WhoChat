from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from whochat.core.models import AIAnalysisEvent, Message, utc_now_iso
from whochat.storage.database import Database


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class KnowledgeGapRepository:
    """Durable, local-only coverage observations derived from formal messages."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record_evaluation(
        self,
        *,
        source_message: Message,
        crop: str,
        growth_stage: str,
        problem: str,
        issue_type: str,
        issue_text: str,
        matched_product_ids: list[str],
        matched_solution_ids: list[str],
        missing_types: list[str],
        reply_capability: str,
        affects_reply: bool,
        signature_hash: str,
        available_knowledge: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now_iso()
        products_json = json.dumps(matched_product_ids, ensure_ascii=False, separators=(",", ":"))
        solutions_json = json.dumps(matched_solution_ids, ensure_ascii=False, separators=(",", ":"))
        missing_json = json.dumps(missing_types, ensure_ascii=False, separators=(",", ":"))
        available_json = json.dumps(available_knowledge, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        evaluation_id = _id("coverage")
        gap_id = ""
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            identity = conn.execute(
                "SELECT account_scope_id,contact_id,conversation_id FROM messages WHERE id=?",
                (source_message.id,),
            ).fetchone()
            if identity is None:
                raise ValueError("knowledge_gap_source_message_missing")
            if (
                identity["account_scope_id"] != source_message.account_scope_id
                or identity["contact_id"] != source_message.contact_id
                or identity["conversation_id"] != source_message.conversation_id
            ):
                raise ValueError("knowledge_gap_source_identity_mismatch")
            existing = conn.execute(
                "SELECT id FROM knowledge_coverage_evaluations WHERE source_message_id=?",
                (source_message.id,),
            ).fetchone()
            if existing is not None:
                linked = conn.execute(
                    "SELECT gap_id FROM knowledge_gap_occurrences WHERE evaluation_id=? LIMIT 1",
                    (existing["id"],),
                ).fetchone()
                return {
                    "created": False,
                    "evaluation_id": existing["id"],
                    "gap_id": linked["gap_id"] if linked else "",
                }
            conn.execute(
                """
                INSERT INTO knowledge_coverage_evaluations(
                    id,source_message_id,account_scope_id,contact_id,conversation_id,
                    crop,growth_stage,problem,issue_type,issue_text,
                    matched_product_ids_json,matched_solution_ids_json,missing_types_json,
                    reply_capability,affects_reply,evaluated_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    evaluation_id, source_message.id, source_message.account_scope_id,
                    source_message.contact_id, source_message.conversation_id or "",
                    crop, growth_stage, problem, issue_type, issue_text[:500],
                    products_json, solutions_json, missing_json, reply_capability,
                    1 if affects_reply else 0, now, now, now,
                ),
            )
            if missing_types:
                gap = conn.execute(
                    "SELECT id FROM knowledge_gaps WHERE signature_hash=?",
                    (signature_hash,),
                ).fetchone()
                if gap is None:
                    gap_id = _id("knowledge_gap")
                    aggregate_issue = (problem or issue_type or "unknown")[:120]
                    conn.execute(
                        """
                        INSERT INTO knowledge_gaps(
                            id,signature_hash,crop,growth_stage,problem,issue_type,issue_text,
                            missing_types_json,available_knowledge_json,occurrence_count,
                            first_seen_at,last_seen_at,affects_reply,status,created_at,updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,0,?,?,?,'open',?,?)
                        """,
                        (
                            gap_id, signature_hash, crop, growth_stage, problem, issue_type,
                            aggregate_issue, missing_json, available_json, now, now,
                            1 if affects_reply else 0, now, now,
                        ),
                    )
                else:
                    gap_id = str(gap["id"])
                    conn.execute(
                        """
                        UPDATE knowledge_gaps
                           SET last_seen_at=?, updated_at=?,
                               affects_reply=MAX(affects_reply,?),
                               available_knowledge_json=?
                         WHERE id=?
                        """,
                        (now, now, 1 if affects_reply else 0, available_json, gap_id),
                    )
                occurrence = conn.execute(
                    """
                    INSERT OR IGNORE INTO knowledge_gap_occurrences(
                        id,gap_id,evaluation_id,source_message_id,observed_at,created_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (_id("gap_occurrence"), gap_id, evaluation_id, source_message.id, source_message.observed_at, now),
                )
                if occurrence.rowcount:
                    conn.execute(
                        "UPDATE knowledge_gaps SET occurrence_count=occurrence_count+1,last_seen_at=?,updated_at=? WHERE id=?",
                        (now, now, gap_id),
                    )
        return {"created": True, "evaluation_id": evaluation_id, "gap_id": gap_id}

    def counts(self) -> dict[str, int]:
        with self.db.connect() as conn:
            return {
                "evaluations": int(conn.execute("SELECT COUNT(*) FROM knowledge_coverage_evaluations").fetchone()[0]),
                "gaps": int(conn.execute("SELECT COUNT(*) FROM knowledge_gaps").fetchone()[0]),
                "open_gaps": int(conn.execute("SELECT COUNT(*) FROM knowledge_gaps WHERE status='open'").fetchone()[0]),
                "occurrences": int(conn.execute("SELECT COUNT(*) FROM knowledge_gap_occurrences").fetchone()[0]),
            }

    def recent_evaluations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_coverage_evaluations ORDER BY evaluated_at DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def top_gaps(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM knowledge_gaps
                 ORDER BY status='open' DESC, affects_reply DESC,
                          occurrence_count DESC, last_seen_at DESC
                 LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [dict(row) for row in rows]


class ReplyPipelineMetricRepository:
    _FIELDS = {
        "context_started_at", "context_completed_at", "reply_started_at",
        "reply_completed_at", "ui_ready_at", "capture_cycle_started_at",
        "capture_started_at", "capture_completed_at", "title_ocr_started_at",
        "title_ocr_completed_at", "content_ocr_started_at", "content_ocr_completed_at",
        "identity_started_at", "identity_stable_at", "message_classified_at",
        "message_committed_at", "reply_event_created_at", "reply_queued_at",
        "reply_worker_started_at", "provider_request_started_at", "provider_first_response_at",
        "provider_response_completed_at", "reply_parse_completed_at", "result_persisted_at",
        "ui_event_received_at", "ui_render_started_at", "ui_render_completed_at",
    }

    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure(self, event: AIAnalysisEvent, message: Message) -> None:
        now = utc_now_iso()
        try:
            snapshot = json.loads(event.trigger_snapshot_json or "{}")
        except json.JSONDecodeError:
            snapshot = {}
        trace = snapshot.get("performance_trace") if isinstance(snapshot, dict) else {}
        trace = trace if isinstance(trace, dict) else {}
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO reply_pipeline_metrics(
                    event_id,source_message_id,account_scope_id,contact_id,conversation_id,
                    message_observed_at,event_created_at,reply_event_created_at,
                    capture_cycle_started_at,capture_started_at,capture_completed_at,
                    title_ocr_started_at,title_ocr_completed_at,content_ocr_started_at,content_ocr_completed_at,
                    identity_started_at,identity_stable_at,message_classified_at,message_committed_at,
                    trace_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.id, message.id, event.account_scope_id, event.contact_id,
                    event.conversation_id or "", message.observed_at, event.created_at, event.created_at,
                    trace.get("capture_cycle_started_at"), trace.get("capture_started_at"),
                    trace.get("capture_completed_at"), trace.get("title_ocr_started_at"),
                    trace.get("title_ocr_completed_at"), trace.get("content_ocr_started_at"),
                    trace.get("content_ocr_completed_at"), trace.get("identity_started_at"),
                    trace.get("identity_stable_at"), trace.get("message_classified_at"),
                    trace.get("message_committed_at"),
                    json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    now, now,
                ),
            )

    def mark(self, event_id: str, field: str, value: str | None = None) -> None:
        if field not in self._FIELDS:
            raise ValueError(f"unsupported_reply_metric_field:{field}")
        now = value or utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE reply_pipeline_metrics SET {field}=COALESCE({field},?),updated_at=? WHERE event_id=?",
                (now, utc_now_iso(), event_id),
            )

    def mark_many(self, event_id: str, values: dict[str, str | None]) -> None:
        clean = {key: value for key, value in values.items() if key in self._FIELDS and value}
        if not clean:
            return
        assignments = ",".join(f"{key}=COALESCE({key},?)" for key in clean)
        params = [*clean.values(), utc_now_iso(), event_id]
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE reply_pipeline_metrics SET {assignments},updated_at=? WHERE event_id=?",
                params,
            )

    def attach_profiles(
        self,
        event_id: str,
        *,
        prompt_profile: dict[str, Any] | None = None,
        provider_profile: dict[str, Any] | None = None,
    ) -> None:
        values: list[str] = []
        assignments: list[str] = []
        if prompt_profile is not None:
            assignments.append("prompt_profile_json=?")
            values.append(json.dumps(prompt_profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if provider_profile is not None:
            assignments.append("provider_profile_json=?")
            values.append(json.dumps(provider_profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if not assignments:
            return
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE reply_pipeline_metrics SET {','.join(assignments)},updated_at=? WHERE event_id=?",
                (*values, utc_now_iso(), event_id),
            )

    def timing_summary(self, limit: int = 100) -> dict[str, Any]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reply_pipeline_metrics
                 WHERE reply_completed_at IS NOT NULL
                 ORDER BY event_created_at DESC LIMIT ?
                """,
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        samples = [dict(row) for row in rows]
        return {
            "samples": len(samples),
            "context_ms": _average_duration(samples, "context_started_at", "context_completed_at"),
            "reply_ms": _average_duration(samples, "reply_started_at", "reply_completed_at"),
            "recognized_to_reply_ms": _average_duration(samples, "message_observed_at", "reply_completed_at"),
            "recognized_to_ui_ms": _average_duration(samples, "message_observed_at", "ui_ready_at"),
            "event_to_ui_ms": _average_duration(samples, "event_created_at", "ui_ready_at"),
            "distributions": {
                "capture_to_provider_ms": _duration_distribution(samples, "capture_started_at", "provider_request_started_at"),
                "capture_ms": _duration_distribution(samples, "capture_started_at", "capture_completed_at"),
                "title_ocr_ms": _duration_distribution(samples, "title_ocr_started_at", "title_ocr_completed_at"),
                "content_ocr_ms": _duration_distribution(samples, "content_ocr_started_at", "content_ocr_completed_at"),
                "identity_ms": _duration_distribution(samples, "identity_started_at", "identity_stable_at"),
                "db_and_event_ms": _duration_distribution(samples, "message_classified_at", "reply_event_created_at"),
                "context_ms": _duration_distribution(samples, "context_started_at", "context_completed_at"),
                "reply_queue_ms": _duration_distribution(samples, "reply_queued_at", "reply_worker_started_at"),
                "provider_ms": _duration_distribution(samples, "provider_request_started_at", "provider_response_completed_at"),
                "parse_ms": _duration_distribution(samples, "provider_response_completed_at", "reply_parse_completed_at"),
                "result_write_ms": _duration_distribution(samples, "reply_parse_completed_at", "result_persisted_at"),
                "ui_dispatch_ms": _duration_distribution(samples, "result_persisted_at", "ui_event_received_at"),
                "ui_render_ms": _duration_distribution(samples, "ui_render_started_at", "ui_render_completed_at"),
                "end_to_end_ms": _duration_distribution(samples, "capture_started_at", "ui_render_completed_at"),
            },
        }


def _average_duration(rows: list[dict[str, Any]], start: str, end: str) -> int | None:
    values: list[float] = []
    for row in rows:
        if not row.get(start) or not row.get(end):
            continue
        try:
            started = datetime.fromisoformat(str(row[start]))
            finished = datetime.fromisoformat(str(row[end]))
        except ValueError:
            continue
        values.append(max(0.0, (finished - started).total_seconds() * 1000.0))
    return round(sum(values) / len(values)) if values else None


def _duration_distribution(rows: list[dict[str, Any]], start: str, end: str) -> dict[str, int | None]:
    values: list[float] = []
    for row in rows:
        if not row.get(start) or not row.get(end):
            continue
        try:
            started = datetime.fromisoformat(str(row[start]))
            finished = datetime.fromisoformat(str(row[end]))
        except ValueError:
            continue
        values.append(max(0.0, (finished - started).total_seconds() * 1000.0))
    if not values:
        return {"count": 0, "min": None, "median": None, "p75": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0]),
        "median": round(_percentile(ordered, 0.50)),
        "p75": round(_percentile(ordered, 0.75)),
        "p90": round(_percentile(ordered, 0.90)),
        "max": round(ordered[-1]),
    }


def _percentile(values: list[float], ratio: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * ratio
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
