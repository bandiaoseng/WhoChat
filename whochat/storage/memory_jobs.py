from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from whochat.core.models import (
    LEGACY_ACCOUNT_SCOPE_ID,
    MemoryExtractionJob,
    MemoryExtractionJobStatus,
    Message,
    Speaker,
    utc_now_iso,
)
from whochat.storage.database import Database


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class MemoryExtractionJobRepository:
    """Durable once-per-message queue for structured customer-memory extraction."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_activation_cursor(self) -> int:
        """Start after the current tail, so pre-v20 history is never backfilled."""
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT last_message_rowid FROM memory_extraction_state WHERE id = 'global'").fetchone()
            if row is not None:
                return int(row["last_message_rowid"])
            maximum = int(conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM messages").fetchone()[0])
            conn.execute(
                """
                INSERT INTO memory_extraction_state(
                    id, activated_at, last_message_rowid, last_scan_at, created_at, updated_at
                ) VALUES ('global', ?, ?, ?, ?, ?)
                """,
                (now, maximum, now, now, now),
            )
            return maximum

    def discover_new_messages(self, *, provider: str, model: str, limit: int = 100) -> list[MemoryExtractionJob]:
        """Create at most one job for each post-activation formal message."""
        self.ensure_activation_cursor()
        now = utc_now_iso()
        created: list[MemoryExtractionJob] = []
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            state = conn.execute(
                "SELECT last_message_rowid FROM memory_extraction_state WHERE id = 'global'"
            ).fetchone()
            cursor = int(state["last_message_rowid"])
            rows = conn.execute(
                """
                SELECT
                    m.rowid AS source_message_rowid,
                    m.*,
                    c.status AS contact_status,
                    c.merged_into AS contact_merged_into,
                    c.allow_cloud_ai AS contact_allow_cloud_ai,
                    c.account_scope_id AS contact_scope_id,
                    c.conversation_type AS contact_conversation_type,
                    a.status AS account_scope_status,
                    cv.account_scope_id AS conversation_scope_id,
                    cv.contact_id AS conversation_contact_id,
                    s.account_scope_id AS session_scope_id,
                    s.conversation_id AS session_conversation_id,
                    sg.account_scope_id AS segment_scope_id,
                    sg.conversation_id AS segment_conversation_id,
                    sg.session_id AS segment_session_id,
                    sg.target_instance_id AS segment_target_instance_id,
                    sg.reliable AS segment_reliable,
                    sg.order_uncertain AS segment_order_uncertain,
                    ti.account_scope_id AS target_scope_id
                FROM messages m
                LEFT JOIN contacts c ON c.id = m.contact_id
                LEFT JOIN account_scopes a ON a.id = m.account_scope_id
                LEFT JOIN conversations cv ON cv.id = m.conversation_id
                LEFT JOIN conversation_sessions s ON s.id = m.session_id
                LEFT JOIN conversation_segments sg ON sg.id = m.segment_id
                LEFT JOIN target_instances ti ON ti.id = NULLIF(sg.target_instance_id, '')
                WHERE m.rowid > ?
                ORDER BY m.rowid ASC
                LIMIT ?
                """,
                (cursor, max(1, int(limit))),
            ).fetchall()
            last_rowid = cursor
            for row in rows:
                last_rowid = int(row["source_message_rowid"])
                blocked_reason = _eligibility_error(row)
                status = MemoryExtractionJobStatus.BLOCKED if blocked_reason else MemoryExtractionJobStatus.PENDING
                target_instance_id = str(row["segment_target_instance_id"] or "").strip() or None
                item = MemoryExtractionJob(
                    id=_new_id("memory_job"),
                    source_message_id=row["id"],
                    source_message_rowid=last_rowid,
                    account_scope_id=row["account_scope_id"],
                    target_instance_id=target_instance_id,
                    contact_id=row["contact_id"],
                    conversation_id=row["conversation_id"],
                    customer_profile_id=None,
                    status=status,
                    provider=provider,
                    model=model,
                    input_hash="",
                    input_context_json="{}",
                    output_json="{}",
                    write_result_json="{}",
                    error=blocked_reason or "",
                    ai_call_count=0,
                    provider_elapsed_ms=None,
                    created_at=now,
                    started_at=None,
                    completed_at=now if blocked_reason else None,
                    updated_at=now,
                )
                inserted = conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_extraction_jobs(
                        id, source_message_id, source_message_rowid, account_scope_id,
                        target_instance_id, contact_id, conversation_id, customer_profile_id,
                        status, provider, model, input_hash, input_context_json, output_json,
                        write_result_json, error, ai_call_count, provider_elapsed_ms,
                        created_at, started_at, completed_at, updated_at
                    ) VALUES (
                        :id, :source_message_id, :source_message_rowid, :account_scope_id,
                        :target_instance_id, :contact_id, :conversation_id, :customer_profile_id,
                        :status, :provider, :model, :input_hash, :input_context_json, :output_json,
                        :write_result_json, :error, :ai_call_count, :provider_elapsed_ms,
                        :created_at, :started_at, :completed_at, :updated_at
                    )
                    """,
                    _job_params(item),
                )
                if inserted.rowcount:
                    created.append(item)
            conn.execute(
                """
                UPDATE memory_extraction_state
                SET last_message_rowid = ?, last_scan_at = ?, updated_at = ?
                WHERE id = 'global'
                """,
                (last_rowid, now, now),
            )
        return created

    def get(self, job_id: str) -> MemoryExtractionJob | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM memory_extraction_jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_from_row(row) if row else None

    def get_for_message(self, message_id: str) -> MemoryExtractionJob | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_extraction_jobs WHERE source_message_id = ?", (message_id,)
            ).fetchone()
        return _job_from_row(row) if row else None

    def list_by_status(self, status: MemoryExtractionJobStatus | str) -> list[MemoryExtractionJob]:
        value = MemoryExtractionJobStatus(status).value
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_extraction_jobs WHERE status = ? ORDER BY created_at, rowid", (value,)
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def recover_interrupted(self) -> tuple[int, int]:
        """Never repeat a possibly-sent provider request after a crash."""
        now = utc_now_iso()
        with self.db.connect() as conn:
            resumable = conn.execute(
                """
                UPDATE memory_extraction_jobs
                SET status = 'pending', started_at = NULL, updated_at = ?,
                    error = 'restart_resume_local_output'
                WHERE status = 'running' AND output_json <> '{}'
                """,
                (now,),
            ).rowcount
            untouched = conn.execute(
                """
                UPDATE memory_extraction_jobs
                SET status = 'pending', started_at = NULL, updated_at = ?,
                    error = 'restart_before_provider_call'
                WHERE status = 'running' AND ai_call_count = 0 AND output_json = '{}'
                """,
                (now,),
            ).rowcount
            failed = conn.execute(
                """
                UPDATE memory_extraction_jobs
                SET status = 'failed', completed_at = ?, updated_at = ?,
                    error = 'interrupted_after_provider_call_no_retry'
                WHERE status = 'running' AND ai_call_count = 1 AND output_json = '{}'
                """,
                (now, now),
            ).rowcount
        return resumable + untouched, failed

    def claim_next(self, *, reply_priority_grace_seconds: int = 30) -> tuple[MemoryExtractionJob, bool] | None:
        """Return ``(job, provider_call_required)`` and reserve one job.

        A new customer message can create both a reply-analysis event and a
        memory job.  They remain independent durable queues, but the user-facing
        reply gets a short head start: while the matching reply event is pending
        or running, a fresh memory job is not claimed.  The bounded grace window
        prevents a crashed or unavailable reply worker from starving memory
        extraction forever; normal restart recovery remains unchanged.
        """
        now = utc_now_iso()
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=max(0, int(reply_priority_grace_seconds)))
        ).isoformat()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT job.*
                  FROM memory_extraction_jobs AS job
                  LEFT JOIN ai_analysis_events AS event
                    ON event.message_id = job.source_message_id
                 WHERE job.status = 'pending'
                   AND (
                       event.id IS NULL
                       OR event.status NOT IN ('pending','running')
                       OR event.created_at <= ?
                   )
                 ORDER BY job.created_at, job.rowid
                 LIMIT 1
                """,
                (cutoff,),
            ).fetchone()
            if row is None:
                return None
            has_output = row["output_json"] != "{}"
            if not has_output and int(row["ai_call_count"]) >= 1:
                conn.execute(
                    """
                    UPDATE memory_extraction_jobs
                    SET status = 'failed', completed_at = ?, updated_at = ?, error = 'provider_call_already_consumed'
                    WHERE id = ?
                    """,
                    (now, now, row["id"]),
                )
                return None
            if has_output:
                conn.execute(
                    "UPDATE memory_extraction_jobs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ?",
                    (now, now, row["id"]),
                )
                requires_provider = False
            else:
                conn.execute(
                    "UPDATE memory_extraction_jobs SET status = 'running', started_at = ?, error = '', updated_at = ? WHERE id = ? AND status = 'pending'",
                    (now, now, row["id"]),
                )
                requires_provider = True
            claimed = conn.execute("SELECT * FROM memory_extraction_jobs WHERE id = ?", (row["id"],)).fetchone()
        return _job_from_row(claimed), requires_provider

    def begin_provider_call(self, job_id: str) -> bool:
        """Consume the single provider-call allowance immediately before network I/O."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_extraction_jobs
                SET ai_call_count = 1, updated_at = ?
                WHERE id = ? AND status = 'running' AND ai_call_count = 0 AND output_json = '{}'
                """,
                (utc_now_iso(), job_id),
            )
        return cursor.rowcount == 1

    def load_source_message(self, job: MemoryExtractionJob) -> Message:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (job.source_message_id,)).fetchone()
        if row is None:
            raise ValueError("source message no longer exists")
        message = _message_from_row(row)
        if (
            message.account_scope_id != job.account_scope_id
            or message.contact_id != job.contact_id
            or message.conversation_id != job.conversation_id
        ):
            raise ValueError("source message identity changed")
        return message

    def list_context_messages(self, job: MemoryExtractionJob, limit: int = 10) -> list[Message]:
        source = self.load_source_message(job)
        with self.db.connect() as conn:
            if source.sequence_no is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE account_scope_id = ? AND contact_id = ? AND conversation_id = ?
                      AND sequence_no <= ?
                    ORDER BY sequence_no DESC, observed_at DESC, id DESC
                    LIMIT ?
                    """,
                    (
                        job.account_scope_id,
                        job.contact_id,
                        job.conversation_id,
                        source.sequence_no,
                        max(1, min(int(limit), 15)),
                    ),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE account_scope_id = ? AND contact_id = ? AND conversation_id = ?
                      AND observed_at <= ?
                    ORDER BY observed_at DESC, id DESC
                    LIMIT ?
                    """,
                    (
                        job.account_scope_id,
                        job.contact_id,
                        job.conversation_id,
                        source.observed_at,
                        max(1, min(int(limit), 15)),
                    ),
                ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def save_input(
        self,
        job_id: str,
        *,
        customer_profile_id: str | None,
        input_hash: str,
        input_context_json: str,
    ) -> None:
        json.loads(input_context_json)
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE memory_extraction_jobs
                SET customer_profile_id = ?, input_hash = ?, input_context_json = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (customer_profile_id, input_hash, input_context_json, utc_now_iso(), job_id),
            )

    def attach_customer_profile(self, job_id: str, customer_profile_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE memory_extraction_jobs SET customer_profile_id=?,updated_at=? WHERE id=?",
                (customer_profile_id, utc_now_iso(), job_id),
            )

    def save_provider_output(self, job_id: str, output_json: str, elapsed_ms: int | None) -> None:
        json.loads(output_json)
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE memory_extraction_jobs
                SET output_json = ?, provider_elapsed_ms = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (output_json, elapsed_ms, utc_now_iso(), job_id),
            )

    def mark_performance(self, job_id: str, field: str, value: str | None = None) -> None:
        allowed = {
            "provider_started_at", "provider_completed_at",
            "write_started_at", "write_completed_at",
        }
        if field not in allowed:
            raise ValueError(f"unsupported_memory_performance_field:{field}")
        now = value or utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE memory_extraction_jobs SET {field}=COALESCE({field},?),updated_at=? WHERE id=?",
                (now, utc_now_iso(), job_id),
            )

    def finish(
        self,
        job_id: str,
        status: MemoryExtractionJobStatus | str,
        *,
        write_result_json: str = "{}",
        error: str = "",
    ) -> None:
        status = MemoryExtractionJobStatus(status)
        if status in {MemoryExtractionJobStatus.PENDING, MemoryExtractionJobStatus.RUNNING}:
            raise ValueError("finish requires a terminal job status")
        json.loads(write_result_json)
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE memory_extraction_jobs
                SET status = ?, write_result_json = ?, error = ?, completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, write_result_json, error[:2000], now, now, job_id),
            )

    def store_invalid_output(self, job_id: str, raw_output: str, error: str) -> None:
        wrapped = json.dumps({"invalid_raw": raw_output[:10000]}, ensure_ascii=False, separators=(",", ":"))
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE memory_extraction_jobs SET output_json = ?, updated_at = ? WHERE id = ?",
                (wrapped, utc_now_iso(), job_id),
            )
        self.finish(job_id, MemoryExtractionJobStatus.INVALID_OUTPUT, error=error)

    def counts(self) -> dict[str, int]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM memory_extraction_jobs GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}


def _eligibility_error(row: sqlite3.Row) -> str | None:
    if row["speaker"] not in {Speaker.ME.value, Speaker.OTHER.value}:
        return "unsupported_sender"
    if row["content_type"] != "text" or not str(row["text"] or "").strip():
        return "unsupported_content"
    if bool(row["partial"]) or bool(row["order_uncertain"]):
        return "uncertain_message"
    if ";verified_new=" not in str(row["source"] or ""):
        return "not_verified_new_message"
    if not row["session_id"] or not row["segment_id"] or not row["conversation_id"]:
        return "missing_timeline_identity"
    if not row["account_scope_id"] or row["account_scope_id"] == LEGACY_ACCOUNT_SCOPE_ID:
        return "account_scope_unconfirmed"
    if row["account_scope_status"] != "active":
        return "account_scope_inactive"
    if row["contact_scope_id"] != row["account_scope_id"]:
        return "contact_scope_mismatch"
    if row["contact_status"] != "confirmed" or row["contact_merged_into"]:
        return "contact_unconfirmed"
    if not bool(row["contact_allow_cloud_ai"]):
        return "contact_cloud_ai_not_allowed"
    if row["contact_conversation_type"] != "dm":
        return "group_scope_not_supported"
    if (
        row["conversation_scope_id"] != row["account_scope_id"]
        or row["conversation_contact_id"] != row["contact_id"]
    ):
        return "conversation_identity_mismatch"
    if (
        row["session_scope_id"] != row["account_scope_id"]
        or row["session_conversation_id"] != row["conversation_id"]
    ):
        return "session_identity_mismatch"
    if (
        row["segment_scope_id"] != row["account_scope_id"]
        or row["segment_conversation_id"] != row["conversation_id"]
        or row["segment_session_id"] != row["session_id"]
        or not bool(row["segment_reliable"])
        or bool(row["segment_order_uncertain"])
    ):
        return "segment_unreliable"
    target_instance_id = str(row["segment_target_instance_id"] or "").strip()
    if not target_instance_id or row["target_scope_id"] != row["account_scope_id"]:
        return "target_instance_unconfirmed"
    confidence = row["ocr_confidence"]
    if confidence is not None and float(confidence) < 0.42:
        return "ocr_confidence_too_low"
    return None


def _job_from_row(row: sqlite3.Row) -> MemoryExtractionJob:
    data = dict(row)
    data["status"] = MemoryExtractionJobStatus(data["status"])
    return MemoryExtractionJob(**data)


def _job_params(job: MemoryExtractionJob) -> dict:
    data = asdict(job)
    data["status"] = job.status.value
    return data


def _message_from_row(row: sqlite3.Row) -> Message:
    data = dict(row)
    data.pop("source_message_rowid", None)
    for extra in (
        "contact_status", "contact_merged_into", "contact_allow_cloud_ai",
        "contact_scope_id", "contact_conversation_type", "account_scope_status",
        "conversation_scope_id", "conversation_contact_id", "session_scope_id",
        "session_conversation_id", "segment_scope_id", "segment_conversation_id",
        "segment_session_id", "segment_target_instance_id", "segment_reliable",
        "segment_order_uncertain", "target_scope_id",
    ):
        data.pop(extra, None)
    data["speaker"] = Speaker(data["speaker"])
    data["partial"] = bool(data["partial"])
    data["order_uncertain"] = bool(data.get("order_uncertain", 0))
    data.setdefault("sender_name", "")
    return Message(**data)
