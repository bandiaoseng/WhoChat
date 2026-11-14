from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from whochat.storage.database import Database


OPEN_STATES = (
    "selecting", "prepared", "authorized", "running", "paused_manual",
    "paused_automatic", "paused_realtime", "paused_window", "paused_budget",
    "paused_quality", "paused_technical",
)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class HistoricalSchedulerRepository:
    """Persistent scheduler boundary; extraction stays in existing v28 tables."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def control(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM historical_scheduler_control WHERE id='default'"
            ).fetchone()
        if row is None:
            raise RuntimeError("historical_scheduler_control_missing")
        return dict(row)

    def set_enabled(self, enabled: bool, reason: str = "") -> dict[str, Any]:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE historical_scheduler_control SET enabled=?,pause_reason=?,updated_at=? WHERE id='default'",
                (int(enabled), reason, _now()),
            )
        return self.control()

    def set_manual_pause(self, paused: bool, reason: str) -> dict[str, Any]:
        now = _now()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE historical_scheduler_control SET manual_paused=?,pause_reason=?,updated_at=? WHERE id='default'",
                (int(paused), reason, now),
            )
            if paused:
                conn.execute(
                    """UPDATE historical_scheduler_batches SET status='paused_manual',
                       stop_reason=?,updated_at=?
                       WHERE id=(SELECT current_batch_id FROM historical_scheduler_control WHERE id='default')
                         AND status IN ('authorized','running','paused_realtime','paused_window')""",
                    (reason, now),
                )
        return self.control()

    def set_automatic_pause(self, paused: bool, reason: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE historical_scheduler_control SET automatic_paused=?,pause_reason=?,updated_at=? WHERE id='default'",
                (int(paused), reason, _now()),
            )
        return self.control()

    def latest_open_batch(self) -> dict[str, Any] | None:
        marks = ",".join("?" for _ in OPEN_STATES)
        with self.db.connect() as conn:
            row = conn.execute(
                f"SELECT * FROM historical_scheduler_batches WHERE status IN ({marks}) ORDER BY sequence_no DESC LIMIT 1",
                OPEN_STATES,
            ).fetchone()
        return dict(row) if row else None

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM historical_scheduler_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if row is None:
                raise KeyError("historical_scheduler_batch_not_found")
            result = dict(row)
            result["customers"] = [
                dict(item) for item in conn.execute(
                    "SELECT * FROM historical_scheduler_batch_customers WHERE batch_id=? ORDER BY priority_rank",
                    (batch_id,),
                ).fetchall()
            ]
        return result

    def create_prepared_batch(
        self, *, phase: str, scheduled_local_date: str, planned_chunks: int,
        provider_call_budget: int, max_runtime_seconds: int,
        config_snapshot: dict[str, Any], customers: Iterable[dict[str, Any]],
        database_size_before: int,
    ) -> dict[str, Any]:
        rows = list(customers)
        if not rows or len(rows) > 20:
            raise ValueError("historical_scheduler_customer_count_invalid")
        if sum(int(row["expected_chunks"]) for row in rows) != int(planned_chunks):
            raise ValueError("historical_scheduler_chunk_sum_mismatch")
        if int(provider_call_budget) != int(planned_chunks) + 3:
            raise ValueError("historical_scheduler_call_budget_must_equal_chunks_plus_three")
        now, batch_id = _now(), _id("historical_batch")
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT id,status FROM historical_scheduler_batches WHERE status NOT IN ('completed','failed') LIMIT 1"
            ).fetchone()
            if current:
                raise ValueError(f"historical_scheduler_open_batch_exists:{current['id']}")
            sequence = int(conn.execute(
                "SELECT COALESCE(MAX(sequence_no),0)+1 FROM historical_scheduler_batches"
            ).fetchone()[0])
            observation = int(conn.execute(
                "SELECT observation_batches_completed FROM historical_scheduler_control WHERE id='default'"
            ).fetchone()[0]) + 1
            conn.execute(
                """INSERT INTO historical_scheduler_batches(
                   id,sequence_no,phase,status,observation_number,scheduled_local_date,
                   selected_customers,planned_chunks,provider_call_budget,provider_calls_used,
                   max_runtime_seconds,config_snapshot_json,metrics_json,stop_reason,
                   database_size_before,created_at,updated_at)
                   VALUES (?,?,?,'prepared',?,?,?,?,?,0,?,?,'{}',?,?,?,?)""",
                (batch_id, sequence, phase, observation, scheduled_local_date, len(rows),
                 planned_chunks, provider_call_budget, max_runtime_seconds,
                 _json(config_snapshot), "awaiting_first_long_running_batch_authorization",
                 database_size_before, now, now),
            )
            for rank, row in enumerate(rows, 1):
                conn.execute(
                    """INSERT INTO historical_scheduler_batch_customers(
                       batch_id,safe_contact_reference,mapping_id,account_scope_id,contact_id,
                       conversation_id,value_tier,priority_score,priority_rank,expected_chunks,
                       state,identity_basis,identity_evidence_json,error,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,'identity_validated',?,?,?,?,?)""",
                    (batch_id, row["safe_contact_reference"], row["mapping_id"],
                     row["account_scope_id"], row.get("contact_id"), row.get("conversation_id"),
                     row["value_tier"], int(row["priority_score"]), rank,
                     int(row["expected_chunks"]), row["identity_basis"],
                     _json(row.get("identity_evidence") or {}), str(row.get("error") or ""),
                     now, now),
                )
            conn.execute(
                """UPDATE historical_scheduler_control SET current_batch_id=?,manual_paused=1,
                   pause_reason='awaiting_first_long_running_batch_authorization',updated_at=?
                   WHERE id='default'""", (batch_id, now),
            )
        return self.get_batch(batch_id)

    def authorize_batch(self, batch_id: str) -> dict[str, Any]:
        now = _now()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM historical_scheduler_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if row is None or row["status"] not in {"prepared", "paused_manual"}:
                raise ValueError("historical_scheduler_batch_not_authorizable")
            conn.execute(
                "UPDATE historical_scheduler_batches SET status='authorized',authorized_at=?,stop_reason='',updated_at=? WHERE id=?",
                (now, now, batch_id),
            )
            conn.execute(
                """UPDATE historical_scheduler_control SET enabled=1,manual_paused=0,
                   automatic_paused=0,pause_reason='',updated_at=? WHERE id='default'""", (now,),
            )
        return self.get_batch(batch_id)

    def update_batch_state(self, batch_id: str, *, status: str, stop_reason: str = "",
                           metrics: dict[str, Any] | None = None,
                           database_size_after: int | None = None,
                           report_json_path: str | None = None,
                           report_markdown_path: str | None = None) -> dict[str, Any]:
        now = _now(); terminal = now if status in {"completed", "failed"} else None
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """UPDATE historical_scheduler_batches SET status=?,stop_reason=?,
                   metrics_json=COALESCE(?,metrics_json),database_size_after=COALESCE(?,database_size_after),
                   report_json_path=COALESCE(?,report_json_path),
                   report_markdown_path=COALESCE(?,report_markdown_path),
                   started_at=CASE WHEN ?='running' THEN COALESCE(started_at,?) ELSE started_at END,
                   completed_at=COALESCE(?,completed_at),updated_at=? WHERE id=?""",
                (status, stop_reason, _json(metrics) if metrics is not None else None,
                 database_size_after, report_json_path, report_markdown_path,
                 status, now, terminal, now, batch_id),
            )
            if status == "completed":
                conn.execute(
                    """UPDATE historical_scheduler_control SET
                       observation_batches_completed=observation_batches_completed+1,
                       last_batch_local_date=(SELECT scheduled_local_date FROM historical_scheduler_batches WHERE id=?),
                       current_batch_id=NULL,updated_at=? WHERE id='default'""", (batch_id, now),
                )
        return self.get_batch(batch_id)

    def update_customer_state(self, *, batch_id: str, safe_contact_reference: str,
                              state: str, contact_id: str | None = None,
                              conversation_id: str | None = None,
                              backfill_job_id: str | None = None, error: str = "") -> None:
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE historical_scheduler_batch_customers SET state=?,
                   contact_id=COALESCE(?,contact_id),conversation_id=COALESCE(?,conversation_id),
                   backfill_job_id=COALESCE(?,backfill_job_id),error=?,updated_at=?
                   WHERE batch_id=? AND safe_contact_reference=?""",
                (state, contact_id, conversation_id, backfill_job_id, error, _now(),
                 batch_id, safe_contact_reference),
            )

    def provider_calls_between(self, start_utc: str, end_utc: str) -> int:
        with self.db.connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM historical_scheduler_provider_calls WHERE started_at>=? AND started_at<?",
                (start_utc, end_utc),
            ).fetchone()[0])

    def record_provider_call_started(self, *, batch_id: str, backfill_job_id: str,
                                     chunk_id: str, attempt_no: int, provider: str, model: str,
                                     calls_today: int, daily_cap: int) -> str:
        now, call_id = _now(), _id("historical_call")
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            batch = conn.execute(
                "SELECT provider_call_budget,provider_calls_used,status FROM historical_scheduler_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
            if batch is None or batch["status"] != "running":
                raise ValueError("historical_scheduler_batch_not_running")
            if int(batch["provider_calls_used"]) >= int(batch["provider_call_budget"]):
                raise ValueError("historical_scheduler_batch_call_cap")
            if calls_today >= daily_cap:
                raise ValueError("historical_scheduler_daily_call_cap")
            conn.execute(
                """INSERT INTO historical_scheduler_provider_calls(
                   id,batch_id,backfill_job_id,chunk_id,attempt_no,call_kind,outcome,
                   provider,model,started_at,created_at,updated_at)
                   VALUES (?,?,?,?,?,?, 'started',?,?,?,?,?)""",
                (call_id, batch_id, backfill_job_id, chunk_id, attempt_no,
                 "initial" if attempt_no == 1 else "transport_retry", provider, model,
                 now, now, now),
            )
            conn.execute(
                "UPDATE historical_scheduler_batches SET provider_calls_used=provider_calls_used+1,updated_at=? WHERE id=?",
                (now, batch_id),
            )
        return call_id

    def finish_provider_call(self, call_id: str, *, outcome: str, elapsed_ms: int,
                             error_class: str = "") -> None:
        now = _now()
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE historical_scheduler_provider_calls SET outcome=?,completed_at=?,
                   elapsed_ms=?,error_class=?,updated_at=? WHERE id=?""",
                (outcome, now, max(0, elapsed_ms), error_class, now, call_id),
            )

    def add_incident(self, *, batch_id: str | None, incident_type: str, reason: str,
                     evidence: dict[str, Any], circuit_breaker: bool = True) -> str:
        incident_id = _id("historical_incident")
        with self.db.connect() as conn:
            conn.execute(
                """INSERT INTO historical_scheduler_incidents(
                   id,batch_id,incident_type,severity,reason,evidence_json,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (incident_id, batch_id, incident_type,
                 "circuit_breaker" if circuit_breaker else "warning",
                 reason, _json(evidence), _now()),
            )
        return incident_id
