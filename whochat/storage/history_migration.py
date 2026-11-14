from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from whochat.core.models import utc_now_iso
from whochat.storage.database import Database


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class BatchImportResult:
    processed: int
    committed: int
    duplicates: int
    checkpoint_sequence: int


class HistoryMigrationRepository:
    """v29 persistence for one-time local history migration.

    This repository only writes the historical archive and its audit metadata.
    It deliberately has no dependency on the live message-ingestion pipeline.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_source_account(
        self,
        *,
        source_account_id: str,
        stable_identifier: str,
        public_alias: str,
        source_data_directory: str,
        account_scope_id: str | None,
        target_instance_id: str | None,
        mapping_status: str,
        match_basis: str,
        confidence: float,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        if mapping_status not in {"resolved", "unresolved", "conflicting"}:
            raise ValueError("invalid_source_account_mapping_status")
        if mapping_status == "resolved" and not account_scope_id:
            raise ValueError("resolved_source_account_requires_scope")
        now = utc_now_iso()
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.db.connect() as conn:
            if account_scope_id:
                if conn.execute("SELECT 1 FROM account_scopes WHERE id=?", (account_scope_id,)).fetchone() is None:
                    raise ValueError("source_account_scope_not_found")
            if target_instance_id:
                target = conn.execute(
                    "SELECT account_scope_id FROM target_instances WHERE id=?", (target_instance_id,)
                ).fetchone()
                if target is None or (account_scope_id and target["account_scope_id"] != account_scope_id):
                    raise ValueError("source_target_instance_scope_mismatch")
            row = conn.execute(
                "SELECT id FROM historical_source_accounts WHERE source_account_id=?",
                (source_account_id,),
            ).fetchone()
            record_id = str(row["id"]) if row else _id("history_account")
            conn.execute(
                """INSERT INTO historical_source_accounts(
                       id,source_account_id,stable_identifier,public_alias,source_data_directory,
                       account_scope_id,target_instance_id,mapping_status,match_basis,confidence,
                       manifest_json,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_account_id) DO UPDATE SET
                       stable_identifier=excluded.stable_identifier,
                       public_alias=excluded.public_alias,
                       source_data_directory=excluded.source_data_directory,
                       account_scope_id=excluded.account_scope_id,
                       target_instance_id=excluded.target_instance_id,
                       mapping_status=excluded.mapping_status,
                       match_basis=excluded.match_basis,
                       confidence=excluded.confidence,
                       manifest_json=excluded.manifest_json,
                       updated_at=excluded.updated_at""",
                (
                    record_id, source_account_id, stable_identifier, public_alias,
                    source_data_directory, account_scope_id, target_instance_id, mapping_status,
                    match_basis, max(0.0, min(1.0, float(confidence))), payload, now, now,
                ),
            )
            return dict(conn.execute(
                "SELECT * FROM historical_source_accounts WHERE source_account_id=?",
                (source_account_id,),
            ).fetchone())

    def upsert_snapshot(
        self,
        *,
        source_account_record_id: str,
        import_source_id: str | None,
        snapshot_label: str,
        source_directory: str,
        decrypted_directory: str,
        status: str,
        snapshot_started_at: str,
        snapshot_completed_at: str | None,
        latest_historical_message_at: str | None,
        cutover_at: str | None,
        database_count: int,
        database_size_bytes: int,
        message_count: int,
        snapshot_hash: str,
        manifest: dict[str, Any],
        key_material_status: str = "pending_cleanup",
    ) -> dict[str, Any]:
        now = utc_now_iso()
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT id FROM historical_snapshots
                   WHERE source_account_record_id=? AND snapshot_label=?""",
                (source_account_record_id, snapshot_label),
            ).fetchone()
            snapshot_id = str(row["id"]) if row else _id("history_snapshot")
            conn.execute(
                """INSERT INTO historical_snapshots(
                       id,source_account_record_id,import_source_id,snapshot_label,source_directory,
                       decrypted_directory,status,snapshot_started_at,snapshot_completed_at,
                       latest_historical_message_at,cutover_at,database_count,database_size_bytes,
                       message_count,snapshot_hash,manifest_json,key_material_status,
                       key_material_deleted_at,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)
                   ON CONFLICT(source_account_record_id,snapshot_label) DO UPDATE SET
                       import_source_id=excluded.import_source_id,
                       source_directory=excluded.source_directory,
                       decrypted_directory=excluded.decrypted_directory,
                       status=excluded.status,
                       snapshot_completed_at=excluded.snapshot_completed_at,
                       latest_historical_message_at=excluded.latest_historical_message_at,
                       cutover_at=excluded.cutover_at,
                       database_count=excluded.database_count,
                       database_size_bytes=excluded.database_size_bytes,
                       message_count=excluded.message_count,
                       snapshot_hash=excluded.snapshot_hash,
                       manifest_json=excluded.manifest_json,
                       key_material_status=excluded.key_material_status,
                       updated_at=excluded.updated_at""",
                (
                    snapshot_id, source_account_record_id, import_source_id, snapshot_label,
                    source_directory, decrypted_directory, status, snapshot_started_at,
                    snapshot_completed_at, latest_historical_message_at, cutover_at,
                    int(database_count), int(database_size_bytes), int(message_count),
                    snapshot_hash, payload, key_material_status, now, now,
                ),
            )
            return dict(conn.execute(
                "SELECT * FROM historical_snapshots WHERE id=?", (snapshot_id,)
            ).fetchone())

    def attach_snapshot_source(self, snapshot_id: str, import_source_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE historical_snapshots SET import_source_id=?,updated_at=? WHERE id=?",
                (import_source_id, utc_now_iso(), snapshot_id),
            )

    def upsert_mappings_batch(
        self,
        *,
        import_source_id: str,
        account_scope_id: str,
        conversations: Iterable[dict[str, Any]],
        resolved_overrides: dict[str, tuple[str, str]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Create source-contact mappings in one transaction.

        Stable source identifiers are stored for every conversation, but no
        nickname heuristic can resolve a customer.  Only explicit overrides or
        an already-confirmed mapping may produce ``resolved``.
        """
        now = utc_now_iso()
        overrides = resolved_overrides or {}
        items = list(conversations)
        result: dict[str, dict[str, Any]] = {}
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM account_scopes WHERE id=?", (account_scope_id,)).fetchone() is None:
                raise ValueError("history_account_scope_not_found")
            for key, (contact_id, conversation_id) in overrides.items():
                row = conn.execute(
                    """SELECT c.account_scope_id AS contact_scope,
                              v.account_scope_id AS conversation_scope,v.contact_id
                       FROM contacts c JOIN conversations v ON v.id=?
                       WHERE c.id=?""",
                    (conversation_id, contact_id),
                ).fetchone()
                if (
                    row is None or row["contact_scope"] != account_scope_id
                    or row["conversation_scope"] != account_scope_id or row["contact_id"] != contact_id
                ):
                    raise ValueError(f"history_override_scope_mismatch:{key}")
            for item in items:
                key = str(item["mapping_key"])
                existing = conn.execute(
                    """SELECT * FROM historical_contact_mappings
                       WHERE import_source_id=? AND account_scope_id=? AND mapping_key=?""",
                    (import_source_id, account_scope_id, key),
                ).fetchone()
                override = overrides.get(key)
                if override:
                    contact_id, conversation_id = override
                    status, basis, confidence = "resolved", "explicit_user_named_pilot", 1.0
                elif existing is not None and existing["status"] == "resolved":
                    contact_id, conversation_id = existing["contact_id"], existing["conversation_id"]
                    status, basis, confidence = "resolved", existing["match_basis"], float(existing["confidence"])
                else:
                    contact_id = conversation_id = None
                    status, basis, confidence = "unresolved", "stable_source_identifier_requires_explicit_mapping", 0.0
                mapping_id = str(existing["id"]) if existing else _id("history_mapping")
                conn.execute(
                    """INSERT INTO historical_contact_mappings(
                           id,import_source_id,mapping_key,account_scope_id,external_contact_id,
                           external_conversation_id,source_display_name,contact_id,conversation_id,
                           status,match_basis,confidence,created_at,updated_at
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(import_source_id,account_scope_id,mapping_key) DO UPDATE SET
                           external_contact_id=excluded.external_contact_id,
                           external_conversation_id=excluded.external_conversation_id,
                           source_display_name=excluded.source_display_name,
                           contact_id=excluded.contact_id,conversation_id=excluded.conversation_id,
                           status=excluded.status,match_basis=excluded.match_basis,
                           confidence=excluded.confidence,updated_at=excluded.updated_at""",
                    (
                        mapping_id, import_source_id, key, account_scope_id,
                        str(item.get("external_contact_id") or ""),
                        str(item.get("external_conversation_id") or ""),
                        str(item.get("display_name") or ""), contact_id, conversation_id,
                        status, basis, confidence, now, now,
                    ),
                )
            rows = conn.execute(
                """SELECT * FROM historical_contact_mappings
                   WHERE import_source_id=? AND account_scope_id=?""",
                (import_source_id, account_scope_id),
            ).fetchall()
            result = {str(row["mapping_key"]): dict(row) for row in rows}
        return result

    def mark_snapshot_imported(self, snapshot_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE historical_snapshots SET status='imported',updated_at=? WHERE id=?",
                (utc_now_iso(), snapshot_id),
            )

    def mark_key_material_deleted(self, snapshot_ids: Iterable[str]) -> None:
        ids = tuple(dict.fromkeys(str(item) for item in snapshot_ids if item))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                f"""UPDATE historical_snapshots SET key_material_status='deleted',
                    key_material_deleted_at=?,updated_at=? WHERE id IN ({placeholders})""",
                (now, now, *ids),
            )

    def commit_messages_batch(
        self,
        *,
        job_id: str,
        snapshot_id: str,
        rows: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    ) -> BatchImportResult:
        """Insert a bounded batch and advance its checkpoint atomically."""
        materialized = list(rows)
        if not materialized:
            job = self._job(job_id)
            return BatchImportResult(0, 0, 0, int(job["checkpoint_sequence"]))
        now = utc_now_iso()
        committed = duplicates = 0
        checkpoint = -1
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute("SELECT * FROM historical_import_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise ValueError("historical_import_job_not_found")
            for mapping, message in materialized:
                if mapping["account_scope_id"] != job["account_scope_id"]:
                    raise ValueError("history_job_mapping_scope_mismatch")
                sequence = int(message["source_sequence"])
                checkpoint = max(checkpoint, sequence)
                historical_id = _id("historical_message")
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO historical_chat_messages(
                           historical_message_id,import_source_id,import_job_id,mapping_id,
                           account_scope_id,contact_id,conversation_id,external_contact_id,
                           external_conversation_id,external_message_id,sender,message_type,text,
                           sent_at,source_sequence,source_hash,mapping_status,imported_at,
                           raw_metadata_json,snapshot_id
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        historical_id, job["import_source_id"], job_id, mapping["id"],
                        job["account_scope_id"], mapping.get("contact_id"),
                        mapping.get("conversation_id"), mapping.get("external_contact_id", ""),
                        mapping.get("external_conversation_id", ""),
                        message.get("external_message_id", ""), message["sender"],
                        message.get("message_type", "text"), message.get("text", ""),
                        message["sent_at"], sequence, message["source_hash"],
                        mapping["status"], now,
                        json.dumps(message.get("raw_metadata", {}), ensure_ascii=False,
                                   sort_keys=True, separators=(",", ":")),
                        snapshot_id,
                    ),
                )
                created = cursor.rowcount == 1
                committed += int(created)
                duplicates += int(not created)
            conn.execute(
                """UPDATE historical_import_jobs SET
                       checkpoint_sequence=MAX(checkpoint_sequence,?),
                       committed_count=committed_count+?,duplicate_count=duplicate_count+?,
                       updated_at=? WHERE id=?""",
                (checkpoint, committed, duplicates, now, job_id),
            )
        return BatchImportResult(len(materialized), committed, duplicates, checkpoint)

    def _job(self, job_id: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM historical_import_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("historical_import_job_not_found")
        return dict(row)
