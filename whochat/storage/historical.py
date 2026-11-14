from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from whochat.core.models import utc_now_iso
from whochat.storage.database import Database


MAPPING_STATES = {"resolved", "probable", "unresolved", "conflicting"}


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class HistoricalMessageRow:
    historical_message_id: str
    import_source_id: str
    import_job_id: str
    mapping_id: str
    account_scope_id: str
    contact_id: str | None
    conversation_id: str | None
    external_contact_id: str
    external_conversation_id: str
    external_message_id: str
    sender: str
    message_type: str
    text: str
    sent_at: str
    source_sequence: int
    source_hash: str
    mapping_status: str
    imported_at: str
    raw_metadata_json: str
    snapshot_id: str | None


class HistoricalArchiveRepository:
    """Persistence boundary for raw history, mapping and resumable jobs.

    Nothing in this repository writes to ``messages`` or any live-ingestion
    cursor.  That separation is a deliberate safety property, not merely an
    implementation detail.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def register_source(
        self,
        *,
        source_kind: str,
        display_name: str,
        source_reference: str = "",
        adapter_version: str = "1",
        source_hash: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = utc_now_iso()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.db.connect() as conn:
            if source_reference:
                row = conn.execute(
                    "SELECT id FROM historical_import_sources WHERE source_kind=? AND source_reference=?",
                    (source_kind, source_reference),
                ).fetchone()
                if row:
                    conn.execute(
                        """UPDATE historical_import_sources
                           SET display_name=?,adapter_version=?,source_hash=?,metadata_json=?,updated_at=?
                           WHERE id=?""",
                        (display_name, adapter_version, source_hash, metadata_json, now, row["id"]),
                    )
                    return str(row["id"])
            source_id = _id("history_source")
            conn.execute(
                """INSERT INTO historical_import_sources(
                       id,source_kind,display_name,source_reference,adapter_version,
                       source_hash,metadata_json,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    source_id, source_kind, display_name, source_reference, adapter_version,
                    source_hash, metadata_json, now, now,
                ),
            )
            return source_id

    def upsert_mapping(
        self,
        *,
        import_source_id: str,
        mapping_key: str,
        account_scope_id: str,
        external_contact_id: str,
        external_conversation_id: str,
        source_display_name: str,
        status: str,
        contact_id: str | None = None,
        conversation_id: str | None = None,
        match_basis: str = "",
        confidence: float = 0.0,
    ) -> dict[str, Any]:
        if status not in MAPPING_STATES:
            raise ValueError("invalid_history_mapping_status")
        now = utc_now_iso()
        with self.db.connect() as conn:
            self._validate_scope_targets(
                conn,
                account_scope_id=account_scope_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
                require_resolved=status == "resolved",
            )
            existing = conn.execute(
                """SELECT * FROM historical_contact_mappings
                   WHERE import_source_id=? AND account_scope_id=? AND mapping_key=?""",
                (import_source_id, account_scope_id, mapping_key),
            ).fetchone()
            if existing:
                # A previously confirmed mapping is never silently replaced by
                # nickname heuristics from a new import preview.
                if existing["status"] == "resolved" and status != "resolved":
                    return dict(existing)
                conn.execute(
                    """UPDATE historical_contact_mappings SET
                       external_contact_id=?,external_conversation_id=?,source_display_name=?,
                       contact_id=?,conversation_id=?,status=?,match_basis=?,confidence=?,updated_at=?
                       WHERE id=?""",
                    (
                        external_contact_id, external_conversation_id, source_display_name,
                        contact_id, conversation_id, status, match_basis,
                        max(0.0, min(1.0, float(confidence))), now, existing["id"],
                    ),
                )
                if status == "resolved":
                    self._attach_resolved_messages(
                        conn,
                        mapping_id=str(existing["id"]),
                        account_scope_id=account_scope_id,
                        contact_id=str(contact_id),
                        conversation_id=str(conversation_id),
                    )
                row = conn.execute(
                    "SELECT * FROM historical_contact_mappings WHERE id=?", (existing["id"],)
                ).fetchone()
                return dict(row)
            mapping_id = _id("history_mapping")
            conn.execute(
                """INSERT INTO historical_contact_mappings(
                    id,import_source_id,mapping_key,account_scope_id,external_contact_id,
                    external_conversation_id,source_display_name,contact_id,conversation_id,
                    status,match_basis,confidence,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mapping_id, import_source_id, mapping_key, account_scope_id,
                    external_contact_id, external_conversation_id, source_display_name,
                    contact_id, conversation_id, status, match_basis,
                    max(0.0, min(1.0, float(confidence))), now, now,
                ),
            )
            if status == "resolved":
                self._attach_resolved_messages(
                    conn,
                    mapping_id=mapping_id,
                    account_scope_id=account_scope_id,
                    contact_id=str(contact_id),
                    conversation_id=str(conversation_id),
                )
            row = conn.execute(
                "SELECT * FROM historical_contact_mappings WHERE id=?", (mapping_id,)
            ).fetchone()
            return dict(row)

    @staticmethod
    def _attach_resolved_messages(
        conn: sqlite3.Connection,
        *,
        mapping_id: str,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
    ) -> None:
        """Atomically bind already-imported raw history after explicit mapping.

        Full imports intentionally store unresolved rows before a user chooses
        customers.  Resolving a mapping later must update the row-level identity
        snapshot as part of the same transaction; otherwise the conservative
        business-message gate correctly treats every row as unresolved and
        prepares zero chunks.
        """
        invalid = conn.execute(
            """SELECT COUNT(*) AS n FROM historical_chat_messages
               WHERE mapping_id=? AND account_scope_id<>?""",
            (mapping_id, account_scope_id),
        ).fetchone()
        if int(invalid["n"] or 0):
            raise ValueError("historical_message_scope_mismatch")
        conn.execute(
            """UPDATE historical_chat_messages
               SET contact_id=?,conversation_id=?,mapping_status='resolved'
               WHERE mapping_id=? AND account_scope_id=?""",
            (contact_id, conversation_id, mapping_id, account_scope_id),
        )

    def enroll_explicit_local_identity(
        self,
        *,
        mapping_id: str,
        expected_account_scope_id: str,
        expected_external_contact_id: str,
        expected_external_conversation_id: str,
        safe_contact_reference: str,
        authorization_batch: str = "batch1",
        match_basis_prefix: str = "explicit_user_authorized",
    ) -> dict[str, Any]:
        """Atomically create a local identity for one user-authorised mapping.

        This method deliberately never searches by nickname to choose an
        existing contact.  A same-account display-name/alias collision is a
        review condition because only the stable source identifiers were
        authorised.  The display name remains presentation metadata; the
        mapping row retains the authoritative source identity.
        """
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            mapping = conn.execute(
                "SELECT * FROM historical_contact_mappings WHERE id=?",
                (mapping_id,),
            ).fetchone()
            if mapping is None:
                raise ValueError("historical_explicit_mapping_not_found")
            if mapping["account_scope_id"] != expected_account_scope_id:
                raise ValueError("historical_explicit_mapping_scope_changed")
            if mapping["external_contact_id"] != expected_external_contact_id:
                raise ValueError("historical_explicit_mapping_contact_identity_changed")
            if mapping["external_conversation_id"] != expected_external_conversation_id:
                raise ValueError("historical_explicit_mapping_conversation_identity_changed")
            if not expected_external_contact_id or not expected_external_conversation_id:
                raise ValueError("historical_explicit_mapping_stable_identity_missing")
            if not re.fullmatch(r"[a-z0-9_]{1,64}", authorization_batch):
                raise ValueError("historical_explicit_mapping_authorization_batch_invalid")
            if match_basis_prefix not in {
                "explicit_user_authorized", "deterministic_external_identity"
            }:
                raise ValueError("historical_mapping_basis_prefix_invalid")
            expected_basis = (
                f"{match_basis_prefix}_{authorization_batch}:{safe_contact_reference}"
            )
            if match_basis_prefix == "deterministic_external_identity":
                duplicate_identity = int(conn.execute(
                    """SELECT COUNT(*) FROM historical_contact_mappings
                       WHERE account_scope_id=? AND external_contact_id=?
                         AND external_conversation_id=?""",
                    (
                        expected_account_scope_id,
                        expected_external_contact_id,
                        expected_external_conversation_id,
                    ),
                ).fetchone()[0])
                if duplicate_identity != 1:
                    raise ValueError("historical_deterministic_identity_not_unique")
                evidence = conn.execute(
                    """SELECT COUNT(*) AS total,
                              SUM(CASE WHEN account_scope_id=?
                                        AND external_contact_id=?
                                        AND external_conversation_id=?
                                       THEN 1 ELSE 0 END) AS exact_rows,
                              COUNT(DISTINCT external_contact_id) AS contact_ids,
                              COUNT(DISTINCT external_conversation_id) AS conversation_ids
                       FROM historical_chat_messages WHERE mapping_id=?""",
                    (
                        expected_account_scope_id,
                        expected_external_contact_id,
                        expected_external_conversation_id,
                        mapping_id,
                    ),
                ).fetchone()
                if (
                    int(evidence["total"] or 0) <= 0
                    or int(evidence["exact_rows"] or 0) != int(evidence["total"] or 0)
                    or int(evidence["contact_ids"] or 0) != 1
                    or int(evidence["conversation_ids"] or 0) != 1
                ):
                    raise ValueError("historical_deterministic_conversation_evidence_invalid")
            if mapping["status"] == "resolved":
                if (
                    mapping["match_basis"] != expected_basis
                    or not mapping["contact_id"]
                    or not mapping["conversation_id"]
                ):
                    raise ValueError("historical_explicit_mapping_already_resolved_elsewhere")
                return {
                    "mapping_id": str(mapping["id"]),
                    "contact_id": str(mapping["contact_id"]),
                    "conversation_id": str(mapping["conversation_id"]),
                    "status": "already_resolved",
                }
            if mapping["status"] != "unresolved":
                raise ValueError("historical_explicit_mapping_requires_unresolved")
            display_name = str(mapping["source_display_name"] or "").strip()
            if not display_name:
                raise ValueError("historical_explicit_mapping_display_name_missing")
            collision = conn.execute(
                """SELECT contacts.id FROM contacts
                   LEFT JOIN contact_aliases ON contact_aliases.contact_id=contacts.id
                   WHERE contacts.account_scope_id=? AND contacts.platform='wechat'
                     AND contacts.merged_into IS NULL
                     AND (contacts.display_name=? OR contact_aliases.alias=?)
                   LIMIT 1""",
                (expected_account_scope_id, display_name, display_name),
            ).fetchone()
            if collision is not None:
                raise ValueError("historical_explicit_mapping_local_title_collision")
            strategy = conn.execute(
                "SELECT id FROM strategies WHERE id='default'"
            ).fetchone()
            if strategy is None:
                raise ValueError("historical_explicit_mapping_default_strategy_missing")
            contact_id = _id("contact")
            conversation_id = _id("conversation")
            conn.execute(
                """INSERT INTO contacts(
                       id,platform,display_name,conversation_type,status,strategy_id,
                       allow_cloud_ai,remark,avatar_fingerprint,merged_into,created_at,
                       updated_at,account_scope_id
                   ) VALUES (?,'wechat',?,'dm','confirmed','default',0,'','',NULL,?,?,?)""",
                (contact_id, display_name, now, now, expected_account_scope_id),
            )
            conn.execute(
                """INSERT INTO contact_aliases(id,contact_id,alias,source,created_at)
                   VALUES (?,?,?,?,?)""",
                (_id("alias"), contact_id, display_name, "historical_explicit_identity", now),
            )
            conn.execute(
                """INSERT INTO conversations(
                       id,contact_id,platform,conversation_type,identity_key,created_at,
                       updated_at,account_scope_id
                   ) VALUES (?,?,'wechat','dm',?,?,?,?)""",
                (
                    conversation_id, contact_id,
                    f"historical:{safe_contact_reference}", now, now,
                    expected_account_scope_id,
                ),
            )
            conn.execute(
                """UPDATE historical_contact_mappings SET
                       contact_id=?,conversation_id=?,status='resolved',match_basis=?,
                       confidence=1.0,updated_at=? WHERE id=?""",
                (contact_id, conversation_id, expected_basis, now, mapping_id),
            )
            self._attach_resolved_messages(
                conn,
                mapping_id=mapping_id,
                account_scope_id=expected_account_scope_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
            )
            message_count = int(conn.execute(
                "SELECT COUNT(*) FROM historical_chat_messages WHERE mapping_id=?",
                (mapping_id,),
            ).fetchone()[0])
            attached_count = int(conn.execute(
                """SELECT COUNT(*) FROM historical_chat_messages
                   WHERE mapping_id=? AND account_scope_id=? AND contact_id=?
                     AND conversation_id=? AND mapping_status='resolved'""",
                (
                    mapping_id, expected_account_scope_id, contact_id,
                    conversation_id,
                ),
            ).fetchone()[0])
            if message_count <= 0 or attached_count != message_count:
                raise ValueError("historical_explicit_mapping_message_attach_incomplete")
            return {
                "mapping_id": mapping_id,
                "contact_id": contact_id,
                "conversation_id": conversation_id,
                "status": "resolved",
                "message_count": message_count,
                "safe_contact_reference": safe_contact_reference,
            }

    @staticmethod
    def _validate_scope_targets(
        conn: sqlite3.Connection,
        *,
        account_scope_id: str,
        contact_id: str | None,
        conversation_id: str | None,
        require_resolved: bool,
    ) -> None:
        scope = conn.execute("SELECT id FROM account_scopes WHERE id=?", (account_scope_id,)).fetchone()
        if scope is None:
            raise ValueError("history_account_scope_not_found")
        if require_resolved and (not contact_id or not conversation_id):
            raise ValueError("resolved_mapping_requires_contact_and_conversation")
        if contact_id:
            contact = conn.execute(
                "SELECT account_scope_id FROM contacts WHERE id=?", (contact_id,)
            ).fetchone()
            if contact is None or contact["account_scope_id"] != account_scope_id:
                raise ValueError("history_contact_scope_mismatch")
        if conversation_id:
            conversation = conn.execute(
                "SELECT account_scope_id,contact_id FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
            if (
                conversation is None
                or conversation["account_scope_id"] != account_scope_id
                or (contact_id and conversation["contact_id"] != contact_id)
            ):
                raise ValueError("history_conversation_scope_mismatch")

    def get_mapping(self, mapping_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM historical_contact_mappings WHERE id=?", (mapping_id,)
            ).fetchone()
        return dict(row) if row else None

    def find_mapping(
        self, *, import_source_id: str, account_scope_id: str, mapping_key: str
    ) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT * FROM historical_contact_mappings
                   WHERE import_source_id=? AND account_scope_id=? AND mapping_key=?""",
                (import_source_id, account_scope_id, mapping_key),
            ).fetchone()
        return dict(row) if row else None

    def create_or_get_import_job(
        self,
        *,
        import_source_id: str,
        account_scope_id: str,
        preview_hash: str,
        preview: dict[str, Any],
        mapping_id: str | None = None,
        contact_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        payload = json.dumps(preview, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT * FROM historical_import_jobs
                   WHERE import_source_id=? AND account_scope_id=? AND preview_hash=?""",
                (import_source_id, account_scope_id, preview_hash),
            ).fetchone()
            if row:
                return dict(row)
            job_id = _id("history_import")
            conn.execute(
                """INSERT INTO historical_import_jobs(
                    id,import_source_id,mapping_id,account_scope_id,contact_id,conversation_id,
                    start_time,end_time,message_count,duplicate_count,committed_count,status,
                    checkpoint_sequence,preview_hash,preview_json,error,started_at,completed_at,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,0,'previewed',-1,?,?, '',NULL,NULL,?,?)""",
                (
                    job_id, import_source_id, mapping_id, account_scope_id, contact_id,
                    conversation_id, preview.get("start_time"), preview.get("end_time"),
                    int(preview.get("message_count", 0)), int(preview.get("duplicate_count", 0)),
                    preview_hash, payload, now, now,
                ),
            )
            row = conn.execute("SELECT * FROM historical_import_jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row)

    def get_import_job(self, job_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM historical_import_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def begin_import(self, job_id: str) -> dict[str, Any]:
        now = utc_now_iso()
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM historical_import_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("historical_import_job_not_found")
            if row["status"] == "completed":
                return dict(row)
            conn.execute(
                """UPDATE historical_import_jobs SET status='running',
                   started_at=COALESCE(started_at,?),error='',updated_at=? WHERE id=?""",
                (now, now, job_id),
            )
            return dict(conn.execute("SELECT * FROM historical_import_jobs WHERE id=?", (job_id,)).fetchone())

    def commit_message(
        self,
        *,
        job_id: str,
        mapping: dict[str, Any],
        message: dict[str, Any],
    ) -> tuple[str, bool]:
        """Commit one raw record and checkpoint it in the same transaction."""
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute("SELECT * FROM historical_import_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise ValueError("historical_import_job_not_found")
            if job["account_scope_id"] != mapping["account_scope_id"]:
                raise ValueError("history_job_mapping_scope_mismatch")
            sequence = int(message["source_sequence"])
            existing = conn.execute(
                """SELECT historical_message_id FROM historical_chat_messages
                   WHERE import_source_id=? AND account_scope_id=? AND source_hash=?""",
                (job["import_source_id"], job["account_scope_id"], message["source_hash"]),
            ).fetchone()
            created = existing is None
            if created:
                historical_id = _id("historical_message")
                conn.execute(
                    """INSERT INTO historical_chat_messages(
                        historical_message_id,import_source_id,import_job_id,mapping_id,
                        account_scope_id,contact_id,conversation_id,external_contact_id,
                        external_conversation_id,external_message_id,sender,message_type,text,
                        sent_at,source_sequence,source_hash,mapping_status,imported_at,raw_metadata_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        historical_id, job["import_source_id"], job_id, mapping["id"],
                        job["account_scope_id"], mapping.get("contact_id"),
                        mapping.get("conversation_id"), mapping.get("external_contact_id", ""),
                        mapping.get("external_conversation_id", ""),
                        message.get("external_message_id", ""), message["sender"],
                        message.get("message_type", "text"), message.get("text", ""),
                        message["sent_at"], sequence, message["source_hash"],
                        mapping["status"], now,
                        json.dumps(message.get("raw_metadata", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    ),
                )
            else:
                historical_id = str(existing["historical_message_id"])
            conn.execute(
                """UPDATE historical_import_jobs SET
                   checkpoint_sequence=MAX(checkpoint_sequence,?),
                   committed_count=committed_count+?,
                   duplicate_count=duplicate_count+?,updated_at=? WHERE id=?""",
                (sequence, int(created), int(not created), now, job_id),
            )
            return historical_id, created

    def finish_import(self, job_id: str, *, status: str = "completed", error: str = "") -> None:
        if status not in {"paused", "completed", "failed", "blocked"}:
            raise ValueError("invalid_import_terminal_status")
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE historical_import_jobs SET status=?,error=?,
                   completed_at=CASE WHEN ?='completed' THEN ? ELSE completed_at END,
                   updated_at=? WHERE id=?""",
                (status, error[:2000], status, now, now, job_id),
            )

    def list_messages(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        page: int = 1,
        page_size: int = 50,
        resolved_only: bool = True,
    ) -> list[HistoricalMessageRow]:
        size = max(1, min(200, int(page_size)))
        offset = max(0, int(page) - 1) * size
        status = " AND mapping_status='resolved'" if resolved_only else ""
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM historical_chat_messages
                    WHERE account_scope_id=? AND contact_id=?{status}
                    ORDER BY sent_at DESC,source_sequence DESC,historical_message_id DESC
                    LIMIT ? OFFSET ?""",
                (account_scope_id, contact_id, size, offset),
            ).fetchall()
        return [HistoricalMessageRow(**dict(row)) for row in rows]

    def list_mapping_messages(self, mapping_id: str, *, after_sequence: int = -1) -> list[HistoricalMessageRow]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM historical_chat_messages
                   WHERE mapping_id=? AND source_sequence>?
                   ORDER BY sent_at,source_sequence,historical_message_id""",
                (mapping_id, int(after_sequence)),
            ).fetchall()
        return [HistoricalMessageRow(**dict(row)) for row in rows]

    def create_backfill_job(self, *, import_job_id: str, mapping_id: str) -> dict[str, Any]:
        now = utc_now_iso()
        with self.db.connect() as conn:
            mapping = conn.execute(
                "SELECT * FROM historical_contact_mappings WHERE id=?", (mapping_id,)
            ).fetchone()
            if mapping is None or mapping["status"] != "resolved":
                raise ValueError("backfill_requires_resolved_mapping")
            existing = conn.execute(
                """SELECT * FROM historical_memory_backfill_jobs
                   WHERE import_job_id=? AND contact_id=?""",
                (import_job_id, mapping["contact_id"]),
            ).fetchone()
            if existing:
                return dict(existing)
            counts = conn.execute(
                "SELECT COUNT(*) AS n FROM historical_chat_messages WHERE import_job_id=? AND mapping_id=?",
                (import_job_id, mapping_id),
            ).fetchone()
            job_id = _id("history_backfill")
            conn.execute(
                """INSERT INTO historical_memory_backfill_jobs(
                    id,import_job_id,mapping_id,account_scope_id,contact_id,conversation_id,
                    customer_profile_id,status,total_messages,eligible_messages,total_chunks,
                    completed_chunks,checkpoint_chunk,provider,model,error,started_at,
                    completed_at,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,NULL,'pending',?,0,0,0,-1,'','','',NULL,NULL,?,?)""",
                (
                    job_id, import_job_id, mapping_id, mapping["account_scope_id"],
                    mapping["contact_id"], mapping["conversation_id"], int(counts["n"]), now, now,
                ),
            )
            return dict(conn.execute(
                "SELECT * FROM historical_memory_backfill_jobs WHERE id=?", (job_id,)
            ).fetchone())

    def get_backfill_job(self, job_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM historical_memory_backfill_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def link_backfill_profile(self, job_id: str, customer_profile_id: str) -> None:
        """Persist the exact profile produced/used by a selected Pilot job."""
        now = utc_now_iso()
        with self.db.connect() as conn:
            job = conn.execute(
                "SELECT account_scope_id,contact_id,conversation_id FROM historical_memory_backfill_jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            profile = conn.execute(
                "SELECT account_scope_id,contact_id,conversation_id FROM customer_profiles WHERE id=?",
                (customer_profile_id,),
            ).fetchone()
            if job is None or profile is None:
                raise ValueError("historical_backfill_profile_link_missing")
            if (
                job["account_scope_id"] != profile["account_scope_id"]
                or job["contact_id"] != profile["contact_id"]
                or job["conversation_id"] != profile["conversation_id"]
            ):
                raise ValueError("historical_backfill_profile_link_identity_mismatch")
            conn.execute(
                "UPDATE historical_memory_backfill_jobs SET customer_profile_id=?,updated_at=? WHERE id=?",
                (customer_profile_id, now, job_id),
            )

    def get_backfill_chunk(self, job_id: str, chunk_index: int) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT * FROM historical_memory_backfill_chunks
                   WHERE backfill_job_id=? AND chunk_index=?""",
                (job_id, int(chunk_index)),
            ).fetchone()
        return dict(row) if row else None

    def replace_pending_chunks(
        self, job_id: str, chunks: Iterable[dict[str, Any]], *, eligible_messages: int
    ) -> list[dict[str, Any]]:
        now = utc_now_iso()
        chunk_list = list(chunks)
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute(
                "SELECT * FROM historical_memory_backfill_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise ValueError("historical_backfill_job_not_found")
            existing = conn.execute(
                """SELECT COUNT(*) AS n,
                          SUM(CASE WHEN ai_call_count>0 OR status<>'pending' THEN 1 ELSE 0 END) AS protected
                   FROM historical_memory_backfill_chunks WHERE backfill_job_id=?""",
                (job_id,),
            ).fetchone()
            # Before the first provider call a preview policy may be tuned from
            # real message-density benchmarks. Once any chunk has run, its
            # checkpoint/input hash is immutable and can never be regenerated.
            if int(existing["n"]) > 0 and int(existing["protected"] or 0) == 0:
                conn.execute(
                    "DELETE FROM historical_memory_backfill_chunks WHERE backfill_job_id=?",
                    (job_id,),
                )
                existing = {"n": 0}
            if int(existing["n"]) == 0:
                for item in chunk_list:
                    conn.execute(
                        """INSERT INTO historical_memory_backfill_chunks(
                            id,backfill_job_id,chunk_index,first_message_id,last_message_id,
                            message_count,character_count,input_hash,input_context_json,
                            output_json,write_result_json,status,ai_call_count,error,
                            started_at,completed_at,created_at,updated_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,'{}','{}','pending',0,'',NULL,NULL,?,?)""",
                        (
                            _id("history_chunk"), job_id, item["chunk_index"],
                            item["first_message_id"], item["last_message_id"],
                            item["message_count"], item["character_count"],
                            item["input_hash"], item["input_context_json"], now, now,
                        ),
                    )
            conn.execute(
                """UPDATE historical_memory_backfill_jobs SET eligible_messages=?,total_chunks=?,
                   status=CASE WHEN status='completed' THEN status ELSE 'pending' END,updated_at=?
                   WHERE id=?""",
                (eligible_messages, len(chunk_list), now, job_id),
            )
            rows = conn.execute(
                """SELECT * FROM historical_memory_backfill_chunks
                   WHERE backfill_job_id=? ORDER BY chunk_index""",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_next_chunk(self, job_id: str, *, provider: str, model: str) -> dict[str, Any] | None:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute(
                "SELECT * FROM historical_memory_backfill_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if job is None or job["status"] in {"completed", "blocked", "paused"}:
                return None
            row = conn.execute(
                """SELECT * FROM historical_memory_backfill_chunks
                   WHERE backfill_job_id=? AND (
                       status='pending'
                       OR (status='provider_timeout_pending' AND ai_call_count < 2)
                       OR (status='failed' AND error LIKE 'retry_authorized:%'
                           AND ai_call_count < 2)
                   )
                   ORDER BY chunk_index LIMIT 1""",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            retry_authorized = row["status"] in {"failed", "provider_timeout_pending"}
            max_attempts = 2 if retry_authorized else 1
            if int(row["ai_call_count"]) >= max_attempts:
                conn.execute(
                    "UPDATE historical_memory_backfill_chunks SET status='blocked',error='provider_call_already_used',updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                return None
            conn.execute(
                """UPDATE historical_memory_backfill_chunks SET status='running',
                   ai_call_count=ai_call_count+1,started_at=COALESCE(started_at,?),updated_at=? WHERE id=?""",
                (now, now, row["id"]),
            )
            conn.execute(
                """UPDATE historical_memory_backfill_jobs SET status='running',provider=?,model=?,
                   started_at=COALESCE(started_at,?),updated_at=? WHERE id=?""",
                (provider, model, now, now, job_id),
            )
            return dict(conn.execute(
                "SELECT * FROM historical_memory_backfill_chunks WHERE id=?", (row["id"],)
            ).fetchone())

    def authorize_chunk_retry(self, job_id: str, chunk_index: int, *, reason: str) -> dict[str, Any]:
        """Allow one explicit retry without hiding the first provider call.

        Completed/no-memory chunks are immutable.  The call counter is kept,
        so cost reports show both attempts and ``claim_next_chunk`` will block
        a third call even if this method is invoked again.
        """
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM historical_memory_backfill_chunks
                   WHERE backfill_job_id=? AND chunk_index=?""",
                (job_id, int(chunk_index)),
            ).fetchone()
            if row is None:
                raise ValueError("historical_backfill_chunk_not_found")
            if row["status"] not in {"invalid_output", "failed", "blocked"}:
                raise ValueError("historical_backfill_retry_requires_failed_chunk")
            if int(row["ai_call_count"]) != 1:
                raise ValueError("historical_backfill_retry_limit_reached")
            if str(row["write_result_json"] or "{}") not in {"", "{}"}:
                raise ValueError("historical_backfill_retry_has_committed_writes")
            conn.execute(
                """UPDATE historical_memory_backfill_chunks
                   SET status='failed',error=?,completed_at=NULL,updated_at=?
                   WHERE id=?""",
                (f"retry_authorized:{reason}"[:2000], now, row["id"]),
            )
            conn.execute(
                """UPDATE historical_memory_backfill_jobs
                   SET status='partial',error='',completed_at=NULL,updated_at=? WHERE id=?""",
                (now, job_id),
            )
            return dict(conn.execute(
                "SELECT * FROM historical_memory_backfill_chunks WHERE id=?", (row["id"],)
            ).fetchone())

    def finish_chunk(
        self,
        chunk_id: str,
        *,
        status: str,
        output_json: str = "{}",
        write_result_json: str = "{}",
        error: str = "",
    ) -> None:
        if status not in {
            "completed", "no_memory", "invalid_output", "failed", "blocked",
            "provider_timeout_pending",
        }:
            raise ValueError("invalid_backfill_chunk_status")
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            chunk = conn.execute(
                "SELECT * FROM historical_memory_backfill_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
            if chunk is None:
                raise ValueError("historical_backfill_chunk_not_found")
            conn.execute(
                """UPDATE historical_memory_backfill_chunks SET status=?,output_json=?,
                   write_result_json=?,error=?,completed_at=?,updated_at=? WHERE id=?""",
                (status, output_json, write_result_json, error[:2000], now, now, chunk_id),
            )
            job_id = chunk["backfill_job_id"]
            summary = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN status IN ('completed','no_memory') THEN 1 ELSE 0 END) AS done,
                          MAX(CASE WHEN status IN ('completed','no_memory') THEN chunk_index ELSE -1 END) AS checkpoint,
                          SUM(CASE WHEN status IN (
                              'invalid_output','failed','blocked','provider_timeout_pending'
                          ) THEN 1 ELSE 0 END) AS bad
                   FROM historical_memory_backfill_chunks WHERE backfill_job_id=?""",
                (job_id,),
            ).fetchone()
            terminal = int(summary["done"] or 0) == int(summary["total"] or 0)
            job_status = "completed" if terminal else ("partial" if int(summary["bad"] or 0) else "running")
            conn.execute(
                """UPDATE historical_memory_backfill_jobs SET completed_chunks=?,checkpoint_chunk=?,
                   status=?,completed_at=CASE WHEN ?='completed' THEN ? ELSE completed_at END,
                   updated_at=? WHERE id=?""",
                (
                    int(summary["done"] or 0), int(summary["checkpoint"] or -1),
                    job_status, job_status, now, now, job_id,
                ),
            )

    def pause_backfill(self, job_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE historical_memory_backfill_jobs SET status='paused',updated_at=?
                   WHERE id=? AND status NOT IN ('completed','blocked')""",
                (utc_now_iso(), job_id),
            )

    def resume_backfill(self, job_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE historical_memory_backfill_jobs SET status='pending',updated_at=?
                   WHERE id=? AND status IN ('paused','partial','failed')""",
                (utc_now_iso(), job_id),
            )

    def finish_empty_backfill(self, job_id: str) -> None:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE historical_memory_backfill_jobs SET status='completed',
                   completed_at=?,updated_at=? WHERE id=? AND total_chunks=0""",
                (now, now, job_id),
            )

    def recover_interrupted_backfills(self) -> int:
        """Never repeat a provider call whose outcome became uncertain.

        A process crash after claiming a chunk leaves ``running`` plus
        ``ai_call_count=1``. On restart it is blocked for explicit review,
        while earlier completed chunks remain checkpointed.
        """
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT id,backfill_job_id FROM historical_memory_backfill_chunks
                   WHERE status='running'"""
            ).fetchall()
            if not rows:
                return 0
            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""UPDATE historical_memory_backfill_chunks
                    SET status='blocked',error='interrupted_provider_outcome_unknown',updated_at=?
                    WHERE id IN ({placeholders})""",
                (now, *ids),
            )
            job_ids = sorted({str(row["backfill_job_id"]) for row in rows})
            job_placeholders = ",".join("?" for _ in job_ids)
            conn.execute(
                f"""UPDATE historical_memory_backfill_jobs
                    SET status='partial',error='interrupted_chunk_requires_review',updated_at=?
                    WHERE id IN ({job_placeholders})""",
                (now, *job_ids),
            )
            return len(rows)

    def set_profile_history_status(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        status: str,
        detail: str = "",
    ) -> None:
        now = utc_now_iso()
        imported = now if status in {"history_imported", "memory_backfill_pending"} else None
        backfilled = now if status == "memory_backfilled" else None
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE customer_profiles SET history_status=?,history_status_detail=?,
                   history_last_imported_at=COALESCE(?,history_last_imported_at),
                   history_last_backfilled_at=COALESCE(?,history_last_backfilled_at),updated_at=?
                   WHERE account_scope_id=? AND contact_id=?""",
                (status, detail[:500], imported, backfilled, now, account_scope_id, contact_id),
            )

