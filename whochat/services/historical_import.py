from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from whochat.storage.historical import HistoricalArchiveRepository


@dataclass(frozen=True)
class ParsedHistoricalMessage:
    mapping_key: str
    external_contact_id: str
    external_conversation_id: str
    source_display_name: str
    external_message_id: str
    sender: str
    message_type: str
    text: str
    sent_at: str
    source_sequence: int
    raw_metadata: dict[str, Any]


@dataclass(frozen=True)
class ParsedHistoricalExport:
    source_kind: str
    display_name: str
    source_reference: str
    source_hash: str
    metadata: dict[str, Any]
    messages: tuple[ParsedHistoricalMessage, ...]


class HistoricalParserAdapter(Protocol):
    adapter_version: str

    def parse(self, source: Path) -> ParsedHistoricalExport: ...


class FixtureJsonHistoryAdapter:
    """Safe JSON/export adapter used by fixtures and compliant user exports.

    It never opens WeChat databases, processes or memory.  The accepted shape
    is intentionally explicit so sender and stable identifiers are never
    inferred from a nickname.
    """

    adapter_version = "1"

    def parse(self, source: Path) -> ParsedHistoricalExport:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("conversations"), list):
            raise ValueError("history_export_invalid_root")
        items: list[ParsedHistoricalMessage] = []
        global_sequence = 0
        for conversation in payload["conversations"]:
            if not isinstance(conversation, dict):
                raise ValueError("history_export_invalid_conversation")
            external_contact_id = _text(conversation.get("external_contact_id"))
            external_conversation_id = _text(conversation.get("external_conversation_id"))
            display_name = _text(conversation.get("display_name"))
            mapping_key = external_conversation_id or external_contact_id
            if not mapping_key:
                # A display name is kept as an unresolved bucket only; it can
                # never become a resolved mapping automatically.
                mapping_key = f"display:{display_name or 'unknown'}"
            messages = conversation.get("messages")
            if not isinstance(messages, list):
                raise ValueError("history_export_messages_not_list")
            for local_sequence, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError("history_export_invalid_message")
                sender = _text(message.get("sender")).lower()
                if sender not in {"other", "me", "system", "unknown"}:
                    raise ValueError("history_export_invalid_sender")
                sent_at = _text(message.get("sent_at"))
                if not sent_at:
                    raise ValueError("history_export_missing_sent_at")
                sequence = message.get("source_sequence", global_sequence)
                try:
                    sequence = int(sequence)
                except (TypeError, ValueError) as exc:
                    raise ValueError("history_export_invalid_sequence") from exc
                items.append(
                    ParsedHistoricalMessage(
                        mapping_key=mapping_key,
                        external_contact_id=external_contact_id,
                        external_conversation_id=external_conversation_id,
                        source_display_name=display_name,
                        external_message_id=_text(message.get("external_message_id")),
                        sender=sender,
                        message_type=_text(message.get("message_type")) or "text",
                        text=_text(message.get("text")),
                        sent_at=sent_at,
                        source_sequence=sequence,
                        raw_metadata=dict(message.get("raw_metadata") or {}),
                    )
                )
                global_sequence += 1
        items.sort(key=lambda item: (item.sent_at, item.source_sequence, item.external_message_id))
        # Re-number only duplicate/non-monotonic source sequences. The stable
        # source sequence is part of fallback identity and protects two genuine
        # identical short messages from accidental collapse.
        normalized: list[ParsedHistoricalMessage] = []
        for sequence, item in enumerate(items):
            normalized.append(
                ParsedHistoricalMessage(**{**item.__dict__, "source_sequence": sequence})
            )
        return ParsedHistoricalExport(
            source_kind=_text(payload.get("source_kind")) or "fixture_json",
            display_name=_text(payload.get("display_name")) or source.name,
            source_reference=_text(payload.get("source_reference")) or str(source.resolve()),
            source_hash=hashlib.sha256(raw).hexdigest(),
            metadata=dict(payload.get("metadata") or {}),
            messages=tuple(normalized),
        )


class HistoricalImportService:
    def __init__(self, repository: HistoricalArchiveRepository) -> None:
        self.repository = repository

    def preview(
        self,
        *,
        parsed: ParsedHistoricalExport,
        account_scope_id: str,
        mapping_overrides: dict[str, tuple[str, str]] | None = None,
        adapter_version: str = "1",
    ) -> dict[str, Any]:
        source_id = self.repository.register_source(
            source_kind=parsed.source_kind,
            display_name=parsed.display_name,
            source_reference=parsed.source_reference,
            adapter_version=adapter_version,
            source_hash=parsed.source_hash,
            metadata=parsed.metadata,
        )
        grouped: dict[str, list[ParsedHistoricalMessage]] = {}
        for item in parsed.messages:
            grouped.setdefault(item.mapping_key, []).append(item)
        mapping_overrides = mapping_overrides or {}
        mappings: dict[str, dict[str, Any]] = {}
        status_counts = {key: 0 for key in ("resolved", "probable", "unresolved", "conflicting")}
        for mapping_key, items in grouped.items():
            first = items[0]
            existing = self.repository.find_mapping(
                import_source_id=source_id,
                account_scope_id=account_scope_id,
                mapping_key=mapping_key,
            )
            override = mapping_overrides.get(mapping_key)
            if override:
                contact_id, conversation_id = override
                status, basis, confidence = "resolved", "explicit_user_mapping", 1.0
            elif existing and existing["status"] == "resolved":
                contact_id, conversation_id = existing["contact_id"], existing["conversation_id"]
                status, basis, confidence = "resolved", existing["match_basis"], existing["confidence"]
            else:
                contact_id = conversation_id = None
                status = "unresolved"
                basis = "stable_source_identifier_requires_explicit_mapping" if (
                    first.external_contact_id or first.external_conversation_id
                ) else "display_name_only_never_auto_maps"
                confidence = 0.0
            mapping = self.repository.upsert_mapping(
                import_source_id=source_id,
                mapping_key=mapping_key,
                account_scope_id=account_scope_id,
                external_contact_id=first.external_contact_id,
                external_conversation_id=first.external_conversation_id,
                source_display_name=first.source_display_name,
                contact_id=contact_id,
                conversation_id=conversation_id,
                status=status,
                match_basis=basis,
                confidence=confidence,
            )
            mappings[mapping_key] = mapping
            status_counts[mapping["status"]] += 1

        hashes: set[str] = set()
        duplicates = 0
        sender_counts: dict[str, int] = {}
        rows: list[dict[str, Any]] = []
        for item in parsed.messages:
            fingerprint = history_source_hash(item)
            duplicates += int(fingerprint in hashes)
            hashes.add(fingerprint)
            sender_counts[item.sender] = sender_counts.get(item.sender, 0) + 1
            rows.append({**item.__dict__, "source_hash": fingerprint})
        times = [item.sent_at for item in parsed.messages]
        preview = {
            "account_count": 1,
            "conversation_count": len(grouped),
            "contact_count": len(grouped),
            "resolved_count": status_counts["resolved"],
            "probable_count": status_counts["probable"],
            "unresolved_count": status_counts["unresolved"],
            "conflicting_count": status_counts["conflicting"],
            "message_count": len(rows),
            "start_time": min(times) if times else None,
            "end_time": max(times) if times else None,
            "sender_counts": sender_counts,
            "duplicate_count": duplicates,
            "source_hash": parsed.source_hash,
        }
        preview_hash = hashlib.sha256(
            json.dumps(
                {"preview": preview, "rows": rows, "mappings": {key: value["id"] for key, value in mappings.items()}},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        job = self.repository.create_or_get_import_job(
            import_source_id=source_id,
            account_scope_id=account_scope_id,
            preview_hash=preview_hash,
            preview=preview,
        )
        return {
            "source_id": source_id,
            "job_id": job["id"],
            "preview_hash": preview_hash,
            "preview": preview,
            "mappings": mappings,
            "rows": tuple(rows),
        }

    def commit(
        self,
        preview_result: dict[str, Any],
        *,
        max_records: int | None = None,
    ) -> dict[str, int | str]:
        job_id = str(preview_result["job_id"])
        job = self.repository.begin_import(job_id)
        checkpoint = int(job["checkpoint_sequence"])
        committed = duplicates = processed = 0
        rows = preview_result["rows"]
        mappings = preview_result["mappings"]
        try:
            for row in rows:
                if int(row["source_sequence"]) <= checkpoint:
                    continue
                if max_records is not None and processed >= max_records:
                    self.repository.finish_import(job_id, status="paused")
                    break
                _, created = self.repository.commit_message(
                    job_id=job_id,
                    mapping=mappings[row["mapping_key"]],
                    message=row,
                )
                committed += int(created)
                duplicates += int(not created)
                processed += 1
            else:
                self.repository.finish_import(job_id, status="completed")
                for mapping in mappings.values():
                    if mapping["status"] == "resolved" and mapping.get("contact_id"):
                        self.repository.set_profile_history_status(
                            account_scope_id=mapping["account_scope_id"],
                            contact_id=mapping["contact_id"],
                            status="history_imported",
                            detail="历史聊天已安全导入，等待结构化初始化",
                        )
        except Exception as exc:
            self.repository.finish_import(
                job_id, status="failed", error=f"{type(exc).__name__}:{exc}"
            )
            raise
        final = self.repository.get_import_job(job_id) or {}
        return {
            "job_id": job_id,
            "status": str(final.get("status", "")),
            "processed": processed,
            "committed": committed,
            "duplicates": duplicates,
            "checkpoint_sequence": int(final.get("checkpoint_sequence", -1)),
        }


def history_source_hash(item: ParsedHistoricalMessage) -> str:
    if item.external_message_id:
        identity = {
            "mapping_key": item.mapping_key,
            "external_message_id": item.external_message_id,
        }
    else:
        identity = {
            "mapping_key": item.mapping_key,
            "sent_at": item.sent_at,
            "sender": item.sender,
            "text": _normalize_text(item.text),
            "source_sequence": item.source_sequence,
        }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _text(value: Any) -> str:
    return str(value or "").strip()

