from __future__ import annotations

import json
import hashlib
import sqlite3
import unicodedata
from dataclasses import replace
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from whochat.ai.models import ReplyMemoryContext
from whochat.core.current_conversation import (
    CurrentMessageView, build_current_message_view, current_screen_evidence, _OVERLAP_ONLY_REASONS,
)
from whochat.core.models import ContactStatus, Conversation, ConversationObservation, ConversationType
from whochat.services.memory_canonicalization import canonicalize_crop, crop_terms
from whochat.storage.database import Database
from whochat.storage.repositories import LogRepository


_EVENT_TYPES = (
    "problem_reported",
    "diagnosis",
    "solution_given",
    "product_recommended",
    "dosage_given",
    "quote_given",
    "order_intent",
    "order_placed",
    "purchased",
    "usage_started",
    "effect_feedback",
    "complaint",
    "customer_concern",
    "followup",
    "management_plan",
)
_EVENT_PRIORITY = {name: len(_EVENT_TYPES) - index for index, name in enumerate(_EVENT_TYPES)}
_SEMANTIC_TERMS = (
    "黄叶", "烧根", "根", "叶", "病", "虫", "药", "肥", "方案", "使用", "用过",
    "效果", "改善", "减轻", "再用", "第二遍", "报价", "价格", "购买", "收到",
    "回访", "膨果", "转色", "苹果", "葡萄", "草莓", "玉米", "水稻",
)


class CustomerMemoryContextBuilder:
    """Build a compact, identity-scoped reply-memory snapshot using local SQLite only."""

    def __init__(self, db: Database, *, logs: LogRepository | None = None, observed_history=None,
                 history_enabled=None) -> None:
        self.db = db
        self.logs = logs
        self.observed_history = observed_history
        self.history_enabled = history_enabled

    def build(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
        current_message_id: str,
    ) -> ReplyMemoryContext:
        try:
            return self._build(
                account_scope_id=account_scope_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
                current_message_id=current_message_id,
            )
        except Exception as exc:
            self._log_failure(exc, account_scope_id, contact_id, conversation_id)
            return ReplyMemoryContext(
                status="fallback_error",
                account_scope_id=account_scope_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
                current_message_id=current_message_id,
                error=type(exc).__name__,
            )

    def build_for_current_view(
        self, view: CurrentMessageView, *, now: datetime | None = None,
    ) -> ReplyMemoryContext:
        """Build bounded memory only after re-reading its actual local source.

        A window baseline has no formal Message ID.  It does not create one,
        does not trigger memory extraction, and cannot select a crop from an
        unrelated old formal-message tail.
        """
        if type(view) is not CurrentMessageView:
            return ReplyMemoryContext("fallback_error", "", "", "", "", error="InvalidCurrentView")
        try:
            result = self._build(
                account_scope_id=view.account_scope_id, contact_id=view.contact_id,
                conversation_id=view.conversation_id, current_message_id="",
                current_view=view, now=now,
            )
            if self.observed_history is not None and self._history_is_enabled():
                result = replace(result, observed_history=self.observed_history.build(view, now=now))
            if self._history_is_enabled():
                from whochat.services.imported_history import build_imported_history
                with self._readonly_view_connection() as conn:
                    # Revalidate in the same read transaction as the archive
                    # join; an expired observation never becomes history-only
                    # permission to call AI.
                    self._verified_current_view(conn, view, now=now)
                    result = replace(result, imported_history=build_imported_history(conn, view, now))
            return result
        except Exception as exc:
            self._log_failure(exc, view.account_scope_id, view.contact_id, view.conversation_id)
            return ReplyMemoryContext(
                status="fallback_error", account_scope_id=view.account_scope_id,
                contact_id=view.contact_id, conversation_id=view.conversation_id,
                current_message_id="", error=type(exc).__name__,
                current_view_revision=view.content_revision,
                current_view_source_ref=view.observation_id,
                current_view_source_kind=view.source_kind,
                current_view_verified=False,
            )

    def record_current_view(self, view: CurrentMessageView):
        """Explicit local capture only; never synthesize a new Message/AI job."""
        if not self._history_is_enabled():
            return {"status": "disabled", "stored": 0, "duplicates": 0}
        if self.observed_history is None:
            raise ValueError("observed_history_unavailable")
        return self.observed_history.record(view)

    def validate_history_context(self, memory: ReplyMemoryContext | None, view: CurrentMessageView) -> bool:
        """Deleted/changed history must invalidate a pending result, not leak it."""
        try:
            if self._history_is_enabled():
                from whochat.services.imported_history import build_imported_history
                with self._readonly_view_connection() as conn:
                    self._verified_current_view(conn, view)
                    fresh_imported = build_imported_history(conn, view)
                if fresh_imported != (getattr(memory, 'imported_history', None) if memory else None):
                    return False
            elif getattr(memory, 'imported_history', None):
                return False
        except Exception:
            return False
        if self.observed_history is None or not self._history_is_enabled():
            return memory is None or not memory.observed_history
        try:
            fresh = self.observed_history.build(view)
            return memory is not None and fresh == memory.observed_history
        except Exception:
            return False

    def _history_is_enabled(self) -> bool:
        try:
            return self.history_enabled is None or self.history_enabled() is True
        except Exception:
            return False

    def validate_historical_source(self, source: dict[str, Any]) -> bool:
        """Recheck an archived quote; this NEVER grants a live capture lease.

        The original view was validated at recording time. Observation status,
        capture time and TTL can subsequently refresh/expire in-place. Check
        unchanged business content without treating that old lease as current.
        """
        try:
            from whochat.services.observed_history import message_evidence_hash, sanitize_excerpt
            from whochat.core.current_conversation import _visible_message
            if source.get("schema") != "whochat.observed_history.source.v1":
                return False
            with self._readonly_view_connection() as conn:
                contact = conn.execute("SELECT * FROM contacts WHERE id=?", (source["contact_id"],)).fetchone()
                conversation = conn.execute("SELECT * FROM conversations WHERE id=?", (source["conversation_id"],)).fetchone()
                row = conn.execute("SELECT * FROM conversation_observations WHERE id=?", (source["source_ref"],)).fetchone()
                session = conn.execute("SELECT * FROM conversation_sessions WHERE id=?", (source["session_id"],)).fetchone()
                if (contact is None or conversation is None or row is None or session is None
                        or contact["account_scope_id"] != source["account_scope_id"]
                        or str(contact["platform"]).casefold() != "wechat"
                        or contact["conversation_type"] != ConversationType.DM.value
                        or contact["status"] in {ContactStatus.IGNORED.value, ContactStatus.MERGED.value}
                        or contact["merged_into"]
                        or conversation["contact_id"] != source["contact_id"]
                        or conversation["account_scope_id"] != source["account_scope_id"]
                        or conversation["platform"].casefold() != "wechat"
                        or conversation["conversation_type"] != ConversationType.DM.value
                        or row["status"] not in {"baseline", "confirmed", "promoted", "quarantined", "expired"}
                        or any(row[key] != source[key] for key in (
                            "account_scope_id", "conversation_id", "target_instance_id", "session_id"))
                        or any(session[key] != source[key] for key in (
                            "account_scope_id", "conversation_id", "target_instance_id"))):
                    return False
                if (source.get("source_kind") != "visible_observation"
                        or source.get("original_observation_status") not in {"baseline", "confirmed", "promoted", "quarantined"}
                        or (source.get("original_observation_status") == "quarantined"
                            and row["reason"] != source.get("source_reason"))
                        or (source.get("original_observation_status") != "quarantined"
                            and row["status"] not in {"baseline", "confirmed", "promoted"})
                        or type(source.get("source_job_id")) is not int
                        or type(source.get("capture_generation")) is not int
                        or row["source_job_id"] < source["source_job_id"]
                        or row["capture_generation"] < source["capture_generation"]):
                    return False
                # Historical evidence keeps its original lineage even after
                # a lease expires. An orphaned/cross-owner parent cannot be
                # resurrected just because its text hash still matches.
                if source["original_observation_status"] == "quarantined":
                    cursor = dict(row)
                    seen = {cursor["id"]}
                    for _ in range(64):
                        reason = cursor["reason"]
                        if reason in _OVERLAP_ONLY_REASONS:
                            break
                        prefix = "persistent_tail_fallback_repeated_quarantine:"
                        if not reason.startswith(prefix):
                            return False
                        parent_id = reason[len(prefix):]
                        if not parent_id or parent_id in seen:
                            return False
                        parent = conn.execute(
                            "SELECT rowid AS source_rowid,* FROM conversation_observations WHERE id=?", (parent_id,),
                        ).fetchone()
                        child_order = conn.execute(
                            "SELECT rowid FROM conversation_observations WHERE id=?", (cursor["id"],),
                        ).fetchone()
                        if (parent is None or child_order is None or parent["source_rowid"] >= child_order[0]
                                or parent["status"] not in {"quarantined", "expired"}
                                or any(parent[key] != row[key] for key in (
                                    "account_scope_id", "conversation_id", "target_instance_id", "session_id"))):
                            return False
                        cursor = dict(parent)
                        seen.add(parent_id)
                    else:
                        return False
                captured = datetime.fromisoformat(row["observed_at"])
                recorded = datetime.fromisoformat(source["observed_at"])
                if (captured.tzinfo is None or recorded.tzinfo is None
                        or recorded > captured or captured > datetime.now(timezone.utc)):
                    return False
                if not isinstance(row["messages_json"], str) or len(row["messages_json"]) > 512000:
                    return False
                rows = json.loads(row["messages_json"])
                if not isinstance(rows, list) or not 0 < len(rows) <= 120:
                    return False
                messages = []
                for raw in rows:
                    message = _visible_message(raw, len(messages))
                    if message.text or message.media_refs:
                        messages.append(message)
                payload = dict(schema="whochat.reply_business_view.v1", account_scope_id=source["account_scope_id"],
                               contact_id=source["contact_id"], conversation_id=source["conversation_id"],
                               source_kind="visible_observation", messages=[item.business_payload() for item in messages])
                digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":")).encode("utf-8")).hexdigest()
                index = source["source_message_ordinal"]
                if type(index) is not int or not 0 <= index < len(messages):
                    return False
                message = messages[index]
                return (digest == source["reply_business_revision"]
                        and message_evidence_hash(message) == source["source_message_sha256"]
                        and message.message_time == source["message_time"]
                        and message.time_source == source["time_source"]
                        and message.time_text == source["time_text"]
                        and sanitize_excerpt(message.text) == source["excerpt"])
        except (ValueError, TypeError, KeyError, AttributeError, sqlite3.Error):
            return False

    def validate_current_view(self, view: CurrentMessageView, *, now: datetime | None = None) -> bool:
        """Recheck persisted evidence before/after work without loading memory."""
        try:
            with self._readonly_view_connection() as conn:
                self._verified_current_view(conn, view, now=now)
            return True
        except Exception:
            return False

    @contextmanager
    def _readonly_view_connection(self) -> Iterator[sqlite3.Connection]:
        # Avoid Database.connect's mkdir/WAL setup on this read-only path.
        uri = self.db.path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA busy_timeout = 1000")
            conn.execute("BEGIN")
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _verified_current_view(
        conn: sqlite3.Connection, view: CurrentMessageView, *, now: datetime | None = None,
    ) -> CurrentMessageView:
        if type(view) is not CurrentMessageView:
            raise ValueError("reply_memory_current_view_required")
        contact = conn.execute("SELECT * FROM contacts WHERE id=?", (view.contact_id,)).fetchone()
        conversation_row = conn.execute("SELECT * FROM conversations WHERE id=?", (view.conversation_id,)).fetchone()
        observation_row = conn.execute("SELECT * FROM conversation_observations WHERE id=?", (view.observation_id,)).fetchone()
        session = conn.execute(
            "SELECT conversation_id,account_scope_id,target_instance_id,ended_at FROM conversation_sessions WHERE id=?",
            (view.session_id,),
        ).fetchone()
        if (
            contact is None or contact["account_scope_id"] != view.account_scope_id
            or str(contact["platform"]).casefold() != "wechat"
            or contact["conversation_type"] != ConversationType.DM.value
            or contact["status"] in {ContactStatus.IGNORED.value, ContactStatus.MERGED.value}
            or contact["merged_into"]
            or conversation_row is None or observation_row is None or session is None
            or session["conversation_id"] != view.conversation_id
            or session["account_scope_id"] != view.account_scope_id
            or session["target_instance_id"] != view.target_instance_id
            or session["ended_at"] is not None
        ):
            raise ValueError("reply_memory_current_view_source_owner_mismatch")
        conversation_data = dict(conversation_row)
        conversation_data["conversation_type"] = ConversationType(conversation_data["conversation_type"])
        rebuilt = build_current_message_view(
            ConversationObservation(**dict(observation_row)), conversation=Conversation(**conversation_data),
            expected_account_scope_id=view.account_scope_id, expected_contact_id=view.contact_id,
            expected_conversation_id=view.conversation_id, expected_target_instance_id=view.target_instance_id,
            expected_session_id=view.session_id, max_source_job_id=view.source_job_id,
            max_capture_generation=view.capture_generation, now=now or datetime.now(timezone.utc),
        )
        if rebuilt != view:
            raise ValueError("reply_memory_current_view_source_changed")
        if rebuilt.screen_evidence_json:
            # A repeated quarantine's JSON is not authority for its own
            # history. Follow bounded, actual, earlier same-owner DB rows.
            cursor = dict(observation_row)
            seen = {cursor["id"]}
            for _ in range(64):
                reason = cursor["reason"]
                if reason in _OVERLAP_ONLY_REASONS or cursor["status"] != "quarantined":
                    break
                prefix = "persistent_tail_fallback_repeated_quarantine:"
                # A prior negative marker intentionally contains no positive
                # proof. Its actual fixed reason can still identify history.
                parent_id = reason[len(prefix):] if reason.startswith(prefix) else None
                if (not isinstance(parent_id, str) or parent_id in seen
                        or reason != "persistent_tail_fallback_repeated_quarantine:" + parent_id):
                    raise ValueError("reply_memory_current_view_history_lineage_invalid")
                parent = conn.execute(
                    "SELECT rowid AS source_rowid,* FROM conversation_observations WHERE id=?", (parent_id,),
                ).fetchone()
                child_order = conn.execute(
                    "SELECT rowid FROM conversation_observations WHERE id=?", (cursor["id"],),
                ).fetchone()
                if (parent is None or child_order is None or parent["source_rowid"] >= child_order[0]
                        or any(parent[key] != observation_row[key] for key in (
                            "account_scope_id", "conversation_id", "target_instance_id", "session_id"))
                        or parent["status"] != "quarantined"):
                    raise ValueError("reply_memory_current_view_history_lineage_invalid")
                cursor = dict(parent)
                del cursor["source_rowid"]
                seen.add(parent_id)
            else:
                raise ValueError("reply_memory_current_view_history_lineage_invalid")
            if cursor["status"] == "quarantined" and cursor["reason"] not in _OVERLAP_ONLY_REASONS:
                raise ValueError("reply_memory_current_view_history_lineage_invalid")
        latest = conn.execute(
            """SELECT id FROM conversation_observations
               WHERE account_scope_id=? AND conversation_id=? AND target_instance_id=? AND session_id=?
               ORDER BY rowid DESC LIMIT 1""",
            (view.account_scope_id, view.conversation_id, view.target_instance_id, view.session_id),
        ).fetchone()
        # Even a newer quarantined frame invalidates this source.  Never
        # silently fall back to the previous apparently safe customer text.
        if latest is None or latest["id"] != view.observation_id:
            raise ValueError("reply_memory_current_view_not_latest")
        return rebuilt

    def _build(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
        current_message_id: str,
        current_view: CurrentMessageView | None = None,
        now: datetime | None = None,
    ) -> ReplyMemoryContext:
        if not account_scope_id or not contact_id or not conversation_id:
            raise ValueError("reply_memory_identity_incomplete")
        with (self._readonly_view_connection() if current_view is not None else self.db.connect()) as conn:
            view_fields: dict[str, Any] = {}
            if current_view is not None:
                self._verified_current_view(conn, current_view, now=now)
                if current_message_id:
                    raise ValueError("reply_memory_mixed_source_anchor")
                view_fields = {
                    "current_view_revision": current_view.content_revision,
                    "current_view_source_ref": current_view.observation_id,
                    "current_view_source_kind": current_view.source_kind,
                    "current_view_verified": True,
                }
            contact = conn.execute(
                "SELECT id,account_scope_id,display_name FROM contacts WHERE id=?",
                (contact_id,),
            ).fetchone()
            if contact is None or contact["account_scope_id"] != account_scope_id:
                raise ValueError("reply_memory_contact_scope_mismatch")
            conversation = conn.execute(
                "SELECT id,account_scope_id,contact_id FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
            if (
                conversation is None
                or conversation["account_scope_id"] != account_scope_id
                or conversation["contact_id"] != contact_id
            ):
                raise ValueError("reply_memory_conversation_mismatch")

            current = None
            if current_message_id:
                current = conn.execute(
                    "SELECT rowid,* FROM messages WHERE id=?",
                    (current_message_id,),
                ).fetchone()
                if (
                    current is None
                    or current["account_scope_id"] != account_scope_id
                    or current["contact_id"] != contact_id
                    or current["conversation_id"] != conversation_id
                ):
                    raise ValueError("reply_memory_message_identity_mismatch")

            profile = conn.execute(
                "SELECT * FROM customer_profiles WHERE account_scope_id=? AND contact_id=?",
                (account_scope_id, contact_id),
            ).fetchone()
            if profile is None:
                return ReplyMemoryContext(
                    status="empty",
                    account_scope_id=account_scope_id,
                    contact_id=contact_id,
                    conversation_id=conversation_id,
                    current_message_id=current_message_id,
                    **view_fields,
                )

            contexts = conn.execute(
                """
                SELECT * FROM customer_growing_contexts
                WHERE customer_profile_id=? AND status='active'
                ORDER BY last_confirmed_at DESC,created_at DESC,id DESC
                LIMIT 8
                """,
                (profile["id"],),
            ).fetchall()
            current_text = (
                current_view.current_message.text if current_view is not None and current_view.current_message is not None
                else current["text"] if current is not None else ""
            )
            recent_messages = (
                [{"text": item.text} for item in reversed(current_view.messages[-12:])]
                if current_view is not None else self._recent_messages(
                    conn, account_scope_id=account_scope_id, contact_id=contact_id,
                    conversation_id=conversation_id,
                    through_observed_at=current["observed_at"] if current is not None else None,
                    through_rowid=current["rowid"] if current is not None else None,
                )
            )
            event_rows = conn.execute(
                """
                SELECT * FROM customer_events
                WHERE customer_profile_id=? AND status<>'invalid'
                ORDER BY event_at DESC,created_at DESC,id DESC
                LIMIT 100
                """,
                (profile["id"],),
            ).fetchall()
            primary, candidates, resolution, ambiguous = self._select_context(
                contexts,
                current_text,
                recent_messages,
                event_rows,
            )
            allowed_context_ids = self._allowed_context_ids(primary, candidates, ambiguous)
            confirmed, qualified = self._facts(
                conn, profile["id"], allowed_context_ids, contexts
            )
            events = self._events(
                event_rows,
                allowed_context_ids,
                contexts,
                current_text,
            )
            followups = self._followups(
                conn, profile["id"], allowed_context_ids, contexts
            )

            return ReplyMemoryContext(
                status="available",
                account_scope_id=account_scope_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
                current_message_id=current_message_id,
                profile={
                    "id": profile["id"],
                    "display_name": _clip(profile["display_name"], 120),
                    "sales_stage": profile["sales_stage"],
                    "latest_contact_at": profile["latest_contact_at"],
                },
                primary_context=self._context_payload(primary) if primary is not None else None,
                candidate_contexts=tuple(self._context_payload(item) for item in candidates[:3]),
                context_resolution=resolution,
                context_ambiguous=ambiguous,
                confirmed_facts=tuple(confirmed[:20]),
                qualified_facts=tuple(qualified[:8]),
                relevant_events=tuple(events[:12]),
                pending_followups=tuple(followups[:5]),
                **view_fields,
            )

    @staticmethod
    def _recent_messages(
        conn: sqlite3.Connection,
        *,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
        through_observed_at: str | None,
        through_rowid: int | None,
    ) -> list[sqlite3.Row]:
        if through_rowid is None:
            return conn.execute(
                """SELECT rowid,id,speaker,text,observed_at FROM messages
                   WHERE account_scope_id=? AND contact_id=? AND conversation_id=?
                   ORDER BY observed_at DESC,rowid DESC LIMIT 12""",
                (account_scope_id, contact_id, conversation_id),
            ).fetchall()
        return conn.execute(
            """SELECT rowid,id,speaker,text,observed_at FROM messages
               WHERE account_scope_id=? AND contact_id=? AND conversation_id=?
                 AND (observed_at<? OR (observed_at=? AND rowid<=?))
               ORDER BY observed_at DESC,rowid DESC LIMIT 12""",
            (
                account_scope_id, contact_id, conversation_id,
                through_observed_at, through_observed_at, through_rowid,
            ),
        ).fetchall()

    def _select_context(
        self,
        contexts: list[sqlite3.Row],
        current_text: str,
        recent_messages: list[sqlite3.Row],
        events: list[sqlite3.Row],
    ) -> tuple[sqlite3.Row | None, list[sqlite3.Row], str, bool]:
        if not contexts:
            return None, [], "none", False
        current_matches = self._mentioned_contexts(current_text, contexts)
        if len(current_matches) == 1:
            return current_matches[0], [], "current_message", False
        if len(current_matches) > 1:
            return None, current_matches[:3], "ambiguous_current_message", True

        for message in recent_messages:
            if _normal(message["text"]) == _normal(current_text):
                continue
            matches = self._mentioned_contexts(message["text"], contexts)
            if len(matches) == 1:
                return matches[0], [], "recent_chat", False
            if len(matches) > 1:
                return None, matches[:3], "ambiguous_recent_chat", True

        by_id = {item["id"]: item for item in contexts}
        for event in events:
            context = by_id.get(event["growing_context_id"])
            if context is not None:
                return context, [], "recent_event", False
        if len(contexts) == 1:
            return contexts[0], [], "unique_active_context", False
        return None, list(contexts[:3]), "ambiguous", True

    @staticmethod
    def _mentioned_contexts(text: str, contexts: list[sqlite3.Row]) -> list[sqlite3.Row]:
        source = _normal(text)
        matches: list[sqlite3.Row] = []
        for context in contexts:
            crop = canonicalize_crop(context["crop"])
            if crop and any(_normal(term) in source for term in crop_terms(crop)):
                matches.append(context)
        return matches

    @staticmethod
    def _allowed_context_ids(
        primary: sqlite3.Row | None,
        candidates: list[sqlite3.Row],
        ambiguous: bool,
    ) -> set[str]:
        if primary is not None:
            return {primary["id"]}
        if ambiguous:
            return {item["id"] for item in candidates[:3]}
        return set()

    @staticmethod
    def _context_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "crop": _clip(row["crop"], 80),
            "variety": _clip(row["variety"], 80),
            "region": _clip(row["region"], 120),
            "acreage_value": row["acreage_value"],
            "acreage_unit": _clip(row["acreage_unit"], 20),
            "growth_stage": _clip(row["growth_stage"], 80),
            "cultivation_type": _clip(row["cultivation_type"], 80),
            "season_or_year": _clip(row["season_or_year"], 80),
            "last_confirmed_at": row["last_confirmed_at"],
        }

    @staticmethod
    def _facts(
        conn: sqlite3.Connection,
        profile_id: str,
        allowed_context_ids: set[str],
        contexts: list[sqlite3.Row],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = conn.execute(
            """
            SELECT * FROM customer_facts
            WHERE customer_profile_id=? AND status IN ('active','uncertain','conflicting')
            ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'uncertain' THEN 1 ELSE 2 END,
                     last_confirmed_at DESC,created_at DESC,id DESC
            LIMIT 80
            """,
            (profile_id,),
        ).fetchall()
        context_crop = {item["id"]: item["crop"] for item in contexts}
        confirmed: list[dict[str, Any]] = []
        qualified: list[dict[str, Any]] = []
        for row in rows:
            context_id = row["growing_context_id"]
            if context_id is not None and context_id not in allowed_context_ids:
                continue
            payload = {
                "id": row["id"],
                "context_id": context_id,
                "context_crop": context_crop.get(context_id, ""),
                "category": _clip(row["fact_category"], 80),
                "key": _clip(row["fact_key"], 80),
                "value": _clip(row["value_text"], 240),
                "normalized_value": _clip(row["normalized_value"], 160),
                "status": row["status"],
                "provenance": row["provenance"],
                "confidence": row["confidence"],
                "last_confirmed_at": row["last_confirmed_at"],
            }
            (confirmed if row["status"] == "active" else qualified).append(payload)
        return confirmed[:20], qualified[:8]

    @staticmethod
    def _events(
        rows: list[sqlite3.Row],
        allowed_context_ids: set[str],
        contexts: list[sqlite3.Row],
        current_text: str,
    ) -> list[dict[str, Any]]:
        context_crop = {item["id"]: item["crop"] for item in contexts}
        current_terms = {term for term in _SEMANTIC_TERMS if term in current_text}
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for row in rows:
            if row["event_type"] not in _EVENT_PRIORITY:
                continue
            context_id = row["growing_context_id"]
            if context_id is not None and context_id not in allowed_context_ids:
                continue
            structured = _json_object(row["structured_data_json"])
            searchable = " ".join(
                (str(row["title"] or ""), str(row["summary"] or ""), json.dumps(structured, ensure_ascii=False))
            )
            overlap = sum(1 for term in current_terms if term in searchable)
            score = (100 if context_id in allowed_context_ids else 40) + overlap * 25 + _EVENT_PRIORITY[row["event_type"]]
            payload = {
                "id": row["id"],
                "context_id": context_id,
                "context_crop": context_crop.get(context_id, ""),
                "type": row["event_type"],
                "title": _clip(row["title"], 180),
                "summary": _clip(row["summary"], 500),
                "structured_data": _compact_json(structured),
                "event_at": row["event_at"],
                "status": row["status"],
                "provenance": row["provenance"],
            }
            ranked.append((score, str(row["event_at"] or ""), payload))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[:12]]

    @staticmethod
    def _followups(
        conn: sqlite3.Connection,
        profile_id: str,
        allowed_context_ids: set[str],
        contexts: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT * FROM customer_followups
            WHERE customer_profile_id=? AND status='pending'
            ORDER BY due_at ASC,priority DESC,created_at ASC
            LIMIT 30
            """,
            (profile_id,),
        ).fetchall()
        context_crop = {item["id"]: item["crop"] for item in contexts}
        result = []
        for row in rows:
            context_id = row["growing_context_id"]
            if context_id is not None and context_id not in allowed_context_ids:
                continue
            result.append(
                {
                    "id": row["id"],
                    "context_id": context_id,
                    "context_crop": context_crop.get(context_id, ""),
                    "type": _clip(row["followup_type"], 80),
                    "title": _clip(row["title"], 180),
                    "due_at": row["due_at"],
                    "priority": row["priority"],
                    "status": row["status"],
                }
            )
        return result[:5]

    def _log_failure(self, exc: Exception, account_scope_id: str, contact_id: str, conversation_id: str) -> None:
        if self.logs is None:
            return
        try:
            self.logs.append(
                "error",
                "reply_memory",
                "reply_memory_context_fallback",
                (
                    f"error={type(exc).__name__} scope={account_scope_id} "
                    f"contact={contact_id} conversation={conversation_id}"
                ),
            )
        except Exception:
            pass


def _normal(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _compact_json(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in list(value.items())[:12]:
        clean_key = _clip(key, 60)
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[clean_key] = _clip(item, 180) if isinstance(item, str) else item
    return result
