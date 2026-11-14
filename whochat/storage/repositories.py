from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import asdict, replace

from whochat.core.models import (
    AccountScope,
    AIAnalysisEvent,
    AppLog,
    CaptureSample,
    Contact,
    ContactAlias,
    ContactIdentityBinding,
    ConversationObservation,
    ContactPersonLink,
    ContactStatus,
    ConversationType,
    Conversation,
    ConversationSegment,
    ConversationSession,
    GroupMember,
    IdentityStatus,
    GenerationLog,
    Memory,
    MemoryKind,
    MemoryStatus,
    Message,
    Person,
    PersonAlias,
    ReplyFeedback,
    SettingsAudit,
    Speaker,
    Strategy,
    TargetInstance,
    LEGACY_ACCOUNT_SCOPE_ID,
    utc_now_iso,
)
from whochat.core.runtime import LayoutCalibration, RelativeRect, TargetApp, ThemeMode
from whochat.storage.database import Database


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class AccountScopeRepository:
    """Persistently separates local WeChat accounts from runtime HWND/PID values."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_scope(self, scope_id: str) -> AccountScope | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM account_scopes WHERE id = ?", (scope_id,)).fetchone()
        return _account_scope_from_row(row) if row else None

    def native_account_mappings(self) -> list[dict]:
        """Read existing verified mappings; never create or repair a binding."""
        with self.db.connect() as conn:
            rows = conn.execute("""SELECT h.account_scope_id,h.stable_identifier,
                h.source_data_directory,h.mapping_status,h.match_basis,h.confidence
                FROM historical_source_accounts h JOIN account_scopes a ON a.id=h.account_scope_id
                WHERE a.status='active' AND a.platform='wechat'""").fetchall()
        return [dict(row) for row in rows]

    def get_or_create_scope(self, local_name: str, platform: str = "wechat") -> AccountScope:
        name = " ".join(local_name.strip().split())[:80]
        if not name:
            raise ValueError("account scope local name is required")
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM account_scopes WHERE platform = ? AND local_name = ?",
                (platform, name),
            ).fetchone()
            if row:
                return _account_scope_from_row(row)
            now = utc_now_iso()
            item = AccountScope(new_id("account_scope"), platform, name, "active", now, now)
            conn.execute(
                """
                INSERT INTO account_scopes(id, platform, local_name, status, created_at, updated_at)
                VALUES (:id, :platform, :local_name, :status, :created_at, :updated_at)
                """,
                asdict(item),
            )
        return item

    def find_instance(self, runtime_key: str, visual_hash: str) -> TargetInstance | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM target_instances WHERE runtime_key = ? AND visual_hash = ?",
                (runtime_key, visual_hash),
            ).fetchone()
        return _target_instance_from_row(row) if row else None

    def find_runtime_instance(self, runtime_key: str) -> TargetInstance | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM target_instances WHERE runtime_key = ? ORDER BY last_seen_at DESC LIMIT 1",
                (runtime_key,),
            ).fetchone()
        return _target_instance_from_row(row) if row else None

    def list_runtime_instances(self, runtime_key: str) -> list[TargetInstance]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM target_instances WHERE runtime_key = ? ORDER BY last_seen_at DESC",
                (runtime_key,),
            ).fetchall()
        return [_target_instance_from_row(row) for row in rows]

    def list_bound_instances(self, platform: str) -> list[TargetInstance]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM target_instances
                WHERE platform = ? AND account_scope_id IS NOT NULL AND visual_reliable = 1
                ORDER BY last_seen_at DESC
                """,
                (platform,),
            ).fetchall()
        return [_target_instance_from_row(row) for row in rows]

    def observe_instance(
        self,
        *,
        platform: str,
        runtime_key: str,
        visual_hash: str,
        visual_reliable: bool,
        process_name: str,
        executable_path: str,
        pid: int | None,
        hwnd: int | None,
        rect_json: str,
        matched_scope_id: str | None = None,
        binding_source: str = "unbound",
        confidence: float = 0.0,
        reuse_runtime_instance: bool = True,
    ) -> TargetInstance:
        current = self.find_instance(runtime_key, visual_hash)
        if current is None and reuse_runtime_instance:
            current = self.find_runtime_instance(runtime_key)
        now = utc_now_iso()
        if current is not None:
            scope_id = current.account_scope_id or matched_scope_id
            source = current.binding_source if current.account_scope_id else binding_source
            next_visual_hash = current.visual_hash
            next_visual_reliable = current.visual_reliable
            if visual_reliable and (not current.visual_reliable or not current.visual_hash):
                next_visual_hash = visual_hash
                next_visual_reliable = True
            with self.db.connect() as conn:
                conn.execute(
                    """
                    UPDATE target_instances
                    SET account_scope_id = ?, visual_hash = ?, visual_reliable = ?, process_name = ?, executable_path = ?,
                        last_pid = ?, last_hwnd = ?, last_rect_json = ?, binding_source = ?,
                        confidence = MAX(confidence, ?), last_seen_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        scope_id, next_visual_hash, int(next_visual_reliable), process_name, executable_path, pid, hwnd, rect_json,
                        source, confidence, now, now, current.id,
                    ),
                )
            return self.get_instance(current.id) or current
        item = TargetInstance(
            id=new_id("target_instance"),
            platform=platform,
            account_scope_id=matched_scope_id,
            runtime_key=runtime_key,
            visual_hash=visual_hash,
            visual_reliable=visual_reliable,
            process_name=process_name,
            executable_path=executable_path,
            last_pid=pid,
            last_hwnd=hwnd,
            last_rect_json=rect_json,
            binding_source=binding_source,
            confidence=confidence,
            first_seen_at=now,
            last_seen_at=now,
            created_at=now,
            updated_at=now,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO target_instances(
                    id, platform, account_scope_id, runtime_key, visual_hash, visual_reliable,
                    process_name, executable_path, last_pid, last_hwnd, last_rect_json,
                    binding_source, confidence, first_seen_at, last_seen_at, created_at, updated_at
                ) VALUES (
                    :id, :platform, :account_scope_id, :runtime_key, :visual_hash, :visual_reliable,
                    :process_name, :executable_path, :last_pid, :last_hwnd, :last_rect_json,
                    :binding_source, :confidence, :first_seen_at, :last_seen_at, :created_at, :updated_at
                )
                """,
                _target_instance_params(item),
            )
        return item

    def get_instance(self, instance_id: str) -> TargetInstance | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM target_instances WHERE id = ?", (instance_id,)).fetchone()
        return _target_instance_from_row(row) if row else None

    def bind_instance(self, instance_id: str, local_name: str) -> TargetInstance:
        instance = self.get_instance(instance_id)
        if instance is None:
            raise ValueError("target instance not found")
        if instance.account_scope_id:
            current_scope = self.get_scope(instance.account_scope_id)
            normalized_name = " ".join(local_name.strip().split())[:80]
            if current_scope is not None and current_scope.local_name != normalized_name:
                raise ValueError(f"target instance is already bound to {current_scope.local_name}")
            return instance
        scope = self.get_or_create_scope(local_name, instance.platform)
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE target_instances
                SET account_scope_id = ?, binding_source = 'manual', confidence = 1.0,
                    last_seen_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (scope.id, now, now, instance_id),
            )
        return self.get_instance(instance_id) or instance

    def complete_avatar_fingerprint_migration(
        self,
        *,
        instance_id: str,
        account_scope_id: str,
        visual_hash: str,
    ) -> TargetInstance | None:
        """Atomically complete an explicitly staged avatar ROI migration."""

        if not visual_hash.startswith("v2:"):
            return None
        now = utc_now_iso()
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE target_instances
                SET visual_hash = ?, visual_reliable = 1,
                    binding_source = 'avatar_roi_v2_migrated',
                    confidence = MAX(confidence, 0.95),
                    last_seen_at = ?, updated_at = ?
                WHERE id = ? AND account_scope_id = ?
                  AND binding_source = 'avatar_roi_v2_migration_pending'
                """,
                (visual_hash, now, now, instance_id, account_scope_id),
            )
            if cursor.rowcount != 1:
                return None
        return self.get_instance(instance_id)

    def list_scopes(self) -> list[AccountScope]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM account_scopes WHERE status != 'legacy' ORDER BY local_name"
            ).fetchall()
        return [_account_scope_from_row(row) for row in rows]


class StrategyRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_defaults(self) -> None:
        now = utc_now_iso()
        defaults = [
            Strategy("default", "默认", "根据上下文给出自然、稳妥的回复建议", "默认", "自然、清晰", "", "回复1,回复2", False, now, now),
            Strategy("leader_boundary", "领导-保持边界", "清楚表达进度和边界，避免过度承诺", "保持边界", "尊重、简洁、事实导向", "抱怨,甩锅,阴阳怪气", "回复1,回复2", False, now, now),
            Strategy("client", "客户", "推进下一步并降低不确定感", "推进", "专业、明确、有耐心", "过度承诺,模糊时间", "回复1,回复2", False, now, now),
            Strategy("manual_protect", "手动回复保护", "只整理信息，不主动代写", "保护", "用户亲自回复", "代写亲密关系回复", "", True, now, now),
        ]
        with self.db.connect() as conn:
            for item in defaults:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO strategies
                    (id, name, goal, mode, tone, avoid, reply_variants, requires_manual_reply, created_at, updated_at)
                    VALUES (:id, :name, :goal, :mode, :tone, :avoid, :reply_variants, :requires_manual_reply, :created_at, :updated_at)
                    """,
                    {**asdict(item), "requires_manual_reply": int(item.requires_manual_reply)},
                )

    def list_all(self, include_archived: bool = True) -> list[Strategy]:
        with self.db.connect() as conn:
            where = "" if include_archived else "WHERE archived = 0"
            rows = conn.execute(
                f"SELECT * FROM strategies {where} ORDER BY archived, requires_manual_reply, name"
            ).fetchall()
        return [_strategy_from_row(row) for row in rows]

    def list_active(self) -> list[Strategy]:
        return self.list_all(include_archived=False)

    def get(self, strategy_id: str) -> Strategy | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        return _strategy_from_row(row) if row else None

    def create(
        self,
        name: str,
        goal: str,
        mode: str,
        tone: str,
        avoid: str = "",
        reply_variants: str = "",
        requires_manual_reply: bool = False,
    ) -> Strategy:
        now = utc_now_iso()
        strategy = Strategy(
            id=new_id("strategy"),
            name=name,
            goal=goal,
            mode=mode,
            tone=tone,
            avoid=avoid,
            reply_variants=reply_variants,
            requires_manual_reply=requires_manual_reply,
            created_at=now,
            updated_at=now,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO strategies
                (id, name, goal, mode, tone, avoid, reply_variants, requires_manual_reply, created_at, updated_at)
                VALUES (:id, :name, :goal, :mode, :tone, :avoid, :reply_variants, :requires_manual_reply, :created_at, :updated_at)
                """,
                {**asdict(strategy), "requires_manual_reply": int(strategy.requires_manual_reply)},
            )
        return strategy

    def update(self, strategy: Strategy) -> Strategy:
        updated = Strategy(**{**asdict(strategy), "updated_at": utc_now_iso()})
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE strategies
                SET name = :name,
                    goal = :goal,
                    mode = :mode,
                    tone = :tone,
                    avoid = :avoid,
                    reply_variants = :reply_variants,
                    requires_manual_reply = :requires_manual_reply,
                    archived = :archived,
                    updated_at = :updated_at
                WHERE id = :id
                """,
                {
                    **asdict(updated),
                    "requires_manual_reply": int(updated.requires_manual_reply),
                    "archived": int(updated.archived),
                },
            )
        return updated

    def set_archived(self, strategy_id: str, archived: bool) -> Strategy:
        current = self.get(strategy_id)
        if current is None:
            raise ValueError(f"strategy not found: {strategy_id}")
        if strategy_id in {"default", "manual_protect"} and archived:
            raise ValueError("built-in safety strategies cannot be archived")
        updated = Strategy(**{**asdict(current), "archived": archived, "updated_at": utc_now_iso()})
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE strategies SET archived = ?, updated_at = ? WHERE id = ?",
                (int(archived), updated.updated_at, strategy_id),
            )
        return updated

    def count_assigned_contacts(self, strategy_id: str) -> int:
        with self.db.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM contacts WHERE strategy_id = ?", (strategy_id,)).fetchone()
        return int(row["count"]) if row else 0


class ContactRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_or_get_by_display_name(
        self,
        display_name: str,
        platform: str = "wechat",
        conversation_type: ConversationType = ConversationType.UNKNOWN,
        strategy_id: str = "default",
        account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID,
    ) -> Contact:
        with self.db.connect() as conn:
            existing = conn.execute(
                """
                SELECT contacts.* FROM contacts
                LEFT JOIN contact_aliases ON contact_aliases.contact_id = contacts.id
                WHERE contacts.platform = ?
                  AND contacts.account_scope_id = ?
                  AND contacts.merged_into IS NULL
                  AND (contacts.display_name = ? OR contact_aliases.alias = ?)
                ORDER BY contacts.created_at LIMIT 1
                """,
                (platform, account_scope_id, display_name, display_name),
            ).fetchone()
            if existing:
                return _contact_from_row(existing)
            normalized_display_name = _normalized_contact_name(display_name)
            normalized_matches: dict[str, object] = {}
            for row in conn.execute(
                """
                SELECT contacts.*, contact_aliases.alias AS candidate_alias
                FROM contacts
                LEFT JOIN contact_aliases ON contact_aliases.contact_id = contacts.id
                WHERE contacts.platform = ?
                  AND contacts.account_scope_id = ?
                  AND contacts.merged_into IS NULL
                ORDER BY contacts.created_at
                """,
                (platform, account_scope_id),
            ).fetchall():
                if (
                    _normalized_contact_name(str(row["display_name"] or "")) == normalized_display_name
                    or _normalized_contact_name(str(row["candidate_alias"] or "")) == normalized_display_name
                ):
                    normalized_matches[str(row["id"])] = row
            if normalized_matches:
                # Exact-name lookup already has the same oldest-record
                # behavior. This extension only makes Unicode/whitespace OCR
                # variants converge within the same account and platform.
                matched_id = next(iter(normalized_matches))
                matched = conn.execute(
                    "SELECT * FROM contacts WHERE id = ?",
                    (matched_id,),
                ).fetchone()
                if matched is not None:
                    return _contact_from_row(matched)
            now = utc_now_iso()
            contact = Contact(
                id=new_id("contact"),
                platform=platform,
                display_name=display_name,
                conversation_type=conversation_type,
                status=ContactStatus.UNCONFIRMED,
                strategy_id=strategy_id,
                # Cloud AI is opt-in: a contact discovered by OCR must never have its
                # chat content sent to a third-party provider until the user grants
                # it explicitly ("AI开/AI关" on the contacts page, or the checkbox in
                # the contact edit dialog).
                allow_cloud_ai=False,
                remark="",
                avatar_fingerprint="",
                merged_into=None,
                created_at=now,
                updated_at=now,
                account_scope_id=account_scope_id,
            )
            conn.execute(
                """
                INSERT INTO contacts
                (id, platform, display_name, conversation_type, status, strategy_id, allow_cloud_ai, remark, avatar_fingerprint, merged_into, created_at, updated_at, account_scope_id)
                VALUES (:id, :platform, :display_name, :conversation_type, :status, :strategy_id, :allow_cloud_ai, :remark, :avatar_fingerprint, :merged_into, :created_at, :updated_at, :account_scope_id)
                """,
                _contact_params(contact),
            )
            conn.execute(
                "INSERT INTO contact_aliases(id, contact_id, alias, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (new_id("alias"), contact.id, display_name, "display_name", now),
            )
            return contact

    def list_recent(self, limit: int = 100, include_merged: bool = False) -> list[Contact]:
        with self.db.connect() as conn:
            where = "" if include_merged else "WHERE merged_into IS NULL"
            rows = conn.execute(f"SELECT * FROM contacts {where} ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [_contact_from_row(row) for row in rows]

    def get(self, contact_id: str) -> Contact | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        return _contact_from_row(row) if row else None

    def update_profile(
        self,
        contact_id: str,
        *,
        display_name: str | None = None,
        status: ContactStatus | None = None,
        strategy_id: str | None = None,
        allow_cloud_ai: bool | None = None,
        remark: str | None = None,
    ) -> Contact:
        current = self.get(contact_id)
        if current is None:
            raise ValueError(f"contact not found: {contact_id}")
        if (
            (display_name is None or display_name == current.display_name)
            and (status is None or status == current.status)
            and (strategy_id is None or strategy_id == current.strategy_id)
            and (allow_cloud_ai is None or allow_cloud_ai == current.allow_cloud_ai)
            and (remark is None or remark == current.remark)
        ):
            # Do not fire the contacts UPDATE trigger for a semantic no-op.
            # Besides unnecessary disk work, that trigger invalidates the
            # contact's Activation snapshot and can create rebuild loops.
            return current
        updated = Contact(
            id=current.id,
            platform=current.platform,
            display_name=display_name if display_name is not None else current.display_name,
            conversation_type=current.conversation_type,
            status=status if status is not None else current.status,
            strategy_id=strategy_id if strategy_id is not None else current.strategy_id,
            allow_cloud_ai=allow_cloud_ai if allow_cloud_ai is not None else current.allow_cloud_ai,
            remark=remark if remark is not None else current.remark,
            avatar_fingerprint=current.avatar_fingerprint,
            merged_into=current.merged_into,
            created_at=current.created_at,
            updated_at=utc_now_iso(),
            account_scope_id=current.account_scope_id,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE contacts
                SET display_name = :display_name,
                    status = :status,
                    strategy_id = :strategy_id,
                    allow_cloud_ai = :allow_cloud_ai,
                    remark = :remark,
                    updated_at = :updated_at
                WHERE id = :id
                """,
                _contact_params(updated),
            )
        return updated

    def list_aliases(self, contact_id: str) -> list[ContactAlias]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM contact_aliases WHERE contact_id = ? ORDER BY created_at",
                (contact_id,),
            ).fetchall()
        return [ContactAlias(**dict(row)) for row in rows]

    def add_alias(self, contact_id: str, alias: str, source: str = "manual") -> ContactAlias | None:
        alias = alias.strip()
        if not alias:
            return None
        if self.get(contact_id) is None:
            raise ValueError(f"contact not found: {contact_id}")
        now = utc_now_iso()
        item = ContactAlias(new_id("alias"), contact_id, alias, source, now)
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO contact_aliases(id, contact_id, alias, source, created_at)
                VALUES (:id, :contact_id, :alias, :source, :created_at)
                """,
                asdict(item),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT * FROM contact_aliases WHERE contact_id = ? AND alias = ?",
                    (contact_id, alias),
                ).fetchone()
                return ContactAlias(**dict(row)) if row else None
        return item

    def merge_contacts(self, source_id: str, target_id: str) -> Contact:
        if source_id == target_id:
            raise ValueError("cannot merge contact into itself")
        source = self.get(source_id)
        target = self.get(target_id)
        if source is None or target is None:
            raise ValueError("source or target contact not found")
        if source.merged_into or source.status == ContactStatus.MERGED:
            raise ValueError("source contact has already been merged")
        if target.merged_into or target.status == ContactStatus.MERGED:
            raise ValueError("target contact cannot be a merged contact")
        if source.platform != target.platform:
            raise ValueError("cannot merge contacts from different platforms")
        if source.account_scope_id != target.account_scope_id:
            raise ValueError("cannot merge contacts from different account scopes")

        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                DELETE FROM messages
                WHERE contact_id = ?
                  AND fingerprint IN (SELECT fingerprint FROM messages WHERE contact_id = ?)
                """,
                (source_id, target_id),
            )
            source_conversations = conn.execute(
                "SELECT * FROM conversations WHERE contact_id = ? ORDER BY created_at",
                (source_id,),
            ).fetchall()
            for source_conversation in source_conversations:
                target_conversation = conn.execute(
                    """
                    SELECT * FROM conversations
                    WHERE contact_id = ? AND account_scope_id = ? AND platform = ? AND conversation_type = ?
                    """,
                    (
                        target_id,
                        source_conversation["account_scope_id"],
                        source_conversation["platform"],
                        source_conversation["conversation_type"],
                    ),
                ).fetchone()
                if target_conversation is None:
                    conn.execute(
                        "UPDATE conversations SET contact_id = ?, updated_at = ? WHERE id = ?",
                        (target_id, now, source_conversation["id"]),
                    )
                    continue

                target_conversation_id = target_conversation["id"]
                source_conversation_id = source_conversation["id"]
                target_max = conn.execute(
                    "SELECT COALESCE(MAX(sequence_no), 0) AS maximum FROM messages WHERE conversation_id = ?",
                    (target_conversation_id,),
                ).fetchone()["maximum"]
                conn.execute(
                    """
                    UPDATE messages
                    SET sequence_no = ? + COALESCE(sequence_no, 0), order_uncertain = 1
                    WHERE conversation_id = ?
                    """,
                    (int(target_max), source_conversation_id),
                )
                for source_segment in conn.execute(
                    "SELECT * FROM conversation_segments WHERE conversation_id = ?",
                    (source_conversation_id,),
                ).fetchall():
                    duplicate_segment = conn.execute(
                        """
                        SELECT * FROM conversation_segments
                        WHERE conversation_id = ? AND snapshot_hash = ?
                        """,
                        (target_conversation_id, source_segment["snapshot_hash"]),
                    ).fetchone()
                    if duplicate_segment is not None:
                        conn.execute(
                            """
                            UPDATE messages
                            SET conversation_id = ?, session_id = ?, segment_id = ?, order_uncertain = 1
                            WHERE segment_id = ?
                            """,
                            (
                                target_conversation_id,
                                duplicate_segment["session_id"],
                                duplicate_segment["id"],
                                source_segment["id"],
                            ),
                        )
                        conn.execute("DELETE FROM conversation_segments WHERE id = ?", (source_segment["id"],))
                    else:
                        conn.execute(
                            "UPDATE conversation_segments SET conversation_id = ? WHERE id = ?",
                            (target_conversation_id, source_segment["id"]),
                        )
                conn.execute(
                    "UPDATE messages SET conversation_id = ? WHERE conversation_id = ?",
                    (target_conversation_id, source_conversation_id),
                )
                conn.execute(
                    "UPDATE conversation_sessions SET conversation_id = ? WHERE conversation_id = ?",
                    (target_conversation_id, source_conversation_id),
                )
                conn.execute("DELETE FROM conversations WHERE id = ?", (source_conversation_id,))

            conn.execute("UPDATE messages SET contact_id = ? WHERE contact_id = ?", (target_id, source_id))
            conn.execute("UPDATE ai_analysis_events SET contact_id = ? WHERE contact_id = ?", (target_id, source_id))
            conn.execute("UPDATE contact_identity_bindings SET contact_id = ?, updated_at = ? WHERE contact_id = ?", (target_id, now, source_id))
            conn.execute("UPDATE reply_feedback SET contact_id = ? WHERE contact_id = ?", (target_id, source_id))
            conn.execute(
                """
                DELETE FROM memories
                WHERE contact_id = ?
                  AND fingerprint IN (SELECT fingerprint FROM memories WHERE contact_id = ?)
                """,
                (source_id, target_id),
            )
            conn.execute("UPDATE memories SET contact_id = ?, updated_at = ? WHERE contact_id = ?", (target_id, now, source_id))
            conn.execute("UPDATE generation_logs SET contact_id = ? WHERE contact_id = ?", (target_id, source_id))
            conn.execute(
                """
                INSERT OR IGNORE INTO contact_aliases(id, contact_id, alias, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (new_id("alias"), target_id, source.display_name, "merged_display_name", now),
            )
            for row in conn.execute("SELECT alias, source FROM contact_aliases WHERE contact_id = ?", (source_id,)):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO contact_aliases(id, contact_id, alias, source, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (new_id("alias"), target_id, row["alias"], f"merged:{row['source']}", now),
                )
            conn.execute("DELETE FROM contact_aliases WHERE contact_id = ?", (source_id,))
            conn.execute(
                """
                UPDATE contacts
                SET status = ?, merged_into = ?, updated_at = ?
                WHERE id = ?
                """,
                (ContactStatus.MERGED.value, target_id, now, source_id),
            )
            conn.execute("UPDATE contacts SET updated_at = ? WHERE id = ?", (now, target_id))
        refreshed = self.get(target_id)
        if refreshed is None:
            raise ValueError(f"target contact not found after merge: {target_id}")
        return refreshed


class IdentityRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_person(self, display_name: str, source: str = "manual", status: IdentityStatus = IdentityStatus.SUSPECTED) -> Person:
        now = utc_now_iso()
        person = Person(
            id=new_id("person"),
            display_name=display_name.strip(),
            status=status,
            remark="",
            created_at=now,
            updated_at=now,
        )
        if not person.display_name:
            raise ValueError("person display name cannot be empty")
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO people(id, display_name, status, remark, created_at, updated_at)
                VALUES (:id, :display_name, :status, :remark, :created_at, :updated_at)
                """,
                _person_params(person),
            )
            conn.execute(
                "INSERT OR IGNORE INTO person_aliases(id, person_id, alias, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (new_id("person_alias"), person.id, person.display_name, source, now),
            )
        return person

    def get_person(self, person_id: str) -> Person | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
        return _person_from_row(row) if row else None

    def find_people_by_alias(self, alias: str) -> list[Person]:
        alias = alias.strip()
        if not alias:
            return []
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT people.* FROM people
                LEFT JOIN person_aliases ON person_aliases.person_id = people.id
                WHERE people.display_name = ? OR person_aliases.alias = ?
                ORDER BY people.updated_at DESC
                """,
                (alias, alias),
            ).fetchall()
        return [_person_from_row(row) for row in rows]

    def add_person_alias(self, person_id: str, alias: str, source: str = "manual") -> PersonAlias | None:
        alias = alias.strip()
        if not alias:
            return None
        if self.get_person(person_id) is None:
            raise ValueError(f"person not found: {person_id}")
        now = utc_now_iso()
        item = PersonAlias(new_id("person_alias"), person_id, alias, source, now)
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO person_aliases(id, person_id, alias, source, created_at)
                VALUES (:id, :person_id, :alias, :source, :created_at)
                """,
                asdict(item),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT * FROM person_aliases WHERE person_id = ? AND alias = ?",
                    (person_id, alias),
                ).fetchone()
                return PersonAlias(**dict(row)) if row else None
        return item

    def list_person_aliases(self, person_id: str) -> list[PersonAlias]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM person_aliases WHERE person_id = ? ORDER BY created_at",
                (person_id,),
            ).fetchall()
        return [PersonAlias(**dict(row)) for row in rows]

    def link_contact_to_person(
        self,
        contact_id: str,
        person_id: str,
        *,
        confidence: float,
        source: str,
        verified: bool = False,
    ) -> ContactPersonLink:
        now = utc_now_iso()
        link = ContactPersonLink(
            id=new_id("contact_person"),
            contact_id=contact_id,
            person_id=person_id,
            confidence=max(0.0, min(confidence, 1.0)),
            source=source,
            verified=verified,
            created_at=now,
            updated_at=now,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO contact_person_links(id, contact_id, person_id, confidence, source, verified, created_at, updated_at)
                VALUES (:id, :contact_id, :person_id, :confidence, :source, :verified, :created_at, :updated_at)
                ON CONFLICT(contact_id, person_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    source = excluded.source,
                    verified = excluded.verified,
                    updated_at = excluded.updated_at
                """,
                _contact_person_link_params(link),
            )
            row = conn.execute(
                """
                SELECT * FROM contact_person_links
                WHERE contact_id = ? AND person_id = ?
                """,
                (contact_id, person_id),
            ).fetchone()
        return _contact_person_link_from_row(row)

    def list_people_for_contact(self, contact_id: str) -> list[tuple[Person, ContactPersonLink]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT people.*, contact_person_links.id AS link_id,
                       contact_person_links.contact_id AS link_contact_id,
                       contact_person_links.person_id AS link_person_id,
                       contact_person_links.confidence AS link_confidence,
                       contact_person_links.source AS link_source,
                       contact_person_links.verified AS link_verified,
                       contact_person_links.created_at AS link_created_at,
                       contact_person_links.updated_at AS link_updated_at
                FROM contact_person_links
                JOIN people ON people.id = contact_person_links.person_id
                WHERE contact_person_links.contact_id = ?
                ORDER BY contact_person_links.verified DESC, contact_person_links.confidence DESC
                """,
                (contact_id,),
            ).fetchall()
        return [(_person_from_row(row), _contact_person_link_from_joined_row(row)) for row in rows]

    def add_group_member(
        self,
        *,
        group_contact_id: str,
        member_display_name: str,
        person_id: str | None = None,
        platform_contact_id: str | None = None,
        confidence: float = 0.5,
        source: str = "ocr",
    ) -> GroupMember:
        now = utc_now_iso()
        item = GroupMember(
            id=new_id("group_member"),
            group_contact_id=group_contact_id,
            member_display_name=member_display_name.strip(),
            person_id=person_id,
            platform_contact_id=platform_contact_id,
            confidence=max(0.0, min(confidence, 1.0)),
            source=source,
            created_at=now,
            updated_at=now,
        )
        if not item.member_display_name:
            raise ValueError("group member display name cannot be empty")
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO group_members
                (id, group_contact_id, member_display_name, person_id, platform_contact_id, confidence, source, created_at, updated_at)
                VALUES
                (:id, :group_contact_id, :member_display_name, :person_id, :platform_contact_id, :confidence, :source, :created_at, :updated_at)
                ON CONFLICT(group_contact_id, member_display_name) DO UPDATE SET
                    person_id = excluded.person_id,
                    platform_contact_id = excluded.platform_contact_id,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                _group_member_params(item),
            )
            row = conn.execute(
                "SELECT * FROM group_members WHERE group_contact_id = ? AND member_display_name = ?",
                (group_contact_id, item.member_display_name),
            ).fetchone()
        return _group_member_from_row(row)

    def list_group_members(self, group_contact_id: str) -> list[GroupMember]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM group_members WHERE group_contact_id = ? ORDER BY member_display_name",
                (group_contact_id,),
            ).fetchall()
        return [_group_member_from_row(row) for row in rows]


class IdentityBindingRepository:
    """Maps stable on-screen conversation identity to one local contact.

    Avatar fingerprints are intentionally optional in phase 1.  Keeping the
    column and lookup shape here lets a future non-invasive avatar matcher add
    evidence without changing the contact model again.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def find(
        self,
        platform: str,
        conversation_type: ConversationType,
        normalized_title: str,
        account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID,
    ) -> ContactIdentityBinding | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM contact_identity_bindings
                WHERE account_scope_id = ? AND platform = ? AND conversation_type = ? AND normalized_title = ?
                """,
                (account_scope_id, platform, conversation_type.value, _identity_title(normalized_title)),
            ).fetchone()
        return _identity_binding_from_row(row) if row else None

    def upsert(
        self,
        *,
        platform: str,
        conversation_type: ConversationType,
        normalized_title: str,
        contact_id: str,
        confidence: float,
        verified: bool = False,
        avatar_fingerprint: str = "",
        account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID,
    ) -> ContactIdentityBinding:
        now = utc_now_iso()
        title_key = _identity_title(normalized_title)
        current = self.find(platform, conversation_type, title_key, account_scope_id)
        record = ContactIdentityBinding(
            id=current.id if current else new_id("identity_binding"),
            platform=platform,
            conversation_type=conversation_type,
            normalized_title=title_key,
            contact_id=contact_id,
            avatar_fingerprint=avatar_fingerprint or (current.avatar_fingerprint if current else ""),
            confidence=max(confidence, current.confidence if current else 0.0),
            verified=verified or bool(current and current.verified),
            last_confirmed_at=now,
            created_at=current.created_at if current else now,
            updated_at=now,
            account_scope_id=account_scope_id,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO contact_identity_bindings
                (id, account_scope_id, platform, conversation_type, normalized_title, contact_id, avatar_fingerprint,
                 confidence, verified, last_confirmed_at, created_at, updated_at)
                VALUES (:id, :account_scope_id, :platform, :conversation_type, :normalized_title, :contact_id, :avatar_fingerprint,
                        :confidence, :verified, :last_confirmed_at, :created_at, :updated_at)
                ON CONFLICT(account_scope_id, platform, conversation_type, normalized_title) DO UPDATE SET
                    contact_id = excluded.contact_id,
                    avatar_fingerprint = CASE WHEN excluded.avatar_fingerprint = ''
                                              THEN contact_identity_bindings.avatar_fingerprint
                                              ELSE excluded.avatar_fingerprint END,
                    confidence = MAX(contact_identity_bindings.confidence, excluded.confidence),
                    verified = MAX(contact_identity_bindings.verified, excluded.verified),
                    last_confirmed_at = excluded.last_confirmed_at,
                    updated_at = excluded.updated_at
                """,
                _identity_binding_params(record),
            )
        return self.find(platform, conversation_type, title_key, account_scope_id) or record

    def mark_contact_verified(self, contact_id: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute(
                "UPDATE contact_identity_bindings SET verified = 1, updated_at = ? WHERE contact_id = ?",
                (utc_now_iso(), contact_id),
            )
            return cursor.rowcount


class ConversationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create(self, contact: Contact, identity_key: str) -> Conversation:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE contact_id = ? AND account_scope_id = ? AND platform = ? AND conversation_type = ?",
                (contact.id, contact.account_scope_id, contact.platform, contact.conversation_type.value),
            ).fetchone()
            if row:
                conn.execute("UPDATE conversations SET identity_key = ?, updated_at = ? WHERE id = ?", (identity_key, utc_now_iso(), row["id"]))
                return _conversation_from_row(row)
            now = utc_now_iso()
            item = Conversation(
                id=new_id("conversation"),
                contact_id=contact.id,
                platform=contact.platform,
                conversation_type=contact.conversation_type,
                identity_key=identity_key,
                created_at=now,
                updated_at=now,
                account_scope_id=contact.account_scope_id,
            )
            conn.execute(
                """
                INSERT INTO conversations(id, contact_id, platform, conversation_type, identity_key, created_at, updated_at, account_scope_id)
                VALUES (:id, :contact_id, :platform, :conversation_type, :identity_key, :created_at, :updated_at, :account_scope_id)
                """,
                _conversation_params(item),
            )
        return item

    def get(self, conversation_id: str) -> Conversation | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return _conversation_from_row(row) if row else None

    def start_session(
        self,
        conversation_id: str,
        *,
        hwnd: int | None,
        stable_title: str,
        identity_confidence: float,
        source: str = "stable_ocr_identity",
        account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID,
        target_instance_id: str = "",
        capture_generation: int = 0,
    ) -> ConversationSession:
        now = utc_now_iso()
        item = ConversationSession(
            id=new_id("session"),
            conversation_id=conversation_id,
            hwnd=hwnd,
            stable_title=stable_title,
            identity_confidence=identity_confidence,
            started_at=now,
            last_seen_at=now,
            ended_at=None,
            source=source,
            account_scope_id=account_scope_id,
            target_instance_id=target_instance_id,
            capture_generation=capture_generation,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_sessions
                (id, conversation_id, hwnd, stable_title, identity_confidence, started_at, last_seen_at, ended_at, source,
                 account_scope_id, target_instance_id, capture_generation)
                VALUES (:id, :conversation_id, :hwnd, :stable_title, :identity_confidence, :started_at,
                        :last_seen_at, :ended_at, :source, :account_scope_id, :target_instance_id, :capture_generation)
                """,
                asdict(item),
            )
        return item

    def touch_session(self, session_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("UPDATE conversation_sessions SET last_seen_at = ? WHERE id = ?", (utc_now_iso(), session_id))

    def close_session(self, session_id: str) -> None:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE conversation_sessions SET last_seen_at = ?, ended_at = COALESCE(ended_at, ?) WHERE id = ?",
                (now, now, session_id),
            )

    def add_segment(
        self,
        *,
        conversation_id: str,
        session_id: str,
        snapshot_hash: str,
        observed_at: str,
        page_type: str,
        page_confidence: float,
        ocr_engine: str,
        reliable: bool,
        order_uncertain: bool,
        source_job_id: int | None,
        message_count: int,
        account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID,
        target_instance_id: str = "",
    ) -> tuple[ConversationSegment, bool]:
        item = ConversationSegment(
            id=new_id("segment"),
            conversation_id=conversation_id,
            session_id=session_id,
            snapshot_hash=snapshot_hash,
            observed_at=observed_at,
            page_type=page_type,
            page_confidence=page_confidence,
            ocr_engine=ocr_engine,
            reliable=reliable,
            order_uncertain=order_uncertain,
            source_job_id=source_job_id,
            message_count=message_count,
            account_scope_id=account_scope_id,
            target_instance_id=target_instance_id,
        )
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO conversation_segments
                (id, conversation_id, session_id, snapshot_hash, observed_at, page_type, page_confidence,
                 ocr_engine, reliable, order_uncertain, source_job_id, message_count, account_scope_id, target_instance_id)
                VALUES (:id, :conversation_id, :session_id, :snapshot_hash, :observed_at, :page_type,
                        :page_confidence, :ocr_engine, :reliable, :order_uncertain, :source_job_id, :message_count,
                        :account_scope_id, :target_instance_id)
                """,
                _conversation_segment_params(item),
            )
            if cursor.rowcount > 0:
                return item, True
            row = conn.execute(
                "SELECT * FROM conversation_segments WHERE conversation_id = ? AND snapshot_hash = ?",
                (conversation_id, snapshot_hash),
            ).fetchone()
        if row is None:
            raise RuntimeError("conversation segment insert failed")
        return _conversation_segment_from_row(row), False


class AIAnalysisEventRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def claim(
        self,
        message: Message,
        context_hash: str = "",
        *,
        target_instance_id: str = "",
        conversation_id: str | None = None,
        trigger_snapshot_json: str = "{}",
    ) -> AIAnalysisEvent | None:
        now = utc_now_iso()
        item = AIAnalysisEvent(
            id=new_id("analysis"),
            message_id=message.id,
            contact_id=message.contact_id,
            session_id=message.session_id,
            segment_id=message.segment_id,
            status="pending",
            context_hash=context_hash,
            created_at=now,
            updated_at=now,
            account_scope_id=message.account_scope_id,
            target_instance_id=target_instance_id,
            conversation_id=conversation_id or message.conversation_id,
            trigger_snapshot_json=trigger_snapshot_json,
        )
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO ai_analysis_events
                (id, message_id, contact_id, session_id, segment_id, status, context_hash, created_at, updated_at,
                 account_scope_id, target_instance_id, conversation_id, trigger_snapshot_json,
                 started_at, completed_at, failure_reason, result_json)
                VALUES (:id, :message_id, :contact_id, :session_id, :segment_id, :status, :context_hash, :created_at, :updated_at,
                        :account_scope_id, :target_instance_id, :conversation_id, :trigger_snapshot_json,
                        :started_at, :completed_at, :failure_reason, :result_json)
                """,
                asdict(item),
            )
        return item if cursor.rowcount > 0 else None

    def get(self, event_id: str) -> AIAnalysisEvent | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM ai_analysis_events WHERE id = ?", (event_id,)).fetchone()
        return _analysis_event_from_row(row) if row else None

    def get_for_message(self, message_id: str) -> AIAnalysisEvent | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM ai_analysis_events WHERE message_id = ?", (message_id,)).fetchone()
        return _analysis_event_from_row(row) if row else None

    def list_pending_for_binding(
        self,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[AIAnalysisEvent]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ai_analysis_events
                WHERE status = 'pending' AND account_scope_id = ? AND contact_id = ? AND conversation_id = ?
                ORDER BY created_at ASC LIMIT ?
                """,
                (account_scope_id, contact_id, conversation_id, limit),
            ).fetchall()
        return [_analysis_event_from_row(row) for row in rows]

    def latest_for_binding(
        self,
        account_scope_id: str,
        target_instance_id: str,
        contact_id: str,
        conversation_id: str,
    ) -> AIAnalysisEvent | None:
        """Return the newest event for one exact UI identity binding.

        This deliberately includes pending/running/failed rows.  A newer event
        must prevent an older completed result from masquerading as the latest
        analysis while work for a newer customer message is still in flight.
        """
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM ai_analysis_events
                WHERE account_scope_id = ? AND target_instance_id = ?
                  AND contact_id = ? AND conversation_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (account_scope_id, target_instance_id, contact_id, conversation_id),
            ).fetchone()
        return _analysis_event_from_row(row) if row else None

    def latest_for_conversation_binding(
        self,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
    ) -> AIAnalysisEvent | None:
        """Return the newest durable event for a stable customer conversation.

        ``target_instance_id`` identifies one runtime WeChat window incarnation.
        It is intentionally *not* part of this display-only lookup because the
        same manually bound work account can reconnect with a new window/target
        instance after WeChat or WhoChat restarts.  Account, contact and
        conversation remain mandatory, so results can never cross an account or
        customer boundary.

        New event execution continues to use :meth:`latest_for_binding` and the
        exact target-instance gate in the UI worker path.
        """
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM ai_analysis_events
                WHERE account_scope_id = ? AND contact_id = ? AND conversation_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (account_scope_id, contact_id, conversation_id),
            ).fetchone()
        return _analysis_event_from_row(row) if row else None

    def mark_started(self, message_id: str, context_hash: str = "") -> None:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE ai_analysis_events
                SET status = 'running',
                    context_hash = CASE WHEN ? = '' THEN context_hash ELSE ? END,
                    started_at = COALESCE(started_at, ?),
                    failure_reason = '',
                    updated_at = ?
                WHERE message_id = ? AND status = 'pending'
                """,
                (context_hash, context_hash, now, now, message_id),
            )

    def attach_memory_context_snapshot(
        self,
        message_id: str,
        *,
        payload: dict,
        memory_hash: str,
        status: str,
        error: str = "",
    ) -> bool:
        """Persist the first deterministic reply-memory snapshot on a durable event."""
        identity = payload.get("identity") if isinstance(payload, dict) else None
        if not isinstance(identity, dict) or identity.get("current_message_id") != message_id:
            raise ValueError("reply_memory_snapshot_message_mismatch")
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM ai_analysis_events WHERE message_id=?",
                (message_id,),
            ).fetchone()
            if row is None:
                raise ValueError("ai_analysis_event_not_found")
            if (
                identity.get("account_scope_id") != row["account_scope_id"]
                or identity.get("contact_id") != row["contact_id"]
                or identity.get("conversation_id") != row["conversation_id"]
            ):
                raise ValueError("reply_memory_snapshot_identity_mismatch")
            try:
                snapshot = json.loads(row["trigger_snapshot_json"] or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("invalid_ai_trigger_snapshot") from exc
            if not isinstance(snapshot, dict):
                raise ValueError("invalid_ai_trigger_snapshot")
            if isinstance(snapshot.get("reply_memory_context"), dict):
                return False
            snapshot["reply_memory_context"] = payload
            snapshot["reply_memory_hash"] = memory_hash
            snapshot["reply_memory_status"] = status
            snapshot["reply_memory_error"] = error
            conn.execute(
                "UPDATE ai_analysis_events SET trigger_snapshot_json=?,updated_at=? WHERE message_id=?",
                (
                    json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    utc_now_iso(),
                    message_id,
                ),
            )
            return True

    def update_status(
        self,
        message_id: str,
        status: str,
        context_hash: str = "",
        *,
        failure_reason: str = "",
        result_json: str | None = None,
    ) -> None:
        now = utc_now_iso()
        terminal = status not in {"pending", "running"}
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE ai_analysis_events SET status = ?, context_hash = CASE WHEN ? = '' THEN context_hash ELSE ? END,
                                              failure_reason = ?,
                                              result_json = CASE WHEN ? IS NULL THEN result_json ELSE ? END,
                                              completed_at = CASE WHEN ? THEN ? ELSE completed_at END,
                                              updated_at = ?
                WHERE message_id = ?
                """,
                (
                    status,
                    context_hash,
                    context_hash,
                    failure_reason,
                    result_json,
                    result_json,
                    terminal,
                    now,
                    now,
                    message_id,
                ),
            )

    def defer_pending(self, message_id: str, reason: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE ai_analysis_events
                SET status = 'pending', failure_reason = ?, started_at = NULL, updated_at = ?
                WHERE message_id = ? AND status IN ('pending', 'running')
                """,
                (reason, utc_now_iso(), message_id),
            )

    def release_pending(self, message_id: str) -> None:
        """Legacy compatibility: defer instead of deleting a durable trigger."""
        self.defer_pending(message_id, "legacy_release_requested")

    def recover_interrupted(self) -> int:
        """Return interrupted workers to pending without duplicating their event."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE ai_analysis_events
                SET status = 'pending', started_at = NULL,
                    failure_reason = 'worker_interrupted_by_restart', updated_at = ?
                WHERE status = 'running'
                """,
                (utc_now_iso(),),
            )
        return cursor.rowcount


class ObservationRepository:
    """Stores first-screen and uncertain OCR data outside the formal message timeline."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, item: ConversationObservation) -> tuple[ConversationObservation, bool]:
        from datetime import datetime, timedelta, timezone

        from whochat.core.current_conversation import (
            SCREEN_EVIDENCE_SCHEMA, build_current_message_view,
            current_screen_evidence, observation_semantic_payload,
            WINDOW_OBSERVATION_SCHEMA,
        )

        now = datetime.now(timezone.utc)

        def put_screen_marker(payload, marker):
            # Window-only observations carry no artificial first message.
            # Reuse the exact same validation/revocation/refresh rules below.
            if isinstance(payload,dict) and payload.get("schema") == WINDOW_OBSERVATION_SCHEMA:
                return {**payload,"screen_evidence":marker}
            if isinstance(payload,list) and payload and isinstance(payload[0],dict):
                payload[0]["_current_screen_evidence"]=marker
                return payload
            raise ValueError("conversation_observation_marker_shape_invalid")

        def evidence_is_fresh(observation: ConversationObservation) -> bool:
            try:
                observed = datetime.fromisoformat(observation.observed_at.replace("Z", "+00:00"))
                expires = datetime.fromisoformat(observation.expires_at.replace("Z", "+00:00"))
                return (
                    observed.tzinfo is not None and observed.utcoffset() is not None
                    and expires.tzinfo is not None and expires.utcoffset() is not None
                    and expires > now and expires > observed
                    and observed <= now + timedelta(minutes=5)
                    and type(observation.source_job_id) is int and observation.source_job_id > 0
                    and type(observation.capture_generation) is int and observation.capture_generation >= 0
                )
            except (ValueError, TypeError, AttributeError):
                return False

        semantic = observation_semantic_payload(item)
        trusted_relations = {
            "baseline", "known", "history", "append", "append_after_quarantine",
            "quarantine_confirmed", "duplicate_snapshot", "duplicate_segment",
        }
        lifecycle_keys = (
            "hwnd", "process_id", "window_generation", "capture_generation",
            "account_identity_epoch", "account_identity_runtime_key", "title_exact_hash",
        )
        with self.db.connect() as conn:
            # Serialize the semantic comparison and the write.  Pixel-only
            # repaints must not race into two inserts and fire invalidation.
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                """
                SELECT c.* FROM conversations AS c
                JOIN conversation_sessions AS s ON s.conversation_id=c.id
                JOIN target_instances AS t ON t.id=s.target_instance_id
                WHERE c.id=? AND s.id=? AND t.id=?
                  AND c.account_scope_id=? AND s.account_scope_id=? AND t.account_scope_id=?
                """,
                (item.conversation_id, item.session_id, item.target_instance_id,
                 item.account_scope_id, item.account_scope_id, item.account_scope_id),
            ).fetchone()
            if owner is None:
                raise ValueError("conversation_observation_owner_mismatch")
            conversation = _conversation_from_row(owner)
            live_owner = conn.execute(
                """SELECT c.id FROM contacts AS c
                   JOIN conversation_sessions AS s ON s.id=?
                   WHERE c.id=? AND c.account_scope_id=?
                     AND c.platform='wechat' AND c.conversation_type='dm'
                     AND c.status NOT IN ('ignored','merged') AND c.merged_into IS NULL
                     AND s.ended_at IS NULL""",
                (item.session_id, conversation.contact_id, item.account_scope_id),
            ).fetchone() is not None

            def validated_screen(observation: ConversationObservation, at: datetime) -> dict | None:
                proof = current_screen_evidence(observation)
                if proof is None or not live_owner:
                    return None
                try:
                    build_current_message_view(
                        observation, conversation=conversation,
                        expected_account_scope_id=item.account_scope_id,
                        expected_contact_id=conversation.contact_id,
                        expected_conversation_id=item.conversation_id,
                        expected_target_instance_id=item.target_instance_id,
                        expected_session_id=item.session_id,
                        max_source_job_id=observation.source_job_id,
                        max_capture_generation=observation.capture_generation, now=at,
                    )
                except (ValueError, TypeError, AttributeError, KeyError):
                    return None
                return proof

            supplied_proof = current_screen_evidence(item)
            screen = validated_screen(item, now)
            if supplied_proof is not None and screen is None:
                # Persist a negative capability, not a temporarily bad grant
                # which could become valid later as the clock catches up.
                rows = json.loads(item.messages_json)
                marker = {
                    "schema": SCREEN_EVIDENCE_SCHEMA, "valid": False,
                    "rejection_code": "repository_screen_validation_failed",
                    "source_job_id": item.source_job_id,
                }
                # Keep only type-safe lifecycle bindings for a later genuine
                # capture comparison. Never persist exception text here.
                for key in lifecycle_keys:
                    value = supplied_proof.get(key)
                    if ((key in {"account_identity_runtime_key", "title_exact_hash"}
                         and isinstance(value, str) and 0 < len(value) <= 512)
                            or (key not in {"account_identity_runtime_key", "title_exact_hash"}
                                and type(value) is int and value >= 0)):
                        marker[key] = value
                rows = put_screen_marker(rows,marker)
                item = replace(item, messages_json=json.dumps(rows, ensure_ascii=False))
                supplied_proof = marker

            latest_row = conn.execute(
                """SELECT * FROM conversation_observations
                   WHERE account_scope_id=? AND target_instance_id=?
                     AND conversation_id=? AND session_id=?
                   ORDER BY rowid DESC LIMIT 1""",
                (item.account_scope_id, item.target_instance_id, item.conversation_id, item.session_id),
            ).fetchone()
            latest = _observation_from_row(latest_row) if latest_row is not None else None
            if screen is not None and latest is not None and semantic is not None:
                previous_screen = validated_screen(latest, now)
                if (previous_screen is not None
                        and observation_semantic_payload(latest) == semantic
                        and all(previous_screen.get(key) == screen.get(key) for key in lifecycle_keys)):
                    # Repainting cannot refresh an existing 120s capability.
                    # In particular never scan past a newer negative frame.
                    return latest, False
            if (
                semantic is not None and item.status in {"baseline", "confirmed", "promoted"}
                and item.relation in trusted_relations and evidence_is_fresh(item)
                and supplied_proof is None
            ):
                row = conn.execute(
                    """
                    SELECT o.* FROM conversation_observations AS o
                    JOIN conversations AS c ON c.id=o.conversation_id
                    JOIN conversation_sessions AS s ON s.id=o.session_id
                    JOIN target_instances AS t ON t.id=o.target_instance_id
                    WHERE o.account_scope_id=? AND o.target_instance_id=?
                      AND o.conversation_id=? AND o.session_id=?
                      AND c.account_scope_id=o.account_scope_id
                      AND s.account_scope_id=o.account_scope_id
                      AND s.conversation_id=o.conversation_id
                      AND s.target_instance_id=o.target_instance_id
                      AND t.account_scope_id=o.account_scope_id
                    ORDER BY julianday(o.observed_at) DESC,
                             julianday(o.updated_at) DESC, o.rowid DESC LIMIT 1
                    """,
                    (item.account_scope_id, item.target_instance_id, item.conversation_id, item.session_id),
                ).fetchone()
                if row is not None:
                    previous = _observation_from_row(row)
                    if (
                        previous.status == item.status and previous.relation in trusted_relations
                        and evidence_is_fresh(previous)
                        and current_screen_evidence(previous) is None
                        and latest is not None and previous.id == latest.id
                        and observation_semantic_payload(previous) == semantic
                    ):
                        # No UPDATE either: schema 39 invalidates snapshots on
                        # every observation write.  The ingestion service keeps
                        # the latest capture/identity evidence independently.
                        return previous, False
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO conversation_observations(
                    id, account_scope_id, target_instance_id, conversation_id, session_id,
                    snapshot_hash, observed_at, status, relation, reason, messages_json,
                    keys_json, source_job_id, capture_generation, expires_at, created_at, updated_at
                ) VALUES (
                    :id, :account_scope_id, :target_instance_id, :conversation_id, :session_id,
                    :snapshot_hash, :observed_at, :status, :relation, :reason, :messages_json,
                    :keys_json, :source_job_id, :capture_generation, :expires_at, :created_at, :updated_at
                )
                """,
                _observation_params(item),
            )
            if cursor.rowcount > 0:
                return item, True
            row = conn.execute(
                """
                SELECT * FROM conversation_observations
                WHERE target_instance_id = ? AND conversation_id = ? AND session_id = ? AND snapshot_hash = ?
                """,
                (item.target_instance_id, item.conversation_id, item.session_id, item.snapshot_hash),
            ).fetchone()
            if row is None:
                raise RuntimeError("conversation observation insert failed")
            previous = _observation_from_row(row)
            if (
                previous.account_scope_id != item.account_scope_id
                or previous.target_instance_id != item.target_instance_id
                or previous.conversation_id != item.conversation_id
                or previous.session_id != item.session_id
            ):
                raise ValueError("conversation_observation_conflict_owner_mismatch")
            # An identical pixel hash does not imply identical semantic or
            # trust evidence.  In-place evidence transitions retain the real
            # pixel hash and stable ID rather than fabricate a second hash.
            previous_proof = current_screen_evidence(previous)
            negative_capability = screen is None and (supplied_proof is not None or previous_proof is not None)

            def existing_overlap_root() -> str | None:
                # Only actual, earlier same-owner rows classify isolation.
                # Neither a marker's claim nor a self-parent is authority.
                root_reasons = {
                    "persistent_tail_fallback_uncertain:runtime_anchor_missing",
                    "persistent_tail_fallback_uncertain:runtime_anchor_expired",
                }
                current = previous
                seen = set()
                for _ in range(64):
                    if current.id in seen or current.status != "quarantined":
                        return None
                    seen.add(current.id)
                    if current.reason in root_reasons:
                        return current.reason
                    prefix = "persistent_tail_fallback_repeated_quarantine:"
                    if not current.reason.startswith(prefix):
                        return None
                    parent = conn.execute(
                        "SELECT rowid AS source_rowid,* FROM conversation_observations WHERE id=?",
                        (current.reason[len(prefix):],),
                    ).fetchone()
                    child = conn.execute(
                        "SELECT rowid FROM conversation_observations WHERE id=?", (current.id,),
                    ).fetchone()
                    if (parent is None or child is None or parent["source_rowid"] >= child[0]
                            or any(parent[key] != getattr(previous, key) for key in (
                                "account_scope_id", "conversation_id", "target_instance_id", "session_id"))):
                        return None
                    parent_data = dict(parent)
                    del parent_data["source_rowid"]
                    current = ConversationObservation(**parent_data)
                return None

            if not negative_capability and not evidence_is_fresh(item):
                raise ValueError("conversation_observation_refresh_requires_fresh_evidence")
            if screen is not None:
                # A UNIQUE pixel collision may replace evidence only for a
                # genuinely later capture, never by extending an old proof.
                try:
                    previous_observed = datetime.fromisoformat(previous.observed_at.replace("Z", "+00:00"))
                    new_observed = datetime.fromisoformat(item.observed_at.replace("Z", "+00:00"))
                    native_fields_valid = (
                        type(previous.source_job_id) is int and previous.source_job_id > 0
                        and type(previous.capture_generation) is int and previous.capture_generation >= 0
                        and previous_observed.tzinfo is not None and previous_observed.utcoffset() is not None
                    )
                    old_structural = (validated_screen(previous, previous_observed)
                                      if previous_proof is not None and previous_proof.get("valid") is True else None)
                    old_negative = previous_proof is None or previous_proof.get("valid") is not True
                    compatible_lifecycle = previous_proof is None or all(
                        key not in previous_proof or previous_proof[key] == screen[key] for key in lifecycle_keys
                    )
                    fresh_capture = (
                        native_fields_valid and item.source_job_id > previous.source_job_id
                        and new_observed > previous_observed
                        and latest is not None and latest.id == previous.id
                        and compatible_lifecycle and (old_structural is not None or old_negative)
                    )
                except (ValueError, TypeError, AttributeError, KeyError):
                    fresh_capture = False
                self_parent = "persistent_tail_fallback_repeated_quarantine:" + previous.id
                if fresh_capture and item.status == "quarantined" and item.reason == self_parent:
                    root_reason = existing_overlap_root()
                    if root_reason is None:
                        fresh_capture = False
                    else:
                        rows = json.loads(item.messages_json)
                        parent_prefix = "persistent_tail_fallback_repeated_quarantine:"
                        rebound = {
                            **screen, "history_reason": previous.reason,
                            "history_root_reason": root_reason,
                            "history_parent_observation_id": (
                                previous.reason[len(parent_prefix):] if previous.reason.startswith(parent_prefix) else ""
                            ),
                        }
                        rows = put_screen_marker(rows,rebound)
                        candidate = replace(item, reason=previous.reason, messages_json=json.dumps(rows, ensure_ascii=False))
                        checked = validated_screen(candidate, now)
                        if checked is None:
                            fresh_capture = False
                        else:
                            item, screen = candidate, checked
                if not fresh_capture:
                    # A failed transition must revoke the old DB capability,
                    # not leave it usable during a queued live-lease race.
                    rows = json.loads(item.messages_json)
                    marker = {
                        "schema": SCREEN_EVIDENCE_SCHEMA, "valid": False,
                        "rejection_code": "repository_screen_refresh_not_new_capture",
                        "source_job_id": item.source_job_id,
                        **{key: screen[key] for key in lifecycle_keys},
                    }
                    rows = put_screen_marker(rows,marker)
                    item = replace(item, messages_json=json.dumps(rows, ensure_ascii=False))
                    negative_capability = True
            elif previous_proof is not None and supplied_proof is None:
                # Loss of native proof must revoke a previous capability even
                # when a caller supplied a legacy trusted-status snapshot.
                try:
                    rows = json.loads(item.messages_json)
                except (ValueError, TypeError):
                    rows = None
                if not ((isinstance(rows,list) and rows and isinstance(rows[0],dict))
                        or isinstance(rows,dict) and rows.get("schema") == WINDOW_OBSERVATION_SCHEMA):
                    rows = json.loads(previous.messages_json)
                rows = put_screen_marker(rows, {
                    "schema": SCREEN_EVIDENCE_SCHEMA, "valid": False,
                    "rejection_code": "repository_screen_evidence_missing",
                    "source_job_id": item.source_job_id,
                })
                item = replace(item, messages_json=json.dumps(rows, ensure_ascii=False))
            if (negative_capability and item.status == "quarantined"
                    and item.reason == "persistent_tail_fallback_repeated_quarantine:" + previous.id
                    and existing_overlap_root() is not None):
                # Revocation must not rewrite the retained row into its own
                # historical parent. This preserves classification only; the
                # permanent negative marker still grants no visible source.
                item = replace(item, reason=previous.reason, relation=previous.relation)
            if not negative_capability and not evidence_is_fresh(previous) and not (
                datetime.fromisoformat(item.observed_at.replace("Z", "+00:00"))
                > datetime.fromisoformat(previous.observed_at.replace("Z", "+00:00"))
                and (
                    item.source_job_id > (previous.source_job_id or 0)
                    or item.capture_generation > previous.capture_generation
                )
            ):
                raise ValueError("conversation_observation_expired_refresh_requires_new_capture")
            updated = replace(item, id=previous.id, updated_at=utc_now_iso())
            conn.execute(
                """
                UPDATE conversation_observations SET
                    observed_at=:observed_at,status=:status,relation=:relation,reason=:reason,
                    messages_json=:messages_json,keys_json=:keys_json,source_job_id=:source_job_id,
                    capture_generation=:capture_generation,expires_at=:expires_at,updated_at=:updated_at
                WHERE id=:id
                """,
                _observation_params(updated),
            )
            return updated, True

    def get(self, observation_id: str) -> ConversationObservation | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
        return _observation_from_row(row) if row else None

    def latest_confirmed_for_conversation(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        conversation_id: str,
        now_iso: str | None = None,
    ) -> ConversationObservation | None:
        """Return one unexpired two-frame-confirmed OCR snapshot.

        Account scope and conversation identity are both mandatory.  Baseline,
        quarantined and expired observations are intentionally excluded so a
        proactive assessment cannot promote uncertain screen text.
        """
        now_iso = now_iso or utc_now_iso()
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT o.* FROM conversation_observations AS o
                JOIN conversations AS c ON c.id = o.conversation_id
                WHERE o.account_scope_id = ? AND o.conversation_id = ?
                  AND c.contact_id = ? AND c.account_scope_id = ?
                  AND o.status IN ('confirmed','promoted')
                  AND o.expires_at > ?
                ORDER BY o.observed_at DESC, o.created_at DESC LIMIT 1
                """,
                (account_scope_id, conversation_id, contact_id, account_scope_id, now_iso),
            ).fetchone()
        return _observation_from_row(row) if row else None

    def latest_anchor(self, conversation_id: str, session_id: str) -> ConversationObservation | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM conversation_observations
                WHERE conversation_id = ? AND session_id = ?
                  AND status IN ('baseline', 'promoted', 'confirmed')
                ORDER BY observed_at DESC, created_at DESC LIMIT 1
                """,
                (conversation_id, session_id),
            ).fetchone()
        return _observation_from_row(row) if row else None

    def latest_quarantine(self, conversation_id: str, session_id: str) -> ConversationObservation | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM conversation_observations
                WHERE conversation_id = ? AND session_id = ? AND status = 'quarantined'
                ORDER BY observed_at DESC, created_at DESC LIMIT 1
                """,
                (conversation_id, session_id),
            ).fetchone()
        return _observation_from_row(row) if row else None

    def update_status(self, observation_id: str, status: str, reason: str = "") -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE conversation_observations
                SET status = ?, reason = CASE WHEN ? = '' THEN reason ELSE ? END, updated_at = ?
                WHERE id = ?
                """,
                (status, reason, reason, utc_now_iso(), observation_id),
            )

    def expire_due(self, now_iso: str | None = None) -> int:
        now = now_iso or utc_now_iso()
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE conversation_observations
                SET status = 'expired', updated_at = ?
                WHERE status = 'quarantined' AND expires_at <= ?
                """,
                (now, now),
            )
        return cursor.rowcount


class MessageRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add_message(self, message: Message) -> bool:
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO messages
                (id, contact_id, speaker, text, content_type, ocr_confidence, observed_at, message_time, time_source,
                 partial, fingerprint, source, sender_name, conversation_id, session_id, segment_id,
                 sequence_no, segment_index, bbox_json, order_uncertain, account_scope_id)
                VALUES (:id, :contact_id, :speaker, :text, :content_type, :ocr_confidence, :observed_at, :message_time,
                        :time_source, :partial, :fingerprint, :source, :sender_name, :conversation_id, :session_id,
                        :segment_id, :sequence_no, :segment_index, :bbox_json, :order_uncertain, :account_scope_id)
                """,
                _message_params(message),
            )
            return cursor.rowcount > 0

    def get(self, message_id: str) -> Message | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _message_from_row(row) if row is not None else None

    def list_for_contact(self, contact_id: str, limit: int = 200) -> list[Message]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE contact_id = ? ORDER BY observed_at DESC LIMIT ?",
                (contact_id, limit),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def list_timeline(self, conversation_id: str, limit: int = 1000) -> list[Message]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY sequence_no ASC, observed_at ASC, id ASC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def list_timeline_tail(self, conversation_id: str, limit: int = 64) -> list[Message]:
        """Return the persisted conversation tail in chronological order.

        A new OCR session cannot use only its in-memory baseline: a scroll or
        relock can expose messages that were already committed by an older
        session. The timeline reconciler uses this bounded tail as a durable
        sequence anchor before it promotes any visible delta.
        """
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY sequence_no DESC, observed_at DESC, id DESC
                LIMIT ?
                """,
                (conversation_id, max(1, limit)),
            ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def sequence_bounds(self, conversation_id: str) -> tuple[int, int]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT MIN(sequence_no) AS minimum, MAX(sequence_no) AS maximum FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None or row["minimum"] is None:
            return (0, 0)
        return (int(row["minimum"]), int(row["maximum"]))

    def delete_for_contact(self, contact_id: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE contact_id = ?", (contact_id,))
            return cursor.rowcount


class MemoryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add_pending(self, contact_id: str, kind: MemoryKind, content: str, confidence: float | None = None) -> Memory:
        """Insert a pending memory, or return the existing row for identical content.

        The memories table carries a unique index on (contact_id, fingerprint), so
        calling this twice with the same content is a no-op rather than a duplicate.
        """
        memory = self._build_pending(contact_id, kind, content, confidence, None, None)
        if self._insert_memory(memory):
            return memory
        existing = self._find_by_fingerprint(contact_id, memory.fingerprint)
        return existing if existing is not None else memory

    def add_or_ignore_pending(
        self,
        contact_id: str,
        kind: MemoryKind,
        content: str,
        *,
        confidence: float | None = None,
        source_message_id: str | None = None,
        expires_at: str | None = None,
    ) -> Memory | None:
        """Insert a pending memory, returning ``None`` when the same content already exists."""
        memory = self._build_pending(contact_id, kind, content, confidence, source_message_id, expires_at)
        return memory if self._insert_memory(memory) else None

    def list_for_contact(self, contact_id: str) -> list[Memory]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE contact_id = ? ORDER BY created_at DESC",
                (contact_id,),
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def list_confirmed_for_contact(self, contact_id: str) -> list[Memory]:
        """Only user-confirmed memories may be sent to the AI provider."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE contact_id = ? AND status = ? ORDER BY created_at DESC",
                (contact_id, MemoryStatus.CONFIRMED.value),
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def count_for_contact(self, contact_id: str) -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM memories WHERE contact_id = ?",
                (contact_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def delete_for_contact(self, contact_id: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE contact_id = ?", (contact_id,))
            return cursor.rowcount

    def list_by_status(self, status: MemoryStatus, limit: int = 200) -> list[Memory]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit),
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def update_status(self, memory_id: str, status: MemoryStatus) -> Memory:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("UPDATE memories SET status = ?, updated_at = ? WHERE id = ?", (status.value, now, memory_id))
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise ValueError(f"memory not found: {memory_id}")
        return _memory_from_row(row)

    def _build_pending(
        self,
        contact_id: str,
        kind: MemoryKind,
        content: str,
        confidence: float | None,
        source_message_id: str | None,
        expires_at: str | None,
    ) -> Memory:
        now = utc_now_iso()
        return Memory(
            id=new_id("memory"),
            contact_id=contact_id,
            kind=kind,
            status=MemoryStatus.PENDING,
            content=content,
            confidence=confidence,
            source_message_id=source_message_id,
            expires_at=expires_at,
            created_at=now,
            updated_at=now,
            fingerprint=memory_fingerprint(contact_id, kind, content),
        )

    def _insert_memory(self, memory: Memory) -> bool:
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memories
                (id, contact_id, kind, status, content, confidence, source_message_id, expires_at, created_at, updated_at, fingerprint)
                VALUES (:id, :contact_id, :kind, :status, :content, :confidence, :source_message_id, :expires_at, :created_at, :updated_at, :fingerprint)
                """,
                _memory_params(memory),
            )
            return cursor.rowcount > 0

    def _find_by_fingerprint(self, contact_id: str, fingerprint: str) -> Memory | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE contact_id = ? AND fingerprint = ?",
                (contact_id, fingerprint),
            ).fetchone()
        return _memory_from_row(row) if row is not None else None


class LogRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(self, level: str, module: str, event: str, message: str, context: dict | None = None) -> AppLog:
        log = AppLog(
            id=new_id("log"),
            ts=utc_now_iso(),
            level=level,
            module=module,
            event=event,
            message=message,
            context_json=json.dumps(context or {}, ensure_ascii=False),
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO app_logs(id, ts, level, module, event, message, context_json)
                VALUES (:id, :ts, :level, :module, :event, :message, :context_json)
                """,
                asdict(log),
            )
        return log

    def tail(self, limit: int = 200) -> list[AppLog]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM app_logs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [AppLog(**dict(row)) for row in rows]

    def delete_older_than(self, cutoff_iso: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM app_logs WHERE ts < ?", (cutoff_iso,))
            return cursor.rowcount

    def enforce_retention(self, cutoff_iso: str, max_rows: int = 20000) -> int:
        """Bound diagnostic history by age and row count.

        This runs at startup, not on the 800 ms window-follow heartbeat.
        """
        deleted = self.delete_older_than(cutoff_iso)
        max_rows = max(1000, int(max_rows))
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM app_logs
                WHERE id IN (
                    SELECT id FROM app_logs ORDER BY ts DESC LIMIT -1 OFFSET ?
                )
                """,
                (max_rows,),
            )
            deleted += cursor.rowcount
        return deleted


class GenerationLogRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(
        self,
        *,
        contact_id: str | None,
        strategy_id: str | None,
        provider: str,
        model: str,
        allowed: bool,
        status: str,
        suggestion_count: int,
        risk_summary: str,
        context_hash: str,
        page_type: str,
        page_confidence: float,
        message_count: int,
        memory_count: int,
    ) -> GenerationLog:
        log = GenerationLog(
            id=new_id("generation"),
            ts=utc_now_iso(),
            contact_id=contact_id,
            strategy_id=strategy_id,
            provider=provider,
            model=model,
            allowed=allowed,
            status=status,
            suggestion_count=suggestion_count,
            risk_summary=risk_summary,
            context_hash=context_hash,
            page_type=page_type,
            page_confidence=page_confidence,
            message_count=message_count,
            memory_count=memory_count,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO generation_logs
                (id, ts, contact_id, strategy_id, provider, model, allowed, status, suggestion_count,
                 risk_summary, context_hash, page_type, page_confidence, message_count, memory_count)
                VALUES
                (:id, :ts, :contact_id, :strategy_id, :provider, :model, :allowed, :status, :suggestion_count,
                 :risk_summary, :context_hash, :page_type, :page_confidence, :message_count, :memory_count)
                """,
                _generation_log_params(log),
            )
        return log

    def tail(self, limit: int = 100) -> list[GenerationLog]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM generation_logs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [_generation_log_from_row(row) for row in rows]

    def list_for_contact(self, contact_id: str, limit: int = 500) -> list[GenerationLog]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM generation_logs WHERE contact_id = ? ORDER BY ts DESC LIMIT ?",
                (contact_id, limit),
            ).fetchall()
        return [_generation_log_from_row(row) for row in rows]

    def latest_for_context(self, context_hash: str) -> GenerationLog | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM generation_logs WHERE context_hash = ? ORDER BY ts DESC LIMIT 1",
                (context_hash,),
            ).fetchone()
        return _generation_log_from_row(row) if row else None

    def count_cloud_attempts_since(self, since_iso: str) -> int:
        # These two exact legacy statuses were emitted by the old transport
        # guard before HTTP. Keep the audit rows, but not the false usage lock.
        # New unknown exceptions use the v2 prefix and can never match them.
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count FROM generation_logs
                WHERE ts >= ?
                  AND provider IN ('OpenAI', 'OpenAI Compatible')
                  AND status NOT LIKE 'blocked:%'
                  AND status NOT LIKE 'ai_cached:%'
                  AND status NOT IN ('attempted:exception:BudgetExhausted', 'attempted:exception:BudgetUnavailable')
                """,
                (since_iso,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def latest_cloud_attempt_for_context(self, context_hash: str) -> GenerationLog | None:
        # Showing a cached result or a pre-dispatch refusal must not move the
        # original dedupe window forward and strand an expired result cache.
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM generation_logs WHERE context_hash = ?
                  AND provider IN ('OpenAI', 'OpenAI Compatible')
                  AND status NOT LIKE 'blocked:%'
                  AND status NOT LIKE 'ai_cached:%'
                  AND status NOT IN ('attempted:exception:BudgetExhausted', 'attempted:exception:BudgetUnavailable')
                ORDER BY ts DESC LIMIT 1
                """, (context_hash,),
            ).fetchone()
        return _generation_log_from_row(row) if row else None

    def delete_for_contact(self, contact_id: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM generation_logs WHERE contact_id = ?", (contact_id,))
            return cursor.rowcount


class ReplyFeedbackRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(
        self,
        *,
        contact_id: str | None,
        strategy_id: str | None,
        provider: str,
        status: str,
        suggestion_label: str,
        suggestion_text: str,
        risk: str,
        feedback: str,
        context_hash: str,
        page_type: str,
        message_count: int,
        memory_count: int,
    ) -> ReplyFeedback:
        record = ReplyFeedback(
            id=new_id("reply_feedback"),
            ts=utc_now_iso(),
            contact_id=contact_id,
            strategy_id=strategy_id,
            provider=provider,
            status=status,
            suggestion_label=suggestion_label[:48],
            suggestion_text_preview=_preview_text(suggestion_text, 96),
            risk=risk[:24],
            feedback=feedback[:24],
            context_hash=context_hash,
            page_type=page_type,
            message_count=message_count,
            memory_count=memory_count,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO reply_feedback
                (id, ts, contact_id, strategy_id, provider, status, suggestion_label, suggestion_text_preview,
                 risk, feedback, context_hash, page_type, message_count, memory_count)
                VALUES
                (:id, :ts, :contact_id, :strategy_id, :provider, :status, :suggestion_label, :suggestion_text_preview,
                 :risk, :feedback, :context_hash, :page_type, :message_count, :memory_count)
                """,
                asdict(record),
            )
        return record

    def tail(self, limit: int = 100) -> list[ReplyFeedback]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM reply_feedback ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [ReplyFeedback(**dict(row)) for row in rows]

    def list_for_contact(self, contact_id: str, limit: int = 500) -> list[ReplyFeedback]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reply_feedback WHERE contact_id = ? ORDER BY ts DESC LIMIT ?",
                (contact_id, limit),
            ).fetchall()
        return [ReplyFeedback(**dict(row)) for row in rows]

    def delete_for_contact(self, contact_id: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM reply_feedback WHERE contact_id = ?", (contact_id,))
            return cursor.rowcount

    def delete_older_than(self, cutoff_iso: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM reply_feedback WHERE ts < ?", (cutoff_iso,))
            return cursor.rowcount


class SettingsAuditRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(self, *, actor: str, scope: str, changes: dict, secret_backend: str) -> SettingsAudit:
        audit = SettingsAudit(
            id=new_id("settings_audit"),
            ts=utc_now_iso(),
            actor=actor,
            scope=scope,
            changes_json=json.dumps(changes, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            secret_backend=secret_backend,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO settings_audit(id, ts, actor, scope, changes_json, secret_backend)
                VALUES (:id, :ts, :actor, :scope, :changes_json, :secret_backend)
                """,
                asdict(audit),
            )
        return audit

    def tail(self, limit: int = 100) -> list[SettingsAudit]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM settings_audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [SettingsAudit(**dict(row)) for row in rows]


class CaptureSampleRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(
        self,
        *,
        job_id: int,
        hwnd: int | None,
        target_app: str,
        app_label: str,
        snapshot_hash: str,
        image_path: str,
        ocr_image_path: str,
        crop_rect_json: str,
        ocr_engine: str,
        ocr_warning: str,
        page_type: str,
        page_confidence: float,
        message_count: int,
        retained_image: bool,
        title_ocr_image_path: str = "",
        title_crop_rect_json: str = "",
        title_ocr_elapsed_ms: int | None = None,
        content_ocr_elapsed_ms: int | None = None,
        total_elapsed_ms: int | None = None,
        capture_elapsed_ms: int | None = None,
        layout_elapsed_ms: int | None = None,
        title_input_elapsed_ms: int | None = None,
        content_input_elapsed_ms: int | None = None,
        title_preprocess_elapsed_ms: int | None = None,
        title_inference_elapsed_ms: int | None = None,
        title_postprocess_elapsed_ms: int | None = None,
        content_preprocess_elapsed_ms: int | None = None,
        content_inference_elapsed_ms: int | None = None,
        content_postprocess_elapsed_ms: int | None = None,
        verify_capture_elapsed_ms: int | None = None,
        page_classification_elapsed_ms: int | None = None,
        title_cache_hit: bool = False,
    ) -> CaptureSample:
        sample = CaptureSample(
            id=new_id("capture"),
            ts=utc_now_iso(),
            job_id=job_id,
            hwnd=hwnd,
            target_app=target_app,
            app_label=app_label,
            snapshot_hash=snapshot_hash,
            image_path=image_path,
            ocr_image_path=ocr_image_path,
            crop_rect_json=crop_rect_json,
            title_ocr_image_path=title_ocr_image_path,
            title_crop_rect_json=title_crop_rect_json,
            title_ocr_elapsed_ms=title_ocr_elapsed_ms,
            content_ocr_elapsed_ms=content_ocr_elapsed_ms,
            total_elapsed_ms=total_elapsed_ms,
            capture_elapsed_ms=capture_elapsed_ms,
            layout_elapsed_ms=layout_elapsed_ms,
            title_input_elapsed_ms=title_input_elapsed_ms,
            content_input_elapsed_ms=content_input_elapsed_ms,
            title_preprocess_elapsed_ms=title_preprocess_elapsed_ms,
            title_inference_elapsed_ms=title_inference_elapsed_ms,
            title_postprocess_elapsed_ms=title_postprocess_elapsed_ms,
            content_preprocess_elapsed_ms=content_preprocess_elapsed_ms,
            content_inference_elapsed_ms=content_inference_elapsed_ms,
            content_postprocess_elapsed_ms=content_postprocess_elapsed_ms,
            verify_capture_elapsed_ms=verify_capture_elapsed_ms,
            page_classification_elapsed_ms=page_classification_elapsed_ms,
            title_cache_hit=title_cache_hit,
            ocr_engine=ocr_engine,
            ocr_warning=ocr_warning,
            page_type=page_type,
            page_confidence=page_confidence,
            message_count=message_count,
            retained_image=retained_image,
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO capture_samples
                (id, ts, job_id, hwnd, target_app, app_label, snapshot_hash, image_path, ocr_image_path, crop_rect_json,
                 title_ocr_image_path, title_crop_rect_json, title_ocr_elapsed_ms, content_ocr_elapsed_ms, total_elapsed_ms,
                 capture_elapsed_ms, layout_elapsed_ms, title_input_elapsed_ms, content_input_elapsed_ms,
                 title_preprocess_elapsed_ms, title_inference_elapsed_ms, title_postprocess_elapsed_ms,
                 content_preprocess_elapsed_ms, content_inference_elapsed_ms, content_postprocess_elapsed_ms,
                 verify_capture_elapsed_ms, page_classification_elapsed_ms, title_cache_hit,
                 ocr_engine, ocr_warning, page_type, page_confidence, message_count, retained_image)
                VALUES
                (:id, :ts, :job_id, :hwnd, :target_app, :app_label, :snapshot_hash, :image_path, :ocr_image_path, :crop_rect_json,
                 :title_ocr_image_path, :title_crop_rect_json, :title_ocr_elapsed_ms, :content_ocr_elapsed_ms, :total_elapsed_ms,
                 :capture_elapsed_ms, :layout_elapsed_ms, :title_input_elapsed_ms, :content_input_elapsed_ms,
                 :title_preprocess_elapsed_ms, :title_inference_elapsed_ms, :title_postprocess_elapsed_ms,
                 :content_preprocess_elapsed_ms, :content_inference_elapsed_ms, :content_postprocess_elapsed_ms,
                 :verify_capture_elapsed_ms, :page_classification_elapsed_ms, :title_cache_hit,
                 :ocr_engine, :ocr_warning, :page_type, :page_confidence, :message_count, :retained_image)
                """,
                _capture_sample_params(sample),
            )
        return sample

    def tail(self, limit: int = 100) -> list[CaptureSample]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM capture_samples ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [_capture_sample_from_row(row) for row in rows]

    def delete_all(self) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM capture_samples")
            return cursor.rowcount


class CalibrationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_active(self, target: TargetApp = TargetApp.WECHAT) -> LayoutCalibration | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM layout_calibrations WHERE target = ? AND active = 1 ORDER BY updated_at DESC LIMIT 1",
                (target.value,),
            ).fetchone()
        return _calibration_from_row(row) if row else None

    def list_all(self, target: TargetApp = TargetApp.WECHAT) -> list[LayoutCalibration]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM layout_calibrations WHERE target = ? ORDER BY active DESC, updated_at DESC",
                (target.value,),
            ).fetchall()
        return [_calibration_from_row(row) for row in rows]

    def save(self, calibration: LayoutCalibration) -> LayoutCalibration:
        updated = replace(calibration, updated_at=utc_now_iso())
        with self.db.connect() as conn:
            if updated.active:
                conn.execute("UPDATE layout_calibrations SET active = 0 WHERE target = ?", (updated.target.value,))
            conn.execute(
                """
                INSERT INTO layout_calibrations
                (id, target, name, theme, dpi_scale, nav_rect_json, chat_list_rect_json, content_rect_json,
                 title_rect_json, message_rect_json, input_rect_json, active, created_at, updated_at)
                VALUES
                (:id, :target, :name, :theme, :dpi_scale, :nav_rect_json, :chat_list_rect_json, :content_rect_json,
                 :title_rect_json, :message_rect_json, :input_rect_json, :active, :created_at, :updated_at)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    theme = excluded.theme,
                    dpi_scale = excluded.dpi_scale,
                    nav_rect_json = excluded.nav_rect_json,
                    chat_list_rect_json = excluded.chat_list_rect_json,
                    content_rect_json = excluded.content_rect_json,
                    title_rect_json = excluded.title_rect_json,
                    message_rect_json = excluded.message_rect_json,
                    input_rect_json = excluded.input_rect_json,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                _calibration_params(updated),
            )
        return updated

    def create_from_layout(
        self,
        *,
        name: str,
        target: TargetApp,
        window_rect,
        layout,
        theme: ThemeMode = ThemeMode.UNKNOWN,
        dpi_scale: float = 1.0,
        active: bool = True,
    ) -> LayoutCalibration:
        now = utc_now_iso()
        calibration = LayoutCalibration(
            id=new_id("calibration"),
            target=target,
            name=name,
            theme=theme,
            dpi_scale=dpi_scale,
            nav_rect=RelativeRect.from_absolute(window_rect, layout.nav_rect),
            chat_list_rect=RelativeRect.from_absolute(window_rect, layout.chat_list_rect),
            content_rect=RelativeRect.from_absolute(window_rect, layout.content_rect),
            title_rect=RelativeRect.from_absolute(window_rect, layout.title_rect),
            message_rect=RelativeRect.from_absolute(window_rect, layout.message_rect),
            input_rect=RelativeRect.from_absolute(window_rect, layout.input_rect),
            active=active,
            created_at=now,
            updated_at=now,
        )
        return self.save(calibration)


def _strategy_from_row(row) -> Strategy:
    data = dict(row)
    data["requires_manual_reply"] = bool(data["requires_manual_reply"])
    data["archived"] = bool(data.get("archived", 0))
    return Strategy(**data)


def _account_scope_from_row(row) -> AccountScope:
    return AccountScope(**dict(row))


def _target_instance_from_row(row) -> TargetInstance:
    data = dict(row)
    data["visual_reliable"] = bool(data["visual_reliable"])
    return TargetInstance(**data)


def _observation_from_row(row) -> ConversationObservation:
    return ConversationObservation(**dict(row))


def _analysis_event_from_row(row) -> AIAnalysisEvent:
    return AIAnalysisEvent(**dict(row))


def _contact_from_row(row) -> Contact:
    data = dict(row)
    data["conversation_type"] = ConversationType(data["conversation_type"])
    data["status"] = ContactStatus(data["status"])
    data["allow_cloud_ai"] = bool(data["allow_cloud_ai"])
    return Contact(**data)


def _person_from_row(row) -> Person:
    data = dict(row)
    data["status"] = IdentityStatus(data["status"])
    return Person(
        id=data["id"],
        display_name=data["display_name"],
        status=data["status"],
        remark=data["remark"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


def _contact_person_link_from_row(row) -> ContactPersonLink:
    data = dict(row)
    data["verified"] = bool(data["verified"])
    return ContactPersonLink(**data)


def _contact_person_link_from_joined_row(row) -> ContactPersonLink:
    return ContactPersonLink(
        id=row["link_id"],
        contact_id=row["link_contact_id"],
        person_id=row["link_person_id"],
        confidence=float(row["link_confidence"]),
        source=row["link_source"],
        verified=bool(row["link_verified"]),
        created_at=row["link_created_at"],
        updated_at=row["link_updated_at"],
    )


def _group_member_from_row(row) -> GroupMember:
    data = dict(row)
    return GroupMember(
        id=data["id"],
        group_contact_id=data["group_contact_id"],
        member_display_name=data["member_display_name"],
        person_id=data["person_id"],
        platform_contact_id=data["platform_contact_id"],
        confidence=float(data["confidence"]),
        source=data["source"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


def _message_from_row(row) -> Message:
    data = dict(row)
    data["speaker"] = Speaker(data["speaker"])
    data["partial"] = bool(data["partial"])
    data["order_uncertain"] = bool(data.get("order_uncertain", 0))
    data.setdefault("sender_name", "")
    return Message(**data)


def _identity_binding_from_row(row) -> ContactIdentityBinding:
    data = dict(row)
    data["conversation_type"] = ConversationType(data["conversation_type"])
    data["verified"] = bool(data["verified"])
    return ContactIdentityBinding(**data)


def _conversation_from_row(row) -> Conversation:
    data = dict(row)
    data["conversation_type"] = ConversationType(data["conversation_type"])
    return Conversation(**data)


def _conversation_segment_from_row(row) -> ConversationSegment:
    data = dict(row)
    data["reliable"] = bool(data["reliable"])
    data["order_uncertain"] = bool(data["order_uncertain"])
    return ConversationSegment(**data)


def _memory_from_row(row) -> Memory:
    data = dict(row)
    data["kind"] = MemoryKind(data["kind"])
    data["status"] = MemoryStatus(data["status"])
    data.setdefault("fingerprint", "")
    return Memory(**data)


def _generation_log_from_row(row) -> GenerationLog:
    data = dict(row)
    data["allowed"] = bool(data["allowed"])
    return GenerationLog(**data)


def _capture_sample_from_row(row) -> CaptureSample:
    data = dict(row)
    data["retained_image"] = bool(data["retained_image"])
    data["title_cache_hit"] = bool(data.get("title_cache_hit", 0))
    return CaptureSample(**data)


def _contact_params(contact: Contact) -> dict:
    data = asdict(contact)
    data["conversation_type"] = contact.conversation_type.value
    data["status"] = contact.status.value
    data["allow_cloud_ai"] = int(contact.allow_cloud_ai)
    return data


def _target_instance_params(instance: TargetInstance) -> dict:
    data = asdict(instance)
    data["visual_reliable"] = int(instance.visual_reliable)
    return data


def _observation_params(observation: ConversationObservation) -> dict:
    return asdict(observation)


def _person_params(person: Person) -> dict:
    data = asdict(person)
    data["status"] = person.status.value
    return data


def _contact_person_link_params(link: ContactPersonLink) -> dict:
    data = asdict(link)
    data["verified"] = int(link.verified)
    return data


def _group_member_params(member: GroupMember) -> dict:
    return asdict(member)


def _message_params(message: Message) -> dict:
    data = asdict(message)
    data["speaker"] = message.speaker.value
    data["partial"] = int(message.partial)
    data["order_uncertain"] = int(message.order_uncertain)
    data["sender_name"] = message.sender_name or ""
    return data


def _identity_binding_params(binding: ContactIdentityBinding) -> dict:
    data = asdict(binding)
    data["conversation_type"] = binding.conversation_type.value
    data["verified"] = int(binding.verified)
    return data


def _conversation_params(conversation: Conversation) -> dict:
    data = asdict(conversation)
    data["conversation_type"] = conversation.conversation_type.value
    return data


def _conversation_segment_params(segment: ConversationSegment) -> dict:
    data = asdict(segment)
    data["reliable"] = int(segment.reliable)
    data["order_uncertain"] = int(segment.order_uncertain)
    return data


def _identity_title(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _normalized_contact_name(value: str) -> str:
    """Comparison key for harmless OCR whitespace variants of the same remark."""
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", normalized).casefold()


def memory_fingerprint(contact_id: str, kind: MemoryKind, content: str) -> str:
    """Stable identity for a memory row: contact + kind + whitespace/punctuation-free text."""
    normalized = re.sub(r"[\s\u3000]+", "", content or "").casefold()
    bucket = f"{contact_id}|{kind.value}|{normalized}"
    return hashlib.sha256(bucket.encode("utf-8")).hexdigest()[:32]


def _memory_params(memory: Memory) -> dict:
    data = asdict(memory)
    data["kind"] = memory.kind.value
    data["status"] = memory.status.value
    return data


def _generation_log_params(log: GenerationLog) -> dict:
    data = asdict(log)
    data["allowed"] = int(log.allowed)
    return data


def _preview_text(value: str, limit: int) -> str:
    text = " ".join(value.strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _capture_sample_params(sample: CaptureSample) -> dict:
    data = asdict(sample)
    data["retained_image"] = int(sample.retained_image)
    data["title_cache_hit"] = int(sample.title_cache_hit)
    return data


def _calibration_from_row(row) -> LayoutCalibration:
    data = dict(row)
    return LayoutCalibration(
        id=data["id"],
        target=TargetApp(data["target"]),
        name=data["name"],
        theme=ThemeMode(data["theme"]),
        dpi_scale=float(data["dpi_scale"]),
        nav_rect=_relative_rect_from_json(data["nav_rect_json"]),
        chat_list_rect=_relative_rect_from_json(data["chat_list_rect_json"]),
        content_rect=_relative_rect_from_json(data["content_rect_json"]),
        title_rect=_relative_rect_from_json(data["title_rect_json"]),
        message_rect=_relative_rect_from_json(data["message_rect_json"]),
        input_rect=_relative_rect_from_json(data["input_rect_json"]),
        active=bool(data["active"]),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


def _calibration_params(calibration: LayoutCalibration) -> dict:
    return {
        "id": calibration.id,
        "target": calibration.target.value,
        "name": calibration.name,
        "theme": calibration.theme.value,
        "dpi_scale": calibration.dpi_scale,
        "nav_rect_json": _relative_rect_to_json(calibration.nav_rect),
        "chat_list_rect_json": _relative_rect_to_json(calibration.chat_list_rect),
        "content_rect_json": _relative_rect_to_json(calibration.content_rect),
        "title_rect_json": _relative_rect_to_json(calibration.title_rect),
        "message_rect_json": _relative_rect_to_json(calibration.message_rect),
        "input_rect_json": _relative_rect_to_json(calibration.input_rect),
        "active": int(calibration.active),
        "created_at": calibration.created_at,
        "updated_at": calibration.updated_at,
    }


def _relative_rect_from_json(value: str) -> RelativeRect:
    data = json.loads(value)
    return RelativeRect(float(data["left"]), float(data["top"]), float(data["right"]), float(data["bottom"])).clamp()


def _relative_rect_to_json(value: RelativeRect) -> str:
    rect = value.clamp()
    return json.dumps(asdict(rect), ensure_ascii=False, separators=(",", ":"))
