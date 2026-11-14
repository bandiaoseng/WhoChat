"""Short-lived completed Reply results, separate from the business database.

This is not an authorization, incoming-message event or request-budget store.
The caller must validate the *fresh* source and live identity lease before and
after recovery. Only hashes of ownership/configuration and a strict completed
result are saved; no source context, API key or provider-stage timestamps are
persisted. Results themselves may contain the normal customer summary/history.

TTL is a logical validity limit, not a promise of physical erasure while the
application is closed. Expired rows are removed on subsequent cache operations.
Unknown/corrupt files are never migrated, replaced, deleted or treated as a
reason to retry HTTP. All cache failures are conservative misses/no-ops.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
from threading import RLock
import time

from whochat.ai.current_work_authorization import cloud_reply_authorized, current_context_source_error
from whochat.ai.models import (
    INGREDIENT_REPLY_POLICY_VERSION, ReplyContext, ReplyGenerationResult,
    reply_generation_result_from_json, structured_reply_result_error,
)
from whochat.config import AppConfig
from whochat.core.models import ContactStatus, ConversationType, LEGACY_ACCOUNT_SCOPE_ID


_APPLICATION_ID = 0x57435243  # WCRC; never WhoChat's main database.
_SCHEMA_VERSION = 1
_SCHEMA_MARKER = "whochat.reply_result_cache.v1"
_MAX_TTL_SECONDS = 24 * 60 * 60
_MAX_RESULT_BYTES = 256_000
_MAX_ROWS = 512
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TABLE_COLUMNS = {
    "reply_cache_meta": ("key", "value"),
    "reply_results": (
        "owner_hash", "context_digest", "profile_hash", "policy_version",
        "created_at", "expires_at", "result_json", "payload_sha256",
    ),
}


class ReplyResultCache:
    """Lazy, fail-closed cache restricted to ``cache/reply_results.sqlite3``.

    Construction performs no filesystem IO, so damaged cache data cannot stop
    WhoChat startup. Each operation rechecks the dedicated application/schema
    markers before any SQL write. Connections are short-lived and serialized
    within this instance; no worker, timer, retry or automatic cleanup job is
    created here.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._clock = time.time

    def get(
        self, context_digest: str, config: AppConfig, context: ReplyContext,
    ) -> ReplyGenerationResult | None:
        try:
            identity = _cache_identity(context_digest, config, context)
            ttl = _ttl_seconds(config)
            if identity is None or ttl <= 0:
                return None
            owner_hash, profile_hash = identity
            now = self._now()
            with self._lock:
                connection = self._open_owned(create=False)
                if connection is None:
                    return None
                with closing(connection), connection:
                    self._purge_expired(connection, now)
                    row = connection.execute(
                        "SELECT policy_version,created_at,expires_at,result_json,payload_sha256 "
                        "FROM reply_results WHERE owner_hash=? AND context_digest=? AND profile_hash=?",
                        (owner_hash, context_digest, profile_hash),
                    ).fetchone()
                    if row is None:
                        return None
                    version, created, expires, payload, payload_hash = row
                    if (
                        version != INGREDIENT_REPLY_POLICY_VERSION
                        or not isinstance(created, (int, float)) or not math.isfinite(created)
                        or not isinstance(expires, (int, float)) or not math.isfinite(expires)
                        or created > now or expires <= now or expires <= created
                        or expires - created > _MAX_TTL_SECONDS
                        or now - created >= ttl
                        or not isinstance(payload, str)
                        or len(payload.encode("utf-8")) > _MAX_RESULT_BYTES
                        or _sha256(payload) != payload_hash
                    ):
                        return None
                    result, error = reply_generation_result_from_json(payload, context=context)
                    if error or result is None or result.provider != config.ai.provider:
                        return None
                    return _cache_result(result, profile_hash, restored=True)
        except Exception:
            # The service retains its original dedupe/rate/budget decision.
            # In particular an IO failure does not authorize another request.
            return None

    def put(
        self, context_digest: str, config: AppConfig, context: ReplyContext,
        result: ReplyGenerationResult,
    ) -> None:
        try:
            identity = _cache_identity(context_digest, config, context)
            ttl = _ttl_seconds(config)
            if (
                identity is None or ttl <= 0 or result.allowed is not True
                or result.provider != config.ai.provider
                or structured_reply_result_error(result, context=context)
            ):
                return
            owner_hash, profile_hash = identity
            clean_result = _cache_result(result, profile_hash, restored=False)
            payload = json.dumps(asdict(clean_result), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len(payload.encode("utf-8")) > _MAX_RESULT_BYTES:
                return
            # Use the durable reader too: e.g. a malformed suggestion label
            # must not enter storage just because the in-memory check missed it.
            restored, error = reply_generation_result_from_json(payload, context=context)
            if error or restored is None:
                return
            now = self._now()
            with self._lock:
                connection = self._open_owned(create=True)
                if connection is None:
                    return
                with closing(connection), connection:
                    self._purge_expired(connection, now)
                    existing = connection.execute(
                        "SELECT 1 FROM reply_results WHERE owner_hash=? AND context_digest=? AND profile_hash=?",
                        (owner_hash, context_digest, profile_hash),
                    ).fetchone()
                    count = connection.execute("SELECT COUNT(*) FROM reply_results").fetchone()[0]
                    if existing is None and count >= _MAX_ROWS:
                        return  # Bounded cache; never evict another live row.
                    connection.execute(
                        "INSERT INTO reply_results "
                        "(owner_hash,context_digest,profile_hash,policy_version,created_at,expires_at,result_json,payload_sha256) "
                        "VALUES (?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(owner_hash,context_digest,profile_hash) DO UPDATE SET "
                        "policy_version=excluded.policy_version,created_at=excluded.created_at,"
                        "expires_at=excluded.expires_at,result_json=excluded.result_json,payload_sha256=excluded.payload_sha256",
                        (owner_hash, context_digest, profile_hash, INGREDIENT_REPLY_POLICY_VERSION,
                         now, now + ttl, payload, _sha256(payload)),
                    )
        except Exception:
            return

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now) or now < 0:
            raise ValueError("invalid_cache_clock")
        return now

    def _checked_path(self) -> Path | None:
        path = self.path
        if (
            not path.is_absolute() or path.name != "reply_results.sqlite3"
            or path.parent.name != "cache" or str(path.anchor).startswith("\\\\")
            or path.resolve(strict=False) != path.absolute()
        ):
            return None
        # Resolve rejects redirected parents; the explicit Windows reparse bit
        # also covers junctions, not just ordinary Python-recognized symlinks.
        for part in (path, *path.parents):
            if not part.exists():
                continue
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                return None
        if path.exists():
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                return None
        return path

    def _open_owned(self, *, create: bool) -> sqlite3.Connection | None:
        path = self._checked_path()
        if path is None:
            return None
        created = False
        if not path.exists():
            if not create:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._checked_path() is None:
                return None
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
                created = True
        if not created:
            # immutable=1 prevents even journal recovery or sidecar writes to
            # an unrelated database before ownership has been established.
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=.2)) as check:
                if not self._is_owned(check):
                    return None
        connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=.2)
        try:
            if created:
                self._initialize(connection)
            if not self._is_owned(connection):
                connection.close()
                return None
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            connection.execute("CREATE TABLE reply_cache_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            connection.execute("INSERT INTO reply_cache_meta VALUES ('schema',?)", (_SCHEMA_MARKER,))
            connection.execute(
                "CREATE TABLE reply_results ("
                "owner_hash TEXT NOT NULL,context_digest TEXT NOT NULL,profile_hash TEXT NOT NULL,"
                "policy_version TEXT NOT NULL,created_at REAL NOT NULL,expires_at REAL NOT NULL,"
                "result_json TEXT NOT NULL,payload_sha256 TEXT NOT NULL,"
                "PRIMARY KEY(owner_hash,context_digest,profile_hash))"
            )
            connection.execute("CREATE INDEX reply_results_expiry ON reply_results(expires_at)")

    @staticmethod
    def _is_owned(connection: sqlite3.Connection) -> bool:
        if (
            connection.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
            or connection.execute("PRAGMA user_version").fetchone()[0] != _SCHEMA_VERSION
        ):
            return False
        objects = connection.execute(
            "SELECT name,type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if set(objects) != {
            ("reply_cache_meta", "table"), ("reply_results", "table"), ("reply_results_expiry", "index"),
        }:
            return False
        for table, expected in _TABLE_COLUMNS.items():
            actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if actual != expected:
                return False
        return connection.execute("SELECT key,value FROM reply_cache_meta").fetchall() == [("schema", _SCHEMA_MARKER)]

    @staticmethod
    def _purge_expired(connection: sqlite3.Connection, now: float) -> None:
        connection.execute("DELETE FROM reply_results WHERE expires_at<=?", (now,))


def _ttl_seconds(config: AppConfig) -> float:
    minutes = float(config.ai.dedupe_context_minutes)
    return min(_MAX_TTL_SECONDS, max(0.0, minutes * 60)) if math.isfinite(minutes) else 0.0


def _cache_identity(
    context_digest: str, config: AppConfig, context: ReplyContext,
) -> tuple[str, str] | None:
    if (
        not isinstance(context_digest, str) or not _DIGEST.fullmatch(context_digest)
        or config.ai.provider not in {"OpenAI", "OpenAI Compatible"} or not config.ai.api_key
        or current_context_source_error(context) or not cloud_reply_authorized(context)
    ):
        return None
    contact = context.contact
    if (
        contact is None or not contact.id or not contact.account_scope_id
        or contact.account_scope_id == LEGACY_ACCOUNT_SCOPE_ID
        or contact.platform.casefold() != "wechat" or contact.conversation_type != ConversationType.DM
        or contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED} or contact.merged_into
    ):
        return None
    view = context.current_message_view
    if view is not None:
        conversation_id, source_kind = view.conversation_id, view.source_kind
    else:
        current = context.effective_current_message
        if current is None:
            return None
        conversation_id = current.conversation_id or (
            context.memory_context.conversation_id if context.memory_context is not None else ""
        )
        source_kind = "formal_messages"
        if any(
            item.contact_id != contact.id or item.account_scope_id != contact.account_scope_id
            or (item.conversation_id and item.conversation_id != conversation_id)
            for item in context.messages
        ):
            return None
    if not conversation_id:
        return None
    owner = {"account_scope_id": contact.account_scope_id, "contact_id": contact.id,
             "conversation_id": conversation_id, "source_kind": source_kind}
    return _hash_payload(owner), reply_cache_profile_hash(config)


def reply_cache_profile_hash(config: AppConfig) -> str:
    """One shared hashed generation profile for memory and durable caches."""
    profile = {
        "schema": _SCHEMA_MARKER, "policy_version": INGREDIENT_REPLY_POLICY_VERSION,
        "provider": config.ai.provider, "base_url": config.ai.base_url.strip().rstrip("/"),
        "model": config.ai.model.strip(), "api_key": config.ai.api_key,
        "temperature": config.ai.temperature, "context_tokens": config.ai.context_tokens,
        "trim_context_for_cloud": config.privacy.trim_context_for_cloud,
        "require_cloud_prompt_review": config.privacy.require_cloud_prompt_review,
        "manual_protection_blocks_replies": config.privacy.manual_protection_blocks_replies,
    }
    return _hash_payload(profile)


def _cache_result(result: ReplyGenerationResult, profile_hash: str, *, restored: bool) -> ReplyGenerationResult:
    return replace(result, status="ai_cached:durable_result", performance_trace={
        "prompt_profile": {"policy_version": INGREDIENT_REPLY_POLICY_VERSION},
        "cache_profile": {"schema": _SCHEMA_MARKER, "profile_hash": profile_hash, "restored": restored},
    })


def _hash_payload(payload: object) -> str:
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
