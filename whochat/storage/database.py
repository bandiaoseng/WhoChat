from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Callable

from whochat.core.paths import database_path
from whochat.storage.migrations import MIGRATIONS


_ACTIVATION_MUTATION_TABLES = (
    "activation_snapshots",
    "contacts",
    "conversation_observations",
    "customer_conversation_styles",
    "customer_events",
    "customer_facts",
    "customer_followups",
    "customer_growing_contexts",
    "customer_profiles",
    "memories",
    "messages",
    "suggestion_feedback",
)
_ACTIVATION_MUTATION_RE = re.compile(
    r"\b(?:insert(?:\s+or\s+\w+)?\s+into|update|delete\s+from)\s+"
    r"[\[\`\"]?(" + "|".join(_ACTIVATION_MUTATION_TABLES) + r")\b",
    re.IGNORECASE,
)
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivationInvalidationNotice:
    account_scope_id: str
    contact_id: str
    reason: str
    invalidated_at: str

    @property
    def token(self) -> str:
        return f"db:{self.invalidated_at}:{self.reason}"


class _WhoChatConnection(sqlite3.Connection):
    """Publish trigger-driven activation invalidation only after commit."""

    _whochat_database: "Database | None" = None
    _activation_write_seen: bool = False

    def bind_database(self, database: "Database") -> None:
        self._whochat_database = database
        self._activation_write_seen = False
        self.set_trace_callback(self._trace_statement)

    def _trace_statement(self, sql: str) -> None:
        if _ACTIVATION_MUTATION_RE.search(sql or ""):
            self._activation_write_seen = True

    def _take_activation_write(self) -> bool:
        dirty = self._activation_write_seen
        self._activation_write_seen = False
        return dirty

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        dirty = self._take_activation_write()
        result = super().__exit__(exc_type, exc_value, traceback)
        if exc_type is None and dirty and self._whochat_database is not None:
            self._whochat_database._after_activation_source_commit(self)
        return bool(result)

    def commit(self) -> None:
        dirty = self._take_activation_write()
        super().commit()
        if dirty and self._whochat_database is not None:
            self._whochat_database._after_activation_source_commit(self)

    def rollback(self) -> None:
        self._activation_write_seen = False
        super().rollback()


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or database_path()
        self._activation_lock = RLock()
        self._activation_invalidation_listeners: list[
            Callable[[ActivationInvalidationNotice], None]
        ] = []
        self._activation_invalidated_signatures: dict[tuple[str, str], tuple[str, str]] = {}

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, factory=_WhoChatConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.bind_database(self)
        return conn

    def connect_readonly(self) -> sqlite3.Connection:
        """Open an existing database without write-path initialization."""
        conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            return conn
        except BaseException:
            conn.close()
            raise

    def register_activation_invalidation_listener(
        self,
        listener: Callable[[ActivationInvalidationNotice], None],
    ) -> None:
        """Subscribe to committed trigger invalidations and prime existing state."""

        self._refresh_activation_invalidation_state(notify=False)
        with self._activation_lock:
            if listener not in self._activation_invalidation_listeners:
                self._activation_invalidation_listeners.append(listener)

    def unregister_activation_invalidation_listener(
        self,
        listener: Callable[[ActivationInvalidationNotice], None],
    ) -> None:
        with self._activation_lock:
            self._activation_invalidation_listeners = [
                item for item in self._activation_invalidation_listeners if item != listener
            ]

    def _after_activation_source_commit(self, conn: sqlite3.Connection) -> None:
        self._refresh_activation_invalidation_state(conn=conn, notify=True)

    def _refresh_activation_invalidation_state(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        notify: bool,
    ) -> None:
        owned = conn is None
        current_conn = conn or self.connect()
        try:
            rows = current_conn.execute(
                """
                SELECT account_scope_id,contact_id,invalidation_reason,invalidated_at
                FROM activation_snapshots
                WHERE invalidated_at IS NOT NULL
                """
            ).fetchall()
        except sqlite3.OperationalError:
            # Migrations may connect before the activation table exists.
            return
        finally:
            if owned:
                current_conn.close()

        current = {
            (str(row["account_scope_id"]), str(row["contact_id"])): (
                str(row["invalidated_at"]),
                str(row["invalidation_reason"] or "unknown"),
            )
            for row in rows
        }
        notices: list[ActivationInvalidationNotice] = []
        with self._activation_lock:
            previous = self._activation_invalidated_signatures
            if notify:
                for key, signature in current.items():
                    if previous.get(key) == signature:
                        continue
                    notices.append(
                        ActivationInvalidationNotice(
                            key[0], key[1], signature[1], signature[0]
                        )
                    )
            self._activation_invalidated_signatures = current
            listeners = tuple(self._activation_invalidation_listeners)
        for notice in notices:
            for listener in listeners:
                try:
                    listener(notice)
                except Exception:
                    # The business transaction is already committed.  A faulty
                    # listener must never turn that success into a rollback.
                    _LOG.exception("activation invalidation listener failed")

    def migrate(self) -> None:
        validate_migrations(MIGRATIONS)
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
            for version, sql in MIGRATIONS:
                if version in applied:
                    continue
                conn.executescript(sql)
                conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))


def validate_migrations(migrations: list[tuple[int, str]]) -> None:
    versions = [version for version, _sql in migrations]
    if not versions:
        raise ValueError("database migrations are empty")
    invalid = [version for version in versions if not isinstance(version, int) or version <= 0]
    if invalid:
        raise ValueError(f"database migration versions must be positive integers: {invalid}")
    duplicates = sorted({version for version in versions if versions.count(version) > 1})
    if duplicates:
        raise ValueError(f"duplicate database migration versions: {duplicates}")
    expected = list(range(1, max(versions) + 1))
    if versions != expected:
        missing = [version for version in expected if version not in versions]
        raise ValueError(f"database migrations must be ordered and contiguous; missing={missing} actual={versions}")
