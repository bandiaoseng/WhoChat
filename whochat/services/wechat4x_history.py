from __future__ import annotations

import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator

from whochat.services.historical_import import history_source_hash, ParsedHistoricalMessage


CHINA_TZ = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class WeChat4xConversation:
    mapping_key: str
    external_contact_id: str
    external_conversation_id: str
    display_name: str
    source_database: str
    source_table: str
    is_group: bool


@dataclass(frozen=True)
class WeChat4xSnapshotManifest:
    account_alias: str
    stable_identifier: str
    decrypted_directory: str
    database_count: int
    database_size_bytes: int
    conversation_count: int
    contact_count: int
    message_count: int
    start_time: str | None
    end_time: str | None
    message_type_counts: dict[str, int]
    sender_counts: dict[str, int]
    snapshot_hash: str


class WeChat4xHistoryAdapter:
    """Read-only adapter for a previously decrypted WeChat 4.x snapshot.

    The adapter never opens a live WeChat process and never writes the source
    databases.  It translates stable internal usernames/table identities into
    WhoChat's historical archive format.  Display names are metadata only and
    are never used as account or contact identity.
    """

    adapter_version = "wechat4x-raw-v1"

    def __init__(self, snapshot_directory: Path, *, stable_identifier: str) -> None:
        self.snapshot_directory = snapshot_directory.resolve()
        self.stable_identifier = stable_identifier.strip()
        if not self.snapshot_directory.is_dir():
            raise ValueError("wechat_snapshot_directory_not_found")
        if not self.stable_identifier:
            raise ValueError("wechat_snapshot_stable_identifier_required")
        self.contact_db = self.snapshot_directory / "contact" / "contact.db"
        self.message_directory = self.snapshot_directory / "message"
        if not self.contact_db.is_file() or not self.message_directory.is_dir():
            raise ValueError("wechat_snapshot_layout_invalid")
        self._contacts = self._load_contacts()
        self.account_alias = self._verify_self_identity()

    def conversations(self) -> tuple[WeChat4xConversation, ...]:
        result: list[WeChat4xConversation] = []
        for shard in self._message_shards():
            with closing(_read_only(shard)) as connection:
                result.extend(self._shard_conversations(connection, shard.name))
        return tuple(sorted(result, key=lambda item: (item.source_database, item.mapping_key)))

    def iter_messages(self) -> Iterator[dict[str, Any]]:
        sequence = 0
        for shard in self._message_shards():
            with closing(_read_only(shard)) as connection:
                names = {
                    int(row["rowid"]): _as_text(row["user_name"])
                    for row in connection.execute("SELECT rowid,user_name FROM Name2Id")
                }
                for conversation in self._shard_conversations(connection, shard.name):
                    query = f"""SELECT local_id,server_id,local_type,sort_seq,real_sender_id,
                                        create_time,status,message_content
                                 FROM [{conversation.source_table}]
                                 ORDER BY create_time,sort_seq,local_id"""
                    for row in connection.execute(query):
                        local_type = int(row["local_type"] or 0)
                        sender_id = int(row["real_sender_id"] or 0)
                        sender_username = names.get(sender_id, "")
                        sender = _sender(
                            sender_username=sender_username,
                            stable_identifier=self.stable_identifier,
                            conversation=conversation,
                            local_type=local_type,
                        )
                        message_type = _message_type(local_type)
                        raw_content = _as_text(row["message_content"])
                        text = _display_text(raw_content, message_type)
                        sent_at = _timestamp(int(row["create_time"] or 0))
                        server_id = _as_text(row["server_id"])
                        external_id = _external_message_id(
                            conversation.external_conversation_id,
                            conversation.source_database,
                            int(row["local_id"] or 0),
                            server_id,
                            int(row["sort_seq"] or 0),
                            int(row["create_time"] or 0),
                            local_type,
                        )
                        parsed = ParsedHistoricalMessage(
                            mapping_key=conversation.mapping_key,
                            external_contact_id=conversation.external_contact_id,
                            external_conversation_id=conversation.external_conversation_id,
                            source_display_name=conversation.display_name,
                            external_message_id=external_id,
                            sender=sender,
                            message_type=message_type,
                            text=text,
                            sent_at=sent_at,
                            source_sequence=sequence,
                            raw_metadata={
                                "adapter": self.adapter_version,
                                "source_database": conversation.source_database,
                                "source_table_hash": conversation.source_table.removeprefix("Msg_"),
                                "local_id": int(row["local_id"] or 0),
                                "server_id_present": bool(server_id and server_id != "0"),
                                "sort_seq": int(row["sort_seq"] or 0),
                                "local_type": local_type,
                                "status": int(row["status"] or 0),
                                "sender_identifier": sender_username,
                                "is_group": conversation.is_group,
                            },
                        )
                        yield {**asdict(parsed), "source_hash": history_source_hash(parsed)}
                        sequence += 1

    def build_manifest(self) -> WeChat4xSnapshotManifest:
        conversations = self.conversations()
        unique_conversations = {item.mapping_key for item in conversations}
        contacts = {item.external_contact_id for item in conversations}
        types: Counter[str] = Counter()
        senders: Counter[str] = Counter()
        start: str | None = None
        end: str | None = None
        count = 0
        for message in self.iter_messages():
            count += 1
            types[str(message["message_type"])] += 1
            senders[str(message["sender"])] += 1
            sent_at = str(message["sent_at"])
            start = sent_at if start is None or sent_at < start else start
            end = sent_at if end is None or sent_at > end else end
        databases = sorted(self.snapshot_directory.rglob("*.db"))
        size = sum(path.stat().st_size for path in databases)
        return WeChat4xSnapshotManifest(
            account_alias=self.account_alias,
            stable_identifier=self.stable_identifier,
            decrypted_directory=str(self.snapshot_directory),
            database_count=len(databases),
            database_size_bytes=size,
            conversation_count=len(unique_conversations),
            contact_count=len(contacts),
            message_count=count,
            start_time=start,
            end_time=end,
            message_type_counts=dict(sorted(types.items())),
            sender_counts=dict(sorted(senders.items())),
            snapshot_hash=self._snapshot_hash(databases),
        )

    def write_normalized_jsonl(self, destination: Path) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        count = 0
        digest = hashlib.sha256()
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for message in self.iter_messages():
                line = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))
                count += 1
        temporary.replace(destination)
        return {"path": str(destination), "message_count": count, "sha256": digest.hexdigest(),
                "size_bytes": destination.stat().st_size}

    def _load_contacts(self) -> dict[str, dict[str, str]]:
        with closing(_read_only(self.contact_db)) as connection:
            rows = connection.execute(
                "SELECT username,alias,remark,nick_name FROM contact"
            ).fetchall()
        return {
            _as_text(row["username"]): {
                "alias": _as_text(row["alias"]),
                "remark": _as_text(row["remark"]),
                "nick_name": _as_text(row["nick_name"]),
            }
            for row in rows if _as_text(row["username"])
        }

    def _verify_self_identity(self) -> str:
        self_contact = self._contacts.get(self.stable_identifier)
        if self_contact is None:
            raise ValueError("wechat_snapshot_self_contact_not_found")
        alias = _as_text(self_contact.get("alias"))
        if not alias:
            raise ValueError("wechat_snapshot_self_alias_missing")
        return alias

    def _message_shards(self) -> tuple[Path, ...]:
        shards = tuple(sorted(self.message_directory.glob("message_[0-9]*.db")))
        if not shards:
            raise ValueError("wechat_snapshot_message_shards_missing")
        return shards

    def _shard_conversations(
        self, connection: sqlite3.Connection, database_name: str
    ) -> tuple[WeChat4xConversation, ...]:
        result: list[WeChat4xConversation] = []
        sessions = connection.execute(
            "SELECT user_name FROM Name2Id WHERE is_session=1 ORDER BY rowid"
        ).fetchall()
        for row in sessions:
            username = _as_text(row["user_name"])
            if not username:
                continue
            table = "Msg_" + hashlib.md5(username.encode("utf-8")).hexdigest()
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is None:
                continue
            contact = self._contacts.get(username, {})
            display = (
                _as_text(contact.get("remark")) or _as_text(contact.get("nick_name"))
                or _as_text(contact.get("alias")) or username
            )
            result.append(WeChat4xConversation(
                mapping_key=username, external_contact_id=username,
                external_conversation_id=username, display_name=display,
                source_database=database_name, source_table=table,
                is_group=username.endswith("@chatroom"),
            ))
        return tuple(result)

    def _snapshot_hash(self, databases: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in databases:
            relative = path.relative_to(self.snapshot_directory).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()


def _read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _timestamp(value: int) -> str:
    if value <= 0:
        return datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()
    return datetime.fromtimestamp(value, tz=CHINA_TZ).isoformat()


def _sender(
    *, sender_username: str, stable_identifier: str,
    conversation: WeChat4xConversation, local_type: int,
) -> str:
    if (local_type & 0xFFFFFFFF) == 10000:
        return "system"
    if sender_username == stable_identifier:
        return "me"
    if sender_username:
        return "other"
    # Never infer a missing sender in a group. For direct chats the raw schema
    # normally stores the self/contact rowid; missing identity stays unknown.
    return "unknown"


def _message_type(local_type: int) -> str:
    base = local_type & 0xFFFFFFFF
    subtype = local_type >> 32
    if base == 1:
        return "text"
    if base == 3:
        return "image"
    if base == 34:
        return "voice"
    if base == 43:
        return "video"
    if base == 47:
        return "sticker"
    if base == 49:
        return {5: "link", 6: "file", 19: "forwarded_chat", 51: "channels_video"}.get(subtype, "app_message")
    if base == 10000:
        return "system"
    return f"wechat_type_{local_type}"


def _display_text(content: str, message_type: str) -> str:
    if message_type in {"text", "system"}:
        return content
    labels = {
        "image": "[图片]", "voice": "[语音]", "video": "[视频]", "sticker": "[表情]",
        "file": "[文件]", "link": "[链接]", "forwarded_chat": "[合并转发]",
        "channels_video": "[视频号]", "app_message": "[应用消息]",
    }
    label = labels.get(message_type, f"[{message_type}]")
    if message_type in {"file", "link", "forwarded_chat", "channels_video", "app_message"}:
        title = _xml_title(content)
        return f"{label} {title}".strip()
    return label


def _xml_title(content: str) -> str:
    if not content or "<" not in content:
        return ""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return ""
    for path in (".//title", ".//appmsg/title", ".//des"):
        node = root.find(path)
        if node is not None and node.text and node.text.strip():
            return node.text.strip()[:500]
    return ""


def _external_message_id(
    conversation_id: str,
    database: str,
    local_id: int,
    server_id: str,
    sort_seq: int,
    create_time: int,
    local_type: int,
) -> str:
    identity = {
        "conversation_id": conversation_id,
        "database": database,
        "local_id": local_id,
        "server_id": server_id if server_id and server_id != "0" else "",
        "sort_seq": sort_seq,
        "create_time": create_time,
        "local_type": local_type,
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
