"""Bounded local customer statements, not confirmed memory or new messages.

Every write requires a freshly DB-verified CurrentMessageView. Historical reads
recheck each retained source through a separate DB-backed callback; expired live
leases do not grant new authority and deleted/changed sources are not reused.
No network, formal-message writes, model calls, or pending-memory promotion.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Callable

from whochat.core.current_conversation import CurrentMessageView
from whochat.core.models import LEGACY_ACCOUNT_SCOPE_ID, Speaker
from whochat.security.redaction import redact_diagnostics_text

SCHEMA = "whochat.observed_history.v1"
SOURCE_SCHEMA = "whochat.observed_history.source.v1"
_APPLICATION_ID = 0x57434831
_TZ = timezone(timedelta(hours=8))
_TRUSTED = {"baseline", "confirmed", "promoted"}
_MAX_ENTRIES = 2048
_MAX_GLOBAL = 50000
_MAX_SOURCES = 8
_MAX_READ_CANDIDATES = 48
_MAX_SOURCE_CHECKS = 64
_STAGES = {
    "幼果期": ("幼果期", "幼果"), "开花期": ("开花期", "花期", "盛花", "初花"),
    "坐果期": ("坐果期", "坐果", "座果"), "膨果期": ("膨果期", "膨果", "膨大期"),
    "转色期": ("转色期", "转色", "着色期"), "采收期": ("采收期", "正在采收"),
    "苗期": ("苗期", "幼苗期"), "缓苗期": ("缓苗期", "缓苗"),
    "萌芽期": ("萌芽期", "萌芽"), "成熟期": ("成熟期",),
}
_CROPS = ("草莓", "蓝莓", "葡萄", "柑橘", "砂糖橘", "柚子", "杨梅", "苹果", "桃树", "桃子",
          "西瓜", "甜瓜", "黄瓜", "辣椒", "番茄", "西红柿", "玉米", "水稻", "小麦", "大豆")
_NONCURRENT = re.compile(r"[?？]|吗|么|是否|哪个|什么|几时|请问|要不要|不|没|未|并非|"
                         r"去年|上季|之前|以前|曾经|原来|那时|昨天|前天|已经过了|下周|明天|后天|将要|准备|打算|预计|快到|马上|等到|"
                         r"如果|假如|要是|可能|应该|大概|好像|听说|他说|她说|别人|朋友|邻居|他家|她家|他的|她的|假设|例如|比如|快|过几天|过两天|要到|就到|建议|应当|一般|通常|最好")
_SELF_CURRENT = re.compile(r"我|咱|现在|目前|已经|都是|进入|到了|了[。！!\s]*$")
_PRIVATE = re.compile(r"(?:收货|寄货|配送|家庭)?地址|身份证|银行卡|收件人|收货人|密码|验证码|微信号|账号|账户|(?i:api.?key|authorization|bearer)")
_QUANTITY = re.compile(r"\d+(?:\.\d+)?\s*(?:毫升|毫克|公斤|千克|克|斤|两|升|桶|瓶|袋|倍|亩|喷雾器|ml|mL|kg|g|L)")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def message_evidence_hash(message) -> str:
    return _digest(message.business_payload())


def sanitize_excerpt(text: str) -> str:
    """Keep bounded business text only; never retain addresses or secrets."""
    if not isinstance(text, str) or not text.strip() or len(text) > 1000 or _PRIVATE.search(text):
        return ""
    # Preserve statement boundaries; stripping all whitespace can join clauses.
    text = re.sub(r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9])", "[已脱敏:邮箱]", text)
    return redact_diagnostics_text(text).strip()


def _time(value: str) -> datetime | None:
    try:
        result = datetime.fromisoformat(value)
        return result if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def _now(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("history_now_timezone_required")
    return value


def _empty(status: str, warnings=()) -> dict:
    payload = {"schema": SCHEMA, "status": status, "facts": [], "excerpts": [], "warnings": list(warnings)}
    payload["revision"] = _digest(payload)
    return payload


def _message_identity(view, message, excerpt):
    return _digest([[view.account_scope_id, view.contact_id, view.conversation_id],
                    message.speaker.value, excerpt, message.message_time, message.time_source, message.time_text])


class ObservedHistoryService:
    def __init__(self, path: str | Path, *, verify_view: Callable, verify_source: Callable):
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("history_path_must_be_absolute")
        if not callable(verify_view) or not callable(verify_source):
            raise TypeError("history_db_verifiers_required")
        self.verify_view = verify_view
        self.verify_source = verify_source

    def _valid_view(self, view, now: datetime) -> bool:
        if (type(view) is not CurrentMessageView or view.source_kind != "visible_observation"
                or (view.observation_status not in _TRUSTED and not (
                    view.observation_status == "quarantined" and view.screen_evidence_json)) or not view.account_scope_id
                or view.account_scope_id == LEGACY_ACCOUNT_SCOPE_ID or not view.contact_id
                or not view.conversation_id or not view.messages):
            return False
        if not view.validate_binding(
                expected_account_scope_id=view.account_scope_id, expected_contact_id=view.contact_id,
                expected_conversation_id=view.conversation_id, expected_target_instance_id=view.target_instance_id,
                expected_session_id=view.session_id, max_source_job_id=view.source_job_id,
                max_capture_generation=view.capture_generation, now=now):
            return False
        try:
            return self.verify_view(view) is True
        except Exception:
            return False

    def _write_connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=1)
        try:
            application = conn.execute("PRAGMA application_id").fetchone()[0]
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if application not in {0, _APPLICATION_ID} or (application == 0 and tables):
                raise ValueError("history_database_identity_mismatch")
            conn.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evidence(
                  id TEXT PRIMARY KEY, account_id TEXT NOT NULL, contact_id TEXT NOT NULL,
                  conversation_id TEXT NOT NULL, excerpt TEXT NOT NULL, message_time TEXT,
                  time_source TEXT NOT NULL, time_text TEXT NOT NULL, first_observed_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS evidence_owner ON evidence(account_id,contact_id,conversation_id);
                CREATE TABLE IF NOT EXISTS sources(
                  evidence_id TEXT NOT NULL, source_ref TEXT NOT NULL, ordinal INTEGER NOT NULL,
                  payload TEXT NOT NULL, PRIMARY KEY(evidence_id,source_ref,ordinal));
            """)
            return conn
        except Exception:
            conn.close()
            raise

    def record(self, view, *, now: datetime | None = None) -> dict:
        now = _now(now)
        if not self._valid_view(view, now):
            return {"status": "unverified", "stored": 0, "duplicates": 0}
        candidates = []
        for message in view.messages:
            if (message.speaker != Speaker.OTHER or message.content_type != "text"
                    or message.partial or message.order_uncertain):
                continue
            excerpt = sanitize_excerpt(message.text)
            if not excerpt:
                continue
            identity = _message_identity(view, message, excerpt)
            source = dict(schema=SOURCE_SCHEMA, account_scope_id=view.account_scope_id,
                          contact_id=view.contact_id, conversation_id=view.conversation_id,
                          target_instance_id=view.target_instance_id, session_id=view.session_id,
                          source_ref=view.observation_id, content_revision=view.content_revision,
                          reply_business_revision=view.reply_business_revision,
                          original_observation_status=view.observation_status, source_kind=view.source_kind,
                          source_reason=view.source_reason, source_job_id=view.source_job_id,
                          capture_generation=view.capture_generation,
                          source_message_ordinal=message.ordinal,
                          source_message_sha256=message_evidence_hash(message), excerpt=excerpt,
                          message_time=message.message_time, time_source=message.time_source,
                          time_text=message.time_text, observed_at=view.observed_at)
            candidates.append((identity, message, excerpt, source))
        if not candidates:
            return {"status": "no_usable_customer_text", "stored": 0, "duplicates": 0}
        stored = duplicate = omitted = 0
        conn = self._write_connection()
        try:
            with conn:
                for identity, message, excerpt, source in candidates:
                    exists = conn.execute("SELECT 1 FROM evidence WHERE id=?", (identity,)).fetchone()
                    if exists:
                        duplicate += 1
                    else:
                        owner_count = conn.execute("SELECT count(*) FROM evidence WHERE account_id=? AND contact_id=? AND conversation_id=?",
                                                   (view.account_scope_id, view.contact_id, view.conversation_id)).fetchone()[0]
                        if owner_count >= _MAX_ENTRIES or conn.execute("SELECT count(*) FROM evidence").fetchone()[0] >= _MAX_GLOBAL:
                            omitted += 1
                            continue
                        conn.execute("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?)",
                                     (identity, view.account_scope_id, view.contact_id, view.conversation_id,
                                      excerpt, message.message_time, message.time_source, message.time_text, view.observed_at))
                        stored += 1
                    count = conn.execute("SELECT count(*) FROM sources WHERE evidence_id=?", (identity,)).fetchone()[0]
                    if count < _MAX_SOURCES:
                        conn.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?,?)",
                                     (identity, view.observation_id, message.ordinal, _json(source)))
        finally:
            conn.close()
        return {"status": "capacity_exceeded" if omitted else "recorded", "stored": stored, "duplicates": duplicate, "omitted": omitted}

    def clear_contact(self, account_scope_id: str, contact_id: str) -> int:
        if not account_scope_id or not contact_id:
            raise ValueError("history_clear_owner_required")
        return self._clear((account_scope_id, contact_id))

    def clear_all(self) -> int:
        return self._clear(None)

    def _clear(self, owner) -> int:
        if not self.path.exists():
            return 0
        conn = sqlite3.connect(self.path, timeout=1)
        try:
            if conn.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID:
                raise ValueError("history_database_identity_mismatch")
            conn.execute("PRAGMA secure_delete=ON")
            with conn:
                if owner is None:
                    count = conn.execute("SELECT count(*) FROM evidence").fetchone()[0]
                    conn.execute("DELETE FROM sources")
                    conn.execute("DELETE FROM evidence")
                else:
                    conn.execute("DELETE FROM sources WHERE evidence_id IN(SELECT id FROM evidence WHERE account_id=? AND contact_id=?)", owner)
                    count = conn.execute("DELETE FROM evidence WHERE account_id=? AND contact_id=?", owner).rowcount
            return count
        finally:
            conn.close()

    def build(self, view, *, now: datetime | None = None) -> dict:
        now = _now(now)
        if not self._valid_view(view, now):
            return _empty("unverified")
        if not self.path.exists():
            return _empty("empty")
        try:
            conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=1)
            conn.row_factory = sqlite3.Row
            try:
                if conn.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID:
                    return _empty("unavailable", ["history_database_identity_mismatch"])
                # Bound source verification independently of store capacity.
                # Explicit stage statements outrank routine short acknowledgments.
                stage_terms = [term for variants in _STAGES.values() for term in variants]
                priority = " OR ".join("instr(excerpt,?)>0" for _ in stage_terms)
                rows = conn.execute("SELECT * FROM evidence WHERE account_id=? AND contact_id=? AND conversation_id=? "
                                    "AND ((message_time IS NOT NULL AND julianday(message_time)>=julianday(?)) "
                                    "OR (message_time IS NULL AND julianday(first_observed_at)>=julianday(?))) "
                                    f"ORDER BY CASE WHEN {priority} THEN 0 ELSE 1 END,first_observed_at DESC,id LIMIT ?",
                                    (view.account_scope_id, view.contact_id, view.conversation_id,
                                     (now-timedelta(days=7)).isoformat(),
                                     now.astimezone(_TZ).replace(hour=0,minute=0,second=0,microsecond=0).isoformat(),
                                     *stage_terms, _MAX_READ_CANDIDATES)).fetchall()
                usable = []
                warnings = set()
                current_ids = {_message_identity(view, m, sanitize_excerpt(m.text)) for m in view.messages}
                source_checks = 0
                for row in rows:
                    if row["id"] in current_ids:
                        continue
                    observed = _time(row["first_observed_at"])
                    sent = _time(row["message_time"]) if row["time_source"] in {"ocr", "verified", "platform", "source"} else None
                    if observed is None or observed > now + timedelta(minutes=1):
                        continue
                    if sent is not None:
                        if sent > observed + timedelta(minutes=1) or sent > now or now - sent > timedelta(days=7):
                            continue
                        time_status = "same_day" if sent.astimezone(_TZ).date() == now.astimezone(_TZ).date() else "historical"
                    elif observed.astimezone(_TZ).date() == now.astimezone(_TZ).date():
                        time_status = "date_unconfirmed"
                    else:
                        continue
                    valid_source = None
                    for item in conn.execute("SELECT payload FROM sources WHERE evidence_id=? ORDER BY rowid LIMIT ?", (row["id"], _MAX_SOURCES)):
                        if source_checks >= _MAX_SOURCE_CHECKS:
                            warnings.add("history_source_check_limit")
                            break
                        try:
                            payload = json.loads(item[0])
                            if (payload.get("schema") != SOURCE_SCHEMA or payload.get("excerpt") != row["excerpt"]
                                    or [payload.get(k) for k in ("account_scope_id", "contact_id", "conversation_id")]
                                    != [view.account_scope_id, view.contact_id, view.conversation_id]
                                    or any(payload.get(k) != row[k] for k in ("message_time", "time_source", "time_text"))
                                    or _time(payload.get("observed_at")) is None
                                    or observed > _time(payload["observed_at"])):
                                continue
                            source_checks += 1
                            if self.verify_source(payload) is True:
                                valid_source = payload
                                break
                        except Exception:
                            continue
                    if valid_source is None:
                        warnings.add("history_source_unavailable")
                        continue
                    usable.append({"text": row["excerpt"], "message_time": row["message_time"],
                                   "time_text": row["time_text"], "time_status": time_status,
                                   "first_observed_at": row["first_observed_at"],
                                   "source_ref": valid_source["source_ref"],
                                   "source_ordinal": valid_source["source_message_ordinal"],
                                   "source_kind": "verified_customer_statement", "human_confirmed": False,
                                   "contains_quantity": bool(_QUANTITY.search(row["excerpt"]))})
            finally:
                conn.close()
        except (sqlite3.Error, ValueError, OSError):
            return _empty("unavailable", ["history_read_failed"])
        facts = []
        stages = []
        for item in usable:
            text = item["text"]
            found = [name for name, variants in _STAGES.items() if any(part in text for part in variants)]
            crops = [crop for crop in _CROPS if crop in text]
            if found:
                if _NONCURRENT.search(text) or not _SELF_CURRENT.search(text) or len(found) != 1 or len(crops) > 1:
                    item["stage_qualification"] = "not_a_clear_current_statement"
                    warnings.add("stage_requires_context")
                elif item["time_status"] == "same_day":
                    stages.append((item, found[0], crops[0] if crops else ""))
                else:
                    item["stage_qualification"] = "historical_stage_not_current"
                    warnings.add("stage_date_unconfirmed_or_old")
        stage_values = {value for _, value, _ in stages}
        disputed_current_stage = False
        for message in view.messages:
            if message.speaker == Speaker.OTHER:
                mentioned = {name for name, variants in _STAGES.items() if any(part in message.text for part in variants)}
                if mentioned and _NONCURRENT.search(message.text):
                    disputed_current_stage = True
                else:
                    stage_values.update(mentioned)
        crop_values = {crop for item in usable for crop in _CROPS if crop in item["text"]}
        crop_values.update(crop for message in view.messages for crop in _CROPS if crop in message.text)
        if disputed_current_stage or len(stage_values) > 1 or len(crop_values) > 1:
            warnings.add("conflicting_or_multiple_crop_stages")
        elif stages:
            item, value, crop = stages[0]
            facts.append({"key": "growth_stage", "value": value, "crop": crop,
                          "status": "customer_statement_not_human_confirmed", "message_time": item["message_time"],
                          "source_ref": item["source_ref"], "source_ordinal": item["source_ordinal"],
                          "evidence": item["text"], "usable_as_current": True})
        # Stage evidence must survive a long later exchange; remainder is recent
        # customer business text. Nothing is projected into confirmed memory.
        usable.sort(key=lambda item: (not any(v in item["text"] for values in _STAGES.values() for v in values),
                                      -(_time(item["message_time"] or item["first_observed_at"]).timestamp())))
        selected = []
        characters = 0
        for item in usable:
            if len(selected) >= 12:
                break
            if characters + len(item["text"]) > 2400:
                continue
            selected.append(item)
            characters += len(item["text"])
        payload = {"schema": SCHEMA, "status": "available" if usable else "empty",
                   "facts": facts, "excerpts": selected, "warnings": sorted(warnings)}
        payload["revision"] = _digest(payload)
        return payload
