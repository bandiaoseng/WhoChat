"""Local operator feedback, deliberately separate from chat facts and AI caches.

No provider, capture, clipboard or main business database dependency. Cross-
customer learning exposes only closed-vocabulary descriptions, never raw cases.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock


APPLICATION_ID = 0x57434642
MAX_TEXT = 2000
MAX_RECENT = 200


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _scene(context):
    if context.current_work_mode in {"reactivation", "next_step"}:
        return context.current_work_mode
    text = " ".join(str(m.text) for m in list(context.effective_messages)[-6:])
    for name, pattern in (("effect", r"效果|用了|用后|反馈"), ("order", r"物流|发货|收货|单号"),
                          ("price", r"多少钱|价格|报价|贵"), ("problem", r"怎么|病|虫|黄叶|不长")):
        if re.search(pattern, text):
            return name
    return "reply"


def _traits(text):
    """Only predefined descriptions may cross customer/account boundaries."""
    traits = set()
    if len(text) <= 100:
        traits.add("表达简短，控制在一小段")
    if "\n" in text.strip():
        traits.add("分成短段，便于微信阅读")
    if len(re.findall(r"[？?]", text)) <= 1:
        traits.add("一次最多追问一个核心问题")
    if re.search(r"[\U0001F300-\U0001FAFF]", text):
        traits.add("可以少量使用自然的表情")
    # These are communication moves, NOT retained factual claims or advice.
    moves = ((r"明白|理解|确实|辛苦|好的|嗯", "先承接客户的话或表示理解"),
             (r"最近|这几天|现在.*(怎样|怎么样|多不多)|目前.*(情况|如何)", "先问近期情况，再展开沟通"),
             (r"效果|用后|用了.*(如何|怎么样)|反馈", "先回访使用效果，再决定下一步"),
             (r"照片|拍.*(看|图)|发.*图", "信息不足时先请客户补充图片"),
             (r"先.*(确认|看看|了解)|确定.*再|了解.*再", "先核实条件，再给具体建议"),
             (r"有需要|方便.*再|不着急|随时|您看", "给客户选择空间，避免催促"))
    traits.update(label for pattern, label in moves if re.search(pattern, text))
    return traits


class FeedbackConflict(ValueError):
    pass


class WritingFeedbackService:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = RLock()
        self._snapshot_key = None
        self._snapshot = (0, [])
        self._learning_cache = {}

    @contextmanager
    def _db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=2)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            app_id = conn.execute("PRAGMA application_id").fetchone()[0]
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if app_id == 0 and version == 0 and not tables:
                conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
                conn.execute("PRAGMA user_version=1")
                conn.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)")
                conn.execute("INSERT INTO meta VALUES(1,0)")
                conn.execute("""CREATE TABLE suggestions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, customer TEXT NOT NULL,
                    conversation TEXT NOT NULL, name TEXT NOT NULL, scene TEXT NOT NULL,
                    mode TEXT NOT NULL, context_hash TEXT NOT NULL, context_text TEXT NOT NULL,
                    original TEXT NOT NULL, final TEXT, active INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL, updated TEXT NOT NULL,
                    label TEXT NOT NULL, position INTEGER NOT NULL, learning_refs TEXT NOT NULL)""")
                conn.execute("CREATE INDEX feedback_active ON suggestions(active,updated)")
            elif app_id != APPLICATION_ID or version != 1:
                raise ValueError("反馈库版本不匹配；未修改原文件")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def capture(self, context, result):
        """Called only for a successfully displayed, source-bound original result."""
        if not result.allowed or not result.suggestions or context is None or context.contact is None:
            return ()
        from whochat.services.reply import context_hash
        contact = context.contact
        view = context.current_message_view
        conversation = view.conversation_id if view else (context.memory_context.conversation_id if context.memory_context else "")
        if not contact.account_scope_id or not contact.id or not conversation:
            return ()
        digest = context_hash(context)
        # Include the immutable model output; same content repainted is one case.
        group = _digest([contact.account_scope_id, contact.id, conversation, digest,
                         [(s.label, s.text) for s in result.suggestions]])
        ids = tuple(_digest([group, i]) for i in range(len(result.suggestions)))
        excerpt = "\n".join(f"{getattr(m.speaker, 'value', m.speaker)}: {m.text}" for m in list(context.effective_messages)[-8:])[-6000:]
        refs = [e["id"] for e in (context.writing_feedback or {}).get("examples", [])]
        with self._lock, self._db() as db:
            for i, (key, suggestion) in enumerate(zip(ids, result.suggestions)):
                db.execute("""INSERT OR IGNORE INTO suggestions
                    (id,owner,customer,conversation,name,scene,mode,context_hash,context_text,
                     original,created,updated,label,position,learning_refs)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (key, contact.account_scope_id, contact.id, conversation, contact.display_name,
                     _scene(context), context.current_work_mode, digest, excerpt, suggestion.text,
                     _now(), _now(), suggestion.label, i, _json(refs)))
            # Unedited history is bounded. Explicit feedback remains until deleted.
            db.execute("""DELETE FROM suggestions WHERE active=0 AND id NOT IN
                (SELECT id FROM suggestions ORDER BY created DESC LIMIT ?)""", (MAX_RECENT,))
        return ids

    def get(self, key):
        with self._lock, self._db() as db:
            row = db.execute("SELECT * FROM suggestions WHERE id=?", (key,)).fetchone()
            if row is None:
                raise ValueError("这条历史建议已不存在")
            return dict(row)

    def recent(self, *, feedback_only=False):
        with self._lock, self._db() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM suggestions " + ("WHERE active=1 " if feedback_only else "") +
                "ORDER BY updated DESC, id LIMIT ?", (MAX_RECENT,))]

    def save(self, key, text, expected_revision):
        text = str(text).replace("\r\n", "\n").strip()
        if not text or len(text) > MAX_TEXT or "\x00" in text:
            raise ValueError(f"请输入 1～{MAX_TEXT} 字的最终话术")
        with self._lock, self._db() as db:
            row = db.execute("SELECT * FROM suggestions WHERE id=?", (key,)).fetchone()
            if row is None or row["revision"] != expected_revision:
                raise FeedbackConflict("该建议已在另一窗口修改，请重新打开后编辑")
            if row["active"] and row["final"] == text:
                return dict(row)
            db.execute("UPDATE suggestions SET final=?,active=1,revision=revision+1,updated=? WHERE id=?",
                       (text, _now(), key))
            db.execute("UPDATE meta SET revision=revision+1 WHERE id=1")
        self._invalidate()
        return self.get(key)

    def delete(self, key, expected_revision):
        """Forget learning; preserve the user's adopted text and original history."""
        with self._lock, self._db() as db:
            row = db.execute("SELECT * FROM suggestions WHERE id=?", (key,)).fetchone()
            if row is None or row["revision"] != expected_revision:
                raise FeedbackConflict("该反馈已变化，请刷新后操作")
            if row["active"]:
                db.execute("UPDATE suggestions SET active=0,revision=revision+1,updated=? WHERE id=?", (_now(), key))
                db.execute("UPDATE meta SET revision=revision+1 WHERE id=1")
        self._invalidate()

    def overlay(self, result, ids):
        if len(ids) != len(result.suggestions):
            return result
        suggestions = list(result.suggestions)
        for i, key in enumerate(ids):
            row = self.get(key)
            if row["final"] is not None:
                suggestions[i] = replace(suggestions[i], text=row["final"])
        return replace(result, suggestions=suggestions)

    def _invalidate(self):
        self._snapshot_key = None
        self._learning_cache.clear()

    def learning_for(self, context):
        if context.contact is None or not self.path.exists():
            return None
        with self._lock:
            stat = self.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp != self._snapshot_key:
                with self._db() as db:
                    revision = db.execute("SELECT revision FROM meta WHERE id=1").fetchone()[0]
                    # Hard bound; each original suggestion has one current vote.
                    rows = [dict(r) for r in db.execute("SELECT * FROM suggestions WHERE active=1 ORDER BY updated DESC LIMIT 1000")]
                self._snapshot = revision, rows
                self._snapshot_key = (self.path.stat().st_mtime_ns, self.path.stat().st_size)
                self._learning_cache.clear()
            revision, rows = self._snapshot
            scene = _scene(context)
            if scene in self._learning_cache:
                return deepcopy(self._learning_cache[scene])
            if not revision:
                return None
            candidates = [r for r in rows if r["scene"] == scene]
            examples = []
            votes = Counter()
            owners = {}
            for row in candidates:
                before, after = _traits(row["original"]), _traits(row["final"])
                preferred = sorted(after)
                added, removed = sorted(after - before), sorted(before - after)
                if not (added or removed):
                    continue
                for trait in added:
                    votes[trait] += 1
                    owners.setdefault(trait, set()).add((row["owner"], row["customer"]))
                if len(examples) < 3:
                    examples.append({"id": row["id"][:16], "prefer": preferred,
                                     "increase": added, "decrease": removed,
                                     "weight": "单条场景样例，不是通用规则"})
            stable = sorted(t for t, n in votes.items() if n >= 3 and len(owners[t]) >= 2)
            payload = {"schema": "whochat.writing_feedback.v1", "revision": revision,
                       "scope": "本机操作者的四个工作微信共用表达与沟通思路；不含客户事实",
                       "scene": scene, "examples": examples, "repeated_preferences": stable,
                       "instruction": "仅在当前场景合适时参考。不得覆盖当前客户事实、安全约束或核验要求；不是模型训练。"}
            self._learning_cache[scene] = payload
            return deepcopy(payload)
