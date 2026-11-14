from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import unicodedata

from whochat.core.models import utc_now_iso
from whochat.storage.database import Database
from whochat.storage.repositories import new_id


CONVERSATION_STYLE_VERSION = "customer-salutation-v1.0-20261010"
_RECENT_DAYS = 180
_MAX_OUTGOING_SAMPLES = 2000
_PLACEHOLDER = re.compile(r"^\s*\[[^\]]{1,20}\]\s*[。.]?$", re.S)
_CORRUPT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_PREAMBLE = r"(?:在的|好的?|嗯嗯?|对了|没事|理解|收到|中|行)"
_ROLE = r"(?:老板娘|老师|师傅|叔叔|大爷|大哥|兄弟|老板|哥|姐|叔|总)"
_SPECIFIC = r"(?:[\u4e00-\u9fff]{1,3}(?:哥|姐|总|老师|师傅))"
_OPENING = re.compile(
    rf"^\s*(?:{_PREAMBLE}\s*[，,。.!！、：:]?\s*)?(?P<value>{_SPECIFIC}|{_ROLE})"
)
_FALSE_SPECIFIC = {
    "我哥", "你哥", "他哥", "她哥", "咱哥", "咱们哥", "我姐", "你姐", "他姐", "她姐",
    "我老板", "你老板", "他老板", "她老板", "这个哥", "那个哥", "这个姐", "那个姐",
}
_NON_NAME_PREFIXES = {
    "这种", "这个", "那个", "这些", "那些", "今年", "现在", "今天", "明天", "昨天",
    "我们", "你们", "他们", "她们", "客户", "对方", "一个", "每个", "所有", "什么",
    "怎么", "如果", "因为", "还是", "就是", "已经", "目前", "之前", "这里", "那里",
}


@dataclass(frozen=True)
class SalutationStat:
    salutation: str
    count: int
    recent_count: int
    current_year_count: int
    last_used_at: str | None
    weighted_score: float
    feedback_candidates: int = 0


@dataclass(frozen=True)
class CustomerConversationStyleProfile:
    account_scope_id: str
    contact_id: str
    preferred_salutation: str
    salutation_confidence: str
    salutation_evidence_count: int
    salutation_recent_count: int
    salutation_current_year_count: int
    outgoing_sample_count: int
    last_salutation_used_at: str | None
    prefers_no_salutation: bool
    manual_preferred_salutation: str
    salutation_stats: tuple[SalutationStat, ...]
    evidence_hash: str
    style_version: str = CONVERSATION_STYLE_VERSION

    @property
    def usable_salutation(self) -> str:
        if self.salutation_confidence not in {"medium", "high"}:
            return ""
        return self.preferred_salutation


class CustomerSalutationResolver:
    """Resolve one customer's salutation from *our* outgoing messages only.

    This service is intentionally absent from the click path.  It runs while
    local Activation Snapshots are built/refreshed and persists a compact,
    account-scoped style profile for later memory-only display.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def get(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
    ) -> CustomerConversationStyleProfile | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM customer_conversation_styles
                WHERE account_scope_id=? AND contact_id=?
                """,
                (account_scope_id, contact_id),
            ).fetchone()
        if row is None:
            return None
        return _profile_from_row(row)

    def resolve_and_store(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        now: datetime | None = None,
    ) -> CustomerConversationStyleProfile:
        now = now or datetime.now().astimezone()
        existing = self.get(account_scope_id=account_scope_id, contact_id=contact_id)
        manual = _clean_manual(existing.manual_preferred_salutation if existing else "")
        messages = self._outgoing_messages(account_scope_id, contact_id)
        feedback = self._feedback_candidates(account_scope_id, contact_id)
        profile = resolve_customer_salutation(
            account_scope_id=account_scope_id,
            contact_id=contact_id,
            outgoing_messages=messages,
            edit_feedback=feedback,
            manual_preferred_salutation=manual,
            now=now,
        )
        if existing is None or not _same_profile(existing, profile):
            self._save(profile)
        return profile

    def set_manual(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        salutation: str,
    ) -> CustomerConversationStyleProfile:
        value = _clean_manual(salutation)
        if salutation.strip() and not value:
            raise ValueError("invalid_manual_salutation")
        current = self.resolve_and_store(
            account_scope_id=account_scope_id,
            contact_id=contact_id,
        )
        if current.manual_preferred_salutation == value:
            return current
        profile = resolve_customer_salutation(
            account_scope_id=account_scope_id,
            contact_id=contact_id,
            outgoing_messages=self._outgoing_messages(account_scope_id, contact_id),
            edit_feedback=self._feedback_candidates(account_scope_id, contact_id),
            manual_preferred_salutation=value,
            now=datetime.now().astimezone(),
        )
        self._save(profile)
        return profile

    def backfill_activation_contacts(self) -> dict[str, int]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT s.account_scope_id,s.contact_id
                FROM activation_snapshots s
                JOIN contacts c ON c.id=s.contact_id
                WHERE s.value_tier IN ('A','B')
                  AND c.account_scope_id=s.account_scope_id
                  AND c.merged_into IS NULL
                ORDER BY s.account_scope_id,s.contact_id
                """
            ).fetchall()
        result = {"total": len(rows), "generated": 0, "failed": 0, "with_salutation": 0, "prefers_none": 0}
        for row in rows:
            try:
                profile = self.resolve_and_store(
                    account_scope_id=str(row["account_scope_id"]),
                    contact_id=str(row["contact_id"]),
                )
            except Exception:
                result["failed"] += 1
                continue
            result["generated"] += 1
            if profile.usable_salutation:
                result["with_salutation"] += 1
            if profile.prefers_no_salutation:
                result["prefers_none"] += 1
        return result

    def _outgoing_messages(self, account_scope_id: str, contact_id: str) -> list[dict[str, str]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT evidence_id,text,used_at,source FROM (
                    SELECT 'live:'||id AS evidence_id,text,observed_at AS used_at,'live' AS source
                    FROM messages
                    WHERE account_scope_id=? AND contact_id=? AND speaker='me'
                    UNION ALL
                    SELECT 'historical:'||historical_message_id AS evidence_id,text,sent_at AS used_at,
                           'historical' AS source
                    FROM historical_chat_messages
                    WHERE account_scope_id=? AND contact_id=? AND sender='me'
                )
                ORDER BY used_at DESC LIMIT ?
                """,
                (
                    account_scope_id,
                    contact_id,
                    account_scope_id,
                    contact_id,
                    _MAX_OUTGOING_SAMPLES,
                ),
            ).fetchall()
        return [
            {
                "evidence_id": str(row["evidence_id"]),
                "text": str(row["text"] or ""),
                "used_at": str(row["used_at"] or ""),
                "source": str(row["source"] or ""),
            }
            for row in rows
        ]

    def _feedback_candidates(self, account_scope_id: str, contact_id: str) -> list[dict[str, str]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT feedback_id,original_text,final_text,updated_at
                FROM suggestion_feedback
                WHERE account_scope_id=? AND contact_id=?
                  AND feedback_type='edited_and_accepted' AND final_text<>''
                ORDER BY updated_at DESC LIMIT 50
                """,
                (account_scope_id, contact_id),
            ).fetchall()
        result: list[dict[str, str]] = []
        for row in rows:
            before = extract_salutation(str(row["original_text"] or ""))
            after = extract_salutation(str(row["final_text"] or ""))
            if after and after != before:
                result.append(
                    {
                        "evidence_id": f"feedback:{row['feedback_id']}",
                        "salutation": after,
                        "used_at": str(row["updated_at"] or ""),
                    }
                )
        return result

    def _save(self, profile: CustomerConversationStyleProfile) -> None:
        now = utc_now_iso()
        payload = json.dumps(
            {"salutations": [asdict(item) for item in profile.salutation_stats]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO customer_conversation_styles(
                    id,account_scope_id,contact_id,preferred_salutation,salutation_confidence,
                    salutation_evidence_count,salutation_recent_count,
                    salutation_current_year_count,outgoing_sample_count,last_salutation_used_at,
                    prefers_no_salutation,manual_preferred_salutation,evidence_json,evidence_hash,
                    style_version,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(account_scope_id,contact_id) DO UPDATE SET
                    preferred_salutation=excluded.preferred_salutation,
                    salutation_confidence=excluded.salutation_confidence,
                    salutation_evidence_count=excluded.salutation_evidence_count,
                    salutation_recent_count=excluded.salutation_recent_count,
                    salutation_current_year_count=excluded.salutation_current_year_count,
                    outgoing_sample_count=excluded.outgoing_sample_count,
                    last_salutation_used_at=excluded.last_salutation_used_at,
                    prefers_no_salutation=excluded.prefers_no_salutation,
                    manual_preferred_salutation=excluded.manual_preferred_salutation,
                    evidence_json=excluded.evidence_json,evidence_hash=excluded.evidence_hash,
                    style_version=excluded.style_version,updated_at=excluded.updated_at
                """,
                (
                    new_id("conversation_style"),
                    profile.account_scope_id,
                    profile.contact_id,
                    profile.preferred_salutation,
                    profile.salutation_confidence,
                    profile.salutation_evidence_count,
                    profile.salutation_recent_count,
                    profile.salutation_current_year_count,
                    profile.outgoing_sample_count,
                    profile.last_salutation_used_at,
                    int(profile.prefers_no_salutation),
                    profile.manual_preferred_salutation,
                    payload,
                    profile.evidence_hash,
                    profile.style_version,
                    now,
                    now,
                ),
            )


def resolve_customer_salutation(
    *,
    account_scope_id: str,
    contact_id: str,
    outgoing_messages: list[dict[str, str]],
    edit_feedback: list[dict[str, str]] | None = None,
    manual_preferred_salutation: str = "",
    now: datetime | None = None,
) -> CustomerConversationStyleProfile:
    now = now or datetime.now().astimezone()
    recent_cutoff = now - timedelta(days=_RECENT_DAYS)
    samples: list[dict[str, str]] = []
    evidence: list[tuple[str, str, str]] = []
    counts: dict[str, int] = defaultdict(int)
    recent: dict[str, int] = defaultdict(int)
    current_year: dict[str, int] = defaultdict(int)
    scores: dict[str, float] = defaultdict(float)
    last_used: dict[str, datetime] = {}
    feedback_counts: dict[str, int] = defaultdict(int)

    for item in outgoing_messages[:_MAX_OUTGOING_SAMPLES]:
        text = _clean_text(item.get("text", ""))
        if not _usable_sample(text):
            continue
        samples.append(item)
        salutation = extract_salutation(text)
        if not salutation:
            continue
        used_at = _parse_dt(item.get("used_at", "")) or datetime.min.replace(tzinfo=timezone.utc)
        local_used = used_at.astimezone(now.tzinfo)
        counts[salutation] += 1
        if local_used >= recent_cutoff:
            recent[salutation] += 1
            weight = 5.0
        elif local_used.year == now.year:
            weight = 3.0
        elif local_used.year == now.year - 1:
            weight = 1.0
        else:
            weight = 0.35
        if local_used.year == now.year:
            current_year[salutation] += 1
        scores[salutation] += weight
        if salutation not in last_used or used_at > last_used[salutation]:
            last_used[salutation] = used_at
        evidence.append((str(item.get("evidence_id") or ""), salutation, str(item.get("used_at") or "")))

    for item in edit_feedback or []:
        salutation = _clean_manual(item.get("salutation", ""))
        if not salutation:
            continue
        feedback_counts[salutation] += 1
        scores[salutation] += 0.35
        evidence.append((str(item.get("evidence_id") or ""), f"feedback:{salutation}", str(item.get("used_at") or "")))

    ranked = sorted(
        scores,
        key=lambda value: (
            scores[value],
            recent[value],
            current_year[value],
            counts[value],
            len(value),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else ""
    total_salutations = sum(counts.values())
    selected_count = counts[selected] if selected else 0
    selected_recent = recent[selected] if selected else 0
    selected_year = current_year[selected] if selected else 0
    ratio = selected_count / max(1, total_salutations)
    recent_total = sum(recent.values())
    recent_ratio = selected_recent / max(1, recent_total)

    manual = _clean_manual(manual_preferred_salutation)
    if manual:
        preferred = manual
        confidence = "high"
    elif (
        selected_count >= 5
        and selected_year >= 3
        and ratio >= 0.55
        and (selected_recent >= 2 or recent_total == 0)
        and (recent_ratio >= 0.55 or recent_total == 0)
    ):
        preferred = selected
        confidence = "high"
    elif selected_count >= 3 and ratio >= 0.45 and (selected_recent >= 2 or selected_year >= 3):
        preferred = selected
        confidence = "medium"
    else:
        preferred = ""
        confidence = "low"

    salutation_density = total_salutations / max(1, len(samples))
    prefers_none = not preferred and len(samples) >= 8 and salutation_density <= 0.08
    if prefers_none:
        confidence = "high" if len(samples) >= 20 and salutation_density <= 0.03 else "medium"

    stats = tuple(
        SalutationStat(
            salutation=value,
            count=counts[value],
            recent_count=recent[value],
            current_year_count=current_year[value],
            last_used_at=last_used[value].isoformat() if value in last_used else None,
            weighted_score=round(scores[value], 3),
            feedback_candidates=feedback_counts[value],
        )
        for value in ranked
    )
    selected_stat = next((item for item in stats if item.salutation == preferred), None)
    digest = hashlib.sha256(
        json.dumps(
            {
                "version": CONVERSATION_STYLE_VERSION,
                "manual": manual,
                "evidence": sorted(evidence),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return CustomerConversationStyleProfile(
        account_scope_id=account_scope_id,
        contact_id=contact_id,
        preferred_salutation=preferred,
        salutation_confidence=confidence,
        salutation_evidence_count=selected_stat.count if selected_stat else 0,
        salutation_recent_count=selected_stat.recent_count if selected_stat else 0,
        salutation_current_year_count=selected_stat.current_year_count if selected_stat else 0,
        outgoing_sample_count=len(samples),
        last_salutation_used_at=selected_stat.last_used_at if selected_stat else None,
        prefers_no_salutation=prefers_none,
        manual_preferred_salutation=manual,
        salutation_stats=stats,
        evidence_hash=digest,
    )


def extract_salutation(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned or _PLACEHOLDER.match(cleaned):
        return ""
    match = _OPENING.match(cleaned)
    if match is None:
        return ""
    value = _clean_manual(match.group("value"))
    if value in _FALSE_SPECIFIC or value.startswith(("我哥", "你哥", "他哥", "她哥", "我姐", "你姐", "他姐", "她姐")):
        return ""
    for role in ("老师", "师傅", "哥", "姐", "总"):
        if value.endswith(role):
            prefix = value[: -len(role)]
            if prefix in _NON_NAME_PREFIXES:
                return ""
            break
    # A multi-character candidate ending in a role must not swallow a normal
    # subject phrase such as “今年哥…”.  Keep specific forms compact.
    if len(value) > 4:
        return ""
    return value


def _profile_from_row(row) -> CustomerConversationStyleProfile:
    try:
        payload = json.loads(str(row["evidence_json"] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    stats: list[SalutationStat] = []
    for item in payload.get("salutations", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            stats.append(SalutationStat(**item))
        except TypeError:
            continue
    return CustomerConversationStyleProfile(
        account_scope_id=str(row["account_scope_id"]),
        contact_id=str(row["contact_id"]),
        preferred_salutation=str(row["preferred_salutation"] or ""),
        salutation_confidence=str(row["salutation_confidence"] or "low"),
        salutation_evidence_count=int(row["salutation_evidence_count"] or 0),
        salutation_recent_count=int(row["salutation_recent_count"] or 0),
        salutation_current_year_count=int(row["salutation_current_year_count"] or 0),
        outgoing_sample_count=int(row["outgoing_sample_count"] or 0),
        last_salutation_used_at=row["last_salutation_used_at"],
        prefers_no_salutation=bool(row["prefers_no_salutation"]),
        manual_preferred_salutation=str(row["manual_preferred_salutation"] or ""),
        salutation_stats=tuple(stats),
        evidence_hash=str(row["evidence_hash"] or ""),
        style_version=str(row["style_version"] or ""),
    )


def _same_profile(
    left: CustomerConversationStyleProfile,
    right: CustomerConversationStyleProfile,
) -> bool:
    """Keep style refresh idempotent so unchanged evidence does not stale snapshots."""

    return (
        left.account_scope_id == right.account_scope_id
        and left.contact_id == right.contact_id
        and left.preferred_salutation == right.preferred_salutation
        and left.salutation_confidence == right.salutation_confidence
        and left.salutation_evidence_count == right.salutation_evidence_count
        and left.salutation_recent_count == right.salutation_recent_count
        and left.salutation_current_year_count == right.salutation_current_year_count
        and left.outgoing_sample_count == right.outgoing_sample_count
        and left.last_salutation_used_at == right.last_salutation_used_at
        and left.prefers_no_salutation == right.prefers_no_salutation
        and left.manual_preferred_salutation == right.manual_preferred_salutation
        and left.evidence_hash == right.evidence_hash
        and left.style_version == right.style_version
    )


def _clean_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _usable_sample(text: str) -> bool:
    if not text or _PLACEHOLDER.match(text) or _CORRUPT.search(text):
        return False
    if text.count("�") >= 2:
        return False
    return any(character.isalnum() or "\u4e00" <= character <= "\u9fff" for character in text)


def _clean_manual(value: str) -> str:
    cleaned = "".join(_clean_text(value).split()).strip("，,。.!！?？、：:")
    if not cleaned or len(cleaned) > 8 or _CORRUPT.search(cleaned):
        return ""
    if cleaned in {"您好", "你好"}:
        return ""
    return cleaned


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
