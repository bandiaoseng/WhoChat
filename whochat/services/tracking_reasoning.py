from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Any, Iterable

from whochat.services.crop_display import KNOWN_CROPS
from whochat.services.memory_extraction import _CROPS as MEMORY_EXTRACTION_CROPS


TRACKING_REASONING_VERSION = "existing-customer-tracking-v2.0-20261009"


@dataclass(frozen=True)
class ActiveCropAssessment:
    crop: str
    confidence: str
    score: float
    runner_up: str
    runner_up_score: float
    ambiguous: bool
    evidence: tuple[str, ...]
    evidence_hash: str
    candidate_scores: tuple[tuple[str, float], ...]
    reasoning_version: str = TRACKING_REASONING_VERSION


@dataclass(frozen=True)
class CropSeasonTimeline:
    crop: str
    season_key: str
    stage_label: str
    status: str
    confidence: str
    planted_at: str | None
    estimated_age_days: int | None
    next_window: str
    source: str
    evidence: tuple[str, ...]
    evidence_hash: str
    reasoning_version: str = TRACKING_REASONING_VERSION


_CANONICAL_CROP_ALIASES = {
    "甘薯": "红薯",
    "地瓜": "红薯",
    "番薯": "红薯",
    "土豆": "马铃薯",
    "番茄": "西红柿",
}

# Cultivar knowledge is crop-level knowledge, not a customer-specific rule.
# These aliases let structured variety facts contribute to crop continuity.
_CULTIVAR_CROP_SIGNALS = {
    "西瓜红": "红薯",
    "济薯": "红薯",
    "贵妃紫": "红薯",
    "红瑶": "红薯",
    "丝滑": "红薯",
    "烟薯": "红薯",
    "普薯": "红薯",
    "龙薯": "红薯",
}

_CROP_TERMS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            **{
                crop: _CANONICAL_CROP_ALIASES.get(crop, crop)
                for crop in (*KNOWN_CROPS, *MEMORY_EXTRACTION_CROPS)
            },
            **_CANONICAL_CROP_ALIASES,
            **_CULTIVAR_CROP_SIGNALS,
        }.items(),
        key=lambda item: (-len(item[0]), item[0]),
    )
)

_CURRENT_CROP_CUES = (
    "今年主要",
    "今年种",
    "今年在种",
    "现在主要",
    "当前主要",
    "改种",
    "换种",
    "开始种",
    "开始起垄",
    "播种",
    "定植",
    "移栽",
    "苗",
)
_EXPLICIT_ACTIVE_CROP_CUES = (
    "今年主要",
    "现在主要",
    "当前主要",
    "今年改种",
    "现在改种",
    "已经改种",
    "换种",
    "新一茬改",
)
_CONTINUITY_CUES = (
    "起垄",
    "覆膜",
    "滴灌",
    "苗期",
    "控旺",
    "开花",
    "坐果",
    "膨大",
    "薯型",
    "采收",
    "收获",
    "采完",
)
_PLANTING_ACTION_RE = re.compile(
    r"(?:(?:又重新|重新|新一茬|开始|准备|正在)(?:起垄|播种|栽种|栽了|种植|定植|移栽)"
    r"|起垄|播种|栽种|栽了|定植|移栽)"
)
_AGE_RE = re.compile(r"(?<!\d)(\d{1,3})\s*天(?:左右|上下|了)?")
_MONTH_PLANT_RE = re.compile(
    r"(?<!\d)(1[0-2]|0?[1-9])\s*月(?:份)?(?:左右|前后)?[^。！？]{0,12}"
    r"(?:开始|准备|正在)?(?:起垄|播种|栽种|种植|定植|移栽)"
)


class ActiveCropResolver:
    """Score current-season crop continuity from bounded structured evidence."""

    def resolve(
        self,
        *,
        evidence_payload: dict[str, Any],
        current_year: int,
        remark_text: str,
        contexts: Iterable[Any] = (),
        recent_customer_texts: Iterable[str] = (),
    ) -> ActiveCropAssessment:
        scores: dict[str, float] = {}
        evidence_by_crop: dict[str, list[tuple[float, str]]] = {}
        months_by_crop: dict[str, set[str]] = {}
        explicit_override: tuple[str, str] | None = None

        def add(crop: str, points: float, label: str, occurred_at: str = "") -> None:
            crop = _canonical_crop(crop)
            if not crop or points <= 0:
                return
            scores[crop] = scores.get(crop, 0.0) + points
            evidence_by_crop.setdefault(crop, []).append((points, label[:140]))
            if occurred_at:
                months_by_crop.setdefault(crop, set()).add(occurred_at[:7])

        for text in recent_customer_texts:
            explicit_crops = _explicit_active_crops(text)
            crops = explicit_crops or _extract_crops(text)
            if not crops:
                continue
            for crop in crops:
                if _contains_any(text, _EXPLICIT_ACTIVE_CROP_CUES):
                    points = 120.0
                    explicit_override = (crop, f"最近客户明确当前主作物：{crop}")
                elif _contains_any(text, _CURRENT_CROP_CUES):
                    points = 24.0
                else:
                    points = 9.0
                add(crop, points, f"最近客户表述：{_compact(text)[:90]}", f"{current_year}-12")

        for context in contexts:
            status = str(getattr(context, "status", "") or "")
            confirmed_at = str(getattr(context, "last_confirmed_at", "") or "")
            year_factor = _year_factor(confirmed_at, current_year)
            points = (12.0 if status == "active" else 5.0) * year_factor
            for crop in _extract_crops(str(getattr(context, "crop", "") or "")):
                add(crop, points, f"{status or 'profile'} Growing Context：{crop}", confirmed_at)

        for kind in ("facts", "events"):
            for row in evidence_payload.get(kind, ()):
                if not isinstance(row, dict):
                    continue
                occurred_at = str(row.get("occurred_at") or "")
                year_factor = _year_factor(occurred_at, current_year)
                if year_factor <= 0:
                    continue
                text = _evidence_text(row)
                explicit_crops = _explicit_active_crops(text)
                crops = explicit_crops or _extract_crops(text)
                if not crops:
                    continue
                category = str(row.get("category") or "")
                key = str(row.get("evidence_key") or "")
                confidence = max(0.35, min(float(row.get("confidence") or 0.5), 1.0))
                base = 5.0
                if kind == "facts" and _contains_any(f"{category} {key}", ("crop", "variety", "品种", "作物", "seedling")):
                    base += 5.0
                if _contains_any(text, _CURRENT_CROP_CUES):
                    base += 4.0
                if _contains_any(text, _EXPLICIT_ACTIVE_CROP_CUES):
                    base += 70.0
                    if explicit_override is None:
                        explicit_override = (crops[0], f"结构化证据明确当前主作物：{crops[0]}")
                if _contains_any(text, _CONTINUITY_CUES):
                    base += 2.0
                if len(crops) > 1:
                    base *= 0.72
                points = base * confidence * year_factor
                label = f"{occurred_at[:10] or '未标日期'} {kind[:-1]}：{_best_evidence_text(row)[:90]}"
                for crop in crops:
                    add(crop, points, label, occurred_at)

        # A WeChat remark is a useful fallback, but it cannot beat a sustained
        # current-season structured evidence chain on its own.
        for crop in _extract_crops(remark_text):
            add(crop, 1.5, f"微信备注低权重候选：{crop}")

        for crop, months in months_by_crop.items():
            if len(months) >= 2:
                bonus = min(8.0, float(len(months) * 2))
                scores[crop] = scores.get(crop, 0.0) + bonus
                evidence_by_crop.setdefault(crop, []).append(
                    (bonus, f"当季连续证据覆盖 {len(months)} 个月")
                )

        if explicit_override is not None:
            selected, label = explicit_override
            other_max = max((value for crop, value in scores.items() if crop != selected), default=0.0)
            scores[selected] = max(scores.get(selected, 0.0), other_max + 20.0)
            evidence_by_crop.setdefault(selected, []).append((200.0, label))

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            return ActiveCropAssessment("", "low", 0.0, "", 0.0, False, (), _stable_hash([]), ())
        crop, score = ranked[0]
        runner, runner_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
        gap = score - runner_score
        ambiguous = bool(
            explicit_override is None
            and runner_score > 0
            and gap < max(4.0, score * 0.22)
        )
        confidence = "high" if score >= 24 and gap >= 8 else "medium" if score >= 12 and gap >= 4 else "low"
        selected_evidence = tuple(
            label
            for _, label in sorted(evidence_by_crop.get(crop, ()), key=lambda item: -item[0])[:6]
        )
        candidate_scores = tuple((name, round(value, 3)) for name, value in ranked[:5])
        return ActiveCropAssessment(
            "" if ambiguous else crop,
            confidence,
            round(score, 3),
            runner,
            round(runner_score, 3),
            ambiguous,
            selected_evidence,
            _stable_hash({"crop": crop, "evidence": selected_evidence, "scores": candidate_scores}),
            candidate_scores,
        )


class CropSeasonTimelineService:
    """Build a non-persistent current-season timeline for one resolved crop."""

    def resolve(
        self,
        *,
        crop: str,
        crop_confidence: str,
        evidence_payload: dict[str, Any],
        now: datetime,
        region: str = "",
    ) -> CropSeasonTimeline:
        if not crop:
            return self._unknown(crop, (), "active_crop_unresolved")
        crop = _canonical_crop(crop)
        rows = [
            row
            for kind in ("facts", "events")
            for row in evidence_payload.get(kind, ())
            if isinstance(row, dict) and _row_relevant_to_crop(row, crop, now.year)
        ]
        rows.sort(key=lambda row: str(row.get("occurred_at") or ""), reverse=True)

        explicit_current = self._explicit_current_stage(rows, now)
        if explicit_current is not None:
            label, evidence = explicit_current
            return self._result(
                crop=crop,
                season_key=f"{now.year}当季",
                stage_label=label,
                status="confirmed",
                confidence="high",
                planted_at=None,
                estimated_age_days=None,
                next_window=_window_for_stage(label, crop),
                source="customer_explicit_current_stage",
                evidence=evidence,
            )

        explicit_harvest = self._explicit_harvest(rows, now)
        if explicit_harvest is not None:
            label, status, source, evidence = explicit_harvest
            return self._result(
                crop=crop,
                season_key=f"{now.year}当季",
                stage_label=label,
                status=status,
                confidence="high",
                planted_at=None,
                estimated_age_days=None,
                next_window=_window_for_stage(label, crop),
                source=source,
                evidence=evidence,
            )

        anchors: list[tuple[datetime, float, str]] = []
        for row in rows:
            occurred = _parse_dt(str(row.get("occurred_at") or ""))
            if occurred is None or occurred.year != now.year:
                continue
            text = _evidence_text(row)
            month_match = _MONTH_PLANT_RE.search(text)
            if month_match:
                month = int(month_match.group(1))
                anchor = datetime(now.year, month, 15, tzinfo=now.tzinfo)
                anchors.append((anchor, 0.9, f"客户明确 {now.year} 年 {month} 月种植时间"))
            for match in _AGE_RE.finditer(text):
                age = int(match.group(1))
                if 5 <= age <= 330:
                    anchor = occurred.astimezone(now.tzinfo) - timedelta(days=age)
                    anchors.append((anchor, 1.0, f"{occurred.date()} 客户明确作物约 {age} 天"))
            action = _PLANTING_ACTION_RE.search(text)
            if action:
                offset = 10 if "起垄" in action.group(0) else 0
                anchor = occurred.astimezone(now.tzinfo) + timedelta(days=offset)
                boundary = _contains_any(action.group(0), ("重新", "新一茬"))
                anchors.append(
                    (
                        anchor,
                        1.0 if boundary else 0.68,
                        f"{occurred.date()} 明确{action.group(0)}",
                    )
                )

        if not anchors:
            dated_rows = [
                (_parse_dt(str(row.get("occurred_at") or "")), row)
                for row in rows
            ]
            dated_rows = [
                (date.astimezone(now.tzinfo), row)
                for date, row in dated_rows
                if date is not None and date.year == now.year
            ]
            months = {date.strftime("%Y-%m") for date, _ in dated_rows}
            if len(dated_rows) >= 3 and len(months) >= 2:
                first_date = min(date for date, _ in dated_rows)
                anchors.append(
                    (
                        first_date,
                        0.45,
                        f"当季结构化沟通从 {first_date.date()} 起并连续覆盖 {len(months)} 个月",
                    )
                )

        if anchors:
            anchors.sort(key=lambda item: (-item[1], -item[0].timestamp()))
            planted_at, anchor_confidence, anchor_evidence = anchors[0]
            age_days = max(0, (now.date() - planted_at.date()).days)
            stage = _stage_from_crop_age(crop, age_days)
            if stage:
                label, next_window = stage
                confidence = "high" if anchor_confidence >= 0.95 and crop_confidence == "high" else "medium"
                season_key = f"{planted_at.year}{_season_name(planted_at.month)}"
                evidence = [anchor_evidence]
                evidence.extend(_timeline_evidence(rows, crop, limit=4))
                if region:
                    evidence.append(f"地区：{region}（仅作农时边界，不覆盖客户证据）")
                return self._result(
                    crop=crop,
                    season_key=season_key,
                    stage_label=label,
                    status="inferred",
                    confidence=confidence,
                    planted_at=planted_at.date().isoformat(),
                    estimated_age_days=age_days,
                    next_window=next_window,
                    source="customer_timing_timeline",
                    evidence=evidence,
                )

        # When exact timing is absent, repeated current-year stage evidence can
        # still support a conservative stage family.  It never becomes a fact.
        stage_cues = _latest_stage_cue(rows, now)
        if stage_cues is not None:
            label, evidence = stage_cues
            return self._result(
                crop=crop,
                season_key=f"{now.year}当季",
                stage_label=label,
                status="inferred",
                confidence="medium" if crop_confidence != "low" else "low",
                planted_at=None,
                estimated_age_days=None,
                next_window=_window_for_stage(label, crop),
                source="structured_season_continuity",
                evidence=evidence,
            )
        return self._unknown(crop, _timeline_evidence(rows, crop, limit=4), "season_timeline_insufficient")

    @staticmethod
    def _explicit_current_stage(
        rows: list[dict[str, Any]],
        now: datetime,
    ) -> tuple[str, tuple[str, ...]] | None:
        rules = (
            (re.compile(r"(?:刚栽|刚移栽|栽了\s*\d{1,3}\s*天|移栽了\s*\d{1,3}\s*天)"), "缓苗期"),
            (re.compile(r"(?:现在|目前|正在|已经).{0,6}(?:开花|花期)"), "花期"),
            (re.compile(r"(?:现在|目前|正在|已经).{0,6}(?:坐果|幼果)"), "坐果/幼果期"),
            (re.compile(r"(?:现在|目前|正在|已经).{0,6}(?:膨大|膨果)"), "膨大期"),
        )
        for row in rows:
            occurred = _parse_dt(str(row.get("occurred_at") or ""))
            if occurred is None or occurred.year != now.year:
                continue
            if (now.date() - occurred.astimezone(now.tzinfo).date()).days > 45:
                continue
            text = _evidence_text(row)
            for pattern, label in rules:
                match = pattern.search(text)
                if match:
                    return label, (f"{occurred.date()} 客户明确：{match.group(0)[:80]}",)
        return None

    @staticmethod
    def _explicit_harvest(
        rows: list[dict[str, Any]],
        now: datetime,
    ) -> tuple[str, str, str, tuple[str, ...]] | None:
        for row in rows:
            occurred = _parse_dt(str(row.get("occurred_at") or ""))
            if occurred is None or occurred.year != now.year:
                continue
            text = _evidence_text(row)
            if _contains_any(text, ("已经采完", "采收完", "已经收完", "收完了")):
                return "采后阶段", "confirmed", "customer_explicit_harvest", (
                    f"{occurred.date()} 客户明确：{_best_evidence_text(row)[:90]}",
                )
            if _contains_any(text, ("开始采收", "正在采收", "开始收获", "正在收获")):
                return "采收期", "confirmed", "customer_explicit_harvest", (
                    f"{occurred.date()} 客户明确：{_best_evidence_text(row)[:90]}",
                )
        return None

    @staticmethod
    def _result(**values: Any) -> CropSeasonTimeline:
        evidence = tuple(dict.fromkeys(str(item) for item in values.pop("evidence") if item))
        return CropSeasonTimeline(
            evidence=evidence,
            evidence_hash=_stable_hash(evidence),
            **values,
        )

    @staticmethod
    def _unknown(crop: str, evidence: Iterable[str], source: str) -> CropSeasonTimeline:
        items = tuple(evidence)
        return CropSeasonTimeline(
            crop,
            "",
            "当前阶段待确认",
            "unknown",
            "low",
            None,
            None,
            "未来7～20天先确认当前实际管理节点",
            source,
            items,
            _stable_hash(items),
        )


def _extract_crops(text: str) -> tuple[str, ...]:
    remaining = _compact(text)
    found: list[str] = []
    cultivar_hit = False
    for term, crop in _CROP_TERMS:
        if len(term) == 1 and remaining != term:
            # Single-character crop names (桃、梨、枣...) are also common in
            # personal names.  Only accept them as an exact structured value;
            # substring matches are unsafe for automatic identity reasoning.
            continue
        if term and term in remaining:
            canonical = _canonical_crop(crop)
            if canonical not in found:
                found.append(canonical)
            if term in _CULTIVAR_CROP_SIGNALS:
                cultivar_hit = True
            remaining = remaining.replace(term, " ")
    # A cultivar such as 西瓜红 must not be double-counted as 西瓜.
    if cultivar_hit and "红薯" in found and "西瓜" in found and "西瓜" not in text.replace("西瓜红", ""):
        found.remove("西瓜")
    return tuple(found)


def _explicit_active_crops(text: str) -> tuple[str, ...]:
    compact = _compact(text)
    for cue in _EXPLICIT_ACTIVE_CROP_CUES:
        index = compact.find(cue)
        if index < 0:
            continue
        selected = _extract_crops(compact[index + len(cue) : index + len(cue) + 24])
        if selected:
            return selected[:1]
    return ()


def _canonical_crop(value: str) -> str:
    cleaned = _compact(value)
    return _CANONICAL_CROP_ALIASES.get(cleaned, cleaned)


def _year_factor(value: str, current_year: int) -> float:
    parsed = _parse_dt(value)
    if parsed is None:
        return 0.55
    if parsed.year == current_year:
        return 1.0
    if parsed.year == current_year - 1:
        return 0.10
    return 0.0


def _row_relevant_to_crop(row: dict[str, Any], crop: str, current_year: int) -> bool:
    occurred = _parse_dt(str(row.get("occurred_at") or ""))
    if occurred is not None and occurred.year < current_year - 1:
        return False
    text = _evidence_text(row)
    crops = _extract_crops(text)
    if crops:
        return crop in crops
    # Once a single active crop is resolved with continuity, unlabeled
    # lifecycle rows in the same season remain useful timeline evidence.
    return bool(occurred and occurred.year == current_year and _contains_any(text, _CONTINUITY_CUES))


def _stage_from_crop_age(crop: str, age_days: int) -> tuple[str, str] | None:
    root_crops = {"红薯", "马铃薯", "大蒜", "花生", "生姜", "山药", "芋头"}
    fruiting_crops = {"南瓜", "西瓜", "甜瓜", "黄瓜", "辣椒", "西红柿", "茄子"}
    grain_crops = {"水稻", "玉米", "小麦", "大豆", "棉花"}
    if crop in root_crops:
        if age_days < 35:
            return "苗期/根系建立期（待核实）", "未来7～20天关注苗势、根系与早期风险"
        if age_days < 85:
            return "营养生长/块根形成期（待核实）", "未来7～20天关注长势与块根形成"
        if age_days < 125:
            return "块根膨大期（待核实）", "未来7～20天关注块根膨大、品质与水肥衔接"
        return "采收/采后窗口（待核实）", "未来7～20天优先确认采收进度、品质与采后安排"
    if crop in fruiting_crops:
        if age_days < 30:
            return "苗期/缓苗期（待核实）", "未来7～20天关注缓苗、根系与苗势"
        if age_days < 60:
            return "开花坐果期（待核实）", "未来7～20天关注花果与长势"
        if age_days < 105:
            return "果实膨大期（待核实）", "未来7～20天关注膨大、品质与采收衔接"
        return "采收/采后窗口（待核实）", "未来7～20天优先确认采收进度与采后安排"
    if crop in grain_crops:
        if age_days < 45:
            return "苗期（待核实）", "未来7～20天关注苗情与根系"
        if age_days < 100:
            return "营养生长/生殖转换期（待核实）", "未来7～20天关注长势与关键生育节点"
        return "成熟/收获窗口（待核实）", "未来7～20天优先确认成熟与收获进度"
    return None


def _latest_stage_cue(
    rows: list[dict[str, Any]],
    now: datetime,
) -> tuple[str, tuple[str, ...]] | None:
    rules = (
        (("已经采完", "采收完", "收完了"), "采后阶段"),
        (("开始采收", "正在采收", "开始收获"), "采收期"),
        (("膨大", "膨果", "薯型"), "膨大至品质形成阶段（待核实）"),
        (("坐果", "幼果"), "坐果/幼果期（待核实）"),
        (("开花", "花期"), "花期（待核实）"),
        (("苗期", "缓苗", "出苗"), "苗期（待核实）"),
    )
    for row in rows:
        occurred = _parse_dt(str(row.get("occurred_at") or ""))
        if occurred is None or occurred.year != now.year:
            continue
        text = _evidence_text(row)
        for words, label in rules:
            if _contains_any(text, words):
                evidence = tuple(_timeline_evidence(rows, "", limit=4))
                return label, evidence
    return None


def _window_for_stage(stage: str, crop: str) -> str:
    if _contains_any(stage, ("采收", "收获", "采后")):
        noun = "薯块品质" if crop == "红薯" else "商品性"
        return f"未来7～20天优先确认采收进度、{noun}与采后安排"
    if _contains_any(stage, ("膨大", "膨果")):
        return "未来7～20天关注膨大、品质与采收衔接"
    if _contains_any(stage, ("开花", "坐果", "幼果")):
        return "未来7～20天关注花果、长势与下一管理节点"
    if _contains_any(stage, ("苗期", "缓苗", "根系")):
        return "未来7～20天关注苗势、根系与早期风险"
    return "未来7～20天先确认当前实际管理节点"


def _timeline_evidence(rows: list[dict[str, Any]], crop: str, *, limit: int) -> list[str]:
    values: list[str] = []
    for row in rows:
        text = _best_evidence_text(row)
        if not text:
            continue
        occurred = str(row.get("occurred_at") or "")[:10]
        label = f"{occurred} {text[:90]}".strip()
        if label not in values:
            values.append(label)
        if len(values) >= limit:
            break
    return values


def _season_name(month: int) -> str:
    if month in {12, 1, 2}:
        return "冬茬"
    if month in {3, 4, 5}:
        return "春茬"
    if month in {6, 7, 8}:
        return "夏茬"
    return "秋茬"


def _evidence_text(row: dict[str, Any]) -> str:
    return " ".join(
        _compact(row.get(key))
        for key in ("category", "evidence_key", "text", "normalized_value", "structured_json", "source_text")
        if row.get(key)
    )


def _best_evidence_text(row: dict[str, Any]) -> str:
    return _compact(row.get("source_text") or row.get("text") or row.get("normalized_value") or "")


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _contains_any(text: str, words: Iterable[str]) -> bool:
    compact = "".join(str(text or "").split()).casefold()
    return any(str(word).casefold() in compact for word in words)


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
