from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


def _clean(value: Any) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _token(value: str) -> str:
    return _clean(value).casefold()


@dataclass(frozen=True)
class CanonicalFactIdentity:
    canonical_category: str
    canonical_key: str
    canonical_value: str
    customer_profile_id: str
    growing_context_id: str | None
    temporal_scope: str | None = None


_KEY_ALIASES = {
    "region": {"region", "location", "地区", "区域", "所在地区", "farm_location"},
    "crop": {"crop", "crop_name", "planted_crop", "作物", "种植作物"},
    "variety": {"variety", "cultivar", "品种", "作物品种"},
    "acreage": {"acreage", "area", "planting_area", "种植面积", "面积"},
    "growth_stage": {"growth_stage", "stage", "生育阶段", "生长阶段"},
    "cultivation_type": {"cultivation_type", "planting_method", "种植方式", "栽培方式"},
    "season_or_year": {"season_or_year", "season", "year", "季节", "年份", "种植季"},
    "planting_date": {"planting_date", "sowing_date", "定植日期", "播种日期"},
    "expected_harvest": {"expected_harvest", "harvest_date", "预计采收", "预计收获"},
    "harvest_status": {"harvest_status", "采收状态", "收获状态"},
    "customer_concern": {"customer_concern", "concern", "担心", "顾虑", "关注点"},
    "customer_preference": {"customer_preference", "preference", "偏好"},
    "price_sensitivity": {"price_sensitivity", "price_concern", "价格敏感", "价格顾虑"},
}
_ALIAS_TO_KEY = {
    _token(alias): canonical
    for canonical, aliases in _KEY_ALIASES.items()
    for alias in aliases
}
_GROWING_CONTEXT_KEYS = {
    "region", "crop", "variety", "acreage", "growth_stage", "cultivation_type",
    "season_or_year", "planting_date", "expected_harvest", "harvest_status",
}
_PREFERENCE_KEYS = {"customer_concern", "customer_preference", "price_sensitivity"}
_CROP_ALIASES = {
    "番茄": "西红柿",
    "西红柿": "西红柿",
    "玉蜀黍": "玉米",
}
_STAGE_ORDER = {
    "定植期": 10,
    "缓苗期": 20,
    "营养生长期": 30,
    "花期": 40,
    "坐果期": 50,
    "幼果期": 60,
    "膨果初期": 70,
    "膨果期": 75,
    "膨果后期": 80,
    "转色期": 90,
    "成熟期": 100,
    "采收期": 110,
    "采后": 120,
    "清园期": 130,
    "休眠期": 140,
}
_AREA_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)\s*(?P<unit>亩|个棚|棚)"
)


def canonicalize_fact_key(fact_category: str, fact_key: str, value_text: str = "") -> tuple[str, str]:
    category_token = _token(fact_category)
    key_token = _token(fact_key)
    canonical_key = _ALIAS_TO_KEY.get(key_token)
    if canonical_key is None and key_token in {"value", "name"}:
        canonical_key = _ALIAS_TO_KEY.get(category_token)
    if canonical_key is None and category_token in {"location", "region", "farm"} and key_token == "location":
        canonical_key = "region"
    canonical_key = canonical_key or key_token or category_token or "general"
    if canonical_key == "customer_concern" and _is_price_concern(value_text):
        canonical_key = "price_sensitivity"
    if canonical_key in _GROWING_CONTEXT_KEYS:
        category = "growing_context"
    elif canonical_key in _PREFERENCE_KEYS:
        category = "customer_preference"
    else:
        category = category_token or "general"
    return category, canonical_key


def canonicalize_fact_identity(
    *,
    customer_profile_id: str,
    growing_context_id: str | None,
    fact_category: str,
    fact_key: str,
    value_text: str,
    normalized_value: Any,
    value_json: str | dict[str, Any] | list[Any] | None = None,
    source_text: str = "",
    temporal_scope: str | None = None,
) -> CanonicalFactIdentity:
    category, key = canonicalize_fact_key(fact_category, fact_key, value_text)
    value = canonicalize_fact_value(
        key,
        value_text=value_text,
        normalized_value=normalized_value,
        value_json=value_json,
        source_text=source_text,
    )
    return CanonicalFactIdentity(
        canonical_category=category,
        canonical_key=key,
        canonical_value=value,
        customer_profile_id=customer_profile_id,
        growing_context_id=growing_context_id,
        temporal_scope=_clean(temporal_scope) or None,
    )


def canonicalize_fact_value(
    canonical_key: str,
    *,
    value_text: str,
    normalized_value: Any,
    value_json: str | dict[str, Any] | list[Any] | None = None,
    source_text: str = "",
) -> str:
    parsed_json = _json_value(value_json)
    if canonical_key == "crop":
        return canonicalize_crop(str(normalized_value or value_text))
    if canonical_key == "variety":
        return canonicalize_variety(str(normalized_value or value_text))
    if canonical_key == "region":
        return canonicalize_region(str(normalized_value or value_text))
    if canonical_key == "acreage":
        return canonicalize_acreage(
            value_text=value_text,
            normalized_value=normalized_value,
            value_json=parsed_json,
            source_text=source_text,
        )
    if canonical_key == "growth_stage":
        return canonicalize_growth_stage(str(normalized_value or value_text), source_text=source_text)
    if canonical_key == "customer_concern":
        return canonicalize_concern(str(normalized_value or value_text))
    if canonical_key == "price_sensitivity":
        return "价格敏感"
    return canonicalize_scalar(normalized_value, fallback=value_text)


def canonicalize_crop(value: str) -> str:
    cleaned = _clean(value)
    return _CROP_ALIASES.get(cleaned, cleaned)


def crop_terms(canonical: str) -> tuple[str, ...]:
    aliases = [alias for alias, target in _CROP_ALIASES.items() if target == canonical]
    return tuple(dict.fromkeys([canonical, *aliases]))


def canonicalize_variety(value: str) -> str:
    return _clean(value)


def canonicalize_region(value: str) -> str:
    cleaned = _clean(value)
    if "省" in cleaned:
        cleaned = cleaned.replace("省", "", 1)
    if cleaned.endswith("市"):
        cleaned = cleaned[:-1]
    return cleaned


def canonicalize_acreage(
    *,
    value_text: str,
    normalized_value: Any = None,
    value_json: dict[str, Any] | list[Any] | None = None,
    source_text: str = "",
) -> str:
    if isinstance(normalized_value, str) and "|" in normalized_value:
        raw_value, raw_unit = normalized_value.split("|", 1)
        number = _number(raw_value)
        unit = _canonical_area_unit(raw_unit)
        if number is not None and unit:
            return f"{format_number(number)}|{unit}"
    if isinstance(value_json, dict):
        number = _number(value_json.get("value"))
        unit = _canonical_area_unit(str(value_json.get("unit") or ""))
        if number is not None and unit:
            return f"{format_number(number)}|{unit}"
    for candidate in (value_text, str(normalized_value or ""), source_text):
        parsed = extract_acreage(candidate)
        if parsed is not None:
            number, unit = parsed
            return f"{format_number(number)}|{unit}"
    return canonicalize_scalar(normalized_value, fallback=value_text)


def extract_acreage(text: str) -> tuple[float, str] | None:
    values = extract_acreages(text)
    return values[0] if values else None


def extract_acreages(text: str) -> tuple[tuple[float, str], ...]:
    values: list[tuple[float, str]] = []
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    for match in _AREA_RE.finditer(normalized):
        number = _number(match.group("value"))
        if number is None:
            continue
        value = (float(number), _canonical_area_unit(match.group("unit")))
        if value not in values:
            values.append(value)
    return tuple(values)


def canonicalize_growth_stage(value: str, *, source_text: str = "") -> str:
    raw = _clean(value)
    evidence = _clean(source_text)
    combined = evidence or raw
    if re.search(r"(?:刚开始|开始|进入|最近(?:已经)?开始|已经开始).*膨果", combined):
        return "膨果初期"
    if any(term in combined for term in ("膨果后期", "膨果末期")):
        return "膨果后期"
    if "快成熟" in combined:
        return "膨果后期"
    rules = (
        (("转色",), "转色期"),
        (("成熟",), "成熟期"),
        (("采收", "收获"), "采收期"),
        (("采后",), "采后"),
        (("清园",), "清园期"),
        (("休眠",), "休眠期"),
        (("幼果",), "幼果期"),
        (("坐果",), "坐果期"),
        (("花期", "开花"), "花期"),
        (("营养生长",), "营养生长期"),
        (("缓苗",), "缓苗期"),
        (("定植",), "定植期"),
        (("膨果初期",), "膨果初期"),
        (("膨果",), "膨果期"),
    )
    for terms, canonical in rules:
        if any(term in combined for term in terms):
            return canonical
    return raw


def growth_stage_relation(existing: str, incoming: str, *, source_text: str = "") -> str:
    old = canonicalize_growth_stage(existing)
    new = canonicalize_growth_stage(incoming, source_text=source_text)
    if old == new:
        return "same"
    if {old, new} <= {"膨果初期", "膨果期"}:
        return "compatible"
    old_order = _STAGE_ORDER.get(old)
    new_order = _STAGE_ORDER.get(new)
    if old_order is not None and new_order is not None and new_order > old_order:
        if _explicit_stage_progress(source_text, new):
            return "advance"
    return "uncertain"


def canonicalize_concern(value: str) -> str:
    cleaned = _clean(value)
    if "烧根" in cleaned or "根烧" in cleaned:
        return "烧根"
    cleaned = re.sub(r"^(?:我)?(?:最)?(?:担心|怕|顾虑)(?:的)?(?:还是)?", "", cleaned)
    cleaned = cleaned.removeprefix("把").removesuffix("了")
    return cleaned or _clean(value)


def canonicalize_scalar(value: Any, *, fallback: str) -> str:
    if value is None:
        return _clean(fallback)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_number(value)
    return _clean(str(value)) or _clean(fallback)


def format_number(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number).rstrip("0").rstrip(".")


def _explicit_stage_progress(source_text: str, canonical_stage: str) -> bool:
    text = _clean(source_text)
    roots = {
        "转色期": ("转色",),
        "成熟期": ("成熟",),
        "采收期": ("采收", "收获"),
        "膨果后期": ("膨果后期", "膨果末期", "快成熟"),
    }.get(canonical_stage, (canonical_stage.removesuffix("期"),))
    return any(root and root in text for root in roots) and any(
        cue in text for cue in ("现在", "目前", "已经", "进入", "到了", "开始", "转入", "最近")
    )


def _is_price_concern(value: str) -> bool:
    cleaned = _clean(value)
    return any(term in cleaned for term in ("价格", "贵", "成本"))


def _canonical_area_unit(value: str) -> str:
    cleaned = _clean(value)
    if cleaned in {"棚", "个棚"}:
        return "个棚"
    return cleaned


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        parsed = _chinese_number(_clean(str(value)))
        return float(parsed) if parsed is not None else None


def _chinese_number(value: str) -> int | None:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value in digits:
        return digits[value]
    total = 0
    section = 0
    current = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in units:
            section += (current or 1) * units[char]
            current = 0
        elif char == "万":
            total += (section + current) * 10000
            section = current = 0
        else:
            return None
    return total + section + current


def _json_value(value: str | dict[str, Any] | list[Any] | None) -> dict[str, Any] | list[Any] | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, (dict, list)) else None
    return value
