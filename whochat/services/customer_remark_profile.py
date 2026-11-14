from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from whochat.services.crop_display import KNOWN_CROPS
from whochat.services.memory_canonicalization import extract_acreages, format_number


_LEADING_TAGS = re.compile(r"^[\s\-_/|+＋,，、;；:：()（）\[\]【】{}<>《》.!！?？·]*[ABCHXSZD]{1,3}(?=[\u3400-\u9fff\d])", re.IGNORECASE)
_ADMIN_SUFFIX = re.compile(r"^([\u3400-\u9fff]{1,6}(?:自治州|地区|林区|新区|市|县|区|州|盟|旗))")
_TAG_SEPARATORS = re.compile(r"[\s\-_/|+＋,，、;；:：()（）\[\]【】{}<>《》.!！?？·]+")
_COMPACT_TAG_PREFIX = re.compile(r"^([ABCHXSZD]{1,3})(?=[\u3400-\u9fff\d]|$)", re.IGNORECASE)
_NUMBER_RUN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")

# Province-level information is useful even when a city/county is absent.  The
# aliases are deliberately explicit so a person's name is never guessed to be
# a location.
_PROVINCES = {
    "北京市": "北京", "北京": "北京", "天津市": "天津", "天津": "天津",
    "上海市": "上海", "上海": "上海", "重庆市": "重庆", "重庆": "重庆",
    "河北省": "河北", "河北": "河北", "山西省": "山西", "山西": "山西",
    "辽宁省": "辽宁", "辽宁": "辽宁", "吉林省": "吉林", "吉林": "吉林",
    "黑龙江省": "黑龙江", "黑龙江": "黑龙江", "江苏省": "江苏", "江苏": "江苏",
    "浙江省": "浙江", "浙江": "浙江", "安徽省": "安徽", "安徽": "安徽",
    "福建省": "福建", "福建": "福建", "江西省": "江西", "江西": "江西",
    "山东省": "山东", "山东": "山东", "河南省": "河南", "河南": "河南",
    "湖北省": "湖北", "湖北": "湖北", "湖南省": "湖南", "湖南": "湖南",
    "广东省": "广东", "广东": "广东", "海南省": "海南", "海南": "海南",
    "四川省": "四川", "四川": "四川", "贵州省": "贵州", "贵州": "贵州",
    "云南省": "云南", "云南": "云南", "陕西省": "陕西", "陕西": "陕西",
    "甘肃省": "甘肃", "甘肃": "甘肃", "青海省": "青海", "青海": "青海",
    "台湾省": "台湾", "台湾": "台湾",
    "内蒙古自治区": "内蒙古", "内蒙古": "内蒙古",
    "广西壮族自治区": "广西", "广西": "广西",
    "西藏自治区": "西藏", "西藏": "西藏",
    "宁夏回族自治区": "宁夏", "宁夏": "宁夏",
    "新疆维吾尔自治区": "新疆", "新疆": "新疆",
    "香港特别行政区": "香港", "香港": "香港",
    "澳门特别行政区": "澳门", "澳门": "澳门",
}
_PROVINCE_ALIASES = tuple(sorted(_PROVINCES, key=len, reverse=True))

# Common suffix-less city/county abbreviations seen in customer remarks.  A
# token outside this list is not inferred; explicit 市/县/区/州/盟/旗 suffixes
# are still accepted by _ADMIN_SUFFIX.
_KNOWN_SUBREGIONS = tuple(
    sorted(
        {
            "济宁", "金乡", "洛阳", "青岛", "烟台", "潍坊", "临沂", "菏泽",
            "德州", "聊城", "泰安", "淄博", "枣庄", "威海", "日照", "滨州",
            "郑州", "开封", "南阳", "商丘", "周口", "驻马店", "新乡", "焦作",
            "成都", "绵阳", "德阳", "眉山", "乐山", "宜宾", "百色", "富宁",
            "昆明", "曲靖", "大理", "保山", "西安", "咸阳", "宝鸡", "渭南",
        },
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class RemarkProfileField:
    value: str = ""
    source_text: str = ""


@dataclass(frozen=True)
class RemarkBusinessTags:
    parsed_tags: tuple[str, ...]
    is_worthless: bool
    worthless_reason: str = ""


@dataclass(frozen=True)
class RemarkProfileResolution:
    region: RemarkProfileField
    crop: RemarkProfileField
    variety: RemarkProfileField
    acreage: RemarkProfileField
    tags: RemarkBusinessTags
    source_kind: str

    @property
    def is_worthless(self) -> bool:
        return self.tags.is_worthless

    def field(self, key: str) -> RemarkProfileField:
        return getattr(self, key, RemarkProfileField())


class CustomerRemarkProfileResolver:
    """Conservative, local-only customer remark fallback.

    This resolver does not write facts.  It exposes explicit province/crop/area
    tokens from the current WeChat remark/display name so the profile can show
    useful low-priority values while retaining the exact local source.
    """

    def resolve(self, *, customer_remark: str = "", customer_display_name: str = "") -> RemarkProfileResolution:
        sources = tuple(
            (kind, unicodedata.normalize("NFKC", str(raw or "")).strip())
            for kind, raw in (("remark", customer_remark), ("display_name", customer_display_name))
            if str(raw or "").strip()
        )
        combined = " ".join(raw for _, raw in sources)
        tags = self._parse_business_tags(combined)
        empty = RemarkProfileField()
        if tags.is_worthless:
            return RemarkProfileResolution(empty, empty, empty, empty, tags, "")

        resolved: dict[str, RemarkProfileField] = {}
        source_kind = ""
        for kind, raw in sources:
            cleaned = self._strip_business_prefix(raw)
            candidates = {
                "region": self._extract_region(cleaned),
                "crop": self._extract_crops(cleaned),
                "acreage": self._extract_acreage(cleaned),
                "variety": "",  # Never infer a variety from an arbitrary name.
            }
            for key, value in candidates.items():
                if value and key not in resolved:
                    resolved[key] = RemarkProfileField(value=value, source_text=raw)
                    source_kind = source_kind or kind
        return RemarkProfileResolution(
            region=resolved.get("region", empty),
            crop=resolved.get("crop", empty),
            variety=resolved.get("variety", empty),
            acreage=resolved.get("acreage", empty),
            tags=tags,
            source_kind=source_kind,
        )

    @staticmethod
    def _parse_business_tags(text: str) -> RemarkBusinessTags:
        normalized = unicodedata.normalize("NFKC", text or "").strip()
        upper = normalized.upper()
        found: list[str] = []

        def add(tag: str) -> None:
            if tag in {"A", "B", "C", "Z", "H", "X", "D", "S", "0"} and tag not in found:
                found.append(tag)

        if "无价值" in normalized:
            found.append("无价值")
        for token in (item for item in _TAG_SEPARATORS.split(upper) if item):
            if token in {"A", "B", "C", "Z", "H", "X", "D", "S"}:
                add(token)
        leading = upper.lstrip(" -_/|+＋,，、;；:：()（）[]【】{}<>《》.!！?？·")
        compact = _COMPACT_TAG_PREFIX.match(leading)
        if compact:
            for tag in compact.group(1):
                add(tag)
        # Match the canonical eligibility rule: only a complete numeric run
        # whose normalized value is exactly zero is the no-value marker.
        if any(float(match.group(0)) == 0 for match in _NUMBER_RUN.finditer(upper)):
            add("0")
        worthless_reason = "explicit_no_value" if "无价值" in found else "tag_s" if "S" in found else "standalone_zero" if "0" in found else ""
        return RemarkBusinessTags(tuple(found), bool(worthless_reason), worthless_reason)

    @staticmethod
    def _strip_business_prefix(text: str) -> str:
        current = text.strip()
        # Compact prefixes such as Bh/Ah and delimited prefixes such as A-H-
        # are tags, not pieces of region/crop names.
        for _ in range(4):
            updated = _LEADING_TAGS.sub("", current, count=1).lstrip(" -_/|+＋,，、;；:：")
            if updated == current:
                break
            current = updated
        return current

    @staticmethod
    def _extract_crops(text: str) -> str:
        positions: list[tuple[int, str]] = []
        for crop in KNOWN_CROPS:
            index = text.find(crop)
            if index >= 0:
                positions.append((index, crop))
        positions.sort(key=lambda item: (item[0], -len(item[1])))
        found: list[str] = []
        covered_until = -1
        for index, crop in positions:
            if index < covered_until or crop in found:
                continue
            found.append(crop)
            covered_until = index + len(crop)
        return "/".join(found)

    @staticmethod
    def _extract_acreage(text: str) -> str:
        values = extract_acreages(text)
        return "/".join(f"{format_number(number)}{unit}" for number, unit in values)

    @classmethod
    def _extract_region(cls, text: str) -> str:
        match_alias = ""
        match_index: int | None = None
        for alias in _PROVINCE_ALIASES:
            index = text.find(alias)
            if index >= 0 and (match_index is None or index < match_index or (index == match_index and len(alias) > len(match_alias))):
                match_alias = alias
                match_index = index
        if match_index is None:
            return ""
        province = _PROVINCES[match_alias]
        tail = text[match_index + len(match_alias):]
        parts: list[str] = []
        while tail:
            explicit = _ADMIN_SUFFIX.match(tail)
            if explicit:
                token = explicit.group(1)
                parts.append(token)
                tail = tail[len(token):]
                continue
            token = next((item for item in _KNOWN_SUBREGIONS if tail.startswith(item)), "")
            if not token:
                break
            parts.append(token)
            tail = tail[len(token):]
        return province + "".join(parts)


def remark_values_compatible(key: str, authoritative: str, remark: str) -> bool:
    """Return True when the fallback agrees with a stronger local value."""
    left = str(authoritative or "").strip()
    right = str(remark or "").strip()
    if not left or not right:
        return True
    if key == "region":
        return left.startswith(right) or right.startswith(left)
    if key == "crop":
        return bool(set(left.replace("，", "/").split("/")) & set(right.replace("，", "/").split("/")))
    if key == "acreage":
        return set(left.replace("，", "/").split("/")) == set(right.replace("，", "/").split("/"))
    return left == right
