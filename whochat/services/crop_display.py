from __future__ import annotations

from dataclasses import dataclass


# Deliberately explicit: customer names/remarks may contain arbitrary text and
# must never be treated as authoritative agriculture knowledge by inference.
KNOWN_CROPS = tuple(
    sorted(
        {
            "西红柿", "番茄", "莴笋", "生菜", "芹菜", "菠菜", "白菜", "甘蓝",
            "黄瓜", "西瓜", "甜瓜", "南瓜", "冬瓜", "丝瓜", "苦瓜",
            "茄子", "辣椒", "豆角", "架豆", "豇豆", "草莓", "蓝莓",
            "葡萄", "苹果", "梨", "桃", "樱桃", "猕猴桃", "柑橘", "橙",
            "柚", "芒果", "香蕉", "火龙果", "玉米", "小麦", "水稻",
            "大豆", "花生", "棉花", "油菜", "马铃薯", "土豆", "红薯",
            "魔芋", "大葱", "大蒜", "韭菜", "洋葱", "烟草", "茶叶",
            "白芨", "三七", "药材",
        },
        key=lambda item: (-len(item), item),
    )
)


@dataclass(frozen=True)
class CropDisplayResolution:
    value: str
    source_type: str


class CropDisplayResolver:
    """Resolve a display crop without inventing or mutating customer memory."""

    @classmethod
    def resolve(
        cls,
        *,
        followup_crop: str = "",
        event_crop: str = "",
        active_context_crops: str | tuple[str, ...] = (),
        customer_display_name: str = "",
        customer_remark: str = "",
    ) -> CropDisplayResolution:
        if followup_crop.strip():
            return CropDisplayResolution(followup_crop.strip(), "followup_context")
        if event_crop.strip():
            return CropDisplayResolution(event_crop.strip(), "source_event_context")

        context_crops = cls._normalise_values(active_context_crops)
        if len(context_crops) == 1:
            return CropDisplayResolution(context_crops[0], "active_profile_context")
        if len(context_crops) > 1:
            return CropDisplayResolution("/".join(context_crops), "multiple_active_contexts")

        remark_crops = cls._extract_explicit_crops(
            " ".join(item for item in (customer_remark, customer_display_name) if item)
        )
        if remark_crops:
            return CropDisplayResolution("/".join(remark_crops), "customer_remark")
        return CropDisplayResolution("未指定作物", "unresolved")

    @staticmethod
    def _normalise_values(values: str | tuple[str, ...]) -> tuple[str, ...]:
        raw = values.replace("，", ",").replace("、", ",").split(",") if isinstance(values, str) else values
        result: list[str] = []
        for value in raw:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return tuple(result)

    @staticmethod
    def _extract_explicit_crops(text: str) -> tuple[str, ...]:
        remaining = str(text or "")
        found: list[str] = []
        for crop in KNOWN_CROPS:
            if crop in remaining and crop not in found:
                found.append(crop)
                remaining = remaining.replace(crop, " ")
        return tuple(found)
