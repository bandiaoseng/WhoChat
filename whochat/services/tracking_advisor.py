from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TrackingAdvice:
    number: int
    text: str
    rule: str


class CustomerTrackingAdvisor:
    """Local deterministic tracking guidance; never calls an AI provider."""

    def build(
        self,
        *,
        events: Iterable[Any],
        followups: Iterable[Any],
        core_values: dict[str, str],
        history_status: str,
        sales_stage: str,
    ) -> tuple[TrackingAdvice, ...]:
        event_list = list(events)
        followup_list = list(followups)
        suggestions: list[tuple[str, str]] = []

        if followup_list:
            first = followup_list[0]
            suggestions.append((f"优先完成待回访：{first.title}。", "pending_followup"))

        types = [str(item.event_type) for item in event_list]
        latest_index = {event_type: types.index(event_type) for event_type in set(types)}
        # Events arrive newest first. A smaller index therefore happened later.
        usage = latest_index.get("usage_started")
        effect = latest_index.get("effect_feedback")
        if usage is not None and (effect is None or usage < effect):
            suggestions.append(("客户已有使用记录但缺少后续反馈，建议先询问当前效果。", "usage_without_feedback"))

        quote = latest_index.get("quote_given")
        order = min(
            [value for key, value in latest_index.items() if key in {"order_placed", "purchased"}],
            default=None,
        )
        if quote is not None and (order is None or quote < order):
            suggestions.append(("已有报价但尚无后续订单记录，建议确认客户是否仍在考虑。", "quote_without_order"))

        placed = latest_index.get("order_placed")
        purchased = latest_index.get("purchased")
        ordered = min([value for value in (placed, purchased) if value is not None], default=None)
        if ordered is not None and usage is None:
            suggestions.append(("已有下单或购买记录，建议确认是否收货以及是否开始使用。", "order_without_usage"))

        problem = latest_index.get("problem_reported")
        solution = min(
            [value for key, value in latest_index.items() if key in {"solution_given", "management_plan"}],
            default=None,
        )
        if problem is not None and (solution is None or problem < solution):
            suggestions.append(("最近问题尚无对应方案记录，先补充关键情况再完善处理方向。", "problem_without_solution"))

        missing = [
            label for key, label in (("acreage", "面积"), ("growth_stage", "当前阶段"))
            if not core_values.get(key)
        ]
        if missing:
            suggestions.append((f"后续沟通时一次补充{'和'.join(missing)}，便于判断方案适用性。", "core_profile_gap"))

        if history_status == "history_not_initialized":
            suggestions.append(("历史档案尚未初始化；在完成安全映射前，不把同名客户历史并入当前档案。", "history_not_initialized"))
        elif history_status in {"history_imported", "memory_backfill_pending", "partial", "failed"}:
            suggestions.append(("历史档案尚未完整提取；现阶段仅依据已确认的本地资料跟进。", "history_partial"))

        if not followup_list:
            suggestions.append(("当前没有明确待回访任务；下次联系时先确认客户近期最需要解决的问题。", "no_pending_followup"))

        if sales_stage == "new":
            suggestions.append(("当前仍是新客户阶段，优先补齐需求信息，不急于直接推荐。", "new_customer"))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for text, rule in suggestions:
            if text in seen:
                continue
            seen.add(text)
            deduped.append((text, rule))
        if len(deduped) < 3:
            deduped.append(("查看最近关键记录，确认本次跟进与客户当前作物和问题一致。", "review_recent_context"))
        if len(deduped) < 3:
            deduped.append(("沟通后及时记录新的明确事实、效果反馈或回访时间。", "record_outcome"))
        return tuple(
            TrackingAdvice(number=index + 1, text=text, rule=rule)
            for index, (text, rule) in enumerate(deduped[:6])
        )

