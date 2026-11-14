from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from whochat.core.models import (
    CustomerEvent,
    CustomerEventStatus,
    CustomerEventType,
    CustomerFollowup,
    CustomerFollowupSource,
    CustomerFollowupStatus,
    CustomerProfile,
    CustomerSalesStage,
    CustomerSalesStageSource,
)
from whochat.storage.customer_profiles import CustomerProfileRepository
from whochat.storage.repositories import LogRepository
from whochat.services.crop_display import CropDisplayResolver


SALES_STAGE_LABELS = {
    "new": "新客户",
    "needs_identified": "已发现需求",
    "solution_provided": "已给方案",
    "product_recommended": "已推荐产品",
    "quoted": "已报价",
    "considering": "考虑中",
    "ordered": "已下单",
    "using": "使用中",
    "result_followup": "效果回访",
    "repeat_customer": "复购客户",
    "dormant": "沉默客户",
}


@dataclass(frozen=True)
class FollowupRule:
    followup_type: str | None
    title: str
    delay_days: int
    priority: int
    closes: tuple[str, ...] = ()


# Delays are intentionally centralised and conservative.  This is a sales /
# communication reminder set, not a crop-calendar knowledge base.
FOLLOWUP_RULES: dict[CustomerEventType, FollowupRule] = {
    CustomerEventType.QUOTE_GIVEN: FollowupRule(
        "quote_followup", "跟进报价与客户考虑情况", 2, 60
    ),
    CustomerEventType.ORDER_PLACED: FollowupRule(
        "order_followup", "确认订单后续、收货与使用安排", 3, 75,
        closes=("quote_followup",),
    ),
    CustomerEventType.USAGE_STARTED: FollowupRule(
        "effect_followup", "回访使用效果和当前问题", 3, 75,
        closes=("order_followup", "delivery_followup", "usage_followup"),
    ),
    CustomerEventType.EFFECT_FEEDBACK: FollowupRule(
        None, "", 0, 0,
        closes=("usage_followup", "effect_followup"),
    ),
}


EVENT_STAGE: dict[CustomerEventType, CustomerSalesStage] = {
    CustomerEventType.PROBLEM_REPORTED: CustomerSalesStage.NEEDS_IDENTIFIED,
    CustomerEventType.DIAGNOSIS: CustomerSalesStage.NEEDS_IDENTIFIED,
    CustomerEventType.SOLUTION_GIVEN: CustomerSalesStage.SOLUTION_PROVIDED,
    CustomerEventType.MANAGEMENT_PLAN: CustomerSalesStage.SOLUTION_PROVIDED,
    CustomerEventType.DOSAGE_GIVEN: CustomerSalesStage.SOLUTION_PROVIDED,
    CustomerEventType.PRODUCT_RECOMMENDED: CustomerSalesStage.PRODUCT_RECOMMENDED,
    CustomerEventType.QUOTE_GIVEN: CustomerSalesStage.QUOTED,
    CustomerEventType.ORDER_INTENT: CustomerSalesStage.CONSIDERING,
    CustomerEventType.ORDER_PLACED: CustomerSalesStage.ORDERED,
    CustomerEventType.PURCHASED: CustomerSalesStage.ORDERED,
    CustomerEventType.USAGE_STARTED: CustomerSalesStage.USING,
    CustomerEventType.EFFECT_FEEDBACK: CustomerSalesStage.RESULT_FOLLOWUP,
}


STAGE_RANK = {
    CustomerSalesStage.NEW: 0,
    CustomerSalesStage.NEEDS_IDENTIFIED: 1,
    CustomerSalesStage.SOLUTION_PROVIDED: 2,
    CustomerSalesStage.PRODUCT_RECOMMENDED: 3,
    CustomerSalesStage.QUOTED: 4,
    CustomerSalesStage.CONSIDERING: 5,
    CustomerSalesStage.REPEAT_CUSTOMER: 5,
    CustomerSalesStage.ORDERED: 6,
    CustomerSalesStage.USING: 7,
    CustomerSalesStage.RESULT_FOLLOWUP: 8,
    CustomerSalesStage.DORMANT: 0,
}


@dataclass(frozen=True)
class SalesStageDecision:
    stage: CustomerSalesStage
    reason: str
    source_event_id: str | None
    source_event_at: str | None


@dataclass(frozen=True)
class FollowupWorkbenchItem:
    id: str
    customer_profile_id: str
    account_scope_id: str
    contact_id: str
    customer_name: str
    account_name: str
    crop: str
    crop_source_type: str
    title: str
    due_at: str
    sales_stage: str
    sales_stage_label: str
    priority: int
    followup_type: str
    source: str
    reason: str
    bucket: str
    latest_business_at: str
    additional_followup_count: int = 0


@dataclass(frozen=True)
class FollowupWorkbench:
    overdue: tuple[FollowupWorkbenchItem, ...]
    today: tuple[FollowupWorkbenchItem, ...]
    future7: tuple[FollowupWorkbenchItem, ...]
    revision: str

    @property
    def all_items(self) -> tuple[FollowupWorkbenchItem, ...]:
        return self.overdue + self.today + self.future7


class CustomerSalesStateEngine:
    """Derive a light-weight current sales stage from authoritative events."""

    def __init__(self, profiles: CustomerProfileRepository) -> None:
        self.profiles = profiles

    def decide(self, customer_profile_id: str) -> SalesStageDecision:
        events = [
            item
            for item in self.profiles.list_customer_events(customer_profile_id, limit=1000)
            if item.status == CustomerEventStatus.ACTIVE and item.event_type in EVENT_STAGE
        ]
        if not events:
            return SalesStageDecision(CustomerSalesStage.NEW, "尚无有效业务事件", None, None)
        events.sort(key=lambda item: (_event_time(item), item.created_at, item.id))

        # A new problem after a progressed cycle starts a light-weight new
        # cycle.  We deliberately keep this implicit; no CRM-cycle table is
        # required for the first version.
        cycle_start = 0
        for index, item in enumerate(events):
            if item.event_type != CustomerEventType.PROBLEM_REPORTED or index == 0:
                continue
            prior = events[:index]
            if any(STAGE_RANK[EVENT_STAGE[event.event_type]] >= STAGE_RANK[CustomerSalesStage.QUOTED] for event in prior):
                cycle_start = index
        current = events[cycle_start:]
        selected = max(
            current,
            key=lambda item: (STAGE_RANK[EVENT_STAGE[item.event_type]], _event_time(item), item.created_at),
        )
        stage = EVENT_STAGE[selected.event_type]
        historical_purchase = any(
            item.event_type
            in {
                CustomerEventType.ORDER_PLACED,
                CustomerEventType.PURCHASED,
                CustomerEventType.USAGE_STARTED,
                CustomerEventType.EFFECT_FEEDBACK,
            }
            for item in events[:cycle_start]
        )
        if historical_purchase and STAGE_RANK[stage] <= STAGE_RANK[CustomerSalesStage.CONSIDERING]:
            stage = CustomerSalesStage.REPEAT_CUSTOMER
            reason = f"历史已有购买/使用记录，当前出现新一轮需求事件：{selected.title}"
        else:
            reason = f"最近销售周期存在 {selected.event_type.value}：{selected.title}"
        return SalesStageDecision(stage, reason, selected.id, _event_time(selected))

    def reconcile(self, customer_profile_id: str) -> CustomerProfile:
        profile = self.profiles.get_customer_profile(customer_profile_id)
        if profile is None:
            raise ValueError("customer profile does not exist")
        decision = self.decide(customer_profile_id)
        if profile.sales_stage_source == CustomerSalesStageSource.USER_EXPLICIT:
            if decision.source_event_id is None or not _after_manual_stage(decision.source_event_at, profile.sales_stage_updated_at):
                return profile
            if decision.stage == profile.sales_stage:
                return profile
        updated, _ = self.profiles.set_sales_stage(
            customer_profile_id,
            decision.stage,
            source=CustomerSalesStageSource.SYSTEM_DERIVED,
            reason=decision.reason,
            source_event_id=decision.source_event_id,
        )
        return updated

    def set_user_stage(
        self,
        customer_profile_id: str,
        stage: CustomerSalesStage | str,
        *,
        reason: str,
    ) -> CustomerProfile:
        updated, _ = self.profiles.set_sales_stage(
            customer_profile_id,
            stage,
            source=CustomerSalesStageSource.USER_EXPLICIT,
            reason=reason or "用户在客户档案中手工设置销售阶段",
        )
        return updated


class FollowupPlanner:
    def __init__(self, profiles: CustomerProfileRepository) -> None:
        self.profiles = profiles

    def apply_events(self, customer_profile_id: str, event_ids: Iterable[str]) -> dict[str, int]:
        created = 0
        merged = 0
        closed = 0
        for event_id in dict.fromkeys(event_ids):
            event = self.profiles.get_customer_event(event_id)
            if event is None or event.customer_profile_id != customer_profile_id:
                continue
            if event.status != CustomerEventStatus.ACTIVE:
                continue
            rule = FOLLOWUP_RULES.get(event.event_type)
            if rule is None:
                continue
            for followup in self.profiles.list_followups(
                customer_profile_id,
                status=CustomerFollowupStatus.PENDING,
            ):
                if followup.followup_type not in rule.closes:
                    continue
                if event.growing_context_id is not None and followup.growing_context_id not in {None, event.growing_context_id}:
                    continue
                self.profiles.resolve_followup(followup.id, status=CustomerFollowupStatus.COMPLETED)
                closed += 1
            if rule.followup_type is None:
                continue
            due_at = (_parse_time(event.event_at) + timedelta(days=rule.delay_days)).isoformat()
            sources = self.profiles.list_event_sources(event.id)
            source_message_id = next((item.message_id for item in sources if item.message_id), None)
            _, was_created = self.profiles.upsert_pending_followup(
                customer_profile_id,
                growing_context_id=event.growing_context_id,
                related_event_id=event.id,
                followup_type=rule.followup_type,
                title=rule.title,
                due_at=due_at,
                priority=rule.priority,
                source=CustomerFollowupSource.SYSTEM_RULE,
                source_message_id=source_message_id,
                reason=f"由 {event.event_type.value} 事件触发，默认 {rule.delay_days} 天后跟进",
            )
            created += int(was_created)
            merged += int(not was_created)
        return {"created": created, "merged": merged, "closed": closed}


class SalesFollowupService:
    """Local-only sales automation and the single follow-up UI boundary."""

    def __init__(
        self,
        profiles: CustomerProfileRepository,
        *,
        logs: LogRepository | None = None,
    ) -> None:
        self.profiles = profiles
        self.engine = CustomerSalesStateEngine(profiles)
        self.planner = FollowupPlanner(profiles)
        self.logs = logs

    def on_memory_written(self, customer_profile_id: str, event_ids: Iterable[str]) -> None:
        event_ids = tuple(event_ids)
        if event_ids:
            self.planner.apply_events(customer_profile_id, event_ids)
        self.engine.reconcile(customer_profile_id)

    def reconcile_profile(self, customer_profile_id: str) -> CustomerProfile:
        return self.engine.reconcile(customer_profile_id)

    def set_user_stage(self, customer_profile_id: str, stage: str, *, reason: str = "") -> CustomerProfile:
        return self.engine.set_user_stage(customer_profile_id, stage, reason=reason)

    def create_user_followup(
        self,
        customer_profile_id: str,
        *,
        title: str,
        due_at: str,
        priority: int,
        growing_context_id: str | None = None,
        followup_type: str = "general_followup",
    ) -> CustomerFollowup:
        item, _ = self.profiles.upsert_pending_followup(
            customer_profile_id,
            growing_context_id=growing_context_id,
            followup_type=followup_type,
            title=title,
            due_at=_parse_time(due_at).isoformat(),
            priority=priority,
            source=CustomerFollowupSource.USER_CREATED,
            reason="用户手工创建回访任务",
        )
        return item

    def update_followup(self, followup_id: str, *, title: str, due_at: str, priority: int) -> CustomerFollowup:
        return self.profiles.update_followup(
            followup_id,
            title=title,
            due_at=_parse_time(due_at).isoformat(),
            priority=priority,
        )

    def complete_followup(self, followup_id: str) -> CustomerFollowup:
        return self.profiles.resolve_followup(followup_id, status=CustomerFollowupStatus.COMPLETED)

    def cancel_followup(self, followup_id: str) -> CustomerFollowup:
        return self.profiles.resolve_followup(followup_id, status=CustomerFollowupStatus.CANCELLED)

    def get_workbench(self, *, now: datetime | None = None) -> FollowupWorkbench:
        # Stage reconciliation happens when memory/events are written.  A read
        # only workbench must not synchronously scan and rewrite every profile.
        current = (now or datetime.now().astimezone()).astimezone()
        start = current.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = start + timedelta(days=1)
        horizon = tomorrow + timedelta(days=7)
        candidates: list[FollowupWorkbenchItem] = []
        for row in self.profiles.list_pending_followup_details(limit=5000):
            due = _parse_time(row["due_at"]).astimezone(current.tzinfo)
            if due < start:
                bucket = "overdue"
            elif due < tomorrow:
                bucket = "today"
            elif due < horizon:
                bucket = "future7"
            else:
                continue
            crop = CropDisplayResolver.resolve(
                followup_crop=row["followup_crop"] or "",
                event_crop=row["event_crop"] or "",
                active_context_crops=row["active_context_crops"] or "",
                customer_display_name=row["contact_display_name"] or row["display_name"] or "",
                customer_remark=row["contact_remark"] or "",
            )
            candidates.append(FollowupWorkbenchItem(
                id=row["id"],
                customer_profile_id=row["customer_profile_id"],
                account_scope_id=row["account_scope_id"],
                contact_id=row["contact_id"],
                customer_name=row["display_name"],
                account_name=row["account_name"],
                crop=crop.value,
                crop_source_type=crop.source_type,
                title=row["title"],
                due_at=row["due_at"],
                sales_stage=row["sales_stage"],
                sales_stage_label=SALES_STAGE_LABELS.get(row["sales_stage"], row["sales_stage"]),
                priority=int(row["priority"]),
                followup_type=row["followup_type"],
                source=row["source"],
                reason=row["reason"],
                bucket=bucket,
                latest_business_at=row["latest_business_at"] or "",
            ))

        grouped: dict[tuple[str, str], list[FollowupWorkbenchItem]] = {}
        for item in candidates:
            grouped.setdefault((item.account_scope_id, item.contact_id), []).append(item)
        buckets: dict[str, list[FollowupWorkbenchItem]] = {"overdue": [], "today": [], "future7": []}
        for items in grouped.values():
            selected = max(items, key=_followup_value_key)
            selected = FollowupWorkbenchItem(
                **{**asdict(selected), "additional_followup_count": len(items) - 1}
            )
            buckets[selected.bucket].append(selected)
        for key in buckets:
            buckets[key].sort(
                key=lambda item: (
                    -item.priority,
                    _descending_time_key(item.latest_business_at),
                    _parse_time(item.due_at),
                    item.account_name,
                    item.customer_name,
                )
            )
        payload = [asdict(item) for key in ("overdue", "today", "future7") for item in buckets[key]]
        revision = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return FollowupWorkbench(
            overdue=tuple(buckets["overdue"]),
            today=tuple(buckets["today"]),
            future7=tuple(buckets["future7"]),
            revision=revision,
        )


def _parse_time(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("time is required")
    if len(text) == 16 and text[4] == "-" and text[7] == "-":
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M").astimezone()
    else:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
    return parsed


def _event_time(event: CustomerEvent) -> str:
    return event.event_at or event.created_at


def _after_manual_stage(event_at: str | None, stage_at: str | None) -> bool:
    if not event_at:
        return False
    if not stage_at:
        return True
    return _parse_time(event_at) > _parse_time(stage_at)


def _descending_time_key(value: str) -> float:
    if not value:
        return 0.0
    try:
        return -_parse_time(value).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _followup_value_key(item: FollowupWorkbenchItem) -> tuple[int, int, float, int, float, str]:
    followup_type = item.followup_type.lower()
    if item.source in {"explicit", "user_created"}:
        rank = 600
    elif any(token in followup_type for token in ("order", "payment", "shipment", "delivery", "usage")):
        rank = 500
    elif "effect" in followup_type:
        rank = 400
    elif "quote" in followup_type:
        rank = 300
    elif any(token in followup_type for token in ("solution", "problem", "diagnosis")):
        rank = 200
    else:
        rank = 100
    source_rank = {"explicit": 3, "user_created": 2, "system_rule": 1}.get(item.source, 0)
    latest = 0.0
    due = 0.0
    try:
        latest = _parse_time(item.latest_business_at).timestamp() if item.latest_business_at else 0.0
    except (TypeError, ValueError):
        pass
    try:
        due = _parse_time(item.due_at).timestamp()
    except (TypeError, ValueError):
        pass
    return rank, source_rank, latest, item.priority, due, item.id
