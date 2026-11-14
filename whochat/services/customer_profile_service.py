from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from whochat.core.models import (
    CustomerEvent,
    CustomerFact,
    CustomerFactStatus,
    CustomerFollowup,
    CustomerGrowingContext,
    CustomerProfile,
)
from whochat.services.memory_canonicalization import format_number
from whochat.services.customer_remark_profile import (
    CustomerRemarkProfileResolver,
    RemarkProfileResolution,
    remark_values_compatible,
)
from whochat.services.sales_followup import SALES_STAGE_LABELS, SalesFollowupService
from whochat.services.tracking_advisor import CustomerTrackingAdvisor, TrackingAdvice
from whochat.storage.customer_profiles import CustomerProfileRepository
from whochat.storage.repositories import AccountScopeRepository, ContactRepository, LogRepository


EVENT_LABELS = {
    "problem_reported": ("⚠", "问题"),
    "diagnosis": ("🔎", "诊断"),
    "solution_given": ("📋", "方案"),
    "product_recommended": ("🌱", "产品推荐"),
    "dosage_given": ("🧪", "用量"),
    "quote_given": ("¥", "报价"),
    "order_intent": ("💬", "购买意向"),
    "order_placed": ("🧾", "已下单"),
    "purchased": ("✅", "已购买"),
    "usage_started": ("🧪", "已使用方案"),
    "effect_feedback": ("✅", "效果反馈"),
    "complaint": ("!", "投诉"),
    "customer_concern": ("💭", "客户顾虑"),
    "followup": ("📅", "回访"),
    "management_plan": ("🗂", "管理计划"),
    "other": ("•", "其他"),
}

TIMELINE_FILTERS = {
    "all": None,
    "problem": {"problem_reported", "diagnosis", "complaint", "customer_concern"},
    "solution": {"solution_given", "dosage_given", "management_plan"},
    "product_quote": {"product_recommended", "quote_given", "order_intent"},
    "purchase_usage": {"order_placed", "purchased", "usage_started"},
    "effect": {"effect_feedback"},
    "followup": {"followup"},
}

CORE_FIELD_LABELS = {
    "region": "地区",
    "crop": "作物",
    "variety": "品种",
    "acreage": "面积",
    "growth_stage": "当前阶段",
    "cultivation_type": "种植方式",
    "season_or_year": "年度/季节",
}

CONCERN_KEYS = {"customer_concern", "customer_preference", "price_sensitivity"}


@dataclass(frozen=True)
class DashboardContext:
    id: str
    crop: str
    variety: str
    region: str
    acreage: str
    growth_stage: str
    cultivation_type: str
    season_or_year: str
    status: str
    last_confirmed_at: str

    @property
    def short_label(self) -> str:
        parts = [item for item in (self.crop, self.variety, self.acreage) if item]
        return "｜".join(parts) or "未命名种植项目"


@dataclass(frozen=True)
class DashboardFact:
    id: str
    growing_context_id: str | None
    category: str
    key: str
    label: str
    value: str
    status: str
    provenance: str
    last_confirmed_at: str


@dataclass(frozen=True)
class DashboardField:
    key: str
    label: str
    value: str
    fact_id: str | None
    source_type: str = ""
    source_text: str = ""


@dataclass(frozen=True)
class DashboardEvent:
    id: str
    growing_context_id: str | None
    event_type: str
    icon: str
    label: str
    title: str
    summary: str
    event_at: str


@dataclass(frozen=True)
class DashboardFollowup:
    id: str
    growing_context_id: str | None
    followup_type: str
    title: str
    due_at: str
    priority: int
    source: str
    reason: str


@dataclass(frozen=True)
class FactSourceView:
    observed_at: str
    sender: str
    source_text: str
    message_id: str | None
    historical_message_id: str | None


@dataclass(frozen=True)
class OrderSourceView:
    observed_at: str
    sender: str
    source_text: str
    message_id: str | None
    historical_message_id: str | None


@dataclass(frozen=True)
class OrderRecord:
    event_id: str
    event_type: str
    event_at: str
    product: str
    specification: str
    quantity: str
    unit_price: str
    amount: str
    status: str
    payment: str
    shipping: str
    sources: tuple[OrderSourceView, ...]


@dataclass(frozen=True)
class DashboardArchiveItem:
    number: int
    key: str
    label: str
    value: str


@dataclass(frozen=True)
class CustomerDashboard:
    available: bool
    account_scope_id: str
    contact_id: str
    profile_id: str
    display_name: str
    account_name: str
    sales_stage: str
    sales_stage_label: str
    sales_stage_source: str
    contexts: tuple[DashboardContext, ...]
    selected_context_id: str | None
    other_context_count: int
    core_fields: tuple[DashboardField, ...]
    concerns: tuple[DashboardFact, ...]
    pending_facts: tuple[DashboardFact, ...]
    events: tuple[DashboardEvent, ...]
    followups: tuple[DashboardFollowup, ...]
    order_records: tuple[OrderRecord, ...]
    history_status: str
    history_status_label: str
    archive_items: tuple[DashboardArchiveItem, ...]
    tracking_advice: tuple[TrackingAdvice, ...]
    revision: str


class CustomerProfileService:
    """Local-only customer dashboard and auditable manual correction boundary."""

    def __init__(
        self,
        profiles: CustomerProfileRepository,
        account_scopes: AccountScopeRepository,
        contacts: ContactRepository,
        *,
        logs: LogRepository | None = None,
        sales_followup: SalesFollowupService | None = None,
        tracking_advisor: CustomerTrackingAdvisor | None = None,
    ) -> None:
        self.profiles = profiles
        self.account_scopes = account_scopes
        self.contacts = contacts
        self.logs = logs
        self.sales_followup = sales_followup
        self.tracking_advisor = tracking_advisor or CustomerTrackingAdvisor()
        self.remark_profile = CustomerRemarkProfileResolver()
        self._logged_remark_conflicts: set[tuple[str, str, str, str]] = set()

    def get_customer_dashboard(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        selected_context_id: str | None = None,
        timeline_filter: str = "all",
        event_limit: int = 15,
    ) -> CustomerDashboard:
        contact = self.contacts.get(contact_id)
        if contact is None or contact.account_scope_id != account_scope_id:
            raise ValueError("customer_dashboard_contact_scope_mismatch")
        scope = self.account_scopes.get_scope(account_scope_id)
        account_name = scope.local_name if scope else "未绑定工作微信"
        profile = self.profiles.get_customer_profile(
            account_scope_id=account_scope_id,
            contact_id=contact_id,
        )
        remark_profile = self.remark_profile.resolve(
            customer_remark=contact.remark,
            customer_display_name=contact.display_name,
        )
        if profile is None:
            core_fields = self._core_fields(None, [], remark_profile)
            history_status = "history_not_initialized"
            tracking = self.tracking_advisor.build(
                events=(), followups=(),
                core_values={item.key: item.value for item in core_fields},
                history_status=history_status, sales_stage="new",
            )
            archive_items = self._archive_items(
                core_fields=core_fields, events=(), concerns=(), followups=(),
                order_records=(),
                sales_stage_label=SALES_STAGE_LABELS["new"],
            )
            return CustomerDashboard(
                available=False,
                account_scope_id=account_scope_id,
                contact_id=contact_id,
                profile_id="",
                display_name=contact.display_name,
                account_name=account_name,
                sales_stage="new",
                sales_stage_label=SALES_STAGE_LABELS["new"],
                sales_stage_source="system_derived",
                contexts=(),
                selected_context_id=None,
                other_context_count=0,
                core_fields=core_fields,
                concerns=(),
                pending_facts=(),
                events=(),
                followups=(),
                order_records=(),
                history_status=history_status,
                history_status_label="历史档案尚未初始化",
                archive_items=archive_items,
                tracking_advice=tracking,
                revision=hashlib.sha256(
                    f"empty:{account_scope_id}:{contact_id}:{contact.display_name}".encode("utf-8")
                ).hexdigest(),
            )

        contexts = tuple(self._context_view(item) for item in self.profiles.list_growing_contexts(profile.id))
        valid_ids = {item.id for item in contexts}
        if selected_context_id not in valid_ids:
            selected_context_id = next((item.id for item in contexts if item.status == "active"), None)
            selected_context_id = selected_context_id or (contexts[0].id if contexts else None)
        facts = self.profiles.list_facts(
            profile.id,
            statuses=(CustomerFactStatus.ACTIVE, CustomerFactStatus.UNCERTAIN, CustomerFactStatus.CONFLICTING),
        )
        relevant = [item for item in facts if item.growing_context_id in {None, selected_context_id}]
        active = [item for item in relevant if item.status == CustomerFactStatus.ACTIVE]
        selected_context = next((item for item in contexts if item.id == selected_context_id), None)
        core_fields = self._core_fields(selected_context, active, remark_profile)
        self._audit_remark_conflicts(
            account_scope_id=account_scope_id,
            contact_id=contact_id,
            profile_id=profile.id,
            core_fields=core_fields,
            remark_profile=remark_profile,
        )
        concerns = tuple(
            self._fact_view(item)
            for item in active
            if item.fact_key in CONCERN_KEYS or item.fact_category in {"customer_concern", "customer_preference", "price_sensitivity"}
        )
        pending = tuple(
            self._fact_view(item)
            for item in relevant
            if item.status in {CustomerFactStatus.UNCERTAIN, CustomerFactStatus.CONFLICTING}
        )
        allowed_types = TIMELINE_FILTERS.get(timeline_filter, TIMELINE_FILTERS["all"])
        all_relevant_event_rows = [
            item for item in self.profiles.list_customer_events(profile.id, limit=500)
            if item.growing_context_id in {None, selected_context_id}
            and (allowed_types is None or item.event_type.value in allowed_types)
        ]
        events = tuple(
            self._event_view(item)
            for item in all_relevant_event_rows[: max(1, int(event_limit))]
        )
        # The visible timeline stays compact, but numbered profile fields and
        # local tracking must not lose an older problem/solution merely because
        # a Historical Replay adds many newer sales events.
        archive_events = tuple(self._event_view(item) for item in all_relevant_event_rows)
        followups = tuple(
            self._followup_view(item)
            for item in self.profiles.list_pending_followups(profile.id, limit=100)
            if item.growing_context_id in {None, selected_context_id}
        )
        order_records = self._order_records(all_relevant_event_rows)
        history_status = profile.history_status
        archive_items = self._archive_items(
            core_fields=core_fields,
            events=archive_events,
            concerns=concerns,
            followups=followups,
            order_records=order_records,
            sales_stage_label=SALES_STAGE_LABELS.get(profile.sales_stage.value, profile.sales_stage.value),
        )
        tracking = self.tracking_advisor.build(
            events=archive_events,
            followups=followups,
            core_values={item.key: item.value for item in core_fields},
            history_status=history_status,
            sales_stage=profile.sales_stage.value,
        )
        payload = {
            "profile": asdict(profile),
            "core_fields": [asdict(item) for item in core_fields],
            "contexts": [asdict(item) for item in contexts],
            "selected_context_id": selected_context_id,
            "facts": [asdict(self._fact_view(item)) for item in relevant],
            "events": [asdict(item) for item in events],
            "followups": [asdict(item) for item in followups],
            "order_records": [asdict(item) for item in order_records],
            "history_status": history_status,
            "archive_items": [asdict(item) for item in archive_items],
            "tracking_advice": [asdict(item) for item in tracking],
            "edits": [asdict(item) for item in self.profiles.list_memory_edits(profile.id, limit=30)],
        }
        revision = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return CustomerDashboard(
            available=True,
            account_scope_id=account_scope_id,
            contact_id=contact_id,
            profile_id=profile.id,
            display_name=profile.display_name,
            account_name=account_name,
            sales_stage=profile.sales_stage.value,
            sales_stage_label=SALES_STAGE_LABELS.get(profile.sales_stage.value, profile.sales_stage.value),
            sales_stage_source=profile.sales_stage_source.value,
            contexts=contexts,
            selected_context_id=selected_context_id,
            other_context_count=max(0, len(contexts) - 1),
            core_fields=core_fields,
            concerns=concerns,
            pending_facts=pending,
            events=events,
            followups=followups,
            order_records=order_records,
            history_status=history_status,
            history_status_label=_history_status_label(history_status),
            archive_items=archive_items,
            tracking_advice=tracking,
            revision=revision,
        )

    def get_tracking_evidence(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        current_year: int,
        fact_limit: int = 120,
        event_limit: int = 160,
    ) -> dict[str, Any]:
        """Read the bounded current/previous-season structured evidence set."""
        contact = self.contacts.get(contact_id)
        if contact is None or contact.account_scope_id != account_scope_id:
            raise ValueError("customer_tracking_scope_mismatch")
        profile = self.profiles.get_customer_profile(
            account_scope_id=account_scope_id,
            contact_id=contact_id,
        )
        if profile is None:
            return {
                "schema": "whochat.tracking_evidence.v2",
                "profile_id": "",
                "window_start": f"{current_year - 1:04d}-01-01T00:00:00",
                "current_year": current_year,
                "facts": [],
                "events": [],
            }
        since_iso = f"{current_year - 1:04d}-01-01T00:00:00"
        evidence = self.profiles.list_tracking_evidence(
            profile.id,
            since_iso=since_iso,
            fact_limit=fact_limit,
            event_limit=event_limit,
        )
        return {
            "schema": "whochat.tracking_evidence.v2",
            "profile_id": profile.id,
            "window_start": since_iso,
            "current_year": current_year,
            "facts": evidence["facts"],
            "events": evidence["events"],
        }

    def get_fact_sources(self, fact_id: str) -> tuple[FactSourceView, ...]:
        fact = self.profiles.get_fact(fact_id)
        if fact is None:
            raise ValueError("fact does not exist")
        return tuple(
            FactSourceView(
                observed_at=str(item["observed_at"]),
                sender=str(item["sender"]),
                source_text=str(item["source_text"]),
                message_id=item["message_id"],
                historical_message_id=item.get("historical_message_id"),
            )
            for item in self.profiles.list_fact_source_details(fact_id)
        )

    def confirm_fact(self, fact_id: str, *, reason: str = "") -> CustomerFact:
        return self.profiles.confirm_fact(fact_id, reason=reason, actor="local_user")

    def correct_fact(self, fact_id: str, *, value_text: str, reason: str = "") -> tuple[CustomerFact, CustomerFact]:
        return self.profiles.correct_fact(fact_id, value_text=value_text, reason=reason, actor="local_user")

    def invalidate_fact(self, fact_id: str, *, reason: str = "") -> CustomerFact:
        return self.profiles.invalidate_fact(fact_id, reason=reason, actor="local_user")

    def set_sales_stage(self, profile_id: str, stage: str, *, reason: str = ""):
        if self.sales_followup is None:
            raise RuntimeError("sales follow-up service is unavailable")
        return self.sales_followup.set_user_stage(profile_id, stage, reason=reason)

    @staticmethod
    def _context_view(item: CustomerGrowingContext) -> DashboardContext:
        acreage = ""
        if item.acreage_value is not None:
            acreage = f"{format_number(item.acreage_value)}{item.acreage_unit}"
        return DashboardContext(
            id=item.id,
            crop=item.crop,
            variety=item.variety,
            region=item.region,
            acreage=acreage,
            growth_stage=item.growth_stage,
            cultivation_type=item.cultivation_type,
            season_or_year=item.season_or_year,
            status=item.status.value,
            last_confirmed_at=item.last_confirmed_at,
        )

    @staticmethod
    def _fact_view(item: CustomerFact) -> DashboardFact:
        return DashboardFact(
            id=item.id,
            growing_context_id=item.growing_context_id,
            category=item.fact_category,
            key=item.fact_key,
            label=CORE_FIELD_LABELS.get(item.fact_key, _fact_label(item.fact_key)),
            value=item.value_text,
            status=item.status.value,
            provenance=item.provenance.value,
            last_confirmed_at=item.last_confirmed_at,
        )

    @classmethod
    def _core_fields(
        cls,
        context: DashboardContext | None,
        active_facts: list[CustomerFact],
        remark_profile: RemarkProfileResolution | None = None,
    ) -> tuple[DashboardField, ...]:
        facts_by_key: dict[str, CustomerFact] = {}
        # Context-specific facts must win over customer-wide facts when the
        # operator switches between crops. Recency still decides within a tier.
        prioritized_facts = sorted(
            enumerate(active_facts),
            key=lambda pair: (
                0 if pair[1].provenance.value == "user_explicit" else
                1 if pair[1].provenance.value == "customer_explicit" else
                2 if pair[1].provenance.value == "system_imported" else 3,
                0
                if context is not None and pair[1].growing_context_id == context.id
                else 1
                if pair[1].growing_context_id is None
                else 2,
                -len(pair[1].value_text or "") if pair[1].fact_key == "region" else 0,
                pair[0],
            ),
        )
        for _, fact in prioritized_facts:
            if fact.fact_key in CORE_FIELD_LABELS and fact.fact_key not in facts_by_key:
                facts_by_key[fact.fact_key] = fact
        fallback = {
            "region": context.region if context else "",
            "crop": context.crop if context else "",
            "variety": context.variety if context else "",
            "acreage": context.acreage if context else "",
            "growth_stage": context.growth_stage if context else "",
            "cultivation_type": context.cultivation_type if context else "",
            "season_or_year": context.season_or_year if context else "",
        }
        fields: list[DashboardField] = []
        for key, label in CORE_FIELD_LABELS.items():
            fact = facts_by_key.get(key)
            if fact is not None:
                fields.append(
                    DashboardField(
                        key=key,
                        label=label,
                        value=fact.value_text,
                        fact_id=fact.id,
                        source_type="manual" if fact.provenance.value == "user_explicit" else "structured_memory",
                    )
                )
                continue
            if fallback[key]:
                fields.append(
                    DashboardField(
                        key=key,
                        label=label,
                        value=fallback[key],
                        fact_id=None,
                        source_type="structured_memory",
                    )
                )
                continue
            remark_field = remark_profile.field(key) if remark_profile is not None else None
            fields.append(
                DashboardField(
                    key=key,
                    label=label,
                    value=remark_field.value if remark_field is not None else "",
                    fact_id=None,
                    source_type="remark" if remark_field is not None and remark_field.value else "",
                    source_text=remark_field.source_text if remark_field is not None else "",
                )
            )
        return tuple(fields)

    def _audit_remark_conflicts(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        profile_id: str,
        core_fields: tuple[DashboardField, ...],
        remark_profile: RemarkProfileResolution,
    ) -> None:
        if self.logs is None:
            return
        for field in core_fields:
            if field.key not in {"region", "crop", "variety", "acreage"} or field.source_type == "remark":
                continue
            remark = remark_profile.field(field.key)
            if not remark.value or remark_values_compatible(field.key, field.value, remark.value):
                continue
            conflict_key = (profile_id, field.key, field.value, remark.value)
            if conflict_key in self._logged_remark_conflicts:
                continue
            self._logged_remark_conflicts.add(conflict_key)
            self.logs.append(
                "warning",
                "customer_profile",
                "remark_conflict",
                "微信备注与更高优先级客户资料不一致；保留高优先级资料",
                {
                    "account_scope_id": account_scope_id,
                    "contact_safe_ref": hashlib.sha256(contact_id.encode("utf-8")).hexdigest()[:12],
                    "profile_id": profile_id,
                    "field": field.key,
                    "kept_value": field.value,
                    "remark_value": remark.value,
                    "remark_hash": hashlib.sha256(remark.source_text.encode("utf-8")).hexdigest(),
                },
            )

    @staticmethod
    def _event_view(item: CustomerEvent) -> DashboardEvent:
        icon, label = EVENT_LABELS.get(item.event_type.value, EVENT_LABELS["other"])
        return DashboardEvent(
            id=item.id,
            growing_context_id=item.growing_context_id,
            event_type=item.event_type.value,
            icon=icon,
            label=label,
            title=item.title,
            summary=item.summary,
            event_at=item.event_at,
        )

    @staticmethod
    def _followup_view(item: CustomerFollowup) -> DashboardFollowup:
        return DashboardFollowup(
            id=item.id,
            growing_context_id=item.growing_context_id,
            followup_type=item.followup_type,
            title=item.title,
            due_at=item.due_at,
            priority=item.priority,
            source=item.source.value,
            reason=item.reason,
        )

    def _order_records(self, events: list[CustomerEvent]) -> tuple[OrderRecord, ...]:
        order_events = [
            item
            for item in events
            if item.status.value != "invalid"
            and item.event_type.value in {"order_intent", "order_placed", "purchased", "quote_given"}
        ]
        source_rows = self.profiles.list_event_source_details_bulk([item.id for item in order_events])
        sources_by_event: dict[str, list[OrderSourceView]] = {}
        for row in source_rows:
            source_text = str(row.get("source_text") or row.get("historical_text") or "").strip()
            sources_by_event.setdefault(str(row["event_id"]), []).append(
                OrderSourceView(
                    observed_at=str(row.get("historical_sent_at") or row.get("observed_at") or ""),
                    sender=str(row.get("message_speaker") or row.get("historical_sender") or "unknown"),
                    source_text=source_text,
                    message_id=row.get("message_id"),
                    historical_message_id=row.get("historical_message_id"),
                )
            )
        records: list[OrderRecord] = []
        for event in order_events:
            try:
                data = json.loads(event.structured_data_json or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            fields = _explicit_order_fields(data)
            records.append(
                OrderRecord(
                    event_id=event.id,
                    event_type=event.event_type.value,
                    event_at=event.event_at,
                    product=fields["product"],
                    specification=fields["specification"],
                    quantity=fields["quantity"],
                    unit_price=fields["unit_price"],
                    amount=fields["amount"],
                    status=_order_status_label(event.event_type.value),
                    payment=fields["payment"],
                    shipping=fields["shipping"],
                    sources=tuple(sources_by_event.get(event.id, ())),
                )
            )
        return tuple(records)

    @staticmethod
    def _archive_items(
        *,
        core_fields: tuple[DashboardField, ...],
        events: tuple[DashboardEvent, ...],
        concerns: tuple[DashboardFact, ...],
        followups: tuple[DashboardFollowup, ...],
        order_records: tuple[OrderRecord, ...],
        sales_stage_label: str,
    ) -> tuple[DashboardArchiveItem, ...]:
        missing = "暂未确认"
        core = {item.key: item.value or missing for item in core_fields}

        def event_values(types: set[str], *, limit: int = 2) -> str:
            values: list[str] = []
            for item in events:
                if item.event_type in types and item.title not in values:
                    values.append(item.title)
                if len(values) >= limit:
                    break
            return "；".join(values) or missing

        values = (
            ("region", "地区", core.get("region", missing)),
            ("crop", "作物", core.get("crop", missing)),
            ("variety", "品种", core.get("variety", missing)),
            ("acreage", "面积", core.get("acreage", missing)),
            ("growth_stage", "当前阶段", core.get("growth_stage", missing)),
            ("cultivation_type", "种植方式", core.get("cultivation_type", missing)),
            ("recent_problem", "最近问题", event_values({"problem_reported", "complaint"})),
            ("historical_solution", "历史方案", event_values({"solution_given", "management_plan", "product_recommended"})),
            ("purchase_order", "购买/订单", _order_summary(order_records)),
            ("usage", "使用情况", event_values({"usage_started"})),
            ("effect_feedback", "效果反馈", event_values({"effect_feedback"})),
            ("customer_concern", "客户关注", "；".join(item.value for item in concerns) or missing),
            ("sales_stage", "销售阶段", sales_stage_label or missing),
            ("followup", "待回访", "；".join(item.title for item in followups[:3]) or missing),
        )
        return tuple(
            DashboardArchiveItem(index + 1, key, label, value)
            for index, (key, label, value) in enumerate(values)
        )


def _order_summary(records: tuple[OrderRecord, ...]) -> str:
    missing = "暂未确认"
    orders = [item for item in records if item.event_type == "order_placed"]
    purchased = [item for item in records if item.event_type == "purchased"]
    if orders:
        latest = orders[0]
        detail = latest.product if latest.product != missing else latest.status
        return f"共{len(orders)}笔订单；最近一笔：{_short_date(latest.event_at)} {detail}"
    if purchased:
        latest = purchased[0]
        detail = latest.product if latest.product != missing else latest.status
        return f"共{len(purchased)}条购买记录；最近一条：{_short_date(latest.event_at)} {detail}"
    quotes = [item for item in records if item.event_type == "quote_given"]
    return f"暂无明确订单；相关报价{len(quotes)}条" if quotes else missing


def _explicit_order_fields(data: dict[str, Any]) -> dict[str, str]:
    missing = "暂未确认"
    items = data.get("items") if isinstance(data.get("items"), list) else []
    products: list[str] = []
    specifications: list[str] = []
    quantities: list[str] = []
    unit_prices: list[str] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        product = _first_text(raw, "product", "name", "product_name")
        if product:
            products.append(product)
        specification = _first_text(raw, "specification", "spec", "package", "package_spec")
        if specification:
            specifications.append(f"{product}：{specification}" if product else specification)
        quantity = _first_text(raw, "quantity", "count", "barrels", "boxes")
        unit = _first_text(raw, "unit")
        if quantity:
            text = f"{quantity}{unit}" if unit else quantity
            quantities.append(f"{product}：{text}" if product else text)
        price = _first_text(raw, "unit_price", "price")
        if price:
            text = _money(price, raw.get("currency") or data.get("currency"))
            unit_prices.append(f"{product}：{text}" if product else text)

    product = "；".join(_unique(products)) or _first_text(data, "product", "product_name", "name") or missing
    specification = "；".join(_unique(specifications)) or _first_text(
        data, "specification", "spec", "package", "package_spec"
    ) or missing
    quantity = "；".join(_unique(quantities))
    if not quantity:
        direct_quantity = _first_text(data, "quantity", "count", "barrels", "boxes")
        direct_unit = _first_text(data, "unit")
        quantity = f"{direct_quantity}{direct_unit}" if direct_quantity and direct_unit else (direct_quantity or missing)
    unit_price = "；".join(_unique(unit_prices))
    if not unit_price:
        direct_price = _first_text(data, "unit_price", "price")
        unit_price = _money(direct_price, data.get("currency")) if direct_price else missing
    amount_value = next(
        (data.get(key) for key in ("final_amount", "total_amount", "total", "amount", "payable") if data.get(key) not in (None, "")),
        None,
    )
    amount = _money(amount_value, data.get("currency")) if amount_value is not None else missing
    payment = _first_text(data, "payment_status", "payment_method")
    if not payment and data.get("payment_received") is True:
        payment = "已收款"
    shipping = _first_text(data, "shipping_status", "delivery_status", "delivery_date", "ship_date")
    return {
        "product": product,
        "specification": specification,
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
        "payment": payment or missing,
        "shipping": shipping or missing,
    }


def _first_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value not in (None, "") and not isinstance(value, (dict, list)):
            return str(value).strip()
    return ""


def _money(value: Any, currency: Any = None) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if "元" in text or str(currency or "").upper() not in {"CNY", "RMB"}:
        return text
    return f"{text}元"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _short_date(value: str) -> str:
    return str(value or "")[:10] or "日期未确认"


def _order_status_label(event_type: str) -> str:
    return {
        "order_intent": "购买意向",
        "order_placed": "已下单",
        "purchased": "已购买/成交",
        "quote_given": "已报价",
    }.get(event_type, event_type)


def _fact_label(key: str) -> str:
    return {
        "customer_concern": "客户关注",
        "customer_preference": "客户偏好",
        "price_sensitivity": "价格关注",
        "planting_date": "种植日期",
        "expected_harvest": "预计采收",
        "harvest_status": "采收状态",
    }.get(key, key.replace("_", " "))


def _history_status_label(status: str) -> str:
    return {
        "history_not_initialized": "历史档案尚未初始化",
        "history_imported": "历史聊天已安全导入",
        "memory_backfill_pending": "历史档案正在初始化",
        "memory_backfilled": "历史档案已初始化",
        "partial": "历史档案部分完成",
        "failed": "历史档案初始化失败",
    }.get(status, "历史档案状态未知")
