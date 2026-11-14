from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import re
import time as monotonic_time
from typing import Any, Callable, Iterable, Protocol
import urllib.error
import urllib.parse
import urllib.request

from PySide6.QtCore import QObject, Signal

from whochat.ai.models import ReplyGenerationResult, ReplySuggestion
from whochat.config import AppConfig
from whochat.core.current_conversation import (
    CurrentMessageView, VisibleMessage, build_current_message_view, observation_semantic_payload,
)
from whochat.core.models import Contact, ContactStatus, Memory, Message, Speaker, utc_now_iso
from whochat.diagnostics import append_diagnostics_log
from whochat.services.conversation_style import (
    CONVERSATION_STYLE_VERSION,
    CustomerConversationStyleProfile,
    CustomerSalutationResolver,
    extract_salutation,
)
from whochat.services.customer_profile_service import CustomerDashboard, CustomerProfileService
from whochat.services.customer_memory_context import CustomerMemoryContextBuilder
from whochat.services.historical_eligibility import CustomerRemarkTagParser, RemarkTagParseResult
from whochat.services.memory_canonicalization import canonicalize_crop
from whochat.services.tracking_reasoning import (
    TRACKING_REASONING_VERSION,
    ActiveCropResolver,
    CropSeasonTimelineService,
    _extract_crops,
)
from whochat.storage.database import Database
from whochat.storage.repositories import (
    ContactRepository,
    ConversationRepository,
    GenerationLogRepository,
    LogRepository,
    MemoryRepository,
    MessageRepository,
    ObservationRepository,
    StrategyRepository,
    new_id,
)


PROACTIVE_REASON_TYPES = frozenset(
    {
        "ineligible",
        "reply_obligation",
        "order_followup",
        "usage_followup",
        "effect_followup",
        "predicted_window",
        "reactivation",
        "do_not_contact_now",
    }
)
PROACTIVE_CONTACT_V1_ENABLED = True
PROACTIVE_CLOUD_AUTH_SCOPE = "ab_proactive_v1:stable_chat_dm"
ACTIVATION_PREWARM_CLOUD_AUTH_SCOPE = "a_activation_prewarm_v1:idle_local_snapshot"
PROACTIVE_POLICY_VERSION = "activation-v1-current-view-v4-20261015"
# These are fixed local validation codes, never arbitrary exception text.
# A refusal to use a current source is not a provider/worker failure and must
# not fall back to old formal history or turn quarantined data into consent.
_PROACTIVE_SOURCE_BLOCK_REASONS = frozenset({
    "proactive_contact_scope_mismatch", "proactive_current_view_binding_incomplete",
    "proactive_current_view_source_missing", "proactive_current_view_not_current",
    "current_view_persisted_source_required", "current_view_conversation_owner_mismatch",
    "current_view_messages_payload_invalid", "current_view_message_count_invalid",
    "current_view_messages_empty", "current_view_expected_binding_incomplete",
    "current_view_unscoped_source", "current_view_source_binding_mismatch",
    "current_view_untrusted_source", "current_view_evidence_generation_mismatch",
    "current_view_expired_or_future_source", "current_view_message_shape_invalid",
    "current_view_uncertain_flag_invalid", "current_view_uncertain_message",
    "current_view_raw_message", "current_view_speaker_invalid", "current_view_group_member_in_dm",
    "current_view_confidence_invalid", "current_view_media_invalid",
    "current_view_media_reference_missing", "current_view_media_reference_invalid",
    "current_view_text_invalid", "current_view_timezone_required", "current_view_time_invalid",
    "reply_memory_current_view_required", "reply_memory_current_view_source_owner_mismatch",
    "reply_memory_current_view_source_changed", "reply_memory_current_view_not_latest",
    "current_view_screen_proof_invalid", "current_view_screen_proof_incomplete",
    "current_view_screen_proof_page_invalid", "current_view_screen_proof_omitted_content",
    "current_view_screen_proof_binding_invalid", "current_view_screen_proof_rows_invalid",
    "current_view_screen_proof_low_confidence", "current_view_screen_proof_expired",
    "current_view_screen_proof_identity_invalid", "current_view_screen_proof_history_reason_invalid",
    "current_view_screen_proof_owner_mismatch", "current_view_screen_proof_capture_mismatch",
    "current_view_screen_proof_rows_mismatch",
    "current_view_live_lease_required", "current_view_runtime_lease_required",
    "current_view_runtime_lifecycle_mismatch", "current_view_not_current",
    "current_view_missing", "identity_stability_lease", "current_source_unavailable",
})
_REPLY_OBLIGATION_WORDS = ("回复", "答复", "报价", "资料", "方案", "明细", "确认", "补发")
_ORDER_WORDS = ("订单", "下单", "购买", "发货", "到货", "收货", "物流")
_USAGE_WORDS = ("使用", "施用", "用药", "用肥", "冲施", "喷施")
_EFFECT_WORDS = ("效果", "反馈", "变化", "恢复", "改善")
_ASK_LATER_WORDS = ("以后再说", "过段时间", "晚点", "下次", "过几天", "再联系")
_REJECTION_WORDS = ("不需要", "不考虑", "别联系", "不要联系", "暂时不要", "不买", "不用了")
_POSITIVE_ACK_WORDS = ("收到", "好的", "好", "可以", "行", "知道了", "谢谢", "明白")
_FORBIDDEN_GENERATED_TERMS = (
    "系统显示",
    "根据档案",
    "根据记忆",
    "根据历史记录",
    "作为AI",
    "AI助手",
    "WhoChat",
    "Sales State",
    "confidence",
    "客户等级",
)
_CURRENT_ASSERTION = re.compile(r"(?:目前|现在|当前).{0,10}(?:正|已经|处于|到了)")
_CLAUSE = re.compile(r"[^。！？!?；;，,]+[。！？!?；;，,]?")


@dataclass(frozen=True)
class ProactiveContactRequest:
    account_scope_id: str
    contact_id: str
    conversation_id: str
    target_instance_id: str
    hwnd: int
    window_title: str
    capture_generation: int
    local_date: str
    proactive_cloud_authorization: str = ""
    current_observation_id: str = ""
    current_session_id: str = ""
    source_job_id: int = 0

    @property
    def binding_key(self) -> str:
        return "|".join(
            (
                self.account_scope_id,
                self.contact_id,
                self.conversation_id,
                self.target_instance_id,
                self.current_session_id,
                str(self.hwnd),
                self.local_date,
                self.proactive_cloud_authorization,
            )
        )


@dataclass(frozen=True)
class ProactiveEligibility:
    eligible: bool
    value_tier: str | None
    parsed_tags: tuple[str, ...]
    is_worthless: bool
    parse_status: str
    reason: str


@dataclass(frozen=True)
class ProactiveCloudAuthorization:
    allowed: bool
    reason: str
    scope: str = ""


@dataclass(frozen=True)
class CustomerContactRecency:
    last_customer_message_at: str | None
    last_our_message_at: str | None
    last_any_contact_at: str | None
    days_since_last_contact: int | None
    silence_threshold_days: int
    silence_candidate: bool


@dataclass(frozen=True)
class CurrentStageAssessment:
    crop: str
    stage_label: str
    status: str
    confidence: str
    region_used: str
    region_source: str
    evidence: tuple[str, ...]
    calculated_at: str
    next_window: str
    source: str = ""
    active_crop_confidence: str = "low"
    active_crop_score: float = 0.0
    active_crop_evidence_hash: str = ""
    active_crop_candidates: tuple[tuple[str, float], ...] = ()
    season_key: str = ""
    stage_evidence_hash: str = ""
    tracking_reasoning_version: str = TRACKING_REASONING_VERSION


@dataclass(frozen=True)
class CurrentNeedAssessment:
    crop: str
    stage: str
    possible_needs: tuple[str, ...]
    top_need: str
    confidence: str
    evidence: tuple[str, ...]
    historical_relevance: tuple[str, ...]
    already_stocked: bool
    best_talking_point: str
    need_window: str = ""
    evidence_hash: str = ""
    tracking_reasoning_version: str = TRACKING_REASONING_VERSION


@dataclass(frozen=True)
class ProactiveContactReason:
    eligible: bool
    reason_type: str
    reason_summary: str
    confidence: float
    missing_key_variable: str
    do_not_contact_reason: str
    suggested_next_contact_at: str | None
    context_payload: dict[str, Any] = field(default_factory=dict)
    value_tier: str | None = None


@dataclass(frozen=True)
class ProactiveContactSnapshot:
    request: ProactiveContactRequest
    contact: Contact
    messages: tuple[Message | VisibleMessage, ...]
    memories: tuple[Memory, ...]
    dashboard: CustomerDashboard
    profile_latest_contact_at: str | None
    strategy_requires_manual_reply: bool
    bounded_payload: dict[str, Any]
    component_hashes: dict[str, str]
    build_ms: float
    writing_preferences: tuple[str, ...] = ()
    contact_recency: CustomerContactRecency | None = None
    stage_assessment: CurrentStageAssessment | None = None
    need_assessment: CurrentNeedAssessment | None = None
    assessment_ms: float = 0.0
    recent_observation_messages: tuple[dict[str, Any], ...] = ()
    tracking_evidence: dict[str, Any] = field(default_factory=dict)
    conversation_style: CustomerConversationStyleProfile | None = None
    current_message_view: CurrentMessageView | None = None


@dataclass(frozen=True)
class ProactiveContactOutcome:
    request: ProactiveContactRequest
    decision: ProactiveContactReason
    result: ReplyGenerationResult | None
    context_hash: str
    cache_hit: bool
    build_ms: float

    @property
    def display_mode(self) -> str:
        if (
            self.result is not None and not self.result.allowed
            and self.result.status.removeprefix("proactive:blocked:") in _PROACTIVE_SOURCE_BLOCK_REASONS
            and self.result.status.startswith("proactive:blocked:")
        ):
            # Source refusal occurs before even assessing customer eligibility.
            # Do not disguise it as an assessed non-A/B/ineligible customer.
            return "blocked"
        if not self.decision.eligible:
            return "ineligible"
        if self.result is not None and not self.result.allowed:
            return "blocked"
        if self.decision.reason_type == "do_not_contact_now":
            return "do_not_contact"
        if self.decision.context_payload.get("reply_mode_required"):
            return "reply_priority"
        if self.result is not None and self.result.allowed and len(self.result.suggestions) == 1:
            return "proactive"
        return "blocked"


@dataclass(frozen=True)
class ProactiveTaskResult:
    job_id: int
    outcome: ProactiveContactOutcome
    queued_at: str
    worker_started_at: str
    worker_completed_at: str
    operation: str = "evaluate"


@dataclass(frozen=True)
class _CacheEntry:
    outcome_json: str
    provider: str
    model: str


class CustomerContactRecencyService:
    def assess(
        self,
        snapshot: ProactiveContactSnapshot,
        *,
        now: datetime,
        silence_days: int,
    ) -> CustomerContactRecency:
        threshold = min(365, max(1, int(silence_days)))
        customer_times = [
            value
            for item in snapshot.messages
            if item.speaker == Speaker.OTHER
            for value in (_message_time(item),)
            if value is not None
        ]
        our_times = [
            value
            for item in snapshot.messages
            if item.speaker == Speaker.ME
            for value in (_message_time(item),)
            if value is not None
        ]
        profile_time = _parse_dt(snapshot.profile_latest_contact_at)
        last_customer = max(customer_times, key=lambda item: item.astimezone(timezone.utc)) if customer_times else None
        last_our = max(our_times, key=lambda item: item.astimezone(timezone.utc)) if our_times else None
        all_times = [item for item in (last_customer, last_our, profile_time) if item is not None]
        last_any = max(all_times, key=lambda item: item.astimezone(timezone.utc)) if all_times else None
        days = max(0, (now - last_any.astimezone(now.tzinfo)).days) if last_any is not None else None
        return CustomerContactRecency(
            _iso_or_none(last_customer),
            _iso_or_none(last_our),
            _iso_or_none(last_any),
            days,
            threshold,
            days is None or days >= threshold,
        )


class CropStageInferenceService:
    """Resolve one derived current stage without writing a Customer Fact."""

    def assess(self, snapshot: ProactiveContactSnapshot, *, now: datetime) -> CurrentStageAssessment:
        active_crop = ActiveCropResolver().resolve(
            evidence_payload=snapshot.tracking_evidence,
            current_year=now.year,
            remark_text=" ".join(
                item for item in (snapshot.contact.remark, snapshot.contact.display_name) if item
            ),
            contexts=snapshot.dashboard.contexts,
            recent_customer_texts=(
                item.text for item in snapshot.messages if item.speaker == Speaker.OTHER
            ),
        )
        crop = active_crop.crop
        crop_evidence = list(active_crop.evidence)
        region, region_source, region_evidence = _resolve_region(snapshot)
        calculated_at = now.isoformat()
        common = {
            "active_crop_confidence": active_crop.confidence,
            "active_crop_score": active_crop.score,
            "active_crop_evidence_hash": active_crop.evidence_hash,
            "active_crop_candidates": active_crop.candidate_scores,
            "tracking_reasoning_version": TRACKING_REASONING_VERSION,
        }
        if active_crop.ambiguous:
            return CurrentStageAssessment(
                crop="",
                stage_label="当前作物待确认",
                status="unknown",
                confidence="low",
                region_used=region,
                region_source=region_source,
                evidence=tuple(crop_evidence + region_evidence),
                calculated_at=calculated_at,
                next_window="先确认当前主要管理作物",
                source="multi_crop_unresolved",
                stage_evidence_hash=_stable_hash(crop_evidence + region_evidence),
                **common,
            )

        explicit = _latest_explicit_stage(snapshot, crop, now)
        if explicit is not None:
            stage, evidence, source = explicit
            return CurrentStageAssessment(
                crop=crop,
                stage_label=stage,
                status="confirmed",
                confidence="high",
                region_used=region,
                region_source=region_source,
                evidence=tuple(crop_evidence + region_evidence + [evidence]),
                calculated_at=calculated_at,
                next_window=_next_window_for_stage(stage),
                source=source,
                stage_evidence_hash=_stable_hash([evidence]),
                **common,
            )

        timeline = CropSeasonTimelineService().resolve(
            crop=crop,
            crop_confidence=active_crop.confidence,
            evidence_payload=snapshot.tracking_evidence,
            now=now,
            region=region,
        )
        if timeline.status != "unknown":
            return CurrentStageAssessment(
                crop=crop,
                stage_label=timeline.stage_label,
                status=timeline.status,
                confidence=timeline.confidence,
                region_used=region,
                region_source=region_source,
                evidence=tuple(crop_evidence + region_evidence + list(timeline.evidence)),
                calculated_at=calculated_at,
                next_window=timeline.next_window,
                source=timeline.source,
                season_key=timeline.season_key,
                stage_evidence_hash=timeline.evidence_hash,
                **common,
            )

        selected_context = next(
            (item for item in snapshot.dashboard.contexts if item.id == snapshot.dashboard.selected_context_id),
            None,
        )
        profile_crop = next(
            (item.value for item in snapshot.dashboard.core_fields if item.key == "crop" and item.value),
            "",
        )
        profile_stage_matches = bool(
            crop and (
                (selected_context is not None and canonicalize_crop(selected_context.crop) == canonicalize_crop(crop))
                or (
                    selected_context is None
                    and canonicalize_crop(profile_crop) == canonicalize_crop(crop)
                    and sum(
                        item.status == "active" and canonicalize_crop(item.crop) == canonicalize_crop(crop)
                        for item in snapshot.dashboard.contexts
                    ) <= 1
                    and not any(
                        item.status == "active" and item.crop
                        and canonicalize_crop(item.crop) != canonicalize_crop(crop)
                        for item in snapshot.dashboard.contexts
                    )
                )
            )
        )
        field = next(
            (
                item for item in snapshot.dashboard.core_fields
                if item.key == "growth_stage" and item.value
                and profile_stage_matches
                and item.source_type in {"manual", "structured_memory"}
            ),
            None,
        )
        if field is not None:
            source = "manual_confirmed" if field.source_type == "manual" else "high_confidence_structured_fact"
            return CurrentStageAssessment(
                crop=crop,
                stage_label=field.value.strip(),
                status="confirmed",
                confidence="high",
                region_used=region,
                region_source=region_source,
                evidence=tuple(crop_evidence + region_evidence + [f"已确认阶段：{field.value.strip()}"]),
                calculated_at=calculated_at,
                next_window=_next_window_for_stage(field.value),
                source=source,
                stage_evidence_hash=_stable_hash([field.value.strip(), source]),
                **common,
            )

        matching_contexts = [
            item for item in snapshot.dashboard.contexts
            if item.status == "active" and item.growth_stage and crop
            and canonicalize_crop(item.crop) == canonicalize_crop(crop)
        ]
        # Same-crop batches can also have different stages. Never pick the first
        # other-crop context, or silently collapse conflicting batch stages.
        active_context = matching_contexts[0] if len(matching_contexts) == 1 else None
        if active_context is not None:
            return CurrentStageAssessment(
                crop=crop or active_context.crop,
                stage_label=active_context.growth_stage.strip(),
                status="confirmed",
                confidence="high",
                region_used=region,
                region_source=region_source,
                evidence=tuple(
                    crop_evidence
                    + region_evidence
                    + [f"active Growing Context 已确认阶段：{active_context.growth_stage.strip()}"]
                ),
                calculated_at=calculated_at,
                next_window=_next_window_for_stage(active_context.growth_stage),
                source="high_confidence_structured_fact",
                stage_evidence_hash=_stable_hash([active_context.growth_stage]),
                **common,
            )

        timing = _latest_explicit_planting_timing(snapshot, crop=crop)
        if timing is not None:
            month, action, evidence = timing
            stage, next_window = _stage_from_explicit_month(month, now.month, action)
            return CurrentStageAssessment(
                crop=crop,
                stage_label=stage,
                status="inferred",
                confidence="medium",
                region_used=region,
                region_source=region_source,
                evidence=tuple(crop_evidence + region_evidence + [evidence]),
                calculated_at=calculated_at,
                next_window=next_window,
                source="customer_explicit_timing",
                season_key=f"{now.year}当季",
                stage_evidence_hash=_stable_hash([evidence]),
                **common,
            )

        seasonal = _seasonal_stage(crop, region, now.month)
        if seasonal is not None:
            stage, next_window = seasonal
            confidence = "low" if region_source == "phone_attribution" else "medium"
            return CurrentStageAssessment(
                crop=crop,
                stage_label=stage,
                status="inferred",
                confidence=confidence,
                region_used=region,
                region_source=region_source,
                evidence=tuple(crop_evidence + region_evidence + [f"按{region or '已知地区'}、{crop}与{now.month}月常见农时粗略推算"]),
                calculated_at=calculated_at,
                next_window=next_window,
                source="seasonal_crop_region",
                season_key=f"{now.year}当季",
                stage_evidence_hash=_stable_hash([crop, region, now.month]),
                **common,
            )

        return CurrentStageAssessment(
            crop=crop,
            stage_label="当前阶段待确认",
            status="unknown",
            confidence="low",
            region_used=region,
            region_source=region_source,
            evidence=tuple(crop_evidence + region_evidence),
            calculated_at=calculated_at,
            next_window="先确认今年是否仍种及当前实际阶段",
            source="insufficient_evidence",
            stage_evidence_hash=timeline.evidence_hash,
            **common,
        )


class CurrentNeedInferenceService:
    """Rank at most three possible needs; results are never persisted as facts."""

    def assess(
        self,
        snapshot: ProactiveContactSnapshot,
        stage: CurrentStageAssessment,
        *,
        now: datetime,
    ) -> CurrentNeedAssessment:
        historical = _historical_relevance(snapshot)
        stocked = _recently_stocked(snapshot, now)
        purchase_background = _current_season_purchase(snapshot, stage.crop, now)
        possible = list(_needs_for_stage(stage.stage_label))
        evidence = list(stage.evidence)
        harvest_window = _contains_any(stage.stage_label, ("采收", "收获", "采后"))
        before_harvest = "采收前" in stage.stage_label
        if before_harvest:
            possible = ["采收前长势与管理需求", "后期投入与管理安排", "采收后的种植安排"]
        elif harvest_window:
            possible = ["采收进度回访", "商品性与品质确认", "采后安排或下茬复盘"]
            if stage.crop == "红薯" and _contains_any(" ".join(historical), ("薯型", "品质")):
                possible = ["采收进度与薯块品质回访", "采后储运安排", "下茬复盘"]
        if stocked:
            possible = ["使用进展回访", "到货/备货衔接", "效果观察"]
            evidence.append("近期存在购买、下单或备货线索，避免重复推同一需求")
        elif historical:
            ranked = _rank_needs_with_history(possible, historical)
            possible = ranked or possible
        if purchase_background:
            evidence.append(purchase_background)
        if stage.status == "unknown" and stage.crop and purchase_background:
            possible = ["本季作物长势与实际管理需求", "后续用肥用药安排"]
        elif stage.status == "unknown":
            possible = ["确认当前作物与实际阶段"]
        elif not possible:
            possible = ["当前阶段管理进展"]
        possible = list(dict.fromkeys(item for item in possible if item))[:3]
        top_need = possible[0] if possible else "确认当前实际情况"
        if stocked:
            talking_point = "先回访到货或使用进展，不重复推荐同一需求"
        elif not stage.crop:
            talking_point = "先确认今年当前主要管理的作物"
        elif stage.status == "unknown" and purchase_background:
            talking_point = f"承接本季{stage.crop}相关记录中的作物用途，了解实际长势和当前管理需求；不推定已经使用或出现问题"
        elif stage.status == "unknown":
            talking_point = f"先确认{stage.crop}当前管理到哪一步"
        elif before_harvest:
            talking_point = f"客户已明确尚未采收，围绕{top_need}寻找合适产品需求，不重复询问是否采收"
        elif harvest_window:
            talking_point = f"先确认{stage.crop}是否已经开始采收，再了解品质与采后安排"
        elif stage.status == "inferred":
            talking_point = f"先确认是否已进入{_display_stage(stage.stage_label)}，再了解{top_need}"
        else:
            talking_point = f"围绕已确认的{stage.stage_label}了解{top_need}"
        confidence = "high" if stocked or stage.status == "confirmed" else stage.confidence
        return CurrentNeedAssessment(
            stage.crop,
            stage.stage_label,
            tuple(possible),
            top_need,
            confidence,
            tuple(evidence),
            tuple(historical[:3]),
            stocked,
            talking_point,
            stage.next_window,
            _stable_hash(
                {
                    "crop": stage.crop,
                    "stage": stage.stage_label,
                    "possible": possible,
                    "history": historical[:3],
                    "stocked": stocked,
                    "window": stage.next_window,
                }
            ),
            TRACKING_REASONING_VERSION,
        )


class ProactiveContactContextBuilder:
    """Build a bounded, account-scoped local snapshot for one stable DM binding."""

    def __init__(
        self,
        contacts: ContactRepository,
        messages: MessageRepository,
        memories: MemoryRepository,
        profiles: CustomerProfileService,
        strategies: StrategyRepository,
        observations: ObservationRepository | None = None,
        conversation_styles: CustomerSalutationResolver | None = None,
    ) -> None:
        self.contacts = contacts
        self.messages = messages
        self.memories = memories
        self.profiles = profiles
        self.strategies = strategies
        self.observations = observations
        self.conversation_styles = conversation_styles
        self._tracking_evidence_cache: dict[
            tuple[str, str, int, str], tuple[float, dict[str, Any]]
        ] = {}

    def build(self, request: ProactiveContactRequest) -> ProactiveContactSnapshot:
        started = monotonic_time.perf_counter()
        contact = self.contacts.get(request.contact_id)
        if contact is None or contact.account_scope_id != request.account_scope_id:
            raise ValueError("proactive_contact_scope_mismatch")
        current_view = None
        has_current_proof = bool(request.current_observation_id or request.current_session_id or request.source_job_id)
        if has_current_proof:
            if not (self.observations is not None and request.current_observation_id and request.current_session_id and request.source_job_id):
                raise ValueError("proactive_current_view_binding_incomplete")
            observation = self.observations.get(request.current_observation_id)
            conversation = ConversationRepository(self.messages.db).get(request.conversation_id)
            if observation is None or conversation is None:
                raise ValueError("proactive_current_view_source_missing")
            current_view = build_current_message_view(
                observation, conversation=conversation,
                expected_account_scope_id=request.account_scope_id, expected_contact_id=contact.id,
                expected_conversation_id=request.conversation_id, expected_target_instance_id=request.target_instance_id,
                expected_session_id=request.current_session_id, max_source_job_id=request.source_job_id,
                max_capture_generation=request.capture_generation, now=datetime.now(timezone.utc),
            )
            if not CustomerMemoryContextBuilder(self.messages.db).validate_current_view(current_view):
                raise ValueError("proactive_current_view_not_current")
        recent = tuple(
            item
            for item in self.messages.list_timeline_tail(request.conversation_id, 12)
            if item.contact_id == contact.id and item.account_scope_id == request.account_scope_id
        )
        if not recent:
            recent = tuple(
                reversed(
                    [
                        item
                        for item in self.messages.list_for_contact(contact.id, 12)
                        if item.account_scope_id == request.account_scope_id
                    ]
                )
            )
        confirmed_memories = tuple(self.memories.list_confirmed_for_contact(contact.id)[:8])
        dashboard = self.profiles.get_customer_dashboard(
            account_scope_id=request.account_scope_id,
            contact_id=contact.id,
            event_limit=12,
        )
        try:
            tracking_year = datetime.fromisoformat(request.local_date).year
        except ValueError:
            tracking_year = datetime.now().astimezone().year
        tracking_cache_key = (
            request.account_scope_id,
            contact.id,
            tracking_year,
            dashboard.revision,
        )
        cached_tracking = self._tracking_evidence_cache.get(tracking_cache_key)
        cache_now = monotonic_time.monotonic()
        if cached_tracking is not None and cache_now - cached_tracking[0] <= 5.0:
            tracking_evidence = cached_tracking[1]
        else:
            tracking_evidence = self.profiles.get_tracking_evidence(
                account_scope_id=request.account_scope_id,
                contact_id=contact.id,
                current_year=tracking_year,
                fact_limit=120,
                event_limit=160,
            )
            self._tracking_evidence_cache[tracking_cache_key] = (cache_now, tracking_evidence)
            if len(self._tracking_evidence_cache) > 64:
                self._tracking_evidence_cache.pop(next(iter(self._tracking_evidence_cache)))
        profile_record = self.profiles.profiles.get_customer_profile(
            account_scope_id=request.account_scope_id,
            contact_id=contact.id,
        )
        strategy = self.strategies.get(contact.strategy_id) if contact.strategy_id else None
        conversation_style = (
            self.conversation_styles.resolve_and_store(
                account_scope_id=request.account_scope_id,
                contact_id=contact.id,
            )
            if self.conversation_styles is not None
            else None
        )
        profile_payload = {
            "sales_stage": dashboard.sales_stage,
            "history_status": dashboard.history_status,
            "latest_contact_at": profile_record.latest_contact_at if profile_record else None,
            "core_fields": [asdict(item) for item in dashboard.core_fields[:10]],
            "contexts": [asdict(item) for item in dashboard.contexts[:3]],
            "concerns": [asdict(item) for item in dashboard.concerns[:6]],
        }
        message_payload = [
            {
                "speaker": item.speaker.value,
                "text": " ".join(item.text.split())[:500],
                "observed_at": item.observed_at,
                "message_time": item.message_time,
                "fingerprint": item.fingerprint,
            }
            for item in recent[-12:]
        ]
        if current_view is not None:
            # Visible occurrences remain VisibleMessage objects, never forged
            # formal IDs or events.  The current tail replaces stale DB text.
            recent = current_view.messages[-12:]
            message_payload = [{**item.business_payload(), "text": item.text[:500]} for item in recent]
        memory_payload = [
            {
                "id": item.id,
                "kind": item.kind.value,
                "content": " ".join(item.content.split())[:400],
                "updated_at": item.updated_at,
            }
            for item in confirmed_memories
        ]
        event_payload = [asdict(item) for item in dashboard.events[:12]]
        followup_payload = [asdict(item) for item in dashboard.followups[:8]]
        order_payload = [
            {
                "event_id": item.event_id,
                "event_type": item.event_type,
                "event_at": item.event_at,
                "product": item.product,
                "status": item.status,
            }
            for item in dashboard.order_records[:5]
        ]
        observation_payload: list[dict[str, Any]] = []
        observation_identity: dict[str, Any] = {}
        observation_business: dict[str, Any] = {}
        if current_view is not None:
            observation_payload = [{**item.business_payload(), "text": item.text[:500]} for item in current_view.messages[-20:]]
            observation_identity = {"source_kind": current_view.source_kind, "current": True}
            observation_business = current_view.business_payload()
        elif self.observations is not None:
            observation = self.observations.latest_confirmed_for_conversation(
                account_scope_id=request.account_scope_id,
                contact_id=contact.id,
                conversation_id=request.conversation_id,
            )
            if observation is not None:
                semantic = observation_semantic_payload(observation)
                if semantic is not None:
                    # This unbound observation is historical context only.
                    # Preserve ordered repetitions; do not manufacture a
                    # current-window proof from whichever row happens latest.
                    observation_payload = [{**row, "text": row["text"][:500]} for row in semantic["messages"][-20:]]
                    observation_identity = {"source_kind": "historical_visible_observation", "current": False}
                    observation_business = {"status": semantic["status"], "messages": observation_payload}
        component_hashes = {
            "profile": _stable_hash({"revision": dashboard.revision, "profile": profile_payload}),
            "memory": _stable_hash(memory_payload),
            "messages": _stable_hash(message_payload),
            "events": _stable_hash(event_payload),
            "orders": _stable_hash(order_payload),
            "followups": _stable_hash(followup_payload),
            "confirmed_observation": _stable_hash(
                observation_business
            ),
            "tracking_evidence": _stable_hash(tracking_evidence),
            "tracking_reasoning_version": _stable_hash(TRACKING_REASONING_VERSION),
            "conversation_style": _stable_hash(
                asdict(conversation_style) if conversation_style is not None else {}
            ),
            "conversation_style_version": _stable_hash(CONVERSATION_STYLE_VERSION),
        }
        bounded_payload = {
            "schema": "whochat.proactive_context.v3",
            "tracking_reasoning_version": TRACKING_REASONING_VERSION,
            "conversation_style_version": CONVERSATION_STYLE_VERSION,
            "identity": {
                "account_scope_id": request.account_scope_id,
                "contact_id": contact.id,
                "conversation_id": request.conversation_id,
                "target_instance_id": request.target_instance_id,
            },
            "local_date": request.local_date,
            "profile": profile_payload,
            "recent_messages": message_payload,
            "confirmed_memories": memory_payload,
            "events": event_payload,
            "followups": followup_payload,
            "orders": order_payload,
            "recent_confirmed_observation": {
                "identity": observation_identity,
                "messages": observation_payload,
            },
            "tracking_evidence": tracking_evidence,
            "conversation_style": (
                asdict(conversation_style) if conversation_style is not None else {
                    "preferred_salutation": "",
                    "salutation_confidence": "low",
                    "prefers_no_salutation": False,
                    "style_version": CONVERSATION_STYLE_VERSION,
                }
            ),
            "limits": {
                "recent_messages": 12,
                "confirmed_memories": 8,
                "events": 12,
                "followups": 8,
                "orders": 5,
                "recent_confirmed_observation_messages": 20,
                "tracking_facts": 120,
                "tracking_events": 160,
            },
        }
        return ProactiveContactSnapshot(
            request=request,
            contact=contact,
            messages=recent[-12:],
            memories=confirmed_memories,
            dashboard=dashboard,
            profile_latest_contact_at=(profile_record.latest_contact_at if profile_record else None),
            strategy_requires_manual_reply=bool(strategy and strategy.requires_manual_reply),
            bounded_payload=bounded_payload,
            component_hashes=component_hashes,
            build_ms=round((monotonic_time.perf_counter() - started) * 1000.0, 3),
            recent_observation_messages=tuple(observation_payload),
            tracking_evidence=tracking_evidence,
            conversation_style=conversation_style,
            current_message_view=current_view,
        )


class ProactiveContactReasonEngine:
    """Deterministic local contact-reason engine; it never calls an AI provider."""

    def __init__(self, parser: CustomerRemarkTagParser | None = None) -> None:
        self.parser = parser or CustomerRemarkTagParser()

    def evaluate(
        self,
        snapshot: ProactiveContactSnapshot,
        *,
        now: datetime | None = None,
    ) -> ProactiveContactReason:
        now = now or datetime.now().astimezone()
        snapshot = _with_activation_assessments(
            snapshot,
            now=now,
            silence_days=(
                snapshot.contact_recency.silence_threshold_days
                if snapshot.contact_recency is not None
                else 7
            ),
        )
        eligibility = evaluate_proactive_eligibility(snapshot.contact, self.parser)
        tier = eligibility.value_tier
        stage = snapshot.stage_assessment
        need = snapshot.need_assessment
        recency = snapshot.contact_recency
        base_payload: dict[str, Any] = {
            "value_tier": tier,
            "parsed_tags": eligibility.parsed_tags,
            "tag_parse_status": eligibility.parse_status,
            "eligibility_reason": eligibility.reason,
            "reply_mode_required": False,
            "current_view_revision": snapshot.current_message_view.content_revision if snapshot.current_message_view is not None else "",
            "recommended_need": need.top_need if need else "",
            "recommended_question": _activation_question(snapshot),
            "evidence_refs": [],
            "contact_recency": asdict(recency) if recency else {},
            "current_stage": asdict(stage) if stage else {},
            "current_need": asdict(need) if need else {},
            "stage_inference_hash": _stable_hash(asdict(stage)) if stage else "",
            "need_inference_hash": _stable_hash(asdict(need)) if need else "",
            "talking_point_hash": _stable_hash(need.best_talking_point) if need else "",
            "conversation_style": (
                asdict(snapshot.conversation_style)
                if snapshot.conversation_style is not None
                else {
                    "preferred_salutation": "",
                    "salutation_confidence": "low",
                    "prefers_no_salutation": False,
                    "style_version": CONVERSATION_STYLE_VERSION,
                }
            ),
        }
        if eligibility.is_worthless:
            return ProactiveContactReason(
                False,
                "ineligible",
                "显式无价值标记覆盖 A/B 资格",
                1.0,
                "",
                "客户存在 S、独立 0 或“无价值”标记",
                None,
                base_payload,
                tier,
            )
        if (
            snapshot.contact.status != ContactStatus.CONFIRMED
            or snapshot.contact.merged_into
            or snapshot.strategy_requires_manual_reply
        ):
            return ProactiveContactReason(
                False, "ineligible", "当前身份不可用或处于人工回复保护", 1.0,
                "", "contact_or_manual_protection", None, base_payload, tier,
            )
        current_view = snapshot.current_message_view
        if current_view is not None:
            if not current_view.validate_binding(
                expected_account_scope_id=snapshot.request.account_scope_id,
                expected_contact_id=snapshot.request.contact_id,
                expected_conversation_id=snapshot.request.conversation_id,
                expected_target_instance_id=snapshot.request.target_instance_id,
                expected_session_id=snapshot.request.current_session_id,
                max_source_job_id=snapshot.request.source_job_id,
                max_capture_generation=snapshot.request.capture_generation, now=now,
            ):
                return ProactiveContactReason(
                    False, "ineligible", "当前窗口证据已失效", 1.0,
                    "", "current_view_not_current", None, base_payload, tier,
                )
            if current_view.has_unanswered_customer_message:
                base_payload.update({
                    "reply_mode_required": True,
                    "recommended_need": "优先处理当前窗口中的客户消息",
                    "recommended_question": "",
                    "current_view_revision": current_view.content_revision,
                    "current_context_source": current_view.source_kind,
                })
                # eligible here means Reply dispatch only.  The independent
                # Proactive cloud gate still requires A/B and refuses Reply.
                return ProactiveContactReason(
                    True, "reply_obligation", "当前窗口有待处理的客户回应，应走 Reply", 1.0,
                    "", "", snapshot.request.local_date, base_payload, tier,
                )
        if not eligibility.eligible:
            return ProactiveContactReason(
                False,
                "ineligible",
                "当前客户不是可确认的 A/B 价值层级",
                1.0,
                "value_tier",
                "非 A/B 客户，本次只做本地提示且不调用云端 AI",
                None,
                base_payload,
                tier,
            )

        messages = list(snapshot.messages)
        latest = messages[-1] if messages else None
        latest_dt = _message_time(latest)
        latest_contact_dt = _parse_dt(recency.last_any_contact_at) if recency else None
        latest_text = latest.text.strip() if latest else ""
        local_day = now.date()
        latest_incoming_age = (
            now - latest_dt.astimezone(now.tzinfo)
            if latest_dt is not None
            else None
        )
        if (
            latest
            and latest.speaker == Speaker.OTHER
            and latest_incoming_age is not None
            and -timedelta(minutes=5) <= latest_incoming_age <= timedelta(days=1)
            and not any(
                item.speaker == Speaker.ME for item in messages[messages.index(latest) + 1 :]
            )
        ):
            base_payload.update(
                {
                    "reply_mode_required": True,
                    "recommended_need": "优先回复客户的新消息",
                    "recommended_question": "",
                }
            )
            return ProactiveContactReason(
                True,
                "reply_obligation",
                "存在尚未由我方后续消息覆盖的客户来信，应优先走 Reply",
                1.0,
                "",
                "",
                snapshot.request.local_date,
                base_payload,
                tier,
            )

        refusal = _current_sales_refusal(snapshot, now)
        if refusal is not None:
            base_payload["sales_round_status"] = "waiting_for_new_need"
            base_payload["sales_round_reason"] = refusal
            base_payload["recommended_question"] = ""
            return _do_not_contact(tier, base_payload, refusal, None)

        due_followups = [
            item
            for item in snapshot.dashboard.followups
            if _followup_is_currently_actionable(item.due_at, now)
        ]
        reply_due = next(
            (
                item
                for item in due_followups
                if _contains_any(f"{item.followup_type} {item.title} {item.reason}", _REPLY_OBLIGATION_WORDS)
            ),
            None,
        )
        if reply_due is not None:
            question, missing = _question_for_snapshot(snapshot)
            base_payload.update(
                {
                    "recommended_need": "完成已到期的明确回复或资料承诺",
                    "recommended_question": question,
                    "evidence_refs": [reply_due.id],
                }
            )
            return ProactiveContactReason(
                True,
                "reply_obligation",
                "存在已到期的明确回复义务",
                0.96,
                missing,
                "",
                snapshot.request.local_date,
                base_payload,
                tier,
            )

        future_followup = next(
            (item for item in snapshot.dashboard.followups if (_parse_dt(item.due_at) or now) > now),
            None,
        )
        if _contains_any(latest_text, _ASK_LATER_WORDS):
            next_at = future_followup.due_at if future_followup else None
            return _do_not_contact(tier, base_payload, "客户已表示稍后再联系", next_at)
        if (
            latest
            and latest.speaker == Speaker.OTHER
            and _contains_any(latest_text, _POSITIVE_ACK_WORDS)
            and latest_dt is not None
            and latest_dt.astimezone(now.tzinfo).date() == local_day
        ):
            return _do_not_contact(tier, base_payload, "客户刚完成正向确认，无需立即追加销售消息", None)

        recent_purchase = _latest_event(snapshot.dashboard, {"order_placed", "purchased"})
        if recent_purchase is not None and _age_days(recent_purchase.event_at, now) <= 2:
            return _do_not_contact(
                tier,
                base_payload,
                "客户刚下单或购买，先等待履约/使用节点",
                (now.date() + timedelta(days=3)).isoformat(),
            )
        if latest_contact_dt is not None and latest_contact_dt.astimezone(now.tzinfo).date() == local_day:
            return _do_not_contact(tier, base_payload, "今天已经联系过该客户", None)

        due = due_followups[0] if due_followups else None
        if due is not None:
            text = f"{due.followup_type} {due.title} {due.reason}"
            if _contains_any(text, _ORDER_WORDS):
                reason_type = "order_followup"
                need = "核对订单履约或到货后的使用安排"
            elif _contains_any(text, _USAGE_WORDS):
                reason_type = "usage_followup"
                need = "了解方案是否开始使用以及使用环节"
            else:
                reason_type = "effect_followup"
                need = "回访已有方案或使用后的变化"
            question, missing = _question_for_snapshot(snapshot, reason_type)
            base_payload.update(
                {
                    "recommended_need": need,
                    "recommended_question": question,
                    "evidence_refs": [due.id],
                }
            )
            return ProactiveContactReason(
                True,
                reason_type,
                f"存在已到期的{need}",
                0.92,
                missing,
                "",
                snapshot.request.local_date,
                base_payload,
                tier,
            )

        event_reason = _event_followup_reason(snapshot, now)
        if event_reason is not None:
            reason_type, summary, event_id = event_reason
            question, missing = _question_for_snapshot(snapshot, reason_type)
            base_payload.update(
                {
                    "recommended_need": summary,
                    "recommended_question": question,
                    "evidence_refs": [event_id],
                }
            )
            return ProactiveContactReason(
                True,
                reason_type,
                summary,
                0.84,
                missing,
                "",
                snapshot.request.local_date,
                base_payload,
                tier,
            )

        if recency is not None and not recency.silence_candidate:
            return _do_not_contact(
                tier,
                base_payload,
                f"距上次联系不足配置阈值 {recency.silence_threshold_days} 天，暂不重复打扰",
                (
                    latest_contact_dt.astimezone(now.tzinfo)
                    + timedelta(days=recency.silence_threshold_days)
                ).isoformat() if latest_contact_dt is not None else None,
            )

        predicted = _predicted_window(snapshot, now)
        if predicted is not None:
            summary, question, missing, evidence_ref = predicted
            base_payload.update(
                {
                    "recommended_need": need.top_need if need else summary,
                    "recommended_question": question,
                    "evidence_refs": [evidence_ref] if evidence_ref else [],
                }
            )
            return ProactiveContactReason(
                True,
                "predicted_window",
                summary,
                0.88 if stage and stage.status == "confirmed" else 0.74,
                missing,
                "",
                snapshot.request.local_date,
                base_payload,
                tier,
            )

        question = _activation_question(snapshot)
        missing = "crop" if not stage or not stage.crop else "growth_stage"
        base_payload.update(
            {
                "recommended_need": need.top_need if need else "以低承诺方式重新了解当前种植安排",
                "recommended_question": question,
            }
        )
        return ProactiveContactReason(
            True,
            "reactivation",
            "当前没有更高优先级义务，适合做一次低打扰激活",
            0.66 if messages else 0.58,
            missing,
            "",
            snapshot.request.local_date,
            base_payload,
            tier,
        )


class ProactiveGenerator(Protocol):
    def generate(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
        config: AppConfig,
    ) -> ReplyGenerationResult: ...


class ProactiveContactGenerator:
    """Dedicated proactive prompt/provider path; it cannot send WeChat messages."""

    def generate(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
        config: AppConfig,
    ) -> ReplyGenerationResult:
        blocked = _generation_block(snapshot, decision, config)
        if blocked:
            return ReplyGenerationResult(False, f"proactive:blocked:{blocked}", [], config.ai.provider)
        if config.ai.provider in {"OpenAI", "OpenAI Compatible"} and config.ai.api_key:
            return self._generate_cloud(snapshot, decision, config)
        if config.ai.provider == "Disabled":
            return ReplyGenerationResult(False, "proactive:blocked:provider_disabled", [], config.ai.provider)
        return _local_proactive_preview(snapshot, decision)

    def generate_with_feedback(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
        config: AppConfig,
        reason_code: str,
    ) -> ReplyGenerationResult:
        blocked = _generation_block(snapshot, decision, config)
        if blocked:
            return ReplyGenerationResult(False, f"proactive:blocked:{blocked}", [], config.ai.provider)
        if config.ai.provider in {"OpenAI", "OpenAI Compatible"} and config.ai.api_key:
            return self._generate_cloud(
                snapshot,
                decision,
                config,
                regeneration_reason=reason_code,
            )
        if config.ai.provider == "Disabled":
            return ReplyGenerationResult(False, "proactive:blocked:provider_disabled", [], config.ai.provider)
        return _local_proactive_preview(snapshot, decision)

    def _generate_cloud(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
        config: AppConfig,
        *,
        regeneration_reason: str = "",
    ) -> ReplyGenerationResult:
        endpoint = _chat_completions_endpoint(config.ai.base_url)
        system_prompt, user_prompt = _proactive_prompt(
            snapshot,
            decision,
            regeneration_reason=regeneration_reason,
        )
        payload = {
            "model": config.ai.model,
            "temperature": min(max(config.ai.temperature, 0.0), 0.7),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=encoded,
            headers={
                "Authorization": f"Bearer {config.ai.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "WhoChat/0.1 Proactive-Contact",
            },
            method="POST",
        )
        started = monotonic_time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=config.ai.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            _safe_provider_log(endpoint, started, f"http_error:{exc.code}")
            return ReplyGenerationResult(False, f"proactive:provider_http_error:{exc.code}", [], config.ai.provider)
        except (urllib.error.URLError, TimeoutError) as exc:
            _safe_provider_log(endpoint, started, f"transport_error:{type(exc).__name__}")
            return ReplyGenerationResult(False, f"proactive:provider_transport_error:{type(exc).__name__}", [], config.ai.provider)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _safe_provider_log(endpoint, started, "bad_response_json")
            return ReplyGenerationResult(False, "proactive:provider_bad_response_json", [], config.ai.provider)
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            _safe_provider_log(endpoint, started, "bad_response_shape")
            return ReplyGenerationResult(False, "proactive:provider_bad_response_shape", [], config.ai.provider)
        items = parsed.get("messages") if isinstance(parsed, dict) else None
        if not isinstance(items, list) or len(items) != 1 or any(not isinstance(item, dict) for item in items):
            return ReplyGenerationResult(False, "proactive:invalid_exact_one", [], config.ai.provider)
        suggestions = [
            ReplySuggestion(
                label="激活建议",
                text=str(item.get("text") or "").strip(),
                risk="low",
                rationale=str(item.get("rationale") or "稳妥、低承诺、只追问一个变量").strip(),
            )
            for index, item in enumerate(items)
        ]
        validation_error = _validate_proactive_suggestions(suggestions)
        if validation_error:
            return ReplyGenerationResult(False, f"proactive:invalid_output:{validation_error}", [], config.ai.provider)
        tracking_error = _validate_tracking_suggestions(snapshot, suggestions)
        if tracking_error:
            return ReplyGenerationResult(False, f"proactive:invalid_output:{tracking_error}", [], config.ai.provider)
        _safe_provider_log(endpoint, started, "parsed")
        return _proactive_result(snapshot, decision, suggestions, config.ai.provider, "proactive:ai_generated")


class ProactiveContactCacheRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        local_date: str,
        context_hash: str,
    ) -> _CacheEntry | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT outcome_json,provider,model
                FROM proactive_contact_cache
                WHERE account_scope_id=? AND contact_id=? AND local_date=? AND context_hash=?
                """,
                (account_scope_id, contact_id, local_date, context_hash),
            ).fetchone()
        return _CacheEntry(str(row["outcome_json"]), str(row["provider"]), str(row["model"])) if row else None

    def save(
        self,
        outcome: ProactiveContactOutcome,
        *,
        provider: str,
        model: str,
    ) -> None:
        now = utc_now_iso()
        payload = _outcome_to_json(outcome)
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO proactive_contact_cache(
                    id,account_scope_id,contact_id,local_date,context_hash,reason_type,
                    decision_status,outcome_json,provider,model,suggestion_count,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(account_scope_id,contact_id,local_date,context_hash)
                DO UPDATE SET
                    reason_type=excluded.reason_type,
                    decision_status=excluded.decision_status,
                    outcome_json=excluded.outcome_json,
                    provider=excluded.provider,
                    model=excluded.model,
                    suggestion_count=excluded.suggestion_count,
                    updated_at=excluded.updated_at
                """,
                (
                    new_id("proactive"),
                    outcome.request.account_scope_id,
                    outcome.request.contact_id,
                    outcome.request.local_date,
                    outcome.context_hash,
                    outcome.decision.reason_type,
                    outcome.display_mode,
                    payload,
                    provider,
                    model,
                    len(outcome.result.suggestions) if outcome.result else 0,
                    now,
                    now,
                ),
            )


class ProactiveContactService:
    def __init__(
        self,
        builder: ProactiveContactContextBuilder,
        cache: ProactiveContactCacheRepository,
        generation_logs: GenerationLogRepository,
        logs: LogRepository,
        *,
        engine: ProactiveContactReasonEngine | None = None,
        generator: ProactiveGenerator | None = None,
        feedback: ProactiveSuggestionFeedbackService | None = None,
        writing_preferences: SalesWritingPreferenceService | None = None,
        identity_lease_validator: Callable[[ProactiveContactRequest], bool] | None = None,
        current_runtime_provider: Callable[[], Any] | None = None,
    ) -> None:
        self.builder = builder
        self.cache = cache
        self.generation_logs = generation_logs
        self.logs = logs
        self.engine = engine or ProactiveContactReasonEngine()
        self.generator = generator or ProactiveContactGenerator()
        self.feedback = feedback
        self.writing_preferences = writing_preferences
        self.identity_lease_validator = identity_lease_validator
        self.current_runtime_provider = current_runtime_provider
        self._last_cloud_attempt_at: datetime | None = None

    def evaluate(self, request: ProactiveContactRequest, config: AppConfig) -> ProactiveContactOutcome:
        started = monotonic_time.perf_counter()
        try:
            snapshot = self._build_snapshot(request, config)
        except ValueError as exc:
            reason = str(exc)
            if reason not in _PROACTIVE_SOURCE_BLOCK_REASONS:
                raise
            return self._source_blocked_request(request, config, reason, started)
        source_error = self._snapshot_source_error(snapshot)
        if source_error:
            return self._source_blocked_request(request, config, source_error, started)
        decision = self.engine.evaluate(snapshot)
        snapshot = self._with_writing_preferences(snapshot, decision)
        context_hash = _proactive_context_hash(snapshot, decision, config)
        source_error = self._snapshot_source_error(snapshot)
        if source_error:
            return self._blocked_snapshot_outcome(snapshot, decision, context_hash, config, source_error)
        cached = self.cache.get(
            account_scope_id=request.account_scope_id,
            contact_id=request.contact_id,
            local_date=request.local_date,
            context_hash=context_hash,
        )
        if cached is not None:
            restored = _outcome_from_json(cached.outcome_json, request, context_hash, snapshot.build_ms)
            if restored is not None:
                source_error = self._snapshot_source_error(snapshot)
                if source_error:
                    return self._blocked_snapshot_outcome(snapshot, decision, context_hash, config, source_error)
                self._audit_decision(restored)
                return restored

        if not decision.eligible:
            # Eligibility gates are not business contact reasons and are not
            # persisted as do-not-contact cache entries.  This prevents an old
            # parser miss from pinning a customer to a misleading daily result.
            outcome = ProactiveContactOutcome(request, decision, None, context_hash, False, snapshot.build_ms)
            source_error = self._snapshot_source_error(snapshot)
            if source_error:
                return self._blocked_snapshot_outcome(snapshot, decision, context_hash, config, source_error)
            self._audit_decision(outcome)
            return outcome
        if decision.reason_type == "do_not_contact_now" or decision.context_payload.get("reply_mode_required"):
            outcome = ProactiveContactOutcome(request, decision, None, context_hash, False, snapshot.build_ms)
            source_error = self._snapshot_source_error(snapshot)
            if source_error:
                return self._blocked_snapshot_outcome(snapshot, decision, context_hash, config, source_error)
            self.cache.save(outcome, provider="Local Reason Engine", model="v1")
            self._audit_decision(outcome)
            return outcome

        previous_attempt = self._last_cloud_attempt_at
        try:
            result = self._generate_with_policy(snapshot, decision, config)
        except Exception as exc:
            if self._last_cloud_attempt_at is not previous_attempt:
                self._audit_generation(snapshot, decision, ReplyGenerationResult(
                    False, f"attempted:exception:{type(exc).__name__}", [], config.ai.provider
                ), context_hash, config)
            raise
        return self._finish_generated_snapshot(
            snapshot, decision, result, context_hash, config,
            cloud_dispatched=self._last_cloud_attempt_at is not previous_attempt,
        )

    def regenerate(
        self,
        request: ProactiveContactRequest,
        config: AppConfig,
        feedback_id: str,
    ) -> ProactiveContactOutcome:
        """Run the one explicit feedback regeneration allowed for this context."""
        started = monotonic_time.perf_counter()
        try:
            snapshot = self._build_snapshot(request, config)
        except ValueError as exc:
            reason = str(exc)
            if reason not in _PROACTIVE_SOURCE_BLOCK_REASONS:
                raise
            return self._source_blocked_request(request, config, reason, started)
        source_error = self._snapshot_source_error(snapshot)
        if source_error:
            return self._source_blocked_request(request, config, source_error, started)
        decision = self.engine.evaluate(snapshot)
        snapshot = self._with_writing_preferences(snapshot, decision)
        context_hash = _proactive_context_hash(snapshot, decision, config)
        source_error = self._snapshot_source_error(snapshot)
        if source_error:
            return self._blocked_snapshot_outcome(snapshot, decision, context_hash, config, source_error)
        if self.feedback is None:
            result = ReplyGenerationResult(False, "proactive:blocked:feedback_unavailable", [], config.ai.provider)
            return ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)
        blocked = _generation_block(snapshot, decision, config)
        if blocked:
            result = ReplyGenerationResult(False, f"proactive:blocked:{blocked}", [], config.ai.provider)
            return ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)
        if not decision.eligible or decision.reason_type == "do_not_contact_now":
            result = ReplyGenerationResult(False, "proactive:blocked:reason_not_generatable", [], config.ai.provider)
            return ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)
        if decision.context_payload.get("reply_mode_required"):
            result = ReplyGenerationResult(False, "proactive:blocked:reply_priority", [], config.ai.provider)
            return ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)

        if self.identity_lease_validator is not None and not self.identity_lease_validator(request):
            result = ReplyGenerationResult(False, "proactive:blocked:identity_stability_lease", [], config.ai.provider)
            return ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)

        cloud = config.ai.provider in {"OpenAI", "OpenAI Compatible"} and bool(config.ai.api_key)
        now = datetime.now(timezone.utc)
        if cloud and self.identity_lease_validator is not None and not self.identity_lease_validator(request):
            result = ReplyGenerationResult(
                False,
                "proactive:blocked:identity_stability_lease",
                [],
                config.ai.provider,
            )
            return ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)
        if cloud and config.ai.max_daily_cloud_requests:
            day_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc).isoformat()
            if self.generation_logs.count_cloud_attempts_since(day_start) >= config.ai.max_daily_cloud_requests:
                result = ReplyGenerationResult(False, "proactive:blocked:daily_limit", [], config.ai.provider)
                return ProactiveContactOutcome(request, decision, result, context_hash, False, snapshot.build_ms)

        reason_code = self.feedback.repository.claim_regeneration(
            feedback_id,
            account_scope_id=request.account_scope_id,
            contact_id=request.contact_id,
            local_date=request.local_date,
            context_hash=context_hash,
        )
        source_error = self._snapshot_source_error(snapshot)
        if source_error:
            return self._blocked_snapshot_outcome(snapshot, decision, context_hash, config, source_error)
        if cloud:
            # This is an explicit one-shot user action.  It is still subject to
            # the daily limit and database hard cap, but not the automatic
            # scheduler cooldown that would otherwise consume the only try.
            self._last_cloud_attempt_at = now
        generate_with_feedback = getattr(self.generator, "generate_with_feedback", None)
        try:
            if callable(generate_with_feedback):
                result = generate_with_feedback(snapshot, decision, config, reason_code)
            else:
                result = self.generator.generate(snapshot, decision, config)
        except Exception as exc:
            if cloud:
                self._audit_generation(snapshot, decision, ReplyGenerationResult(
                    False, f"attempted:exception:{type(exc).__name__}", [], config.ai.provider
                ), context_hash, config)
            raise
        return self._finish_generated_snapshot(snapshot, decision, result, context_hash, config, cloud_dispatched=cloud)

    def _snapshot_source_error(self, snapshot: ProactiveContactSnapshot) -> str | None:
        view, request = snapshot.current_message_view, snapshot.request
        try:
            if view is not None:
                if view.source_kind == "current_window_reactivation":
                    return "current_window_reactivation_only"
                if not callable(self.identity_lease_validator):
                    return "current_view_live_lease_required"
                if view.screen_evidence_json:
                    if not callable(self.current_runtime_provider):
                        return "current_view_runtime_lease_required"
                    if not view.current_screen_runtime_matches(self.current_runtime_provider()):
                        return "current_view_runtime_lifecycle_mismatch"
                if not view.validate_binding(
                    expected_account_scope_id=request.account_scope_id, expected_contact_id=request.contact_id,
                    expected_conversation_id=request.conversation_id, expected_target_instance_id=request.target_instance_id,
                    expected_session_id=request.current_session_id, max_source_job_id=request.source_job_id,
                    max_capture_generation=request.capture_generation, now=datetime.now(timezone.utc),
                ) or not CustomerMemoryContextBuilder(self.builder.messages.db).validate_current_view(view):
                    return "current_view_not_current"
            elif request.current_observation_id or request.current_session_id or request.source_job_id:
                return "current_view_missing"
            if self.identity_lease_validator is not None and not self.identity_lease_validator(request):
                return "identity_stability_lease"
        except Exception:
            return "current_source_unavailable"
        return None

    @staticmethod
    def _source_blocked_request(request, config, reason, started):
        # A failed proof never reaches reason inference, feedback claims,
        # cache reads/writes or the model. The original request remains the
        # complete owner binding for the task's terminal state and result.
        decision = ProactiveContactReason(
            False, "do_not_contact_now", "当前聊天来源尚未通过校验", 1.0, "",
            "当前可见内容尚未通过来源校验，未进行客户需求或联系资格判断", None,
            {"source_blocked": True, "source_error": reason},
        )
        result = ReplyGenerationResult(False, "proactive:blocked:" + reason, [], config.ai.provider)
        return ProactiveContactOutcome(
            request, decision, result, "", False,
            round((monotonic_time.perf_counter() - started) * 1000.0, 3),
        )

    @staticmethod
    def _blocked_snapshot_outcome(snapshot, decision, context_hash, config, reason):
        result = ReplyGenerationResult(False, f"proactive:blocked:{reason}", [], config.ai.provider)
        return ProactiveContactOutcome(snapshot.request, decision, result, context_hash, False, snapshot.build_ms)

    def _finish_generated_snapshot(self, snapshot, decision, result, context_hash, config, *, cloud_dispatched):
        source_error = self._snapshot_source_error(snapshot)
        visible_result = (
            self._blocked_snapshot_outcome(snapshot, decision, context_hash, config, source_error).result
            if source_error else result
        )
        audit_result = visible_result
        if cloud_dispatched and (source_error or visible_result.status.startswith("proactive:blocked:")):
            # Display/source revocation does not cancel or refund an already
            # dispatched cloud attempt.  Keep it in the local attempt ledger.
            audit_result = replace(visible_result, status="attempted:" + visible_result.status)
        elif not cloud_dispatched and visible_result.status.startswith("proactive:blocked:"):
            # A refusal before the provider entry is not an HTTP attempt.
            audit_result = replace(visible_result, status="blocked:" + visible_result.status)
        self._audit_generation(snapshot, decision, audit_result, context_hash, config)
        outcome = ProactiveContactOutcome(snapshot.request, decision, visible_result, context_hash, False, snapshot.build_ms)
        if not source_error and result.allowed and len(result.suggestions) == 1:
            self.cache.save(outcome, provider=result.provider, model=config.ai.model)
        self._audit_decision(outcome)
        return outcome

    def _build_snapshot(self, request: ProactiveContactRequest, config: AppConfig) -> ProactiveContactSnapshot:
        snapshot = self.builder.build(request)
        now = _request_local_datetime(request.local_date)
        return _with_activation_assessments(
            snapshot,
            now=now,
            silence_days=config.proactive.silence_days,
            force=True,
        )

    def _with_writing_preferences(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
    ) -> ProactiveContactSnapshot:
        if self.writing_preferences is None:
            return snapshot
        summaries = self.writing_preferences.stable_summaries(
            snapshot.request.account_scope_id,
            decision.reason_type,
        )
        return replace(snapshot, writing_preferences=summaries)

    def _generate_with_policy(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
        config: AppConfig,
    ) -> ReplyGenerationResult:
        source_error = self._snapshot_source_error(snapshot)
        if source_error:
            return ReplyGenerationResult(False, f"proactive:blocked:{source_error}", [], config.ai.provider)
        blocked = _generation_block(snapshot, decision, config)
        if blocked:
            return ReplyGenerationResult(False, f"proactive:blocked:{blocked}", [], config.ai.provider)
        cloud = config.ai.provider in {"OpenAI", "OpenAI Compatible"} and bool(config.ai.api_key)
        now = datetime.now(timezone.utc)
        if cloud:
            if self.identity_lease_validator is not None and not self.identity_lease_validator(snapshot.request):
                return ReplyGenerationResult(
                    False,
                    "proactive:blocked:identity_stability_lease",
                    [],
                    config.ai.provider,
                )
            cooldown = max(0, config.ai.request_cooldown_seconds)
            if self._last_cloud_attempt_at and (now - self._last_cloud_attempt_at).total_seconds() < cooldown:
                return ReplyGenerationResult(False, "proactive:blocked:cooldown", [], config.ai.provider)
            if config.ai.max_daily_cloud_requests:
                day_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc).isoformat()
                if self.generation_logs.count_cloud_attempts_since(day_start) >= config.ai.max_daily_cloud_requests:
                    return ReplyGenerationResult(False, "proactive:blocked:daily_limit", [], config.ai.provider)
            self._last_cloud_attempt_at = now
        return self.generator.generate(snapshot, decision, config)

    def _audit_generation(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
        result: ReplyGenerationResult,
        context_hash: str,
        config: AppConfig,
    ) -> None:
        self.generation_logs.append(
            contact_id=snapshot.contact.id,
            strategy_id=snapshot.contact.strategy_id,
            provider=result.provider,
            model=config.ai.model if config.ai.provider != "Disabled" else "",
            allowed=result.allowed,
            status=result.status,
            suggestion_count=len(result.suggestions),
            risk_summary=f"low:{len(result.suggestions)}" if result.suggestions else "",
            context_hash=context_hash,
            page_type="chat_dm",
            page_confidence=1.0,
            message_count=len(snapshot.messages),
            memory_count=len(snapshot.memories),
        )

    def _audit_decision(self, outcome: ProactiveContactOutcome) -> None:
        safe_ref = hashlib.sha256(
            f"{outcome.request.account_scope_id}|{outcome.request.contact_id}".encode("utf-8")
        ).hexdigest()[:12]
        reason_hash = _stable_hash(asdict(outcome.decision))
        provider_called = bool(
            outcome.result is not None
            and outcome.result.provider in {"OpenAI", "OpenAI Compatible"}
            and outcome.result.status == "proactive:ai_generated"
        )
        suggestion_count = len(outcome.result.suggestions) if outcome.result else 0
        generated_at = utc_now_iso()
        self.logs.append(
            "info",
            "proactive_contact",
            "decision",
            (
                f"safe_contact={safe_ref} reason={outcome.decision.reason_type} "
                f"mode={outcome.display_mode} cache={int(outcome.cache_hit)} "
                f"provider_called={int(provider_called)} suggestions={suggestion_count} "
                f"context={outcome.context_hash[:12]} reason_hash={reason_hash[:12]} "
                f"build_ms={outcome.build_ms:.3f}"
            ),
            {
                "safe_contact_reference": safe_ref,
                "account_scope_id": outcome.request.account_scope_id,
                "reason_type": outcome.decision.reason_type,
                "reason_hash": reason_hash,
                "display_mode": outcome.display_mode,
                "context_hash": outcome.context_hash,
                "cache_hit": outcome.cache_hit,
                "provider_called": provider_called,
                "suggestion_count": suggestion_count,
                "generated_at": generated_at,
                "build_ms": outcome.build_ms,
            },
        )


@dataclass(frozen=True)
class ProactiveTaskState:
    job_id: int
    status: str
    terminal: bool
    reason: str
    request: ProactiveContactRequest
    result_available: bool = False


@dataclass
class _ProactiveJob:
    job_id: int
    request: ProactiveContactRequest
    config: AppConfig
    queued_at: str
    deadline_at: float
    operation: str = "evaluate"
    feedback_id: str = ""
    terminal: bool = False


class ProactiveContactTaskService(QObject):
    result_ready = Signal(object)
    result_discarded = Signal(str)
    status_changed = Signal(str)
    task_state_changed = Signal(object)

    def __init__(
        self,
        service: ProactiveContactService,
        executor: ThreadPoolExecutor | None = None,
        parent: QObject | None = None,
        *,
        task_timeout_seconds: float | None = None,
    ) -> None:
        from math import isfinite
        from threading import RLock
        from PySide6.QtCore import QCoreApplication, QTimer

        super().__init__(parent)
        if task_timeout_seconds is not None and (not isfinite(task_timeout_seconds) or task_timeout_seconds <= 0):
            raise ValueError("task_timeout_seconds_must_be_finite_and_positive")
        self.service = service
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="whochat-proactive")
        self._lock = RLock()
        self._job_id = 0
        self._latest_job_id = 0
        self._running: Future | None = None
        self._active: _ProactiveJob | None = None
        self._running_key = ""
        self._completed_key = ""
        self._closed = False
        self._task_timeout_seconds = task_timeout_seconds
        self._clock = monotonic_time.monotonic
        self.last_status = "idle"
        self.last_state: ProactiveTaskState | None = None
        self._deadline_timer = QTimer(self)
        self._deadline_timer.setInterval(250)
        self._deadline_timer.timeout.connect(self.check_timeouts)
        if QCoreApplication.instance() is not None:
            self._deadline_timer.start()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def running_key(self) -> str:
        with self._lock:
            return self._running_key if self._active is not None else ""

    def submit(self, request: ProactiveContactRequest, config: AppConfig) -> int | None:
        return self._submit(request, config)

    def submit_regeneration(
        self,
        request: ProactiveContactRequest,
        config: AppConfig,
        feedback_id: str,
    ) -> int | None:
        return self._submit(request, config, "regenerate", feedback_id)

    def _submit(
        self, request: ProactiveContactRequest, config: AppConfig,
        operation: str = "evaluate", feedback_id: str = "",
    ) -> int | None:
        with self._lock:
            if self._closed:
                self._reject(request, "cancelled", "shutdown")
                return None
            if operation == "evaluate" and request.binding_key == self._completed_key:
                self._reject(request, "duplicate", "duplicate_suppressed")
                return None
            if self._active is not None:
                # Deliberately no proactive queue/retry: current Reply work
                # owns the priority.  Rejection does not invalidate its peer.
                self._reject(request, "busy", "worker_busy")
                return None
            try:
                frozen_request, frozen_config = deepcopy(request), deepcopy(config)
                timeout = self._task_timeout_seconds
                if timeout is None:
                    from math import isfinite
                    configured = float(frozen_config.ai.timeout_seconds)
                    timeout = configured + 5 if isfinite(configured) and configured >= 0 else 65.0
            except Exception as exc:
                self._reject(request, "failed", "input_snapshot:" + type(exc).__name__)
                return None
            job = _ProactiveJob(
                self._next_token(), frozen_request, frozen_config, utc_now_iso(),
                self._clock() + timeout, operation, feedback_id,
            )
            self._active = job
            self._running_key = frozen_request.binding_key
            if operation == "regenerate":
                self._running_key += "|regenerate|" + feedback_id
            self._publish(job, "running", False)
            # Direct Qt slots may invalidate while the started event is sent.
            if job.terminal or self._closed or job.job_id != self._latest_job_id:
                if self._active is job:
                    self._active = None
                    self._running_key = ""
                return job.job_id
            try:
                future = self.executor.submit(self._run_job, job)
            except Exception as exc:
                if self._active is job:
                    self._active = None
                    self._running_key = ""
                state, reason = self._exception_state(exc)
                self._finish(job, state, reason)
                self.result_discarded.emit("proactive_failed:" + type(exc).__name__)
                return job.job_id
            self._running = future
            future.add_done_callback(lambda done: self._on_done(done, job))
            return job.job_id

    def _next_token(self) -> int:
        self._job_id = max(self._job_id, self._latest_job_id) + 1
        self._latest_job_id = self._job_id
        return self._job_id

    def invalidate(self, reason: str) -> None:
        with self._lock:
            token = self._next_token()
            old_job, old_future = self._active, self._running
            if old_future is None:
                self._active = None
            self._running_key = ""
            self._completed_key = ""
            self.last_state = None
            if old_job is not None:
                self._finish(old_job, "cancelled", "context_invalidated")
            # This never stops an executing HTTP request or refunds its budget.
            if old_future is not None and not old_future.done():
                old_future.cancel()
            if self._latest_job_id == token:
                self._set_status(f"proactive_invalidated:{reason}")

    def shutdown(self) -> None:
        from PySide6.QtCore import QThread

        with self._lock:
            self._closed = True
            self.invalidate("shutdown")
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self.executor.shutdown(wait=False)
            if QThread.currentThread() == self.thread():
                self._deadline_timer.stop()
            self._set_status("proactive_shutdown")

    def check_timeouts(self) -> None:
        with self._lock:
            job = self._active
            if self._closed or job is None or job.terminal or self._clock() < job.deadline_at:
                return
            old_future = self._running
            self._finish(job, "timed_out", "task_deadline_exceeded")
            if old_future is not None:
                old_future.cancel()

    def _run_job(self, job: _ProactiveJob) -> ProactiveTaskResult:
        from concurrent.futures import CancelledError

        with self._lock:
            if job.terminal or self._closed or job.job_id != self._latest_job_id:
                raise CancelledError()
            if self._clock() >= job.deadline_at:
                self._finish(job, "timed_out", "task_deadline_exceeded")
                raise CancelledError()
        if job.operation == "regenerate":
            return self._run_regeneration(job.job_id, job.request, job.config, job.feedback_id, job.queued_at)
        return self._run(job.job_id, job.request, job.config, job.queued_at)

    def _run(
        self,
        job_id: int,
        request: ProactiveContactRequest,
        config: AppConfig,
        queued_at: str,
    ) -> ProactiveTaskResult:
        worker_started_at = utc_now_iso()
        outcome = self.service.evaluate(request, config)
        return ProactiveTaskResult(job_id, outcome, queued_at, worker_started_at, utc_now_iso())

    def _run_regeneration(
        self,
        job_id: int,
        request: ProactiveContactRequest,
        config: AppConfig,
        feedback_id: str,
        queued_at: str,
    ) -> ProactiveTaskResult:
        worker_started_at = utc_now_iso()
        outcome = self.service.regenerate(request, config, feedback_id)
        return ProactiveTaskResult(
            job_id,
            outcome,
            queued_at,
            worker_started_at,
            utc_now_iso(),
            "regenerate",
        )

    def _on_done(self, future: Future, job: _ProactiveJob | None = None) -> None:
        with self._lock:
            if job is None:
                job = self._active if future is self._running else None
            if job is not None and self._active is job:
                self._active = None
                self._running = None
                self._running_key = ""
            try:
                task = future.result()
            except Exception as exc:
                if job is not None and job.job_id == self._latest_job_id and not job.terminal:
                    state, reason = self._exception_state(exc)
                    self._finish(job, state, reason)
                self.result_discarded.emit("proactive_failed:" + type(exc).__name__)
                return
            if job is not None and not job.terminal and self._clock() >= job.deadline_at:
                self._finish(job, "timed_out", "task_deadline_exceeded")
            if task.job_id != self._latest_job_id or self._closed or (job is not None and job.terminal):
                if job is not None:
                    self._finish(job, "superseded", "newer_context")
                self.result_discarded.emit(f"proactive_stale_result:{task.job_id}")
                return
            if job is not None and task.outcome.request != job.request:
                self._finish(job, "failed", "outcome_binding_mismatch")
                self.result_discarded.emit("proactive_failed:outcome_binding_mismatch")
                return
            self._completed_key = task.outcome.request.binding_key
            if job is not None:
                status = "blocked" if task.outcome.display_mode == "blocked" else "succeeded"
                reason = "outcome_" + task.outcome.display_mode
                if task.outcome.display_mode == "blocked" and task.outcome.result is not None:
                    result_status = task.outcome.result.status.lower()
                    source_reason = task.outcome.result.status.removeprefix("proactive:blocked:")
                    if task.outcome.result.status.startswith("proactive:blocked:") and source_reason in _PROACTIVE_SOURCE_BLOCK_REASONS:
                        reason = "source_blocked:" + source_reason
                    elif "timeout" in result_status or "超时" in result_status:
                        status, reason = "timed_out", "provider_timeout"
                    elif any(marker in result_status for marker in ("budget", "quota", "额度")):
                        reason = "request_budget_blocked"
                self._finish(job, status, reason, result_available=True)
            else:
                self._set_status(f"proactive_finished: job={task.job_id} mode={task.outcome.display_mode}")
            if task.job_id == self._latest_job_id and not self._closed:
                self.result_ready.emit(task)

    def _finish(self, job: _ProactiveJob, status: str, reason: str, result_available: bool = False) -> None:
        if job.terminal:
            return
        job.terminal = True
        self._publish(job, status, True, reason, result_available)

    def _publish(
        self, job: _ProactiveJob, status: str, terminal: bool,
        reason: str = "", result_available: bool = False,
    ) -> None:
        state = ProactiveTaskState(job.job_id, status, terminal, reason, job.request, result_available)
        if job.job_id == self._latest_job_id:
            self.last_state = state
            prefix = {"running": "regeneration_started" if job.operation == "regenerate" else "started", "succeeded": "finished"}.get(status, status)
            self._set_status(f"proactive_{prefix}: job={job.job_id}, reason={reason}")
        self.task_state_changed.emit(state)

    def _reject(self, request: ProactiveContactRequest, status: str, reason: str) -> None:
        # job 0 is explicitly a rejected submission, not the running job's ID.
        self.task_state_changed.emit(ProactiveTaskState(0, status, True, reason, request))
        self.result_discarded.emit("proactive_" + ("duplicate_suppressed" if status == "duplicate" else status))

    @staticmethod
    def _exception_state(exc: Exception) -> tuple[str, str]:
        from concurrent.futures import CancelledError

        if isinstance(exc, CancelledError):
            return "cancelled", "not_started_cancelled"
        if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
            return "timed_out", "provider_timeout"
        text = str(exc).lower()
        if any(marker in text for marker in ("budget", "quota", "额度")):
            return "blocked", "request_budget_blocked"
        if isinstance(exc, PermissionError) or any(marker in text for marker in ("permission", "authorization", "unauthorized")):
            return "blocked", "permission_blocked"
        return "failed", "worker_failure:" + type(exc).__name__

    def _set_status(self, value: str) -> None:
        self.last_status = value
        self.status_changed.emit(value)


def _merge_tag_results(
    remark: RemarkTagParseResult,
    display: RemarkTagParseResult,
) -> tuple[str | None, tuple[str, ...], bool, str]:
    tags = tuple(dict.fromkeys((*remark.parsed_tags, *display.parsed_tags)))
    tiers = {item for item in (remark.value_tier, display.value_tier) if item}
    worthless = remark.is_worthless or display.is_worthless
    if len(tiers) > 1 or remark.tag_parse_status == "conflicting" or display.tag_parse_status == "conflicting":
        return None, tags, worthless, "conflicting"
    tier = next(iter(tiers), None)
    statuses = {remark.tag_parse_status, display.tag_parse_status}
    status = "ambiguous" if "ambiguous" in statuses else ("parsed" if tags else "none")
    return tier, tags, worthless, status


def evaluate_proactive_eligibility(
    contact: Contact,
    parser: CustomerRemarkTagParser | None = None,
) -> ProactiveEligibility:
    """Return the single canonical A/B eligibility decision for Proactive V1.

    WeChat OCR-backed contacts commonly carry the user's remark in
    ``display_name`` while imported contacts may use ``remark``.  Both fields
    are parsed by the same canonical parser and merged once here; no separate
    Proactive regex or legacy eligibility rule is allowed.
    """
    canonical = parser or CustomerRemarkTagParser()
    remark_tags = canonical.parse(contact.remark)
    display_tags = canonical.parse(contact.display_name)
    tier, tags, worthless, parse_status = _merge_tag_results(remark_tags, display_tags)
    if worthless:
        reason = "worthless"
    elif parse_status == "conflicting":
        reason = "conflicting_value_tier"
    elif tier not in {"A", "B"}:
        reason = "non_ab"
    else:
        reason = "ab"
    return ProactiveEligibility(
        eligible=reason == "ab",
        value_tier=tier,
        parsed_tags=tags,
        is_worthless=worthless,
        parse_status=parse_status,
        reason=reason,
    )


class ProactiveCloudAuthorizationGate:
    """Issue and validate a narrow, per-request A/B Proactive authorization.

    The scope is never persisted onto ``contacts.allow_cloud_ai`` and therefore
    cannot authorize Reply, Memory extraction, Historical, or another customer.
    The UI may issue it only for the current stable ``chat_dm`` binding after
    confirming that no fresh Reply event owns the panel.
    """

    def __init__(self, parser: CustomerRemarkTagParser | None = None) -> None:
        self.parser = parser or CustomerRemarkTagParser()

    def issue_scope(
        self,
        *,
        contact: Contact,
        page_type: str,
        identity_stable: bool,
        account_scope_id: str,
        active_contact_id: str,
        reply_pending: bool,
    ) -> str:
        eligibility = evaluate_proactive_eligibility(contact, self.parser)
        if not PROACTIVE_CONTACT_V1_ENABLED:
            return ""
        if page_type != "chat_dm" or not identity_stable or reply_pending:
            return ""
        if not account_scope_id or contact.account_scope_id != account_scope_id:
            return ""
        if not active_contact_id or contact.id != active_contact_id:
            return ""
        if contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED} or contact.merged_into:
            return ""
        return PROACTIVE_CLOUD_AUTH_SCOPE if eligibility.eligible else ""

    def authorize(
        self,
        snapshot: ProactiveContactSnapshot,
        decision: ProactiveContactReason,
    ) -> ProactiveCloudAuthorization:
        scope = snapshot.request.proactive_cloud_authorization
        if scope not in {PROACTIVE_CLOUD_AUTH_SCOPE, ACTIVATION_PREWARM_CLOUD_AUTH_SCOPE}:
            return ProactiveCloudAuthorization(False, "scope_not_issued")
        if snapshot.contact.account_scope_id != snapshot.request.account_scope_id:
            return ProactiveCloudAuthorization(False, "account_scope_mismatch")
        eligibility = evaluate_proactive_eligibility(snapshot.contact, self.parser)
        if not eligibility.eligible or not decision.eligible:
            return ProactiveCloudAuthorization(False, eligibility.reason)
        if decision.value_tier != eligibility.value_tier or decision.value_tier not in {"A", "B"}:
            return ProactiveCloudAuthorization(False, "tier_mismatch")
        if scope == ACTIVATION_PREWARM_CLOUD_AUTH_SCOPE and decision.value_tier != "A":
            return ProactiveCloudAuthorization(False, "prewarm_a_only")
        if decision.reason_type == "do_not_contact_now":
            return ProactiveCloudAuthorization(False, "business_do_not_contact")
        if decision.context_payload.get("reply_mode_required"):
            return ProactiveCloudAuthorization(False, "reply_has_priority")
        if snapshot.contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED} or snapshot.contact.merged_into:
            return ProactiveCloudAuthorization(False, "contact_not_usable")
        reason = "a_activation_prewarm_only" if scope == ACTIVATION_PREWARM_CLOUD_AUTH_SCOPE else "ab_proactive_only"
        return ProactiveCloudAuthorization(True, reason, scope)


def _do_not_contact(
    tier: str,
    payload: dict[str, Any],
    reason: str,
    next_at: str | None,
) -> ProactiveContactReason:
    return ProactiveContactReason(
        True,
        "do_not_contact_now",
        "当前不适合追加主动联系",
        0.95,
        "",
        reason,
        next_at,
        payload,
        tier,
    )


def _with_activation_assessments(
    snapshot: ProactiveContactSnapshot,
    *,
    now: datetime,
    silence_days: int,
    force: bool = False,
) -> ProactiveContactSnapshot:
    if (
        not force
        and snapshot.contact_recency is not None
        and snapshot.stage_assessment is not None
        and snapshot.need_assessment is not None
    ):
        return snapshot
    started = monotonic_time.perf_counter()
    recency = CustomerContactRecencyService().assess(
        snapshot,
        now=now,
        silence_days=silence_days,
    )
    stage = CropStageInferenceService().assess(snapshot, now=now)
    need = CurrentNeedInferenceService().assess(snapshot, stage, now=now)
    assessment_payload = {
        "contact_recency": asdict(recency),
        "current_stage": asdict(stage),
        "current_need": asdict(need),
    }
    components = dict(snapshot.component_hashes)
    components.update(
        {
            "stage_inference": _stable_hash(asdict(stage)),
            "need_inference": _stable_hash(asdict(need)),
            "talking_point": _stable_hash(need.best_talking_point),
        }
    )
    bounded = deepcopy(snapshot.bounded_payload)
    bounded["activation_assessment"] = assessment_payload
    return replace(
        snapshot,
        bounded_payload=bounded,
        component_hashes=components,
        contact_recency=recency,
        stage_assessment=stage,
        need_assessment=need,
        assessment_ms=round((monotonic_time.perf_counter() - started) * 1000.0, 3),
    )


def _request_local_datetime(local_date: str) -> datetime:
    local_tz = datetime.now().astimezone().tzinfo
    try:
        day = datetime.fromisoformat(local_date).date()
    except ValueError:
        return datetime.now().astimezone()
    return datetime.combine(day, time(hour=12), tzinfo=local_tz)


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


_CROP_WORDS = (
    "辣椒", "蓝莓", "番茄", "西红柿", "黄瓜", "葡萄", "玉米", "小麦", "草莓",
    "苹果", "柑橘", "橙子", "红薯", "南瓜", "大蒜", "花生", "水稻", "棉花",
)
_PROVINCES = (
    "山东", "广西", "广东", "海南", "福建", "云南", "贵州", "四川", "重庆",
    "河南", "河北", "山西", "陕西", "甘肃", "辽宁", "吉林", "黑龙江", "江苏",
    "浙江", "安徽", "江西", "湖北", "湖南", "内蒙古", "新疆", "宁夏", "青海",
    "北京", "天津", "上海",
)


def _resolve_current_crop(snapshot: ProactiveContactSnapshot) -> tuple[str, list[str], bool]:
    known = tuple(dict.fromkeys((*_CROP_WORDS, *(item.crop for item in snapshot.dashboard.contexts if item.crop))))
    for message in reversed(snapshot.messages):
        crops = [crop for crop in known if crop and crop in message.text]
        crops = list(dict.fromkeys(crops))
        if len(crops) == 1:
            return crops[0], [f"最近沟通明确作物：{crops[0]}"], False
        if len(crops) > 1:
            return "", [f"最近沟通同时出现多个作物：{'、'.join(crops[:3])}"], True
    active = [item for item in snapshot.dashboard.contexts if item.status == "active" and item.crop]
    if len(active) == 1:
        return active[0].crop, [f"当前 active Growing Context：{active[0].crop}"], False
    event_text = " ".join(f"{item.title} {item.summary}" for item in snapshot.dashboard.events[:5])
    event_crops = list(dict.fromkeys(crop for crop in known if crop and crop in event_text))
    if len(event_crops) == 1:
        return event_crops[0], [f"最近事件对应作物：{event_crops[0]}"], False
    profile_crop = next((item.value for item in snapshot.dashboard.core_fields if item.key == "crop" and item.value), "")
    if profile_crop and len(snapshot.dashboard.contexts) <= 1:
        return profile_crop, [f"客户档案作物：{profile_crop}"], False
    if len(active) > 1 or len(event_crops) > 1 or snapshot.dashboard.other_context_count > 0:
        return "", ["存在多个作物且近期上下文不足，禁止自动绑定"], True
    return "", ["缺少可确认的当前作物"], False


def _resolve_region(snapshot: ProactiveContactSnapshot) -> tuple[str, str, list[str]]:
    for message in reversed(snapshot.messages):
        province = _extract_province(message.text)
        if province and _contains_any(message.text, ("我这边", "我们这边", "这里", "本地", "在")):
            return province, "customer_explicit", [f"最近聊天确认地区：{province}"]
    region_field = next(
        (item for item in snapshot.dashboard.core_fields if item.key == "region" and item.value),
        None,
    )
    if region_field is not None and region_field.source_type in {"manual", "structured_memory"}:
        return region_field.value, "confirmed_profile", [f"已确认客户档案地区：{region_field.value}"]
    active_region = next(
        (item.region for item in snapshot.dashboard.contexts if item.status == "active" and item.region),
        "",
    )
    if active_region:
        return active_region, "confirmed_profile", [f"active Growing Context 地区：{active_region}"]
    if region_field is not None and region_field.source_type == "remark":
        return region_field.value, "wechat_remark", [f"微信备注地区：{region_field.value}"]
    order_text = " ".join(
        source.source_text
        for order in snapshot.dashboard.order_records[:5]
        for source in order.sources
        if source.source_text
    )
    shipping_region = _extract_province(order_text)
    if shipping_region:
        return shipping_region, "shipping_address", [f"可靠收货信息地区：{shipping_region}"]
    phone_texts = [
        item.content for item in snapshot.memories
        if _contains_any(item.content, ("手机号归属", "号码归属", "手机归属"))
    ]
    if region_field is not None and region_field.source_type == "phone_attribution":
        phone_texts.insert(0, region_field.value)
    phone_region = _extract_province(" ".join(phone_texts))
    if phone_region:
        return phone_region, "phone_attribution", [f"手机号归属地低置信兜底：{phone_region}"]
    return "", "unknown", ["地区不足，不做地区农时强推"]


def _extract_province(text: str) -> str:
    return next((item for item in _PROVINCES if item in str(text)), "")


def _latest_explicit_stage(
    snapshot: ProactiveContactSnapshot,
    crop: str,
    now: datetime,
) -> tuple[str, str, str] | None:
    patterns = (
        (re.compile(r"(?:还没|还没有|尚未|没有|未)(?:开始)?(?:采收|收获|收薯|收红薯)|(?:快要|临近|马上要)(?:采收|收获|收薯)|(?:红薯|这茬|这批)?还没收"), "采收前管理期"),
        (re.compile(r"(?:已经|刚|全部).{0,4}(?:采完|采收完|收完)|收完了"), "采后阶段"),
        (re.compile(r"(?:已经|现在|正在|开始).{0,4}(?:采收|收获)"), "采收期"),
        (re.compile(r"(?:(?:已经|刚|现在|目前|正在).{0,4}(?:定植|移栽)|(?:栽了|刚栽).{0,6}(?:天|周)?)"), "缓苗期"),
        (re.compile(r"(?:现在|目前|已经|正在).{0,5}(?:开花|花期)"), "开花期"),
        (re.compile(r"(?:现在|目前|已经|正在).{0,5}(?:坐果|挂果)"), "坐果期"),
        (re.compile(r"(?:现在|目前|已经|正在).{0,5}(?:膨果|果实膨大)"), "膨果期"),
        (re.compile(r"(?:刚|已经).{0,4}(?:采完|采收完)"), "采后恢复期"),
        (re.compile(r"(?:已经|刚).{0,4}拔园"), "已拔园"),
    )
    candidates = [
        (_message_time(item), item.text, "客户最近明确说")
        for item in snapshot.messages if item.speaker == Speaker.OTHER
    ]
    for kind in ("facts", "events"):
        for row in snapshot.tracking_evidence.get(kind, ()):
            if not isinstance(row, dict) or not _reliable_evidence(row):
                continue
            text = str(row.get("source_text") or row.get("text") or "")
            # An unlabeled historical row cannot safely identify a batch/crop.
            if crop and canonicalize_crop(crop) in _extract_crops(text):
                candidates.append((_parse_dt(str(row.get("occurred_at") or "")), text, "已确认记录"))
    candidates.sort(
        key=lambda item: item[0].timestamp() if item[0] is not None else float("-inf"),
        reverse=True,
    )
    active_crops = {
        canonicalize_crop(item.crop) for item in snapshot.dashboard.contexts
        if item.status == "active" and item.crop
    }
    for observed, raw_text, evidence_source in candidates:
        if observed is None or not -1 <= (now - observed.astimezone(now.tzinfo)).days <= 45:
            continue
        text = "".join(raw_text.split())
        mentioned_crops = _extract_crops(text)
        if crop and mentioned_crops and canonicalize_crop(crop) not in mentioned_crops:
            continue
        if len(mentioned_crops) > 1 or (not mentioned_crops and len(active_crops) > 1):
            continue
        for pattern, label in patterns:
            match = pattern.search(text)
            if match:
                return label, f"{observed.date()} {evidence_source}：{match.group(0)[:40]}", "customer_explicit"
    return None


_CN_NUMBER = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_TIMING_PATTERN = re.compile(
    r"([一二三四五六七八九十两]|1[0-2]|0?[1-9])\s*月(?:份)?(?:左右|前后)?[^。！？]{0,16}?(开始)?\s*(种|播种|定植|移栽|采收)"
)


def _latest_explicit_planting_timing(
    snapshot: ProactiveContactSnapshot, *, crop: str = ""
) -> tuple[int, str, str] | None:
    texts: list[str] = [item.text for item in reversed(snapshot.messages) if item.speaker == Speaker.OTHER]
    texts.extend(
        str(item.get("text") or "")
        for item in snapshot.recent_observation_messages
        if item.get("speaker") == "other" and float(item.get("confidence") or 0.0) >= 0.5
    )
    texts.extend(item.content for item in snapshot.memories)
    for text in texts:
        mentioned_crops = _extract_crops(str(text))
        if crop and mentioned_crops and canonicalize_crop(crop) not in mentioned_crops:
            continue
        if len(mentioned_crops) > 1 or (
            crop and not mentioned_crops
            and len({canonicalize_crop(item.crop) for item in snapshot.dashboard.contexts if item.status == "active" and item.crop}) > 1
        ):
            continue
        match = _TIMING_PATTERN.search("".join(str(text).split()))
        if not match:
            continue
        raw_month = match.group(1)
        month = int(raw_month) if raw_month.isdigit() else _CN_NUMBER.get(raw_month, 0)
        if 1 <= month <= 12:
            action = match.group(3)
            return month, action, f"客户历史明确时间：{month}月左右{action}"
    return None


def _stage_from_explicit_month(target_month: int, current_month: int, action: str) -> tuple[str, str]:
    delta = (current_month - target_month) % 12
    if delta == 0:
        label = "预计进入采收窗口" if action == "采收" else "预计进入种植/定植窗口"
        return label, "未来7～20天确认实际进度与早期管理衔接"
    if action == "采收":
        return ("预计处于采后恢复窗口", "未来7～20天关注采后恢复与下茬安排")
    if delta == 1:
        return ("预计处于定植后缓苗/苗期窗口", "未来7～20天关注缓苗、生根与苗势恢复")
    if delta == 2:
        return ("预计进入苗期生长管理窗口", "未来7～20天关注苗势与下一生育节点")
    if delta >= 10:
        return ("预计临近种植/定植窗口", "未来7～20天确认是否仍按原计划种植")
    return ("历史种植时间距今较久，当前阶段待确认", "先确认今年实际种植进度")


def _seasonal_stage(crop: str, region: str, month: int) -> tuple[str, str] | None:
    south = any(item in region for item in ("广西", "广东", "海南", "福建", "云南"))
    north = any(item in region for item in ("山东", "河南", "河北", "山西", "陕西", "辽宁"))
    if crop == "蓝莓" and north and month in {9, 10, 11}:
        return "预计处于秋季树势恢复/越冬准备窗口", "未来7～20天关注树势恢复、花芽与越冬准备"
    if crop in {"辣椒", "番茄", "西红柿"} and south and month in {9, 10, 11}:
        return "预计进入秋冬茬种植/定植窗口", "未来7～20天关注缓苗、生根与定植后根部风险"
    if crop == "草莓" and month in {9, 10}:
        return "预计进入定植/缓苗窗口", "未来7～20天关注缓苗、生根与苗势恢复"
    if crop in {"小麦", "大蒜"} and north and month in {9, 10, 11}:
        return "预计进入播种/出苗窗口", "未来7～20天关注出苗、根系与苗齐苗壮"
    if crop == "葡萄" and north and month in {9, 10, 11}:
        return "预计进入采后恢复/越冬准备窗口", "未来7～20天关注树势恢复与越冬准备"
    return None


def _next_window_for_stage(stage: str) -> str:
    for words, value in (
        (("定植", "移栽", "缓苗", "苗期"), "未来7～20天关注生根、缓苗与苗势恢复"),
        (("开花", "花期"), "未来7～20天关注花期管理与坐果衔接"),
        (("坐果", "幼果"), "未来7～20天关注幼果与膨果衔接"),
        (("膨果",), "未来7～20天关注膨果与品质管理"),
        (("采收", "收获"), "未来7～20天优先确认采收进度、品质与采后安排"),
        (("采后", "采完"), "未来7～20天关注采后恢复与下一茬安排"),
    ):
        if _contains_any(stage, words):
            return value
    return "未来7～20天先确认下一管理节点"


def _needs_for_stage(stage: str) -> tuple[str, ...]:
    for words, needs in (
        (("种植", "定植", "移栽", "缓苗"), ("缓苗生根", "苗势恢复", "定植后根部风险管理")),
        (("播种", "出苗", "苗期", "幼苗"), ("出苗与根系", "苗齐苗壮", "早期根部风险管理")),
        (("开花", "花期"), ("花期营养", "保花坐果", "授粉与环境确认")),
        (("坐果", "幼果"), ("幼果稳果", "膨果衔接", "养分供应")),
        (("膨果",), ("膨果与品质", "养分供应", "转色采收衔接")),
        (("转色", "成熟"), ("品质与成熟度确认", "采收安排", "商品性回访")),
        (("采收", "收获"), ("采收进度回访", "商品性与品质确认", "采后安排或下茬复盘")),
        (("树势恢复", "采后恢复", "越冬"), ("树势恢复", "花芽/越冬管理", "根系养护")),
    ):
        if _contains_any(stage, words):
            return needs
    return ()


def _historical_relevance(snapshot: ProactiveContactSnapshot) -> list[str]:
    values = [item.value for item in snapshot.dashboard.concerns if item.value]
    for event in snapshot.dashboard.events[:8]:
        if event.event_type in {"problem_reported", "diagnosis", "solution_given", "customer_concern"}:
            text = " ".join(f"{event.title} {event.summary}".split())
            if text:
                values.append(text[:100])
    return list(dict.fromkeys(values))


def _rank_needs_with_history(needs: list[str], historical: list[str]) -> list[str]:
    text = " ".join(historical)
    keywords = {
        "生根": ("生根", "根腐", "死苗", "根系"),
        "花芽": ("花芽", "开花", "坐果"),
        "养分": ("黄叶", "缺素", "营养", "肥"),
        "品质": ("品质", "膨果", "转色"),
    }
    scores = []
    for index, need in enumerate(needs):
        score = -index
        for token, history_words in keywords.items():
            if token in need and _contains_any(text, history_words):
                score += 10
        scores.append((score, index, need))
    return [item[2] for item in sorted(scores, key=lambda item: (-item[0], item[1]))]


def _recently_stocked(snapshot: ProactiveContactSnapshot, now: datetime) -> bool:
    order = _latest_event(snapshot.dashboard, {"order_placed", "purchased"})
    if order is not None and _age_days(order.event_at, now) <= 45:
        return True
    return any(
        _contains_any(item.text, ("已经备货", "备好了", "已经买", "下单了", "货到了"))
        and not _contains_any(item.text, ("还没买", "没有买", "没下单", "准备下单", "打算买"))
        for item in snapshot.messages if item.speaker == Speaker.OTHER
    )


def _reliable_evidence(row: dict[str, Any]) -> bool:
    return (
        str(row.get("status") or "") not in {"uncertain", "conflicting", "invalid", "superseded", "cancelled"}
        and str(row.get("provenance") or "") in {"customer_explicit", "user_explicit"}
    )


def _current_season_purchase(snapshot: ProactiveContactSnapshot, crop: str, now: datetime) -> str:
    """Keep order and purchase events distinct; neither proves use or symptoms."""
    if not crop:
        return ""
    for row in snapshot.tracking_evidence.get("events", ()):
        if not isinstance(row, dict) or not _reliable_evidence(row):
            continue
        category = str(row.get("category") or row.get("evidence_key") or "")
        if category not in {"purchased", "order_placed"}:
            continue
        occurred = _parse_dt(str(row.get("occurred_at") or ""))
        if occurred is None or occurred.astimezone(now.tzinfo).year != now.year or occurred > now:
            continue
        text = " ".join(str(row.get(key) or "") for key in ("text", "source_text", "normalized_value"))
        if canonicalize_crop(crop) in _extract_crops(text):
            record_label = "购买记录" if category == "purchased" else "下单记录"
            event_scope = "仅证明购买及用途" if category == "purchased" else "仅证明下单及用途，不据此认定已购买"
            return f"{occurred.date()} 已确认本季{crop}{record_label}：{text[:100]}；{event_scope}，不证明已使用、仍有库存或存在症状"
    return ""


def _current_sales_refusal(snapshot: ProactiveContactSnapshot, now: datetime) -> str | None:
    """Keep one declined opportunity closed until a new, explicit demand exists.

    Time passing, our outgoing pitch, or an AI-inferred seasonal stage cannot
    reopen it. No customer fact, tier or permanent-contact status is changed.
    """
    refusal = next(
        (item for item in reversed(snapshot.messages)
         if item.speaker == Speaker.OTHER and _contains_any(item.text, _REJECTION_WORDS)
         and not item.text.rstrip().endswith(("?", "？"))),
        None,
    )
    if refusal is None:
        return None
    stopped = _contains_any(refusal.text, ("别联系", "不要联系", "不再联系"))
    refusal_at = _message_time(refusal)
    after_refusal = snapshot.messages[list(snapshot.messages).index(refusal) + 1 :]
    explicit_need_cues = ("需要", "想买", "要买", "帮我", "怎么治", "用什么", "下茬", "明年", "改种", "重新种", "新一茬")
    reopened_by_customer = any(
        item.speaker == Speaker.OTHER
        and _contains_any(item.text, explicit_need_cues)
        and not _contains_any(item.text, _REJECTION_WORDS)
        for item in after_refusal
    )
    if reopened_by_customer:
        return None
    if stopped:
        return "客户明确要求停止联系，等待客户主动重新提出需求"
    if refusal_at is not None:
        for row in snapshot.tracking_evidence.get("events", ()):
            if not isinstance(row, dict) or not _reliable_evidence(row):
                continue
            occurred = _parse_dt(str(row.get("occurred_at") or ""))
            if occurred is None or not refusal_at < occurred <= now:
                continue
            category = str(row.get("category") or row.get("evidence_key") or "")
            text = str(row.get("source_text") or row.get("text") or "")
            if category in {"problem_reported", "order_intent", "order_placed", "purchased", "usage_started"} or (
                category == "management_plan"
                and _contains_any(text, ("下茬", "明年", "改种", "重新种", "新一茬"))
            ):
                return None
    return "客户已拒绝本轮方案，等待有依据的新需求点；不重复推本次产品，不改变客户价值等级"


def _display_stage(stage: str) -> str:
    value = stage.replace("预计进入", "").replace("预计处于", "").replace("预计临近", "")
    return value.replace("窗口", "").strip(" /｜") or "当前管理节点"


def _activation_question(snapshot: ProactiveContactSnapshot) -> str:
    stage = snapshot.stage_assessment
    need = snapshot.need_assessment
    opening = _salutation_opening(snapshot)
    if need is not None and need.already_stocked:
        return f"{opening}之前备的货现在开始用了没有？"
    if stage is None or not stage.crop:
        return f"{opening}今年当前主要在管哪个作物？"
    crop = stage.crop
    purchase_background = _current_season_purchase(snapshot, crop, _request_local_datetime(snapshot.request.local_date))
    if "采收前" in stage.stage_label:
        return f"{opening}这茬{crop}最近长势怎么样，采收前还有什么管理问题需要处理？"
    if stage.status == "unknown":
        if purchase_background:
            return f"{opening}今年这茬{crop}最近长势怎么样，有没有需要处理的管理问题？"
        return f"{opening}今年{crop}现在管理到哪一步了？"
    if _contains_any(stage.stage_label, ("采收", "收获", "采后")):
        return f"{opening}今年{crop}这批已经开始收了吗？"
    if stage.status == "inferred":
        if _contains_any(stage.stage_label, ("种植", "定植", "移栽")):
            return f"{opening}今年{crop}是不是已经进入种植定植这一步了？"
        return f"{opening}今年{crop}是不是已经到{_display_stage(stage.stage_label)}这一步了？"
    top_need = need.top_need if need is not None and need.top_need else "当前管理"
    if top_need in {"确认当前实际情况", "当前阶段管理进展"}:
        return f"{opening}{crop}现在{_display_stage(stage.stage_label)}这块进展还顺利吗？"
    return f"{opening}{crop}现在{top_need}这块进展还顺利吗？"


def _salutation_opening(snapshot: ProactiveContactSnapshot) -> str:
    profile = snapshot.conversation_style
    if profile is None:
        return ""
    salutation = profile.usable_salutation.strip()
    return f"{salutation}，" if salutation else ""


def _question_for_snapshot(
    snapshot: ProactiveContactSnapshot,
    reason_type: str = "reactivation",
) -> tuple[str, str]:
    stage_assessment = snapshot.stage_assessment
    fields = {item.key: item.value for item in snapshot.dashboard.core_fields if item.value}
    active = next((item for item in snapshot.dashboard.contexts if item.status == "active"), None)
    crop = (fields.get("crop", "") or (active.crop if active else "")).strip()
    stage = (fields.get("growth_stage", "") or (active.growth_stage if active else "")).strip()
    opening = _salutation_opening(snapshot)
    if reason_type == "order_followup":
        return f"{opening}之前的货现在是否已经收到、使用安排是否确定了？", "delivery_or_usage_status"
    if reason_type == "usage_followup":
        return f"{opening}目前是否已经开始使用，使用到哪一步了？", "usage_status"
    if reason_type == "effect_followup":
        return f"{opening}使用后观察到的变化怎么样？", "effect_feedback"
    if stage_assessment is not None and stage_assessment.status != "unknown":
        return _activation_question(snapshot), "current_stage_confirmation"
    if not crop:
        return f"{opening}今年主要还在种什么？", "crop"
    if not stage:
        return f"{opening}{crop}现在大概到哪个阶段了？", "growth_stage"
    return f"{opening}{crop}最近最想先解决哪一项管理问题？", "current_need"


def _event_followup_reason(
    snapshot: ProactiveContactSnapshot,
    now: datetime,
) -> tuple[str, str, str] | None:
    order = _latest_event(snapshot.dashboard, {"order_placed", "purchased"})
    if order is not None and 3 <= _age_days(order.event_at, now) <= 60:
        return "order_followup", "适合核对既有订单的履约或后续使用安排", order.id
    usage = _latest_event(snapshot.dashboard, {"usage_started"})
    if usage is not None and 3 <= _age_days(usage.event_at, now) <= 75:
        return "usage_followup", "适合了解既有方案是否开始使用及使用环节", usage.id
    effect = _latest_event(snapshot.dashboard, {"solution_given", "product_recommended", "effect_feedback"})
    if effect is not None and 5 <= _age_days(effect.event_at, now) <= 60:
        return "effect_followup", "适合回访此前方案或使用后的实际变化", effect.id
    return None


def _followup_is_currently_actionable(due_at: str, now: datetime) -> bool:
    due = _parse_dt(due_at)
    if due is None:
        return False
    age = (now.date() - due.astimezone(now.tzinfo).date()).days
    return 0 <= age <= 90


def _predicted_window(
    snapshot: ProactiveContactSnapshot,
    now: datetime,
) -> tuple[str, str, str, str] | None:
    stage = snapshot.stage_assessment
    need = snapshot.need_assessment
    if stage is None or need is None or stage.status == "unknown":
        return None
    marker = "已确认" if stage.status == "confirmed" else "推算，待确认"
    summary = f"当前阶段：{stage.stage_label}（{marker}）；可能需求：{need.top_need}"
    missing = "current_need" if stage.status == "confirmed" else "current_stage_confirmation"
    return summary, _activation_question(snapshot), missing, stage.source


def _latest_event(dashboard: CustomerDashboard, types: set[str]):
    return next((item for item in dashboard.events if item.event_type in types), None)


def _message_time(message: Message | VisibleMessage | None) -> datetime | None:
    if message is None:
        return None
    if isinstance(message, VisibleMessage):
        # Unknown sent time stays unknown.  Observation/capture time is not
        # evidence of when the customer actually sent this visible message.
        return _parse_dt(message.message_time)
    return _parse_dt(message.observed_at or message.message_time or "")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_days(value: str, now: datetime) -> int:
    parsed = _parse_dt(value)
    if parsed is None:
        return 9999
    return max(0, (now - parsed.astimezone(now.tzinfo)).days)


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    compact = "".join(str(text).split()).casefold()
    return any(word.casefold() in compact for word in words)


def _generation_block(
    snapshot: ProactiveContactSnapshot,
    decision: ProactiveContactReason,
    config: AppConfig,
) -> str | None:
    contact = snapshot.contact
    if not decision.eligible or decision.reason_type == "do_not_contact_now":
        return "reason_not_contactable"
    if decision.context_payload.get("reply_mode_required"):
        return "reply_has_priority"
    if contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED} or contact.merged_into:
        return "contact_not_usable"
    if config.privacy.manual_protection_blocks_replies and snapshot.strategy_requires_manual_reply:
        return "manual_reply_protection"
    if config.ai.provider in {"OpenAI", "OpenAI Compatible"} and config.ai.api_key and not contact.allow_cloud_ai:
        authorization = ProactiveCloudAuthorizationGate().authorize(snapshot, decision)
        if not authorization.allowed:
            return "cloud_not_allowed"
    return None


def _local_proactive_preview(
    snapshot: ProactiveContactSnapshot,
    decision: ProactiveContactReason,
) -> ReplyGenerationResult:
    question = str(decision.context_payload.get("recommended_question") or _activation_question(snapshot))
    if _validate_tracking_suggestions(
        snapshot,
        [ReplySuggestion("激活建议", question, "low", "tracking_quality_guard")],
    ):
        question = _activation_question(snapshot)
    suggestions = [ReplySuggestion("激活建议", question, "low", "只问一个当前关键变量，不把推算当事实")]
    return _proactive_result(snapshot, decision, suggestions, "Local Preview", "proactive:local_preview")


def _validate_tracking_suggestions(
    snapshot: ProactiveContactSnapshot,
    suggestions: list[ReplySuggestion],
) -> str | None:
    profile = snapshot.conversation_style
    for suggestion in suggestions:
        text = suggestion.text.strip()
        expected = profile.usable_salutation.strip() if profile is not None else ""
        if expected:
            if not text.startswith(expected):
                return "customer_salutation_not_preserved"
        elif text.startswith(("您好", "你好")) or extract_salutation(text):
            return "unverified_salutation"
    stage = snapshot.stage_assessment
    if stage is None or stage.status == "unknown" or not stage.crop:
        return None
    text = " ".join(item.text for item in suggestions)
    if _contains_any(
        text,
        (
            "到哪个阶段",
            "管理到哪一步",
            "现在什么阶段",
            "主要还在种什么",
            "今年主要在种什么",
        ),
    ):
        return "generic_stage_question_despite_usable_timeline"
    return None


def _proactive_result(
    snapshot: ProactiveContactSnapshot,
    decision: ProactiveContactReason,
    suggestions: list[ReplySuggestion],
    provider: str,
    status: str,
) -> ReplyGenerationResult:
    return ReplyGenerationResult(
        True,
        status,
        suggestions,
        provider,
        customer_summary=decision.reason_summary,
        intent=decision.reason_type,
        relevant_history=(
            snapshot.need_assessment.evidence[:4]
            if snapshot.need_assessment is not None and snapshot.need_assessment.evidence
            else ("结合证据的日期与来源；历史购买不等于已使用或有当前症状",)
        ),
        ask_first=decision.missing_key_variable or "current_need",
        sales_judgement=f"围绕{decision.context_payload.get('recommended_need') or '真实管理需求'}发现或推进适合产品的销售；不自动发送",
        next_action="复制前人工确认；客户有新消息时立即切回 Reply",
        performance_trace={
            "context_build_ms": snapshot.build_ms,
            "activation_assessment_ms": snapshot.assessment_ms,
        },
    )


def _proactive_prompt(
    snapshot: ProactiveContactSnapshot,
    decision: ProactiveContactReason,
    *,
    regeneration_reason: str = "",
) -> tuple[str, str]:
    system = (
        "你是农资销售的主动联系文案助手。只输出 JSON 对象，不要 Markdown。"
        "必须返回 messages 数组且恰好一项，只有 text 和 rationale。"
        "这一条只能表达一个联系切入点并只提出一个核心问题；整条只能有一个问号。"
        "目的：选择当前能给客户实际价值、并最有助于发现或推进适合产品需求的切入点；说明为什么现在值得聊，客户回答后应有后续方向。"
        "按日期、作物/批次和来源区分仍有效资料、历史事件与未知情况。已确认的本季购买及用途可以直接承接，不能由购买推定已经使用、仍有库存或存在病虫害/黄叶等症状。"
        "旧备注或往年作物不能默认仍在种；但不能把可靠本季资料一律抛开，重新问客户种什么。"
        "客户近期已明确尚未采收时，围绕采收前长势、实际管理需求或已知后续安排联系，不重复问是否采收；已知答案不重复问。"
        "作物阶段是推算时保留不确定性，但优先用能发现实际需求的问题确认，不机械追问阶段。通用作物需求不是客户已有问题。"
        "客户明确拒绝本轮方案后等待有依据的新需求，不反复劝买，不把本次拒绝记作永久无价值。"
        "严禁虚构作物阶段、订单/购买状态、效果、天气、价格、库存、剂量、用法或客户意向。"
        "不要提及 AI、WhoChat、系统、档案、记忆、历史记录、Sales State、confidence、标签或客户等级。"
        "语气自然克制，不施压，不承诺效果，不自动发送。"
        "除非 allowed_product 非空且 confidence>=0.9，否则不要提具体产品；最多一个产品。"
        "称呼必须严格服从 conversation_style：存在 medium/high preferred_salutation 时自然沿用；"
        "prefers_no_salutation=true 或没有可靠客户级称呼时直接说事，禁止机械添加“您好”或猜哥/姐/老板。"
    )
    if regeneration_reason:
        feedback_instruction = {
            "too_formal": "上一版被判定太官方；改得更像熟客微信沟通，更短、更直接、更自然。",
            "too_long": "上一版被判定太长；显著缩短，但保留一个具体问题。",
            "too_salesy": "上一版太像推销；先了解情况，不要急着提产品。",
            "wrong_question": "上一版沟通方向不对；结合已知情况，选择当前有实际价值并能发现或推进合适产品需求的问题，不重复问已知答案。",
            "missing_history": "上一版没有结合历史；承接有日期/来源的有效资料与购买用途，区分历史事实和当前未知，不把本季可靠背景一律重新问一遍。",
            "unnatural_salutation": "上一版称呼不自然；严格按 conversation_style 使用客户级称呼，无可靠证据时不加称呼。",
            "too_stiff": "上一版太生硬；改成简短、自然的微信口语。",
            "other": "上一版未被采用；换一种简短、自然、低压力的表达。",
        }.get(regeneration_reason, "根据本次明确负反馈重写，保持简短、自然和事实安全。")
        system += feedback_instruction
    allowed_product = ""
    if decision.confidence >= 0.9:
        allowed_product = next(
            (item.product for item in snapshot.dashboard.order_records if item.product),
            "",
        )
    user_payload = {
        "reason": asdict(decision),
        "allowed_product": allowed_product,
        "context": snapshot.bounded_payload,
        "output_schema": {
            "messages": [
                {"text": "一条自然中文且只问一个核心问题", "rationale": "安全理由"},
            ]
        },
    }
    if snapshot.writing_preferences:
        user_payload["writing_preferences"] = list(snapshot.writing_preferences)
    if regeneration_reason:
        user_payload["regeneration_feedback"] = {"reason_code": regeneration_reason}
    return system, json.dumps(user_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_proactive_suggestions(items: list[ReplySuggestion]) -> str | None:
    if len(items) != 1:
        return "exact_one_required"
    texts = [" ".join(item.text.split()) for item in items]
    if any(len(text) < 4 or len(text) > 180 for text in texts):
        return "message_length"
    combined = "\n".join(texts)
    if sum(combined.count(mark) for mark in ("?", "？")) != 1:
        return "exact_one_question"
    if any(term.casefold() in combined.casefold() for term in _FORBIDDEN_GENERATED_TERMS):
        return "system_language"
    if _has_unverified_current_assertion(texts):
        return "unverified_current_assertion"
    return None


def _has_unverified_current_assertion(texts: list[str]) -> bool:
    """Reject current-state assertions without rejecting confirmation questions.

    ``目前是否已经开始使用？`` is exactly the safe question V1 asks for,
    while ``现在已经到了膨果期了，需要方案吗？`` still contains an unsafe
    assertion in the clause before the comma and remains blocked.
    """
    for text in texts:
        for match in _CLAUSE.finditer(text):
            clause = match.group(0).strip()
            if not _CURRENT_ASSERTION.search(clause):
                continue
            if clause.endswith(("?", "？")):
                continue
            return True
    return False


def _proactive_context_hash(
    snapshot: ProactiveContactSnapshot,
    decision: ProactiveContactReason,
    config: AppConfig,
) -> str:
    payload = {
        "policy_version": PROACTIVE_POLICY_VERSION,
        "account_scope_id": snapshot.request.account_scope_id,
        "contact_id": snapshot.request.contact_id,
        "conversation_id": snapshot.request.conversation_id,
        "local_date": snapshot.request.local_date,
        "contact_updated_at": snapshot.contact.updated_at,
        "strategy_requires_manual_reply": snapshot.strategy_requires_manual_reply,
        "component_hashes": snapshot.component_hashes,
        "reason": asdict(decision),
        "provider": config.ai.provider,
        "base_url": config.ai.base_url.strip().rstrip("/"),
        "model": config.ai.model,
        "cloud_configured": bool(config.ai.api_key),
        "allow_cloud_ai": snapshot.contact.allow_cloud_ai,
        "proactive_cloud_authorization": snapshot.request.proactive_cloud_authorization,
    }
    return _stable_hash(payload)


def _stable_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _outcome_to_json(outcome: ProactiveContactOutcome) -> str:
    payload = {
        "decision": asdict(outcome.decision),
        "result": asdict(outcome.result) if outcome.result is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _outcome_from_json(
    payload: str,
    request: ProactiveContactRequest,
    context_hash: str,
    build_ms: float,
) -> ProactiveContactOutcome | None:
    try:
        data = json.loads(payload)
        decision_data = data["decision"]
        decision = ProactiveContactReason(**decision_data)
        result_data = data.get("result")
        result = None
        if isinstance(result_data, dict):
            suggestions = [ReplySuggestion(**item) for item in result_data.get("suggestions", [])]
            result = ReplyGenerationResult(
                allowed=bool(result_data.get("allowed")),
                status=str(result_data.get("status") or ""),
                suggestions=suggestions,
                provider=str(result_data.get("provider") or ""),
                customer_summary=str(result_data.get("customer_summary") or ""),
                intent=str(result_data.get("intent") or ""),
                relevant_history=tuple(result_data.get("relevant_history") or ()),
                ask_first=str(result_data.get("ask_first") or ""),
                sales_judgement=str(result_data.get("sales_judgement") or ""),
                next_action=str(result_data.get("next_action") or ""),
                performance_trace=(
                    result_data.get("performance_trace")
                    if isinstance(result_data.get("performance_trace"), dict)
                    else {}
                ),
            )
        return ProactiveContactOutcome(request, decision, result, context_hash, True, build_ms)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _parse_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("response_not_object")
    return data


def _chat_completions_endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def _safe_provider_log(endpoint: str, started: float, status: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    elapsed_ms = int((monotonic_time.perf_counter() - started) * 1000)
    append_diagnostics_log(
        "ai_provider",
        f"target={parsed.scheme}://{parsed.netloc}{parsed.path} elapsed_ms={elapsed_ms} status=proactive:{status}",
    )


# Feedback V1 intentionally lives beside the proactive cache implementation.
# Keeping both writes in one module makes the cache-text replacement atomic and
# prevents this expression-only learning data from reaching fact repositories.
REJECTION_REASON_LABELS = {
    "too_formal": "太官方", "too_long": "太长", "too_salesy": "太像推销",
    "wrong_question": "问的问题不对", "missing_history": "没结合客户历史",
    "unnatural_salutation": "称呼不自然", "too_stiff": "太生硬", "other": "其他",
}
_REJECTION_PREFERENCES = {
    "too_formal": ("avoid_formal_tone", "少用客服腔和书面套话，像熟客微信沟通"),
    "too_long": ("prefer_short", "句子更短、更直接"),
    "too_salesy": ("avoid_early_sales_pitch", "开场先了解情况，不急着推产品"),
    "wrong_question": ("ask_one_specific_question", "每次只问一个具体问题"),
    "missing_history": ("mention_specific_history", "先引用之前聊过的具体事情再追问"),
    "unnatural_salutation": ("natural_salutation", "只使用自然、确定的熟客称呼"),
    "too_stiff": ("conversational_wording", "优先使用自然口语表达"),
}
_FORMAL_PHRASES = ("如果方便", "方便的话", "想请您说说", "请您说说", "请问", "烦请")
_NATURAL_ENDINGS = ("怎么样了", "缓过来没有", "现在咋样", "好点没有", "长得咋样")
_SALUTATION = re.compile(r"^(?:哥|姐|师傅|老板)[，,。\s]")
_DIRECTION_PREFERENCE_KEYS = frozenset({"confirm_crop_continuity", "prefer_actionable_need"})


@dataclass(frozen=True)
class SuggestionFeedbackRecord:
    feedback_id: str
    account_scope_id: str
    contact_id: str
    local_date: str
    context_hash: str
    suggestion_index: int
    reason_type: str
    original_text: str
    final_text: str
    feedback_type: str
    reason_code: str
    business_context_summary: str
    provider: str
    model: str
    generated_at: str
    accepted_at: str | None
    regeneration_requested_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class WritingPreference:
    key: str
    summary: str
    support_count: int
    level: str
    reason_type: str = ""


@dataclass(frozen=True)
class FeedbackViewState:
    statuses: dict[int, str]
    rejected: dict[int, tuple[str, str]]
    regeneration_used: bool


class SuggestionFeedbackRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(
        self, *, outcome: ProactiveContactOutcome, suggestion_index: int,
        feedback_type: str, original_text: str, final_text: str,
        reason_code: str, model: str, replace_cached_text: bool = False,
    ) -> SuggestionFeedbackRecord:
        _validate_feedback_input(suggestion_index, feedback_type, reason_code, final_text)
        now = utc_now_iso()
        request = outcome.request
        feedback_id = new_id("suggestion_feedback")
        accepted_at = now if feedback_type in {"accepted", "edited_and_accepted"} else None
        with self.db.connect() as conn:
            cache = conn.execute(
                """SELECT outcome_json,created_at FROM proactive_contact_cache
                WHERE account_scope_id=? AND contact_id=? AND local_date=? AND context_hash=?""",
                (request.account_scope_id, request.contact_id, request.local_date, outcome.context_hash),
            ).fetchone()
            conn.execute(
                """INSERT OR IGNORE INTO suggestion_feedback(
                    feedback_id,account_scope_id,contact_id,local_date,context_hash,
                    suggestion_kind,suggestion_index,reason_type,original_text,final_text,
                    feedback_type,reason_code,business_context_summary,provider,model,
                    generated_at,accepted_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,'proactive',?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    feedback_id, request.account_scope_id, request.contact_id, request.local_date,
                    outcome.context_hash, suggestion_index, outcome.decision.reason_type,
                    original_text, final_text, feedback_type, reason_code,
                    _safe_feedback_business_summary(outcome),
                    outcome.result.provider if outcome.result else "", model,
                    str(cache["created_at"]) if cache is not None else now,
                    accepted_at, now, now,
                ),
            )
            if replace_cached_text and cache is not None:
                payload = json.loads(str(cache["outcome_json"]))
                result = payload.get("result")
                suggestions = result.get("suggestions") if isinstance(result, dict) else None
                if (
                    not isinstance(suggestions, list)
                    or suggestion_index < 0
                    or suggestion_index >= len(suggestions)
                ):
                    raise ValueError("proactive_feedback_cache_invalid")
                item = suggestions[suggestion_index]
                if not isinstance(item, dict):
                    raise ValueError("proactive_feedback_cache_invalid")
                item["text"] = final_text
                conn.execute(
                    """UPDATE proactive_contact_cache SET outcome_json=?,updated_at=?
                    WHERE account_scope_id=? AND contact_id=? AND local_date=? AND context_hash=?""",
                    (
                        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        now, request.account_scope_id, request.contact_id,
                        request.local_date, outcome.context_hash,
                    ),
                )
            row = conn.execute(
                """SELECT * FROM suggestion_feedback
                WHERE account_scope_id=? AND contact_id=? AND local_date=? AND context_hash=?
                  AND suggestion_index=? AND feedback_type=? AND original_text=?
                  AND final_text=? AND reason_code=?""",
                (
                    request.account_scope_id, request.contact_id, request.local_date,
                    outcome.context_hash, suggestion_index, feedback_type,
                    original_text, final_text, reason_code,
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("proactive_feedback_write_failed")
        return _feedback_record(row)

    def canonical_original(self, outcome: ProactiveContactOutcome, suggestion_index: int) -> str:
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT original_text FROM suggestion_feedback
                WHERE account_scope_id=? AND contact_id=? AND local_date=? AND context_hash=?
                  AND suggestion_index=? AND feedback_type='edited_and_accepted'
                ORDER BY created_at ASC LIMIT 1""",
                (
                    outcome.request.account_scope_id, outcome.request.contact_id,
                    outcome.request.local_date, outcome.context_hash, suggestion_index,
                ),
            ).fetchone()
        return str(row["original_text"]) if row else ""

    def list_for_account(self, account_scope_id: str, *, limit: int = 500) -> list[SuggestionFeedbackRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM suggestion_feedback WHERE account_scope_id=? ORDER BY created_at DESC LIMIT ?",
                (account_scope_id, max(1, min(limit, 2000))),
            ).fetchall()
        return [_feedback_record(row) for row in rows]

    def list_for_context(self, outcome: ProactiveContactOutcome) -> list[SuggestionFeedbackRecord]:
        request = outcome.request
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM suggestion_feedback
                WHERE account_scope_id=? AND contact_id=? AND local_date=? AND context_hash=?
                ORDER BY created_at DESC,updated_at DESC""",
                (request.account_scope_id, request.contact_id, request.local_date, outcome.context_hash),
            ).fetchall()
        return [_feedback_record(row) for row in rows]

    def claim_regeneration(
        self, feedback_id: str, *, account_scope_id: str, contact_id: str,
        local_date: str, context_hash: str,
    ) -> str:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM suggestion_feedback
                WHERE feedback_id=? AND account_scope_id=? AND contact_id=?
                  AND local_date=? AND context_hash=? AND feedback_type='rejected'""",
                (feedback_id, account_scope_id, contact_id, local_date, context_hash),
            ).fetchone()
            if row is None:
                raise ValueError("proactive_regeneration_feedback_mismatch")
            used = conn.execute(
                """SELECT 1 FROM suggestion_feedback
                WHERE account_scope_id=? AND contact_id=? AND local_date=? AND context_hash=?
                  AND regeneration_requested_at IS NOT NULL LIMIT 1""",
                (account_scope_id, contact_id, local_date, context_hash),
            ).fetchone()
            if used is not None:
                raise ValueError("proactive_regeneration_limit_reached")
            conn.execute(
                """UPDATE suggestion_feedback SET regeneration_requested_at=?,updated_at=?
                WHERE feedback_id=? AND regeneration_requested_at IS NULL""",
                (now, now, feedback_id),
            )
        return str(row["reason_code"])


class SalesWritingPreferenceService:
    """Deterministic local statistics; never calls a model."""

    def __init__(self, feedback: SuggestionFeedbackRepository, *, stable_threshold: int = 3) -> None:
        self.feedback = feedback
        self.stable_threshold = max(3, stable_threshold)

    def preferences(self, account_scope_id: str, *, reason_type: str = "") -> list[WritingPreference]:
        rows = self.feedback.list_for_account(account_scope_id)
        global_support = _preference_support(rows)
        reason_support = _preference_support(row for row in rows if row.reason_type == reason_type)
        result: list[WritingPreference] = []
        for key, (summary, evidence) in global_support.items():
            count = len(evidence)
            distinct_customers = {value.split(":", 2)[1] for value in evidence if value.startswith("direction:")}
            stable = count >= self.stable_threshold and (
                key not in _DIRECTION_PREFERENCE_KEYS or len(distinct_customers) >= 2
            )
            level = "stable" if stable else "candidate"
            result.append(WritingPreference(key, summary, count, level))
        if reason_type:
            global_stable = {item.key for item in result if item.level == "stable"}
            for key, (summary, evidence) in reason_support.items():
                count = len(evidence)
                distinct_customers = {value.split(":", 2)[1] for value in evidence if value.startswith("direction:")}
                generalizable = key not in _DIRECTION_PREFERENCE_KEYS or len(distinct_customers) >= 2
                if count >= self.stable_threshold and generalizable and key not in global_stable:
                    result.append(WritingPreference(key, summary, count, "stable", reason_type))
        return sorted(result, key=lambda item: (item.level != "stable", -item.support_count, item.key))

    def stable_summaries(self, account_scope_id: str, reason_type: str = "") -> tuple[str, ...]:
        return tuple(
            item.summary for item in self.preferences(account_scope_id, reason_type=reason_type)
            if item.level == "stable"
        )[:6]


class ProactiveSuggestionFeedbackService:
    def __init__(self, repository: SuggestionFeedbackRepository, preferences: SalesWritingPreferenceService) -> None:
        self.repository = repository
        self.preferences = preferences

    def record_accepted(
        self, outcome: ProactiveContactOutcome, suggestion_index: int, *, model: str,
    ) -> SuggestionFeedbackRecord:
        suggestion = _feedback_suggestion(outcome, suggestion_index)
        original = self.repository.canonical_original(outcome, suggestion_index) or suggestion.text
        return self.repository.append(
            outcome=outcome, suggestion_index=suggestion_index, feedback_type="accepted",
            original_text=original, final_text=suggestion.text, reason_code="", model=model,
        )

    def record_edited_and_accepted(
        self, outcome: ProactiveContactOutcome, suggestion_index: int,
        final_text: str, *, model: str,
    ) -> tuple[SuggestionFeedbackRecord, ProactiveContactOutcome]:
        final_text = " ".join(final_text.strip().split())
        if not final_text or len(final_text) > 500:
            raise ValueError("proactive_feedback_final_text_invalid")
        suggestion = _feedback_suggestion(outcome, suggestion_index)
        original = self.repository.canonical_original(outcome, suggestion_index) or suggestion.text
        record = self.repository.append(
            outcome=outcome, suggestion_index=suggestion_index,
            feedback_type="edited_and_accepted", original_text=original,
            final_text=final_text, reason_code="", model=model, replace_cached_text=True,
        )
        suggestions = list(outcome.result.suggestions)
        suggestions[suggestion_index] = replace(suggestion, text=final_text)
        updated = replace(outcome, result=replace(outcome.result, suggestions=suggestions), cache_hit=True)
        return record, updated

    def record_rejected(
        self, outcome: ProactiveContactOutcome, suggestion_index: int,
        reason_code: str, *, model: str,
    ) -> SuggestionFeedbackRecord:
        suggestion = _feedback_suggestion(outcome, suggestion_index)
        return self.repository.append(
            outcome=outcome, suggestion_index=suggestion_index, feedback_type="rejected",
            original_text=suggestion.text, final_text="", reason_code=reason_code, model=model,
        )

    def view_state(self, outcome: ProactiveContactOutcome) -> FeedbackViewState:
        rows = self.repository.list_for_context(outcome)
        regeneration_used = any(row.regeneration_requested_at for row in rows)
        statuses: dict[int, str] = {}
        rejected: dict[int, tuple[str, str]] = {}
        suggestions = outcome.result.suggestions if outcome.result else []
        for row in rows:
            if row.suggestion_index in statuses or row.suggestion_index >= len(suggestions):
                continue
            current = suggestions[row.suggestion_index].text
            matches = row.final_text == current if row.final_text else row.original_text == current
            if not matches:
                continue
            if row.feedback_type == "edited_and_accepted":
                statuses[row.suggestion_index] = "已按你的修改保存"
            elif row.feedback_type == "accepted":
                statuses[row.suggestion_index] = "已记录采用"
            else:
                label = REJECTION_REASON_LABELS.get(row.reason_code, "不满意")
                statuses[row.suggestion_index] = f"已记录不满意：{label}"
                rejected[row.suggestion_index] = (row.feedback_id, row.reason_code)
        return FeedbackViewState(statuses, rejected, regeneration_used)


def _validate_feedback_input(index: int, feedback_type: str, reason_code: str, final_text: str) -> None:
    if index not in (0, 1):
        raise ValueError("proactive_feedback_index_invalid")
    if feedback_type not in {"accepted", "edited_and_accepted", "rejected"}:
        raise ValueError("proactive_feedback_type_invalid")
    if feedback_type == "rejected" and reason_code not in REJECTION_REASON_LABELS:
        raise ValueError("proactive_feedback_reason_invalid")
    if feedback_type != "rejected" and reason_code:
        raise ValueError("proactive_feedback_reason_unexpected")
    if feedback_type == "edited_and_accepted" and not final_text.strip():
        raise ValueError("proactive_feedback_final_text_invalid")


def _feedback_suggestion(outcome: ProactiveContactOutcome, index: int) -> ReplySuggestion:
    if outcome.result is None or not outcome.result.allowed or not outcome.result.suggestions:
        raise ValueError("proactive_feedback_outcome_unavailable")
    if index < 0 or index >= len(outcome.result.suggestions):
        raise ValueError("proactive_feedback_index_invalid")
    return outcome.result.suggestions[index]


def _safe_feedback_business_summary(outcome: ProactiveContactOutcome) -> str:
    parts = [
        outcome.decision.reason_summary,
        f"missing={outcome.decision.missing_key_variable}" if outcome.decision.missing_key_variable else "",
    ]
    return "；".join(value for value in parts if value)[:300]


def _feedback_record(row) -> SuggestionFeedbackRecord:
    return SuggestionFeedbackRecord(
        feedback_id=str(row["feedback_id"]), account_scope_id=str(row["account_scope_id"]),
        contact_id=str(row["contact_id"]), local_date=str(row["local_date"]),
        context_hash=str(row["context_hash"]), suggestion_index=int(row["suggestion_index"]),
        reason_type=str(row["reason_type"]), original_text=str(row["original_text"]),
        final_text=str(row["final_text"]), feedback_type=str(row["feedback_type"]),
        reason_code=str(row["reason_code"]),
        business_context_summary=str(row["business_context_summary"]),
        provider=str(row["provider"]), model=str(row["model"]),
        generated_at=str(row["generated_at"]),
        accepted_at=str(row["accepted_at"]) if row["accepted_at"] else None,
        regeneration_requested_at=(
            str(row["regeneration_requested_at"]) if row["regeneration_requested_at"] else None
        ),
        created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
    )


def _preference_support(
    rows: Iterable[SuggestionFeedbackRecord],
) -> dict[str, tuple[str, set[str]]]:
    support: dict[str, tuple[str, set[str]]] = {}

    def add(key: str, summary: str, feedback_id: str) -> None:
        if key not in support:
            support[key] = (summary, set())
        support[key][1].add(feedback_id)

    for row in rows:
        explicit = _REJECTION_PREFERENCES.get(row.reason_code)
        if explicit:
            add(explicit[0], explicit[1], row.feedback_id)
        positive = row.final_text or (row.original_text if row.feedback_type == "accepted" else "")
        if not positive:
            continue
        if len(positive) <= 44:
            add("prefer_short", "句子更短、更直接", row.feedback_id)
        if positive.count("？") + positive.count("?") == 1:
            add("ask_one_specific_question", "每次只问一个具体问题", row.feedback_id)
        if _SALUTATION.search(positive):
            add("natural_salutation", "只使用自然、确定的熟客称呼", row.feedback_id)
        if positive.endswith(_NATURAL_ENDINGS):
            add("conversational_wording", "优先使用自然口语表达", row.feedback_id)
        if row.feedback_type != "edited_and_accepted":
            continue
        # Learn only clear directional corrections, with deduped contexts. The
        # summaries are conditional methods, never customer crops or symptoms.
        original = "".join(row.original_text.split())
        edited = "".join(positive.split())
        direction_evidence = f"direction:{row.contact_id}:{row.context_hash}"
        if _contains_any(original, ("管理到哪一步", "到哪个阶段", "现在什么阶段")):
            if _contains_any(edited, ("还是", "仍然", "还在种")) and _extract_crops(original) == _extract_crops(edited) and _extract_crops(edited):
                add(
                    "confirm_crop_continuity",
                    "仅有旧作物线索时先确认这茬是否仍种；有可靠本季资料时承接已有情况，不重复确认",
                    direction_evidence,
                )
            elif _contains_any(edited, ("长势", "黄叶", "早衰", "用肥", "用药", "下茬", "明年", "种植安排")):
                add(
                    "prefer_actionable_need",
                    "已有当前作物或阶段依据时，围绕实际管理需求或后续安排寻找销售切口；具体问题随客户变化，不固定问某种症状",
                    direction_evidence,
                )
        if len(positive) <= max(8, int(len(row.original_text) * 0.85)):
            add("prefer_short", "句子更短、更直接", row.feedback_id)
        if any(phrase in row.original_text and phrase not in positive for phrase in _FORMAL_PHRASES):
            add("avoid_formal_tone", "少用客服腔和书面套话，像熟客微信沟通", row.feedback_id)
    return support
