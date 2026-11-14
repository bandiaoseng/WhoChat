from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


LEGACY_ACCOUNT_SCOPE_ID = "account_scope_legacy_unscoped"


class ConversationType(StrEnum):
    DM = "dm"
    GROUP = "group"
    OFFICIAL = "official"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ContactStatus(StrEnum):
    UNCONFIRMED = "unconfirmed"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    MERGED = "merged"
    IGNORED = "ignored"


class IdentityStatus(StrEnum):
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class Speaker(StrEnum):
    ME = "me"
    OTHER = "other"
    MEMBER = "member"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class MemoryKind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    AVOID = "avoid"
    TEMPORARY = "temporary"
    SUMMARY = "summary"


class MemoryStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class CustomerSalesStage(StrEnum):
    NEW = "new"
    NEEDS_IDENTIFIED = "needs_identified"
    SOLUTION_PROVIDED = "solution_provided"
    PRODUCT_RECOMMENDED = "product_recommended"
    QUOTED = "quoted"
    CONSIDERING = "considering"
    ORDERED = "ordered"
    USING = "using"
    RESULT_FOLLOWUP = "result_followup"
    REPEAT_CUSTOMER = "repeat_customer"
    DORMANT = "dormant"


class CustomerSalesStageSource(StrEnum):
    SYSTEM_DERIVED = "system_derived"
    USER_EXPLICIT = "user_explicit"


class GrowingContextStatus(StrEnum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    UNCERTAIN = "uncertain"


class CustomerFactProvenance(StrEnum):
    CUSTOMER_EXPLICIT = "customer_explicit"
    USER_EXPLICIT = "user_explicit"
    AI_INFERRED = "ai_inferred"
    SYSTEM_IMPORTED = "system_imported"


class CustomerFactStatus(StrEnum):
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"
    INVALID = "invalid"


class CustomerEventType(StrEnum):
    PROBLEM_REPORTED = "problem_reported"
    DIAGNOSIS = "diagnosis"
    SOLUTION_GIVEN = "solution_given"
    PRODUCT_RECOMMENDED = "product_recommended"
    DOSAGE_GIVEN = "dosage_given"
    QUOTE_GIVEN = "quote_given"
    ORDER_INTENT = "order_intent"
    ORDER_PLACED = "order_placed"
    PURCHASED = "purchased"
    USAGE_STARTED = "usage_started"
    EFFECT_FEEDBACK = "effect_feedback"
    COMPLAINT = "complaint"
    FOLLOWUP = "followup"
    CUSTOMER_CONCERN = "customer_concern"
    MANAGEMENT_PLAN = "management_plan"
    OTHER = "other"


class CustomerEventStatus(StrEnum):
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    SUPERSEDED = "superseded"
    INVALID = "invalid"


class CustomerFollowupStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CustomerFollowupSource(StrEnum):
    EXPLICIT = "explicit"
    SYSTEM_RULE = "system_rule"
    USER_CREATED = "user_created"


class MemoryExtractionJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    NO_MEMORY = "no_memory"
    BLOCKED = "blocked"
    FAILED = "failed"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    goal: str
    mode: str
    tone: str
    avoid: str
    reply_variants: str
    requires_manual_reply: bool
    created_at: str
    updated_at: str
    archived: bool = False


@dataclass(frozen=True)
class Contact:
    id: str
    platform: str
    display_name: str
    conversation_type: ConversationType
    status: ContactStatus
    strategy_id: str
    allow_cloud_ai: bool
    remark: str
    avatar_fingerprint: str
    merged_into: str | None
    created_at: str
    updated_at: str
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID


@dataclass(frozen=True)
class ContactAlias:
    id: str
    contact_id: str
    alias: str
    source: str
    created_at: str


@dataclass(frozen=True)
class Person:
    id: str
    display_name: str
    status: IdentityStatus
    remark: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class PersonAlias:
    id: str
    person_id: str
    alias: str
    source: str
    created_at: str


@dataclass(frozen=True)
class ContactPersonLink:
    id: str
    contact_id: str
    person_id: str
    confidence: float
    source: str
    verified: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class GroupMember:
    id: str
    group_contact_id: str
    member_display_name: str
    person_id: str | None
    platform_contact_id: str | None
    confidence: float
    source: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: str
    contact_id: str
    speaker: Speaker
    text: str
    content_type: str
    ocr_confidence: float | None
    observed_at: str
    message_time: str | None
    time_source: str
    partial: bool
    fingerprint: str
    source: str
    sender_name: str = ""
    conversation_id: str | None = None
    session_id: str | None = None
    segment_id: str | None = None
    sequence_no: int | None = None
    segment_index: int | None = None
    bbox_json: str = ""
    order_uncertain: bool = False
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID


@dataclass(frozen=True)
class AccountScope:
    id: str
    platform: str
    local_name: str
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TargetInstance:
    id: str
    platform: str
    account_scope_id: str | None
    runtime_key: str
    visual_hash: str
    visual_reliable: bool
    process_name: str
    executable_path: str
    last_pid: int | None
    last_hwnd: int | None
    last_rect_json: str
    binding_source: str
    confidence: float
    first_seen_at: str
    last_seen_at: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ContactIdentityBinding:
    id: str
    platform: str
    conversation_type: ConversationType
    normalized_title: str
    contact_id: str
    avatar_fingerprint: str
    confidence: float
    verified: bool
    last_confirmed_at: str
    created_at: str
    updated_at: str
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID


@dataclass(frozen=True)
class Conversation:
    id: str
    contact_id: str
    platform: str
    conversation_type: ConversationType
    identity_key: str
    created_at: str
    updated_at: str
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID


@dataclass(frozen=True)
class ConversationSession:
    id: str
    conversation_id: str
    hwnd: int | None
    stable_title: str
    identity_confidence: float
    started_at: str
    last_seen_at: str
    ended_at: str | None
    source: str
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID
    target_instance_id: str = ""
    capture_generation: int = 0


@dataclass(frozen=True)
class ConversationSegment:
    id: str
    conversation_id: str
    session_id: str
    snapshot_hash: str
    observed_at: str
    page_type: str
    page_confidence: float
    ocr_engine: str
    reliable: bool
    order_uncertain: bool
    source_job_id: int | None
    message_count: int
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID
    target_instance_id: str = ""


@dataclass(frozen=True)
class AIAnalysisEvent:
    id: str
    message_id: str
    contact_id: str
    session_id: str | None
    segment_id: str | None
    status: str
    context_hash: str
    created_at: str
    updated_at: str
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID
    target_instance_id: str = ""
    conversation_id: str | None = None
    trigger_snapshot_json: str = "{}"
    started_at: str | None = None
    completed_at: str | None = None
    failure_reason: str = ""
    result_json: str = "{}"


@dataclass(frozen=True)
class ConversationObservation:
    id: str
    account_scope_id: str
    target_instance_id: str
    conversation_id: str
    session_id: str
    snapshot_hash: str
    observed_at: str
    status: str
    relation: str
    reason: str
    messages_json: str
    keys_json: str
    source_job_id: int | None
    capture_generation: int
    expires_at: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Memory:
    id: str
    contact_id: str
    kind: MemoryKind
    status: MemoryStatus
    content: str
    confidence: float | None
    source_message_id: str | None
    expires_at: str | None
    created_at: str
    updated_at: str
    # Content fingerprint (contact + kind + normalised content).  The unique index
    # ``idx_memories_fingerprint`` makes repeated extraction idempotent, so the same
    # fact observed again on a later screen does not become a second memory row.
    fingerprint: str = ""


@dataclass(frozen=True)
class CustomerProfile:
    id: str
    account_scope_id: str
    target_instance_id: str | None
    contact_id: str
    conversation_id: str | None
    display_name: str
    sales_stage: CustomerSalesStage
    sales_stage_source: CustomerSalesStageSource
    sales_stage_updated_at: str | None
    latest_contact_at: str | None
    next_followup_at: str | None
    created_at: str
    updated_at: str
    history_status: str = "history_not_initialized"
    history_last_imported_at: str | None = None
    history_last_backfilled_at: str | None = None
    history_status_detail: str = ""


@dataclass(frozen=True)
class CustomerGrowingContext:
    id: str
    customer_profile_id: str
    crop: str
    variety: str
    region: str
    acreage_value: float | None
    acreage_unit: str
    growth_stage: str
    cultivation_type: str
    season_or_year: str
    status: GrowingContextStatus
    first_seen_at: str
    last_confirmed_at: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CustomerFact:
    id: str
    customer_profile_id: str
    growing_context_id: str | None
    fact_category: str
    fact_key: str
    value_text: str
    value_json: str
    normalized_value: str
    confidence: float
    provenance: CustomerFactProvenance
    status: CustomerFactStatus
    first_seen_at: str
    last_confirmed_at: str
    valid_from: str | None
    valid_to: str | None
    conflict_group_id: str | None
    superseded_by_fact_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CustomerFactSource:
    id: str
    fact_id: str
    message_id: str | None
    conversation_id: str | None
    source_text: str
    observed_at: str
    created_at: str
    historical_message_id: str | None = None


@dataclass(frozen=True)
class CustomerEvent:
    id: str
    customer_profile_id: str
    growing_context_id: str | None
    event_type: CustomerEventType
    title: str
    summary: str
    structured_data_json: str
    event_at: str
    confidence: float
    provenance: CustomerFactProvenance
    status: CustomerEventStatus
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CustomerEventSource:
    id: str
    event_id: str
    message_id: str | None
    conversation_id: str | None
    source_text: str
    observed_at: str
    created_at: str
    historical_message_id: str | None = None


@dataclass(frozen=True)
class CustomerFollowup:
    id: str
    customer_profile_id: str
    growing_context_id: str | None
    related_event_id: str | None
    followup_type: str
    title: str
    due_at: str
    status: CustomerFollowupStatus
    priority: int
    completed_at: str | None
    cancelled_at: str | None
    source: CustomerFollowupSource
    source_message_id: str | None
    reason: str
    created_at: str
    updated_at: str
    source_historical_message_id: str | None = None


@dataclass(frozen=True)
class CustomerSalesStageHistory:
    id: str
    customer_profile_id: str
    old_stage: CustomerSalesStage
    new_stage: CustomerSalesStage
    reason: str
    source_event_id: str | None
    source: CustomerSalesStageSource
    changed_at: str


@dataclass(frozen=True)
class CustomerMemoryEdit:
    id: str
    customer_profile_id: str
    entity_type: str
    entity_id: str
    operation: str
    old_value_json: str
    new_value_json: str
    reason: str
    actor: str
    created_at: str


@dataclass(frozen=True)
class MemoryExtractionJob:
    id: str
    source_message_id: str
    source_message_rowid: int
    account_scope_id: str
    target_instance_id: str | None
    contact_id: str
    conversation_id: str | None
    customer_profile_id: str | None
    status: MemoryExtractionJobStatus
    provider: str
    model: str
    input_hash: str
    input_context_json: str
    output_json: str
    write_result_json: str
    error: str
    ai_call_count: int
    provider_elapsed_ms: int | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    updated_at: str
    provider_started_at: str | None = None
    provider_completed_at: str | None = None
    write_started_at: str | None = None
    write_completed_at: str | None = None


@dataclass(frozen=True)
class AppLog:
    id: str
    ts: str
    level: str
    module: str
    event: str
    message: str
    context_json: str


@dataclass(frozen=True)
class GenerationLog:
    id: str
    ts: str
    contact_id: str | None
    strategy_id: str | None
    provider: str
    model: str
    allowed: bool
    status: str
    suggestion_count: int
    risk_summary: str
    context_hash: str
    page_type: str
    page_confidence: float
    message_count: int
    memory_count: int


@dataclass(frozen=True)
class ReplyFeedback:
    id: str
    ts: str
    contact_id: str | None
    strategy_id: str | None
    provider: str
    status: str
    suggestion_label: str
    suggestion_text_preview: str
    risk: str
    feedback: str
    context_hash: str
    page_type: str
    message_count: int
    memory_count: int


@dataclass(frozen=True)
class SettingsAudit:
    id: str
    ts: str
    actor: str
    scope: str
    changes_json: str
    secret_backend: str


@dataclass(frozen=True)
class CaptureSample:
    id: str
    ts: str
    job_id: int
    hwnd: int | None
    target_app: str
    app_label: str
    snapshot_hash: str
    image_path: str
    ocr_image_path: str
    crop_rect_json: str
    ocr_engine: str
    ocr_warning: str
    page_type: str
    page_confidence: float
    message_count: int
    retained_image: bool
    title_ocr_image_path: str = ""
    title_crop_rect_json: str = ""
    title_ocr_elapsed_ms: int | None = None
    content_ocr_elapsed_ms: int | None = None
    total_elapsed_ms: int | None = None
    capture_elapsed_ms: int | None = None
    layout_elapsed_ms: int | None = None
    title_input_elapsed_ms: int | None = None
    content_input_elapsed_ms: int | None = None
    title_preprocess_elapsed_ms: int | None = None
    title_inference_elapsed_ms: int | None = None
    title_postprocess_elapsed_ms: int | None = None
    content_preprocess_elapsed_ms: int | None = None
    content_inference_elapsed_ms: int | None = None
    content_postprocess_elapsed_ms: int | None = None
    verify_capture_elapsed_ms: int | None = None
    page_classification_elapsed_ms: int | None = None
    title_cache_hit: bool = False
