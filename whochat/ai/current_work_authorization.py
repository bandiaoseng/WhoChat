from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from whochat.core.current_conversation import CurrentMessageView, WINDOW_REACTIVATION_SOURCE
from whochat.core.models import ContactStatus, ConversationType, LEGACY_ACCOUNT_SCOPE_ID, MemoryStatus, Speaker
from whochat.core.runtime import PageType, WindowState

if TYPE_CHECKING:
    from whochat.ai.models import ReplyContext


@dataclass(frozen=True)
class CurrentWorkCloudAuthorization:
    """One current-customer reply/next-step grant, never a Contact permission.

    Reply was approved on 2026-10-15; current-customer next-step work was
    explicitly approved on 2026-10-16. Neither applies to background screening.
    Only the UI's live identity-lease check may issue it. Memory extraction,
    Historical and idle prewarming do not receive this capability.
    """

    binding: tuple[str, ...]
    target_instance_id: str = ""


def current_context_source_error(context: ReplyContext) -> str | None:
    """Validate visible-source provenance even with permanent cloud permission.

    A verified memory marker is issued only by the local DB-backed builder;
    restored JSON cannot issue it.  This pure check is not the live lease or
    the DB recheck, which the service must run before and after Provider work.
    Formal-message contexts keep their existing compatibility contract.
    """
    try:
        view = getattr(context, "current_message_view", None)
        memory = getattr(context, "memory_context", None)
        mode = getattr(context, "current_work_mode", "reply")
        if mode not in {"reply", "next_step", "reactivation"}:
            return "current_work_mode_invalid"
        if view is None:
            if mode != "reply":
                return "current_work_mode_requires_visible_source"
            if memory is not None and (
                memory.current_view_revision or memory.current_view_source_ref
                or memory.current_view_source_kind or memory.current_view_verified
            ):
                return "current_view_missing"
            return None
        if type(view) is not CurrentMessageView:
            return "current_view_type_invalid"
        window_only=view.source_kind == WINDOW_REACTIVATION_SOURCE
        from whochat.services.communication_intent import select_communication_intent
        closed_shipping = (not window_only and mode == 'reactivation'
                           and select_communication_intent(view).reason == 'old_outgoing_shipment_topic_closed')
        if (window_only and (mode != "reactivation" or view.messages or view.current_message is not None)
                or not window_only and ((mode == "reactivation" and not closed_shipping) or view.current_message is None
                    or (mode == "next_step" and view.current_message.speaker != Speaker.ME))):
            return "current_work_mode_source_mismatch"
        if not view.current_screen_runtime_matches(context.runtime):
            return "current_view_screen_runtime_mismatch"
        contact, runtime = context.contact, context.runtime
        if (
            contact is None or contact.platform.casefold() != "wechat"
            or contact.conversation_type != ConversationType.DM
            or contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED}
            or contact.merged_into or not contact.account_scope_id
            or contact.account_scope_id == LEGACY_ACCOUNT_SCOPE_ID
            or runtime.page.page_type != PageType.CHAT_DM or runtime.page.confidence < .65
            or runtime.window.state != WindowState.VISIBLE
            or not runtime.window.hwnd or not runtime.window.process_id
            or not runtime.account_identity_runtime_key
            or (context.strategy is not None and context.strategy.requires_manual_reply)
        ):
            return "current_view_identity_or_runtime_invalid"
        if (
            memory is None or memory.status not in {"available", "empty"}
            or memory.current_view_verified is not True
            or memory.account_scope_id != contact.account_scope_id
            or memory.contact_id != contact.id
            or memory.conversation_id != view.conversation_id
            or memory.current_view_revision != view.content_revision
            or memory.current_view_source_ref != view.observation_id
            or memory.current_view_source_kind != view.source_kind
            or context.current_message_id or memory.current_message_id
        ):
            return "current_view_memory_binding_invalid"
        if not view.validate_binding(
            expected_account_scope_id=contact.account_scope_id, expected_contact_id=contact.id,
            expected_conversation_id=memory.conversation_id,
            expected_target_instance_id=view.target_instance_id, expected_session_id=view.session_id,
            max_source_job_id=view.source_job_id, max_capture_generation=runtime.capture_generation,
            now=datetime.now(timezone.utc),
        ):
            return "current_view_evidence_invalid"
        if any(
            item.contact_id != contact.id or item.account_scope_id != contact.account_scope_id
            or (item.conversation_id and item.conversation_id != view.conversation_id)
            for item in context.messages
        ):
            return "current_view_formal_history_owner_invalid"
        if any(item.contact_id != contact.id or item.status != MemoryStatus.CONFIRMED for item in context.memories):
            return "current_view_confirmed_memory_owner_invalid"
        return None
    except (AttributeError, TypeError, ValueError):
        return "current_view_malformed"


def _binding(context: ReplyContext) -> tuple[str, ...] | None:
    contact = context.contact
    runtime = context.runtime
    memory = context.memory_context
    view = context.current_message_view
    if (
        contact is None
        or contact.platform.casefold() != "wechat"
        or contact.conversation_type != ConversationType.DM
        or contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED}
        or contact.merged_into
        or not contact.account_scope_id
        or contact.account_scope_id == LEGACY_ACCOUNT_SCOPE_ID
        or runtime.page.page_type != PageType.CHAT_DM
        or runtime.page.confidence < 0.65
        or runtime.window.state != WindowState.VISIBLE
        or not runtime.window.hwnd
        or not runtime.window.process_id
        or not runtime.account_identity_runtime_key
        or memory is None
        or memory.account_scope_id != contact.account_scope_id
        or memory.contact_id != contact.id
        or not memory.conversation_id
    ):
        return None
    if current_context_source_error(context) is not None:
        return None
    if context.strategy is not None and context.strategy.requires_manual_reply:
        return None
    if view is not None:
        return (
            "visible_observation", context.current_work_mode, contact.account_scope_id, contact.id, view.conversation_id,
            view.content_revision, view.observation_id, view.target_instance_id, view.session_id,
            str(view.source_job_id), str(view.capture_generation), memory.memory_hash,
            str(runtime.window.hwnd), str(runtime.window.process_id), str(runtime.window_generation),
            str(runtime.capture_generation), str(runtime.account_identity_epoch),
            runtime.account_identity_runtime_key,
        )
    if (
        not context.current_message_id or memory.current_message_id != context.current_message_id
        or not any(message.id == context.current_message_id for message in context.messages)
    ):
        return None
    if any(
        message.contact_id != contact.id
        or message.account_scope_id != contact.account_scope_id
        or (message.conversation_id and message.conversation_id != memory.conversation_id)
        for message in context.messages
    ):
        return None
    return (
        context.current_work_mode,
        contact.account_scope_id,
        contact.id,
        memory.conversation_id,
        context.current_message_id,
        str(runtime.window.hwnd),
        str(runtime.window.process_id),
        str(runtime.window_generation),
        str(runtime.capture_generation),
        str(runtime.account_identity_epoch),
        runtime.account_identity_runtime_key,
    )


def authorize_current_work_context(
    context: ReplyContext, *, identity_lease_valid: bool
) -> ReplyContext:
    binding = _binding(context) if identity_lease_valid else None
    grant = CurrentWorkCloudAuthorization(
        binding, context.current_message_view.target_instance_id if context.current_message_view is not None else ""
    ) if binding is not None else None
    return replace(context, current_work_cloud_authorization=grant)


def current_work_context_authorized(context: ReplyContext) -> bool:
    grant = context.current_work_cloud_authorization
    binding = _binding(context)
    return bool(
        isinstance(grant, CurrentWorkCloudAuthorization)
        and binding is not None
        and grant.binding == binding
        and grant.target_instance_id == (
            context.current_message_view.target_instance_id if context.current_message_view is not None else ""
        )
    )


def cloud_reply_authorized(context: ReplyContext) -> bool:
    return bool(
        context.contact is not None
        and current_context_source_error(context) is None
        and (context.contact.allow_cloud_ai or current_work_context_authorized(context))
    )
