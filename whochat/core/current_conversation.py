"""A bounded current-window view, never a formal message or incoming event.

The factory consumes an already persisted observation and its verified
conversation owner.  It does no IO and does not establish a live identity lease:
callers must additionally validate the current account/window lifecycle before
dispatching or displaying work.  Capture evidence and business revision are
deliberately separate so a repaint does not become a new customer message.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any

from whochat.core.models import (
    Conversation, ConversationObservation, ConversationType,
    LEGACY_ACCOUNT_SCOPE_ID, Speaker,
)


_TRUSTED_STATUSES = frozenset({"baseline", "confirmed", "promoted"})
_TRUSTED_RELATIONS = frozenset({
    "baseline", "known", "history", "append", "append_after_quarantine",
    "quarantine_confirmed", "duplicate_snapshot", "duplicate_segment",
})
_RELIABLE_TIME_SOURCES = frozenset({"ocr", "verified", "platform", "source"})
_MAX_MESSAGES = 64
_MAX_TEXT = 8000
_MAX_JSON = 512_000
SCREEN_CAPTURE_SCHEMA = "whochat.current_screen_capture.v1"
SCREEN_EVIDENCE_SCHEMA = "whochat.current_screen_evidence.v1"
SCREEN_TTL_SECONDS = 120
_SCREEN_KEY = "_current_screen_evidence"
WINDOW_REACTIVATION_SOURCE = "current_window_reactivation"
WINDOW_OBSERVATION_SCHEMA = "whochat.current_window_observation.v1"
_OVERLAP_ONLY_REASONS = frozenset({
    "persistent_tail_fallback_uncertain:runtime_anchor_missing",
    "persistent_tail_fallback_uncertain:runtime_anchor_expired",
})


@dataclass(frozen=True)
class VisibleMessage:
    """One ordered occurrence; intentionally has no ``id`` or Message adapter."""

    ordinal: int
    speaker: Speaker
    text: str
    content_type: str = "text"
    sender_name: str = ""
    message_time: str | None = None
    time_source: str = "unknown"
    time_text: str = ""
    media_refs: tuple[str, ...] = ()
    evidence_status: str = ""
    partial: bool = False
    order_uncertain: bool = False

    def business_payload(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal, "speaker": self.speaker.value,
            "text": self.text, "content_type": self.content_type,
            "sender_name": self.sender_name, "message_time": self.message_time,
            "time_source": self.time_source, "time_text": self.time_text,
            "media_refs": list(self.media_refs), "evidence_status": self.evidence_status,
            "partial": self.partial, "order_uncertain": self.order_uncertain,
        }


@dataclass(frozen=True)
class CurrentMessageView:
    account_scope_id: str
    contact_id: str
    conversation_id: str
    target_instance_id: str
    session_id: str
    messages: tuple[VisibleMessage, ...]
    observation_status: str
    relation: str
    content_revision: str
    observation_id: str
    source_job_id: int
    capture_generation: int
    observed_at: str
    expires_at: str
    source_kind: str = "visible_observation"
    screen_evidence_json: str = ""
    source_reason: str = ""
    window_content_digest: str = ""

    def current_screen_runtime_matches(self, runtime: Any) -> bool:
        """Capture proof is not a live lease; it must still match this runtime."""
        if not self.screen_evidence_json:
            return True
        try:
            proof = json.loads(self.screen_evidence_json)
            return all(proof[key] == value for key, value in {
                "hwnd": runtime.window.hwnd, "process_id": runtime.window.process_id,
                "window_generation": runtime.window_generation,
                "capture_generation": runtime.capture_generation,
                "account_identity_epoch": runtime.account_identity_epoch,
                "account_identity_runtime_key": runtime.account_identity_runtime_key,
            }.items())
        except (ValueError, TypeError, KeyError, AttributeError):
            return False

    @property
    def current_message(self) -> VisibleMessage | None:
        return self.messages[-1] if self.messages else None

    @property
    def has_unanswered_customer_message(self) -> bool:
        """A simple routing hint, not proof of newness or a sales judgement."""
        return bool(self.current_message and self.current_message.speaker == Speaker.OTHER)

    def business_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "whochat.current_message_view.v1",
            "account_scope_id": self.account_scope_id, "contact_id": self.contact_id,
            "conversation_id": self.conversation_id, "source_kind": self.source_kind,
            "observation_status": self.observation_status,
            "messages": [item.business_payload() for item in self.messages],
        }
        if self.source_kind == WINDOW_REACTIVATION_SOURCE:
            payload["window_content_digest"] = self.window_content_digest
        return payload

    def reply_business_payload(self) -> dict[str, Any]:
        """Stable reply/cache semantics, never a substitute for source proof.

        ``baseline`` becoming ``confirmed`` describes observation ingestion,
        not a new customer message.  Keep the original status-bearing payload
        and ``content_revision`` for source reconstruction/lease validation;
        only a reply business digest uses this versioned projection.  Message
        content, uncertainty, ownership and source kind remain unchanged.
        """
        payload = self.business_payload()
        payload["schema"] = "whochat.reply_business_view.v1"
        payload.pop("observation_status")
        return payload

    @property
    def reply_business_revision(self) -> str:
        """A semantic hash with no authority to validate or refresh a lease."""
        return _revision(self.reply_business_payload())

    def validate_binding(
        self, *, expected_account_scope_id: str, expected_contact_id: str,
        expected_conversation_id: str, expected_target_instance_id: str,
        expected_session_id: str, max_source_job_id: int,
        max_capture_generation: int, now: datetime,
    ) -> bool:
        """Check the submitted view; a separate live identity lease is required."""
        try:
            _validate_evidence(
                self, expected_account_scope_id=expected_account_scope_id,
                expected_contact_id=expected_contact_id,
                expected_conversation_id=expected_conversation_id,
                expected_target_instance_id=expected_target_instance_id,
                expected_session_id=expected_session_id,
                max_source_job_id=max_source_job_id,
                max_capture_generation=max_capture_generation, now=now,
            )
            if self.source_kind == WINDOW_REACTIVATION_SOURCE:
                proof = json.loads(self.screen_evidence_json)
                return (type(self.messages) is tuple and not self.messages
                        and self.window_content_digest == proof["window_body_sha256"]
                        and self.content_revision == _revision(self.business_payload()))
            return (
                self.source_kind == "visible_observation"
                and type(self.messages) is tuple
                and 0 < len(self.messages) <= _MAX_MESSAGES
                and all(_message_is_valid(item, index) for index, item in enumerate(self.messages))
                and self.content_revision == _revision(self.business_payload())
                and (not self.screen_evidence_json or json.loads(self.screen_evidence_json)["business_rows_sha256"]
                     == _revision([item.business_payload() for item in self.messages]))
            )
        except (ValueError, TypeError, AttributeError, KeyError):
            return False


def build_current_message_view(
    observation: ConversationObservation, *, conversation: Conversation,
    expected_account_scope_id: str, expected_contact_id: str,
    expected_conversation_id: str, expected_target_instance_id: str,
    expected_session_id: str, max_source_job_id: int,
    max_capture_generation: int, now: datetime,
) -> CurrentMessageView:
    """Validate ownership and snapshot evidence without creating Message IDs.

    Source job/capture are upper bounds: semantic observation deduplication can
    legitimately return an older evidence row from this exact session.  This
    never permits another session/target, expired evidence or a future job.
    """
    if type(observation) is not ConversationObservation or type(conversation) is not Conversation:
        raise ValueError("current_view_persisted_source_required")
    if (
        conversation.id != expected_conversation_id
        or conversation.contact_id != expected_contact_id
        or conversation.account_scope_id != expected_account_scope_id
        or conversation.platform.casefold() != "wechat"
        or conversation.conversation_type != ConversationType.DM
    ):
        raise ValueError("current_view_conversation_owner_mismatch")
    proof = current_screen_evidence(observation)
    _validate_evidence(
        observation, expected_account_scope_id=expected_account_scope_id,
        expected_contact_id=expected_contact_id,
        expected_conversation_id=expected_conversation_id,
        expected_target_instance_id=expected_target_instance_id,
        expected_session_id=expected_session_id, max_source_job_id=max_source_job_id,
        max_capture_generation=max_capture_generation, now=now, screen_proof=proof,
    )
    if proof is not None and proof.get("source_kind") == WINDOW_REACTIVATION_SOURCE:
        if not isinstance(observation.messages_json,str) or len(observation.messages_json)>_MAX_JSON:
            raise ValueError("current_window_envelope_invalid")
        _validate_screen_evidence(proof, observation, expected_contact_id, now)
        envelope = json.loads(observation.messages_json)
        if (not isinstance(envelope, dict) or envelope.get("schema") != WINDOW_OBSERVATION_SCHEMA
                or envelope.get("messages") != [] or set(envelope) != {"schema", "messages", "screen_evidence"}):
            raise ValueError("current_window_envelope_invalid")
        values = dict(account_scope_id=expected_account_scope_id, contact_id=expected_contact_id,
            conversation_id=expected_conversation_id, target_instance_id=expected_target_instance_id,
            session_id=expected_session_id, messages=(), observation_status=observation.status,
            relation=observation.relation, observation_id=observation.id, source_job_id=observation.source_job_id,
            capture_generation=observation.capture_generation, observed_at=observation.observed_at,
            expires_at=observation.expires_at, source_kind=WINDOW_REACTIVATION_SOURCE,
            screen_evidence_json=json.dumps(proof,ensure_ascii=False,sort_keys=True,separators=(",",":")),
            source_reason=observation.reason, window_content_digest=proof["window_body_sha256"])
        draft = CurrentMessageView(content_revision="", **values)
        return CurrentMessageView(content_revision=_revision(draft.business_payload()), **values)
    if not isinstance(observation.messages_json, str) or len(observation.messages_json) > _MAX_JSON:
        raise ValueError("current_view_messages_payload_invalid")
    try:
        rows = json.loads(observation.messages_json)
    except (ValueError, TypeError) as exc:
        raise ValueError("current_view_messages_payload_invalid") from exc
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_MESSAGES:
        raise ValueError("current_view_message_count_invalid")
    items: list[VisibleMessage] = []
    for row in rows:
        item = _visible_message(row, len(items))
        if item.text or item.media_refs:
            items.append(item)
    if not items:
        raise ValueError("current_view_messages_empty")
    if proof is not None:
        _validate_screen_evidence(proof, observation, expected_contact_id, now, rows=rows)
        evidence_json = json.dumps(proof, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        evidence_json = ""
    values = dict(
        account_scope_id=expected_account_scope_id, contact_id=expected_contact_id,
        conversation_id=expected_conversation_id, target_instance_id=expected_target_instance_id,
        session_id=expected_session_id, messages=tuple(items),
        observation_status=observation.status, relation=observation.relation,
        observation_id=observation.id, source_job_id=observation.source_job_id,
        capture_generation=observation.capture_generation, observed_at=observation.observed_at,
        expires_at=observation.expires_at,
        screen_evidence_json=evidence_json, source_reason=observation.reason if evidence_json else "",
    )
    draft = CurrentMessageView(content_revision="", **values)
    return CurrentMessageView(content_revision=_revision(draft.business_payload()), **values)


def observation_semantic_payload(observation: ConversationObservation) -> dict[str, Any] | None:
    """Normalize observation business content for scoped repository deduping.

    This is NOT an authorization or current-view constructor.  The repository
    must independently match owner/target/session, status and freshness before
    reusing a row.  Uncertainty is preserved here, never upgraded or discarded;
    the current-view factory continues to reject it.  Invalid input is not a
    dedupe candidate.
    """
    try:
        if (
            type(observation) is not ConversationObservation
            or not isinstance(observation.status, str) or not observation.status.strip()
            or len(observation.status) > 80
            or not isinstance(observation.messages_json, str)
            or len(observation.messages_json) > _MAX_JSON
        ):
            return None
        rows = json.loads(observation.messages_json)
        if not isinstance(rows, list) or not rows or len(rows) > _MAX_MESSAGES:
            return None
        items: list[VisibleMessage] = []
        for row in rows:
            item = _visible_message(row, len(items), allow_uncertain=True)
            if item.text or item.media_refs or item.partial or item.order_uncertain:
                items.append(item)
        if not items:
            return None
        return {
            "schema": "whochat.observation_semantic.v1", "status": observation.status,
            "messages": [item.business_payload() for item in items],
        }
    except (ValueError, TypeError, AttributeError):
        return None


def _validate_evidence(
    source: ConversationObservation | CurrentMessageView, *, expected_account_scope_id: str,
    expected_contact_id: str, expected_conversation_id: str, expected_target_instance_id: str,
    expected_session_id: str, max_source_job_id: int, max_capture_generation: int, now: datetime,
    screen_proof: dict[str, Any] | None = None,
) -> None:
    expected = (expected_account_scope_id, expected_contact_id, expected_conversation_id,
                expected_target_instance_id, expected_session_id)
    if any(not isinstance(value, str) or not value.strip() for value in expected):
        raise ValueError("current_view_expected_binding_incomplete")
    if expected_account_scope_id == LEGACY_ACCOUNT_SCOPE_ID:
        raise ValueError("current_view_unscoped_source")
    if (
        source.account_scope_id != expected_account_scope_id
        or source.conversation_id != expected_conversation_id
        or source.target_instance_id != expected_target_instance_id
        or source.session_id != expected_session_id
        or (isinstance(source, CurrentMessageView) and source.contact_id != expected_contact_id)
    ):
        raise ValueError("current_view_source_binding_mismatch")
    status = source.status if isinstance(source, ConversationObservation) else source.observation_status
    evidence_id = source.id if isinstance(source, ConversationObservation) else source.observation_id
    independently_visible = False
    if isinstance(source, CurrentMessageView) and source.screen_evidence_json:
        try:
            screen_proof = json.loads(source.screen_evidence_json)
        except (ValueError, TypeError) as exc:
            raise ValueError("current_view_screen_proof_invalid") from exc
    if screen_proof is not None:
        _validate_screen_evidence(screen_proof, source, expected_contact_id, now)
        independently_visible = (status == "quarantined" or (
            screen_proof.get("source_kind") == WINDOW_REACTIVATION_SOURCE
            and status == "window_only" and source.relation == "window_only"))
    if (
        not isinstance(evidence_id, str) or not evidence_id.strip() or len(evidence_id) > 256
        or not isinstance(status, str) or (status not in _TRUSTED_STATUSES and not independently_visible)
        or not isinstance(source.relation, str) or (source.relation not in _TRUSTED_RELATIONS and not independently_visible)
    ):
        raise ValueError("current_view_untrusted_source")
    if (
        type(max_source_job_id) is not int or max_source_job_id < 1
        or type(source.source_job_id) is not int or not 1 <= source.source_job_id <= max_source_job_id
        or type(max_capture_generation) is not int or max_capture_generation < 0
        or type(source.capture_generation) is not int
        or not 0 <= source.capture_generation <= max_capture_generation
    ):
        raise ValueError("current_view_evidence_generation_mismatch")
    current = _aware_datetime(now)
    observed = _parse_time(source.observed_at)
    expires = _parse_time(source.expires_at)
    if expires <= current or expires <= observed or observed > current + timedelta(minutes=5):
        raise ValueError("current_view_expired_or_future_source")


def current_screen_evidence(observation: ConversationObservation) -> dict[str, Any] | None:
    """Read proof metadata, never infer a grant from a legacy quarantine row."""
    try:
        rows = json.loads(observation.messages_json)
        if isinstance(rows, dict) and rows.get("schema") == WINDOW_OBSERVATION_SCHEMA:
            proof = rows.get("screen_evidence")
            return proof if isinstance(proof,dict) else {"valid": False, "rejection_code": "malformed_window_marker"}
        if not (isinstance(rows, list) and rows and isinstance(rows[0], dict) and _SCREEN_KEY in rows[0]):
            return None
        proof = rows[0][_SCREEN_KEY]
        return proof if isinstance(proof, dict) else {"valid": False, "rejection_code": "malformed_marker"}
    except (ValueError, TypeError, AttributeError):
        return None


def screen_message_payload(parsed: Any) -> dict[str, Any]:
    """One shared producer shape; capture completeness precedes all filtering."""
    message_time, time_source = None, "unknown"
    supplied, supplied_source = getattr(parsed, "message_time", None), getattr(parsed, "time_source", "unknown")
    if isinstance(supplied, str) and supplied_source in _RELIABLE_TIME_SOURCES:
        try:
            _parse_time(supplied)  # Validate awareness without changing the producer's timezone.
            message_time = datetime.fromisoformat(supplied.replace("Z", "+00:00")).isoformat()
            time_source = supplied_source
        except (ValueError, TypeError, AttributeError):
            pass
    return {"speaker": parsed.speaker.value, "sender_name": parsed.sender_name or "", "text": parsed.text,
            "confidence": parsed.confidence, "partial": parsed.partial, "rect": parsed.rect.as_tuple(),
            "content_type": getattr(parsed, "content_type", "text"), "media_refs": getattr(parsed, "media_refs", []),
            "message_time": message_time, "time_source": time_source, "time_text": parsed.time_text or "",
            "order_uncertain": getattr(parsed, "order_uncertain", False)}


def screen_rows_digest(rows: list[dict[str, Any]]) -> str:
    return _revision([{key: value for key, value in row.items() if key != _SCREEN_KEY} for row in rows])


def screen_business_rows_digest(rows: list[dict[str, Any]]) -> str:
    return _revision([_visible_message(row, index).business_payload() for index, row in enumerate(rows)])


def screen_top_partial_prefix(messages: list[Any], layout: Any) -> int | None:
    """Only an ordered prefix touching the top, never a clipped middle/tail."""
    try:
        band = layout.message_rect
        margin = max(8, round(band.height * .025))
        prefix = 0
        for item in messages:
            if item.partial is not True:
                break
            if (item.rect.top > band.top + margin or item.rect.bottom >= band.bottom - margin
                    or item.rect.bottom <= band.top or item.rect.left < band.left or item.rect.right > band.right):
                return None
            prefix += 1
        if prefix >= len(messages) or any(item.partial is not False for item in messages[prefix:]):
            return None
        return prefix
    except (TypeError, AttributeError, ValueError):
        return None


def bind_current_screen_evidence(
    result: Any, *, contact: Any, conversation: Conversation, session_id: str,
    visible_messages: list[Any], current_state: Any, identity_stable: bool, now: datetime,
) -> dict[str, Any] | None:
    """Bind verified full-capture evidence after the live ingestion identity gate.

    No current-state provider, title-only input, filtered row, unclear boundary,
    or stale lifecycle can issue this capability. It never proves newness.
    """
    native = hasattr(result, "screen_proof")

    def invalid(code):
        return {"schema": SCREEN_EVIDENCE_SCHEMA, "valid": False, "rejection_code": code,
                "source_job_id": getattr(result, "job_id", 0)} if native else None

    try:
        proof = getattr(result, "screen_proof", None)
        if not isinstance(proof, dict) or proof.get("schema") != SCREEN_CAPTURE_SCHEMA:
            return invalid("capture_proof_missing")
        if proof.get("valid") is not True:
            return invalid("capture_incomplete")
        if identity_stable is not True:
            return invalid("identity_unstable")
        if (current_state is None or contact.account_scope_id != result.account_scope_id
                or conversation.account_scope_id != contact.account_scope_id or conversation.contact_id != contact.id
                or conversation.conversation_type != ConversationType.DM or contact.conversation_type != ConversationType.DM
                or getattr(contact.status, "value", str(contact.status)) in {"ignored", "merged"} or contact.merged_into
                or result.target_app != "wechat" or not session_id):
            return invalid("owner_or_live_binding_invalid")
        actual = {
            "account_scope_id": result.account_scope_id, "target_instance_id": result.target_instance_id,
            "source_job_id": result.job_id, "snapshot_hash": result.snapshot_hash,
            "hwnd": current_state.window.hwnd, "process_id": current_state.window.process_id,
            "window_generation": current_state.window_generation, "capture_generation": current_state.capture_generation,
            "account_identity_epoch": current_state.account_identity_epoch,
            "account_identity_runtime_key": current_state.account_identity_runtime_key,
        }
        if any(proof.get(key) != value for key, value in actual.items()):
            return invalid("capture_owner_binding_invalid")
        if any(getattr(result, key, None) != actual[key] for key in (
            "hwnd", "process_id", "window_generation", "capture_generation", "account_identity_epoch", "account_identity_runtime_key")):
            return invalid("capture_lifecycle_mismatch")
        selection_id = getattr(result.layout, 'manual_selection_id', '')
        if bool(selection_id) != (proof.get('capture_method') == 'manual_selection'):
            return invalid('manual_selection_source_mismatch')
        if selection_id and (proof.get('manual_selection_id') != selection_id or any(
                proof.get(key) != list(getattr(result.layout, name).as_tuple()) for key, name in (
                    ('selected_window_rect', 'window_rect'), ('selected_title_rect', 'title_rect'),
                    ('selected_message_rect', 'message_rect')))):
            return invalid('manual_selection_geometry_mismatch')
        if (current_state.page.page_type.value != "chat_dm" or result.page.page_type.value != "chat_dm"
                or result.page.confidence < .65 or result.ocr_result.warning):
            return invalid("page_or_ocr_invalid")
        if proof.get("source_kind") == WINDOW_REACTIVATION_SOURCE:
            explicit_activation = (proof.get('operator_selected_activation') is True
                and (proof.get('window_structure') or {}).get('schema') == 'whochat.manual_activation_structure.v1')
            if explicit_activation and proof.get('unread_current_rows_sha256') != screen_rows_digest(
                    [screen_message_payload(m) for m in result.messages]):
                return invalid('activation_capture_rows_changed')
            if (result.messages or visible_messages) and not explicit_activation:
                return invalid("window_source_contains_messages")
            bound = {**proof, "schema": SCREEN_EVIDENCE_SCHEMA, "contact_id": contact.id,
                     "conversation_id": conversation.id, "session_id": session_id, "identity_stable": True}
            _validate_capture_fields(bound, now)
            return bound
        # Timeline removes partial/empty rows too. Only the explicitly proved
        # continuous top prefix may disappear; no middle/tail omission can.
        rows = [screen_message_payload(item) for item in visible_messages if item.text.strip() and not item.partial]
        raw = [screen_message_payload(item) for item in result.messages]
        prefix = proof.get("omitted_top_prefix_count")
        if (type(prefix) is not int or prefix < 0 or screen_top_partial_prefix(result.messages, result.layout) != prefix
                or not rows or proof.get("raw_count") != len(raw)
                or proof.get("kept_count") != len(rows)
                or proof.get("raw_rows_sha256") != screen_rows_digest(raw)
                or proof.get("omitted_prefix_sha256") != screen_rows_digest(raw[:prefix])
                or screen_rows_digest(raw[prefix:]) != screen_rows_digest(rows)
                or proof.get("rows_sha256") != screen_rows_digest(rows)
                or proof.get("business_rows_sha256") != screen_business_rows_digest(rows)):
            return invalid("filtered_middle_or_tail")
        bound = {**proof, "schema": SCREEN_EVIDENCE_SCHEMA, "contact_id": contact.id,
                 "conversation_id": conversation.id, "session_id": session_id, "identity_stable": True}
        _validate_capture_fields(bound, now)
        return bound
    except (ValueError, TypeError, AttributeError, KeyError):
        return invalid("capture_proof_invalid")


def _validate_capture_fields(proof: dict[str, Any], now: datetime) -> None:
    if not isinstance(proof, dict) or proof.get("schema") not in {SCREEN_CAPTURE_SCHEMA, SCREEN_EVIDENCE_SCHEMA}:
        raise ValueError("current_view_screen_proof_invalid")
    if proof.get("valid") is not True:
        raise ValueError("current_view_screen_proof_incomplete")
    if proof.get('capture_method') == 'manual_selection':
        selection_id = proof.get('manual_selection_id')
        if (not isinstance(selection_id, str) or len(selection_id) != 32
                or any(c not in '0123456789abcdef' for c in selection_id)
                or type(proof.get('manual_request_id')) is not int or proof['manual_request_id'] <= 0):
            raise ValueError('manual_selection_request_invalid')
        rectangles = [proof.get(key) for key in ('selected_window_rect', 'selected_title_rect', 'selected_message_rect')]
        if any(not isinstance(r, (list, tuple)) or len(r) != 4 or any(type(v) is not int for v in r)
               or r[0] >= r[2] or r[1] >= r[3] for r in rectangles):
            raise ValueError('manual_selection_geometry_invalid')
        outer, title, body = rectangles
        if (any(not (outer[0] <= r[0] < r[2] <= outer[2] and outer[1] <= r[1] < r[3] <= outer[3])
                for r in (title, body)) or title[3] != body[1]
                or title[0] != body[0] or title[2] != body[2]):
            raise ValueError('manual_selection_geometry_invalid')
    window_only = proof.get("source_kind") == WINDOW_REACTIVATION_SOURCE
    flags = ("capture_verified", "title_verified", "account_verified", "input_boundary_verified")
    if any(proof.get(key) is not True for key in flags + (() if window_only else ("parser_complete", "tail_complete"))):
        raise ValueError("current_view_screen_proof_incomplete")
    if (proof.get("page_type") != "chat_dm" or isinstance(proof.get("page_confidence"), bool)
            or not isinstance(proof.get("page_confidence"), (int, float)) or not .65 <= proof["page_confidence"] <= 1):
        raise ValueError("current_view_screen_proof_page_invalid")
    raw, kept, prefix = proof.get("raw_count"), proof.get("kept_count"), proof.get("omitted_top_prefix_count")
    if window_only:
        _validate_window_capture_fields(proof)
    elif (type(raw) is not int or not 1 <= raw <= _MAX_MESSAGES or type(kept) is not int or not 1 <= kept <= raw
            or type(prefix) is not int or prefix < 0 or raw - kept != prefix
            or proof.get("omitted_count") != prefix or proof.get("partial_count") != prefix
            or any(type(proof.get(key)) is not int or proof[key] != 0 for key in (
                "kept_partial_count", "order_uncertain_count", "unaccounted_box_count"))):
        raise ValueError("current_view_screen_proof_omitted_content")
    for key in ("source_job_id", "hwnd", "process_id"):
        if type(proof.get(key)) is not int or proof[key] <= 0:
            raise ValueError("current_view_screen_proof_binding_invalid")
    for key in ("window_generation", "capture_generation", "account_identity_epoch"):
        if type(proof.get(key)) is not int or proof[key] < 0:
            raise ValueError("current_view_screen_proof_binding_invalid")
    for key in ("snapshot_hash", "account_scope_id", "target_instance_id", "account_identity_runtime_key", "title_exact_hash"):
        if not isinstance(proof.get(key), str) or not proof[key].strip() or len(proof[key]) > 512:
            raise ValueError("current_view_screen_proof_binding_invalid")
    for key in ("rows_sha256", "business_rows_sha256", "raw_rows_sha256", "omitted_prefix_sha256"):
        if not isinstance(proof.get(key), str) or len(proof[key]) != 64 or any(c not in "0123456789abcdef" for c in proof[key]):
            raise ValueError("current_view_screen_proof_rows_invalid")
    minimum = proof.get("min_confidence")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not .65 <= minimum <= 1:
        raise ValueError("current_view_screen_proof_low_confidence")
    observed, expires, current = _parse_time(proof.get("observed_at")), _parse_time(proof.get("expires_at")), _aware_datetime(now)
    if (observed > current + timedelta(seconds=5) or expires <= current or expires <= observed
            or expires - observed > timedelta(seconds=SCREEN_TTL_SECONDS)):
        raise ValueError("current_view_screen_proof_expired")


def _validate_window_capture_fields(proof: dict[str, Any]) -> None:
    """An explicit window-only capability cannot masquerade as parsed messages."""
    if (proof.get("parser_complete") is not False or proof.get("tail_complete") is not False
            or proof.get("purpose") != "reactivation" or type(proof.get("manual_request_id")) is not int
            or proof["manual_request_id"] <= 0 or proof.get("window_capture_complete") is not True):
        raise ValueError("current_window_capture_purpose_invalid")
    for key in ("raw_count", "kept_count", "omitted_count", "omitted_top_prefix_count", "partial_count",
                "kept_partial_count", "order_uncertain_count", "unaccounted_box_count", "message_box_count", "parsed_box_count"):
        if type(proof.get(key)) is not int or proof[key] != 0:
            raise ValueError("current_window_has_message_rows")
    if any(proof.get(key)!=screen_rows_digest([]) for key in (
            "rows_sha256","business_rows_sha256","raw_rows_sha256","omitted_prefix_sha256")):
        raise ValueError("current_window_has_message_rows")
    structure = proof.get("window_structure")
    if isinstance(structure,dict) and structure.get('schema') == 'whochat.manual_activation_structure.v1':
        from whochat.vision.manual_activation_structure import validate_manual_activation_structure
        validate_manual_activation_structure(structure,proof)
        _validate_window_avatar_anchor(proof)
        return
    if (not isinstance(structure,dict) or structure.get("schema") != "whochat.textless_chat_structure.v1"
            or type(structure.get("pair_count")) is not int or not 2 <= structure["pair_count"] <= 8
            or not isinstance(structure.get("pairs"),list) or len(structure["pairs"]) != structure["pair_count"]
            or structure.get("tail_card_complete") is not True):
        raise ValueError("current_window_positive_structure_missing")
    size=structure.get("message_size")
    if (set(structure) != {"schema","pair_count","pairs","message_size","tail_card_complete","structure_sha256","body_sha256"}
            or not isinstance(size,list) or len(size)!=2 or any(type(v) is not int or v<240 for v in size)):
        raise ValueError("current_window_structure_geometry_invalid")
    width,height=size
    previous_bottom=-8
    for pair in structure["pairs"]:
        if not isinstance(pair,dict) or set(pair)!={"avatar","card"}:
            raise ValueError("current_window_structure_geometry_invalid")
        avatar,card=pair["avatar"],pair["card"]
        if any(not isinstance(rect,list) or len(rect)!=4 or any(type(v) is not int for v in rect) for rect in (avatar,card)):
            raise ValueError("current_window_structure_geometry_invalid")
        ax,ay,aw,ah=avatar
        x,y,w,h=card
        if (not 0<=ax<=width*.15 or not 18<=aw<=52 or not 18<=ah<=52 or not .70<=aw/ah<=1.43
                or not 0<=ay<=height-ah or ax+aw>width or not width*.10<=x<=width*.30
                or not 70<=w<=width*.68 or not 90<=h<=height*.92 or not 0<=y<=height-h
                or x+w>width or abs(ay-y)>12 or not 3<=x-(ax+aw)<=36 or previous_bottom+8>y):
            raise ValueError("current_window_structure_geometry_invalid")
        previous_bottom=y+h
    if previous_bottom>height-5:
        raise ValueError("current_window_structure_tail_incomplete")
    for key in ("body_sha256", "structure_sha256"):
        value=structure.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("current_window_structure_digest_invalid")
    structure_payload={key:value for key,value in structure.items() if key not in {"body_sha256","structure_sha256"}}
    if _revision(structure_payload) != structure["structure_sha256"] or proof.get("window_body_sha256") != structure["body_sha256"]:
        raise ValueError("current_window_structure_changed")
    _validate_window_avatar_anchor(proof)


def _validate_window_avatar_anchor(proof):
    fingerprint=proof.get("account_avatar_fingerprint")
    geometry=proof.get("account_avatar_geometry")
    if (proof.get("account_avatar_reliable") is not True or not isinstance(fingerprint,str)
            or len(fingerprint)!=67 or not fingerprint.startswith("v2:")
            or any(c not in "0123456789abcdef" for c in fingerprint[3:])
            or not isinstance(geometry,(list,tuple)) or len(geometry)!=4
            or any(type(value) is not int for value in geometry)):
        raise ValueError("current_window_avatar_anchor_invalid")


def _validate_screen_evidence(proof: dict[str, Any], source: Any, contact_id: str, now: datetime, *, rows=None) -> None:
    _validate_capture_fields(proof, now)
    if proof.get("schema") != SCREEN_EVIDENCE_SCHEMA or proof.get("identity_stable") is not True:
        raise ValueError("current_view_screen_proof_identity_invalid")
    reason = source.reason if isinstance(source, ConversationObservation) else source.source_reason
    status = source.status if isinstance(source, ConversationObservation) else source.observation_status
    repeated = (isinstance(proof.get("history_parent_observation_id"), str)
        and bool(proof["history_parent_observation_id"])
        and reason == "persistent_tail_fallback_repeated_quarantine:" + proof["history_parent_observation_id"]
        and proof.get("history_root_reason") in _OVERLAP_ONLY_REASONS)
    window_only = proof.get("source_kind") == WINDOW_REACTIVATION_SOURCE
    allowed_reason = ((status == "window_only" and source.relation == "window_only" and reason == WINDOW_REACTIVATION_SOURCE)
        if window_only else ((reason in _OVERLAP_ONLY_REASONS or repeated)
        if status == "quarantined" else status in _TRUSTED_STATUSES and source.relation in _TRUSTED_RELATIONS)
    )
    if not allowed_reason or (status == "quarantined" and source.relation not in {"uncertain", "quarantine_repeated"}):
        raise ValueError("current_view_screen_proof_history_reason_invalid")
    expected = {"account_scope_id": source.account_scope_id, "contact_id": contact_id,
                "conversation_id": source.conversation_id, "target_instance_id": source.target_instance_id,
                "session_id": source.session_id, "source_job_id": source.source_job_id,
                "capture_generation": source.capture_generation, "history_reason": reason}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError("current_view_screen_proof_owner_mismatch")
    if isinstance(source, ConversationObservation):
        if proof.get("snapshot_hash") != source.snapshot_hash or _parse_time(proof["observed_at"]) != _parse_time(source.observed_at):
            raise ValueError("current_view_screen_proof_capture_mismatch")
    if rows is not None:
        if (len(rows) != proof["kept_count"] or screen_rows_digest(rows) != proof["rows_sha256"]
                or screen_business_rows_digest(rows) != proof["business_rows_sha256"]
                or any(row.get("partial") is not False or row.get("order_uncertain", False) is not False
                       or isinstance(row.get("confidence"), bool) or not isinstance(row.get("confidence"), (int, float))
                       or not .65 <= row["confidence"] <= 1 for row in rows)):
            raise ValueError("current_view_screen_proof_rows_mismatch")


def _visible_message(row: Any, ordinal: int, *, allow_uncertain: bool = False) -> VisibleMessage:
    if not isinstance(row, dict):
        raise ValueError("current_view_message_shape_invalid")
    partial = row.get("partial", False)
    order_uncertain = row.get("order_uncertain", False)
    if type(partial) is not bool or type(order_uncertain) is not bool:
        raise ValueError("current_view_uncertain_flag_invalid")
    if not allow_uncertain and (partial or order_uncertain):
        raise ValueError("current_view_uncertain_message")
    if str(row.get("source") or "").casefold().startswith(("raw", "uncommitted")):
        raise ValueError("current_view_raw_message")
    try:
        speaker = Speaker(row.get("speaker"))
    except (TypeError, ValueError) as exc:
        raise ValueError("current_view_speaker_invalid") from exc
    if speaker == Speaker.MEMBER:
        raise ValueError("current_view_group_member_in_dm")
    confidence = row.get("confidence", 1.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("current_view_confidence_invalid")
    text = _text(row.get("text", ""), _MAX_TEXT)
    sender = _text(row.get("sender_name", ""), 200)
    content_type = _text(row.get("content_type", "text"), 80) or "text"
    time_source = str(row.get("time_source") or "unknown")
    message_time = None
    if row.get("message_time") and time_source in _RELIABLE_TIME_SOURCES:
        message_time = _parse_time(row["message_time"]).isoformat()
    else:
        time_source = "unknown"
    return VisibleMessage(
        ordinal=ordinal, speaker=speaker, text=text, content_type=content_type,
        sender_name=sender, message_time=message_time, time_source=time_source,
        time_text=_text(row.get("time_text", ""), 120),
        media_refs=_attachments(row.get("media_refs", row.get("attachments", []))),
        evidence_status=_text(row.get("evidence_status", ""), 120),
        partial=partial, order_uncertain=order_uncertain,
    )


def _message_is_valid(item: Any, index: int) -> bool:
    if type(item) is not VisibleMessage or item.ordinal != index:
        return False
    # Frozen classes prevent ordinary mutation, but their public constructors
    # must not turn validation into a bypass for malformed or partial items.
    return item == _visible_message(item.business_payload(), index) and bool(item.text or item.media_refs)


def _attachments(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 16:
        raise ValueError("current_view_media_invalid")
    refs: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and len(item) <= 2000:
            refs.append(item.strip())
        elif isinstance(item, dict):
            selected = {key: _text(item[key], 2000) for key in (
                "reference", "ref", "attachment_id", "media_id", "content_hash", "content_type"
            ) if key in item}
            if not selected or not any(key in selected for key in ("reference", "ref", "attachment_id", "media_id", "content_hash")):
                raise ValueError("current_view_media_reference_missing")
            refs.append(json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        else:
            raise ValueError("current_view_media_reference_invalid")
    return tuple(refs)


def _text(value: Any, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("current_view_text_invalid")
    return " ".join(value.split())


def _aware_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("current_view_timezone_required")
    return value.astimezone(timezone.utc)


def _parse_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("current_view_time_invalid")
    return _aware_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _revision(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
