from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from whochat.core.models import (
    Contact,
    Conversation,
    ConversationObservation,
    ConversationSegment,
    Message,
    Speaker,
    utc_now_iso,
)
from whochat.ocr.models import ParsedOcrMessage
from whochat.services.pipeline import PipelineResult
from whochat.storage.repositories import ConversationRepository, MessageRepository, ObservationRepository, new_id
from whochat.core.current_conversation import (
    current_screen_evidence as current_screen_evidence_from_row, screen_message_payload, _validate_screen_evidence,
    _OVERLAP_ONLY_REASONS,
    SCREEN_EVIDENCE_SCHEMA,
    WINDOW_REACTIVATION_SOURCE, WINDOW_OBSERVATION_SCHEMA,
)


@dataclass(frozen=True)
class TimelineResult:
    conversation: Conversation
    segment: ConversationSegment | None
    relation: str
    inserted_messages: tuple[Message, ...] = field(default_factory=tuple)
    duplicate_messages: int = 0
    new_incoming_message_ids: tuple[str, ...] = field(default_factory=tuple)
    observation_id: str = ""
    observation_status: str = ""
    persistent_relation: str = ""
    historical_overlap_count: int = 0


@dataclass(frozen=True)
class RuntimeConversationAnchor:
    account_scope_id: str
    contact_id: str
    conversation_id: str
    identity_key: str
    page_type: str
    target_instance_id: str
    session_id: str
    observation_id: str
    visible_keys: tuple[tuple[str, str, str], ...]
    capture_generation: int
    observed_at: str
    updated_monotonic: float


class PersistentConversationTimeline:
    """Promotes only proven post-baseline deltas into the formal message table.

    The first stable screen of every session is a local observation baseline.
    Screens that cannot be ordered against that baseline remain in the
    observation quarantine and never feed messages, memory extraction, or AI.
    """

    def __init__(
        self,
        conversations: ConversationRepository,
        messages: MessageRepository,
        observations: ObservationRepository,
        *,
        runtime_anchor_max_age_seconds: float = 300.0,
    ) -> None:
        self.conversations = conversations
        self.messages = messages
        self.observations = observations
        self.runtime_anchor_max_age_seconds = max(30.0, runtime_anchor_max_age_seconds)
        self._runtime_anchors: dict[tuple[str, str, str, str, str, str], RuntimeConversationAnchor] = {}

    def observe(
        self,
        *,
        contact: Contact,
        identity_key: str,
        identity_confidence: float,
        session_id: str,
        result: PipelineResult,
        visible_messages: list[ParsedOcrMessage],
        baseline_reason: str = "first_stable_screen",
        delta_provenance: str = "",
        current_screen_evidence: dict | None = None,
    ) -> TimelineResult:
        del identity_confidence
        if current_screen_evidence is None and hasattr(result, "screen_proof"):
            current_screen_evidence = {"schema": SCREEN_EVIDENCE_SCHEMA, "valid": False,
                                       "rejection_code": "unbound_native_capture", "source_job_id": result.job_id}
        conversation = self.conversations.get_or_create(contact, identity_key)
        if (isinstance(current_screen_evidence,dict)
                and current_screen_evidence.get("source_kind") == WINDOW_REACTIVATION_SOURCE):
            explicit_activation = (current_screen_evidence.get('operator_selected_activation') is True
                and (current_screen_evidence.get('window_structure') or {}).get('schema') == 'whochat.manual_activation_structure.v1')
            if ((result.messages or visible_messages) and not explicit_activation
                    or current_screen_evidence.get("valid") is not True):
                raise ValueError("current_window_timeline_source_invalid")
            # A window baseline is not an occurrence, a tail anchor, or an
            # incoming delta. Keep its envelope apart from message-row JSON.
            now=utc_now_iso()
            proof={**current_screen_evidence,"history_reason":WINDOW_REACTIVATION_SOURCE}
            observation=ConversationObservation(
                id=new_id("observation"),account_scope_id=contact.account_scope_id,
                target_instance_id=result.target_instance_id,conversation_id=conversation.id,session_id=session_id,
                snapshot_hash=result.snapshot_hash,observed_at=result.created_at,status="window_only",relation="window_only",
                reason=WINDOW_REACTIVATION_SOURCE,keys_json="[]",source_job_id=result.job_id,
                capture_generation=result.capture_generation,expires_at=proof["expires_at"],created_at=now,updated_at=now,
                messages_json=json.dumps({"schema":WINDOW_OBSERVATION_SCHEMA,"messages":[],"screen_evidence":proof},
                                         ensure_ascii=False,separators=(",",":")))
            _validate_screen_evidence(proof,observation,contact.id,datetime.now(timezone.utc))
            persisted,_created=self.observations.add(observation)
            return TimelineResult(conversation,None,"window_only",(),0,(),persisted.id,"window_only")
        visible = [message for message in visible_messages if message.text.strip() and not message.partial]
        visible_keys = [_message_occurrence_key(item) for item in visible]
        persisted_tail = self.messages.list_timeline_tail(conversation.id, 64)
        persisted_tail_keys = [
            _message_occurrence_key(item)
            for item in persisted_tail
        ]
        runtime_key = _runtime_anchor_key(contact, conversation, identity_key, result)
        runtime_anchor, runtime_anchor_state = self._runtime_anchor_for(runtime_key)
        anchor = self.observations.latest_anchor(conversation.id, session_id)
        relation = "baseline"
        selected_indices: list[int] = []
        observation_status = "baseline"
        observation_reason = baseline_reason
        comparison_anchor_keys: list[tuple[str, str, str]] = []
        persistent_relation = "empty" if not persisted_tail_keys else "not_compared"
        historical_overlap_count = 0

        if anchor is not None:
            anchor_keys = _keys_from_json(anchor.keys_json)
            comparison_anchor_keys = anchor_keys
            relation, selected_indices = _placement_keys(anchor_keys, visible_keys)
            if relation == "uncertain" and _persistent_tail_placement(anchor_keys, visible_keys)[0] == "known":
                # Resizing may rewrap/crop older rows while the unique ordered
                # tail remains visible. Reuse the stricter suffix proof only
                # to recognize an existing screen, never to invent an append.
                relation, selected_indices = "known", []
            observation_status = "confirmed"
            observation_reason = f"ordered_against:{anchor.id}"
            if delta_provenance:
                observation_reason = f"{delta_provenance}:{observation_reason}"

            if relation == "uncertain":
                quarantine = self.observations.latest_quarantine(conversation.id, session_id)
                if quarantine is not None:
                    quarantine_keys = _keys_from_json(quarantine.keys_json)
                    q_relation, q_selected = _placement_keys(quarantine_keys, visible_keys)
                    if q_relation in {"known", "history"}:
                        self.observations.update_status(quarantine.id, "confirmed", "repeated_consistent_screen")
                        relation = "quarantine_confirmed"
                        selected_indices = []
                        observation_status = "confirmed"
                        observation_reason = f"confirmed_quarantine:{quarantine.id}"
                    elif q_relation == "append":
                        self.observations.update_status(quarantine.id, "promoted", "reliable_delta_observed")
                        relation = "append_after_quarantine"
                        selected_indices = q_selected
                        observation_status = "promoted"
                        observation_reason = f"delta_after_quarantine:{quarantine.id}"

                if relation == "uncertain":
                    observation_status = "quarantined"
                    prefix = f"{delta_provenance}:" if delta_provenance else ""
                    observation_reason = f"{prefix}no_reliable_overlap:{anchor.id}"
                    selected_indices = []

        elif runtime_anchor is not None and runtime_anchor.session_id != session_id:
            comparison_anchor_keys = list(runtime_anchor.visible_keys)
            relation, selected_indices = _placement_keys(comparison_anchor_keys, visible_keys)
            if relation == "uncertain" and _persistent_tail_placement(comparison_anchor_keys, visible_keys)[0] == "known":
                relation, selected_indices = "known", []
            observation_status = "confirmed"
            observation_reason = f"cross_session_runtime_anchor:{runtime_anchor.observation_id}"
            if relation == "append":
                delta_provenance = "cross_session_post_entry_new_message"
            elif relation == "uncertain":
                quarantine = self.observations.latest_quarantine(conversation.id, session_id)
                if quarantine is not None:
                    quarantine_keys = _keys_from_json(quarantine.keys_json)
                    q_relation, q_selected = _placement_keys(quarantine_keys, visible_keys)
                    if q_relation in {"known", "history"}:
                        relation = "quarantine_repeated"
                        selected_indices = []
                        observation_status = "quarantined"
                        observation_reason = f"repeated_cross_session_quarantine:{quarantine.id}"
                    elif q_relation == "append":
                        self.observations.update_status(quarantine.id, "promoted", "reliable_cross_session_delta_observed")
                        relation = "append_after_quarantine"
                        selected_indices = q_selected
                        observation_status = "promoted"
                        observation_reason = f"cross_session_delta_after_quarantine:{quarantine.id}"
                        delta_provenance = "cross_session_post_entry_new_message"
                if relation == "uncertain":
                    observation_status = "quarantined"
                    observation_reason = f"cross_session_no_reliable_overlap:{runtime_anchor.observation_id}"
                    selected_indices = []
        elif runtime_anchor_state in {"expired", "missing"} and persisted_tail_keys:
            # A runtime anchor is deliberately short-lived and is never
            # persisted as permanent truth. Once it expires (or after process
            # restart), the formal SQLite conversation tail is the only safe
            # fallback. Promote a suffix only when sender/text/order provide a
            # unique, non-ambiguous overlap with the current first screen.
            persistent_relation, selected_indices = _persistent_tail_placement(
                persisted_tail_keys,
                visible_keys,
            )
            comparison_anchor_keys = list(visible_keys) if persistent_relation == "known" else persisted_tail_keys
            source = "runtime_anchor_expired" if runtime_anchor_state == "expired" else "runtime_anchor_missing"
            if persistent_relation == "append":
                relation = "append"
                observation_status = "confirmed"
                observation_reason = f"persistent_tail_fallback:{source}"
                delta_provenance = "persistent_tail_post_entry_new_message"
            elif persistent_relation in {"known", "history"}:
                relation = persistent_relation
                observation_status = "confirmed"
                observation_reason = f"persistent_tail_fallback_{persistent_relation}:{source}"
            else:
                relation = "uncertain"
                selected_indices = []
                observation_status = "quarantined"
                observation_reason = f"persistent_tail_fallback_uncertain:{source}"
                quarantine = self.observations.latest_quarantine(conversation.id, session_id)
                if quarantine is not None:
                    quarantine_keys = _keys_from_json(quarantine.keys_json)
                    q_relation, q_selected = _placement_keys(quarantine_keys, visible_keys)
                    if q_relation in {"known", "history"}:
                        relation = "quarantine_repeated"
                        observation_reason = f"persistent_tail_fallback_repeated_quarantine:{quarantine.id}"
                    elif q_relation == "append":
                        self.observations.update_status(
                            quarantine.id,
                            "promoted",
                            "reliable_delta_after_persistent_tail_quarantine",
                        )
                        relation = "append_after_quarantine"
                        selected_indices = q_selected
                        observation_status = "promoted"
                        observation_reason = f"persistent_tail_delta_after_quarantine:{quarantine.id}"
                        delta_provenance = "persistent_tail_quarantine_delta"
        elif runtime_anchor_state == "expired":
            observation_reason = "runtime_anchor_expired"

        # A live session/runtime observation already supplies occurrence order,
        # including the distinction between two identical short replies. Its
        # durable-tail guard therefore uses the occurrence-aware matcher. The
        # stricter matcher above is reserved for the expired/missing-anchor
        # fallback, where no live ordering proof exists.
        if (
            selected_indices
            and persisted_tail_keys
            and delta_provenance != "persistent_tail_post_entry_new_message"
        ):
            session_selected = list(selected_indices)
            persistent_relation, persistent_selected = _placement_keys(
                persisted_tail_keys,
                visible_keys,
            )
            strict_tail_fallback = False
            if persistent_relation == "uncertain":
                # The live anchor can legitimately start before the durable DB
                # tail suffix that is still visible.  This happens when OCR
                # line wrapping changed older keys or the viewport shifted by
                # one row: the ordinary occurrence matcher proves the live
                # append, but cannot align a DB suffix that begins in the
                # middle of the current screen.  Reuse the stricter durable
                # matcher before quarantining.  It accepts only a unique,
                # reliable ordered suffix and continues to reject ambiguous
                # acknowledgements such as a lone repeated "收到".
                strict_relation, strict_selected = _persistent_tail_placement(
                    persisted_tail_keys,
                    visible_keys,
                )
                if strict_relation in {"append", "known", "history"}:
                    persistent_relation = strict_relation
                    persistent_selected = strict_selected
                    strict_tail_fallback = True
            if persistent_relation == "append":
                allowed = set(persistent_selected)
                selected_indices = [index for index in session_selected if index in allowed]
                historical_overlap_count = len(session_selected) - len(selected_indices)
                observation_reason += (
                    f":persistent_tail_append"
                    f"{'_strict_fallback' if strict_tail_fallback else ''}"
                    f":suppressed={historical_overlap_count}"
                )
            elif persistent_relation in {"known", "history"}:
                selected_indices = []
                historical_overlap_count = len(session_selected)
                observation_reason += (
                    f":persistent_tail_{persistent_relation}:suppressed={historical_overlap_count}"
                )
            elif relation == "append_after_quarantine" and observation_status == "promoted":
                # The durable tail may have no overlap because the user scrolled
                # to a wholly different region.  A quarantined frame that is
                # then observed again with a reliable suffix proves only that
                # suffix is new; preserve it without promoting the quarantined
                # history itself.
                persistent_relation = "quarantine_delta"
                observation_reason += ":persistent_tail_independent_quarantine_delta"
            else:
                selected_indices = []
                historical_overlap_count = len(session_selected)
                relation = "uncertain"
                observation_status = "quarantined"
                observation_reason = (
                    f"persistent_tail_no_reliable_overlap:suppressed={historical_overlap_count}"
                )
        elif selected_indices:
            persistent_relation = "empty"

        observation = _observation_from_screen(
            contact=contact,
            conversation=conversation,
            session_id=session_id,
            result=result,
            visible=visible,
            visible_keys=visible_keys,
            status=observation_status,
            relation=relation,
            reason=observation_reason,
            screen_evidence=current_screen_evidence,
            history_parent=(_history_parent_with_verified_root(self.observations, conversation.id, session_id)
                if observation_reason.startswith("persistent_tail_fallback_repeated_quarantine:") else None),
        )
        persisted_observation, observation_created = self.observations.add(observation)
        if not observation_created:
            return TimelineResult(
                conversation, None, "duplicate_snapshot", (), len(visible), (),
                persisted_observation.id, persisted_observation.status,
            )

        remember_runtime_anchor = (
            observation_status in {"baseline", "confirmed", "promoted"}
            and (
                relation in {"baseline", "append", "append_after_quarantine", "quarantine_confirmed"}
                or (relation == "known" and visible_keys == comparison_anchor_keys)
            )
        )
        if not selected_indices:
            if remember_runtime_anchor:
                self._remember_runtime_anchor(
                    runtime_key, contact, conversation, identity_key, session_id,
                    persisted_observation, visible_keys, result,
                )
            duplicates = len(visible) if relation in {"known", "history", "duplicate_snapshot"} else 0
            return TimelineResult(
                conversation, None, relation, (), duplicates, (),
                persisted_observation.id, persisted_observation.status,
                persistent_relation, historical_overlap_count,
            )

        segment, segment_created = self.conversations.add_segment(
            conversation_id=conversation.id,
            session_id=session_id,
            snapshot_hash=result.snapshot_hash,
            observed_at=result.created_at,
            page_type=result.page.page_type.value,
            page_confidence=result.page.confidence,
            ocr_engine=result.ocr_result.engine,
            reliable=True,
            order_uncertain=False,
            source_job_id=result.job_id,
            message_count=len(selected_indices),
            account_scope_id=contact.account_scope_id,
            target_instance_id=result.target_instance_id,
        )
        if not segment_created:
            return TimelineResult(
                conversation, segment, "duplicate_segment", (), len(selected_indices) + historical_overlap_count, (),
                persisted_observation.id, persisted_observation.status,
                persistent_relation, historical_overlap_count,
            )

        _minimum, maximum = self.messages.sequence_bounds(conversation.id)
        first_sequence = maximum + 1
        inserted: list[Message] = []
        verified_provenance = delta_provenance or "confirmed_new_message"
        verification_reason = (
            "persistent_tail_overlap"
            if persisted_tail_keys and persistent_relation == "append"
            else "quarantine_reliable_delta"
            if persistent_relation == "quarantine_delta"
            else "timeline_delta"
        )
        for offset, visible_index in enumerate(selected_indices):
            parsed = visible[visible_index]
            message = _message_from_observation(
                contact=contact,
                conversation=conversation,
                session_id=session_id,
                segment=segment,
                parsed=parsed,
                visible_index=visible_index,
                sequence_no=first_sequence + offset,
                result=result,
                provenance=verified_provenance,
                verification_reason=verification_reason,
            )
            if self.messages.add_message(message):
                inserted.append(message)

        new_incoming = tuple(message.id for message in inserted if message.speaker == Speaker.OTHER)
        if remember_runtime_anchor:
            self._remember_runtime_anchor(
                runtime_key, contact, conversation, identity_key, session_id,
                persisted_observation, visible_keys, result,
            )
        return TimelineResult(
            conversation, segment, relation, tuple(inserted),
            historical_overlap_count + max(0, len(selected_indices) - len(inserted)),
            new_incoming, persisted_observation.id, persisted_observation.status,
            persistent_relation, historical_overlap_count,
        )

    def _runtime_anchor_for(
        self,
        key: tuple[str, str, str, str, str, str],
    ) -> tuple[RuntimeConversationAnchor | None, str]:
        anchor = self._runtime_anchors.get(key)
        if anchor is None:
            return None, "missing"
        if time.monotonic() - anchor.updated_monotonic > self.runtime_anchor_max_age_seconds:
            self._runtime_anchors.pop(key, None)
            return None, "expired"
        return anchor, "eligible"

    def _remember_runtime_anchor(
        self,
        key: tuple[str, str, str, str, str, str],
        contact: Contact,
        conversation: Conversation,
        identity_key: str,
        session_id: str,
        observation: ConversationObservation,
        visible_keys: list[tuple[str, str, str]],
        result: PipelineResult,
    ) -> None:
        self._runtime_anchors[key] = RuntimeConversationAnchor(
            account_scope_id=contact.account_scope_id,
            contact_id=contact.id,
            conversation_id=conversation.id,
            identity_key=identity_key,
            page_type=result.page.page_type.value,
            target_instance_id=result.target_instance_id,
            session_id=session_id,
            observation_id=observation.id,
            visible_keys=tuple(visible_keys),
            capture_generation=result.capture_generation,
            observed_at=result.created_at,
            updated_monotonic=time.monotonic(),
        )


def _placement_keys(
    anchor: list[tuple[str, str, str]],
    visible: list[tuple[str, str, str]],
) -> tuple[str, list[int]]:
    if not visible:
        return "known", []
    if not anchor:
        return "uncertain", []
    if _contains_sequence(anchor, visible):
        return "known", []
    anchor_in_visible = _sequence_index(visible, anchor)
    if anchor_in_visible >= 0:
        end = anchor_in_visible + len(anchor)
        return ("append", list(range(end, len(visible)))) if end < len(visible) else ("history", [])
    maximum_overlap = min(len(anchor), len(visible))
    for size in range(maximum_overlap, 0, -1):
        if anchor[-size:] == visible[:size]:
            return "append", list(range(size, len(visible)))
    for size in range(maximum_overlap, 0, -1):
        if visible[-size:] == anchor[:size]:
            return "history", []
    return "uncertain", []


def _persistent_tail_placement(
    persisted_tail: list[tuple[str, str, str]],
    visible: list[tuple[str, str, str]],
) -> tuple[str, list[int]]:
    """Align a first screen with the durable formal-message tail.

    Unlike the in-session matcher, a durable fallback must reject ambiguous
    one-token overlaps such as a repeated short acknowledgement. It also
    distinguishes a historical slice inside the DB tail from the actual tail
    suffix so scrolling upward can never manufacture a new append.
    """
    if not visible:
        return "known", []
    if not persisted_tail:
        return "uncertain", []

    visible_positions = _sequence_positions(persisted_tail, visible)
    if visible_positions:
        if any(position + len(visible) == len(persisted_tail) for position in visible_positions):
            return "known", []
        return "history", []

    full_tail_positions = _sequence_positions(visible, persisted_tail)
    appended_positions = [
        position
        for position in full_tail_positions
        if position + len(persisted_tail) < len(visible)
    ]
    if len(appended_positions) == 1 and _reliable_persistent_overlap(
        persisted_tail,
        persisted_tail,
        visible,
    ):
        end = appended_positions[0] + len(persisted_tail)
        return "append", list(range(end, len(visible)))
    if appended_positions:
        return "uncertain", []

    maximum_overlap = min(len(persisted_tail), len(visible))
    for size in range(maximum_overlap, 0, -1):
        overlap = persisted_tail[-size:]
        positions = _sequence_positions(visible, overlap)
        if not positions:
            continue
        if len(positions) != 1 or not _reliable_persistent_overlap(overlap, persisted_tail, visible):
            return "uncertain", []
        end = positions[0] + size
        return (
            ("append", list(range(end, len(visible))))
            if end < len(visible)
            else ("known", [])
        )

    for size in range(maximum_overlap, 0, -1):
        if visible[-size:] == persisted_tail[:size]:
            return "history", []
    return "uncertain", []


def _reliable_persistent_overlap(
    overlap: list[tuple[str, str, str]],
    persisted_tail: list[tuple[str, str, str]],
    visible: list[tuple[str, str, str]],
) -> bool:
    if not overlap:
        return False
    distinct = len(set(overlap))
    if len(overlap) >= 2 and distinct >= 2:
        return True
    if len(overlap) > 1 and any(not _weak_repeated_key(key) for key in overlap):
        return True
    key = overlap[0]
    if _weak_repeated_key(key):
        return False
    return persisted_tail.count(key) == 1 and visible.count(key) == 1


def _weak_repeated_key(key: tuple[str, str, str]) -> bool:
    text = "".join(key[2].split()).casefold()
    return text in {
        "收到",
        "好的",
        "好",
        "可以",
        "行",
        "嗯",
        "嗯嗯",
        "ok",
        "谢谢",
        "知道了",
        "明白",
    }


def _sequence_positions(
    haystack: list[tuple[str, str, str]],
    needle: list[tuple[str, str, str]],
) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    return [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index:index + len(needle)] == needle
    ]


def _runtime_anchor_key(
    contact: Contact,
    conversation: Conversation,
    identity_key: str,
    result: PipelineResult,
) -> tuple[str, str, str, str, str, str]:
    return (
        contact.account_scope_id,
        contact.id,
        conversation.id,
        identity_key,
        result.page.page_type.value,
        result.target_instance_id,
    )


def _contains_sequence(haystack: list[tuple[str, str, str]], needle: list[tuple[str, str, str]]) -> bool:
    return _sequence_index(haystack, needle) >= 0


def _sequence_index(haystack: list[tuple[str, str, str]], needle: list[tuple[str, str, str]]) -> int:
    if len(needle) > len(haystack):
        return -1
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == needle:
            return index
    return -1


def _message_key(speaker: Speaker, sender_name: str, text: str) -> tuple[str, str, str]:
    return (speaker.value, " ".join(sender_name.strip().split()).casefold(), " ".join(text.strip().split()).casefold())


def _media_reference_hash(message) -> str:
    """Only the structural producer's one immutable image reference qualifies."""
    if getattr(message, "content_type", "text") != "image":
        return ""
    references = getattr(message, "media_refs", ())
    if not isinstance(references, (list, tuple)) or len(references) != 1:
        return ""
    reference = references[0]
    if (not isinstance(reference, dict) or set(reference) != {"content_hash", "content_type"}
            or reference["content_type"] != "image"):
        return ""
    digest = reference["content_hash"]
    return digest if isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) else ""


def _persisted_media_hash(message) -> str:
    """Read our own formal-occurrence marker, never a user's text or filename.

    No schema migration is needed: source remains provenance plus one narrowly
    formatted media hash. Arbitrary imported source strings and legacy image
    placeholders cannot impersonate this producer's persisted occurrence.
    """
    import re
    if (getattr(message, "content_type", "text") != "image"
            or not all(getattr(message, field, "") for field in ("conversation_id", "session_id", "segment_id"))):
        return ""
    identity = re.fullmatch(r"message-id:([0-9a-f]{32})", getattr(message, "id", ""))
    if identity is None or getattr(message, "fingerprint", "") != "observation:" + identity[1]:
        return ""
    source = getattr(message, "source", "")
    marker = re.fullmatch(
        r"ocr_[a-z_]+:[A-Za-z0-9_.+-]+;verified_new=(?:persistent_tail_overlap|quarantine_reliable_delta|timeline_delta);media_sha256=([0-9a-f]{64})",
        source,
    )
    return marker[1] if marker else ""


def _message_occurrence_key(message) -> tuple[str, str, str]:
    key = _message_key(message.speaker, message.sender_name or "", message.text)
    if getattr(message, "content_type", "text") != "image":
        return key
    digest = _media_reference_hash(message) or _persisted_media_hash(message)
    # Legacy/invalid image references must never match a newly proved picture.
    identity = {"type": "image", "sha256": digest} if digest else {
        "type": "unresolved_image", "id": str(getattr(message, "id", ""))}
    return (*key[:2], json.dumps(identity, sort_keys=True, separators=(",", ":")))


def _keys_from_json(value: str) -> list[tuple[str, str, str]]:
    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    result: list[tuple[str, str, str]] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, list) and len(item) == 3:
            result.append((str(item[0]), str(item[1]), str(item[2])))
    return result


def _history_parent_with_verified_root(observations, conversation_id, session_id):
    """Recover only historical isolation reason, never a positive capability."""
    parent = observations.latest_quarantine(conversation_id, session_id)
    if parent is None:
        return None
    cursor, seen = parent, set()
    for _ in range(64):
        if cursor.id in seen:
            return None
        seen.add(cursor.id)
        if any(getattr(cursor, key) != getattr(parent, key) for key in (
                "account_scope_id", "conversation_id", "target_instance_id", "session_id")) or cursor.status != "quarantined":
            return None
        if cursor.reason in _OVERLAP_ONLY_REASONS:
            try:
                rows = json.loads(parent.messages_json)
                if not rows or not isinstance(rows[0], dict):
                    return None
                old = current_screen_evidence_from_row(parent) or {}
                rows[0]["_current_screen_evidence"] = {**old, "history_root_reason": cursor.reason}
                return replace(parent, messages_json=json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
            except (ValueError, TypeError, AttributeError):
                return None
        prefix = "persistent_tail_fallback_repeated_quarantine:"
        if not cursor.reason.startswith(prefix):
            return None
        parent_id = cursor.reason[len(prefix):]
        cursor = observations.get(parent_id)
        if cursor is None:
            return None
    return None


def _observation_from_screen(
    *,
    contact: Contact,
    conversation: Conversation,
    session_id: str,
    result: PipelineResult,
    visible: list[ParsedOcrMessage],
    visible_keys: list[tuple[str, str, str]],
    status: str,
    relation: str,
    reason: str,
    screen_evidence: dict | None = None,
    history_parent: ConversationObservation | None = None,
) -> ConversationObservation:
    now = utc_now_iso()
    expires = datetime.now(timezone.utc) + timedelta(hours=24 if status == "quarantined" else 24 * 30)
    payload = [_observation_message_payload(item) for item in visible]
    observation = ConversationObservation(
        id=new_id("observation"),
        account_scope_id=contact.account_scope_id,
        target_instance_id=result.target_instance_id,
        conversation_id=conversation.id,
        session_id=session_id,
        snapshot_hash=result.snapshot_hash,
        observed_at=result.created_at,
        status=status,
        relation=relation,
        reason=reason,
        messages_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        keys_json=json.dumps(visible_keys, ensure_ascii=False, separators=(",", ":")),
        source_job_id=result.job_id,
        capture_generation=result.capture_generation,
        expires_at=expires.isoformat(),
        created_at=now,
        updated_at=now,
    )
    # This metadata is independent of timeline trust/newness. Never upgrade
    # quarantine, select additional formal messages or issue incoming events.
    if isinstance(screen_evidence, dict) and payload:
        root_reason = reason if reason in _OVERLAP_ONLY_REASONS else ""
        parent_id = ""
        if reason.startswith("persistent_tail_fallback_repeated_quarantine:") and history_parent is not None:
            parent_id = history_parent.id
            parent_proof = current_screen_evidence_from_row(history_parent)
            root_reason = (history_parent.reason if history_parent.reason in _OVERLAP_ONLY_REASONS
                           else (parent_proof or {}).get("history_root_reason", ""))
        proof = {**screen_evidence, "history_reason": reason,
                 "history_root_reason": root_reason, "history_parent_observation_id": parent_id}
        enriched = [dict(row) for row in payload]
        enriched[0]["_current_screen_evidence"] = proof
        if proof.get("valid") is True:
            try:
                _validate_screen_evidence(proof, observation, contact.id, datetime.now(timezone.utc), rows=enriched)
            except (ValueError, TypeError, AttributeError, KeyError):
                enriched[0]["_current_screen_evidence"] = {"schema": SCREEN_EVIDENCE_SCHEMA, "valid": False,
                    "rejection_code": "persisted_screen_binding_invalid", "source_job_id": result.job_id}
        observation = replace(observation, messages_json=json.dumps(enriched, ensure_ascii=False, separators=(",", ":")))
    return observation


def _observation_message_payload(parsed: ParsedOcrMessage) -> dict:
    """Preserve display evidence without inventing a calendar or media read.

    The current OCR message contract supplies only a display time string.  It
    may lack the calendar date or timezone, so capture time is never promoted
    into sent time.  A future producer can carry a separately reliable,
    timezone-aware message timestamp without changing the formal timeline.
    """
    return screen_message_payload(parsed)


def _message_from_observation(
    *,
    contact: Contact,
    conversation: Conversation,
    session_id: str,
    segment: ConversationSegment,
    parsed: ParsedOcrMessage,
    visible_index: int,
    sequence_no: int,
    result: PipelineResult,
    provenance: str = "",
    verification_reason: str = "timeline_delta",
) -> Message:
    identity = "|".join(
        [segment.id, str(visible_index), parsed.speaker.value, parsed.sender_name or "", " ".join(parsed.text.split())]
    )
    media_hash = _media_reference_hash(parsed)
    if getattr(parsed, "content_type", "text") == "image" and not media_hash:
        raise ValueError("formal_image_requires_proved_media_reference")
    if media_hash:
        identity += "|image:" + media_hash
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    message_time = None
    time_source = "observed"
    if parsed.time_text:
        try:
            from whochat.services.transcript_stitcher import _resolve_message_time

            resolved, source = _resolve_message_time(parsed.time_text, result.created_at)
            if source == "ocr":
                message_time = resolved
                time_source = "ocr"
        except Exception:
            pass
    return Message(
        id=f"message-id:{digest}",
        contact_id=contact.id,
        speaker=parsed.speaker,
        text=parsed.text.strip(),
        content_type="image" if media_hash else "text",
        ocr_confidence=parsed.confidence,
        observed_at=result.created_at,
        message_time=message_time,
        time_source=time_source,
        partial=parsed.partial,
        fingerprint=f"observation:{digest}",
        source=(f"ocr_{provenance or 'confirmed_new_message'}:{result.ocr_result.engine};verified_new={verification_reason}"
                + (f";media_sha256={media_hash}" if media_hash else "")),
        sender_name=parsed.sender_name or "",
        conversation_id=conversation.id,
        session_id=session_id,
        segment_id=segment.id,
        sequence_no=sequence_no,
        segment_index=visible_index,
        bbox_json=json.dumps(parsed.rect.as_tuple(), separators=(",", ":")),
        order_uncertain=False,
        account_scope_id=contact.account_scope_id,
    )
