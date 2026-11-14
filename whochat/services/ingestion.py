from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from whochat.core.models import Contact, ContactStatus, ConversationType, Message, Speaker, utc_now_iso
from whochat.core.runtime import PageType
from whochat.ocr.models import OcrRegion, OcrResult, ParsedOcrMessage
from whochat.ocr.parser import normalize_ocr_regions
from whochat.services.memory_extraction import MemoryExtractionService
from whochat.services.conversation_timeline import PersistentConversationTimeline
from whochat.services.identity_lock import IdentityResolution, StableIdentityLock
from whochat.services.pipeline import PipelineResult
from whochat.core.current_conversation import bind_current_screen_evidence
from whochat.services.transcript_stitcher import StitchedMessage, TranscriptStitcher
from whochat.storage.repositories import (
    AIAnalysisEventRepository,
    ContactRepository,
    ConversationRepository,
    LogRepository,
    MessageRepository,
)


@dataclass(frozen=True)
class IngestionResult:
    accepted: bool
    reason: str
    contact: Contact | None
    inserted_messages: int
    duplicate_messages: int
    title_candidates: tuple[str, ...] = field(default_factory=tuple)
    # Produced by the long-term-memory stage of the same ingestion pass.
    memories_created: int = 0
    memory_reason: str = ""
    identity_stable: bool = False
    identity_key: str = ""
    conversation_id: str | None = None
    session_id: str | None = None
    segment_id: str | None = None
    new_incoming_messages: tuple[Message, ...] = field(default_factory=tuple)
    observation_id: str = ""
    observation_status: str = ""
    account_scope_id: str = ""
    target_instance_id: str = ""
    analysis_event_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def should_trigger_ai(self) -> bool:
        return self.accepted and self.identity_stable and bool(self.new_incoming_messages)


@dataclass(frozen=True)
class _PreBaselineFrame:
    result: PipelineResult
    visible_messages: tuple[ParsedOcrMessage, ...]
    buffered_at: float


class _PreBaselineObservationBuffer:
    """Keep bounded full OCR frames while the identity lock is pending."""

    def __init__(self, *, max_contexts: int = 8, max_frames: int = 8, max_age_seconds: float = 180.0) -> None:
        self.max_contexts = max(1, max_contexts)
        self.max_frames = max(2, max_frames)
        self.max_age_seconds = max(30.0, max_age_seconds)
        self._frames: dict[str, list[_PreBaselineFrame]] = {}
        self._last_touched: dict[str, float] = {}

    def add(self, key: str, result: PipelineResult, visible_messages: list[ParsedOcrMessage]) -> None:
        if not key:
            return
        now = time.monotonic()
        self._prune(now)
        frames = self._frames.setdefault(key, [])
        if any(frame.result.job_id == result.job_id or frame.result.snapshot_hash == result.snapshot_hash for frame in frames):
            self._last_touched[key] = now
            return
        frames.append(_PreBaselineFrame(result, tuple(visible_messages), now))
        frames.sort(key=lambda frame: (frame.result.job_id, frame.result.created_at))
        if len(frames) > self.max_frames:
            frames[:] = [frames[0], *frames[-(self.max_frames - 1):]]
        self._last_touched[key] = now
        self._trim_contexts()

    def take(self, key: str) -> tuple[_PreBaselineFrame, ...]:
        self._prune(time.monotonic())
        self._last_touched.pop(key, None)
        return tuple(self._frames.pop(key, []))

    def clear(self) -> None:
        self._frames.clear()
        self._last_touched.clear()

    def _prune(self, now: float) -> None:
        stale = [key for key, touched in self._last_touched.items() if now - touched > self.max_age_seconds]
        for key in stale:
            self._frames.pop(key, None)
            self._last_touched.pop(key, None)

    def _trim_contexts(self) -> None:
        while len(self._frames) > self.max_contexts:
            oldest = min(self._frames, key=lambda key: self._last_touched.get(key, 0.0))
            self._frames.pop(oldest, None)
            self._last_touched.pop(oldest, None)


class ChatIngestionService:
    def __init__(
        self,
        contacts: ContactRepository,
        messages: MessageRepository,
        logs: LogRepository | None = None,
        transcript_stitcher: TranscriptStitcher | None = None,
        memory_extractor: MemoryExtractionService | None = None,
        identity_lock: StableIdentityLock | None = None,
        conversations: ConversationRepository | None = None,
        timeline: PersistentConversationTimeline | None = None,
        analysis_events: AIAnalysisEventRepository | None = None,
        *,
        min_page_confidence: float = 0.65,
        min_title_confidence: float = 0.50,
        min_message_confidence: float = 0.42,
        current_state_provider=None,
    ) -> None:
        self.contacts = contacts
        self.messages = messages
        self.logs = logs
        self.min_page_confidence = min_page_confidence
        self.min_title_confidence = min_title_confidence
        self.min_message_confidence = min_message_confidence
        self.last_result: IngestionResult | None = None
        self.last_evidence = None
        self.current_state_provider = current_state_provider
        self._last_pipeline_evidence_job_id = 0
        self.transcript_stitcher = transcript_stitcher or TranscriptStitcher()
        self.memory_extractor = memory_extractor
        self.identity_lock = identity_lock
        self.conversations = conversations
        self.timeline = timeline
        self.analysis_events = analysis_events
        self._active_identity_key = ""
        self._active_conversation_id = ""
        self._active_session_id = ""
        self._active_scope_id = ""
        self._active_target_instance_id = ""
        self._active_window_generation = 0
        self._active_capture_generation = 0
        self._active_account_identity_epoch = 0
        self._last_log_signature = ""
        self.on_identity_unstable = None
        self._prebaseline = _PreBaselineObservationBuffer()
        self._identity_candidate_key = ""
        self._identity_candidate_started_monotonic = 0.0

    def ingest_pipeline_result(self, result: PipelineResult) -> IngestionResult:
        stale_reason = self._stale_evidence_reason(result)
        if stale_reason:
            return IngestionResult(False, stale_reason, None, 0, 0)
        self.last_evidence = result
        self._last_pipeline_evidence_job_id = max(self._last_pipeline_evidence_job_id, int(result.job_id))
        ingestion_started_at = utc_now_iso()
        boundary_error = self._account_boundary_error(result)
        if boundary_error:
            if self.identity_lock is not None:
                self.identity_lock.invalidate(boundary_error)
            self._prebaseline.clear()
            self._close_active_session()
            return self._finish(IngestionResult(False, boundary_error, None, 0, 0))
        if not result.page.can_generate_reply or result.page.confidence < self.min_page_confidence:
            if self.identity_lock is not None:
                self.identity_lock.invalidate(f"page_blocked:{result.page.page_type.value}")
            self._prebaseline.clear()
            self._close_active_session()
            return self._finish(IngestionResult(False, f"page_blocked:{result.page.page_type.value}", None, 0, 0))
        if result.ocr_result.warning:
            return self._finish(IngestionResult(False, f"ocr_warning:{result.ocr_result.warning}", None, 0, 0))
        title_candidates = extract_title_candidates(
            result.ocr_result,
            result,
            min_confidence=self.min_title_confidence,
        )
        raw_title = title_candidates[0] if title_candidates else None
        title = normalize_contact_title(
            raw_title,
            ConversationType.GROUP if result.page.page_type == PageType.CHAT_GROUP else ConversationType.DM,
        ) if raw_title else None
        if title is None:
            debug_lines = tuple(_title_candidate_debug_lines(result.ocr_result, result))
            return self._finish(
                IngestionResult(
                    False,
                    f"contact_title_unavailable:{_format_title_candidates(debug_lines)}",
                    None,
                    0,
                    0,
                    debug_lines,
                )
            )

        conversation_type = ConversationType.GROUP if result.page.page_type == PageType.CHAT_GROUP else ConversationType.DM
        identity_started_at = utc_now_iso()
        identity_started_monotonic = time.monotonic()
        identity = self._observe_identity(result, title, conversation_type)
        self._record_identity_timing(result, identity, identity_started_monotonic)
        visible_messages = [message for message in result.messages if message.confidence >= self.min_message_confidence]
        prebaseline_key = _prebaseline_context_key(identity, result)
        if not identity.stable or identity.contact is None:
            if identity.reason.startswith("identity_pending:") or identity.reason == "identity_waiting_page_type":
                self._prebaseline.add(prebaseline_key, result, visible_messages)
            callback = self.on_identity_unstable
            if callable(callback):
                callback(result.snapshot_hash)
            return self._finish(
                IngestionResult(
                    False,
                    identity.reason,
                    None,
                    0,
                    0,
                    tuple(_title_candidate_debug_lines(result.ocr_result, result)),
                    identity_stable=False,
                    identity_key=identity.identity_key,
                )
            )

        contact = identity.contact
        identity_stable_at = utc_now_iso()
        _add_raw_title_alias(self.contacts, contact, raw_title, title)
        if self.conversations is None or self.timeline is None:
            return self._finish(IngestionResult(False, "persistent_timeline_unavailable", contact, 0, 0))
        conversation = self.conversations.get_or_create(contact, identity.identity_key)
        session_id = self._ensure_session(conversation.id, identity, result)
        buffered_frames = list(self._prebaseline.take(prebaseline_key))
        replay_frames: list[tuple[PipelineResult, list[ParsedOcrMessage]]] = [
            (frame.result, list(frame.visible_messages)) for frame in buffered_frames
        ]
        if not any(
            buffered_result.job_id == result.job_id or buffered_result.snapshot_hash == result.snapshot_hash
            for buffered_result, _messages in replay_frames
        ):
            replay_frames.append((result, visible_messages))

        timeline_results = []
        for index, (frame_result, frame_messages) in enumerate(replay_frames):
            has_prebaseline_evidence = bool(buffered_frames)
            # A complete current screen is not proof that it follows the old
            # durable tail. Bind this separate capability only after the full
            # live identity/account/session gates, and before any row omission.
            screen_evidence = bind_current_screen_evidence(
                frame_result, contact=contact, conversation=conversation, session_id=session_id,
                visible_messages=frame_messages,
                current_state=self.current_state_provider() if callable(self.current_state_provider) else None,
                identity_stable=identity.stable, now=datetime.now(timezone.utc),
            )
            timeline_results.append(
                self.timeline.observe(
                    contact=contact,
                    identity_key=identity.identity_key,
                    identity_confidence=identity.confidence,
                    session_id=session_id,
                    result=frame_result,
                    visible_messages=frame_messages,
                    baseline_reason="prebaseline_entry_frame" if has_prebaseline_evidence and index == 0 else "first_stable_screen",
                    delta_provenance="post_entry_new_message" if has_prebaseline_evidence and index > 0 else "",
                    current_screen_evidence=screen_evidence,
                )
            )
        timeline_result = timeline_results[-1]
        inserted_messages = [
            message
            for replay_result in timeline_results
            for message in replay_result.inserted_messages
        ]
        incoming_ids = {
            message_id
            for replay_result in timeline_results
            for message_id in replay_result.new_incoming_message_ids
        }
        new_incoming = tuple(message for message in inserted_messages if message.id in incoming_ids)
        message_classified_at = utc_now_iso()
        message_committed_at = message_classified_at
        frame_by_segment = {
            timeline_item.segment.id: frame_result
            for timeline_item, (frame_result, _frame_messages) in zip(timeline_results, replay_frames)
            if timeline_item.segment is not None
        }
        analysis_event_ids = self._create_analysis_events(
            new_incoming,
            frame_by_segment,
            fallback_result=result,
            identity=identity,
            conversation_id=conversation.id,
            ingestion_started_at=ingestion_started_at,
            identity_started_at=identity_started_at,
            identity_stable_at=identity_stable_at,
            message_classified_at=message_classified_at,
            message_committed_at=message_committed_at,
        )
        memories_created, memory_reason = self._extract_memories(contact, inserted_messages)
        reason = f"messages_ingested:{timeline_result.relation}"
        if memories_created:
            reason += f":memories_created:{memories_created}"
        elif memory_reason and memory_reason != "no_new_messages":
            reason += f":memory_skipped:{memory_reason}"
        return self._finish(
            IngestionResult(
                True,
                reason,
                contact,
                len(inserted_messages),
                sum(item.duplicate_messages for item in timeline_results),
                tuple(_title_candidate_debug_lines(result.ocr_result, result)),
                memories_created,
                memory_reason,
                True,
                identity.identity_key,
                timeline_result.conversation.id,
                session_id,
                next((item.segment.id for item in reversed(timeline_results) if item.segment is not None), None),
                new_incoming,
                timeline_result.observation_id,
                timeline_result.observation_status,
                contact.account_scope_id,
                result.target_instance_id,
                analysis_event_ids,
            )
        )

    def _create_analysis_events(
        self,
        messages: tuple[Message, ...],
        frame_by_segment: dict[str, PipelineResult],
        *,
        fallback_result: PipelineResult,
        identity: IdentityResolution,
        conversation_id: str,
        ingestion_started_at: str,
        identity_started_at: str,
        identity_stable_at: str,
        message_classified_at: str,
        message_committed_at: str,
    ) -> tuple[str, ...]:
        if self.analysis_events is None:
            return ()
        created: list[str] = []
        for message in messages:
            # Timeline promotion and AI triggering are separate safety gates.
            # Even if a future ingestion bug manages to insert an OCR row, an
            # automatic event is allowed only for a row explicitly marked as a
            # verified timeline delta for this exact conversation.
            if not _eligible_for_auto_analysis_event(message, conversation_id):
                continue
            frame = frame_by_segment.get(message.segment_id or "", fallback_result)
            performance_trace = dict(frame.performance_trace)
            performance_trace.update(
                {
                    "ingestion_started_at": ingestion_started_at,
                    "identity_started_at": identity_started_at,
                    "identity_stable_at": identity_stable_at,
                    "message_classified_at": message_classified_at,
                    "message_committed_at": message_committed_at,
                }
            )
            snapshot = {
                "source_message_id": message.id,
                "account_scope_id": message.account_scope_id,
                "target_instance_id": frame.target_instance_id,
                "contact_id": message.contact_id,
                "conversation_id": conversation_id,
                "session_id": message.session_id,
                "segment_id": message.segment_id,
                "page_type_at_trigger": frame.page.page_type.value,
                "page_confidence_at_trigger": frame.page.confidence,
                "identity_state_at_trigger": "stable",
                "identity_key": identity.identity_key,
                "identity_confidence": identity.confidence,
                "ocr_confidence": message.ocr_confidence,
                "ocr_engine": frame.ocr_result.engine,
                "capture_generation": frame.capture_generation,
                "window_generation": frame.window_generation,
                "source_job_id": frame.job_id,
                "snapshot_hash": frame.snapshot_hash,
                "trigger_reason": "reliable_new_incoming_message",
                "message_source": message.source,
                "observed_at": message.observed_at,
                "hwnd": frame.hwnd,
                "target_app": frame.target_app,
                "performance_trace": performance_trace,
            }
            event = self.analysis_events.claim(
                message,
                target_instance_id=frame.target_instance_id,
                conversation_id=conversation_id,
                trigger_snapshot_json=json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
            if event is not None:
                created.append(event.id)
        return tuple(created)

    def _extract_memories(self, contact: Contact, inserted_messages: list[Message]) -> tuple[int, str]:
        """Turn the messages that just landed into pending long-term memories."""
        if not inserted_messages:
            return 0, "no_new_messages"
        if self.memory_extractor is None:
            return 0, "memory_extractor_unavailable"
        extraction = self.memory_extractor.extract(contact, inserted_messages)
        return extraction.created, extraction.skipped_reason or "created"

    def ingest_title_result(self, result) -> IngestionResult:
        stale_reason = self._stale_evidence_reason(result)
        if stale_reason:
            return IngestionResult(False, stale_reason, None, 0, 0)
        self.last_evidence = result
        self._last_pipeline_evidence_job_id = max(self._last_pipeline_evidence_job_id, int(result.job_id))
        boundary_error = self._account_boundary_error(result)
        if boundary_error:
            if self.identity_lock is not None:
                self.identity_lock.invalidate(boundary_error)
            self._close_active_session()
            return self._finish(IngestionResult(False, boundary_error, None, 0, 0))
        if result.ocr_result.warning:
            return self._finish(
                IngestionResult(False, f"title_ocr_warning:{result.ocr_result.warning}", None, 0, 0)
            )
        title_candidates = extract_title_candidates(
            result.ocr_result,
            result,
            min_confidence=self.min_title_confidence,
        )
        raw_title = title_candidates[0] if title_candidates else None
        # A suffix such as "(18)" may be a group member count or part of a
        # legitimate DM remark. Preserve it until full-page classification
        # proves the type. In particular, X (18) must never be stripped into
        # an already trusted DM X and inherit that DM's page/identity proof.
        conversation_type = ConversationType.UNKNOWN
        title = _normalize_spaces(raw_title) if raw_title else None
        if title and re.search(r"[（(]\s*\d{2,}\s*[)）]\s*$", title) and self.identity_lock is not None:
            stripped_title = _strip_group_member_count(title)
            candidate_type = self.identity_lock.candidate_conversation_type
            candidate_title = self.identity_lock.candidate_title.casefold()
            if (
                candidate_type == ConversationType.DM
                and stripped_title.casefold() == candidate_title
                and title.casefold() != candidate_title
            ):
                self.identity_lock.invalidate("title_group_requires_full_page")
                self._prebaseline.clear()
                self._close_active_session()
                return self._finish(IngestionResult(False, "title_group_requires_full_page", None, 0, 0))
            if candidate_type == ConversationType.GROUP:
                title = stripped_title
        if title is None:
            debug_lines = tuple(_title_candidate_debug_lines(result.ocr_result, result))
            return self._finish(
                IngestionResult(
                    False,
                    f"title_unavailable:{_format_title_candidates(debug_lines)}",
                    None,
                    0,
                    0,
                    debug_lines,
                )
            )
        identity_started_monotonic = time.monotonic()
        identity = self._observe_identity(result, title, conversation_type)
        self._record_identity_timing(result, identity, identity_started_monotonic)
        contact = identity.contact if identity.stable else None
        if contact is not None:
            _add_raw_title_alias(self.contacts, contact, raw_title, title)
        return self._finish(
            IngestionResult(
                identity.stable,
                identity.reason,
                contact,
                0,
                0,
                tuple(_title_candidate_debug_lines(result.ocr_result, result)),
                identity_stable=identity.stable,
                identity_key=identity.identity_key,
            )
        )

    def _observe_identity(
        self,
        result,
        title: str,
        conversation_type: ConversationType,
    ) -> IdentityResolution:
        confidence = _title_confidence(result.ocr_result, result, title)
        if self.identity_lock is None:
            return IdentityResolution(False, "identity_lock_unavailable", "", title, conversation_type, confidence, 0)
        return self.identity_lock.observe(
            job_id=result.job_id,
            platform=_platform_from_result(result),
            title=title,
            conversation_type=conversation_type,
            confidence=confidence,
            account_scope_id=result.account_scope_id,
            target_instance_id=result.target_instance_id,
            window_generation=result.window_generation,
            capture_generation=result.capture_generation,
            account_identity_epoch=getattr(result, "account_identity_epoch", 0),
            evidence_source=result,
        )

    def _record_identity_timing(
        self,
        result,
        identity: IdentityResolution,
        resolve_started_monotonic: float,
    ) -> None:
        """Attach local title-to-identity timing without changing evidence."""
        trace = getattr(result, "performance_trace", None)
        if not isinstance(trace, dict):
            return
        now = time.monotonic()
        trace["customer_resolve_ms"] = round((now - resolve_started_monotonic) * 1000.0, 3)
        if identity.stable:
            if self._identity_candidate_key == identity.identity_key and self._identity_candidate_started_monotonic:
                trace["identity_confirm_ms"] = round(
                    (now - self._identity_candidate_started_monotonic) * 1000.0,
                    3,
                )
            else:
                trace["identity_confirm_ms"] = trace["customer_resolve_ms"]
            trace["identity_confirmed_at"] = utc_now_iso()
            self._identity_candidate_key = ""
            self._identity_candidate_started_monotonic = 0.0
            return
        if identity.reason.startswith("identity_pending:") or identity.reason == "identity_waiting_page_type":
            if self._identity_candidate_key != identity.identity_key:
                self._identity_candidate_key = identity.identity_key
                self._identity_candidate_started_monotonic = now
            return
        self._identity_candidate_key = ""
        self._identity_candidate_started_monotonic = 0.0

    def _ensure_session(self, conversation_id: str, identity: IdentityResolution, result: PipelineResult) -> str:
        if self.conversations is None:
            raise RuntimeError("conversation repository unavailable")
        context_changed = (
            self._active_identity_key != identity.identity_key
            or self._active_conversation_id != conversation_id
            or self._active_scope_id != identity.account_scope_id
            or self._active_target_instance_id != identity.target_instance_id
            or self._active_window_generation != identity.window_generation
            or self._active_account_identity_epoch != identity.account_identity_epoch
        )
        if context_changed:
            self._close_active_session()
            session = self.conversations.start_session(
                conversation_id,
                hwnd=result.hwnd,
                stable_title=identity.title,
                identity_confidence=identity.confidence,
                account_scope_id=identity.account_scope_id,
                target_instance_id=identity.target_instance_id,
                capture_generation=identity.capture_generation,
            )
            self._active_identity_key = identity.identity_key
            self._active_conversation_id = conversation_id
            self._active_session_id = session.id
            self._active_scope_id = identity.account_scope_id
            self._active_target_instance_id = identity.target_instance_id
            self._active_window_generation = identity.window_generation
            self._active_capture_generation = identity.capture_generation
            self._active_account_identity_epoch = identity.account_identity_epoch
        else:
            self.conversations.touch_session(self._active_session_id)
            # Keep the newest anti-stale token for diagnostics without turning
            # an ordinary focus transition into a new conversation session.
            self._active_capture_generation = identity.capture_generation
        return self._active_session_id

    def _close_active_session(self) -> None:
        if self.conversations is not None and self._active_session_id:
            self.conversations.close_session(self._active_session_id)
        self._active_identity_key = ""
        self._active_conversation_id = ""
        self._active_session_id = ""
        self._active_scope_id = ""
        self._active_target_instance_id = ""
        self._active_window_generation = 0
        self._active_capture_generation = 0
        self._active_account_identity_epoch = 0

    def invalidate_current_screen(self, reason: str = "current_screen_body_changed") -> None:
        """Revoke message evidence without closing the proven customer's session."""
        self.last_evidence = None
        self.last_result = IngestionResult(False, reason, None, 0, 0)
        self._prebaseline.clear()

    def reset_context(self, reason: str = "manual_context_reset") -> None:
        if self.identity_lock is not None:
            self.identity_lock.invalidate(reason)
        self._prebaseline.clear()
        self._identity_candidate_key = ""
        self._identity_candidate_started_monotonic = 0.0
        self._close_active_session()

    def _stale_evidence_reason(self, result) -> str:
        """Reject old work before it can touch locks, sessions or business rows.

        Runtime's accepted-event fanout is the primary gate. This independent
        check protects direct callers and a callback delayed across a window,
        focus or account lifecycle transition. A rejection deliberately does
        not replace last_result/last_evidence or reset the current lock.
        """
        job_id = int(getattr(result, "job_id", 0) or 0)
        if job_id > 0 and job_id < self._last_pipeline_evidence_job_id:
            return "stale_pipeline_job"
        if self.current_state_provider is None:
            return ""
        current = self.current_state_provider()
        if current is None:
            return "stale_runtime_unavailable"
        expected = (
            current.window.hwnd,
            current.window.process_id,
            current.window_generation,
            current.capture_generation,
            current.account_identity_epoch,
            current.account_identity_runtime_key,
        )
        observed = (
            getattr(result, "hwnd", None),
            getattr(result, "process_id", None),
            getattr(result, "window_generation", 0),
            getattr(result, "capture_generation", 0),
            getattr(result, "account_identity_epoch", 0),
            getattr(result, "account_identity_runtime_key", ""),
        )
        names = ("hwnd", "process_id", "window_generation", "capture_generation", "account_identity_epoch", "account_identity_runtime_key")
        for name, value, authority in zip(names, observed, expected):
            if value != authority:
                return f"stale_{name}_mismatch"
        return ""

    @staticmethod
    def _account_boundary_error(result) -> str:
        if not getattr(result, "target_instance_id", ""):
            return "account_target_instance_unavailable"
        if not getattr(result, "account_scope_id", None):
            reason = getattr(result, "account_scope_reason", "") or "unbound"
            return f"account_scope_unbound:{reason}"
        return ""

    def _finish(self, result: IngestionResult) -> IngestionResult:
        if self.last_evidence is not None:
            result = replace(
                result,
                account_scope_id=result.account_scope_id or getattr(self.last_evidence, "account_scope_id", "") or "",
                target_instance_id=result.target_instance_id or getattr(self.last_evidence, "target_instance_id", ""),
            )
        self.last_result = result
        signature = f"{result.accepted}|{result.reason}|{result.contact.id if result.contact else '-'}"
        if self.logs and signature != self._last_log_signature:
            self._last_log_signature = signature
            contact_id = result.contact.id if result.contact else "-"
            self.logs.append(
                "info" if result.accepted else "warning",
                "ingestion",
                "pipeline_ingestion",
                (
                    f"accepted={result.accepted}, reason={result.reason}, contact={contact_id}, "
                    f"inserted={result.inserted_messages}, duplicate={result.duplicate_messages}, "
                    f"memories={result.memories_created}/{result.memory_reason or '-'}, "
                    f"title_candidates={','.join(result.title_candidates) or '-'}"
                ),
            )
        return result


def _prebaseline_context_key(identity: IdentityResolution, result: PipelineResult) -> str:
    if not identity.identity_key:
        return ""
    return "|".join(
        (
            identity.identity_key,
            result.target_instance_id,
            str(result.window_generation),
            str(result.capture_generation),
            str(result.account_identity_epoch),
        )
    )


def _eligible_for_auto_analysis_event(message: Message, conversation_id: str) -> bool:
    return (
        message.speaker == Speaker.OTHER
        and not message.partial
        and not message.order_uncertain
        and bool(message.contact_id)
        and bool(message.account_scope_id)
        and bool(message.session_id)
        and bool(message.segment_id)
        and message.conversation_id == conversation_id
        and ";verified_new=" in message.source
    )


def extract_contact_title(ocr_result: OcrResult, result: PipelineResult) -> str | None:
    candidates = extract_title_candidates(ocr_result, result)
    return candidates[0] if candidates else None


def extract_title_candidates(
    ocr_result: OcrResult,
    result: PipelineResult,
    *,
    min_confidence: float = 0.50,
) -> list[str]:
    normalized = normalize_ocr_regions(ocr_result, result.layout)
    candidates = [
        box for box in normalized.boxes
        if box.region == OcrRegion.TITLE and box.confidence >= min_confidence and _looks_like_contact_title(box.text)
    ]
    candidates.sort(
        key=lambda box: (
            -_title_candidate_score(box, result.layout),
            -box.confidence,
            box.rect.top,
            box.rect.left,
        )
    )
    values: list[str] = []
    seen: set[str] = set()
    for box in candidates:
        text = _normalize_spaces(box.text.strip())[:80]
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            values.append(text)
    return values


def _title_candidate_score(box, layout) -> float:
    """Rank title text by evidence, not OCR confidence alone."""
    text = _normalize_spaces(box.text)
    title = layout.title_rect
    horizontal = (box.rect.left + box.rect.right) / 2
    vertical = (box.rect.top + box.rect.bottom) / 2
    x_ratio = (horizontal - title.left) / max(1, title.width)
    y_ratio = (vertical - title.top) / max(1, title.height)
    score = float(box.confidence) * 4.0
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        score += 2.0
    if 2 <= len(text) <= 40:
        score += 1.0
    elif len(text) == 1:
        score += 0.35
    # Contact titles are left/centre aligned and vertically centred.  Window
    # buttons live high and at the far right; both are strong negative evidence.
    score += max(0.0, 1.8 - abs(y_ratio - 0.62) * 4.0)
    if x_ratio <= 0.62:
        score += 1.5 - max(0.0, x_ratio) * 0.8
    elif x_ratio >= 0.78:
        score -= 3.0
    if y_ratio < 0.28:
        score -= 3.0
    return score


def normalize_contact_title(title: str | None, conversation_type: ConversationType) -> str | None:
    if title is None:
        return None
    text = _normalize_spaces(title)
    if conversation_type == ConversationType.GROUP:
        text = _strip_group_member_count(text)
    return text[:80] if text else None


def _title_confidence(ocr_result: OcrResult, result, normalized_title: str) -> float:
    normalized = normalize_ocr_regions(ocr_result, result.layout)
    target = _normalize_spaces(normalized_title).casefold()
    matches = [
        float(box.confidence)
        for box in normalized.boxes
        if box.region == OcrRegion.TITLE
        and _normalize_spaces(box.text).casefold() == target
    ]
    if matches:
        return max(matches)
    # Group titles may have a trailing member count removed before identity
    # comparison. In that case use the best reliable title-region box.
    return max(
        (float(box.confidence) for box in normalized.boxes if box.region == OcrRegion.TITLE),
        default=0.0,
    )


def _title_candidate_debug_lines(ocr_result: OcrResult, result: PipelineResult) -> list[str]:
    normalized = normalize_ocr_regions(ocr_result, result.layout)
    boxes = [
        box for box in normalized.boxes
        if box.text.strip() and box.region in {OcrRegion.TITLE, OcrRegion.UNKNOWN}
    ]
    boxes.sort(key=lambda box: (-box.confidence, box.rect.top, box.rect.left))
    return [
        f"{box.region.value}:{box.confidence:.2f}:{_clip_debug_text(box.text)}"
        for box in boxes[:8]
    ]


def _format_title_candidates(lines: tuple[str, ...] | list[str]) -> str:
    return "|".join(lines) if lines else "none"


def _clip_debug_text(value: str, limit: int = 48) -> str:
    text = " ".join(value.strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _platform_from_result(result: PipelineResult) -> str:
    platform = (getattr(result, "target_app", "") or "").strip().lower()
    if platform:
        return platform
    return "wechat"


def _conversation_type_from_title(title: str) -> ConversationType:
    if "群" in title or "讨论组" in title or "交流群" in title:
        return ConversationType.GROUP
    if any(left in title and right in title for left, right in [("（", "）"), ("(", ")")]):
        import re

        if re.search(r"[（(]\s*\d{2,}\s*[)）]", title):
            return ConversationType.GROUP
    return ConversationType.DM


def _strip_group_member_count(title: str) -> str:
    previous = title
    while True:
        text = re.sub(r"\s*[（(]\s*\d{2,}\s*[)）]\s*$", "", previous).strip()
        if text == previous:
            return text
        previous = text


def _normalize_spaces(value: str) -> str:
    return " ".join(value.strip().split())


def _add_raw_title_alias(contacts: ContactRepository, contact: Contact, raw_title: str | None, normalized_title: str | None) -> None:
    if not raw_title or not normalized_title:
        return
    raw = _normalize_spaces(raw_title)
    if raw and raw != normalized_title:
        contacts.add_alias(contact.id, raw, "ocr_title")


def message_from_parsed(contact_id: str, parsed: ParsedOcrMessage, result: PipelineResult) -> Message:
    return Message(
        id=_fingerprint("message-id", contact_id, parsed, result),
        contact_id=contact_id,
        speaker=parsed.speaker,
        text=parsed.text.strip(),
        content_type="text",
        ocr_confidence=parsed.confidence,
        observed_at=result.created_at,
        message_time=_resolved_message_time(parsed, result),
        time_source="ocr" if parsed.time_text else "observed",
        partial=parsed.partial,
        fingerprint=_fingerprint("ocr", contact_id, parsed, result),
        source=f"ocr_pipeline:{result.ocr_result.engine}",
        sender_name=parsed.sender_name or "",
    )


def message_from_stitched(contact_id: str, stitched: StitchedMessage, result: PipelineResult) -> Message:
    return Message(
        id=_stitched_fingerprint("message-id", contact_id, stitched, result),
        contact_id=contact_id,
        speaker=stitched.speaker,
        text=stitched.text.strip(),
        content_type="text",
        ocr_confidence=stitched.confidence,
        observed_at=stitched.first_seen_at,
        message_time=stitched.message_time,
        time_source=stitched.time_source,
        partial=stitched.partial,
        fingerprint=_stitched_fingerprint("stitched", contact_id, stitched, result),
        source=f"ocr_stitched:{result.ocr_result.engine}",
        sender_name=stitched.sender_name,
    )


def _fingerprint(prefix: str, contact_id: str, parsed: ParsedOcrMessage, result: PipelineResult) -> str:
    bucket = (
        contact_id,
        parsed.speaker.value,
        parsed.sender_name or "",
        parsed.text.strip(),
        str(round(parsed.confidence, 1)),
        str(parsed.partial),
        ",".join(str(value) for value in parsed.rect.as_tuple()),
        result.ocr_result.engine,
    )
    digest = hashlib.sha256("|".join(bucket).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def _stitched_fingerprint(prefix: str, contact_id: str, stitched: StitchedMessage, result: PipelineResult) -> str:
    bucket = (
        contact_id,
        stitched.speaker.value,
        stitched.sender_name,
        " ".join(stitched.text.strip().split()).lower(),
        result.ocr_result.engine,
    )
    digest = hashlib.sha256("|".join(bucket).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def _new_stitched_messages(
    messages: list[StitchedMessage],
    inserted_before: int,
    inserted_after: int,
) -> list[StitchedMessage]:
    selected: list[StitchedMessage] = []
    if inserted_before > 0:
        selected.extend(messages[:inserted_before])
    if inserted_after > 0:
        selected.extend(messages[len(messages) - inserted_after:])
    return selected


def _looks_like_contact_title(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    blocked = {
        "微信", "WeChat", "搜索", "更多", "设置", "通讯录", "聊天信息", "文件传输助手",
        "最小化", "最大化", "还原", "关闭", "X", "x", "×", "□", "口", "-", "_", "—", "…", "...",
    }
    if text in blocked or text.casefold() in {value.casefold() for value in blocked}:
        return False
    if text.isdigit() or re.fullmatch(r"[（(]?\s*\d+\s*[)）]?", text):
        return False
    if len(text) <= 1 and not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return False
    if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text):
        return False
    return len(text) <= 80


def _resolved_message_time(parsed: ParsedOcrMessage, result: PipelineResult) -> str | None:
    if not parsed.time_text:
        return None
    observed = result.created_at
    try:
        from whochat.services.transcript_stitcher import _resolve_message_time

        resolved, source = _resolve_message_time(parsed.time_text, observed)
    except Exception:
        return None
    return resolved if source == "ocr" else None
