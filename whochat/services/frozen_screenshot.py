"""Explicit screenshot jobs own pixels and provenance, never the live desktop.

Two independent native captures are made BEFORE the selection overlay.  After
the drag, only these owned bytes are read.  The ordinary live pipeline and its
two-frame rules are unchanged.  No imported file can issue this capability.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
import time
import uuid

from PIL import Image

from whochat.core.models import utc_now_iso
from whochat.core.paths import app_data_dir
from whochat.services.manual_selection import SelectionError, capture_selection_frame


def _temporary():
    root = app_data_dir() / 'capture' / 'frozen'
    root.mkdir(parents=True, exist_ok=True)
    return TemporaryDirectory(prefix='request_', dir=root)


@dataclass(frozen=True)
class CapturedFrame:
    state: object
    image: object = field(repr=False)
    native_token: tuple = field(repr=False)
    account: object
    captured_at: str
    capture_id: str


@dataclass(frozen=True)
class CapturedPair:
    frames: tuple[CapturedFrame, CapturedFrame]

    @property
    def state(self):
        return self.frames[-1].state

    @property
    def image(self):
        return self.frames[-1].image

    @property
    def native_token(self):
        return self.frames[-1].native_token

    def close(self):
        for frame in self.frames:
            frame.image.close()


def close_frame(frame):
    if frame is not None:
        if isinstance(frame, CapturedPair):
            frame.close()
        else:
            frame.image.close()


def capture_frozen_pair(state, resolver, *, cancelled=None):
    """Bounded preparation, with actual foreground/account checks on EACH frame."""
    from whochat.services.manual_suggestions import read_native_window
    cancelled = cancelled or Event()
    frames = []
    started = time.monotonic()
    try:
        with _temporary() as directory:
            for attempt in range(4):
                if cancelled.is_set() or time.monotonic()-started >= 10:
                    raise SelectionError('manual_selection_capture_timeout')
                frame = capture_selection_frame(state, resolver)
                keep = False
                try:
                    captured_at = utc_now_iso()
                    path = Path(directory) / 'account.png'
                    frame.image.save(path)
                    account = resolver.observe(path, state.layout, state.window,
                                               native_capture_token=frame.native_token)
                    if resolver.capture_identity_reason(state.window, frame.native_token):
                        raise SelectionError('manual_selection_account_changed')
                    native = read_native_window(state.window.hwnd)
                    if (not native.foreground or not native.visible or native.minimized
                            or native.process_id != state.window.process_id
                            or native.rect != state.window.rect.as_tuple()
                            or native.client_rect != state.window.client_rect.as_tuple()):
                        raise SelectionError('manual_selection_geometry_changed')
                    if not account.actions_allowed:
                        # Only the documented pending confirmation is retryable.
                        if 'native_account_confirmation_pending' in account.reason:
                            continue
                        raise SelectionError('manual_selection_account_unverified')
                    frozen_state = replace(state, account_identity_epoch=account.identity_epoch,
                                           account_identity_runtime_key=account.runtime_key)
                    captured = CapturedFrame(frozen_state, frame.image, frame.native_token,
                                             account, captured_at, uuid.uuid4().hex)
                    if frames and _owner(frames[0]) != _owner(captured):
                        raise SelectionError('manual_selection_account_changed')
                    frames.append(captured)
                    keep = True
                    if len(frames) == 2:
                        return CapturedPair(tuple(frames))
                finally:
                    if not keep:
                        frame.image.close()
        raise SelectionError('manual_selection_account_unverified')
    except BaseException:
        for frame in frames:
            frame.image.close()
        raise


def _owner(frame):
    return (frame.state.window, frame.state.window_generation,
            frame.account.account_scope_id, frame.account.target_instance_id,
            frame.account.identity_epoch, frame.account.runtime_key, frame.native_token)


@dataclass(frozen=True)
class FrozenFrame:
    pixels: bytes = field(repr=False)
    size: tuple[int, int]
    digest: str
    captured_at: str
    capture_id: str
    account: object


@dataclass(frozen=True)
class FrozenRequest:
    request_id: int
    selection: object
    state: object
    frames: tuple[FrozenFrame, FrozenFrame]
    started: float
    deadline: float
    cancelled: Event = field(repr=False, compare=False)

    def check(self):
        if self.cancelled.is_set():
            raise SelectionError('snapshot_request_cancelled')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('snapshot_request_timeout')
        for frame in self.frames:
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(frame.captured_at)).total_seconds()
            if not 0 <= age < 120:
                raise SelectionError('snapshot_source_expired')


def freeze_selection(pair, selection, request_id):
    """Copy bytes, not a PIL view; verify both independently captured inputs."""
    from whochat.services.manual_suggestions import exact_frame_digests
    from whochat.services.manual_selection import _avatar_digest
    if (type(pair) is not CapturedPair or len(pair.frames) != 2
            or pair.frames[0].capture_id == pair.frames[1].capture_id
            or _owner(pair.frames[0]) != _owner(pair.frames[1])):
        raise SelectionError('snapshot_pair_invalid')
    frames = []
    for frame in pair.frames:
        if (frame.native_token != selection.native_token
                or (frame.state.window.hwnd, frame.state.window.process_id, frame.state.window_generation)
                != (selection.hwnd, selection.process_id, selection.window_generation)):
            raise SelectionError('manual_selection_owner_changed')
        title, body = exact_frame_digests(frame.image, selection.layout)
        if (title, body, _avatar_digest(frame.image, selection.layout)) != (
                selection.title_digest, selection.body_digest, selection.avatar_digest):
            raise SelectionError('snapshot_capture_changed')
        pixels = frame.image.tobytes()
        frames.append(FrozenFrame(pixels, frame.image.size, hashlib.sha256(pixels).hexdigest(),
                                  frame.captured_at, frame.capture_id, frame.account))
    start = time.monotonic()
    request = FrozenRequest(request_id, selection, replace(pair.state, layout=selection.layout),
                            tuple(frames), start, start+30.0, Event())
    request.check()
    return request


class SnapshotAuthority:
    """One process-local, immutable request seal, separate from live identity.

Stored observations/flags cannot mint it.  It is issued only after this module's
owned pair, normal OCR proofs, identity lock and database source checks pass.
It permits generation from that input; it NEVER authorizes clipboard actions.
"""
    def __init__(self, memory_builder):
        self.memory_builder = memory_builder
        self._lock = RLock()
        self._request = None
        self._context = None

    @staticmethod
    def recognizes(context):
        return bool(getattr(context, 'frozen_screenshot_id', ''))

    def revoke(self):
        with self._lock:
            if self._request is not None:
                self._request.cancelled.set()
            self._request = self._context = None

    def _issue(self, request, context):
        from whochat.ai.current_work_authorization import current_context_source_error
        request.check()
        if (context.frozen_screenshot_id != request.selection.selection_id
                or current_context_source_error(context)
                or not self.memory_builder.validate_current_view(context.current_message_view)
                or not self.memory_builder.validate_history_context(context.memory_context, context.current_message_view)):
            raise SelectionError('snapshot_context_invalid')
        with self._lock:
            self._request = request
            self._context = deepcopy(replace(context, current_work_cloud_authorization=None))

    def valid(self, context, *, display=False):
        try:
            with self._lock:
                request, sealed = self._request, self._context
                if request is None or sealed is None or request.cancelled.is_set():
                    return False
                if not display:
                    request.check()
                if replace(context, current_work_cloud_authorization=None) != sealed:
                    return False
            # Read the actual scoped observation/profile again. No source TTL
            # extension, global-current-customer fallback, or imported proof.
            return (self.memory_builder.validate_current_view(context.current_message_view)
                    and self.memory_builder.validate_history_context(context.memory_context, context.current_message_view))
        except (ValueError, TypeError, AttributeError, TimeoutError):
            return False


def _read_frame(request, frame, job_id, engine, directory):
    """Use the same OCR/parser/proof functions, with zero live capture calls."""
    from whochat.services import pipeline as p
    from whochat.services.manual_suggestions import title_identity_screen_rect
    from whochat.ocr.low_confidence import recover_low_confidence_content
    request.check()
    if hashlib.sha256(frame.pixels).hexdigest() != frame.digest:
        raise SelectionError('snapshot_pixels_corrupted')
    layout, state, account = request.selection.layout, request.state, frame.account
    path = Path(directory) / f'frame_{job_id}.png'
    with Image.frombytes('RGB', frame.size, frame.pixels) as image:
        image.save(path)
        title_path, title_layout, title_offset, title_crop = p._prepare_title_ocr_input(path, layout, frame_image=image)
        title_started = time.monotonic()
        title = p._offset_ocr_result(getattr(engine, 'recognize_title', engine.recognize)(title_path, title_layout), title_offset, path)
        title_ms = round((time.monotonic()-title_started)*1000)
        request.check()
        content_path, content_layout, offset, crop = p._prepare_ocr_input(path, layout, frame_image=image)
        body_started = time.monotonic()
        content = p._offset_ocr_result(engine.recognize(content_path, content_layout), offset, path)
        content, recovery = recover_low_confidence_content(content, title, layout, image, engine, path, request.check)
        body_ms = round((time.monotonic()-body_started)*1000)
        request.check()
        ocr = p._merge_ocr_results(title, content)
        page = p.classify_page_from_ocr(ocr, layout)
        if p.account_identity_boundary_reason(ocr, layout, page):
            raise SelectionError('snapshot_account_boundary')
        messages, report = p.parse_visible_messages_with_evidence(
            ocr, layout, direct_chat=page.page_type.value == 'chat_dm', frame_image=image)
        report['low_confidence_recovery_attempts'] = recovery['attempts']
        rect = title_identity_screen_rect(layout)
        band = p._shift_rect(rect, -layout.window_rect.left, -layout.window_rect.top)
        exact = p._exact_region_digest(path, band, frame_image=image)
        # Preserve the established clipboard pixel digest contract. Capture
        # independence lives in capture_id/job_id, not a made-up pixel hash.
        conversation_digest = p._exact_conversation_digest(path, layout, frame_image=image)
        snapshot_hash = hashlib.sha256((
            f'hwnd={state.window.hwnd}|pid={state.window.process_id}|'
            f'window_generation={state.window_generation}|capture_generation={state.capture_generation}|'
            f'pixels={conversation_digest}'
        ).encode('ascii')).hexdigest()[:32]
        owner = (state.window.hwnd, state.window.process_id, state.window_generation,
                 account.identity_epoch, account.runtime_key)
        anchor = p._same_frame_account_avatar_anchor(path, layout, account)
        if anchor is None:
            raise SelectionError('snapshot_account_avatar_unverified')
        proof = p._build_current_screen_capture_proof(
            job_id=job_id, state=state, account_context=account, snapshot_hash=snapshot_hash,
            created_at=frame.captured_at, layout=layout, ocr_result=ocr, page=page,
            messages=messages, parser_report=report, title_exact_hash=exact,
            capture_verified=True, title_verified=True, manual_request_token=(request.request_id, owner),
            account_avatar_anchor=anchor)
        # Truthful provenance: these are earlier native frames, not fresh live
        # screens taken after OCR or after the user moved the window.
        proof.update(snapshot_mode='frozen_pair_v1', snapshot_capture_id=frame.capture_id,
                     snapshot_pixel_sha256=frame.digest)
        if proof.get('valid') is not True:
            raise SelectionError('snapshot_ocr:'+str(proof.get('rejection_code', 'unavailable')))
        return p.PipelineResult(job_id=job_id, hwnd=state.window.hwnd, target_app='wechat',
            app_label=state.window.app_label, snapshot_hash=snapshot_hash, image_path=path,
            ocr_image_path=content_path, crop_rect=crop, layout=layout, ocr_result=ocr,
            page=page, messages=messages, created_at=frame.captured_at,
            title_ocr_image_path=title_path, title_crop_rect=title_crop,
            title_ocr_elapsed_ms=title_ms, content_ocr_elapsed_ms=body_ms,
            process_id=state.window.process_id, window_generation=state.window_generation,
            capture_generation=state.capture_generation, account_scope_id=account.account_scope_id,
            account_scope_name=account.account_scope_name, target_instance_id=account.target_instance_id,
            account_scope_reason=account.reason, account_identity_epoch=account.identity_epoch,
            account_identity_runtime_key=account.runtime_key, title_exact_hash=exact,
            title_visual_hash=str(p._title_visual_digest(path, band, frame_image=image)), screen_proof=proof,
            performance_trace={'manual_title_fresh_required': True, 'title_source': 'roi_ocr',
                               'title_cache_hit': False, 'snapshot_mode': 'frozen_pair_v1'},
            photo_crops=p.collect_photo_crops(path, layout, messages, permitted=True))


def prepare_context(request, services, engine):
    """Normal repositories/identity/history, isolated from live UI invalidation."""
    from whochat.services.ingestion import ChatIngestionService
    from whochat.services.identity_lock import StableIdentityLock
    from whochat.services.conversation_timeline import PersistentConversationTimeline
    from whochat.core.current_conversation import build_current_message_view
    from whochat.ai.models import ReplyContext
    from whochat.ai.photos import bind_reply_photos
    from whochat.ai.current_work_authorization import authorize_current_work_context
    from whochat.services.communication_intent import select_communication_intent

    timeline = PersistentConversationTimeline(services.conversations, services.messages, services.observations)
    state = request.state
    ingestion = ChatIngestionService(services.contacts, services.messages, services.logs,
        identity_lock=StableIdentityLock(services.contacts, services.identity_bindings),
        conversations=services.conversations, timeline=timeline,
        current_state_provider=lambda: state)
    with _temporary() as directory:
        results = [_read_frame(request, frame, request.request_id*2+index, engine, directory)
                   for index, frame in enumerate(request.frames)]
        # Do not let two disagreeing OCR titles become two separate contacts.
        from whochat.services.ingestion import extract_contact_title
        titles = [extract_contact_title(result.ocr_result, result) for result in results]
        if not titles[0] or titles[0] != titles[1]:
            raise SelectionError('snapshot_title_disagreement')
        # First independent title is identity evidence only. Persist the exact
        # selected (second) frame, even when both have the same pixel digest.
        # This avoids assigning the first observation to second-frame photos.
        from whochat.core.models import ConversationType
        from whochat.services.ingestion import normalize_contact_title
        first = ingestion._observe_identity(results[0],
            normalize_contact_title(titles[0], ConversationType.DM), ConversationType.DM)
        if first.stable or first.observations != 1:
            raise SelectionError('snapshot_identity_pair_invalid')
        result = results[1]
        state = replace(request.state, page=result.page)
        outcome = ingestion.ingest_pipeline_result(result)
        request.check()
        if not (outcome.accepted and outcome.identity_stable and outcome.contact and outcome.observation_id):
            raise SelectionError('snapshot_identity:'+outcome.reason)
        contact = outcome.contact
        view = build_current_message_view(services.observations.get(outcome.observation_id),
            conversation=services.conversations.get(outcome.conversation_id),
            expected_account_scope_id=contact.account_scope_id, expected_contact_id=contact.id,
            expected_conversation_id=outcome.conversation_id, expected_target_instance_id=outcome.target_instance_id,
            expected_session_id=outcome.session_id, max_source_job_id=result.job_id,
            max_capture_generation=state.capture_generation, now=datetime.now(timezone.utc))
        # Recording may change local history. Build and seal the context AFTER
        # that write so the transport sees precisely the same verified history.
        try:
            services.reply_memory_context.record_current_view(view)
        except Exception:
            from whochat.diagnostics import append_diagnostics_log
            append_diagnostics_log('capture_pipeline', 'frozen_history_record_failed')
        memory = services.reply_memory_context.build_for_current_view(view)
        context = ReplyContext(runtime=state, contact=contact, strategy=services.strategies.get(contact.strategy_id),
            messages=[], memories=services.memories.list_confirmed_for_contact(contact.id),
            memory_context=memory, current_message_view=view,
            knowledge_context=services.product_knowledge_context.build(memory_context=memory,
                current_message=view.current_message.text if view.current_message else ''),
            current_work_mode=select_communication_intent(view).mode,
            current_work_started_monotonic=request.started, current_work_deadline_monotonic=request.deadline,
            manual_retry_id=uuid.uuid4().hex, frozen_screenshot_id=request.selection.selection_id)
        photos, error = bind_reply_photos(view, result, state)
        context = replace(context, photos=photos, photo_error=error)
        if services.writing_feedback is not None:
            context = replace(context, writing_feedback=services.writing_feedback.learning_for(context))
        authority = services.reply_generator.snapshot_authority
        authority._issue(request, context)
        context = authorize_current_work_context(context, identity_lease_valid=authority.valid(context))
        if context.current_work_cloud_authorization is None:
            raise SelectionError('snapshot_authorization_failed')
        return context, result


def failure_message(reason):
    if 'untranscribed_voice' in reason:
        return '截图里有尚未转成文字的语音，请先在微信转文字，再重新截图。'
    if 'message_text_low_confidence' in reason:
        return '截图里的部分文字仍看不清，请放大微信或避开图片中的小字后重新截图。'
    if 'layout_or_coverage' in reason:
        return '截图中有未完整显示的聊天，请框入客户名称、双方头像和完整消息后重试。'
    if 'title' in reason or 'identity' in reason:
        return '截图中的客户名称未通过核验，请框入完整的顶部客户名称后重试。'
    if 'capture_changed' in reason:
        return '准备截图时聊天画面仍在变化，请等画面停稳后重新截图。'
    if 'timeout' in reason or 'expired' in reason:
        return '本次截图任务已超时，未自动重试；请重新截图生成。'
    if 'budget' in reason or 'quota' in reason:
        return '本地额度暂不可用或已用完；未自动重试，请查看剩余额度。'
    if 'account' in reason or 'owner' in reason:
        return '截图时的工作微信账号未通过核验，请打开目标客户后重新截图。'
    return '本次截图任务未完成，请查看下方具体原因后重新截图。'
