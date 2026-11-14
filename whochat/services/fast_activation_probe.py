from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from threading import Lock, RLock
import time
import unicodedata
from typing import Callable

from PIL import Image
from PySide6.QtCore import QObject, QTimer, Signal

from whochat.capture.policy import image_hash
from whochat.capture.screenshot import capture_rect
from whochat.core.models import utc_now_iso
from whochat.core.paths import app_data_dir
from whochat.core.runtime import LayoutRegions, PageType, Rect, RuntimeState, WindowState
from whochat.diagnostics import append_diagnostics_log
from whochat.ocr.engine import OcrEngine
from whochat.ocr.models import OcrRegion, OcrResult, OcrTextBox
from whochat.ocr.parser import _looks_like_group_title
from whochat.services.pipeline import CaptureContextChange, TitleOcrResult, title_identity_screen_rect
from whochat.services.account_scope import account_avatar_fingerprint, account_avatar_screen_rect


@dataclass(frozen=True)
class _ProbeCapture:
    sequence: int
    state: RuntimeState
    capture_rect: Rect
    image_path: Path
    exact_hash: str
    visual_hash: str
    capture_started_at: str
    capture_started_monotonic: float
    capture_elapsed_ms: int
    account_visual_hash: str = ""
    account_visual_reliable: bool = False
    account_checked: bool = False
    title_layout: LayoutRegions | None = None
    seed_revision: int | None = None


class FastActivationProbeService(QObject):
    """Independent title-only switch detector for the activation read path.

    It never reads the message area and never persists identity by itself. Two
    equal ROI captures are required before title evidence is published. The
    existing full pipeline and StableIdentityLock still own final identity.
    """

    title_ready = Signal(object)
    context_changed = Signal(object)
    status_changed = Signal(str)

    def __init__(
        self,
        *,
        current_state_provider: Callable[[], RuntimeState],
        last_title_provider: Callable[[], TitleOcrResult | None],
        fast_title_lookup: Callable[[str, str, str, str], object | None],
        ocr_engine_factory: Callable[[], OcrEngine],
        same_contact_title_validator: Callable[[TitleOcrResult], bool] | None = None,
        trusted_dm_anchor_validator: Callable[[TitleOcrResult, object], bool] | None = None,
        account_identity_guard: Callable[[object, str, bool], object] | None = None,
        capture_func=None,
        poll_interval_ms: int = 90,
        job_timeout_seconds: float = 10.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.current_state_provider = current_state_provider
        self.last_title_provider = last_title_provider
        self.fast_title_lookup = fast_title_lookup
        self.same_contact_title_validator = same_contact_title_validator
        self.trusted_dm_anchor_validator = trusted_dm_anchor_validator
        self.account_identity_guard = account_identity_guard
        self.ocr_engine_factory = ocr_engine_factory
        self.capture_func = capture_func or capture_rect
        self._timer = QTimer(self)
        self._timer.setInterval(max(80, int(poll_interval_ms)))
        self._timer.timeout.connect(self._tick)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whochat-title-probe")
        self._running: Future | None = None
        self._engine: OcrEngine | None = None
        self._sequence = 0
        self._lifecycle_lock = RLock()
        self._seed_revision = 0
        self._active_sequence: int | None = None
        self._processing_sequence: int | None = None
        self._work_started_monotonic = 0.0
        self._job_timeout_seconds = max(1.0, float(job_timeout_seconds))
        self._timeout_restarts = 0
        self._timeout_boundary: tuple[object, ...] | None = None
        self._timeout_title_hash = ""
        self._retired_engines: dict[int, OcrEngine] = {}
        self._retired_sequences: set[int] = set()
        self._stopped = False
        self._candidate_hash = ""
        self._candidate_count = 0
        self._candidate_started_at = ""
        self._candidate_started_monotonic = 0.0
        self._published_hash = ""
        self._published_title = ""
        self._published_raw_title = ""
        self._published_identity_boundary: tuple[object, ...] | None = None
        self._invalidated_hash = ""
        self._context_signature: tuple[object, ...] | None = None
        self._seed_confirmation_pending = False
        self._trusted_unknown_anchor_key: tuple[object, ...] | None = None
        self._metrics_lock = Lock()
        self._tick_scheduled: deque[float] = deque(maxlen=2048)
        self._tick_started: deque[float] = deque(maxlen=2048)
        self._title_resolve_ms: deque[float] = deque(maxlen=512)
        self._tick_skipped: dict[str, int] = {}
        self._ocr_started = 0
        self._ocr_completed = 0
        self._result_applied = 0
        self._seed_confirmations_started = 0
        self._seed_confirmations_completed = 0
        self._unknown_page_dm_seeds = 0
        self._last_account_check_monotonic = 0.0
        self._account_check_interval_seconds = 0.75

    @property
    def is_running(self) -> bool:
        # The capture Future becomes done before its callback performs title
        # OCR. Count that callback as part of the same job, otherwise a timer
        # tick can start a replacement while the first OCR still commits.
        with self._lifecycle_lock:
            return (
                self._active_sequence == self._sequence
                or self._processing_sequence == self._sequence
                or (self._running is not None and not self._running.done())
            )

    def start(self) -> None:
        self._stopped = False
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        with self._lifecycle_lock:
            self._stopped = True
            active_sequence = self._active_sequence or self._processing_sequence
            self._sequence += 1
            if self._running is not None and not self._running.done():
                self._running.cancel()
            if active_sequence is not None and self._engine is not None:
                self._retired_engines[active_sequence] = self._engine
                self._engine = None
            self._active_sequence = None
            self._processing_sequence = None
            self._work_started_monotonic = 0.0
        self._executor.shutdown(wait=False, cancel_futures=True)
        shutdown = getattr(self._engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

    def seed_from_full_title(self, result: TitleOcrResult) -> None:
        with self._lifecycle_lock:
            self._seed_from_full_title(result)

    def _seed_from_full_title(self, result: TitleOcrResult) -> None:
        if not result.title_exact_hash:
            return
        state = self.current_state_provider()
        if self._boundary_ineligible_reason(state, result):
            return
        signature = _context_signature(state, result)
        same_boundary = signature == self._context_signature
        previous_hash = self._published_hash if same_boundary else ""
        previous_title = self._published_title if same_boundary else ""
        observed_title = _normalized_title_observation(result.ocr_result)
        raw_title = _raw_title_observation(result.ocr_result)
        identity_boundary = _identity_boundary_signature(state, result)
        if (identity_boundary == self._published_identity_boundary and self._published_raw_title
                and raw_title != self._published_raw_title and _has_group_title_evidence(raw_title)):
            # A newly observed group/count title is not a DM heartbeat. Keep
            # the old published fingerprint until two independent probe frames
            # can suspend it; a Full seed must not hide this change by stripping
            # its member count. Still retire every older in-flight callback.
            self._seed_revision += 1
            self._context_signature = signature
            self._candidate_hash = ""
            self._candidate_count = 0
            self._candidate_started_at = ""
            self._candidate_started_monotonic = 0.0
            self._seed_confirmation_pending = False
            self._trusted_unknown_anchor_key = None
            return

        same_contact = False
        if self.same_contact_title_validator is not None:
            try:
                same_contact = bool(self.same_contact_title_validator(result))
            except Exception as exc:
                append_diagnostics_log(
                    "instant_activation",
                    f"seed_same_contact_validation_failed error={type(exc).__name__}:{exc}",
                )

        trusted_exact_dm = self._trust_exact_dm_anchor(state, result, observed_title)
        if state.page.page_type != PageType.UNKNOWN:
            self._trusted_unknown_anchor_key = None

        # The full pipeline's title crop is frame 1.  When it is not merely a
        # heartbeat for the retained Contact, the independent title probe owns
        # frame 2 and can publish identity without waiting for message OCR or
        # the next AutoCapture job.  UNKNOWN pages are admitted only through a
        # Every accelerated switch also requires a previously confirmed, exact
        # account/target-scoped DM fingerprint. This prevents a stale CHAT_DM
        # runtime classification from admitting a new group/non-chat title.
        confirmation_pending = bool(
            trusted_exact_dm
            and not same_contact
            and state.page.page_type in {PageType.CHAT_DM, PageType.UNKNOWN}
        )
        if signature != self._timeout_boundary or result.title_exact_hash != self._timeout_title_hash:
            self._timeout_restarts = 0
            self._timeout_boundary = signature
            self._timeout_title_hash = result.title_exact_hash
        self._seed_revision += 1
        self._context_signature = signature
        self._candidate_hash = result.title_exact_hash
        self._candidate_count = 1 if confirmation_pending else 2
        self._candidate_started_at = str(
            result.performance_trace.get("contact_switch_first_seen_at")
            or result.performance_trace.get("contact_switch_detected_at")
            or result.performance_trace.get("title_capture_started_at")
            or result.created_at
            or ""
        ) if confirmation_pending else ""
        self._candidate_started_monotonic = time.monotonic() if confirmation_pending else 0.0
        self._seed_confirmation_pending = confirmation_pending
        if confirmation_pending:
            self._published_hash = previous_hash
            self._published_title = previous_title
            with self._metrics_lock:
                self._seed_confirmations_started += 1
                if state.page.page_type == PageType.UNKNOWN:
                    self._unknown_page_dm_seeds += 1
        else:
            self._published_hash = result.title_exact_hash
            self._published_title = observed_title
            self._published_raw_title = raw_title
            self._published_identity_boundary = identity_boundary
        self._invalidated_hash = ""

    def _tick(self) -> None:
        now = time.monotonic()
        with self._metrics_lock:
            self._tick_scheduled.append(now)
        with self._lifecycle_lock:
            if self._stopped:
                return
            if self.is_running:
                if not self._expire_timed_out_job(now):
                    self._record_skip("worker_busy")
                    return
        state = self.current_state_provider()
        anchor = self.last_title_provider()
        ineligible = self._ineligible_reason(state, anchor)
        if ineligible:
            self._record_skip(ineligible)
            return
        signature = _context_signature(state, anchor)
        with self._lifecycle_lock:
            if self._stopped or self.is_running:
                return
            if self._timeout_boundary != signature:
                self._timeout_restarts = 0
                self._timeout_boundary = signature
                self._timeout_title_hash = anchor.title_exact_hash
            if len(self._retired_sequences) >= 2:
                self._record_skip("watchdog_workers_quarantined")
                return
            if self._timeout_restarts >= 2:
                # The normal pipeline remains available; never accumulate
                # unlimited native OCR workers on an unchanged stuck context.
                self._record_skip("watchdog_retry_exhausted")
                return
            self._sequence += 1
            sequence = self._sequence
            self._active_sequence = sequence
            self._work_started_monotonic = now
            seed_revision = self._seed_revision
        with self._metrics_lock:
            self._tick_started.append(time.monotonic())
        check_account = bool(self.account_identity_guard) and (
            now - self._last_account_check_monotonic >= self._account_check_interval_seconds
        )
        if check_account:
            self._last_account_check_monotonic = now
        self._running = self._executor.submit(
            self._capture_probe, sequence, state, check_account, anchor.layout, seed_revision,
        )
        self._running.add_done_callback(lambda future: self._on_captured(sequence, anchor, future))

    def _expire_timed_out_job(self, now: float) -> bool:
        if not self._work_started_monotonic or now - self._work_started_monotonic < self._job_timeout_seconds:
            return False
        expired_sequence = self._active_sequence or self._processing_sequence or self._sequence
        self._sequence += 1
        self._active_sequence = None
        self._processing_sequence = None
        self._work_started_monotonic = 0.0
        self._timeout_restarts += 1
        self._retired_sequences.add(expired_sequence)
        self._seed_revision += 1
        self._candidate_hash = ""
        self._candidate_count = 0
        self._candidate_started_at = ""
        self._candidate_started_monotonic = 0.0
        self._seed_confirmation_pending = False
        self._trusted_unknown_anchor_key = None
        future, old_executor, old_engine = self._running, self._executor, self._engine
        self._running = None
        self._engine = None
        if old_engine is not None:
            self._retired_engines[expired_sequence] = old_engine
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whochat-title-probe-recovery")
        if future is not None:
            future.cancel()
        old_executor.shutdown(wait=False, cancel_futures=True)
        self.status_changed.emit(f"title_probe_timeout:sequence={expired_sequence}")
        append_diagnostics_log("instant_activation", f"title_probe_watchdog_expired sequence={expired_sequence}")
        return True

    def _capture_is_current(self, sequence: int, anchor: TitleOcrResult, captured: _ProbeCapture, revision: int | None = None) -> bool:
        current = self.current_state_provider()
        latest_anchor = self.last_title_provider()
        return bool(
            not self._stopped
            and sequence == self._sequence
            and (revision is None or revision == self._seed_revision)
            and (captured.seed_revision is None or captured.seed_revision == self._seed_revision)
            and self._eligible(current, anchor)
            and latest_anchor is not None
            and not self._boundary_ineligible_reason(current, latest_anchor)
            and _context_signature(current, latest_anchor) == _context_signature(current, anchor)
            and latest_anchor.title_exact_hash == anchor.title_exact_hash
            and _context_signature(current, anchor) == _context_signature(captured.state, anchor)
            and (captured.title_layout is None or (
                captured.capture_rect == title_identity_screen_rect(anchor.layout)
                and captured.capture_rect == title_identity_screen_rect(captured.title_layout)
            ))
        )

    def _eligible(self, state: RuntimeState, anchor: TitleOcrResult | None) -> bool:
        return not self._ineligible_reason(state, anchor)

    def _ineligible_reason(self, state: RuntimeState, anchor: TitleOcrResult | None) -> str:
        if anchor is None:
            return "anchor_missing"
        boundary_reason = self._boundary_ineligible_reason(state, anchor)
        if boundary_reason:
            return boundary_reason
        if state.page.page_type == PageType.CHAT_DM:
            return ""
        if state.page.page_type == PageType.UNKNOWN and self._unknown_anchor_is_trusted(state, anchor):
            return ""
        return "not_chat_dm"

    def _boundary_ineligible_reason(self, state: RuntimeState, anchor: TitleOcrResult) -> str:
        if state.paused:
            return "runtime_paused"
        if state.window.state != WindowState.VISIBLE or state.window.hwnd is None:
            return "window_not_visible"
        if not state.window.foreground:
            return "wechat_not_foreground"
        if state.layout is None:
            return "layout_missing"
        if not _anchor_geometry_is_current(state, anchor.layout):
            return "title_anchor_geometry_stale"
        if not anchor.title_exact_hash:
            return "title_anchor_hash_missing"
        if not anchor.account_scope_id or not anchor.target_instance_id:
            return "identity_boundary_missing"
        if anchor.hwnd != state.window.hwnd or anchor.process_id != state.window.process_id:
            return "window_anchor_stale"
        if anchor.window_generation != state.window_generation:
            return "window_generation_stale"
        if anchor.capture_generation != state.capture_generation:
            return "capture_generation_stale"
        if anchor.account_identity_epoch != state.account_identity_epoch:
            return "account_identity_epoch_stale"
        if anchor.account_identity_runtime_key != state.account_identity_runtime_key:
            return "account_identity_runtime_stale"
        return ""

    def _trust_exact_dm_anchor(
        self,
        state: RuntimeState,
        anchor: TitleOcrResult,
        observed_title: str,
    ) -> bool:
        self._trusted_unknown_anchor_key = None
        if (
            not observed_title
            or _has_group_title_evidence(_raw_title_observation(anchor.ocr_result))
            or self.trusted_dm_anchor_validator is None
        ):
            return False
        hint = self.fast_title_lookup(
            anchor.account_scope_id or "",
            anchor.target_instance_id,
            anchor.title_exact_hash,
            "",  # UNKNOWN-page evidence must be exact; visual similarity is insufficient.
        )
        hint_title = str(getattr(hint, "normalized_title", "") or "").strip().casefold()
        if (
            hint is None
            or float(getattr(hint, "confidence", 0.0) or 0.0) < 0.95
            or not hint_title
            or hint_title != observed_title.casefold()
        ):
            return False
        try:
            trusted = bool(self.trusted_dm_anchor_validator(anchor, hint))
        except Exception as exc:
            append_diagnostics_log(
                "instant_activation",
                f"trusted_dm_anchor_validation_failed error={type(exc).__name__}:{exc}",
            )
            trusted = False
        if trusted:
            self._trusted_unknown_anchor_key = _unknown_anchor_key(state, anchor)
        return trusted

    def _unknown_anchor_is_trusted(self, state: RuntimeState, anchor: TitleOcrResult) -> bool:
        return bool(
            self._trusted_unknown_anchor_key is not None
            and self._trusted_unknown_anchor_key == _unknown_anchor_key(state, anchor)
        )

    def _record_skip(self, reason: str) -> None:
        with self._metrics_lock:
            self._tick_skipped[reason] = self._tick_skipped.get(reason, 0) + 1

    def metrics_snapshot(self) -> dict[str, object]:
        """Bounded in-memory watcher timings; no high-frequency log writes."""

        with self._metrics_lock:
            scheduled = tuple(self._tick_scheduled)
            started = tuple(self._tick_started)
            resolve_ms = tuple(self._title_resolve_ms)
            skipped = dict(self._tick_skipped)
            ocr_started = self._ocr_started
            ocr_completed = self._ocr_completed
            result_applied = self._result_applied
            seed_started = self._seed_confirmations_started
            seed_completed = self._seed_confirmations_completed
            unknown_page_dm_seeds = self._unknown_page_dm_seeds
        return {
            "configured_interval_ms": int(self._timer.interval()),
            "tick_scheduled": len(scheduled),
            "tick_started": len(started),
            "tick_skipped_reason": skipped,
            "scheduled_interval_ms": _distribution(_intervals_ms(scheduled)),
            "started_interval_ms": _distribution(_intervals_ms(started)),
            "ocr_started": ocr_started,
            "ocr_completed": ocr_completed,
            "ocr_elapsed_ms": _distribution(resolve_ms),
            "result_applied": result_applied,
            "seed_confirmations_started": seed_started,
            "seed_confirmations_completed": seed_completed,
            "unknown_page_dm_seeds": unknown_page_dm_seeds,
        }

    def _capture_probe(
        self, sequence: int, state: RuntimeState, check_account: bool = False,
        title_layout: LayoutRegions | None = None, seed_revision: int | None = None,
    ) -> _ProbeCapture:
        assert state.layout is not None
        started_at = utc_now_iso()
        started = time.monotonic()
        output = app_data_dir() / "capture" / f"activation_probe_{sequence}.png"
        # Full title fingerprints use screenshot-refined geometry, while the
        # runtime intentionally retains its layout prior. Freeze the accepted
        # Full anchor ROI for this task so stable pixels cannot look changed.
        # Account-avatar capture below still uses the independent runtime ROI.
        if title_layout is None or not _anchor_geometry_is_current(state, title_layout):
            raise RuntimeError("title_anchor_geometry_stale")
        capture_rect = title_identity_screen_rect(title_layout)
        image_path = Path(self.capture_func(capture_rect.as_tuple(), output))
        try:
            with Image.open(image_path) as title_image:
                if title_image.size != (capture_rect.width, capture_rect.height):
                    raise RuntimeError("title_probe_capture_shape_mismatch")
            exact_hash = _exact_image_digest(image_path)
            visual_hash = str(image_hash(image_path, size=8))
        except Exception:
            if image_path.resolve() == output.resolve():
                _safe_unlink(image_path)
            raise
        account_visual_hash = ""
        account_visual_reliable = False
        if check_account:
            account_output = output.with_name(f"{output.stem}_account{output.suffix}")
            account_rect = account_avatar_screen_rect(state.layout)
            account_image = Path(self.capture_func(account_rect.as_tuple(), account_output))
            try:
                account_visual_hash, account_visual_reliable = account_avatar_fingerprint(account_image)
            finally:
                if account_image.resolve() == account_output.resolve():
                    _safe_unlink(account_image)
        return _ProbeCapture(
            sequence,
            state,
            capture_rect,
            image_path,
            exact_hash,
            visual_hash,
            started_at,
            started,
            int((time.monotonic() - started) * 1000),
            account_visual_hash,
            account_visual_reliable,
            check_account,
            title_layout,
            seed_revision,
        )

    def _on_captured(self, sequence: int, anchor: TitleOcrResult, future: Future) -> None:
        captured = None
        try:
            captured = future.result()
            with self._lifecycle_lock:
                if not self._capture_is_current(sequence, anchor, captured):
                    return
                self._processing_sequence = sequence
                if not self._work_started_monotonic:
                    self._work_started_monotonic = time.monotonic()
            self._process_captured(sequence, anchor, captured)
            with self._lifecycle_lock:
                if self._capture_is_current(sequence, anchor, captured):
                    self._timeout_restarts = 0
        except Exception as exc:
            append_diagnostics_log("instant_activation", f"title_probe_failed error={type(exc).__name__}:{exc}")
        finally:
            with self._lifecycle_lock:
                if self._active_sequence == sequence:
                    self._active_sequence = None
                if self._processing_sequence == sequence:
                    self._processing_sequence = None
                if sequence == self._sequence:
                    self._work_started_monotonic = 0.0
                retired_engine = self._retired_engines.pop(sequence, None)
                self._retired_sequences.discard(sequence)
            shutdown = getattr(retired_engine, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
            if captured is not None:
                expected = app_data_dir() / "capture" / f"activation_probe_{captured.sequence}.png"
                if captured.image_path.resolve() == expected.resolve():
                    _safe_unlink(captured.image_path)

    def _process_captured(self, sequence: int, anchor: TitleOcrResult, captured: _ProbeCapture) -> None:
        with self._lifecycle_lock:
            current = self.current_state_provider()
            if not self._capture_is_current(sequence, anchor, captured):
                return
            if captured.account_checked and self.account_identity_guard is not None:
                guard = self.account_identity_guard(
                    current.window,
                    captured.account_visual_hash,
                    captured.account_visual_reliable,
                )
                if not bool(getattr(guard, "trusted", False)):
                    if bool(getattr(guard, "boundary_started", False)):
                        self.context_changed.emit(
                            CaptureContextChange(
                                job_id=-captured.sequence,
                                hwnd=current.window.hwnd,
                                process_id=current.window.process_id,
                                window_generation=current.window_generation,
                                capture_generation=current.capture_generation,
                                snapshot_hash=captured.exact_hash[:32],
                                reason=(
                                    "account_identity_revalidation:"
                                    f"{getattr(guard, 'reason', 'fingerprint_mismatch')}"
                                ),
                                account_scope_id=None,
                                target_instance_id="",
                                account_identity_epoch=int(getattr(guard, "identity_epoch", 0) or 0),
                                account_identity_runtime_key=str(
                                    getattr(guard, "runtime_key", "")
                                    or current.account_identity_runtime_key
                                ),
                            )
                        )
                    return
            signature = (
                *_context_signature(current, anchor),
            )
            if signature != self._context_signature:
                self._context_signature = signature
                self._candidate_hash = ""
                self._candidate_count = 0
                self._candidate_started_at = ""
                self._candidate_started_monotonic = 0.0
                anchor_raw_title = _raw_title_observation(anchor.ocr_result)
                identity_boundary = _identity_boundary_signature(current, anchor)
                changed_group_anchor = bool(
                    identity_boundary == self._published_identity_boundary
                    and self._published_raw_title and anchor_raw_title != self._published_raw_title
                    and _has_group_title_evidence(anchor_raw_title)
                )
                if not changed_group_anchor:
                    self._published_hash = anchor.title_exact_hash or ""
                    self._published_title = _normalized_title_observation(anchor.ocr_result)
                    self._published_raw_title = anchor_raw_title
                    self._published_identity_boundary = identity_boundary
                self._invalidated_hash = ""
                self._seed_confirmation_pending = False
                self._trusted_unknown_anchor_key = None
            if (
                current.page.page_type == PageType.UNKNOWN
                and captured.exact_hash != anchor.title_exact_hash
            ):
                # The only lightweight DM evidence on UNKNOWN is tied to the
                # exact stored title pixels. A changed frame returns to the
                # normal full-page classifier instead of being guessed.
                self._trusted_unknown_anchor_key = None
                self._seed_confirmation_pending = False
                self._candidate_hash = ""
                self._candidate_count = 0
                self._record_skip("unknown_page_exact_fingerprint_changed")
                return
            if captured.exact_hash == self._candidate_hash:
                self._candidate_count += 1
            else:
                self._candidate_hash = captured.exact_hash
                self._candidate_count = 1
                self._candidate_started_at = captured.capture_started_at
                self._candidate_started_monotonic = captured.capture_started_monotonic
            if self._candidate_count < 2 or (
                captured.exact_hash == self._published_hash and not self._seed_confirmation_pending
            ):
                return
            revision = self._seed_revision
        # Never hold the lifecycle lock during native OCR: the main thread
        # must be able to seed a newer title, expire this job or change focus.
        result = self._resolve_title(captured, anchor)
        with self._lifecycle_lock:
            if not self._capture_is_current(sequence, anchor, captured, revision):
                self._record_skip("late_title_result")
                return
            current = self.current_state_provider()
            if result is None:
                self._emit_possible_switch(current, anchor, captured)
                if not self._capture_is_current(sequence, anchor, captured, revision):
                    return
                # Require two fresh stable observations before retrying OCR;
                # this prevents an unknown title from becoming a busy loop.
                self._candidate_count = 0
                self._candidate_started_at = ""
                self._candidate_started_monotonic = 0.0
                return
            previous_title = self._published_title or _normalized_title_observation(anchor.ocr_result)
            resolved_title = _normalized_title_observation(result.ocr_result)
            raw_title = _raw_title_observation(result.ocr_result)
            previous_raw_title = self._published_raw_title or _raw_title_observation(anchor.ocr_result)
            if raw_title != previous_raw_title and _has_group_title_evidence(raw_title):
                # Do not normalize a new group/count suffix into the old DM,
                # or let a same-contact resolver bypass this weak boundary.
                self._emit_possible_switch(current, anchor, captured)
                return
            if previous_title and resolved_title == previous_title and not self._seed_confirmation_pending:
                # The pixels changed, but the normalized conversation title
                # did not. Treat this as a repaint heartbeat: learn the new
                # published fingerprint locally and do not generate another
                # identity/activation event.
                self._published_hash = captured.exact_hash
                self._published_title = resolved_title
                self._published_raw_title = raw_title
                self._invalidated_hash = ""
                return
            if self.same_contact_title_validator is not None:
                try:
                    same_contact = bool(self.same_contact_title_validator(result))
                except Exception as exc:
                    append_diagnostics_log(
                        "instant_activation",
                        f"same_contact_title_validation_failed error={type(exc).__name__}:{exc}",
                    )
                    same_contact = False
                if not self._capture_is_current(sequence, anchor, captured, revision):
                    return
                if same_contact:
                    # A truncated/variant title that still resolves uniquely to
                    # the retained account-scoped Contact is weak evidence, not
                    # a switch. Learn the new pixels in memory without
                    # suspending actions or rebuilding the displayed snapshot.
                    self._published_hash = captured.exact_hash
                    self._published_title = resolved_title
                    self._published_raw_title = raw_title
                    self._invalidated_hash = ""
                    self._seed_confirmation_pending = False
                    self._trusted_unknown_anchor_key = None
                    return
            self._emit_possible_switch(current, anchor, captured)
            if not self._capture_is_current(sequence, anchor, captured, revision):
                return
            self._published_hash = captured.exact_hash
            self._published_title = resolved_title
            self._published_raw_title = raw_title
            self._invalidated_hash = ""
            completed_seed_confirmation = self._seed_confirmation_pending
            self._seed_confirmation_pending = False
            self._trusted_unknown_anchor_key = None
            with self._metrics_lock:
                self._result_applied += 1
                if completed_seed_confirmation:
                    self._seed_confirmations_completed += 1
            self.title_ready.emit(result)

    def _emit_possible_switch(
        self,
        current: RuntimeState,
        anchor: TitleOcrResult,
        captured: _ProbeCapture,
    ) -> None:
        if (
            not self._published_hash
            or captured.exact_hash == self._published_hash
            or captured.exact_hash == self._invalidated_hash
        ):
            return
        self._invalidated_hash = captured.exact_hash
        self.context_changed.emit(
            CaptureContextChange(
                job_id=-captured.sequence,
                hwnd=current.window.hwnd,
                process_id=current.window.process_id,
                window_generation=current.window_generation,
                capture_generation=current.capture_generation,
                snapshot_hash=captured.exact_hash[:32],
                reason="fast_title_probe_changed_stable",
                account_scope_id=anchor.account_scope_id,
                target_instance_id=anchor.target_instance_id,
                account_identity_epoch=current.account_identity_epoch,
                account_identity_runtime_key=current.account_identity_runtime_key,
            )
        )

    def _resolve_title(self, captured: _ProbeCapture, anchor: TitleOcrResult) -> TitleOcrResult | None:
        state = captured.state
        assert state.layout is not None
        lookup_started = time.monotonic()
        with self._metrics_lock:
            self._ocr_started += 1
        trusted_unknown_page = bool(
            state.page.page_type == PageType.UNKNOWN
            and self._unknown_anchor_is_trusted(state, anchor)
        )
        hint = self.fast_title_lookup(
            anchor.account_scope_id or "",
            anchor.target_instance_id,
            captured.exact_hash,
            "" if trusted_unknown_page else captured.visual_hash,
        )
        source = "fast_probe_fingerprint"
        if hint is not None and float(getattr(hint, "confidence", 0.0) or 0.0) >= 0.95:
            text = str(getattr(hint, "normalized_title", "") or "").strip()
            if not text:
                self._record_title_resolve_completed(lookup_started)
                return None
            title_result = OcrResult(
                [OcrTextBox(text, captured.capture_rect, float(hint.confidence), OcrRegion.TITLE, source)],
                str(captured.image_path),
                "Confirmed Title Fingerprint",
                None,
            )
        else:
            source = "fast_probe_roi_ocr"
            with self._lifecycle_lock:
                if not self._capture_is_current(captured.sequence, anchor, captured):
                    self._record_title_resolve_completed(lookup_started)
                    return None
                if self._engine is None:
                    self._engine = self.ocr_engine_factory()
                engine = self._engine
            raw = getattr(engine, "recognize_title", engine.recognize)(captured.image_path, captured.title_layout)
            if raw.warning and not raw.boxes:
                self._record_title_resolve_completed(lookup_started)
                return None
            title_result = OcrResult(
                [
                    OcrTextBox(
                        box.text,
                        Rect(
                            box.rect.left + captured.capture_rect.left,
                            box.rect.top + captured.capture_rect.top,
                            box.rect.right + captured.capture_rect.left,
                            box.rect.bottom + captured.capture_rect.top,
                        ),
                        box.confidence,
                        OcrRegion.TITLE,
                        source,
                    )
                    for box in raw.boxes
                ],
                str(captured.image_path),
                raw.engine,
                raw.warning,
            )
            verify_path = app_data_dir() / "capture" / f"activation_probe_{captured.sequence}_verify.png"
            verify_image = Path(self.capture_func(captured.capture_rect.as_tuple(), verify_path))
            try:
                with Image.open(verify_image) as image:
                    if image.size != (captured.capture_rect.width, captured.capture_rect.height):
                        self._record_title_resolve_completed(lookup_started)
                        return None
                if _exact_image_digest(verify_image) != captured.exact_hash:
                    self._record_title_resolve_completed(lookup_started)
                    return None
            finally:
                if verify_image.resolve() == verify_path.resolve():
                    _safe_unlink(verify_image)
        resolved_at = utc_now_iso()
        elapsed_ms = int((time.monotonic() - lookup_started) * 1000)
        self._record_title_resolve_completed(lookup_started)
        first_changed_at = self._candidate_started_at or captured.capture_started_at
        switch_detection_bound_ms = int(
            max(0.0, captured.capture_started_monotonic - self._candidate_started_monotonic) * 1000
        ) if self._candidate_started_monotonic else 0
        return TitleOcrResult(
            job_id=-captured.sequence,
            hwnd=state.window.hwnd,
            target_app=state.window.target.value,
            app_label=state.window.app_label,
            snapshot_hash=captured.exact_hash[:32],
            image_path=captured.image_path,
            title_ocr_image_path=captured.image_path,
            title_crop_rect=captured.capture_rect,
            layout=captured.title_layout,
            ocr_result=title_result,
            created_at=resolved_at,
            elapsed_ms=elapsed_ms,
            process_id=state.window.process_id,
            window_generation=state.window_generation,
            capture_generation=state.capture_generation,
            account_scope_id=anchor.account_scope_id,
            account_scope_name=anchor.account_scope_name,
            target_instance_id=anchor.target_instance_id,
            account_scope_reason="anchored_same_window_instance",
            account_identity_epoch=state.account_identity_epoch,
            account_identity_runtime_key=state.account_identity_runtime_key,
            title_visual_hash=captured.visual_hash,
            title_exact_hash=captured.exact_hash,
            title_source=source,
            performance_trace={
                "contact_switch_first_seen_at": first_changed_at,
                "contact_switch_detected_at": first_changed_at,
                "contact_switch_confirmed_at": captured.capture_started_at,
                "switch_detection_bound_ms": switch_detection_bound_ms,
                "title_capture_started_at": captured.capture_started_at,
                "title_resolved_at": resolved_at,
                "title_capture_elapsed_ms": captured.capture_elapsed_ms,
                "title_ocr_elapsed_ms": elapsed_ms,
                "title_source": source,
                "page_was_chat_dm": bool(
                    state.page.page_type == PageType.CHAT_DM or trusted_unknown_page
                ),
                "page_evidence": (
                    "exact_dm_fingerprint" if trusted_unknown_page else "runtime_chat_dm"
                ),
                "title_confirmation_path": "independent_title_two_frame",
                "stable_frame_count": 2,
                "observed_sequence": time.monotonic_ns(),
            },
        )

    def _record_title_resolve_completed(self, started: float) -> None:
        with self._metrics_lock:
            self._ocr_completed += 1
            self._title_resolve_ms.append((time.monotonic() - started) * 1000.0)


def _exact_image_digest(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        payload = rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _context_signature(state: RuntimeState, anchor: TitleOcrResult) -> tuple[object, ...]:
    return (
        state.window.hwnd,
        state.window.process_id,
        state.window_generation,
        state.capture_generation,
        state.account_identity_epoch,
        state.account_identity_runtime_key,
        anchor.account_scope_id,
        anchor.target_instance_id,
        title_identity_screen_rect(anchor.layout).as_tuple(),
    )


def _identity_boundary_signature(state: RuntimeState, anchor: TitleOcrResult) -> tuple[object, ...]:
    # Capture/focus/ROI may advance within one account window. Such advances
    # must not erase evidence that its previously trusted DM became a group.
    return (
        state.window.hwnd, state.window.process_id, state.window_generation,
        state.account_identity_epoch, state.account_identity_runtime_key,
        anchor.account_scope_id, anchor.target_instance_id,
    )


def _anchor_geometry_is_current(state: RuntimeState, layout: LayoutRegions | None) -> bool:
    if layout is None or state.window.rect is None or layout.window_rect != state.window.rect:
        return False
    title = layout.title_rect
    window = state.window.rect
    if not (window.left <= title.left < title.right <= window.right
            and window.top <= title.top < title.bottom <= window.bottom):
        return False
    rect = title_identity_screen_rect(layout)
    return bool(rect.width > 0 and rect.height > 0
        and window.left <= rect.left < rect.right <= window.right
        and window.top <= rect.top < rect.bottom <= window.bottom)


def _unknown_anchor_key(state: RuntimeState, anchor: TitleOcrResult) -> tuple[object, ...]:
    return _context_signature(state, anchor) + (anchor.title_exact_hash,)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _raw_title_observation(result: OcrResult) -> str:
    boxes = sorted(
        (box for box in result.boxes if str(box.text or "").strip()),
        key=lambda box: (box.rect.top, box.rect.left),
    )
    text = " ".join(str(box.text).strip() for box in boxes)
    return unicodedata.normalize("NFKC", " ".join(text.split())).casefold()


def _has_group_title_evidence(text: str) -> bool:
    return bool(re.search(r"\s*[（(]\s*\d+\s*[）)]\s*$", text) or _looks_like_group_title(text))


def _normalized_title_observation(result: OcrResult) -> str:
    return re.sub(r"\s*[（(]\s*\d+\s*[）)]\s*$", "", _raw_title_observation(result)).strip()


def _intervals_ms(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((current - previous) * 1000.0 for previous, current in zip(values, values[1:]))


def _distribution(values: tuple[float, ...]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(value: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * value))))
        return round(float(ordered[index]), 3)

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": round(float(ordered[-1]), 3),
    }
