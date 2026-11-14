from __future__ import annotations

import os
import time
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from whochat.core.runtime import PageType, RuntimeState, WindowState
from whochat.platform.window_tracker import WindowInfo
from whochat.services.account_scope import AccountImageEvidenceUnavailable
from whochat.services.pipeline import CapturePipelineService
from whochat.services.runtime import RuntimeStateService


class AutoCaptureController(QObject):
    status_changed = Signal(str)

    def __init__(
        self,
        runtime: RuntimeStateService,
        pipeline: CapturePipelineService,
        *,
        enabled: bool = True,
        interval_ms: int = 5000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.pipeline = pipeline
        self.enabled = enabled
        self.interval_ms = max(1000, int(interval_ms))
        # The timer is only a scheduler tick.  The effective interval below is
        # adaptive: a single foreground target may be sampled faster, while a
        # background/unknown target keeps the configured conservative cadence.
        self._scheduler_tick_ms = min(1000, self.interval_ms)
        self._foreground_interval_ms = _env_interval("WHOCHAT_FOREGROUND_POLL_MS", 2000)
        self._pending_state: RuntimeState | None = None
        self._last_submit_job_id: int | None = None
        self._last_window: WindowInfo | None = None
        self._last_window_signature: tuple[object, ...] | None = None
        self._last_status: str = ""
        self._last_completed_ms = 0
        self._last_completed_context: tuple[object, ...] | None = None
        self._last_submitted_context: tuple[object, ...] | None = None
        self._priority_context: tuple[object, ...] | None = None
        self._priority_frames_remaining = 0
        self._context_refresh_pending = False
        self._priority_tokens: list[tuple[object, ...]] = []
        self._timer = QTimer(self)
        self._timer.setInterval(self._scheduler_tick_ms)
        self._timer.timeout.connect(self._run_cycle)
        # Prefer immutable job ownership. A stale old-window discard must not
        # start a cooldown for whichever customer happens to be visible now.
        result_discarded = getattr(self.pipeline, "result_discarded_detail", None)
        if result_discarded is None:
            result_discarded = getattr(self.pipeline, "result_discarded", None)
        if result_discarded is not None:
            result_discarded.connect(self._on_pipeline_discarded)
        result_ready = getattr(self.pipeline, "result_ready", None)
        if result_ready is not None:
            result_ready.connect(self._on_pipeline_completed)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self._run_cycle()

    def stop(self) -> None:
        self._timer.stop()
        self._pending_state = None

    @property
    def pending(self) -> bool:
        return self._pending_state is not None

    @property
    def last_submit_job_id(self) -> int | None:
        return self._last_submit_job_id

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self._pending_state = None
            self._last_window = None
            self._timer.stop()
            self._emit_status("auto_capture_disabled")
        else:
            self._timer.start(self._scheduler_tick_ms)
            self._emit_status("auto_capture_enabled")

    def on_window_changed(self, window: WindowInfo | None) -> None:
        signature = None if window is None else (
            window.hwnd, window.process_id, window.executable_path, window.target_app,
            window.rect, window.foreground, window.visible, window.minimized, window.covered
        )
        changed = signature != self._last_window_signature
        self._last_window_signature = signature
        self._last_window = window
        # Missing means the canonical native window is gone. It must reach the
        # identity authority even while automatic capture itself is disabled.
        if window is None or not self.enabled:
            self._pending_state = None
            self.runtime.update_from_window_info(window)
            return
        self._queue_current_window()
        if changed:
            self._emit_status("auto_capture_window_updated")

    def _run_cycle(self) -> None:
        if not self._queue_current_window():
            return
        self.flush_pending()

    def _queue_current_window(self) -> bool:
        if not self.enabled or self._last_window is None:
            self._pending_state = None
            return False
        state = self.runtime.update_from_window_info(self._last_window)
        if not state.capture_decision.should_capture:
            # CaptureGate evaluation is stateful: a second identical window
            # notification can report a short throttle immediately after the
            # first one queued a frame. Keep that pending frame; only durable
            # blocks (pause, hidden/covered window, invalid layout) clear it.
            transient = state.capture_decision.reason in {"截图节流中", "窗口区域刚变化，等待稳定"}
            if not transient:
                self._pending_state = None
            self._emit_status(f"auto_capture_blocked:{state.capture_decision.reason}")
            return False
        self._pending_state = state
        return True

    def flush_pending(self, *, identity_recovery: bool = False) -> int | None:
        if not self.enabled or self._pending_state is None:
            return None
        state = self._pending_state
        current = self.runtime.state
        if (
            self._capture_context(state) != self._capture_context(current)
            or current.paused
            or current.window.state != WindowState.VISIBLE
            or current.window.covered
            or current.layout is None
            or current.layout.confidence < 0.5
        ):
            self._pending_state = None
            return None
        if self.pipeline.is_running:
            return None
        minimum_interval_ms = self._effective_min_interval_ms()
        now_ms = int(time.monotonic() * 1000)
        context = self._cooldown_context(state)
        priority = (
            (identity_recovery or self._context_refresh_pending)
            and self._priority_context == self._capture_context(state)
            and self._priority_frames_remaining > 0
        )
        if (
            self._last_completed_context == context
            and self._last_completed_ms
            and now_ms - self._last_completed_ms < minimum_interval_ms
            and not priority
        ):
            return None
        self._pending_state = None
        self._last_submitted_context = context
        if priority:
            self._priority_frames_remaining -= 1
            self._context_refresh_pending = False
        job_id = self.pipeline.submit(state)
        self._last_submit_job_id = job_id
        if job_id is None:
            self._last_completed_ms = now_ms
            self._last_completed_context = context
            self._emit_status("auto_capture_submit_failed")
        return job_id

    def prioritize_identity_context(self, evidence, *, frame_budget: int = 1) -> bool:
        """Grant bounded recovery frames for accepted identity evidence only.

        Ordinary window polls, geometry/focus changes and new message text do
        not grant priority. Priority skips only the *completed-job* cooldown;
        CaptureGate, current lifecycle checks and pipeline-busy still apply.
        This schedules observations, never confirms a customer identity.
        A newly observed title gets one ordinary-scheduler refresh even if
        fast identity confirmation finishes before the recovery timer fires.
        """
        validator = getattr(self.runtime, "evidence_is_current", None)
        if not callable(validator) or not validator(evidence):
            return False
        state = self.runtime.state
        if not self.enabled or not state.window.foreground:
            return False
        accepted_result = getattr(self.runtime, "result_event_was_applied", None)
        is_result = callable(accepted_result) and accepted_result(evidence)
        is_switch = str(getattr(evidence, "reason", "")) == "fast_title_probe_changed_stable"
        if not (is_result or is_switch):
            return False
        context = self._capture_context(state)
        token = (context, getattr(evidence, "job_id", None), str(getattr(evidence, "snapshot_hash", "") or ""))
        if token in self._priority_tokens:
            return False
        self._priority_tokens.append(token)
        del self._priority_tokens[:-64]
        self._priority_context = context
        self._priority_frames_remaining = min(2, max(1, int(frame_budget)))
        self._context_refresh_pending = is_switch
        return True

    def request_identity_reacquire(self) -> int | None:
        """Queue one normal capture for identity recovery.

        The request still passes the ordinary foreground, visibility, layout,
        throttle and pipeline-busy gates. Accepted identity recovery may have
        a bounded priority budget; ordinary repeated calls cannot create one.
        """
        if not self.enabled or self._last_window is None:
            return None
        if not self._queue_current_window():
            return None
        return self.flush_pending(identity_recovery=True)

    def _on_pipeline_discarded(self, event: object) -> None:
        # Per-frame outcomes belong to the pipeline diagnostics. The controller
        # emits only durable state transitions (enabled/window/blocked), so a
        # long-running background session cannot create one DB row per poll.
        if not isinstance(event, str) and int(getattr(event, "job_id", 0) or 0) < (self._last_submit_job_id or 0):
            return
        self._last_completed_ms = int(time.monotonic() * 1000)
        self._last_completed_context = (
            self._evidence_context(event) if not isinstance(event, str)
            else self._last_submitted_context
        )

    def _on_pipeline_completed(self, result: object) -> None:
        if int(getattr(result, "job_id", 0) or 0) < (self._last_submit_job_id or 0):
            return
        self._last_completed_ms = int(time.monotonic() * 1000)
        self._last_completed_context = self._evidence_context(result)

    @staticmethod
    def _cooldown_context(state: RuntimeState) -> tuple[object, ...]:
        # capture_generation also advances on temporary focus/geometry changes;
        # including it here would let routine window notifications evade OCR
        # throttling. Actual window/account lifecycles remain separate.
        return (
            state.window.hwnd, state.window.process_id, state.window_generation,
            state.account_identity_epoch, state.account_identity_runtime_key,
        )

    @classmethod
    def _capture_context(cls, state: RuntimeState) -> tuple[object, ...]:
        return cls._cooldown_context(state) + (state.capture_generation,)

    @staticmethod
    def _evidence_context(evidence: object) -> tuple[object, ...]:
        return tuple(getattr(evidence, name, None) for name in (
            "hwnd", "process_id", "window_generation", "account_identity_epoch",
            "account_identity_runtime_key",
        ))

    def _effective_min_interval_ms(self) -> int:
        base = max(1000, int(getattr(self.runtime.capture_gate.policy, "ocr_min_interval_ms", self.interval_ms)))
        if getattr(getattr(self.pipeline, "ocr_engine", None), "name", "") != "paddleocr":
            foreground = bool(self._last_window and self._last_window.foreground and self._last_window.visible)
            return min(base, self._foreground_interval_ms) if foreground else base
        samples = getattr(self.pipeline, "capture_samples", None)
        if samples is None:
            return max(base, _env_interval("WHOCHAT_HEAVY_OCR_MIN_INTERVAL_MS", 5000))
        totals = [row.total_elapsed_ms for row in samples.tail(4) if row.total_elapsed_ms is not None]
        if not totals:
            return max(base, _env_interval("WHOCHAT_HEAVY_OCR_MIN_INTERVAL_MS", 5000))
        average = sum(totals) / len(totals)
        adaptive = int(average * 2)
        configured = (
            _env_interval("WHOCHAT_SLOW_OCR_MIN_INTERVAL_MS", 60000)
            if average > 45000
            else _env_interval("WHOCHAT_HEAVY_OCR_MIN_INTERVAL_MS", 5000)
        )
        return max(base, configured, adaptive)

    def _emit_status(self, status: str) -> None:
        if status == self._last_status:
            return
        self._last_status = status
        self.status_changed.emit(status)


class ContactIdentityRecoveryController(QObject):
    """Bounded recovery for a visible DM whose contact is still unknown.

    Recovery only schedules ordinary capture-pipeline work. Account scope,
    target instance, capture generations, title evidence and the two-frame
    identity lock remain mandatory in their existing owners.
    """

    status_changed = Signal(str)

    def __init__(
        self,
        runtime,
        pipeline,
        ingestion,
        autocapture,
        *,
        current_identity=None,
        logs=None,
        clock: Callable[[], float] = time.monotonic,
        poll_interval_ms: int = 1000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.pipeline = pipeline
        self.ingestion = ingestion
        self.autocapture = autocapture
        self.current_identity = current_identity
        self.logs = logs
        self._clock = clock
        self._unknown_key: tuple[object, ...] | None = None
        self._unknown_since = 0.0
        self._next_retry_at = 0.0
        self._attempt = 0
        self._last_result = None
        self._last_status = ""
        self._weak_recovery = False
        self._timer = QTimer(self)
        self._timer.setInterval(max(250, int(poll_interval_ms)))
        self._timer.timeout.connect(self.poll_once)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._clear()

    @property
    def armed(self) -> bool:
        return self._unknown_key is not None

    @property
    def attempt(self) -> int:
        return self._attempt

    def observe_pipeline_result(self, result) -> None:
        """Observe only a fully verified, current capture result."""
        if not self._result_is_current(result):
            return
        self._last_result = result
        if result.page.page_type != PageType.CHAT_DM:
            self._clear()
            return
        if self._identity_is_stable():
            self._mark_succeeded_if_armed()
            return
        key = self._result_key(result)
        if self._weak_recovery and (
            key != self._unknown_key
            or str(getattr(self.current_identity.snapshot, "state", "")) != "verifying"
        ):
            self._weak_recovery = False
        now = self._clock()
        if key != self._unknown_key:
            self._unknown_key = key
            self._unknown_since = now
            self._next_retry_at = now + self._initial_delay_seconds()
            self._attempt = 0
            self._prioritize_capture(result, frame_budget=1)
            self._audit("identity_recovery_armed", result=result)
            self._emit_status("identity_recovery_armed")

    def observe_possible_switch(self, event, *, was_active: bool = False) -> bool:
        """Arm new capture, never confirm from the retained customer/old DM."""
        if str(getattr(event, "reason", "")) != "fast_title_probe_changed_stable":
            return False
        validator = getattr(self.runtime, "evidence_is_current", None)
        if not callable(validator) or not validator(event):
            return False
        if not self._verification_anchor_is_current():
            return False
        state = self.runtime.state
        if not state.window.foreground or not self._state_allows_recovery(state):
            return False
        key = self._result_key(self._last_result)
        if self._unknown_key == key and not was_active:
            self._weak_recovery = True
            return True  # Preserve frame one, retry deadline and backoff.
        lock = getattr(self.ingestion, "identity_lock", None)
        if lock is not None and lock.stable_contact is not None:
            anchor = self._last_result
            if (
                lock._candidate_scope_id != anchor.account_scope_id
                or lock._candidate_target_instance_id != anchor.target_instance_id
                or lock._candidate_window_generation != anchor.window_generation
                or lock._candidate_account_identity_epoch != anchor.account_identity_epoch
            ):
                return False
            # Only the former stable lock is expired once per verification
            # cycle. Do NOT reset ingestion: that closes a database session
            # and drops buffered messages. An already-pending lock stays intact.
            lock.invalidate("title_verification_requires_fresh_frames")
        now = self._clock()
        self._unknown_key = key
        self._unknown_since = now
        self._next_retry_at = now + self._initial_delay_seconds()
        self._attempt = 0
        self._weak_recovery = True
        self._prioritize_capture(event, frame_budget=2)
        self._audit("identity_verification_recovery_armed", result=event, reason=event.reason)
        self._emit_status("identity_verification_recovery_armed")
        return True

    def _prioritize_capture(self, evidence, *, frame_budget: int) -> None:
        prioritize = getattr(self.autocapture, "prioritize_identity_context", None)
        if callable(prioritize):
            prioritize(evidence, frame_budget=frame_budget)

    def _verification_anchor_is_current(self) -> bool:
        """Old full-frame DM grants scheduling only, inside the trusted owner.

        A normal focus generation may have advanced. Never retag or publish
        that old frame: new captures still obey the latest capture generation.
        """
        if self.current_identity is None or self._last_result is None:
            return False
        state, anchor = self.runtime.state, self._last_result
        snapshot = self.current_identity.snapshot
        if (
            str(snapshot.state) != "verifying" or not snapshot.retained
            or state.page.page_type != PageType.CHAT_DM
            or anchor.page.page_type != PageType.CHAT_DM
            or not anchor.account_scope_id or not anchor.target_instance_id
            or anchor.capture_generation > state.capture_generation
            or anchor.account_identity_runtime_key != state.account_identity_runtime_key
        ):
            return False
        expected = (state.window.hwnd, state.window.process_id, state.window_generation,
                    state.account_identity_epoch, anchor.account_scope_id, anchor.target_instance_id)
        for evidence in (anchor, snapshot):
            if (evidence.hwnd, evidence.process_id, evidence.window_generation,
                evidence.account_identity_epoch, evidence.account_scope_id, evidence.target_instance_id) != expected:
                return False
        reader = getattr(self.runtime, "_account_checkpoint", None)
        checkpoint = reader(state.window) if callable(reader) else None
        return bool(checkpoint is not None and checkpoint.identity_state == "trusted"
            and checkpoint.identity_epoch == state.account_identity_epoch
            and checkpoint.account_scope_id == anchor.account_scope_id
            and checkpoint.target_instance_id == anchor.target_instance_id)

    def poll_once(self) -> None:
        now = self._clock()
        if self._expire_stuck_pipeline(now):
            return
        if self._identity_is_stable():
            self._mark_succeeded_if_armed()
            return
        if self._weak_recovery and not self._verification_anchor_is_current():
            self._clear()
            return
        if not self.armed or now < self._next_retry_at:
            return
        state = self.runtime.state
        if not self._state_allows_recovery(state):
            # Visibility, foreground, pause and layout gates remain authoritative.
            self._next_retry_at = now + 2.0
            return
        if self.pipeline.is_running:
            return
        evidence_gate = getattr(
            getattr(self.pipeline, "account_scope_resolver", None),
            "require_avatar_evidence_current", None,
        )
        if callable(evidence_gate):
            try:
                evidence_gate(state.window)
            except AccountImageEvidenceUnavailable:
                # Poll-based foreground=True may be stale. Wait without a new
                # job, a retry penalty or consuming the pending first frame.
                self._next_retry_at = now + 2.0
                self._emit_status("identity_recovery_waiting_foreground")
                return
        snapshot_hash = str(getattr(state, "last_snapshot_hash", "") or "")
        if not self.pipeline.prepare_identity_reacquire(
            snapshot_hash,
            reason=f"chat_dm_contact_unknown_attempt_{self._attempt + 1}",
        ):
            self._next_retry_at = now + 2.0
            return
        job_id = self.autocapture.request_identity_reacquire()
        if job_id is None:
            # The normal scheduler may still be inside a short throttle window.
            self._next_retry_at = now + 2.0
            return
        self._attempt += 1
        self._next_retry_at = now + self._retry_delay_seconds(self._attempt)
        self._audit("identity_recovery_retry", result=self._last_result, job_id=job_id)
        self._emit_status(f"identity_recovery_retry:job={job_id},attempt={self._attempt}")

    def _expire_stuck_pipeline(self, now: float) -> bool:
        if not self.pipeline.is_running or not self.pipeline.can_restart_after_timeout:
            return False
        stage = str(self.pipeline.running_stage or "unknown")
        elapsed = float(self.pipeline.running_elapsed_seconds)
        engine_name = str(getattr(getattr(self.pipeline, "ocr_engine", None), "name", "") or "")
        title_stage = stage in {"title_input", "title_ocr"}
        if engine_name == "paddleocr":
            timeout = self._env_seconds(
                "WHOCHAT_PADDLE_TITLE_WATCHDOG_SECONDS" if title_stage else "WHOCHAT_PADDLE_PIPELINE_WATCHDOG_SECONDS",
                110.0 if title_stage else 210.0,
            )
        else:
            timeout = self._env_seconds(
                "WHOCHAT_TITLE_WATCHDOG_SECONDS" if title_stage else "WHOCHAT_PIPELINE_WATCHDOG_SECONDS",
                20.0 if title_stage else 60.0,
            )
        if elapsed < timeout:
            return False
        job_id = self.pipeline.running_job_id
        reason = f"stage={stage},elapsed={elapsed:.1f}s,limit={timeout:.1f}s"
        if not self.pipeline.expire_running_job(reason):
            return False
        self._next_retry_at = now + 1.0
        self._audit("identity_recovery_watchdog", result=self._last_result, job_id=job_id, reason=reason)
        self._emit_status(f"identity_recovery_watchdog:job={job_id},stage={stage}")
        return True

    def _state_allows_recovery(self, state) -> bool:
        if not bool(getattr(self.autocapture, "enabled", False)):
            return False
        if state.paused or state.page.page_type != PageType.CHAT_DM:
            return False
        window = state.window
        if window.state != WindowState.VISIBLE or window.covered:
            return False
        if state.layout is None or state.layout.confidence < 0.5:
            return False
        # A stored throttle/stability decision is transient; auto-capture
        # re-evaluates it with current monotonic time. Durable blocks above stay.
        decision = state.capture_decision
        return bool(decision.should_capture) or decision.reason in {
            "截图节流中",
            "窗口区域刚变化，等待稳定",
        }

    def _identity_is_stable(self) -> bool:
        if self.current_identity is not None:
            snapshot = self.current_identity.snapshot
            state = self.runtime.state
            validator = getattr(self.runtime, "evidence_is_current", None)
            return bool(
                snapshot.active
                and self.current_identity.current_contact() is not None
                and snapshot.hwnd == state.window.hwnd
                and snapshot.process_id == state.window.process_id
                and snapshot.window_generation == state.window_generation
                and snapshot.account_identity_epoch == state.account_identity_epoch
                and (not callable(validator) or validator(snapshot))
            )
        result = getattr(self.ingestion, "last_result", None)
        return bool(
            result is not None
            and result.identity_stable
            and result.contact is not None
            and result.account_scope_id
            and result.target_instance_id
        )

    def _result_is_current(self, result) -> bool:
        state = self.runtime.state
        return (
            result.hwnd == state.window.hwnd
            and result.process_id == state.window.process_id
            and result.window_generation == state.window_generation
            and result.capture_generation == state.capture_generation
            and int(getattr(result, "account_identity_epoch", 0)) == state.account_identity_epoch
            and str(getattr(result, "account_identity_runtime_key", "") or state.account_identity_runtime_key)
            == state.account_identity_runtime_key
        )

    @staticmethod
    def _result_key(result) -> tuple[object, ...]:
        return (
            result.hwnd,
            result.process_id,
            result.window_generation,
            result.target_instance_id,
            getattr(result, "account_identity_epoch", 0),
        )

    @staticmethod
    def _initial_delay_seconds() -> float:
        return ContactIdentityRecoveryController._env_seconds(
            "WHOCHAT_IDENTITY_RECOVERY_INITIAL_SECONDS", 4.0
        )

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        maximum = ContactIdentityRecoveryController._env_seconds(
            "WHOCHAT_IDENTITY_RECOVERY_MAX_SECONDS", 60.0
        )
        return min(maximum, 4.0 * (2 ** min(max(0, attempt - 1), 4)))

    @staticmethod
    def _env_seconds(name: str, default: float) -> float:
        try:
            return max(0.25, float(os.environ.get(name, str(default))))
        except ValueError:
            return default

    def _mark_succeeded_if_armed(self) -> None:
        if self.armed:
            self._audit("identity_recovery_succeeded", result=self._last_result)
            self._emit_status("identity_recovery_succeeded")
        self._clear()

    def _clear(self) -> None:
        self._unknown_key = None
        self._unknown_since = 0.0
        self._next_retry_at = 0.0
        self._attempt = 0
        self._weak_recovery = False

    def _emit_status(self, status: str) -> None:
        if status == self._last_status:
            return
        self._last_status = status
        self.status_changed.emit(status)

    def _audit(self, event: str, *, result=None, job_id=None, reason: str = "") -> None:
        if self.logs is None:
            return
        context = {
            "job_id": job_id if job_id is not None else getattr(result, "job_id", None),
            "window_generation": getattr(result, "window_generation", None),
            "capture_generation": getattr(result, "capture_generation", None),
            "target_instance_id": getattr(result, "target_instance_id", ""),
            "account_scope_id": getattr(result, "account_scope_id", None),
            "attempt": self._attempt,
            "reason": reason,
        }
        try:
            self.logs.append("info", "identity_recovery", event, event, context)
        except Exception:
            pass


def _env_interval(name: str, default: int) -> int:
    try:
        return max(1000, int(os.environ.get(name, str(default))))
    except ValueError:
        return default
