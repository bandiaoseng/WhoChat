from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QObject, Signal

from whochat.capture.policy import CaptureGate
from whochat.core.runtime import (
    CapturePolicy,
    PageClassification,
    RuntimeState,
    account_identity_runtime_key,
    missing_runtime_state,
)
from whochat.services.pipeline import PipelineResult
from whochat.platform.adapters import PlatformAdapter, WeChatAdapter
from whochat.platform.window_tracker import WindowInfo


class RuntimeStateService(QObject):
    state_changed = Signal(object)

    def __init__(
        self,
        adapter: PlatformAdapter | None = None,
        capture_policy: CapturePolicy | None = None,
        calibrations=None,
        logs=None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.adapter = adapter or WeChatAdapter(calibrations=calibrations)
        self.capture_gate = CaptureGate(capture_policy or CapturePolicy())
        self._state = missing_runtime_state()
        self._paused = False
        self._last_confirmed_page: PageClassification | None = None
        self._last_title_job_id = 0
        self._last_result_job_id = 0
        self._last_context_change_job_id = 0
        self._pending_ocr_job_id = 0
        self._state_generation = 0
        # AccountScopeResolver is the only generation authority. Runtime keeps
        # a consumer-side cache so switching A -> B -> A restores each
        # runtime's own epoch instead of comparing process-global integers.
        self._account_identity_epochs: dict[str, int] = {}
        # Bootstrap receives the same Qt signal after Runtime.  Remember the
        # exact account-boundary object that Runtime accepted so downstream
        # identity invalidation cannot run for a rejected/stale boundary.
        self._last_applied_account_boundary_event = None
        self._last_applied_capture_context_event = None
        self._last_applied_result_event = None
        self._account_identity_authority = None
        self._account_revalidation_event_owner = None
        self._account_revalidation_event_job = 0
        self._account_revalidation_event = None
        self._last_account_revalidation_progress = None
        self._logs = logs

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def state_generation(self) -> int:
        return self._state_generation

    def bind_account_identity_authority(self, authority) -> None:
        """Use a read-only current checkpoint, never a delayed OCR payload."""
        self._account_identity_authority = authority

    def _account_checkpoint(self, window):
        if self._account_identity_authority is None or window.hwnd is None:
            return None
        checkpoint = self._account_identity_authority.account_identity_checkpoint(window)
        if checkpoint is None:
            return None
        if (
            checkpoint.runtime_key != _account_runtime_key(window)
            or checkpoint.hwnd != window.hwnd
            or checkpoint.process_id != window.process_id
        ):
            return None
        return checkpoint

    def accept_capture_context_changed(self, event) -> bool:
        self.apply_capture_context_changed(event)
        return self.capture_context_event_was_applied(event)

    def capture_context_event_was_applied(self, event) -> bool:
        return self._last_applied_capture_context_event is event

    def accept_title_result(self, result) -> bool:
        self.apply_title_result(result)
        return self.result_event_was_applied(result)

    def accept_pipeline_result(self, result) -> bool:
        self.apply_pipeline_result(result)
        return self.result_event_was_applied(result)

    def result_event_was_applied(self, result) -> bool:
        return self._last_applied_result_event is result

    def evidence_is_current(self, evidence) -> bool:
        """Lifecycle gate for independent probes without normal-job ordering."""
        return not self._result_reject_reason(evidence)

    def account_revalidation_failure_is_current(self, result) -> bool:
        """Negative UI evidence only; never make an unbound frame admissible.

        Keep the ordinary positive gate unchanged. Its ONLY rejection may be
        the current resolver's frozen account, after all owner/generation/epoch
        checks passed. Delayed frames from another window/request cannot stop
        or replace this customer's work.
        """
        reason = str(getattr(result, "account_scope_reason", ""))
        if (getattr(result, "account_scope_id", None)
                or getattr(result, "target_instance_id", "")
                or reason not in {
                    "account_identity_revalidating:account_fingerprint_outside_trusted_band",
                    "account_identity_revalidating:account_fingerprint_version_mismatch",
                    "account_identity_revalidating:account_fingerprint_prefers_other_scope",
                    "account_identity_revalidating:account_fingerprint_exact_other_scope",
                    "account_identity_revalidating:native_account_evidence_unavailable",
                    "account_identity_revalidating:native_account_evidence_ambiguous",
                    "account_identity_revalidating:native_account_mapping_unverified",
                    "account_identity_revalidating:native_account_owner_changed",
                    "account_identity_revalidating:native_account_binding_conflict",
                    "account_identity_revalidating:native_account_avatar_unavailable",
                }):
            return False
        checkpoint = self._account_checkpoint(self._state.window)
        return bool(checkpoint is not None and checkpoint.identity_state == "revalidating"
                    and checkpoint.identity_epoch == getattr(result, "account_identity_epoch", None)
                    and self._result_reject_reason(result) == "account_identity_context_mismatch"
                    and result.job_id >= max(self._last_result_job_id, self._last_context_change_job_id))

    def capture_context_event_is_current(self, event) -> bool:
        return not self._result_reject_reason(event) and event.job_id >= max(
            self._last_result_job_id,
            self._last_context_change_job_id,
        )

    def _account_revalidation_event_key(self, result) -> tuple:
        return (result.hwnd, result.process_id, result.window_generation,
                result.capture_generation, result.account_identity_epoch,
                result.account_identity_runtime_key)

    def _account_revalidation_event_is_latest(self, result) -> bool:
        job = getattr(result, "job_id", None)
        if type(job) is not int or job <= 0:
            return False
        if self._account_revalidation_event_key(result) != self._account_revalidation_event_owner:
            return True
        return (job > self._account_revalidation_event_job
                or (job == self._account_revalidation_event_job
                    and result is self._account_revalidation_event))

    def _remember_account_revalidation_event(self, result) -> bool:
        if (not self._account_revalidation_event_is_latest(result)
                or result is self._account_revalidation_event):
            return False
        self._account_revalidation_event_owner = self._account_revalidation_event_key(result)
        self._account_revalidation_event_job = result.job_id
        self._account_revalidation_event = result
        self._last_account_revalidation_progress = None
        return True

    def account_revalidation_progress_is_current(self, result) -> bool:
        """Fresh recovery progress can reset a failure counter, never authorize."""
        if (getattr(result, "account_scope_id", None)
                or getattr(result, "target_instance_id", "")
                or getattr(result, "account_scope_reason", "") !=
                    "account_identity_revalidating:native_account_confirmation_pending"
                or not self._account_revalidation_event_is_latest(result)):
            return False
        try:
            captured = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
            if captured.tzinfo is None or not 0 <= (datetime.now(timezone.utc) - captured).total_seconds() <= 30:
                return False
        except (AttributeError, TypeError, ValueError):
            return False
        checkpoint = self._account_checkpoint(self._state.window)
        return bool(checkpoint is not None and checkpoint.identity_state == "revalidating"
                    and getattr(checkpoint, "native_recovery_pending", False) is True
                    and checkpoint.identity_epoch == getattr(result, "account_identity_epoch", None)
                    and self._result_reject_reason(result) == "account_identity_context_mismatch"
                    and result.job_id >= max(self._last_title_job_id, self._last_result_job_id,
                                             self._last_context_change_job_id))

    def accept_account_revalidation_progress(self, result) -> bool:
        if (not self.account_revalidation_progress_is_current(result)
                or not self._remember_account_revalidation_event(result)):
            return False
        self._last_account_revalidation_progress = result
        return True

    def account_revalidation_progress_was_applied(self, result) -> bool:
        return self._last_account_revalidation_progress is result

    def accept_account_revalidation_failure(self, result) -> bool:
        # Keep the existing negative predicate unchanged; only reject repeated
        # or out-of-order delivery on this same account/capture lifecycle.
        return (self.account_revalidation_failure_is_current(result)
                and self._remember_account_revalidation_event(result))

    def account_boundary_event_was_applied(self, event) -> bool:
        """Return whether this exact boundary was adopted by Runtime.

        Qt direct signal connections preserve the event object while invoking
        Runtime and then Bootstrap.  Object identity makes a rejected event
        impossible to confuse with an older accepted event that happened to
        carry the same generation values.
        """

        return self._last_applied_account_boundary_event is event

    def set_paused(self, paused: bool) -> RuntimeState:
        self._paused = paused
        return self.refresh_current()

    def refresh_current(self) -> RuntimeState:
        return self.update_from_window_info(None if self._state.window.hwnd is None else _window_info_from_state(self._state))

    def update_from_window_info(self, window: WindowInfo | None) -> RuntimeState:
        window_snapshot = self.adapter.window_snapshot(window)
        if window_snapshot.hwnd is None and self._state.window.hwnd is not None:
            # The follower reserves None for a missing canonical window;
            # ordinary focus, coverage and minimization retain its WindowInfo.
            # Retire the old account lifecycle before a reused PID/HWND could
            # obtain the old trusted checkpoint without fresh avatar proof.
            previous_checkpoint = self._account_checkpoint(self._state.window)
            closer = getattr(self._account_identity_authority, "mark_runtime_closed", None)
            if previous_checkpoint is not None and callable(closer):
                closer(self._state.window, expected_identity_epoch=previous_checkpoint.identity_epoch)
                closed_checkpoint = self._account_checkpoint(self._state.window)
                if closed_checkpoint is not None:
                    self._account_identity_epochs[closed_checkpoint.runtime_key] = max(
                        closed_checkpoint.identity_epoch,
                        self._account_identity_epochs.get(closed_checkpoint.runtime_key, 0),
                    )
        target_changed = _window_identity(window_snapshot) != _window_identity(self._state.window)
        current_account_runtime_key = self._state.account_identity_runtime_key
        if current_account_runtime_key:
            self._account_identity_epochs[current_account_runtime_key] = max(
                self._state.account_identity_epoch,
                self._account_identity_epochs.get(current_account_runtime_key, 0),
            )
        next_account_runtime_key = _account_runtime_key(window_snapshot)
        next_account_epoch = (
            self._account_identity_epochs.get(next_account_runtime_key, 0)
            if target_changed
            else self._state.account_identity_epoch
        )
        checkpoint = self._account_checkpoint(window_snapshot)
        if checkpoint is not None:
            next_account_epoch = max(next_account_epoch, checkpoint.identity_epoch)
            self._account_identity_epochs[next_account_runtime_key] = next_account_epoch
        account_epoch_changed = (
            not target_changed and next_account_epoch != self._state.account_identity_epoch
        )
        if target_changed:
            self._last_applied_account_boundary_event = None
            self._last_confirmed_page = None
            self._last_title_job_id = 0
            self._last_result_job_id = 0
            self._last_context_change_job_id = 0
        if account_epoch_changed:
            self._last_confirmed_page = None
        if target_changed or account_epoch_changed:
            self._last_applied_capture_context_event = None
            self._last_applied_result_event = None
            self._last_applied_account_boundary_event = None
        capture_context_changed = (
            not target_changed
            and _capture_window_signature(window_snapshot) != _capture_window_signature(self._state.window)
        )
        capture_invalidated = target_changed or capture_context_changed or account_epoch_changed
        if capture_invalidated:
            self._pending_ocr_job_id = 0
        window_generation = self._state.window_generation + int(target_changed)
        # Foreground/coverage/geometry changes invalidate every in-flight
        # screenshot without pretending that the account window changed.  This
        # lets the UI retain the last confirmed contact while a WhoChat or
        # WeChat popup temporarily owns focus, but stale OCR can no longer
        # commit page/title/contact state afterward.
        capture_generation = self._state.capture_generation + int(capture_invalidated)
        if capture_context_changed:
            self._last_applied_capture_context_event = None
            self._last_applied_result_event = None
        layout = self.adapter.estimate_layout(window_snapshot)
        provisional_page = self.adapter.classify_page(window_snapshot, layout)
        # Window-only classification has no screenshot/OCR evidence and is
        # normally UNKNOWN for WeChat 4.x. It must not overwrite a newer page
        # committed by the capture pipeline for the same window generation.
        page = provisional_page
        if (
            not target_changed
            and provisional_page.page_type.value == "unknown"
            and self._last_confirmed_page is not None
        ):
            page = self._last_confirmed_page
        provisional = RuntimeState(
            window=window_snapshot,
            layout=layout,
            page=page,
            capture_decision=self._state.capture_decision,
            paused=self._paused,
            ai_pending=self._state.ai_pending,
            ocr_pending=False if capture_invalidated else self._state.ocr_pending,
            last_snapshot_hash=self._state.last_snapshot_hash,
            visible_message_count=self._state.visible_message_count,
            pipeline_status="idle" if capture_invalidated else self._state.pipeline_status,
            window_generation=window_generation,
            capture_generation=capture_generation,
            account_identity_epoch=next_account_epoch,
            account_identity_runtime_key=next_account_runtime_key,
        )
        decision = self.capture_gate.evaluate(provisional)
        next_state = RuntimeState(
            window=window_snapshot,
            layout=layout,
            page=page,
            capture_decision=decision,
            paused=self._paused,
            ai_pending=self._state.ai_pending,
            ocr_pending=False if capture_invalidated else self._state.ocr_pending,
            last_snapshot_hash=self._state.last_snapshot_hash,
            visible_message_count=self._state.visible_message_count,
            pipeline_status="idle" if capture_invalidated else self._state.pipeline_status,
            window_generation=window_generation,
            capture_generation=capture_generation,
            account_identity_epoch=next_account_epoch,
            account_identity_runtime_key=next_account_runtime_key,
        )
        self._commit(
            next_state,
            "window_target_changed" if target_changed else "account_epoch_reconciled" if account_epoch_changed else "window_poll",
        )
        return self._state

    def apply_pipeline_started(self, job_id: int | None = None) -> RuntimeState:
        if job_id is not None:
            self._pending_ocr_job_id = int(job_id)
        next_state = RuntimeState(
            window=self._state.window,
            layout=self._state.layout,
            page=self._state.page,
            capture_decision=self._state.capture_decision,
            paused=self._state.paused,
            ocr_pending=True,
            ai_pending=self._state.ai_pending,
            last_snapshot_hash=self._state.last_snapshot_hash,
            visible_message_count=self._state.visible_message_count,
            pipeline_status="running",
            window_generation=self._state.window_generation,
            capture_generation=self._state.capture_generation,
            account_identity_epoch=self._state.account_identity_epoch,
            account_identity_runtime_key=self._state.account_identity_runtime_key,
        )
        self._commit(next_state, "pipeline_started")
        return self._state

    def apply_pipeline_result(self, result: PipelineResult) -> RuntimeState:
        self._last_applied_result_event = None
        reject_reason = self._result_reject_reason(result)
        if reject_reason:
            self._audit_reject("pipeline_result", result, reject_reason)
            return self._state
        if result.job_id < max(self._last_title_job_id, self._last_result_job_id, self._last_context_change_job_id):
            self._audit_reject("pipeline_result", result, "older_job")
            return self._state
        self._last_result_job_id = result.job_id
        if not self._pending_ocr_job_id or result.job_id == self._pending_ocr_job_id:
            self._pending_ocr_job_id = 0
        self._last_confirmed_page = result.page
        next_state = RuntimeState(
            window=self._state.window,
            layout=self._state.layout,
            page=result.page,
            capture_decision=self._state.capture_decision,
            paused=self._state.paused,
            ocr_pending=False,
            ai_pending=self._state.ai_pending,
            last_snapshot_hash=result.snapshot_hash,
            visible_message_count=len(result.messages),
            pipeline_status=f"finished:{result.page.page_type.value}",
            window_generation=self._state.window_generation,
            capture_generation=self._state.capture_generation,
            account_identity_epoch=self._state.account_identity_epoch,
            account_identity_runtime_key=self._state.account_identity_runtime_key,
        )
        self._last_applied_result_event = result
        self._commit(next_state, "pipeline_result", result=result)
        return self._state

    def apply_title_result(self, result) -> RuntimeState:
        self._last_applied_result_event = None
        reject_reason = self._result_reject_reason(result)
        if reject_reason:
            self._audit_reject("title_result", result, reject_reason)
            return self._state
        if result.job_id < max(self._last_title_job_id, self._last_result_job_id, self._last_context_change_job_id):
            self._audit_reject("title_result", result, "older_job")
            return self._state
        self._last_title_job_id = result.job_id
        next_state = RuntimeState(
            window=self._state.window,
            layout=self._state.layout,
            page=self._state.page,
            capture_decision=self._state.capture_decision,
            paused=self._state.paused,
            ocr_pending=True,
            ai_pending=self._state.ai_pending,
            last_snapshot_hash=result.snapshot_hash,
            visible_message_count=self._state.visible_message_count,
            pipeline_status="title_ready",
            window_generation=self._state.window_generation,
            capture_generation=self._state.capture_generation,
            account_identity_epoch=self._state.account_identity_epoch,
            account_identity_runtime_key=self._state.account_identity_runtime_key,
        )
        self._last_applied_result_event = result
        self._commit(next_state, "title_result", result=result)
        return self._state

    def apply_capture_context_changed(self, event) -> RuntimeState:
        self._last_applied_capture_context_event = None
        account_boundary = str(getattr(event, "reason", "")).startswith("account_identity_revalidation:")
        if account_boundary:
            # A rejected boundary must never inherit the accepted marker from
            # the preceding event.
            self._last_applied_account_boundary_event = None
        reject_reason = self._result_reject_reason(event, allow_account_boundary=account_boundary)
        if reject_reason:
            self._audit_reject("capture_context_changed", event, reject_reason)
            return self._state
        if not account_boundary and not self.capture_context_event_is_current(event):
            self._audit_reject("capture_context_changed", event, "older_job")
            return self._state
        self._last_context_change_job_id = max(self._last_context_change_job_id, event.job_id)
        self._last_confirmed_page = None
        if account_boundary:
            self._pending_ocr_job_id = 0
        next_account_epoch = self._state.account_identity_epoch
        next_capture_generation = self._state.capture_generation
        if account_boundary:
            account_runtime_key = str(
                getattr(event, "account_identity_runtime_key", "")
                or self._state.account_identity_runtime_key
            )
            next_account_epoch = int(getattr(event, "account_identity_epoch", 0) or 0)
            previous_authority_epoch = self._account_identity_epochs.get(
                account_runtime_key,
                self._state.account_identity_epoch,
            )
            self._account_identity_epochs[account_runtime_key] = next_account_epoch
            next_capture_generation += int(next_account_epoch > previous_authority_epoch)
        unknown_page = PageClassification(
            self._state.page.page_type.__class__.UNKNOWN,
            0.0,
            f"capture_context_changed:{event.reason}",
        )
        next_state = RuntimeState(
            window=self._state.window,
            layout=self._state.layout,
            page=unknown_page,
            capture_decision=self._state.capture_decision,
            paused=self._state.paused,
            ocr_pending=not account_boundary,
            ai_pending=self._state.ai_pending,
            last_snapshot_hash=self._state.last_snapshot_hash,
            visible_message_count=0,
            pipeline_status=(
                f"account_identity_revalidating:{event.reason}"
                if account_boundary
                else f"identity_pending:{event.reason}"
            ),
            window_generation=self._state.window_generation,
            capture_generation=next_capture_generation,
            account_identity_epoch=next_account_epoch,
            account_identity_runtime_key=self._state.account_identity_runtime_key,
        )
        self._last_applied_capture_context_event = event
        if account_boundary:
            self._last_applied_account_boundary_event = event
        self._commit(next_state, "capture_context_changed", result=event)
        return self._state

    def apply_pipeline_discarded(self, event) -> RuntimeState:
        if isinstance(event, str):
            reason = event
            result = None
        else:
            reason = event.reason
            result = event
            reject_reason = self._result_reject_reason(event)
            if reject_reason:
                self._audit_reject("pipeline_discarded", event, reject_reason)
                return self._state
            if event.job_id < max(self._last_result_job_id, self._last_context_change_job_id):
                self._audit_reject("pipeline_discarded", event, "older_job")
                return self._state
            if self._pending_ocr_job_id and event.job_id != self._pending_ocr_job_id:
                self._audit_reject("pipeline_discarded", event, "non_pending_job")
                return self._state
            self._pending_ocr_job_id = 0
        # A discard carries no newer positive page evidence. Retain the last
        # committed state; in particular, never regress chat_dm to UNKNOWN.
        retained_page = self._last_confirmed_page or self._state.page
        next_state = RuntimeState(
            window=self._state.window,
            layout=self._state.layout,
            page=retained_page,
            capture_decision=self._state.capture_decision,
            paused=self._state.paused,
            ocr_pending=False,
            ai_pending=self._state.ai_pending,
            last_snapshot_hash=self._state.last_snapshot_hash,
            visible_message_count=self._state.visible_message_count,
            pipeline_status=f"discarded:{reason}",
            window_generation=self._state.window_generation,
            capture_generation=self._state.capture_generation,
            account_identity_epoch=self._state.account_identity_epoch,
            account_identity_runtime_key=self._state.account_identity_runtime_key,
        )
        self._commit(next_state, "pipeline_discarded", result=result, reject_reason=reason)
        return self._state

    def _result_reject_reason(self, result, *, allow_account_boundary: bool = False) -> str:
        if result.hwnd != self._state.window.hwnd:
            return "hwnd_mismatch"
        if result.process_id != self._state.window.process_id:
            return "process_mismatch"
        if result.window_generation != self._state.window_generation:
            return "window_generation_mismatch"
        result_runtime_key = str(
            getattr(result, "account_identity_runtime_key", "")
            or self._state.account_identity_runtime_key
        )
        if result_runtime_key != self._state.account_identity_runtime_key:
            return "account_identity_runtime_mismatch"
        result_epoch = int(getattr(result, "account_identity_epoch", self._state.account_identity_epoch) or 0)
        if allow_account_boundary:
            authority_epoch = self._account_identity_epochs.get(
                result_runtime_key,
                self._state.account_identity_epoch,
            )
            if result_epoch < max(authority_epoch, self._state.account_identity_epoch):
                return "account_identity_epoch_mismatch"
            if self._account_identity_authority is not None:
                checkpoint = self._account_checkpoint(self._state.window)
                if checkpoint is None or checkpoint.identity_epoch != result_epoch:
                    return "account_identity_authority_mismatch"
                if result_epoch == self._state.account_identity_epoch:
                    return "account_identity_boundary_already_adopted"
            # AccountScopeResolver is the authority for this event. A normal
            # focus/coverage transition may advance capture_generation after
            # the boundary was emitted but before Runtime receives it. The
            # exact runtime/PID/HWND/window-generation checks above keep this
            # exemption inside the current window lifecycle. OCR and all
            # other asynchronous results still take the strict check below.
            return ""
        if result.capture_generation != self._state.capture_generation:
            return "capture_generation_mismatch"
        if result_epoch != self._state.account_identity_epoch:
            return "account_identity_epoch_mismatch"
        checkpoint = self._account_checkpoint(self._state.window)
        if checkpoint is not None and hasattr(result, "account_scope_id"):
            if (
                checkpoint.identity_epoch != result_epoch
                or checkpoint.identity_state != "trusted"
                or str(getattr(result, "account_scope_id", "") or "") != checkpoint.account_scope_id
                or str(getattr(result, "target_instance_id", "") or "") != checkpoint.target_instance_id
            ):
                return "account_identity_context_mismatch"
        return ""

    def _commit(self, state: RuntimeState, source: str, *, result=None, reject_reason: str = "") -> None:
        meaningful = _state_signature(state) != _state_signature(self._state)
        self._state = state
        if meaningful or source != "window_poll":
            self._state_generation += 1
            routine_duplicate = source == "pipeline_discarded" and reject_reason.startswith("duplicate_snapshot:")
            # Scheduler starts and unchanged-snapshot discards are transient UI
            # activity, not durable state transitions. Persisting them every
            # capture interval would recreate the old app_logs growth problem.
            if source != "pipeline_started" and not routine_duplicate:
                self._audit(source, result=result, reject_reason=reject_reason)
        self.state_changed.emit(self._state)

    def _audit(self, source: str, *, result=None, reject_reason: str = "") -> None:
        if self._logs is None:
            return
        context = {
            "state_generation": self._state_generation,
            "source": source,
            "job_id": getattr(result, "job_id", None),
            "snapshot_id": str(getattr(result, "snapshot_hash", "") or "")[:16],
            "account_scope_id": getattr(result, "account_scope_id", None),
            "target_instance_id": getattr(result, "target_instance_id", ""),
            "window_instance": {
                "hwnd": self._state.window.hwnd,
                "process_id": self._state.window.process_id,
                "window_generation": self._state.window_generation,
                "capture_generation": self._state.capture_generation,
                "account_identity_epoch": self._state.account_identity_epoch,
                "account_identity_runtime_key": self._state.account_identity_runtime_key,
            },
            "page": self._state.page.page_type.value,
            "reject_reason": reject_reason,
        }
        try:
            self._logs.append("info", "runtime", "state_commit", source, context)
        except Exception:
            pass

    def _audit_reject(self, source: str, result, reason: str) -> None:
        if self._logs is None:
            return
        try:
            self._logs.append(
                "warning",
                "runtime",
                "state_rejected",
                source,
                {
                    "state_generation": self._state_generation,
                    "job_id": getattr(result, "job_id", None),
                    "snapshot_id": str(getattr(result, "snapshot_hash", "") or "")[:16],
                    "account_scope_id": getattr(result, "account_scope_id", None),
                    "target_instance_id": getattr(result, "target_instance_id", ""),
                    "reject_reason": reason,
                },
            )
        except Exception:
            pass


def _window_info_from_state(state: RuntimeState) -> WindowInfo | None:
    if state.window.hwnd is None or state.window.rect is None:
        return None
    return WindowInfo(
        hwnd=state.window.hwnd,
        title=state.window.title,
        process_name=state.window.process_name,
        rect=state.window.rect.as_tuple(),
        visible=True,
        target_app=state.window.target.value,
        app_label=state.window.app_label,
        diagnostic=state.window.diagnostic,
        foreground=state.window.foreground,
        bubble_profile=state.window.bubble_profile,
        covered=state.window.covered,
        process_id=state.window.process_id,
        executable_path=state.window.executable_path,
        client_rect=state.window.client_rect.as_tuple() if state.window.client_rect is not None else None,
        dpi_scale=getattr(state.window, "dpi_scale", 1.0),
    )


def _window_identity(window) -> tuple[object, ...]:
    return (
        window.target.value,
        window.hwnd,
        window.process_id,
        window.executable_path.casefold() if window.executable_path else "",
    )


def _account_runtime_key(window) -> str:
    if window.hwnd is None:
        return ""
    return account_identity_runtime_key(window)


def _capture_window_signature(window) -> tuple[object, ...]:
    return (
        window.rect.as_tuple() if window.rect is not None else None,
        window.client_rect.as_tuple() if window.client_rect is not None else None,
        getattr(window, "dpi_scale", 1.0),
        window.state.value,
        window.foreground,
        window.covered,
    )


def _state_signature(state: RuntimeState) -> tuple[object, ...]:
    return (
        _window_identity(state.window),
        state.window.state.value,
        state.page.page_type.value,
        round(state.page.confidence, 4),
        state.page.reason,
        state.capture_decision.should_capture,
        state.capture_decision.reason,
        state.ocr_pending,
        state.ai_pending,
        state.last_snapshot_hash,
        state.visible_message_count,
        state.pipeline_status,
        state.window_generation,
        state.capture_generation,
        state.account_identity_epoch,
        state.account_identity_runtime_key,
    )
