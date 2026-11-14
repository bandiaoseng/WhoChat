from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import RLock
from time import monotonic

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Signal

from whochat.ai.models import ReplyContext, ReplyGenerationResult
from whochat.config import AppConfig
from whochat.core.models import utc_now_iso
from whochat.services.reply import ReplyGenerationService, context_hash


@dataclass(frozen=True)
class ReplyTaskResult:
    job_id: int
    result: ReplyGenerationResult
    contact_id: str | None = None
    hwnd: int | None = None
    window_title: str = ""
    context_digest: str = ""
    queued_at: str = ""
    worker_started_at: str = ""
    worker_completed_at: str = ""
    current_view_revision: str = ""
    current_work_started_monotonic: float | None = None
    current_work_deadline_monotonic: float | None = None


@dataclass(frozen=True)
class ReplyTaskState:
    """A lifecycle event belongs to one frozen request, never the current UI.

    Consumers must match job_id and the context binding before changing a
    waiting indicator.  An obsolete request can finish after the next one has
    started; its event is not an instruction to clear that newer request.
    """

    job_id: int
    status: str
    terminal: bool
    reason: str = ""
    contact_id: str | None = None
    account_scope_id: str = ""
    conversation_id: str = ""
    target_instance_id: str = ""
    hwnd: int | None = None
    window_title: str = ""
    window_generation: int = 0
    capture_generation: int = 0
    account_identity_epoch: int = 0
    account_identity_runtime_key: str = ""
    context_digest: str = ""
    current_view_revision: str = ""
    result_available: bool = False


@dataclass
class _ReplyTask:
    job_id: int
    context: ReplyContext
    config: AppConfig
    queued_at: str
    binding: dict[str, object]
    deadline_at: float
    completion: Future = field(default_factory=Future)
    terminal_status: str = ""
    dispatch_not_before: float = 0.0


class ReplyTaskService(QObject):
    result_ready = Signal(object)
    result_discarded = Signal(str)
    status_changed = Signal(str)
    task_state_changed = Signal(object)

    def __init__(
        self,
        generator: ReplyGenerationService,
        executor: ThreadPoolExecutor | None = None,
        parent: QObject | None = None,
        *,
        task_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.generator = generator
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="whochat-reply")
        self._lock = RLock()
        self._job_id = 0
        self._latest_job_id = 0
        self._running: Future | None = None
        self._active: _ReplyTask | None = None
        self._pending: _ReplyTask | None = None
        self._closed = False
        if task_timeout_seconds is not None and (
            not isfinite(task_timeout_seconds) or task_timeout_seconds <= 0
        ):
            raise ValueError("task_timeout_seconds_must_be_finite_and_positive")
        self._task_timeout_seconds = task_timeout_seconds
        self._clock = monotonic
        # One owner-thread timer, not one thread/timer per request.  It only
        # withdraws display/queue rights; it cannot terminate provider I/O.
        self._deadline_timer = QTimer(self)
        self._deadline_timer.setInterval(250)
        self._deadline_timer.timeout.connect(self.check_timeouts)
        if QCoreApplication.instance() is not None:
            self._deadline_timer.start()
        self.last_result: ReplyTaskResult | None = None
        self.last_state: ReplyTaskState | None = None
        self.last_status = "idle"
        self.last_discard_reason: str | None = None

    @property
    def is_running(self) -> bool:
        # A discarded HTTP request can still occupy the only worker.  Keeping
        # it visible here prevents background work from treating it as idle.
        with self._lock:
            return self._active is not None or self._pending is not None

    @property
    def pending_job_id(self) -> int | None:
        with self._lock:
            return self._pending.job_id if self._pending is not None else None

    @property
    def active_job_id(self) -> int | None:
        with self._lock:
            return self._active.job_id if self._active is not None else None

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self.invalidate("shutdown")
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self.executor.shutdown(wait=False)
            if QThread.currentThread() == self.thread():
                self._deadline_timer.stop()
            self._set_status("reply_shutdown")

    def invalidate(self, reason: str) -> None:
        """Withdraw display rights, not an already executing HTTP request.

        Future.cancel() can only cancel work that has not started.  No request
        counter, cloud budget, or provider health value is changed here.
        """

        with self._lock:
            invalidation_token = self._next_token()
            active, active_future = self._active, self._running
            self.last_result = None
            self.last_state = None
            pending, self._pending = self._pending, None
            if pending is not None:
                self._finish(pending, "cancelled", "context_invalidated")
            if active is not None:
                self._finish(active, "cancelled", "context_invalidated")
            # A successful cancellation may invoke _on_done immediately.
            # A terminal direct slot may already have submitted newer work;
            # never cancel that newer future or overwrite its waiting status.
            if active_future is not None and not active_future.done():
                active_future.cancel()
            if self._latest_job_id == invalidation_token:
                self._set_status(f"reply_invalidated:{reason}")

    def submit(self, context: ReplyContext, config: AppConfig) -> int | None:
        task = self._submit(context, config)
        return task.job_id if task is not None else None

    def _submit(self, context: ReplyContext, config: AppConfig) -> _ReplyTask | None:
        with self._lock:
            if self._closed:
                self._discard("reply_shutdown")
                return None
            # Freeze both lists in ReplyContext and mutable settings.  Pending
            # work must not silently adopt a later customer's mutable input.
            try:
                frozen_context = deepcopy(context)
                frozen_config = deepcopy(config)
                frozen_context, deadline_at, deadline_error = self._context_deadline(frozen_context, frozen_config)
                binding = _context_binding(frozen_context)
            except Exception as exc:
                self._discard(f"reply_failed:freeze:{type(exc).__name__}")
                self._set_status("reply_failed:input_snapshot")
                return None
            previous_pending = self._pending
            task = _ReplyTask(
                job_id=self._next_token(),
                context=frozen_context,
                config=frozen_config,
                queued_at=utc_now_iso(),
                binding=binding,
                deadline_at=deadline_at,
            )
            self.last_result = None
            active = self._active
            # Install the replacement before emitting any old terminal event.
            # A direct Qt slot may itself submit an even newer request.
            self._pending = task
            if previous_pending is not None:
                self._finish(previous_pending, "superseded", "newer_request")
            if active is not None:
                self._finish(active, "superseded", "newer_request")
            if task.job_id != self._latest_job_id or task.terminal_status:
                self._finish(task, "superseded", "newer_request")
                return task
            if deadline_error:
                if self._pending is task:
                    self._pending = None
                self._finish(task, "blocked", deadline_error)
                return task
            if self._active is not None:
                # This is the sole queue slot.  No obsolete pending request is
                # ever submitted into ThreadPoolExecutor's unbounded queue.
                self._publish(task, "waiting_latest", False, "previous_worker_finishing")
            else:
                if self._pending is task:
                    self._pending = None
                self._start(task)
            return task

    def run_sync(self, context: ReplyContext, config: AppConfig) -> ReplyTaskResult | None:
        task = self._submit(context, config)
        if task is None:
            return None
        # In particular, a queued latest request must not wait on _running,
        # which still belongs to its superseded predecessor.
        wall_deadline = monotonic() + max(0, task.deadline_at - self._clock())
        while not task.completion.done():
            # A synchronous caller may have no Qt event loop.  Advance only
            # the existing gate/queue, never a second worker or HTTP retry.
            self.check_timeouts()
            remaining = wall_deadline - monotonic()
            if remaining <= 0:
                with self._lock:
                    self._expire_task(task)
                return None
            try:
                return task.completion.result(timeout=min(.25, remaining))
            except TimeoutError:
                continue
        return task.completion.result()

    def _deadline_duration(self, config: AppConfig) -> float:
        timeout = float(config.ai.timeout_seconds)
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid_configured_task_deadline")
        return min(30.0, timeout, self._task_timeout_seconds or 30.0)

    def _context_deadline(self, context: ReplyContext, config: AppConfig) -> tuple[ReplyContext, float, str]:
        """One recognition attempt owns queue, cooldown, I/O and callback time.

        Legacy callers with neither field get one bounded submission clock.
        A malformed supplied clock never silently receives a fresh 30 seconds.
        These process-local fields are deliberately absent from business hashes.
        """
        now = self._clock()
        started = getattr(context, "current_work_started_monotonic", None)
        deadline = getattr(context, "current_work_deadline_monotonic", None)
        duration = self._deadline_duration(config)
        if started is None and deadline is None:
            started, deadline = now, now + duration
        elif (
            type(started) not in {int, float} or type(deadline) not in {int, float}
            or not isfinite(started) or not isfinite(deadline)
            or started < 0 or started > now or deadline <= started
            or deadline - started > 30.0 + 1e-9
        ):
            return context, now, "invalid_current_work_deadline"
        deadline = min(deadline, started + duration)
        return replace(context, current_work_started_monotonic=float(started),
                       current_work_deadline_monotonic=float(deadline)), deadline, ""

    def check_timeouts(self) -> None:
        with self._lock:
            if self._closed:
                return
            now = self._clock()
            # Expire pending before active: an old active callback must never
            # accidentally start a pending request whose deadline has passed.
            for task in (self._pending, self._active):
                if task is not None and not task.terminal_status and now >= task.deadline_at:
                    self._expire_task(task)
            self._drain_pending()

    def _expire_task(self, task: _ReplyTask) -> None:
        if task.terminal_status:
            return
        if self._pending is task:
            self._pending = None
        if task.job_id == self._latest_job_id:
            self.last_result = None
        self._finish(task, "timed_out", "task_deadline_exceeded")
        if self._active is task and self._running is not None:
            self._running.cancel()

    def _next_token(self) -> int:
        # invalidate and submit share one counter: an invalidation must never
        # accidentally give a later task the same token as an old task.
        self._job_id = max(self._job_id, self._latest_job_id) + 1
        self._latest_job_id = self._job_id
        return self._job_id

    def _start(self, task: _ReplyTask) -> None:
        if self._clock() >= task.deadline_at:
            self._expire_task(task)
            return
        try:
            gate = getattr(self.generator, "dispatch_delay_seconds", None)
            delay = float(gate(task.context, task.config)) if callable(gate) else 0.0
            if not isfinite(delay) or delay < 0:
                raise ValueError("invalid_pre_dispatch_delay")
        except Exception as exc:
            state, reason = _exception_state(exc)
            self._finish(task, state, reason)
            self._discard("reply_failed:dispatch_gate:" + type(exc).__name__)
            return
        if task.job_id != self._latest_job_id or task.terminal_status or self._closed:
            self._finish(task, "superseded", "newer_request")
            return
        if self._clock() >= task.deadline_at:
            self._expire_task(task)
            return
        if delay > 0:
            task.dispatch_not_before = self._clock() + delay
            self._pending = task
            if self.last_state is None or self.last_state.job_id != task.job_id or self.last_state.status != "waiting_rate_limit":
                self._publish(task, "waiting_rate_limit", False, "pre_dispatch_cooldown")
            return
        task.dispatch_not_before = 0.0
        self._active = task
        self._publish(task, "running", False)
        # Signals can run direct slots.  Such a slot may invalidate or replace
        # the request before an executor submission has actually happened.
        if self._closed or task.job_id != self._latest_job_id or task.terminal_status:
            self._active = None
            self._drain_pending()
            return
        try:
            future = self.executor.submit(self._run_task, task)
        except Exception as exc:
            self._active = None
            self._running = None
            state, reason = _exception_state(exc)
            self._finish(task, state, reason)
            self._discard(f"reply_failed:{type(exc).__name__}")
            self._drain_pending()
            return
        self._running = future
        future.add_done_callback(lambda finished: self._on_done(finished, task))

    def _run_task(self, task: _ReplyTask) -> ReplyTaskResult:
        with self._lock:
            if self._clock() >= task.deadline_at:
                self._expire_task(task)
            if self._closed or task.job_id != self._latest_job_id or task.terminal_status:
                raise CancelledError()
        return self._run_job(task.job_id, task.context, task.config, task.queued_at)

    def _run_job(self, job_id: int, context: ReplyContext, config: AppConfig, queued_at: str) -> ReplyTaskResult:
        worker_started_at = utc_now_iso()
        # The digest is captured before calling a collaborator, not after it
        # has had the opportunity to mutate the input objects.
        digest = context_hash(context)
        revision = _view_revision(context)
        result = self.generator.generate(context, config)
        return ReplyTaskResult(
            job_id=job_id,
            result=result,
            contact_id=context.contact.id if context.contact else None,
            hwnd=context.runtime.window.hwnd,
            window_title=context.runtime.window.title,
            context_digest=digest,
            queued_at=queued_at,
            worker_started_at=worker_started_at,
            worker_completed_at=utc_now_iso(),
            current_view_revision=revision,
            current_work_started_monotonic=getattr(context, "current_work_started_monotonic", None),
            current_work_deadline_monotonic=getattr(context, "current_work_deadline_monotonic", None),
        )

    def _on_done(self, future: Future, task: _ReplyTask | None = None) -> None:
        with self._lock:
            if task is None:
                task = self._active if future is self._running else None
            if task is not None and self._active is task:
                self._active = None
                self._running = None
            try:
                result = future.result()
            except Exception as exc:
                if task is not None and task.job_id == self._latest_job_id and not task.terminal_status:
                    state, reason = _exception_state(exc)
                    self._finish(task, state, reason)
                # Safe exception type only: exception text can contain request
                # bodies, credentials, or customer information.
                self._discard(f"reply_failed:{type(exc).__name__}")
            else:
                if task is not None and not task.terminal_status and self._clock() >= task.deadline_at:
                    self._expire_task(task)
                if result.job_id != self._latest_job_id or self._closed or (task and task.terminal_status):
                    if task is not None:
                        self._finish(task, "superseded", "newer_request")
                    self._discard(f"reply_stale_result:{result.job_id}")
                else:
                    self.last_result = result
                    if task is not None:
                        state, reason = _result_state(result.result)
                        self._finish(task, state, reason, result)
                    else:
                        self._set_status(
                            f"reply_finished: job={result.job_id}, allowed={result.result.allowed}, status={result.result.status}"
                        )
                    # A direct status slot might have submitted a newer job.
                    if result.job_id == self._latest_job_id and not self._closed:
                        self.result_ready.emit(result)
            finally:
                self._drain_pending()

    def _drain_pending(self) -> None:
        if self._active is not None or self._pending is None or self._closed:
            return
        if self._clock() < self._pending.dispatch_not_before:
            return
        pending, self._pending = self._pending, None
        if pending.job_id != self._latest_job_id or pending.terminal_status:
            self._finish(pending, "superseded", "newer_request")
            return
        self._start(pending)

    def _finish(
        self,
        task: _ReplyTask,
        status: str,
        reason: str,
        result: ReplyTaskResult | None = None,
    ) -> None:
        if task.terminal_status:
            return
        task.terminal_status = status
        if not task.completion.done():
            task.completion.set_result(result)
        self._publish(task, status, True, reason, result_available=result is not None)

    def _publish(self, task: _ReplyTask, status: str, terminal: bool, reason: str = "", *, result_available: bool = False) -> None:
        state = ReplyTaskState(task.job_id, status, terminal, reason, **task.binding, result_available=result_available)
        # Obsolete terminal events are still observable by their owner for
        # cleanup, but must not replace the latest service/UI waiting status.
        if task.job_id == self._latest_job_id:
            self.last_state = state
            prefix = {"succeeded": "finished", "running": "started"}.get(status, status)
            self._set_status(f"reply_{prefix}: job={task.job_id}, reason={reason}")
        self.task_state_changed.emit(state)

    def _discard(self, reason: str) -> None:
        self.last_discard_reason = reason
        self.result_discarded.emit(reason)

    def _set_status(self, status: str) -> None:
        self.last_status = status
        self.status_changed.emit(status)


def _view_revision(context: ReplyContext) -> str:
    view = getattr(context, "current_message_view", None)
    return str(getattr(view, "content_revision", "") or "")


def _context_binding(context: ReplyContext) -> dict[str, object]:
    view = getattr(context, "current_message_view", None)
    authorization = getattr(context, "current_work_cloud_authorization", None)
    memory = context.memory_context
    last_message = context.messages[-1] if context.messages else None
    return {
        "contact_id": context.contact.id if context.contact else None,
        "account_scope_id": context.contact.account_scope_id if context.contact else "",
        "conversation_id": str(
            getattr(view, "conversation_id", "")
            or getattr(memory, "conversation_id", "")
            or getattr(last_message, "conversation_id", "")
            or ""
        ),
        "target_instance_id": str(
            getattr(view, "target_instance_id", "")
            or getattr(authorization, "target_instance_id", "")
            or ""
        ),
        "hwnd": context.runtime.window.hwnd,
        "window_title": context.runtime.window.title,
        "window_generation": context.runtime.window_generation,
        "capture_generation": context.runtime.capture_generation,
        "account_identity_epoch": context.runtime.account_identity_epoch,
        "account_identity_runtime_key": context.runtime.account_identity_runtime_key,
        "context_digest": context_hash(context),
        "current_view_revision": _view_revision(context),
    }


def _exception_state(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, CancelledError):
        return "cancelled", "not_started_cancelled"
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timed_out", "provider_timeout"
    # Classify only recognized marker words; never publish exception text.
    message = str(exc).lower()
    if any(marker in message for marker in ("budget", "quota", "额度")):
        return "blocked", "request_budget_blocked"
    if isinstance(exc, PermissionError) or any(
        marker in message for marker in ("unauthorized", "authorization", "permission")
    ):
        return "blocked", "permission_blocked"
    return "failed", "worker_failure:" + type(exc).__name__


def _result_state(result: ReplyGenerationResult) -> tuple[str, str]:
    if result.allowed and result.suggestions:
        return "succeeded", ""
    status = result.status.lower()
    if "timeout" in status or "超时" in status:
        return "timed_out", "provider_timeout"
    if status == "local_preview":
        return "blocked", "provider_unavailable"
    if status.startswith("blocked:") or any(marker in status for marker in ("budget", "quota", "额度")):
        return "blocked", "generation_blocked"
    return "failed", "empty_result" if result.allowed else "generation_failed"
