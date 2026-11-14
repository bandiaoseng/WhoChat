from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

from whochat.storage.repositories import LogRepository


@dataclass(frozen=True)
class UiDataTaskResult:
    job_id: int
    kind: str
    request_key: str
    value: Any
    query_ms: float


@dataclass(frozen=True)
class UiDataTaskFailure:
    job_id: int
    kind: str
    request_key: str
    error_type: str
    error_message: str
    query_ms: float


class UiDataTaskService(QObject):
    """Run local read models away from Qt's main thread.

    Results carry a request key so the UI can discard an old customer/filter
    result without ever rendering it into the new page state.
    """

    result_ready = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        logs: LogRepository | None = None,
        executor: ThreadPoolExecutor | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.logs = logs
        self.executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="whochat-ui-data")
        self._job_id = 0
        self._futures: set[Future] = set()

    def submit(self, kind: str, request_key: str, loader: Callable[[], Any]) -> int:
        self._job_id += 1
        job_id = self._job_id
        future = self.executor.submit(self._run, job_id, kind, request_key, loader)
        setattr(future, "_whochat_job_id", job_id)
        setattr(future, "_whochat_kind", kind)
        setattr(future, "_whochat_request_key", request_key)
        self._futures.add(future)
        future.add_done_callback(self._on_done)
        return job_id

    def shutdown(self) -> None:
        for future in tuple(self._futures):
            future.cancel()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)

    @staticmethod
    def _run(job_id: int, kind: str, request_key: str, loader: Callable[[], Any]) -> UiDataTaskResult:
        started = perf_counter()
        value = loader()
        return UiDataTaskResult(job_id, kind, request_key, value, (perf_counter() - started) * 1000.0)

    def _on_done(self, future: Future) -> None:
        self._futures.discard(future)
        try:
            result = future.result()
        except Exception as exc:
            kind = getattr(future, "_whochat_kind", "unknown")
            request_key = getattr(future, "_whochat_request_key", "")
            failure = UiDataTaskFailure(
                job_id=int(getattr(future, "_whochat_job_id", 0)),
                kind=kind,
                request_key=request_key,
                error_type=type(exc).__name__,
                error_message=str(exc),
                query_ms=0.0,
            )
            self.failed.emit(failure)
            return
        if result.query_ms >= 500.0 and self.logs is not None:
            try:
                self.logs.append(
                    "warning",
                    "ui_performance",
                    "slow_local_query",
                    f"{result.kind} local query took {result.query_ms:.1f}ms",
                    {"kind": result.kind, "query_ms": round(result.query_ms, 3), "request_key": result.request_key},
                )
            except Exception:
                pass
        self.result_ready.emit(result)
