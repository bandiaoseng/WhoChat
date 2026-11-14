from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class IdentityEventCoordinator(QObject):
    """Commit current evidence once, then notify downstream UI consumers.

    A Qt signal's preceding slot returning early does not stop later slots.
    Raw capture/probe signals therefore enter here, while the UI listens only
    to the accepted signals below. Stale evidence never reaches ingestion,
    the two-frame candidate, recovery, title anchors, or current-customer UI.
    """

    title_ready = Signal(object)
    result_ready = Signal(object)
    account_revalidation_failed = Signal(object)
    account_revalidation_progress = Signal(object)
    context_changed = Signal(object)
    fast_title_ready = Signal(object)
    fast_context_changed = Signal(object)

    def __init__(
        self, runtime, ingestion, current_identity, fast_probe, recovery,
        *, submit_identity, submit_context, logs=None, parent=None,
    ) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.ingestion = ingestion
        self.current_identity = current_identity
        self.fast_probe = fast_probe
        self.recovery = recovery
        self.submit_identity = submit_identity
        self.submit_context = submit_context
        self.logs = logs
        self._fast_lifecycle = None
        self._fast_context_sequence = 0
        self._fast_title_sequence = 0

    def on_title_ready(self, result) -> None:
        if not self.runtime.accept_title_result(result):
            return
        ingested = self.ingestion.ingest_title_result(result)
        if self._ingestion_rejected_stale(ingested):
            return
        self.submit_identity(
            self.current_identity, ingested, result, producer="TitlePipeline",
        )
        # The accepted full-title frame may seed frame two independently of
        # full message OCR. A rejected/expired frame cannot replace this anchor.
        self.fast_probe.seed_from_full_title(result)
        self.title_ready.emit(result)

    def on_result_ready(self, result) -> None:
        if not self.runtime.accept_pipeline_result(result):
            progress = getattr(self.runtime, "accept_account_revalidation_progress", None)
            if callable(progress) and progress(result):
                # Still unbound: no ingestion, title seed, identity proposal,
                # reply/AI submission or copy capability is produced here.
                self.account_revalidation_progress.emit(result)
                return
            negative = getattr(self.runtime, "accept_account_revalidation_failure", None)
            if not callable(negative):
                negative = getattr(self.runtime, "account_revalidation_failure_is_current", None)
            if callable(negative) and negative(result):
                # No ingestion, title seed, identity proposal or recovery.
                self.account_revalidation_failed.emit(result)
            return
        ingested = self.ingestion.ingest_pipeline_result(result)
        if self._ingestion_rejected_stale(ingested):
            return
        self.submit_identity(
            self.current_identity, ingested, result, producer="FullOcrPipeline",
        )
        self.recovery.observe_pipeline_result(result)
        self.result_ready.emit(result)

    def on_context_changed(self, event) -> None:
        if not self.runtime.accept_capture_context_changed(event):
            return
        if event.reason == "current_screen_body_changed":
            # Negative message evidence, not a changed customer/account. Keep
            # the stable identity and active session; only current text expires.
            self.ingestion.invalidate_current_screen("current_screen_body_changed")
            self.context_changed.emit(event)
            return
        if self._is_account_boundary(event):
            self.ingestion.reset_context(f"capture_context_changed:{event.reason}")
        elif event.reason not in {"title_pixels_changed", "fast_title_probe_changed_stable"}:
            self.ingestion.reset_context(f"capture_context_changed:{event.reason}")
        # Changed pixels suspend actions, but are not positive proof of a new
        # title. The lock itself restarts when a new trusted candidate arrives.
        self.submit_context(
            self.current_identity, event, producer="CapturePipelineService", runtime=self.runtime,
        )
        self.context_changed.emit(event)

    def on_fast_title_ready(self, result) -> None:
        # FastProbe has its own negative job ids and observed_sequence. They
        # cannot be compared with the full pipeline's monotonically positive
        # ids. Both producers still obey the same window/account/capture gates.
        if not self.runtime.evidence_is_current(result):
            return
        if not self._accept_fast_sequence(result, title=True):
            return
        self.fast_title_ready.emit(result)

    def on_fast_context_changed(self, event) -> None:
        if self._is_account_boundary(event):
            if not self.runtime.accept_capture_context_changed(event):
                return
            self.ingestion.reset_context(f"capture_context_changed:{event.reason}")
        elif not self.runtime.evidence_is_current(event):
            return
        elif not self._accept_fast_sequence(event, title=False):
            return
        was_active = self.current_identity.snapshot.active
        self.submit_context(
            self.current_identity, event, producer="FastActivationProbe", runtime=self.runtime,
        )
        # A suspended title is not positive identity evidence. Tell the bounded
        # recovery owner to request new ordinary frames: unchanged-screen dedup
        # may otherwise prevent another full result from ever arming recovery.
        if str(getattr(event, "reason", "")) == "fast_title_probe_changed_stable":
            notify = getattr(self.recovery, "observe_possible_switch", None)
            if callable(notify):
                notify(event, was_active=was_active)
        self.fast_context_changed.emit(event)

    def _accept_fast_sequence(self, evidence, *, title: bool) -> bool:
        """Do not let a queued older weak probe undo a newer safe title.

        Probe job ids are negative capture sequence numbers. Context precedes
        title for the same sequence, while a context arriving after that title
        is already obsolete. Keep only this lifecycle's two watermarks.
        """
        sequence = -int(getattr(evidence, "job_id", 0) or 0)
        if sequence <= 0:
            return False
        lifecycle = (
            evidence.hwnd, evidence.process_id, evidence.window_generation,
            evidence.capture_generation, evidence.account_identity_epoch,
            evidence.account_identity_runtime_key,
        )
        if lifecycle != self._fast_lifecycle:
            self._fast_lifecycle = lifecycle
            self._fast_context_sequence = self._fast_title_sequence = 0
        if sequence < max(self._fast_context_sequence, self._fast_title_sequence):
            return False
        if title:
            if sequence <= self._fast_title_sequence:
                return False
            self._fast_title_sequence = sequence
        else:
            if sequence <= max(self._fast_context_sequence, self._fast_title_sequence):
                return False
            self._fast_context_sequence = sequence
        return True

    @staticmethod
    def _is_account_boundary(event) -> bool:
        return str(getattr(event, "reason", "")).startswith("account_identity_revalidation:")

    @staticmethod
    def _ingestion_rejected_stale(result) -> bool:
        return str(getattr(result, "reason", "")).startswith(("stale_", "identity_stale_"))
