from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import threading
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from whochat.core.models import Contact, utc_now_iso
from whochat.core.runtime import WindowState


class CurrentIdentityState(StrEnum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    VERIFYING = "verifying"
    FAST_SAFE = "fast_safe"
    CONFIRMED = "confirmed"


_ACTIVE_STATES = frozenset({CurrentIdentityState.FAST_SAFE, CurrentIdentityState.CONFIRMED})
_STATE_RANK = {
    CurrentIdentityState.UNKNOWN: 0,
    CurrentIdentityState.PENDING: 1,
    CurrentIdentityState.VERIFYING: 1,
    CurrentIdentityState.FAST_SAFE: 2,
    CurrentIdentityState.CONFIRMED: 3,
}


@dataclass(frozen=True)
class CurrentIdentitySnapshot:
    state: CurrentIdentityState = CurrentIdentityState.UNKNOWN
    contact_id: str = ""
    normalized_title: str = ""
    account_scope_id: str = ""
    target_instance_id: str = ""
    hwnd: int | None = None
    process_id: int | None = None
    window_generation: int = 0
    capture_generation: int = 0
    account_identity_epoch: int = 0
    source_job_id: int | None = None
    observed_sequence: int = 0
    producer: str = ""
    reason: str = ""
    updated_at: str = ""

    @property
    def active(self) -> bool:
        return self.state in _ACTIVE_STATES and bool(self.contact_id)

    @property
    def retained(self) -> bool:
        """Whether a previously safe contact remains available for reacquire.

        VERIFYING deliberately is not ``active``: it may keep the profile on
        screen, but must not authorize copy/provider/reply actions until fresh
        positive identity evidence arrives.
        """

        return (self.active or self.state == CurrentIdentityState.VERIFYING) and bool(self.contact_id)

    @property
    def conversation_token(self) -> tuple[object, ...]:
        return (
            self.account_scope_id,
            self.target_instance_id,
            self.hwnd,
            self.process_id,
            self.window_generation,
            self.account_identity_epoch,
            self.normalized_title.casefold(),
            self.contact_id,
        )


@dataclass(frozen=True)
class IdentityTransition:
    old: CurrentIdentitySnapshot
    new: CurrentIdentitySnapshot
    changed: bool
    action: str
    producer: str
    reason: str


class CurrentConversationIdentityController(QObject):
    """Single authority for the customer currently shown by WhoChat.

    OCR, ingestion, fast resolution and UI code submit evidence here.  Missing
    evidence is not negative evidence, capture generations are anti-stale
    tokens rather than identity boundaries, and only explicit stable title or
    window/account boundaries can invalidate an active customer.
    """

    transitioned = Signal(object)

    def __init__(self, contacts, logs=None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.contacts = contacts
        self.logs = logs
        self._lock = threading.RLock()
        self._snapshot = CurrentIdentitySnapshot(updated_at=utc_now_iso())
        # Runtime owns the authoritative live window/capture boundary.  Keep
        # it separately from the conversation token: capture_generation is an
        # anti-stale job token, never evidence that the customer changed.
        self._runtime_window_missing = False
        self._runtime_hwnd: int | None = None
        self._runtime_process_id: int | None = None
        self._runtime_window_generation = 0
        self._runtime_capture_generation = 0
        self._runtime_account_identity_epoch = 0

    @property
    def snapshot(self) -> CurrentIdentitySnapshot:
        with self._lock:
            return self._snapshot

    def current_contact(self) -> Contact | None:
        snapshot = self.snapshot
        if not snapshot.active:
            return None
        contact = self.contacts.get(snapshot.contact_id)
        if contact is None or contact.account_scope_id != snapshot.account_scope_id:
            return None
        return contact

    def retained_contact(self) -> Contact | None:
        snapshot = self.snapshot
        if not snapshot.retained:
            return None
        contact = self.contacts.get(snapshot.contact_id)
        if contact is None or contact.account_scope_id != snapshot.account_scope_id:
            return None
        return contact

    def promote(
        self,
        contact: Contact,
        evidence: object,
        *,
        normalized_title: str = "",
        state: CurrentIdentityState = CurrentIdentityState.CONFIRMED,
        producer: str,
        reason: str,
    ) -> IdentityTransition:
        if state not in _ACTIVE_STATES:
            raise ValueError("active identity promotion requires fast_safe or confirmed")
        fields = _evidence_fields(evidence)
        account_scope_id = str(fields["account_scope_id"] or contact.account_scope_id)
        if not account_scope_id or account_scope_id != contact.account_scope_id:
            return self._reject(producer, "contact_account_scope_mismatch", evidence)
        if not fields["target_instance_id"] or fields["hwnd"] is None:
            return self._reject(producer, "identity_boundary_incomplete", evidence)
        title = " ".join((normalized_title or contact.display_name).strip().split())[:80]
        with self._lock:
            old = self._snapshot
            runtime_reject = self._runtime_reject_reason(fields)
            if runtime_reject:
                return self._reject(producer, runtime_reject, evidence)
            boundary_reason = self._boundary_conflict(old, fields)
            if boundary_reason and old.retained:
                return self._reject(producer, f"stale_{boundary_reason}", evidence)
            if (
                old.retained
                and fields["observed_sequence"]
                and old.observed_sequence
                and fields["observed_sequence"] < old.observed_sequence
            ):
                return self._reject(producer, "stale_observed_sequence", evidence)
            same_identity = bool(
                old.retained
                and old.contact_id == contact.id
                and old.account_scope_id == account_scope_id
                and old.target_instance_id == fields["target_instance_id"]
                and old.hwnd == fields["hwnd"]
                and old.process_id == fields["process_id"]
                and old.window_generation == fields["window_generation"]
                and (not old.normalized_title or old.normalized_title.casefold() == title.casefold())
            )
            effective_state = state
            if same_identity and _STATE_RANK[old.state] > _STATE_RANK[state]:
                effective_state = old.state
            new = CurrentIdentitySnapshot(
                state=effective_state,
                contact_id=contact.id,
                normalized_title=title,
                account_scope_id=account_scope_id,
                target_instance_id=str(fields["target_instance_id"]),
                hwnd=fields["hwnd"],
                process_id=fields["process_id"],
                window_generation=int(fields["window_generation"] or 0),
                capture_generation=int(fields["capture_generation"] or 0),
                account_identity_epoch=int(fields["account_identity_epoch"] or 0),
                source_job_id=fields["source_job_id"],
                observed_sequence=int(
                    fields["observed_sequence"] or (old.observed_sequence if same_identity else 0)
                ),
                producer=producer,
                reason=reason,
                updated_at=utc_now_iso(),
            )
            changed = old.conversation_token != new.conversation_token or old.state != new.state
            self._snapshot = new
        transition = IdentityTransition(old, new, changed, "promote" if changed else "retain", producer, reason)
        if changed:
            self._audit(transition, "identity_state_transition")
            self.transitioned.emit(transition)
        return transition

    def observe_pending(
        self,
        evidence: object,
        *,
        normalized_title: str = "",
        producer: str,
        reason: str,
    ) -> IdentityTransition:
        fields = _evidence_fields(evidence)
        runtime_reject = self._runtime_reject_reason(fields)
        if runtime_reject and producer != "RuntimeStateService":
            return self._reject(producer, runtime_reject, evidence)
        with self._lock:
            old = self._snapshot
        boundary_reason = self._boundary_conflict(old, fields)
        if old.retained:
            if boundary_reason:
                return self.invalidate(
                    evidence,
                    producer=producer,
                    reason=boundary_reason,
                    strong=True,
                )
            transition = IdentityTransition(old, old, False, "retain", producer, reason)
            self._audit(transition, "identity_observation_retained")
            return transition
        title = " ".join(normalized_title.strip().split())[:80]
        new = CurrentIdentitySnapshot(
            state=CurrentIdentityState.PENDING,
            normalized_title=title,
            account_scope_id=str(fields["account_scope_id"] or ""),
            target_instance_id=str(fields["target_instance_id"] or ""),
            hwnd=fields["hwnd"],
            process_id=fields["process_id"],
            window_generation=int(fields["window_generation"] or 0),
            capture_generation=int(fields["capture_generation"] or 0),
            account_identity_epoch=int(fields["account_identity_epoch"] or 0),
            source_job_id=fields["source_job_id"],
            observed_sequence=int(fields["observed_sequence"] or 0),
            producer=producer,
            reason=reason,
            updated_at=utc_now_iso(),
        )
        with self._lock:
            old = self._snapshot
            changed = old.state != new.state or old.conversation_token != new.conversation_token
            self._snapshot = new
        transition = IdentityTransition(old, new, changed, "pending", producer, reason)
        self._audit(transition, "identity_state_transition" if changed else "identity_observation_retained")
        if changed:
            self.transitioned.emit(transition)
        return transition

    def observe_possible_switch(
        self,
        evidence: object,
        *,
        producer: str,
        reason: str,
    ) -> IdentityTransition:
        """Suspend actions while retaining the last safe contact for reacquire.

        A two-frame ROI/fingerprint change proves only that the title surface
        changed. It does not prove which customer is current. The retained
        contact is therefore visible for continuity, while ``active`` becomes
        false until a resolver supplies positive, unique identity evidence.
        """

        fields = _evidence_fields(evidence)
        runtime_reject = self._runtime_reject_reason(fields)
        if runtime_reject and producer != "RuntimeStateService":
            return self._reject(producer, runtime_reject, evidence)
        with self._lock:
            old = self._snapshot
        boundary_reason = self._boundary_conflict(old, fields)
        if boundary_reason:
            return self.invalidate(
                evidence,
                producer=producer,
                reason=boundary_reason,
                strong=True,
            )
        if not old.retained:
            return self.observe_pending(evidence, producer=producer, reason=reason)
        new = replace(
            old,
            state=CurrentIdentityState.VERIFYING,
            capture_generation=int(fields["capture_generation"] or old.capture_generation),
            source_job_id=fields["source_job_id"] or old.source_job_id,
            observed_sequence=max(
                int(fields["observed_sequence"] or 0),
                int(old.observed_sequence or 0),
            ),
            producer=producer,
            reason=reason,
            updated_at=utc_now_iso(),
        )
        with self._lock:
            latest = self._snapshot
            if latest.conversation_token != old.conversation_token or latest.state != old.state:
                return self._reject(producer, "stale_possible_switch_observation", evidence)
            changed = latest.state != CurrentIdentityState.VERIFYING
            self._snapshot = new
        transition = IdentityTransition(latest, new, changed, "suspend" if changed else "retain", producer, reason)
        self._audit(transition, "identity_state_transition" if changed else "identity_observation_retained")
        if changed:
            self.transitioned.emit(transition)
        return transition

    def observe_unavailable(self, evidence: object, *, producer: str, reason: str) -> IdentityTransition:
        fields = _evidence_fields(evidence)
        runtime_reject = self._runtime_reject_reason(fields)
        if runtime_reject and producer != "RuntimeStateService":
            return self._reject(producer, runtime_reject, evidence)
        with self._lock:
            old = self._snapshot
        boundary_reason = self._boundary_conflict(old, fields)
        if old.retained and boundary_reason:
            return self.invalidate(evidence, producer=producer, reason=boundary_reason, strong=True)
        transition = IdentityTransition(old, old, False, "retain", producer, reason)
        if reason != "runtime_heartbeat":
            self._audit(transition, "identity_observation_retained")
        return transition

    def invalidate(
        self,
        evidence: object | None,
        *,
        producer: str,
        reason: str,
        strong: bool = False,
    ) -> IdentityTransition:
        fields = _evidence_fields(evidence)
        runtime_reject = self._runtime_reject_reason(fields)
        if runtime_reject and producer != "RuntimeStateService":
            return self._reject(producer, runtime_reject, evidence)
        with self._lock:
            old = self._snapshot
        boundary_reason = self._boundary_conflict(old, fields)
        if old.retained and not strong and not boundary_reason:
            transition = IdentityTransition(old, old, False, "retain", producer, reason)
            self._audit(transition, "identity_observation_retained")
            return transition
        new = CurrentIdentitySnapshot(
            state=CurrentIdentityState.UNKNOWN,
            account_scope_id=str(fields["account_scope_id"] or ""),
            target_instance_id=str(fields["target_instance_id"] or ""),
            hwnd=fields["hwnd"],
            process_id=fields["process_id"],
            window_generation=int(fields["window_generation"] or 0),
            capture_generation=int(fields["capture_generation"] or 0),
            account_identity_epoch=int(fields["account_identity_epoch"] or 0),
            source_job_id=fields["source_job_id"],
            observed_sequence=int(fields["observed_sequence"] or 0),
            producer=producer,
            reason=reason,
            updated_at=utc_now_iso(),
        )
        with self._lock:
            old = self._snapshot
            changed = old.state != CurrentIdentityState.UNKNOWN or bool(old.contact_id)
            self._snapshot = new
        transition = IdentityTransition(old, new, changed, "invalidate" if changed else "unknown", producer, reason)
        self._audit(transition, "identity_state_transition" if changed else "identity_observation_retained")
        if changed:
            self.transitioned.emit(transition)
        return transition

    def observe_runtime_state(self, state: object) -> IdentityTransition | None:
        window = getattr(state, "window", None)
        if window is None:
            return None
        hwnd = getattr(window, "hwnd", None)
        missing = hwnd is None and getattr(window, "state", None) == WindowState.MISSING
        with self._lock:
            self._runtime_window_missing = missing
            self._runtime_hwnd = hwnd
            self._runtime_process_id = getattr(window, "process_id", None)
            self._runtime_window_generation = int(getattr(state, "window_generation", 0) or 0)
            self._runtime_capture_generation = int(getattr(state, "capture_generation", 0) or 0)
            self._runtime_account_identity_epoch = int(getattr(state, "account_identity_epoch", 0) or 0)
            current = self._snapshot
        if missing:
            if current.retained:
                return self.invalidate(
                    state,
                    producer="RuntimeStateService",
                    reason="wechat_window_closed",
                    strong=True,
                )
            return None
        if hwnd is None:
            if current.retained:
                return self.observe_unavailable(
                    state,
                    producer="RuntimeStateService",
                    reason="window_observation_unavailable",
                )
            return None
        if current.retained:
            boundary_reason = self._boundary_conflict(current, _evidence_fields(state))
            if boundary_reason:
                return self.invalidate(
                    state,
                    producer="RuntimeStateService",
                    reason=boundary_reason,
                    strong=True,
                )
            # Advancing capture_generation quarantines earlier jobs without
            # changing the active conversation or emitting a UI transition.
            with self._lock:
                latest = self._snapshot
                if latest.conversation_token == current.conversation_token:
                    self._snapshot = replace(
                        latest,
                        capture_generation=self._runtime_capture_generation,
                        account_identity_epoch=self._runtime_account_identity_epoch,
                    )
            return self.observe_unavailable(
                state,
                producer="RuntimeStateService",
                reason="runtime_heartbeat",
            )
        # Preserve the current live window boundary while identity is unknown
        # or pending so an old OCR signal cannot repopulate a closed/switched
        # window after Runtime has already rejected it.
        with self._lock:
            latest = self._snapshot
            self._snapshot = replace(
                latest,
                hwnd=hwnd,
                process_id=self._runtime_process_id,
                window_generation=self._runtime_window_generation,
                capture_generation=self._runtime_capture_generation,
                account_identity_epoch=self._runtime_account_identity_epoch,
            )
        return None

    def _boundary_conflict(self, current: CurrentIdentitySnapshot, fields: dict[str, Any]) -> str:
        if current.account_identity_epoch != int(fields["account_identity_epoch"] or 0):
            return "account_identity_epoch_changed"
        comparisons = (
            ("account_scope_changed", current.account_scope_id, fields["account_scope_id"]),
            ("target_instance_changed", current.target_instance_id, fields["target_instance_id"]),
            ("window_handle_changed", current.hwnd, fields["hwnd"]),
            ("window_process_changed", current.process_id, fields["process_id"]),
            ("window_generation_changed", current.window_generation, fields["window_generation"]),
        )
        for reason, old_value, new_value in comparisons:
            if new_value not in (None, "", 0) and old_value not in (None, "", 0) and old_value != new_value:
                return reason
        # capture_generation is intentionally absent: it is an anti-stale job
        # token, never a customer/conversation boundary.
        return ""

    def _runtime_reject_reason(self, fields: dict[str, Any]) -> str:
        with self._lock:
            runtime_window_missing = self._runtime_window_missing
            runtime_hwnd = self._runtime_hwnd
            runtime_process_id = self._runtime_process_id
            runtime_window_generation = self._runtime_window_generation
            runtime_capture_generation = self._runtime_capture_generation
            runtime_account_identity_epoch = self._runtime_account_identity_epoch
        if runtime_window_missing:
            return "runtime_window_missing"
        comparisons = (
            ("stale_runtime_hwnd", runtime_hwnd, fields["hwnd"]),
            ("stale_runtime_process", runtime_process_id, fields["process_id"]),
            (
                "stale_runtime_window_generation",
                runtime_window_generation,
                fields["window_generation"],
            ),
        )
        for reason, current_value, evidence_value in comparisons:
            if (
                current_value not in (None, "", 0)
                and evidence_value not in (None, "", 0)
                and current_value != evidence_value
            ):
                return reason
        if (
            runtime_capture_generation
            and fields["capture_generation"]
            and runtime_capture_generation != fields["capture_generation"]
        ):
            return "stale_runtime_capture_generation"
        if runtime_account_identity_epoch != int(fields["account_identity_epoch"] or 0):
            return "stale_runtime_account_identity_epoch"
        return ""

    def _reject(self, producer: str, reason: str, evidence: object | None) -> IdentityTransition:
        current = self.snapshot
        transition = IdentityTransition(current, current, False, "reject", producer, reason)
        self._audit(transition, "identity_evidence_rejected", evidence=evidence)
        return transition

    def _audit(self, transition: IdentityTransition, event: str, *, evidence: object | None = None) -> None:
        if self.logs is None:
            return
        source = _evidence_fields(evidence) if evidence is not None else {}
        context = {
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "task_id": transition.new.source_job_id or source.get("source_job_id"),
            "producer": transition.producer,
            "action": transition.action,
            "reason": transition.reason,
            "old": _snapshot_context(transition.old),
            "new": _snapshot_context(transition.new),
            "source_job_id": transition.new.source_job_id or source.get("source_job_id"),
        }
        message = (
            f"{transition.old.state.value}({transition.old.contact_id or '-'})"
            f"->{transition.new.state.value}({transition.new.contact_id or '-'}) "
            f"producer={transition.producer} reason={transition.reason}"
        )
        def persist() -> None:
            try:
                self.logs.append("info", "current_identity", event, message, context)
            except Exception:
                pass

        # Fast identity is emitted on the UI thread immediately before an
        # already-built Activation Snapshot is painted. Preserve the audit,
        # but defer its SQLite write until after that first render.
        if transition.producer == "FastCustomerResolver":
            QTimer.singleShot(0, persist)
        else:
            persist()


def _evidence_fields(evidence: object | None) -> dict[str, Any]:
    if evidence is None:
        return {
            "account_scope_id": "", "target_instance_id": "", "hwnd": None,
            "process_id": None, "window_generation": 0, "capture_generation": 0,
            "account_identity_epoch": 0,
            "source_job_id": None,
            "observed_sequence": 0,
        }
    window = getattr(evidence, "window", None)
    return {
        "account_scope_id": getattr(evidence, "account_scope_id", "") or "",
        "target_instance_id": getattr(evidence, "target_instance_id", "") or "",
        "hwnd": getattr(evidence, "hwnd", None) if window is None else getattr(window, "hwnd", None),
        "process_id": (
            getattr(evidence, "process_id", None)
            if window is None else getattr(window, "process_id", None)
        ),
        "window_generation": getattr(evidence, "window_generation", 0) or 0,
        "capture_generation": getattr(evidence, "capture_generation", 0) or 0,
        "account_identity_epoch": getattr(evidence, "account_identity_epoch", 0) or 0,
        "source_job_id": getattr(evidence, "job_id", None),
        "observed_sequence": int(
            (getattr(evidence, "performance_trace", None) or {}).get("observed_sequence") or 0
        ),
    }


def _snapshot_context(snapshot: CurrentIdentitySnapshot) -> dict[str, Any]:
    return {
        "state": snapshot.state.value,
        "account_scope_id": snapshot.account_scope_id,
        "target_instance_id": snapshot.target_instance_id,
        "window_generation": snapshot.window_generation,
        "capture_generation": snapshot.capture_generation,
        "account_identity_epoch": snapshot.account_identity_epoch,
        "normalized_title": snapshot.normalized_title,
        "customer_id": snapshot.contact_id,
        "hwnd": snapshot.hwnd,
        "process_id": snapshot.process_id,
        "source_job_id": snapshot.source_job_id,
        "observed_sequence": snapshot.observed_sequence,
        "updated_at": snapshot.updated_at,
    }
