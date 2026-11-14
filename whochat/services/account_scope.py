from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
from threading import RLock

from PIL import Image, ImageStat

from whochat.core.runtime import LayoutRegions, Rect, WindowSnapshot, account_identity_runtime_key
from whochat.storage.repositories import AccountScopeRepository


@dataclass(frozen=True)
class AccountContext:
    account_scope_id: str | None
    account_scope_name: str
    target_instance_id: str
    runtime_key: str
    visual_hash: str
    visual_reliable: bool
    bound: bool
    confidence: float
    reason: str
    identity_state: str = "unbound"
    identity_epoch: int = 0
    epoch_changed: bool = False
    # Per-observation pixels, not a durable binding or new account authority.
    # visual_hash above intentionally remains the repository's retained anchor.
    observed_visual_hash: str = ""
    observed_visual_reliable: bool = False

    @property
    def actions_allowed(self) -> bool:
        return self.bound and self.identity_state == AccountIdentityState.TRUSTED


class AccountIdentityState(StrEnum):
    TRUSTED = "trusted"
    REVALIDATING = "revalidating"
    UNBOUND = "unbound"


class AccountImageEvidenceUnavailable(RuntimeError):
    """Discard pixels that cannot prove the current native account window."""


@dataclass(frozen=True)
class AccountIdentityGuard:
    trusted: bool
    reason: str
    identity_epoch: int
    boundary_started: bool = False
    runtime_key: str = ""


@dataclass(frozen=True)
class AccountIdentityCheckpoint:
    """Current in-memory authority for one observed native runtime.

    Reading this checkpoint never observes new evidence, rebinds an account,
    writes a target or creates a scope. It repairs a missed notification only
    for the exact runtime requested by the active window consumer.
    """

    runtime_key: str
    hwnd: int
    process_id: int | None
    identity_epoch: int
    identity_state: str
    account_scope_id: str = ""
    target_instance_id: str = ""
    reason: str = ""
    # Current negative-only recovery progress, never account authorization.
    native_recovery_pending: bool = False


@dataclass(frozen=True)
class RawOcrAccountPreparation:
    """Read-only computation partition; never an account action authority.

    This has no trusted state or actions_allowed property. Only an already
    bound native runtime can prepare OCR before its first full observation.
    The manual pipeline must establish its own fresh AccountContext.
    """

    runtime_key: str
    hwnd: int
    process_id: int
    identity_epoch: int
    account_scope_id: str
    target_instance_id: str
    native_token: tuple


@dataclass(frozen=True)
class _TrustedBinding:
    account_scope_id: str
    target_instance_id: str
    visual_hash: str


@dataclass
class _Revalidation:
    previous_scope_id: str
    previous_instance_id: str
    previous_visual_hash: str
    reason: str
    candidate_scope_id: str | None = None
    candidate_visual_hash: str = ""
    stable_count: int = 0
    allow_unbound_confirmation: bool = False


@dataclass
class _AvatarFingerprintMigration:
    instance_id: str
    account_scope_id: str
    candidate_visual_hash: str = ""
    stable_count: int = 0


class AccountScopeResolver:
    """Resolves a visible WeChat window to an explicitly local account scope.

    HWND and PID are deliberately only runtime evidence.  A conservative visual
    fingerprint of the visible navigation avatar can reconnect a previously
    bound account after restart, but ambiguous or concurrently active matches
    remain unbound and therefore cannot feed the message timeline.
    """

    _ACTIVE_CONFLICT_SECONDS = 300
    # An exact/near-exact avatar crop remains the preferred reconnect path.
    # Real WeChat 4.x window reincarnations can move/resize the navigation rail,
    # which shifts an otherwise identical 16x16 aHash much more than two bits.
    # A wider recovery band is therefore allowed only when exactly one account
    # scope is present in that band and it is separated from every other scope
    # by a large margin.  Ambiguous evidence remains unbound.
    _STRICT_HASH_DISTANCE = 2
    _RECOVERY_HASH_DISTANCE = 64
    _RECOVERY_SCOPE_MARGIN = 32
    _RECOVERY_STABILITY_DISTANCE = 8
    _SWITCH_SCOPE_MARGIN = 8
    _SCOPE_SWITCH_STABLE_FRAMES = 3

    def __init__(self, repository: AccountScopeRepository, *, native_identity=None) -> None:
        self.repository = repository
        self.native_identity = native_identity
        self._native_tokens: dict[str, tuple] = {}
        self._native_recovery: dict[str, tuple[tuple, int]] = {}
        self._recovery_pending: dict[str, tuple[str, str, int]] = {}
        self._trusted_runtime: dict[str, _TrustedBinding] = {}
        self._revalidating: dict[str, _Revalidation] = {}
        self._avatar_migrations: dict[str, _AvatarFingerprintMigration] = {}
        self._identity_epochs: dict[str, int] = {}
        self._closed_runtime_epochs: dict[str, int] = {}
        self._lock = RLock()

    def account_identity_checkpoint(self, window: WindowSnapshot) -> AccountIdentityCheckpoint | None:
        if window.hwnd is None:
            return None
        runtime_key = _runtime_key(window)
        with self._lock:
            trusted = self._trusted_runtime.get(runtime_key)
            frozen = runtime_key in self._revalidating or runtime_key in self._avatar_migrations
            if trusted is None and not frozen and runtime_key not in self._identity_epochs:
                return None
            return AccountIdentityCheckpoint(
                runtime_key=runtime_key,
                hwnd=window.hwnd,
                process_id=window.process_id,
                identity_epoch=self._identity_epochs.get(runtime_key, 0),
                identity_state=(
                    AccountIdentityState.REVALIDATING if frozen
                    else AccountIdentityState.TRUSTED if trusted is not None
                    else AccountIdentityState.UNBOUND
                ),
                account_scope_id=trusted.account_scope_id if trusted is not None and not frozen else "",
                target_instance_id=trusted.target_instance_id if trusted is not None and not frozen else "",
                reason=str(getattr(self._revalidating.get(runtime_key), "reason", "")),
                native_recovery_pending=(runtime_key in self._revalidating
                    and self._native_recovery.get(runtime_key, (None, 0))[1] == 1),
            )

    def read_raw_ocr_preparation(self, window: WindowSnapshot, *,
                                 expected_identity_epoch: int,
                                 expected_runtime_key: str) -> RawOcrAccountPreparation | None:
        """Fresh native read for idle OCR only; never observe, bind or unfreeze.

        Run on the OCR worker. Do not hold the resolver lock during native
        I/O, so UI checkpoint reads remain cheap. Recheck the entire authority
        boundary after I/O; an intervening observation cancels preparation.
        """
        key = _runtime_key(window)
        if (self.native_identity is None or key != expected_runtime_key
                or not key or not window.hwnd or not window.process_id):
            return None
        with self._lock:
            if (self.account_identity_checkpoint(window) is not None
                    or key in self._closed_runtime_epochs
                    or self._identity_epochs.get(key, 0) != expected_identity_epoch):
                return None
            previous_token = self._native_tokens.get(key)
        proof = self.native_identity.read(window)
        if (not proof.verified or not proof.account_scope_id or not proof.token
                or (previous_token is not None and previous_token != proof.token)):
            return None
        with self._lock:
            if (self.account_identity_checkpoint(window) is not None
                    or key in self._closed_runtime_epochs
                    or self._identity_epochs.get(key, 0) != expected_identity_epoch
                    or self._native_tokens.get(key) != previous_token):
                return None
            row, conflict = self._native_binding(window, proof)
            if (conflict or row is None or not row.account_scope_id
                    or row.runtime_key != key or not row.id
                    or self.repository.get_scope(row.account_scope_id) is None):
                return None
            return RawOcrAccountPreparation(key, window.hwnd, window.process_id,
                expected_identity_epoch, row.account_scope_id, row.id, proof.token)

    def mark_runtime_closed(self, window: WindowSnapshot, *, expected_identity_epoch: int) -> bool:
        """Retire only the current native lifecycle, without deleting data.

        Runtime invokes this synchronously for an actual missing canonical
        window. A delayed caller from an earlier epoch cannot retire a newly
        verified account that happened to reuse the same PID/HWND.
        """
        if window.hwnd is None:
            return False
        runtime_key = _runtime_key(window)
        with self._lock:
            current_epoch = self._identity_epochs.get(runtime_key, 0)
            if expected_identity_epoch != current_epoch:
                return False
            if self._closed_runtime_epochs.get(runtime_key) == current_epoch:
                return False
            self._native_recovery.pop(runtime_key, None)
            pending = self._revalidating.get(runtime_key)
            migration = self._avatar_migrations.get(runtime_key)
            if pending is not None:
                # Frames from the closed incarnation cannot complete its
                # successor's proof. Preserve the already-frozen epoch.
                pending.candidate_scope_id = None
                pending.candidate_visual_hash = ""
                pending.stable_count = 0
            elif migration is not None:
                migration.candidate_visual_hash = ""
                migration.stable_count = 0
            else:
                trusted = self._trusted_runtime.get(runtime_key)
                if trusted is None:
                    return False
                self._start_revalidation(
                    runtime_key,
                    trusted,
                    "",
                    None,
                    "native_window_closed",
                    stable_count=0,
                    allow_unbound_confirmation=False,
                )
                current_epoch = self._identity_epochs[runtime_key]
            self._closed_runtime_epochs[runtime_key] = current_epoch
            return True

    def observe(
        self,
        image_path: Path,
        layout: LayoutRegions,
        window: WindowSnapshot,
        *, native_capture_token=None,
    ) -> AccountContext:
        self.require_avatar_evidence_current(window)
        visual_hash, visual_reliable = _visual_fingerprint(image_path, layout)
        # Focus can change while the image is opened/cropped/hashed.  Never let
        # those pixels rotate an account epoch or mutate a trusted binding.
        self.require_avatar_evidence_current(window)
        if self.native_identity is not None:
            with self._lock:
                proof = self.native_identity.read(window)
                if not proof.verified:
                    resolved = self._freeze_native_account(window, proof.reason)
                elif native_capture_token is None or native_capture_token != proof.token:
                    resolved = self._freeze_native_account(window, "native_account_owner_changed")
                else:
                    resolved = self._observe_native_account(window, visual_hash, visual_reliable, proof=proof)
        else:
            resolved = self.observe_fingerprint(window, visual_hash, visual_reliable, layout=layout)
        return replace(resolved, observed_visual_hash=visual_hash,
                       observed_visual_reliable=visual_reliable)

    def require_avatar_evidence_current(self, window: WindowSnapshot) -> None:
        # Full image observation and FastProbe deliberately use the same native
        # evidence gate. A lost focus is not proof of a changed account.
        if not _foreground_account_evidence_current(window):
            raise AccountImageEvidenceUnavailable("account_fingerprint_evidence_not_foreground")

    def capture_identity_token(self, window):
        if self.native_identity is None:
            return None
        proof = self.native_identity.read(window)
        return proof.token if proof.verified else None

    def capture_identity_reason(self, window, token) -> str:
        if self.native_identity is None:
            return ""
        proof = self.native_identity.read(window)
        if not proof.verified:
            return proof.reason
        return "" if token is not None and token == proof.token else "native_account_owner_changed"

    def observe_fingerprint(
        self,
        window: WindowSnapshot,
        visual_hash: str,
        visual_reliable: bool,
        *,
        layout: LayoutRegions | None = None,
    ) -> AccountContext:
        """Resolve account evidence while keeping PID/HWND runtime-only.

        A bound runtime is retained across harmless visual drift, but a stable
        fingerprint that points to another scope (or is far outside the trusted
        account band) starts a fail-closed two-frame revalidation.
        """

        runtime_key = _runtime_key(window)
        with self._lock:
            if self.native_identity is not None:
                return self._observe_native_account(window, visual_hash, visual_reliable)
            pending = self._revalidating.get(runtime_key)
            if pending is not None:
                return self._advance_revalidation(
                    window,
                    runtime_key,
                    visual_hash,
                    visual_reliable,
                )

            exact = self.repository.find_instance(runtime_key, visual_hash) if visual_hash else None
            latest = self.repository.find_runtime_instance(runtime_key)
            if _avatar_v2_migration_pending(latest):
                return self._advance_avatar_fingerprint_migration(
                    window,
                    runtime_key,
                    latest,
                    visual_hash,
                    visual_reliable,
                )
            trusted = self._trusted_runtime.get(runtime_key)
            if trusted is None:
                seed = exact if exact is not None and exact.account_scope_id else latest
                if seed is not None and seed.account_scope_id:
                    trusted = _TrustedBinding(seed.account_scope_id, seed.id, seed.visual_hash)

            if trusted is not None:
                mismatch_scope, mismatch_reason = self._runtime_binding_mismatch(
                    window,
                    runtime_key,
                    visual_hash,
                    visual_reliable,
                    trusted,
                    exact,
                )
                if mismatch_reason:
                    return self._start_revalidation(
                        runtime_key,
                        trusted,
                        visual_hash,
                        mismatch_scope,
                        mismatch_reason,
                    )
                current = exact if exact is not None and exact.account_scope_id == trusted.account_scope_id else latest
                if current is None or current.account_scope_id != trusted.account_scope_id:
                    current = self.repository.get_instance(trusted.target_instance_id)
                if current is not None:
                    instance = self.repository.observe_instance(
                        platform=window.target.value,
                        runtime_key=runtime_key,
                        visual_hash=visual_hash,
                        visual_reliable=visual_reliable,
                        process_name=window.process_name,
                        executable_path=window.executable_path,
                        pid=window.process_id,
                        hwnd=window.hwnd,
                        rect_json=json.dumps(window.rect.as_tuple() if window.rect else None, separators=(",", ":")),
                        matched_scope_id=trusted.account_scope_id,
                        binding_source=current.binding_source,
                        confidence=current.confidence,
                    )
                    return self._trusted_context(runtime_key, instance, "runtime_instance_binding")

            return self._resolve_unbound(window, runtime_key, visual_hash, visual_reliable)

    def guard_fingerprint(
        self,
        window: WindowSnapshot,
        visual_hash: str,
        visual_reliable: bool,
    ) -> AccountIdentityGuard:
        """Cheap FastProbe guard; it may freeze, but never rebind or persist."""

        runtime_key = _runtime_key(window)
        # A screen-ROI capture is only account evidence while that exact native
        # WeChat HWND is still foreground. RuntimeState is poll-based and can
        # remain briefly stale after the user switches windows; accepting the
        # now-occluded pixels used to manufacture a large avatar hash drift and
        # freeze the correct account. Block the probe without rotating epoch.
        if not _foreground_account_evidence_current(window):
            return AccountIdentityGuard(
                False,
                "account_fingerprint_evidence_not_foreground",
                self._identity_epochs.get(runtime_key, 0),
                runtime_key=runtime_key,
            )
        with self._lock:
            if self.native_identity is not None:
                return self._guard_native_account(window)
            if runtime_key in self._revalidating:
                return AccountIdentityGuard(
                    False,
                    self._revalidating[runtime_key].reason,
                    self._identity_epochs.get(runtime_key, 0),
                    runtime_key=runtime_key,
                )
            exact = self.repository.find_instance(runtime_key, visual_hash) if visual_hash else None
            latest = self.repository.find_runtime_instance(runtime_key)
            if _avatar_v2_migration_pending(latest):
                context = self._advance_avatar_fingerprint_migration(
                    window,
                    runtime_key,
                    latest,
                    visual_hash,
                    visual_reliable,
                )
                return AccountIdentityGuard(
                    context.actions_allowed,
                    context.reason,
                    context.identity_epoch,
                    context.epoch_changed,
                    runtime_key,
                )
            trusted = self._trusted_runtime.get(runtime_key)
            if trusted is None:
                seed = exact if exact is not None and exact.account_scope_id else latest
                if seed is None or not seed.account_scope_id:
                    return AccountIdentityGuard(
                        False,
                        "account_scope_unbound",
                        self._identity_epochs.get(runtime_key, 0),
                        runtime_key=runtime_key,
                    )
                trusted = _TrustedBinding(seed.account_scope_id, seed.id, seed.visual_hash)
                self._trusted_runtime[runtime_key] = trusted
            mismatch_scope, reason = self._runtime_binding_mismatch(
                window,
                runtime_key,
                visual_hash,
                visual_reliable,
                trusted,
                exact,
            )
            if reason:
                context = self._start_revalidation(
                    runtime_key,
                    trusted,
                    visual_hash,
                    mismatch_scope,
                    reason,
                )
                return AccountIdentityGuard(
                    False,
                    reason,
                    context.identity_epoch,
                    True,
                    runtime_key,
                )
            return AccountIdentityGuard(
                True,
                "account_identity_trusted",
                self._identity_epochs.get(runtime_key, 0),
                runtime_key=runtime_key,
            )

    def freeze_for_explicit_boundary(self, window: WindowSnapshot, reason: str) -> AccountContext:
        runtime_key = _runtime_key(window)
        with self._lock:
            if self.native_identity is not None:
                return self._freeze_native_account(window, reason)
            pending = self._revalidating.get(runtime_key)
            if pending is not None:
                return self._frozen_context(runtime_key, pending, False)
            current = self.repository.find_runtime_instance(runtime_key)
            trusted = self._trusted_runtime.get(runtime_key)
            if trusted is None and current is not None and current.account_scope_id:
                trusted = _TrustedBinding(current.account_scope_id, current.id, current.visual_hash)
            if trusted is None:
                return AccountContext(
                    None, "", "", runtime_key, "", False, False, 0.0,
                    f"account_identity_unbound:{reason}", AccountIdentityState.UNBOUND,
                    self._identity_epochs.get(runtime_key, 0), False,
                )
            return self._start_revalidation(
                runtime_key,
                trusted,
                "",
                None,
                reason,
                stable_count=0,
                allow_unbound_confirmation=True,
            )

    def _resolve_unbound(
        self,
        window: WindowSnapshot,
        runtime_key: str,
        visual_hash: str,
        visual_reliable: bool,
    ) -> AccountContext:
        current = self.repository.find_instance(runtime_key, visual_hash) if visual_hash else None
        if current is None:
            current = self.repository.find_runtime_instance(runtime_key)

        matched_scope_id: str | None = current.account_scope_id if current else None
        binding_source = current.binding_source if current else "unbound"
        confidence = current.confidence if current else 0.0
        reason = "runtime_instance_unbound"

        if matched_scope_id:
            reason = "runtime_instance_binding"
        elif visual_reliable and visual_hash:
            match = self._visual_candidate(window, runtime_key, visual_hash)
            if match is not None:
                candidate, recovery_match, scope_candidates = match
                active_conflict = any(
                    _runtime_still_active(item, window, runtime_key) for item in scope_candidates
                )
                if not active_conflict:
                    if not recovery_match:
                        self._recovery_pending.pop(runtime_key, None)
                    recovery_confirmed = not recovery_match or self._confirm_recovery_candidate(
                        runtime_key,
                        candidate.account_scope_id,
                        visual_hash,
                    )
                    if recovery_confirmed:
                        matched_scope_id = candidate.account_scope_id
                        binding_source = "visual_reconnect_recovery" if recovery_match else "visual_reconnect"
                        if candidate.visual_hash == visual_hash:
                            confidence = 0.93
                        else:
                            confidence = 0.80 if recovery_match else 0.88
                        reason = "unique_visual_reconnect_recovery" if recovery_match else "unique_visual_reconnect"
                    else:
                        reason = "visual_reconnect_recovery_pending_second_frame"
                else:
                    self._recovery_pending.pop(runtime_key, None)
                    reason = "visual_match_active_on_other_window"
            else:
                self._recovery_pending.pop(runtime_key, None)
                ranked = self._ranked_visual_candidates(window, runtime_key, visual_hash)
                nearby_scopes = {
                    item.account_scope_id
                    for distance, item in ranked
                    if item.account_scope_id and distance <= self._RECOVERY_HASH_DISTANCE
                }
                reason = "visual_match_ambiguous" if len(nearby_scopes) > 1 else "visual_binding_not_found"
        elif not visual_reliable:
            self._recovery_pending.pop(runtime_key, None)
            reason = "visual_evidence_unreliable"

        instance = self.repository.observe_instance(
            platform=window.target.value,
            runtime_key=runtime_key,
            visual_hash=visual_hash,
            visual_reliable=visual_reliable,
            process_name=window.process_name,
            executable_path=window.executable_path,
            pid=window.process_id,
            hwnd=window.hwnd,
            rect_json=json.dumps(window.rect.as_tuple() if window.rect else None, separators=(",", ":")),
            matched_scope_id=matched_scope_id,
            binding_source=binding_source,
            confidence=confidence,
        )
        scope = self.repository.get_scope(instance.account_scope_id) if instance.account_scope_id else None
        context = AccountContext(
            account_scope_id=scope.id if scope else None,
            account_scope_name=scope.local_name if scope else "",
            target_instance_id=instance.id,
            runtime_key=runtime_key,
            visual_hash=visual_hash,
            visual_reliable=visual_reliable,
            bound=scope is not None,
            confidence=instance.confidence if scope else 0.0,
            reason=reason,
            identity_state=AccountIdentityState.TRUSTED if scope else AccountIdentityState.UNBOUND,
            identity_epoch=self._identity_epochs.get(runtime_key, 0),
        )
        if scope is not None:
            self._trusted_runtime[runtime_key] = _TrustedBinding(scope.id, instance.id, instance.visual_hash)
        return context

    def _runtime_binding_mismatch(
        self,
        window: WindowSnapshot,
        runtime_key: str,
        visual_hash: str,
        visual_reliable: bool,
        trusted: _TrustedBinding,
        exact,
    ) -> tuple[str | None, str]:
        if not visual_reliable or not visual_hash or not trusted.visual_hash:
            return None, ""
        if not _same_visual_hash_version(visual_hash, trusted.visual_hash):
            return None, "account_fingerprint_version_mismatch"
        if exact is not None and exact.account_scope_id and exact.account_scope_id != trusted.account_scope_id:
            return exact.account_scope_id, "account_fingerprint_exact_other_scope"
        trusted_distance = _hash_distance(visual_hash, trusted.visual_hash)
        ranked = self._ranked_visual_candidates(window, runtime_key, visual_hash)
        scope_distances: dict[str, int] = {}
        for distance, item in ranked:
            if item.account_scope_id:
                scope_distances[item.account_scope_id] = min(
                    distance,
                    scope_distances.get(item.account_scope_id, 10_000),
                )
        others = sorted(
            (distance, scope_id)
            for scope_id, distance in scope_distances.items()
            if scope_id != trusted.account_scope_id
        )
        if others:
            best_distance, best_scope = others[0]
            if (
                best_distance <= self._RECOVERY_HASH_DISTANCE
                and best_distance + self._SWITCH_SCOPE_MARGIN <= trusted_distance
            ):
                return best_scope, "account_fingerprint_prefers_other_scope"
        if trusted_distance > self._RECOVERY_HASH_DISTANCE:
            return None, "account_fingerprint_outside_trusted_band"
        return None, ""

    def _start_revalidation(
        self,
        runtime_key: str,
        trusted: _TrustedBinding,
        visual_hash: str,
        candidate_scope_id: str | None,
        reason: str,
        *,
        stable_count: int = 1,
        allow_unbound_confirmation: bool = False,
    ) -> AccountContext:
        self._identity_epochs[runtime_key] = self._identity_epochs.get(runtime_key, 0) + 1
        pending = _Revalidation(
            trusted.account_scope_id,
            trusted.target_instance_id,
            trusted.visual_hash,
            reason,
            candidate_scope_id,
            visual_hash,
            stable_count if visual_hash else 0,
            allow_unbound_confirmation,
        )
        self._revalidating[runtime_key] = pending
        return self._frozen_context(runtime_key, pending, True)

    def _advance_revalidation(
        self,
        window: WindowSnapshot,
        runtime_key: str,
        visual_hash: str,
        visual_reliable: bool,
    ) -> AccountContext:
        pending = self._revalidating[runtime_key]
        if not visual_reliable or not visual_hash:
            return self._frozen_context(runtime_key, pending, False)
        candidate_scope = self._revalidation_candidate_scope(window, runtime_key, visual_hash, pending)
        same_candidate = (
            pending.candidate_scope_id == candidate_scope
            and pending.candidate_visual_hash
            and _hash_distance(pending.candidate_visual_hash, visual_hash) <= self._RECOVERY_STABILITY_DISTANCE
        )
        if same_candidate:
            pending.stable_count += 1
        else:
            pending.candidate_scope_id = candidate_scope
            pending.candidate_visual_hash = visual_hash
            pending.stable_count = 1
        required_frames = (
            self._SCOPE_SWITCH_STABLE_FRAMES
            if candidate_scope and candidate_scope != pending.previous_scope_id
            else 2
        )
        if pending.stable_count < required_frames:
            return self._frozen_context(runtime_key, pending, False)

        # Avatar-only evidence may fail closed, but it must not manufacture a
        # permanent unbound account. A displaced ROI can be stable across many
        # captures while repeatedly sampling a neighbouring navigation icon.
        # Continue collecting evidence until the previous scope recovers or a
        # strict, available scope is confirmed. Only an explicit login/logout
        # boundary may expose a new unbound target for manual confirmation.
        if candidate_scope is None and not pending.allow_unbound_confirmation:
            pending.stable_count = min(pending.stable_count, 2)
            return self._frozen_context(runtime_key, pending, False)

        instance = self.repository.observe_instance(
            platform=window.target.value,
            runtime_key=runtime_key,
            visual_hash=visual_hash,
            visual_reliable=True,
            process_name=window.process_name,
            executable_path=window.executable_path,
            pid=window.process_id,
            hwnd=window.hwnd,
            rect_json=json.dumps(window.rect.as_tuple() if window.rect else None, separators=(",", ":")),
            matched_scope_id=candidate_scope,
            binding_source="account_identity_revalidated" if candidate_scope else "unbound_revalidated",
            confidence=0.93 if candidate_scope else 0.0,
            reuse_runtime_instance=candidate_scope == pending.previous_scope_id,
        )
        self._revalidating.pop(runtime_key, None)
        if candidate_scope:
            self._trusted_runtime[runtime_key] = _TrustedBinding(candidate_scope, instance.id, instance.visual_hash)
            reason = (
                "account_identity_revalidated_same_scope"
                if candidate_scope == pending.previous_scope_id
                else "account_identity_revalidated_scope_switch"
            )
            return self._trusted_context(runtime_key, instance, reason)
        self._trusted_runtime.pop(runtime_key, None)
        return AccountContext(
            None, "", instance.id, runtime_key, visual_hash, True, False, 0.0,
            "account_identity_revalidated_unknown", AccountIdentityState.UNBOUND,
            self._identity_epochs.get(runtime_key, 0), False,
        )

    def _advance_avatar_fingerprint_migration(
        self,
        window: WindowSnapshot,
        runtime_key: str,
        instance,
        visual_hash: str,
        visual_reliable: bool,
    ) -> AccountContext:
        """Enroll the corrected avatar ROI only for an explicit rollout marker.

        The migration marker is written after an operator has verified the
        runtime-to-account mapping. Even then, account actions remain frozen
        until two stable, reliable v2 avatar frames have been observed.
        """

        pending = self._avatar_migrations.get(runtime_key)
        boundary_started = pending is None
        if pending is None:
            self._identity_epochs[runtime_key] = self._identity_epochs.get(runtime_key, 0) + 1
            pending = _AvatarFingerprintMigration(instance.id, instance.account_scope_id)
            self._avatar_migrations[runtime_key] = pending

        if (
            not visual_reliable
            or not visual_hash
            or not visual_hash.startswith("v2:")
            or instance.id != pending.instance_id
            or instance.account_scope_id != pending.account_scope_id
        ):
            pending.candidate_visual_hash = ""
            pending.stable_count = 0
            return self._avatar_migration_context(runtime_key, pending, boundary_started)

        conflict = any(
            item.account_scope_id != pending.account_scope_id
            and _hash_distance(visual_hash, item.visual_hash) <= self._STRICT_HASH_DISTANCE
            and _runtime_still_active(item, window, runtime_key)
            for item in self.repository.list_bound_instances(window.target.value)
        )
        if conflict:
            pending.candidate_visual_hash = ""
            pending.stable_count = 0
            return self._avatar_migration_context(runtime_key, pending, boundary_started)

        if (
            pending.candidate_visual_hash
            and _hash_distance(pending.candidate_visual_hash, visual_hash)
            <= self._RECOVERY_STABILITY_DISTANCE
        ):
            pending.stable_count += 1
        else:
            pending.candidate_visual_hash = visual_hash
            pending.stable_count = 1
        if pending.stable_count < 2:
            return self._avatar_migration_context(runtime_key, pending, boundary_started)

        migrated = self.repository.complete_avatar_fingerprint_migration(
            instance_id=pending.instance_id,
            account_scope_id=pending.account_scope_id,
            visual_hash=visual_hash,
        )
        if migrated is None:
            pending.candidate_visual_hash = ""
            pending.stable_count = 0
            return self._avatar_migration_context(runtime_key, pending, boundary_started)
        self._avatar_migrations.pop(runtime_key, None)
        self._revalidating.pop(runtime_key, None)
        return self._trusted_context(runtime_key, migrated, "account_avatar_v2_migrated")

    def _avatar_migration_context(
        self,
        runtime_key: str,
        pending: _AvatarFingerprintMigration,
        changed: bool,
    ) -> AccountContext:
        return AccountContext(
            None,
            "",
            "",
            runtime_key,
            pending.candidate_visual_hash,
            bool(pending.candidate_visual_hash),
            False,
            0.0,
            "account_identity_revalidating:avatar_roi_v2_migration",
            AccountIdentityState.REVALIDATING,
            self._identity_epochs.get(runtime_key, 0),
            changed,
        )

    def _revalidation_candidate_scope(
        self,
        window: WindowSnapshot,
        runtime_key: str,
        visual_hash: str,
        pending: _Revalidation,
    ) -> str | None:
        if not _same_visual_hash_version(visual_hash, pending.previous_visual_hash):
            return None
        ranked = self._ranked_visual_candidates(window, runtime_key, visual_hash)

        def scope_is_available(scope_id: str) -> bool:
            return not any(
                item.account_scope_id == scope_id and _runtime_still_active(item, window, runtime_key)
                for _distance, item in ranked
            )

        exact = self.repository.find_instance(runtime_key, visual_hash)
        if exact is not None and exact.account_scope_id and scope_is_available(exact.account_scope_id):
            return exact.account_scope_id
        by_scope: dict[str, int] = {}
        for distance, item in ranked:
            if item.account_scope_id:
                by_scope[item.account_scope_id] = min(distance, by_scope.get(item.account_scope_id, 10_000))
        ordered = sorted((distance, scope_id) for scope_id, distance in by_scope.items())
        strict = [
            (distance, scope_id)
            for distance, scope_id in ordered
            if distance <= self._STRICT_HASH_DISTANCE and scope_is_available(scope_id)
        ]
        if strict:
            strict_scopes = {scope_id for _distance, scope_id in strict}
            if len(strict_scopes) == 1:
                return strict[0][1]

        # The broad recovery band exists only to recover the scope that was
        # already trusted. It must never switch to another account: a shifted
        # crop can consistently resemble another account's avatar and otherwise
        # pass the old two-frame rule.
        if (
            _hash_distance(visual_hash, pending.previous_visual_hash) <= self._RECOVERY_HASH_DISTANCE
            and scope_is_available(pending.previous_scope_id)
        ):
            return pending.previous_scope_id
        return None

    def _trusted_context(self, runtime_key: str, instance, reason: str) -> AccountContext:
        scope = self.repository.get_scope(instance.account_scope_id or "")
        if scope is None:
            return AccountContext(
                None, "", instance.id, runtime_key, instance.visual_hash, instance.visual_reliable,
                False, 0.0, "account_scope_missing", AccountIdentityState.UNBOUND,
                self._identity_epochs.get(runtime_key, 0), False,
            )
        self._trusted_runtime[runtime_key] = _TrustedBinding(scope.id, instance.id, instance.visual_hash)
        self._closed_runtime_epochs.pop(runtime_key, None)
        return AccountContext(
            scope.id, scope.local_name, instance.id, runtime_key, instance.visual_hash,
            instance.visual_reliable, True, instance.confidence, reason,
            AccountIdentityState.TRUSTED, self._identity_epochs.get(runtime_key, 0), False,
        )

    def _frozen_context(self, runtime_key: str, pending: _Revalidation, changed: bool) -> AccountContext:
        return AccountContext(
            None, "", "", runtime_key, pending.candidate_visual_hash,
            bool(pending.candidate_visual_hash), False, 0.0,
            f"account_identity_revalidating:{pending.reason}", AccountIdentityState.REVALIDATING,
            self._identity_epochs.get(runtime_key, 0), changed,
        )

    def _freeze_native_account(self, window, reason: str) -> AccountContext:
        key = _runtime_key(window)
        self._native_recovery.pop(key, None)
        pending = self._revalidating.get(key)
        if pending is not None:
            pending.reason = reason
            return self._frozen_context(key, pending, False)
        trusted = self._trusted_runtime.get(key)
        if trusted is None:
            row = self.repository.find_runtime_instance(key)
            trusted = (_TrustedBinding(row.account_scope_id, row.id, row.visual_hash)
                       if row is not None and row.account_scope_id else _TrustedBinding("", "", ""))
        return self._start_revalidation(key, trusted, "", None, reason, stable_count=0)

    def _native_binding(self, window, proof):
        key = _runtime_key(window)
        trusted = self._trusted_runtime.get(key)
        row = (self.repository.get_instance(trusted.target_instance_id) if trusted is not None
               else self.repository.find_runtime_instance(key))
        if row is not None and row.account_scope_id and row.account_scope_id != proof.account_scope_id:
            return row, "native_account_binding_conflict"
        return row, ""

    def _observe_native_account(self, window, visual_hash: str, visual_reliable: bool, *, proof=None) -> AccountContext:
        """Fresh independently mapped native identity; old avatar anchors are read-only.

        A contradictory existing runtime binding is never reassigned here. This
        also avoids the (runtime_key, visual_hash) collision for identical avatars.
        """
        key = _runtime_key(window)
        proof = proof if proof is not None else self.native_identity.read(window)
        if not proof.verified:
            return self._freeze_native_account(window, proof.reason)
        row, conflict = self._native_binding(window, proof)
        if conflict:
            return self._freeze_native_account(window, conflict)
        if not visual_reliable or not visual_hash:
            return self._freeze_native_account(window, "native_account_avatar_unavailable")
        previous_token = self._native_tokens.get(key)
        if previous_token is not None and previous_token != proof.token and key not in self._revalidating:
            return self._freeze_native_account(window, "native_account_owner_changed")
        if key in self._revalidating:
            previous = self._native_recovery.get(key)
            count = previous[1] + 1 if previous and previous[0] == proof.token else 1
            self._native_recovery[key] = (proof.token, count)
            if count < 2:
                # This frame has fresh good native evidence, but grants no
                # authority until the second full observation. Keep the old
                # fault on pending.reason without reporting it as a new fault.
                return replace(self._frozen_context(key, self._revalidating[key], False),
                    reason="account_identity_revalidating:native_account_confirmation_pending")
        # Current bound instances keep all stored hashes, bindings and rects.
        # New runtimes can reconnect only through the independent mapping.
        if row is None or not row.account_scope_id:
            row = self.repository.observe_instance(
                platform=window.target.value, runtime_key=key, visual_hash=visual_hash,
                visual_reliable=visual_reliable, process_name=window.process_name,
                executable_path=window.executable_path, pid=window.process_id, hwnd=window.hwnd,
                rect_json=json.dumps(window.rect.as_tuple() if window.rect else None, separators=(",", ":")),
                matched_scope_id=proof.account_scope_id, binding_source="verified_native_account_directory", confidence=1.0)
        if row.account_scope_id != proof.account_scope_id:
            return self._freeze_native_account(window, "native_account_binding_conflict")
        self._native_tokens[key] = proof.token
        self._native_recovery.pop(key, None)
        self._revalidating.pop(key, None)
        self._avatar_migrations.pop(key, None)
        return self._trusted_context(key, row, "native_account_verified")

    def _guard_native_account(self, window) -> AccountIdentityGuard:
        """Fresh negative-only check; never unfreeze, rebind, or persist."""
        key = _runtime_key(window)
        proof = self.native_identity.read(window)
        row, conflict = self._native_binding(window, proof) if proof.verified else (None, "")
        reason = proof.reason if not proof.verified else conflict
        recovery = self._native_recovery.get(key)
        # A fast check may invalidate recovery, but a stable check must not
        # erase the full-frame progress just because the old incarnation died.
        expected_token = (recovery[0] if recovery else None) if key in self._revalidating else self._native_tokens.get(key)
        if not reason and expected_token is not None and expected_token != proof.token:
            reason = "native_account_owner_changed"
        if reason:
            context = self._freeze_native_account(window, reason)
            return AccountIdentityGuard(False, reason, context.identity_epoch, context.epoch_changed, key)
        if key in self._revalidating or row is None or not row.account_scope_id:
            return AccountIdentityGuard(False, "native_account_confirmation_pending",
                                        self._identity_epochs.get(key, 0), runtime_key=key)
        self._native_tokens[key] = proof.token
        self._trusted_context(key, row, "native_account_verified")
        return AccountIdentityGuard(True, "native_account_verified", self._identity_epochs.get(key, 0), runtime_key=key)

    def validate_current_native_account(self, result, state) -> str:
        """Action boundary for generation and both copy surfaces; no positive cache."""
        if self.native_identity is None:
            return ""
        with self._lock:
            guard = self._guard_native_account(state.window)
            checkpoint = self.account_identity_checkpoint(state.window)
            if not guard.trusted:
                return guard.reason
            if (checkpoint is None or checkpoint.account_scope_id != getattr(result, "account_scope_id", None)
                    or checkpoint.target_instance_id != getattr(result, "target_instance_id", None)
                    or checkpoint.identity_epoch != getattr(result, "account_identity_epoch", None)
                    or checkpoint.runtime_key != getattr(result, "account_identity_runtime_key", None)):
                return "native_account_binding_conflict"
            return ""

    def bind(self, target_instance_id: str, local_name: str) -> AccountContext:
        instance = self.repository.bind_instance(target_instance_id, local_name)
        scope = self.repository.get_scope(instance.account_scope_id or "")
        if scope is None:
            raise RuntimeError("account scope binding did not persist")
        runtime_key = instance.runtime_key
        with self._lock:
            self._revalidating.pop(runtime_key, None)
            self._trusted_runtime[runtime_key] = _TrustedBinding(scope.id, instance.id, instance.visual_hash)
        return AccountContext(
            account_scope_id=scope.id,
            account_scope_name=scope.local_name,
            target_instance_id=instance.id,
            runtime_key=instance.runtime_key,
            visual_hash=instance.visual_hash,
            visual_reliable=instance.visual_reliable,
            bound=True,
            confidence=1.0,
            reason="manual_binding",
            identity_state=AccountIdentityState.TRUSTED,
            identity_epoch=self._identity_epochs.get(runtime_key, 0),
        )

    def _ranked_visual_candidates(
        self,
        window: WindowSnapshot,
        runtime_key: str,
        visual_hash: str,
    ) -> list[tuple[int, object]]:
        matches: list[tuple[int, object]] = []
        for item in self.repository.list_bound_instances(window.target.value):
            if not _same_process(window, item.process_name, item.executable_path):
                continue
            distance = _hash_distance(visual_hash, item.visual_hash)
            matches.append((distance, item))
        matches.sort(key=lambda pair: (pair[0], pair[1].last_seen_at), reverse=False)
        return matches

    def _visual_candidate(
        self,
        window: WindowSnapshot,
        runtime_key: str,
        visual_hash: str,
    ) -> tuple[object, bool, list[object]] | None:
        ranked = self._ranked_visual_candidates(window, runtime_key, visual_hash)
        if not ranked:
            return None
        by_scope: dict[str, list[tuple[int, object]]] = {}
        for distance, item in ranked:
            if not item.account_scope_id:
                continue
            by_scope.setdefault(item.account_scope_id, []).append((distance, item))
        if not by_scope:
            return None
        scope_ranked = sorted(
            (
                min(items, key=lambda pair: (pair[0], pair[1].last_seen_at))
                for items in by_scope.values()
            ),
            key=lambda pair: (pair[0], pair[1].last_seen_at),
        )
        best_distance, best = scope_ranked[0]
        best_scope_items = [item for _distance, item in by_scope[best.account_scope_id]]
        if best_distance <= self._STRICT_HASH_DISTANCE:
            strict_scopes = {
                item.account_scope_id
                for distance, item in ranked
                if distance <= self._STRICT_HASH_DISTANCE and item.account_scope_id
            }
            if len(strict_scopes) == 1:
                return best, False, best_scope_items
            return None
        if best_distance > self._RECOVERY_HASH_DISTANCE:
            return None
        recovery_scopes = {
            item.account_scope_id
            for distance, item in ranked
            if distance <= self._RECOVERY_HASH_DISTANCE and item.account_scope_id
        }
        if len(recovery_scopes) != 1:
            rect_match = self._unique_rect_recovery_candidate(window, ranked)
            if rect_match is None:
                return None
            best_distance, best = rect_match
            best_scope_items = [item for _distance, item in by_scope[best.account_scope_id]]
            return best, True, best_scope_items
        second_distance = scope_ranked[1][0] if len(scope_ranked) > 1 else 10_000
        if second_distance - best_distance < self._RECOVERY_SCOPE_MARGIN:
            rect_match = self._unique_rect_recovery_candidate(window, ranked)
            if rect_match is None:
                return None
            best_distance, best = rect_match
            best_scope_items = [item for _distance, item in by_scope[best.account_scope_id]]
            return best, True, best_scope_items
        return best, True, best_scope_items

    def _unique_rect_recovery_candidate(
        self,
        window: WindowSnapshot,
        ranked: list[tuple[int, object]],
    ) -> tuple[int, object] | None:
        """Use an exact historical window rectangle only as an ambiguity tie-break.

        Geometry is never sufficient by itself: the avatar must remain inside
        the bounded recovery distance, exactly one account scope may share the
        rectangle, and the normal two-frame plus active-window conflict gates
        still run after this method returns.
        """

        if window.rect is None:
            return None
        current_rect = tuple(int(value) for value in window.rect.as_tuple())
        matches: list[tuple[int, object]] = []
        for distance, item in ranked:
            if distance > self._RECOVERY_HASH_DISTANCE or not item.account_scope_id:
                continue
            try:
                stored = json.loads(str(getattr(item, "last_rect_json", "") or ""))
                stored_rect = tuple(int(value) for value in stored)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if stored_rect == current_rect:
                matches.append((distance, item))
        scopes = {item.account_scope_id for _distance, item in matches}
        if len(scopes) != 1:
            return None
        return min(matches, key=lambda pair: (pair[0], pair[1].last_seen_at))

    def _confirm_recovery_candidate(self, runtime_key: str, scope_id: str, visual_hash: str) -> bool:
        previous = self._recovery_pending.get(runtime_key)
        if previous is None or previous[0] != scope_id or _hash_distance(previous[1], visual_hash) > self._RECOVERY_STABILITY_DISTANCE:
            self._recovery_pending[runtime_key] = (scope_id, visual_hash, 1)
            return False
        count = previous[2] + 1
        if count < 2:
            self._recovery_pending[runtime_key] = (scope_id, visual_hash, count)
            return False
        self._recovery_pending.pop(runtime_key, None)
        return True


def _runtime_key(window: WindowSnapshot) -> str:
    return account_identity_runtime_key(window)


def _same_process(window: WindowSnapshot, process_name: str, executable_path: str) -> bool:
    if window.executable_path and executable_path:
        return window.executable_path.casefold() == executable_path.casefold()
    return bool(window.process_name and process_name and window.process_name.casefold() == process_name.casefold())


def _runtime_still_active(candidate, window: WindowSnapshot, current_key: str) -> bool:
    if candidate.runtime_key == current_key:
        return False
    native_checked = False
    try:
        import psutil

        native_checked = True
        if candidate.last_pid and not psutil.pid_exists(candidate.last_pid):
            return False
    except Exception:
        pass
    try:
        import win32gui, win32process

        native_checked = True
        if candidate.last_hwnd and not win32gui.IsWindow(candidate.last_hwnd):
            return False
        if candidate.last_hwnd:
            owner_pid = int(win32process.GetWindowThreadProcessId(candidate.last_hwnd)[1])
            if not candidate.last_pid or owner_pid != candidate.last_pid:
                return False
            # A recycled PID must not make an old account appear online.
            import psutil
            created = float(psutil.Process(owner_pid).create_time())
            observed = datetime.fromisoformat(candidate.last_seen_at.replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            if created > observed.timestamp():
                return False
            return True
    except Exception:
        pass
    if native_checked and candidate.last_pid and window.process_id and candidate.last_pid != window.process_id:
        return True
    try:
        observed = datetime.fromisoformat(candidate.last_seen_at.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - observed).total_seconds() <= AccountScopeResolver._ACTIVE_CONFLICT_SECONDS
    except (TypeError, ValueError):
        return True


def _visual_fingerprint(image_path: Path, layout: LayoutRegions) -> tuple[str, bool]:
    screen_rect = account_avatar_screen_rect(layout)
    window = layout.window_rect
    if (
        window.width <= 0
        or window.height <= 0
        or screen_rect.width < 18
        or screen_rect.height < 18
        or screen_rect.left < window.left
        or screen_rect.top < window.top
        or screen_rect.right > window.right
        or screen_rect.bottom > window.bottom
    ):
        return "", False
    try:
        with Image.open(image_path) as source:
            if source.width <= 0 or source.height <= 0:
                return "", False
            scale_x = source.width / window.width
            scale_y = source.height / window.height
            if not (0.25 <= scale_x <= 4.0 and 0.25 <= scale_y <= 4.0):
                return "", False
            box = (
                round((screen_rect.left - window.left) * scale_x),
                round((screen_rect.top - window.top) * scale_y),
                round((screen_rect.right - window.left) * scale_x),
                round((screen_rect.bottom - window.top) * scale_y),
            )
            if (
                box[0] < 0
                or box[1] < 0
                or box[2] > source.width
                or box[3] > source.height
                or box[2] - box[0] < 18
                or box[3] - box[1] < 18
            ):
                return "", False
            crop = source.convert("RGB").crop(box)
    except (OSError, ValueError):
        return "", False
    visual_hash, reliable = _fingerprint_crop(crop)
    return (_versioned_visual_hash(visual_hash), reliable) if visual_hash else ("", False)


def account_avatar_screen_rect(layout: LayoutRegions) -> Rect:
    window = layout.window_rect
    # The 4.x navigation avatar is on a fixed UI grid, NOT proportional to
    # the resizable outer frame or estimated/calibrated navigation width.
    # Keep the established v2 96-DPI sampling grid (18,46,40,40), scaled only
    # by native window DPI. All Full/FastProbe/preread/copy consumers call
    # this same function. No hash threshold or account binding is changed.
    scale = getattr(layout, "dpi_scale", 1.0)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not 0.75 <= scale <= 4.0):
        return Rect(window.left, window.top, window.left, window.top)
    side = round(40 * scale)
    left = window.left + round(18 * scale)
    top = window.top + round(46 * scale)
    return Rect(left, top, left + side, top + side)


def account_avatar_fingerprint(image_path: Path) -> tuple[str, bool]:
    try:
        with Image.open(image_path) as source:
            crop = source.convert("RGB")
    except (OSError, ValueError):
        return "", False
    visual_hash, reliable = _fingerprint_crop(crop)
    return (_versioned_visual_hash(visual_hash), reliable) if visual_hash else ("", False)


def _fingerprint_crop(crop: Image.Image) -> tuple[str, bool]:
    if crop.width < 18 or crop.height < 18:
        return "", False
    gray = crop.convert("L")
    stats = ImageStat.Stat(gray)
    variance = float(stats.stddev[0]) if stats.stddev else 0.0
    resized = gray.resize((16, 16), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    if not pixels:
        return "", False
    sorted_pixels = sorted(pixels)
    p10 = sorted_pixels[max(0, round((len(sorted_pixels) - 1) * 0.10))]
    p90 = sorted_pixels[min(len(sorted_pixels) - 1, round((len(sorted_pixels) - 1) * 0.90))]
    tonal_spread = p90 - p10
    mean = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | int(value >= mean)
    # Variance alone accepted a nearly uniform title-bar crop when a tiny
    # avatar sliver contributed a few outliers. Requiring central tonal spread
    # rejects that failure mode without treating one noisy edge as identity.
    return f"{bits:064x}", variance >= 8.0 and tonal_spread >= 12


def _hash_distance(left: str, right: str) -> int:
    left_version, left_digest = _split_visual_hash(left)
    right_version, right_digest = _split_visual_hash(right)
    if (
        not left_digest
        or not right_digest
        or left_version != right_version
        or len(left_digest) != len(right_digest)
    ):
        return 10_000
    try:
        return (int(left_digest, 16) ^ int(right_digest, 16)).bit_count()
    except ValueError:
        return 10_000


def _versioned_visual_hash(visual_hash: str) -> str:
    return f"v2:{visual_hash}" if visual_hash else ""


def _split_visual_hash(visual_hash: str) -> tuple[str, str]:
    if not visual_hash:
        return "", ""
    if ":" not in visual_hash:
        return "legacy", visual_hash
    version, digest = visual_hash.split(":", 1)
    return version, digest


def _same_visual_hash_version(left: str, right: str) -> bool:
    left_version, _left_digest = _split_visual_hash(left)
    right_version, _right_digest = _split_visual_hash(right)
    return bool(left_version and left_version == right_version)


def _avatar_v2_migration_pending(instance) -> bool:
    return bool(
        instance is not None
        and instance.account_scope_id
        and instance.binding_source == "avatar_roi_v2_migration_pending"
    )


def _foreground_account_evidence_current(window: WindowSnapshot) -> bool:
    if not window.foreground or not window.hwnd:
        return False
    try:
        import win32gui

        # Synthetic/offline windows are intentionally left to the caller's
        # WindowSnapshot so the deterministic test/replay path stays portable.
        if not win32gui.IsWindow(int(window.hwnd)):
            return True
        return int(win32gui.GetForegroundWindow() or 0) == int(window.hwnd)
    except Exception:
        # Native verification is defense-in-depth. Existing runtime/capture
        # generation guards remain authoritative if Win32 is unavailable.
        return True
