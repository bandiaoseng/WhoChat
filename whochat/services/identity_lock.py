from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from whochat.core.models import Contact, ContactStatus, ConversationType, LEGACY_ACCOUNT_SCOPE_ID
from whochat.storage.repositories import ContactRepository, IdentityBindingRepository


class AvatarEvidenceProvider(Protocol):
    """Future non-invasive avatar evidence hook.

    Phase 1 intentionally supplies no implementation; screenshots remain the
    only source and no WeChat data is decrypted or modified.
    """

    def fingerprint(self, result: object) -> str:
        ...


class NoopAvatarEvidenceProvider:
    def fingerprint(self, result: object) -> str:
        return ""


@dataclass(frozen=True)
class IdentityResolution:
    stable: bool
    reason: str
    identity_key: str
    title: str
    conversation_type: ConversationType
    confidence: float
    observations: int
    contact: Contact | None = None
    account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID
    target_instance_id: str = ""
    window_generation: int = 0
    capture_generation: int = 0
    account_identity_epoch: int = 0


class StableIdentityLock:
    """Requires two distinct captured frames before a contact becomes active."""

    def __init__(
        self,
        contacts: ContactRepository,
        bindings: IdentityBindingRepository,
        avatar_provider: AvatarEvidenceProvider | None = None,
        *,
        required_observations: int = 2,
        min_confidence: float = 0.50,
    ) -> None:
        self.contacts = contacts
        self.bindings = bindings
        self.avatar_provider = avatar_provider or NoopAvatarEvidenceProvider()
        self.required_observations = max(2, required_observations)
        self.min_confidence = min_confidence
        self._candidate_key = ""
        self._candidate_title = ""
        self._candidate_type = ConversationType.UNKNOWN
        self._candidate_platform = ""
        self._candidate_scope_id = ""
        self._candidate_target_instance_id = ""
        self._candidate_window_generation = 0
        self._candidate_capture_generation = 0
        self._candidate_account_identity_epoch = 0
        self._last_job_id: int | None = None
        self._observations = 0
        self._confidence = 0.0
        self._stable_contact: Contact | None = None

    @property
    def stable_contact(self) -> Contact | None:
        return self._stable_contact

    @property
    def candidate_conversation_type(self) -> ConversationType:
        return self._candidate_type

    @property
    def candidate_title(self) -> str:
        return self._candidate_title

    @property
    def identity_key(self) -> str:
        return self._candidate_key if self._stable_contact else ""

    def invalidate(self, reason: str = "identity_invalidated") -> IdentityResolution:
        title = self._candidate_title
        conversation_type = self._candidate_type
        confidence = self._confidence
        observations = self._observations
        self._candidate_key = ""
        self._candidate_title = ""
        self._candidate_type = ConversationType.UNKNOWN
        self._candidate_platform = ""
        self._candidate_scope_id = ""
        self._candidate_target_instance_id = ""
        self._candidate_window_generation = 0
        self._candidate_capture_generation = 0
        self._candidate_account_identity_epoch = 0
        self._last_job_id = None
        self._observations = 0
        self._confidence = 0.0
        self._stable_contact = None
        return IdentityResolution(False, reason, "", title, conversation_type, confidence, observations)

    def observe(
        self,
        *,
        job_id: int,
        platform: str,
        title: str,
        conversation_type: ConversationType,
        confidence: float,
        account_scope_id: str = LEGACY_ACCOUNT_SCOPE_ID,
        target_instance_id: str = "",
        window_generation: int = 0,
        capture_generation: int = 0,
        account_identity_epoch: int = 0,
        evidence_source: object | None = None,
    ) -> IdentityResolution:
        trace = getattr(evidence_source, "performance_trace", None)
        if isinstance(trace, dict) and trace.get("manual_title_fresh_required") is True:
            # Two jobs that repeat a cached/synthetic title are not two
            # independent title observations for a button-owned request.
            source = getattr(evidence_source, "title_source", None) or trace.get("title_source")
            if source != "roi_ocr" or trace.get("title_cache_hit") is not False:
                return self.invalidate("identity_title_not_independent")
        same_lifecycle = bool(self._candidate_key) and (
            self._candidate_platform == platform
            and self._candidate_scope_id == account_scope_id
            and self._candidate_target_instance_id == target_instance_id
            and self._candidate_window_generation == window_generation
        )
        if same_lifecycle and account_identity_epoch < self._candidate_account_identity_epoch:
            return IdentityResolution(
                False, "identity_stale_account_epoch", "", title, conversation_type, confidence, self._observations,
                account_scope_id=account_scope_id,
                target_instance_id=target_instance_id,
                window_generation=window_generation,
                capture_generation=capture_generation,
                account_identity_epoch=account_identity_epoch,
            )
        if self._last_job_id is not None and job_id < self._last_job_id:
            return IdentityResolution(
                False, "identity_stale_job", "", title, conversation_type, confidence, self._observations,
                account_scope_id=account_scope_id,
                target_instance_id=target_instance_id,
                window_generation=window_generation,
                capture_generation=capture_generation,
                account_identity_epoch=account_identity_epoch,
            )
        normalized_title = _display_title(title)
        identity_boundary_changed = bool(self._candidate_key) and (
            self._candidate_platform != platform
            or self._candidate_scope_id != account_scope_id
            or self._candidate_target_instance_id != target_instance_id
            or self._candidate_window_generation != window_generation
            or self._candidate_account_identity_epoch != account_identity_epoch
        )
        same_title_platform = (
            bool(self._candidate_title)
            and self._candidate_title.casefold() == normalized_title.casefold()
            and self._candidate_platform == platform
            and self._candidate_scope_id == account_scope_id
            and self._candidate_target_instance_id == target_instance_id
            and self._candidate_window_generation == window_generation
            and self._candidate_account_identity_epoch == account_identity_epoch
        )
        effective_type = conversation_type
        if same_title_platform and conversation_type == ConversationType.UNKNOWN and self._candidate_type != ConversationType.UNKNOWN:
            effective_type = self._candidate_type
        key = _identity_key(account_scope_id, platform, effective_type, normalized_title)
        if not normalized_title or confidence < self.min_confidence:
            return self.invalidate("identity_evidence_unreliable")

        if same_title_platform and self._candidate_type == ConversationType.UNKNOWN and conversation_type != ConversationType.UNKNOWN:
            # Full-page OCR refines the title-only candidate. It is the same
            # captured frame, so preserve the count but do not increment it.
            self._candidate_type = conversation_type
            self._candidate_key = key
            self._confidence = min(self._confidence, confidence)
            self._last_job_id = job_id
            self._candidate_capture_generation = capture_generation
            self._candidate_account_identity_epoch = account_identity_epoch
        elif identity_boundary_changed or key != self._candidate_key:
            self._candidate_key = key
            self._candidate_title = normalized_title
            self._candidate_type = effective_type
            self._candidate_platform = platform
            self._candidate_scope_id = account_scope_id
            self._candidate_target_instance_id = target_instance_id
            self._candidate_window_generation = window_generation
            self._candidate_capture_generation = capture_generation
            self._candidate_account_identity_epoch = account_identity_epoch
            self._last_job_id = job_id
            self._observations = 1
            self._confidence = confidence
            self._stable_contact = None
        elif job_id != self._last_job_id:
            self._last_job_id = job_id
            self._observations += 1
            self._confidence = min(self._confidence, confidence)
            # Capture generation is an anti-stale OCR token, not a customer
            # identity boundary.  Foreground/background transitions may bump
            # it while the same account/window/title remains active.
            self._candidate_capture_generation = capture_generation
        else:
            self._confidence = min(self._confidence, confidence)
            self._candidate_capture_generation = capture_generation

        if self._observations < self.required_observations:
            return IdentityResolution(
                False,
                f"identity_pending:{self._observations}/{self.required_observations}",
                key,
                normalized_title,
                conversation_type,
                self._confidence,
                self._observations,
                account_scope_id=account_scope_id,
                target_instance_id=target_instance_id,
                window_generation=window_generation,
                capture_generation=capture_generation,
                account_identity_epoch=account_identity_epoch,
            )

        if self._candidate_type == ConversationType.UNKNOWN:
            return IdentityResolution(
                False,
                "identity_waiting_page_type",
                self._candidate_key,
                normalized_title,
                self._candidate_type,
                self._confidence,
                self._observations,
                account_scope_id=account_scope_id,
                target_instance_id=target_instance_id,
                window_generation=window_generation,
                capture_generation=capture_generation,
                account_identity_epoch=account_identity_epoch,
            )

        conversation_type = self._candidate_type
        key = self._candidate_key
        binding = self.bindings.find(platform, conversation_type, normalized_title, account_scope_id)
        contact = self.contacts.get(binding.contact_id) if binding else None
        if contact is None:
            contact = self.contacts.create_or_get_by_display_name(
                normalized_title,
                platform=platform,
                conversation_type=conversation_type,
                account_scope_id=account_scope_id,
            )
        unknown_dm_compatible = (
            conversation_type == ConversationType.DM
            and contact.conversation_type == ConversationType.UNKNOWN
        )
        if contact.conversation_type != conversation_type and not unknown_dm_compatible:
            # The name-only repository lookup can return a same-name DM for
            # a group. Never bind that record to the proven page type or use
            # its profile/messages merely because the visible names match.
            # A manual/imported UNKNOWN Contact remains compatible with DM
            # only after the two independent frames and proven DM page above;
            # it does not authorize GROUP or rewrite stored contact metadata.
            self._stable_contact = None
            return IdentityResolution(
                False, "identity_contact_type_conflict", key, normalized_title,
                conversation_type, self._confidence, self._observations,
                account_scope_id=account_scope_id,
                target_instance_id=target_instance_id,
                window_generation=window_generation,
                capture_generation=capture_generation,
                account_identity_epoch=account_identity_epoch,
            )
        if contact.status == ContactStatus.UNCONFIRMED:
            contact = self.contacts.update_profile(contact.id, status=ContactStatus.SUSPECTED)
        avatar_fingerprint = self.avatar_provider.fingerprint(evidence_source) if evidence_source is not None else ""
        self.bindings.upsert(
            platform=platform,
            conversation_type=conversation_type,
            normalized_title=normalized_title,
            contact_id=contact.id,
            confidence=self._confidence,
            verified=contact.status == ContactStatus.CONFIRMED,
            avatar_fingerprint=avatar_fingerprint,
            account_scope_id=account_scope_id,
        )
        self._stable_contact = contact
        return IdentityResolution(
            True,
            "identity_stable",
            key,
            normalized_title,
            conversation_type,
            self._confidence,
            self._observations,
            contact,
            account_scope_id,
            target_instance_id,
            window_generation,
            capture_generation,
            account_identity_epoch,
        )


def _display_title(value: str) -> str:
    return " ".join(value.strip().split())[:80]


def _identity_key(account_scope_id: str, platform: str, conversation_type: ConversationType, title: str) -> str:
    return f"{account_scope_id}|{platform.strip().casefold()}|{conversation_type.value}|{title.casefold()}"
