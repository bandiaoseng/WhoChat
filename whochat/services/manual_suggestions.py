"""Manual suggestion policy and copy-only foreground/pixel checks.

Nothing in this module starts a timer, performs OCR, reads a database, writes a
capture file, or grants an identity/dispatch/display lease.  An unchanged
sample is never a replacement for the ordinary source and lease validators.
An explicit copy can request its exact original window's foreground once;
polling remains the caller's bounded responsibility and grants no copy lease.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import os
from typing import Callable

from PIL import Image, ImageGrab

from whochat.core.current_conversation import _validate_capture_fields
from whochat.core.runtime import Rect
from whochat.services.account_scope import (
    _fingerprint_crop, _versioned_visual_hash, account_avatar_screen_rect,
)
from whochat.services.pipeline import _noncontent_scrollbar_rail, title_identity_screen_rect


# A release-level transitional mode, not a persistent configuration migration.
MANUAL_SUGGESTIONS_ONLY = True


@dataclass(frozen=True)
class NativeWindowSample:
    hwnd: int
    process_id: int
    rect: tuple[int, int, int, int]
    foreground: bool
    visible: bool
    minimized: bool
    foreground_hwnd: int = 0
    foreground_process_id: int = 0
    client_rect: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ScreenPixelCheck:
    unchanged: bool
    reason: str
    diagnostic: dict | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CopyFocusPreparation:
    allowed: bool
    needs_wait: bool
    reason: str
    native: NativeWindowSample | None = None


def read_native_window(hwnd: int) -> NativeWindowSample:
    """Read current native ownership/geometry; missing APIs fail closed."""
    import win32gui
    import win32process
    from whochat.platform.window_tracker import read_client_screen_rect

    if not win32gui.IsWindow(hwnd):
        raise ValueError("target_window_missing")
    _, process_id = win32process.GetWindowThreadProcessId(hwnd)
    foreground_hwnd = int(win32gui.GetForegroundWindow() or 0)
    foreground_process_id = (int(win32process.GetWindowThreadProcessId(foreground_hwnd)[1])
                             if foreground_hwnd else 0)
    window_rect = tuple(int(value) for value in win32gui.GetWindowRect(hwnd))
    client_rect = read_client_screen_rect(hwnd, window_rect, expected_process_id=int(process_id))
    return NativeWindowSample(
        hwnd=hwnd,
        process_id=int(process_id),
        rect=window_rect,
        foreground=foreground_hwnd == hwnd,
        visible=bool(win32gui.IsWindowVisible(hwnd)),
        minimized=bool(win32gui.IsIconic(hwnd)),
        foreground_hwnd=foreground_hwnd,
        foreground_process_id=foreground_process_id,
        client_rect=client_rect,
    )


def _client_geometry_error(layout, window, native: NativeWindowSample) -> str | None:
    """Validate frozen client geometry without rebasing or cropping the frame.

    A legacy/synthetic layout with no client evidence retains its original
    full-window pixel contract. It never acquires a client-edge exemption here.
    Proven client layouts require matching fresh Runtime and native evidence;
    only translating the whole outer/client pair preserves that evidence.
    """
    frozen = getattr(layout, "client_rect", None)
    if frozen is None:
        return None

    def relative(client, outer):
        if isinstance(client, Rect):
            client = client.as_tuple()
        if isinstance(outer, Rect):
            outer = outer.as_tuple()
        if (type(client) is not tuple or type(outer) is not tuple
                or len(client) != 4 or len(outer) != 4
                or any(type(value) is not int for value in (*client, *outer))):
            raise ValueError("client_geometry_invalid")
        left, top, right, bottom = client
        if (not outer[0] < outer[2] or not outer[1] < outer[3]
                or not outer[0] <= left < right <= outer[2]
                or not outer[1] <= top < bottom <= outer[3]):
            raise ValueError("client_geometry_invalid")
        return left - outer[0], top - outer[1], right - left, bottom - top

    current = getattr(window, "client_rect", None)
    observed = getattr(native, "client_rect", None)
    if current is None or observed is None:
        return "client_geometry_unavailable"
    if (not isinstance(frozen, Rect) or not isinstance(current, Rect)
            or type(observed) is not tuple):
        return "client_geometry_invalid"
    try:
        expected = relative(frozen, layout.window_rect)
        if (relative(current, window.rect) != expected
                or relative(observed, native.rect) != expected):
            return "client_geometry_changed"
    except (AttributeError, TypeError, ValueError):
        return "client_geometry_invalid"
    return None


def exact_frame_digests(image: Image.Image, layout) -> tuple[str, str]:
    """Match pipeline title/body hashing, without a temporary image file.

    The original full-proof layout is mandatory.  Only the pipeline's proven
    native scrollbar-rail exemption is applied; input drafts/carets are outside
    the message rectangle.  No perceptual threshold hides short replies.
    """
    rgb = image.convert("RGB")
    if rgb.size != (layout.window_rect.width, layout.window_rect.height):
        raise ValueError("sample_size_changed")
    origin = layout.window_rect

    def relative(rect):
        value = Rect(rect.left - origin.left, rect.top - origin.top,
                     rect.right - origin.left, rect.bottom - origin.top)
        if (value.width < 1 or value.height < 1 or value.left < 0 or value.top < 0
                or value.right > rgb.width or value.bottom > rgb.height):
            raise ValueError("sample_layout_out_of_bounds")
        return value

    title, messages = relative(title_identity_screen_rect(layout)), relative(layout.message_rect)
    title_digest = hashlib.sha256(rgb.crop(title.as_tuple()).tobytes()).hexdigest()
    rail = _noncontent_scrollbar_rail(rgb, layout, messages)
    if rail is not None:
        rgb.paste(rail[1], rail[0])
    digest = hashlib.sha256()
    for rect in (title, messages):
        region = rgb.crop(rect.as_tuple())
        digest.update(f"{region.width}x{region.height}|".encode("ascii"))
        digest.update(region.tobytes())
    return title_digest, digest.hexdigest()[:32]


def avatar_fingerprint_from_frame(image: Image.Image, layout, avatar_rect: Rect | None = None) -> tuple[str, bool]:
    """Use the existing v2 avatar algorithm on an in-memory full frame.

    A frozen original geometry is translated to image coordinates.  This
    function only computes pixels; it never calls the account resolver or
    establishes/updates a trusted account binding.
    """
    window = layout.window_rect
    rect = avatar_rect if avatar_rect is not None else account_avatar_screen_rect(layout)
    if (image.size != (window.width, window.height) or rect.width < 18 or rect.height < 18
            or rect.left < window.left or rect.top < window.top
            or rect.right > window.right or rect.bottom > window.bottom):
        return "", False
    crop = image.convert("RGB").crop((rect.left - window.left, rect.top - window.top,
                                      rect.right - window.left, rect.bottom - window.top))
    fingerprint, reliable = _fingerprint_crop(crop)
    return (_versioned_visual_hash(fingerprint), reliable) if fingerprint else ("", False)


def check_current_screen_pixels(
    result,
    state,
    *,
    sampler: Callable[[tuple[int, int, int, int]], Image.Image] | None = None,
    native_window_reader: Callable[[int], NativeWindowSample] | None = None,
    now: datetime | None = None,
    expected_avatar_fingerprint: str | None = None,
    expected_avatar_geometry: Rect | None = None,
    require_avatar_anchor: bool = True,
    account_guard=None,
) -> ScreenPixelCheck:
    """Detect changes after a manual request, never authorize new work.

    The caller may invoke this only while its explicit request or displayed
    result exists.  A mismatch/unavailable/expired frame must revoke that
    request or its copy surface.  Equal pixels must not renew the proof TTL,
    restore an invalid lease, confirm a customer, or dispatch any request.

    Production callers must freeze a reliable v2 avatar fingerprint from the
    target instance named by the trusted checkpoint, and its original frame
    geometry.  Missing anchors fail closed.  ``require_avatar_anchor=False``
    exists only for explicit legacy/synthetic pixel-only callers; it is never
    the production default and must not be used to authorize a copy or reply.

    A pure window translation is sampled using fresh native coordinates.  The
    original frame dimensions and layout stay fixed, and ownership/foreground
    and position are checked both before and after capture.
    """
    try:
        if account_guard is not None:
            account_reason = account_guard(result, state)
            if account_reason:
                return ScreenPixelCheck(False, account_reason)
        proof = result.screen_proof
        _validate_capture_fields(proof, now or datetime.now(timezone.utc))
        if (result.target_app != "wechat" or state.window.target.value != "wechat"
                or state.window.state.value != "visible" or not state.window.foreground
                or state.window.covered or state.paused or state.window.rect is None
                or not result.layout.input_boundary_verified):
            return ScreenPixelCheck(False, "target_not_available")
        owner = {
            "hwnd": result.hwnd, "process_id": result.process_id,
            "window_generation": result.window_generation,
            "account_identity_epoch": result.account_identity_epoch,
            "account_identity_runtime_key": result.account_identity_runtime_key,
        }
        live = {
            "hwnd": state.window.hwnd, "process_id": state.window.process_id,
            "window_generation": state.window_generation,
            "account_identity_epoch": state.account_identity_epoch,
            "account_identity_runtime_key": state.account_identity_runtime_key,
        }
        expected = dict(owner, source_job_id=result.job_id, snapshot_hash=result.snapshot_hash,
                        capture_generation=result.capture_generation,
                        account_scope_id=result.account_scope_id,
                        target_instance_id=result.target_instance_id,
                        title_exact_hash=result.title_exact_hash)
        if owner != live or any(proof.get(key) != value for key, value in expected.items()):
            return ScreenPixelCheck(False, "source_owner_changed")
        # A focus/translation capture generation can differ for this negative
        # check only. It is not relabelled on the original proof or result.
        if state.capture_generation < result.capture_generation:
            return ScreenPixelCheck(False, "source_generation_invalid")
        has_avatar_anchor = bool(
            isinstance(expected_avatar_fingerprint, str)
            and len(expected_avatar_fingerprint) == 67
            and expected_avatar_fingerprint.startswith("v2:")
            and all(char in "0123456789abcdef" for char in expected_avatar_fingerprint[3:])
            and isinstance(expected_avatar_geometry, Rect)
            and expected_avatar_geometry == account_avatar_screen_rect(result.layout)
        )
        if require_avatar_anchor and not has_avatar_anchor:
            return ScreenPixelCheck(False, "account_avatar_anchor_unavailable")
        if (expected_avatar_fingerprint is not None or expected_avatar_geometry is not None) and not has_avatar_anchor:
            return ScreenPixelCheck(False, "account_avatar_anchor_invalid")
        reader = native_window_reader or read_native_window
        before = reader(int(result.hwnd))
        if (before.hwnd != result.hwnd or before.process_id != result.process_id
                or not before.foreground or not before.visible or before.minimized
                or before.rect != state.window.rect.as_tuple()):
            return ScreenPixelCheck(False, "native_target_changed")
        left, top, right, bottom = before.rect
        if ((right - left, bottom - top)
                != (result.layout.window_rect.width, result.layout.window_rect.height)):
            return ScreenPixelCheck(False, "window_size_changed")
        client_error = _client_geometry_error(result.layout, state.window, before)
        if client_error:
            return ScreenPixelCheck(False, client_error)
        capture = sampler or (lambda rect: ImageGrab.grab(bbox=rect))
        image = capture(before.rect)
        after = reader(int(result.hwnd))
        if after != before:
            return ScreenPixelCheck(False, "native_target_changed_during_sample")
        client_error = _client_geometry_error(result.layout, state.window, after)
        if client_error:
            return ScreenPixelCheck(False, client_error)
        if has_avatar_anchor:
            avatar, reliable = avatar_fingerprint_from_frame(image, result.layout, expected_avatar_geometry)
            if not reliable:
                return ScreenPixelCheck(False, "account_avatar_sample_unreliable")
            if avatar != expected_avatar_fingerprint:
                return ScreenPixelCheck(False, "account_avatar_pixels_changed")
        title_digest, body_digest = exact_frame_digests(image, result.layout)
        if title_digest != result.title_exact_hash:
            return ScreenPixelCheck(False, "title_pixels_changed")
        snapshot = hashlib.sha256((
            f"hwnd={result.hwnd}|pid={result.process_id}|"
            f"window_generation={result.window_generation}|capture_generation={result.capture_generation}|"
            f"pixels={body_digest}"
        ).encode("ascii")).hexdigest()[:32]
        if snapshot != result.snapshot_hash:
            return ScreenPixelCheck(False, "message_pixels_changed")
        if account_guard is not None:
            account_reason = account_guard(result, state)
            if account_reason:
                return ScreenPixelCheck(False, account_reason)
            _validate_capture_fields(result.screen_proof, now or datetime.now(timezone.utc))
        return ScreenPixelCheck(True, "pixels_unchanged_negative_check_only")
    except Exception:
        # Native API errors, bad/missing proof, expired TTL and malformed image
        # sizes all revoke, never silently become permission to proceed.
        return ScreenPixelCheck(False, "pixel_check_unavailable")



def _copy_handoff_windows_valid(windows: object, source_hwnd: int) -> bool:
    """Only two caller-owned live Qt top-level handles, never a PID allowlist."""
    if type(windows) is not tuple or len(windows) > 2:
        return False
    seen: set[int] = set()
    for item in windows:
        if (type(item) is not tuple or len(item) != 2
                or any(type(value) is not int for value in item)):
            return False
        hwnd, pid = item
        if (not 0 < hwnd < 2**64 or not 0 < pid < 2**32 or pid != os.getpid()
                or hwnd == source_hwnd or hwnd in seen):
            return False
        seen.add(hwnd)
    return True


def _copy_native_sample_valid(sample: object) -> bool:
    if type(sample) is not NativeWindowSample:
        return False
    if (any(type(value) is not int or value <= 0 for value in (
            sample.hwnd, sample.process_id, sample.foreground_hwnd, sample.foreground_process_id))
            or any(type(value) is not bool for value in (sample.foreground, sample.visible, sample.minimized))
            or type(sample.rect) is not tuple or len(sample.rect) != 4
            or any(type(value) is not int for value in sample.rect)):
        return False
    left, top, right, bottom = sample.rect
    return (right > left and bottom > top
            and sample.foreground == (sample.foreground_hwnd == sample.hwnd)
            and (not sample.foreground or sample.foreground_process_id == sample.process_id))


def _copy_focus_source_error(result, state, now: datetime | None) -> str | None:
    """Validate original provenance without renewing it or trusting focus."""
    try:
        _validate_capture_fields(result.screen_proof, now or datetime.now(timezone.utc))
    except ValueError as error:
        return ("copy_source_expired" if str(error) == "current_view_screen_proof_expired"
                else "copy_source_proof_invalid")
    if (result.target_app != "wechat" or state.window.target.value != "wechat"
            or state.window.state.value != "visible" or state.paused
            or state.window.rect is None or not result.layout.input_boundary_verified):
        return "target_not_available"
    owner = {"hwnd": result.hwnd, "process_id": result.process_id,
             "window_generation": result.window_generation,
             "account_identity_epoch": result.account_identity_epoch,
             "account_identity_runtime_key": result.account_identity_runtime_key}
    live = {"hwnd": state.window.hwnd, "process_id": state.window.process_id,
            "window_generation": state.window_generation,
            "account_identity_epoch": state.account_identity_epoch,
            "account_identity_runtime_key": state.account_identity_runtime_key}
    expected = dict(owner, source_job_id=result.job_id, snapshot_hash=result.snapshot_hash,
                    capture_generation=result.capture_generation,
                    account_scope_id=result.account_scope_id,
                    target_instance_id=result.target_instance_id,
                    title_exact_hash=result.title_exact_hash)
    if owner != live or any(result.screen_proof.get(key) != value for key, value in expected.items()):
        return "source_owner_changed"
    if state.capture_generation < result.capture_generation:
        return "source_generation_invalid"
    return None


def _copy_focus_native_error(result, state, sample) -> str | None:
    if not _copy_native_sample_valid(sample):
        return "copy_native_sample_invalid"
    if sample.hwnd != result.hwnd or sample.process_id != result.process_id:
        return "copy_native_owner_changed"
    if not sample.visible or sample.minimized:
        return "copy_native_target_unavailable"
    if sample.rect != state.window.rect.as_tuple():
        return "copy_native_geometry_changed"
    if ((sample.rect[2] - sample.rect[0], sample.rect[3] - sample.rect[1])
            != (result.layout.window_rect.width, result.layout.window_rect.height)):
        return "window_size_changed"
    return _client_geometry_error(result.layout, state.window, sample)


def _copy_handoff_native_valid(sample, hwnd: int) -> bool:
    return bool(_copy_native_sample_valid(sample) and sample.hwnd == hwnd
                and sample.process_id == os.getpid() and sample.foreground
                and sample.visible and not sample.minimized)


def _request_copy_foreground(hwnd: int) -> None:
    # Only this one native request; never keys, AttachThreadInput, a restore,
    # a guessed window or an event pump. The actual result is polled later.
    import win32gui
    win32gui.SetForegroundWindow(hwnd)


def prepare_copy_target_foreground(
    result, state, *, allowed_handoff_windows: tuple[tuple[int, int], ...],
    reader=None, requester=None, now: datetime | None = None,
) -> CopyFocusPreparation:
    """Prepare one explicit copy, not permission to sample/copy/send.

    The caller must already hold the displayed customer's valid binding and
    must freeze ``native`` with its ticket. Only its exact visible Qt windows
    may hand foreground to the original HWND. No source field is modified.
    """
    try:
        if not _copy_handoff_windows_valid(allowed_handoff_windows, result.hwnd):
            return CopyFocusPreparation(False, False, "copy_handoff_windows_invalid")
        error = _copy_focus_source_error(result, state, now)
        if error:
            return CopyFocusPreparation(False, False, error)
        native_reader = reader or read_native_window
        before = native_reader(int(result.hwnd))
        error = _copy_focus_native_error(result, state, before)
        if error:
            return CopyFocusPreparation(False, False, error)
        if before.foreground:
            error = _copy_focus_source_error(result, state, now)
            if error:
                return CopyFocusPreparation(False, False, error)
            waiting = not state.window.foreground or state.window.covered
            return CopyFocusPreparation(True, waiting,
                "copy_focus_runtime_pending" if waiting else "copy_focus_ready", before)
        foreground = (before.foreground_hwnd, before.foreground_process_id)
        if foreground not in allowed_handoff_windows or foreground[1] != os.getpid():
            return CopyFocusPreparation(False, False, "copy_foreground_not_allowed")
        handoff = native_reader(before.foreground_hwnd)
        if not _copy_handoff_native_valid(handoff, before.foreground_hwnd):
            return CopyFocusPreparation(False, False, "copy_handoff_target_unavailable")
        # Recheck both target ownership and who actually owns foreground just
        # before the single request; a window switch during these reads fails.
        if native_reader(int(result.hwnd)) != before:
            return CopyFocusPreparation(False, False, "copy_native_changed_before_request")
        error = _copy_focus_source_error(result, state, now)
        if error:
            return CopyFocusPreparation(False, False, error)
        requested = (requester or _request_copy_foreground)(int(result.hwnd))
        if requested is not None and requested is not True:
            return CopyFocusPreparation(False, False, "copy_focus_request_failed")
        return CopyFocusPreparation(True, True, "copy_focus_requested", before)
    except Exception:
        return CopyFocusPreparation(False, False, "copy_focus_unavailable")


def copy_focus_poll(
    result, state, initial_native: NativeWindowSample, *, reader=None,
    now: datetime | None = None,
) -> CopyFocusPreparation:
    """One read-only poll of a preparation's immutable native anchor.

    A caller enforces the short monotonic deadline, consecutive ready samples,
    immutable suggestion ticket and all final DB/pixel/copy checks. This helper
    never retries focus, refreshes Runtime, samples pixels or extends the TTL.
    """
    try:
        error = _copy_focus_source_error(result, state, now)
        if error:
            return CopyFocusPreparation(False, False, error)
        if (not _copy_native_sample_valid(initial_native)
                or initial_native.hwnd != result.hwnd
                or initial_native.process_id != result.process_id
                or not initial_native.visible or initial_native.minimized
                or (not initial_native.foreground
                    and initial_native.foreground_process_id != os.getpid())):
            return CopyFocusPreparation(False, False, "copy_focus_anchor_invalid")
        native_reader = reader or read_native_window
        current = native_reader(int(result.hwnd))
        error = _copy_focus_native_error(result, state, current)
        if error:
            return CopyFocusPreparation(False, False, error)
        if (current.rect != initial_native.rect or current.client_rect != initial_native.client_rect):
            return CopyFocusPreparation(False, False, "copy_native_geometry_changed")
        if not current.foreground:
            if (initial_native.foreground
                    or (current.foreground_hwnd, current.foreground_process_id)
                       != (initial_native.foreground_hwnd, initial_native.foreground_process_id)):
                return CopyFocusPreparation(False, False, "copy_focus_stolen")
            handoff = native_reader(current.foreground_hwnd)
            if not _copy_handoff_native_valid(handoff, current.foreground_hwnd):
                return CopyFocusPreparation(False, False, "copy_handoff_target_unavailable")
            if native_reader(int(result.hwnd)) != current:
                return CopyFocusPreparation(False, False, "copy_native_changed_during_poll")
            error = _copy_focus_source_error(result, state, now)
            if error:
                return CopyFocusPreparation(False, False, error)
            return CopyFocusPreparation(True, True, "copy_focus_pending", current)
        error = _copy_focus_source_error(result, state, now)
        if error:
            return CopyFocusPreparation(False, False, error)
        waiting = not state.window.foreground or state.window.covered
        return CopyFocusPreparation(True, waiting,
            "copy_focus_runtime_pending" if waiting else "copy_focus_ready", current)
    except Exception:
        return CopyFocusPreparation(False, False, "copy_focus_unavailable")


def retained_photo_business_matches(saved, current, source, current_source, expected_hash: str) -> bool:
    """Compare retained photo advice only; never repair a dispatch context.

    The original display/DB lease is required before and after this comparison.
    A completed whole-window translation is permitted only with unchanged
    dimensions and client offsets. The final native/pixel checks still prove
    the original account, title and body at the current screen coordinates.
    """
    try:
        from dataclasses import replace
        import json
        from whochat.ai.current_work_authorization import current_context_source_error
        from whochat.ai.photos import bind_reply_photos, photo_source_error
        from whochat.core.current_conversation import CurrentMessageView
        from whochat.services.reply import context_hash

        if (source is None or source is not current_source or not saved.photos
                or saved.photo_error or current.photos or current.photo_error != "photo_source_changed"
                or type(saved.current_message_view) is not CurrentMessageView
                or type(current.current_message_view) is not CurrentMessageView
                or current.current_message_view != saved.current_message_view
                or saved.contact is None or current.contact is None
                or saved.contact.id != current.contact.id
                or saved.contact.account_scope_id != current.contact.account_scope_id
                or current_context_source_error(saved) is not None
                or current_context_source_error(current) not in {None, "current_view_screen_runtime_mismatch"}
                or photo_source_error(saved) is not None):
            return False
        original, live = saved.runtime, current.runtime
        if (original.window.target.value != "wechat" or live.window.target.value != "wechat"
                or original.window.state.value != "visible" or live.window.state.value != "visible"
                or original.paused or live.paused
                or not isinstance(original.window.rect, Rect) or not isinstance(live.window.rect, Rect)
                or (original.window.rect.width, original.window.rect.height)
                   != (live.window.rect.width, live.window.rect.height)
                or source.capture_generation != original.capture_generation
                or live.capture_generation <= original.capture_generation):
            return False
        # Geometry is not customer identity. Keep native owner/epoch/window
        # generations exact and allow only a completed same-size translation.
        for key in ("window_generation", "account_identity_epoch", "account_identity_runtime_key"):
            if getattr(original, key) != getattr(live, key) or getattr(source, key) != getattr(original, key):
                return False
        for key in ("hwnd", "process_id"):
            if getattr(original.window, key) != getattr(live.window, key) or getattr(source, key) != getattr(original.window, key):
                return False
        original_client, live_client = original.window.client_rect, live.window.client_rect
        if original_client is None:
            if live_client is not None or source.layout.client_rect is not None:
                return False
        else:
            if not isinstance(original_client, Rect) or not isinstance(live_client, Rect):
                return False
            dx = live.window.rect.left - original.window.rect.left
            dy = live.window.rect.top - original.window.rect.top
            if (live_client.as_tuple() != (original_client.left + dx, original_client.top + dy,
                                          original_client.right + dx, original_client.bottom + dy)
                    or source.layout.client_rect != original_client):
                return False
        if _copy_focus_source_error(source, original, None):
            return False
        view = saved.current_message_view
        proof = json.loads(view.screen_evidence_json)
        _validate_capture_fields(proof, datetime.now(timezone.utc))
        if (source.account_scope_id != view.account_scope_id
                or source.target_instance_id != view.target_instance_id
                or source.job_id < view.source_job_id
                or source.capture_generation != view.capture_generation
                or any(proof.get(key) != getattr(source, key) for key in (
                    "hwnd", "process_id", "window_generation", "capture_generation",
                    "account_identity_epoch", "account_identity_runtime_key", "snapshot_hash", "title_exact_hash"))):
            return False
        rebound, error = bind_reply_photos(view, source, original)
        if error or rebound != saved.photos or context_hash(saved) != expected_hash:
            return False
        # Comparison-only projection. It is not returned, cached, persisted or
        # dispatched; the real current generation context stays strictly invalid.
        return context_hash(replace(current, photos=saved.photos, photo_error="")) == expected_hash
    except Exception:
        return False


def check_copy_screen_pixels(
    result,
    state,
    *,
    allowed_handoff_windows: tuple[tuple[int, int], ...],
    sampler: Callable[[tuple[int, int, int, int]], Image.Image] | None = None,
    native_window_reader: Callable[[int], NativeWindowSample] | None = None,
    now: datetime | None = None,
    expected_avatar_fingerprint: str | None = None,
    expected_avatar_geometry: Rect | None = None,
    require_avatar_anchor: bool = True,
    account_guard=None,
) -> ScreenPixelCheck:
    """Copy-only negative check after the caller's actual display/source lease.

    A click may put this Python process's exact MainWindow/floating HWND in
    front of WeChat. Only that explicit local handoff can bypass stale Runtime
    foreground/coverage flags. Native ownership, visibility, geometry and the
    actual foreground HWND/PID must be unchanged on BOTH sides of the sample.
    Avatar/title/body must still match the original unexpired capture proof.

    This never changes Runtime/proof, renews TTL, reads/stores a file, confirms
    identity, grants a lease or starts OCR/AI. A True result alone is NOT copy
    authorization. The caller must first validate its displayed result binding,
    latest database source/account lifecycle and unchanged business context,
    then call this immediately before writing the requested suggestion text.
    """
    try:
        if account_guard is not None:
            account_reason = account_guard(result, state)
            if account_reason:
                return ScreenPixelCheck(False, account_reason)
        if not _copy_handoff_windows_valid(allowed_handoff_windows, result.hwnd):
            return ScreenPixelCheck(False, "copy_handoff_windows_invalid")
        if require_avatar_anchor is not True:
            return ScreenPixelCheck(False, "copy_avatar_anchor_required")
        proof = result.screen_proof
        try:
            _validate_capture_fields(proof, now or datetime.now(timezone.utc))
        except ValueError as error:
            reason = ("copy_source_expired" if str(error) == "current_view_screen_proof_expired"
                      else "copy_source_proof_invalid")
            return ScreenPixelCheck(False, reason)
        if (result.target_app != "wechat" or state.window.target.value != "wechat"
                or state.window.state.value != "visible" or state.paused or state.window.rect is None
                or not result.layout.input_boundary_verified):
            return ScreenPixelCheck(False, "target_not_available")
        owner = {
            "hwnd": result.hwnd, "process_id": result.process_id,
            "window_generation": result.window_generation,
            "account_identity_epoch": result.account_identity_epoch,
            "account_identity_runtime_key": result.account_identity_runtime_key,
        }
        live = {
            "hwnd": state.window.hwnd, "process_id": state.window.process_id,
            "window_generation": state.window_generation,
            "account_identity_epoch": state.account_identity_epoch,
            "account_identity_runtime_key": state.account_identity_runtime_key,
        }
        expected = dict(owner, source_job_id=result.job_id, snapshot_hash=result.snapshot_hash,
                        capture_generation=result.capture_generation, account_scope_id=result.account_scope_id,
                        target_instance_id=result.target_instance_id, title_exact_hash=result.title_exact_hash)
        if owner != live or any(proof.get(key) != value for key, value in expected.items()):
            return ScreenPixelCheck(False, "source_owner_changed")
        if state.capture_generation < result.capture_generation:
            return ScreenPixelCheck(False, "source_generation_invalid")
        if not (isinstance(expected_avatar_fingerprint, str)
                and len(expected_avatar_fingerprint) == 67 and expected_avatar_fingerprint.startswith("v2:")
                and all(char in "0123456789abcdef" for char in expected_avatar_fingerprint[3:])
                and isinstance(expected_avatar_geometry, Rect)
                and expected_avatar_geometry == account_avatar_screen_rect(result.layout)):
            return ScreenPixelCheck(False, "account_avatar_anchor_invalid")
        reader = native_window_reader or read_native_window
        before = reader(int(result.hwnd))
        if not _copy_native_sample_valid(before):
            return ScreenPixelCheck(False, "copy_native_sample_invalid")
        if before.hwnd != result.hwnd or before.process_id != result.process_id:
            return ScreenPixelCheck(False, "copy_native_owner_changed")
        if not before.visible or before.minimized:
            return ScreenPixelCheck(False, "copy_native_target_unavailable")
        handoff = (not before.foreground
                   and (before.foreground_hwnd, before.foreground_process_id) in allowed_handoff_windows
                   and before.foreground_process_id == os.getpid())
        if not before.foreground and not handoff:
            return ScreenPixelCheck(False, "copy_foreground_not_allowed")
        if (not state.window.foreground or state.window.covered) and not handoff:
            return ScreenPixelCheck(False, "target_not_available")
        if before.rect != state.window.rect.as_tuple():
            return ScreenPixelCheck(False, "copy_native_geometry_changed")
        left, top, right, bottom = before.rect
        if ((right - left, bottom - top)
                != (result.layout.window_rect.width, result.layout.window_rect.height)):
            return ScreenPixelCheck(False, "window_size_changed")
        client_error = _client_geometry_error(result.layout, state.window, before)
        if client_error:
            return ScreenPixelCheck(False, client_error)
        capture = sampler or (lambda rect: ImageGrab.grab(bbox=rect))
        image = capture(before.rect)
        after = reader(int(result.hwnd))
        if not _copy_native_sample_valid(after) or after != before:
            return ScreenPixelCheck(False, "copy_native_changed_during_sample")
        client_error = _client_geometry_error(result.layout, state.window, after)
        if client_error:
            return ScreenPixelCheck(False, client_error)
        avatar, reliable = avatar_fingerprint_from_frame(image, result.layout, expected_avatar_geometry)
        if not reliable:
            return ScreenPixelCheck(False, "account_avatar_sample_unreliable")
        if avatar != expected_avatar_fingerprint:
            return ScreenPixelCheck(False, "account_avatar_pixels_changed")
        title_digest, body_digest = exact_frame_digests(image, result.layout)
        if title_digest != result.title_exact_hash:
            return ScreenPixelCheck(False, "title_pixels_changed", _copy_pixel_difference(result, image))
        snapshot = hashlib.sha256((
            f"hwnd={result.hwnd}|pid={result.process_id}|"
            f"window_generation={result.window_generation}|capture_generation={result.capture_generation}|"
            f"pixels={body_digest}"
        ).encode("ascii")).hexdigest()[:32]
        if snapshot != result.snapshot_hash:
            return ScreenPixelCheck(False, "message_pixels_changed", _copy_pixel_difference(result, image))
        # Copy must not slip past the source TTL while the capture/hash work
        # was running. This rereads the original proof, never extends it.
        try:
            _validate_capture_fields(proof, now or datetime.now(timezone.utc))
        except ValueError as error:
            reason = ("copy_source_expired" if str(error) == "current_view_screen_proof_expired"
                      else "copy_source_proof_invalid")
            return ScreenPixelCheck(False, reason)
        if account_guard is not None:
            account_reason = account_guard(result, state)
            if account_reason:
                return ScreenPixelCheck(False, account_reason)
            _validate_capture_fields(result.screen_proof, now or datetime.now(timezone.utc))
        return ScreenPixelCheck(True, "copy_pixels_unchanged_negative_check_only")
    except Exception:
        return ScreenPixelCheck(False, "copy_pixel_check_unavailable")



def _copy_pixel_difference(result, image) -> dict:
    """Numeric diagnostics only; the exact authorization verdict is already false."""
    try:
        from whochat.services.frame_diagnostics import compare_signature
        return compare_signature(getattr(result, "frame_diagnostic", None), image,
            result.layout, result.snapshot_hash, result.job_id,
            title_rect=title_identity_screen_rect(result.layout))
    except Exception:
        return {"available": False, "reason": "diagnostic_unavailable"}
