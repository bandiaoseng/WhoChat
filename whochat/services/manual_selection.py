"""One explicit, live user-selected viewport; never a saved-image import.

The user supplies the message boundary. Account, title, text completeness,
independent current captures, dispatch and copy checks remain mandatory.
No functions here invoke AI, ingest messages, or send/modify WeChat content.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import uuid

from whochat.core.runtime import LayoutRegions, Rect, RegionSource


class SelectionError(ValueError):
    pass


def _inside(rect, parent):
    return (isinstance(rect, Rect) and rect.width > 0 and rect.height > 0
            and all(type(x) is int for x in rect.as_tuple())
            and parent.left <= rect.left < rect.right <= parent.right
            and parent.top <= rect.top < rect.bottom <= parent.bottom)


def selection_error_message(reason):
    if 'account' in reason or 'owner' in reason:
        return '截图来源的工作微信尚未核实或已经变化，请回到目标客户重新截图；本次未调用 AI。'
    if 'title' in reason:
        return '请把顶部客户名称一起框入，框选下缘停在打字输入框上方。'
    if 'small' in reason or 'bounds' in reason:
        return '请框入客户名称和双方完整聊天，左右保留头像，底部不要框入打字输入框。'
    if 'pixels' in reason or 'geometry' in reason:
        return '框选后聊天画面或窗口已变化，请重新截图；本次未调用 AI。'
    return '未能取得本次微信截图，请保持客户窗口清晰可见后重新截图；本次未调用 AI。'


@dataclass(frozen=True)
class SelectionFrame:
    state: object
    image: object
    native_token: tuple


@dataclass(frozen=True)
class ManualSelection:
    selection_id: str
    hwnd: int
    process_id: int
    window_generation: int
    native_token: tuple
    layout: LayoutRegions
    title_digest: str
    body_digest: str
    avatar_digest: str

    def for_capture(self, state, image_path, native_token):
        """Check the new independent frame against the original user choice.

        A translation is permitted; a resize/changed DPI requires a new choice.
        Nothing updates the original hashes or extends a source proof lifetime.
        """
        from PIL import Image
        from whochat.services.manual_suggestions import exact_frame_digests
        window = state.window
        if ((window.hwnd, window.process_id, state.window_generation)
                != (self.hwnd, self.process_id, self.window_generation)
                or not self.native_token or native_token != self.native_token):
            raise SelectionError('manual_selection_owner_changed')
        old, current = self.layout.window_rect, window.rect
        if (current is None or current.width != old.width or current.height != old.height
                or window.dpi_scale != self.layout.dpi_scale):
            raise SelectionError('manual_selection_geometry_changed')
        dx, dy = current.left-old.left, current.top-old.top
        def moved(rect):
            return Rect(rect.left+dx, rect.top+dy, rect.right+dx, rect.bottom+dy)
        layout = replace(self.layout, **{name: moved(getattr(self.layout, name)) for name in (
            'window_rect', 'nav_rect', 'chat_list_rect', 'content_rect', 'title_rect',
            'message_rect', 'input_rect', 'client_rect')})
        if layout.client_rect != window.client_rect:
            raise SelectionError('manual_selection_geometry_changed')
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        try:
            title, body = exact_frame_digests(image, layout)
            avatar = _avatar_digest(image, layout)
        finally:
            image.close()
        if (title, body, avatar) != (self.title_digest, self.body_digest, self.avatar_digest):
            raise SelectionError('manual_selection_pixels_changed')
        return layout


def _avatar_digest(image, layout):
    from whochat.services.account_scope import account_avatar_screen_rect
    rect = account_avatar_screen_rect(layout)
    outer = layout.window_rect
    if not _inside(rect, outer):
        raise SelectionError('manual_selection_account_geometry')
    crop = image.crop((rect.left-outer.left, rect.top-outer.top,
                       rect.right-outer.left, rect.bottom-outer.top)).convert('RGB')
    return hashlib.sha256(crop.tobytes()).hexdigest()


def build_selection(frame: SelectionFrame, chosen: Rect) -> ManualSelection:
    """Build geometry only from the user's rectangle and the title band.

    No input separator, closed-panel detector, or automatic message-area
    refinement is called. The existing title-only measurement may refine the
    header; current title OCR must still confirm the customer independently.
    """
    from whochat.vision.layout_refine import _refine_native_title_boundary
    from whochat.vision.visual import VisualContext
    from whochat.services.manual_suggestions import exact_frame_digests
    state, image = frame.state, frame.image
    window, base = state.window, state.layout
    client = window.client_rect
    if (base is None or window.target.value != 'wechat' or client is None
            or window.rect is None or not _inside(client, window.rect)
            or not _inside(chosen, client) or not frame.native_token):
        raise SelectionError('manual_selection_bounds_invalid')
    scale = window.dpi_scale
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
            or not math.isfinite(scale) or not .75 <= scale <= 3.0
            or image.size != (window.rect.width, window.rect.height)):
        raise SelectionError('manual_selection_geometry_invalid')
    if chosen.width < round(240*scale) or chosen.height < round(140*scale):
        raise SelectionError('manual_selection_too_small')
    if (chosen.top > client.top+round(48*scale)
            or chosen.left < client.left+round(100*scale)):
        raise SelectionError('manual_selection_title_missing')
    # WeChat's native client includes its own title strip. This bounded prior
    # is independent of window height; the existing header-only pixel proof
    # refines it where present. It grants no customer identity by itself.
    title_bottom = client.top+round(80*scale)
    if title_bottom >= chosen.bottom-round(60*scale) or chosen.bottom >= client.bottom:
        raise SelectionError('manual_selection_too_small')
    layout = replace(base, window_rect=window.rect, client_rect=client, dpi_scale=scale,
        content_rect=Rect(chosen.left, chosen.top, chosen.right, chosen.bottom),
        title_rect=Rect(chosen.left, chosen.top, chosen.right, title_bottom),
        message_rect=Rect(chosen.left, title_bottom, chosen.right, chosen.bottom),
        input_rect=Rect(chosen.left, chosen.bottom, chosen.right, client.bottom),
        chat_list_rect=Rect(base.chat_list_rect.left, client.top, chosen.left, client.bottom),
        source=RegionSource.AUTO, input_boundary_verified=False)
    visual = VisualContext.load(image)
    if visual is not None:
        layout = _refine_native_title_boundary(visual, layout)
    selection_id = uuid.uuid4().hex
    layout = replace(layout, source=RegionSource.CALIBRATED, input_boundary_verified=True,
        manual_selection_id=selection_id, reason='用户本次明确框选的聊天范围')
    title, body = exact_frame_digests(image, layout)
    return ManualSelection(selection_id, window.hwnd, window.process_id,
        state.window_generation, frame.native_token, layout, title, body, _avatar_digest(image, layout))


def capture_selection_frame(state, resolver, *, reader=None, sampler=None):
    """Capture before the overlay opens, bound to fresh native account reads."""
    from whochat.services.manual_suggestions import read_native_window
    from PIL import ImageGrab
    reader = reader or read_native_window
    sampler = sampler or (lambda rect: ImageGrab.grab(bbox=rect, all_screens=True))
    window = state.window
    if (window.rect is None or window.client_rect is None or window.target.value != 'wechat'
            or state.paused or window.covered or not window.foreground):
        raise SelectionError('manual_selection_target_unavailable')
    before = reader(window.hwnd)
    if (before.hwnd != window.hwnd or before.process_id != window.process_id
            or not before.foreground or not before.visible or before.minimized
            or before.rect != window.rect.as_tuple()
            or before.client_rect != window.client_rect.as_tuple()):
        raise SelectionError('manual_selection_geometry_changed')
    token = resolver.capture_identity_token(window)
    if not token:
        raise SelectionError('manual_selection_account_unverified')
    image = sampler(before.rect).convert('RGB')
    try:
        reason = resolver.capture_identity_reason(window, token)
        if reason:
            raise SelectionError('manual_selection_account_changed')
        if reader(window.hwnd) != before:
            raise SelectionError('manual_selection_geometry_changed')
        if image.size != (window.rect.width, window.rect.height):
            raise SelectionError('manual_selection_geometry_changed')
        return SelectionFrame(state, image, token)
    except Exception:
        image.close()
        raise
