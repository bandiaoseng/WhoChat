from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from whochat.core.models import utc_now_iso


class TargetApp(StrEnum):
    WECHAT = "wechat"
    GENERIC_CHAT = "generic_chat"
    UNKNOWN = "unknown"


class WindowState(StrEnum):
    MISSING = "missing"
    VISIBLE = "visible"
    MINIMIZED = "minimized"
    MOVING = "moving"
    RESIZING = "resizing"
    UNAVAILABLE = "unavailable"


class PageType(StrEnum):
    CHAT_DM = "chat_dm"
    CHAT_GROUP = "chat_group"
    FILE_HELPER = "file_helper"
    OFFICIAL_ACCOUNT = "official_account"
    NEWS_ARTICLE = "news_article"
    FILE_PREVIEW = "file_preview"
    MINI_PROGRAM = "mini_program"
    SEARCH = "search"
    SETTINGS = "settings"
    MEDIA_VIEWER = "media_viewer"
    UNKNOWN = "unknown"


class RegionSource(StrEnum):
    AUTO = "auto"
    CALIBRATED = "calibrated"
    MIXED = "mixed"
    UNAVAILABLE = "unavailable"


class ThemeMode(StrEnum):
    LIGHT = "light"
    DARK = "dark"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_tuple(cls, value: tuple[int, int, int, int]) -> Rect:
        return cls(*value)

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def inset(self, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> Rect:
        return Rect(self.left + left, self.top + top, self.right - right, self.bottom - bottom)


@dataclass(frozen=True)
class WindowSnapshot:
    target: TargetApp
    hwnd: int | None
    title: str
    process_name: str
    rect: Rect | None
    state: WindowState
    process_id: int | None = None
    executable_path: str = ""
    app_label: str = "微信"
    diagnostic: str = ""
    foreground: bool = True
    covered: bool = False
    bubble_profile: str = "auto"
    observed_at: str = field(default_factory=utc_now_iso)
    # Absolute screen coordinates; rect remains the native outer frame.
    client_rect: Rect | None = None
    dpi_scale: float = 1.0


@dataclass(frozen=True)
class LayoutRegions:
    target_app: TargetApp
    bubble_profile: str
    window_rect: Rect
    nav_rect: Rect
    chat_list_rect: Rect
    content_rect: Rect
    title_rect: Rect
    message_rect: Rect
    input_rect: Rect
    confidence: float
    source: RegionSource
    reason: str
    # A ratio prior is not evidence. Either measured screenshot geometry or
    # an explicit live user selection bound to a manual request is required.
    input_boundary_verified: bool = False
    client_rect: Rect | None = None
    dpi_scale: float = 1.0

    # Nonempty only for the dedicated live manual-selection pipeline. Never
    # persisted as a calibration or reused by background automatic capture.
    manual_selection_id: str = ""


def constrain_layout_to_client(layout: LayoutRegions, client_rect: Rect | None) -> LayoutRegions:
    """Exclude only OS-proven non-client pixels without rebasing calibration.

    The screenshot origin, outer frame and navigation/avatar geometry retain
    their existing contract. Unknown legacy/synthetic geometry grants no new
    exclusion; invalid supplied geometry fails closed rather than expanding.
    """
    if client_rect is None:
        return layout
    outer = layout.window_rect
    if (not isinstance(client_rect, Rect) or client_rect.width <= 0 or client_rect.height <= 0
            or client_rect.left < outer.left or client_rect.top < outer.top
            or client_rect.right > outer.right or client_rect.bottom > outer.bottom):
        raise ValueError("client_geometry_invalid")
    updates = {}
    for name in ("content_rect", "title_rect", "message_rect", "input_rect"):
        region = getattr(layout, name)
        clipped = Rect(max(region.left, client_rect.left), max(region.top, client_rect.top),
                       min(region.right, client_rect.right), min(region.bottom, client_rect.bottom))
        if clipped.width <= 0 or clipped.height <= 0:
            raise ValueError(f"client_geometry_empty_{name}")
        updates[name] = clipped
    return replace(layout, client_rect=client_rect, **updates)


@dataclass(frozen=True)
class RelativeRect:
    left: float
    top: float
    right: float
    bottom: float

    def clamp(self) -> RelativeRect:
        left = _clamp_float(self.left, 0.0, 1.0)
        top = _clamp_float(self.top, 0.0, 1.0)
        right = _clamp_float(self.right, left, 1.0)
        bottom = _clamp_float(self.bottom, top, 1.0)
        return RelativeRect(left, top, right, bottom)

    def to_absolute(self, window: Rect) -> Rect:
        rect = self.clamp()
        return Rect(
            window.left + round(window.width * rect.left),
            window.top + round(window.height * rect.top),
            window.left + round(window.width * rect.right),
            window.top + round(window.height * rect.bottom),
        )

    @classmethod
    def from_absolute(cls, window: Rect, rect: Rect) -> RelativeRect:
        width = max(1, window.width)
        height = max(1, window.height)
        return cls(
            (rect.left - window.left) / width,
            (rect.top - window.top) / height,
            (rect.right - window.left) / width,
            (rect.bottom - window.top) / height,
        ).clamp()


@dataclass(frozen=True)
class LayoutCalibration:
    id: str
    target: TargetApp
    name: str
    theme: ThemeMode
    dpi_scale: float
    nav_rect: RelativeRect
    chat_list_rect: RelativeRect
    content_rect: RelativeRect
    title_rect: RelativeRect
    message_rect: RelativeRect
    input_rect: RelativeRect
    active: bool
    created_at: str
    updated_at: str


def layout_from_calibration(calibration: LayoutCalibration, window_rect: Rect) -> LayoutRegions:
    return LayoutRegions(
        target_app=calibration.target,
        bubble_profile="wechat_green" if calibration.target == TargetApp.WECHAT else "geometry",
        window_rect=window_rect,
        nav_rect=calibration.nav_rect.to_absolute(window_rect),
        chat_list_rect=calibration.chat_list_rect.to_absolute(window_rect),
        content_rect=calibration.content_rect.to_absolute(window_rect),
        title_rect=calibration.title_rect.to_absolute(window_rect),
        message_rect=calibration.message_rect.to_absolute(window_rect),
        input_rect=calibration.input_rect.to_absolute(window_rect),
        confidence=0.9,
        source=RegionSource.CALIBRATED,
        reason=f"使用用户校准布局：{calibration.name}",
    )


@dataclass(frozen=True)
class PageClassification:
    page_type: PageType
    confidence: float
    reason: str

    @property
    def can_generate_reply(self) -> bool:
        return self.page_type in {PageType.CHAT_DM, PageType.CHAT_GROUP, PageType.FILE_HELPER} and self.confidence >= 0.65


@dataclass(frozen=True)
class CapturePolicy:
    window_stable_delay_ms: int = 300
    scroll_debounce_ms: int = 500
    screenshot_min_interval_ms: int = 800
    ocr_min_interval_ms: int = 8000
    ai_min_interval_ms: int = 3000
    min_hash_distance: int = 4


@dataclass(frozen=True)
class CaptureDecision:
    should_capture: bool
    reason: str
    snapshot_hash: str | None = None
    elapsed_ms: int | None = None


@dataclass(frozen=True)
class RuntimeState:
    window: WindowSnapshot
    layout: LayoutRegions | None
    page: PageClassification
    capture_decision: CaptureDecision
    paused: bool
    ai_pending: bool = False
    ocr_pending: bool = False
    last_snapshot_hash: str | None = None
    visible_message_count: int = 0
    pipeline_status: str = "idle"
    window_generation: int = 0
    capture_generation: int = 0
    account_identity_epoch: int = 0
    account_identity_runtime_key: str = ""

    @property
    def status_label(self) -> str:
        if self.paused:
            return "paused"
        if self.window.state != WindowState.VISIBLE:
            return self.window.state.value
        if not self.layout:
            return "layout_unavailable"
        if not self.page.can_generate_reply:
            return f"blocked:{self.page.page_type.value}"
        return "ready"


def missing_runtime_state(reason: str = "未发现目标窗口") -> RuntimeState:
    window = WindowSnapshot(
        target=TargetApp.WECHAT,
        hwnd=None,
        title="",
        process_name="",
        rect=None,
        state=WindowState.MISSING,
        bubble_profile="auto",
    )
    return RuntimeState(
        window=window,
        layout=None,
        page=PageClassification(PageType.UNKNOWN, 0.0, reason),
        capture_decision=CaptureDecision(False, reason),
        paused=False,
    )


def account_identity_runtime_key(window: WindowSnapshot) -> str:
    """Return the runtime lifecycle that owns an account identity epoch."""

    executable = (window.executable_path or window.process_name or "unknown").strip().casefold()
    return f"{window.target.value}|{executable}|pid:{window.process_id or 0}|hwnd:{window.hwnd or 0}"


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
