from __future__ import annotations

from dataclasses import dataclass
import time

from whochat.config import TargetWindowConfig, default_target_windows


_PROCESS_NAME_CACHE: dict[int, str] = {}
_PROCESS_NAME_CACHE_EXPIRES_AT = 0.0
_PROCESS_NAME_CACHE_TTL_SECONDS = 2.5


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    process_name: str
    rect: tuple[int, int, int, int]
    visible: bool
    target_app: str = "wechat"
    app_label: str = "微信"
    process_id: int | None = None
    minimized: bool = False
    diagnostic: str = ""
    foreground: bool = True
    bubble_profile: str = "auto"
    covered: bool = False
    executable_path: str = ""
    client_rect: tuple[int, int, int, int] | None = None
    dpi_scale: float = 1.0


@dataclass(frozen=True)
class WindowMatchDiagnostic:
    hwnd: int
    title: str
    process_name: str
    process_id: int | None
    matched: bool
    target_app: str
    app_label: str
    reason: str
    foreground: bool = False


def find_wechat_windows() -> list[WindowInfo]:
    targets = [target for target in default_target_windows() if target.app_id == "wechat"]
    return find_target_windows(targets)


def read_client_screen_rect(
    hwnd: int,
    expected_window_rect: tuple[int, int, int, int],
    expected_process_id: int | None = None,
) -> tuple[int, int, int, int]:
    """Read a stable client's screen bounds for this exact native owner.

    Native failures propagate. No guessed inset, screen capture, focus change
    or coordinate-system mutation is permitted in this geometry proof.
    """
    import win32gui
    import win32process

    expected = tuple(int(value) for value in expected_window_rect)
    before = tuple(int(value) for value in win32gui.GetWindowRect(hwnd))
    process_before = int(win32process.GetWindowThreadProcessId(hwnd)[1])
    if (len(expected) != 4 or before != expected or process_before <= 0
            or (expected_process_id is not None and process_before != expected_process_id)):
        raise ValueError("client_geometry_owner_changed")
    client = tuple(int(value) for value in win32gui.GetClientRect(hwnd))
    if len(client) != 4 or client[2] <= client[0] or client[3] <= client[1]:
        raise ValueError("client_geometry_empty")
    left, top = win32gui.ClientToScreen(hwnd, (client[0], client[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (client[2], client[3]))
    after = tuple(int(value) for value in win32gui.GetWindowRect(hwnd))
    process_after = int(win32process.GetWindowThreadProcessId(hwnd)[1])
    if before != after or process_after != process_before:
        raise ValueError("client_geometry_changed_during_read")
    result = (int(left), int(top), int(right), int(bottom))
    if (result[2] <= result[0] or result[3] <= result[1]
            or result[0] < expected[0] or result[1] < expected[1]
            or result[2] > expected[2] or result[3] > expected[3]):
        raise ValueError("client_geometry_outside_window")
    return result


def find_target_windows(targets: list[TargetWindowConfig]) -> list[WindowInfo]:
    try:
        import win32gui
        import win32process
    except ImportError:
        return []

    windows: list[WindowInfo] = []
    enabled_targets = [target for target in targets if target.enabled]
    process_names_by_pid: dict[int, str] = {}
    foreground_hwnd = foreground_window_handle()

    def enum_handler(hwnd: int, _extra) -> None:
        try:
            visible = bool(win32gui.IsWindowVisible(hwnd))
            if not visible:
                return
            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return
            _thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
            if process_id not in process_names_by_pid:
                process_names_by_pid[process_id] = _process_name_by_pid(process_id)
            process_name = process_names_by_pid[process_id]
            target = _match_target(enabled_targets, title, process_name)
            if target is None:
                return
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            window_class = str(win32gui.GetClassName(hwnd) or "")
            window_style = int(win32gui.GetWindowLong(hwnd, -16))
            owner = int(win32gui.GetWindow(hwnd, 4) or 0)
            if not _is_canonical_capture_window(
                target,
                title=title,
                window_class=window_class,
                rect=(left, top, right, bottom),
                style=window_style,
                owner=owner,
            ):
                return
            minimized = bool(win32gui.IsIconic(hwnd))
            foreground = foreground_hwnd is None or hwnd == foreground_hwnd
            covered = _window_is_covered(hwnd, (left, top, right, bottom))
            client_rect = None
            client_unavailable = False
            if target.app_id == "wechat" and not minimized:
                try:
                    client_rect = read_client_screen_rect(hwnd, (left, top, right, bottom), process_id)
                except Exception:
                    # Retain the canonical owner, but never capture an unknown
                    # client as if the whole outer frame were trusted content.
                    client_unavailable = True
                    covered = True
            windows.append(
                WindowInfo(
                    hwnd=hwnd,
                    title=title,
                    process_name=process_name,
                    rect=(left, top, right, bottom),
                    # Visibility is a native window property.  Losing focus
                    # does not make a window invisible and must not be treated
                    # as proof that another window covers it.
                    visible=visible and not minimized and not client_unavailable,
                    target_app=target.app_id,
                    app_label=target.label,
                    process_id=process_id,
                    minimized=minimized,
                    diagnostic=("客户区几何不可用或已变化，已暂停采集" if client_unavailable else
                                _window_diagnostic(title, process_name, minimized, foreground, covered)),
                    foreground=foreground,
                    bubble_profile=target.bubble_profile,
                    covered=covered,
                    executable_path=_process_executable_by_pid(process_id),
                    client_rect=client_rect,
                    dpi_scale=read_window_dpi_scale(hwnd),
                )
            )
        except Exception:
            return

    try:
        win32gui.EnumWindows(enum_handler, None)
    except Exception:
        return windows
    return windows


def read_window_dpi_scale(hwnd: int) -> float:
    """Read native UI scaling, never infer it from a resizable window's size."""
    try:
        import ctypes
        from ctypes import wintypes
        reader = ctypes.WinDLL("user32", use_last_error=True).GetDpiForWindow
        reader.argtypes = [wintypes.HWND]
        reader.restype = wintypes.UINT
        dpi = int(reader(hwnd))
        if 72 <= dpi <= 384:
            return dpi / 96.0
    except (AttributeError, OSError, ValueError):
        pass
    # Legacy/non-Windows synthetic geometry uses the established 96-DPI grid.
    # This supplies no identity authority; all visual checks still apply.
    return 1.0


def diagnose_target_windows(targets: list[TargetWindowConfig], limit: int = 30) -> list[WindowMatchDiagnostic]:
    try:
        import win32gui
        import win32process
    except ImportError:
        return []

    enabled_targets = [target for target in targets if target.enabled]
    process_names_by_pid: dict[int, str] = {}
    diagnostics: list[WindowMatchDiagnostic] = []
    foreground_hwnd = foreground_window_handle()

    def enum_handler(hwnd: int, _extra) -> None:
        if len(diagnostics) >= limit:
            return
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return
            _thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
            if process_id not in process_names_by_pid:
                process_names_by_pid[process_id] = _process_name_by_pid(process_id)
            process_name = process_names_by_pid[process_id]
            target = _match_target(enabled_targets, title, process_name)
            raw_target = _match_target_raw(enabled_targets, title, process_name)
            related = target is not None or raw_target is not None or _looks_related_to_targets(title, process_name, enabled_targets)
            if not related:
                return
            diagnostics.append(
                WindowMatchDiagnostic(
                    hwnd=hwnd,
                    title=title,
                    process_name=process_name,
                    process_id=process_id,
                    matched=target is not None,
                    target_app=target.app_id if target else raw_target.app_id if raw_target else "",
                    app_label=target.label if target else raw_target.label if raw_target else "",
                    reason=_match_reason(title, process_name, target, raw_target),
                    foreground=foreground_hwnd is not None and hwnd == foreground_hwnd,
                )
            )
        except Exception:
            return

    try:
        win32gui.EnumWindows(enum_handler, None)
    except Exception:
        return diagnostics
    return diagnostics


def foreground_window_handle() -> int | None:
    try:
        import win32gui
    except ImportError:
        return None
    try:
        hwnd = win32gui.GetForegroundWindow()
    except Exception:
        return None
    return int(hwnd) if hwnd else None


def window_process_id(hwnd: int | None) -> int | None:
    """Return the native process id without treating a popup as a target."""
    if not hwnd:
        return None
    try:
        import win32process

        _thread_id, process_id = win32process.GetWindowThreadProcessId(int(hwnd))
    except Exception:
        return None
    return int(process_id) if process_id else None


def _target_process_names(targets: list[TargetWindowConfig]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        names.update(_normalize_process_name(name) for name in target.process_names if name.strip())
    return names


def _match_target(targets: list[TargetWindowConfig], title: str, process_name: str) -> TargetWindowConfig | None:
    target = _match_target_raw(targets, title, process_name)
    if target is None:
        return None
    if _title_is_excluded(title, target):
        return None
    return target


def _match_target_raw(targets: list[TargetWindowConfig], title: str, process_name: str) -> TargetWindowConfig | None:
    normalized_process = _normalize_process_name(process_name)
    for target in targets:
        process_names = {_normalize_process_name(name) for name in target.process_names}
        if normalized_process and normalized_process in process_names:
            return target
        for keyword in target.title_keywords:
            value = keyword.strip()
            if value and _title_keyword_matches(title, value):
                return target
    return None


def _title_is_excluded(title: str, target: TargetWindowConfig) -> bool:
    normalized_title = " ".join(title.strip().split()).lower()
    if not normalized_title:
        return False
    for keyword in target.exclude_title_keywords:
        normalized_keyword = " ".join(keyword.strip().split()).lower()
        if normalized_keyword and normalized_keyword in normalized_title:
            return True
    return False


def _title_keyword_matches(title: str, keyword: str) -> bool:
    normalized_title = " ".join(title.strip().split()).lower()
    normalized_keyword = " ".join(keyword.strip().split()).lower()
    if not normalized_title or not normalized_keyword:
        return False
    if normalized_keyword in {"wechat", "微信"}:
        return normalized_title in {normalized_keyword, f"[{normalized_keyword}]", f"【{normalized_keyword}】"}
    return normalized_keyword in normalized_title


def _process_ids_by_names(names: set[str]) -> dict[int, str]:
    """Return matching processes without spawning ``tasklist`` on the UI thread.

    Window discovery runs every 800ms from a Qt timer. The former implementation
    launched ``tasklist.exe`` for every poll, which blocked the event loop for
    hundreds of milliseconds. A short psutil-backed cache keeps target discovery
    current without repeatedly stalling the interface.
    """
    global _PROCESS_NAME_CACHE, _PROCESS_NAME_CACHE_EXPIRES_AT
    now = time.monotonic()
    if now >= _PROCESS_NAME_CACHE_EXPIRES_AT:
        snapshot: dict[int, str] = {}
        try:
            import psutil

            for process in psutil.process_iter(["pid", "name"]):
                try:
                    process_name = str(process.info.get("name") or "")
                    process_id = int(process.info["pid"])
                except (KeyError, TypeError, ValueError, psutil.Error):
                    continue
                if process_name:
                    snapshot[process_id] = process_name
        except (ImportError, OSError):
            # Per-window lookup through ``_process_name_by_pid`` remains the
            # safe fallback when a process-table snapshot is unavailable.
            snapshot = {}
        _PROCESS_NAME_CACHE = snapshot
        _PROCESS_NAME_CACHE_EXPIRES_AT = now + _PROCESS_NAME_CACHE_TTL_SECONDS

    normalized_names = {_normalize_process_name(name) for name in names}
    return {
        process_id: process_name
        for process_id, process_name in _PROCESS_NAME_CACHE.items()
        if _normalize_process_name(process_name) in normalized_names
    }


def _process_name_by_pid(pid: int) -> str:
    try:
        import psutil

        return str(psutil.Process(pid).name())
    except Exception:
        return ""


def _process_executable_by_pid(pid: int) -> str:
    try:
        import psutil

        return str(psutil.Process(pid).exe())
    except Exception:
        return ""


def _looks_related_to_targets(title: str, process_name: str, targets: list[TargetWindowConfig]) -> bool:
    haystack = f"{title} {process_name}".lower()
    for target in targets:
        for value in [*target.title_keywords, *target.process_names, target.label, target.app_id]:
            marker = value.strip().lower()
            if marker and marker.removesuffix(".exe") in haystack:
                return True
    return False


def _match_reason(
    title: str,
    process_name: str,
    target: TargetWindowConfig | None,
    raw_target: TargetWindowConfig | None = None,
) -> str:
    if target is None:
        if raw_target is not None and _title_is_excluded(title, raw_target):
            return "excluded_by_title"
        return "related_but_not_matched"
    normalized_process = _normalize_process_name(process_name)
    process_names = {_normalize_process_name(name) for name in target.process_names}
    if normalized_process and normalized_process in process_names:
        return "matched_by_process"
    return "matched_by_title"


def _normalize_process_name(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    if not cleaned.endswith(".exe"):
        cleaned += ".exe"
    return cleaned


def _is_canonical_capture_window(
    target: TargetWindowConfig,
    *,
    title: str,
    window_class: str,
    rect: tuple[int, int, int, int],
    style: int,
    owner: int,
) -> bool:
    """Reject transient WeChat surfaces that share Weixin.exe with the chat UI.

    WeChat 4.x creates visible, full-screen Qt popup surfaces while switching
    chats and moving focus.  Matching by process name alone made those HWNDs
    look like new account windows.  The real chat window has the canonical
    application title and behaves as an ordinary captioned top-level window.
    Other configured applications retain the existing permissive matcher.
    """
    if target.app_id != "wechat":
        return True
    normalized_title = " ".join(title.strip().split()).casefold()
    if normalized_title not in {"微信", "wechat", "[微信]", "[wechat]", "【微信】", "【wechat】"}:
        return False
    left, top, right, bottom = rect
    if right - left < 480 or bottom - top < 420 or owner:
        return False
    normalized_class = window_class.strip().casefold()
    known_main_class = any(
        marker in normalized_class
        for marker in ("qwindowicon", "wechatmainwnd", "wechatmainwindow")
    )
    unsigned_style = int(style) & 0xFFFFFFFF
    captioned_resizable = bool(unsigned_style & 0x00C00000) and bool(unsigned_style & 0x00040000)
    return known_main_class or captioned_resizable


def _window_diagnostic(title: str, process_name: str, minimized: bool, foreground: bool, covered: bool = False) -> str:
    if minimized:
        return "目标窗口已最小化，悬浮窗和采集会暂停"
    if covered:
        return "目标窗口部分被其他窗口遮挡，屏幕截图可能不完整，已暂停采集"
    if not foreground:
        return "目标窗口可见但不是当前前景窗口；未检测到遮挡"
    if not process_name:
        return "仅通过窗口标题匹配，未读取到目标进程名；若采集失败，请检查权限或补充进程名规则"
    return ""


def _window_is_covered(hwnd: int, rect: tuple[int, int, int, int]) -> bool:
    """Sample the target's surface without confusing focus with occlusion."""
    try:
        import win32gui

        left, top, right, bottom = rect
        if right <= left or bottom <= top:
            return True
        root = int(win32gui.GetAncestor(hwnd, 2) or hwnd)
        margin_x = max(8, min(24, (right - left) // 12))
        margin_y = max(8, min(24, (bottom - top) // 12))
        points = (
            (left + margin_x, top + margin_y),
            (right - margin_x, top + margin_y),
            (left + margin_x, bottom - margin_y),
            (right - margin_x, bottom - margin_y),
            ((left + right) // 2, (top + bottom) // 2),
        )
        foreign = 0
        for x, y in points:
            hit = int(win32gui.WindowFromPoint((x, y)) or 0)
            if not hit:
                foreign += 1
                continue
            hit_root = int(win32gui.GetAncestor(hit, 2) or hit)
            if hit_root != root:
                foreign += 1
        # One point can land on a border, shadow, or transient tooltip. Require
        # two independent samples before reporting actual occlusion.
        return foreign >= 2
    except Exception:
        # A failed probe is unknown, not covered. Foreground policy can still
        # protect users who explicitly enabled it.
        return False
