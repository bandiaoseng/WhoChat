from __future__ import annotations

import os

from PySide6.QtCore import QObject, QTimer, Signal

from whochat.config import TargetWindowConfig, default_target_windows
from whochat.platform.window_tracker import (
    WindowInfo,
    find_target_windows,
    find_wechat_windows,
    foreground_window_handle,
    window_process_id,
)


class TargetWindowFollowController(QObject):
    window_changed = Signal(object)
    status_changed = Signal(str)

    def __init__(
        self,
        targets: list[TargetWindowConfig] | None = None,
        interval_ms: int = 800,
        foreground_only: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.targets = targets or default_target_windows()
        self.foreground_only = foreground_only
        self._last_hwnd: int | None = None
        self._last_window: WindowInfo | None = None
        self._last_signature: tuple[object, ...] | None = None
        self._last_status: str = ""
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.poll_once)

    def start(self) -> None:
        self._timer.start()
        self.poll_once()

    def stop(self) -> None:
        self._timer.stop()

    def set_targets(self, targets: list[TargetWindowConfig]) -> None:
        self.targets = targets
        self._last_hwnd = None
        self._last_signature = None
        self.poll_once()

    def set_foreground_only(self, foreground_only: bool) -> None:
        self.foreground_only = foreground_only
        self._last_hwnd = None
        self._last_signature = None
        self.poll_once()

    def poll_once(self) -> WindowInfo | None:
        windows = find_target_windows(self.targets)
        if not windows:
            self._emit_absent(("missing",), "未发现已启用的聊天窗口")
            return None
        focused = foreground_window_handle()
        # Prefer an exact foreground handle. Do not infer that every window
        # marked visible is focused, especially when several targets exist.
        window = next((item for item in windows if item.hwnd == focused and item.visible), None)
        previous = next(
            (item for item in windows if self._last_window is not None
             and _same_target_instance(item, self._last_window)),
            None,
        )
        pause_status = ""
        if window is None and focused is not None:
            focused_pid = window_process_id(focused)
            same_process = [
                item for item in windows
                if item.visible and focused_pid is not None and item.process_id == focused_pid
            ]
            if same_process:
                # A popup's PID proves neither a selected main window nor a
                # new account. In particular, a popup from a second WeChat
                # process must not replace the previously selected target.
                # Only an exact foreground canonical HWND above may do that.
                if previous is None:
                    self._emit_absent(
                        ("unselected_target_popup", focused_pid),
                        "聊天临时界面位于前台，等待主聊天窗口确认",
                    )
                    return None
                pause_status = (
                    "微信临时界面位于前台，已保留主聊天身份并暂停采集"
                    if focused_pid == previous.process_id
                    else "其他微信临时界面位于前台，已保留原聊天窗口并暂停采集"
                )
                window = _as_paused_target_window(previous, pause_status)
            elif focused_pid == os.getpid() and previous is not None:
                pause_status = "WhoChat 位于前台，已保留最后确认的聊天身份并暂停采集"
                window = _as_paused_target_window(previous, pause_status)
        if window is None and focused is None:
            window = next((item for item in windows if item.foreground and item.visible), None)
        if window is None:
            if self.foreground_only:
                # An extant previously selected canonical target is a paused
                # observation, not a missing window. Never pick a different
                # account just because another application owns foreground.
                canonical = previous
                if canonical is None:
                    self._emit_absent(("missing_selected",), "此前聊天窗口已关闭或尚未选择聊天窗口")
                    return None
                pause_status = "目标聊天窗口暂离前台，已保留客户身份并暂停采集"
                window = _as_paused_target_window(canonical, pause_status)
            else:
                window = max(windows, key=lambda item: (item.rect[2] - item.rect[0]) * (item.rect[3] - item.rect[1]))
                window = _as_background_test_window(window)
        elif self.foreground_only and not pause_status and (window.minimized or not window.visible or not window.foreground):
            pause_status = "目标聊天窗口不可采集，已保留客户身份并暂停采集"
            window = _as_paused_target_window(window, pause_status)
        # Position alone is not enough. In relaxed/background mode a target
        # can keep the same rect while losing foreground, becoming minimized,
        # or changing its diagnostic safety state.
        signature = (
            window.hwnd,
            window.title,
            window.rect,
            window.client_rect,
            getattr(window, "dpi_scale", 1.0),
            window.visible,
            window.foreground,
            window.minimized,
            window.diagnostic,
            window.target_app,
            window.process_id,
            window.executable_path,
        )
        if signature == self._last_signature:
            return window
        self._last_signature = signature
        self._last_window = window
        status = pause_status or f"已连接：{window.app_label} · {window.title}"
        if status != self._last_status:
            self._last_status = status
            self.status_changed.emit(status)
        self._last_hwnd = window.hwnd
        self.window_changed.emit(window)
        return window

    def _emit_absent(self, signature: tuple[object, ...], status: str) -> None:
        if self._last_signature == signature:
            return
        self._last_hwnd = None
        self._last_window = None
        self._last_signature = signature
        if status != self._last_status:
            self._last_status = status
            self.status_changed.emit(status)
        self.window_changed.emit(None)


class WeChatFollowController(TargetWindowFollowController):
    def __init__(self, interval_ms: int = 800, parent: QObject | None = None) -> None:
        targets = [target for target in default_target_windows() if target.app_id == "wechat"]
        super().__init__(targets=targets, interval_ms=interval_ms, parent=parent)


def _same_target_instance(left: WindowInfo, right: WindowInfo) -> bool:
    return (
        left.hwnd == right.hwnd
        and left.process_id == right.process_id
        and left.target_app == right.target_app
        and left.executable_path.casefold() == right.executable_path.casefold()
    )


def _as_background_test_window(window: WindowInfo) -> WindowInfo:
    return WindowInfo(
        hwnd=window.hwnd,
        title=window.title,
        process_name=window.process_name,
        rect=window.rect,
        visible=not window.minimized,
        target_app=window.target_app,
        app_label=window.app_label,
        process_id=window.process_id,
        minimized=window.minimized,
        diagnostic=("后台测试模式：目标窗口部分被遮挡，截图可能不完整" if window.covered else
                    "后台测试模式：目标窗口可见但不是前台，未检测到遮挡"),
        foreground=False,
        bubble_profile=window.bubble_profile,
        covered=window.covered,
        executable_path=window.executable_path,
        client_rect=window.client_rect,
        dpi_scale=getattr(window, "dpi_scale", 1.0),
    )


def _as_paused_target_window(window: WindowInfo, diagnostic: str) -> WindowInfo:
    """Keep the canonical HWND stable while capture is safely paused."""
    return WindowInfo(
        hwnd=window.hwnd,
        title=window.title,
        process_name=window.process_name,
        rect=window.rect,
        visible=not window.minimized,
        target_app=window.target_app,
        app_label=window.app_label,
        process_id=window.process_id,
        minimized=window.minimized,
        diagnostic=diagnostic,
        foreground=False,
        bubble_profile=window.bubble_profile,
        covered=True,
        executable_path=window.executable_path,
        client_rect=window.client_rect,
        dpi_scale=getattr(window, "dpi_scale", 1.0),
    )
