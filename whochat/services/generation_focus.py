"""Bounded focus acquisition for one explicit generation click.

This is capture preparation, never customer/account proof. All subsequent
independent frames, source/lease checks and manual-only AI gates remain required.
No focus is taken from another application, account, dialog or minimized window.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time


@dataclass
class GenerationFocus:
    key: tuple
    attempted: bool = False
    attempt_at: float = 0.0
    interrupted: bool = False
    reason: str = ""


def read_window(hwnd):
    from whochat.services.manual_suggestions import read_native_window
    return read_native_window(hwnd)


def request_foreground(hwnd):
    import win32gui
    win32gui.SetForegroundWindow(hwnd)


def _valid(sample, hwnd, pid):
    return (sample.hwnd == hwnd and sample.process_id == pid
            and type(hwnd) is int and hwnd > 0 and type(pid) is int and pid > 0
            and sample.visible is True and sample.minimized is False
            and type(sample.foreground) is bool
            and type(sample.foreground_hwnd) is int and sample.foreground_hwnd > 0
            and type(sample.foreground_process_id) is int and sample.foreground_process_id > 0
            and sample.foreground == (sample.foreground_hwnd == hwnd)
            and (not sample.foreground or sample.foreground_process_id == pid)
            and type(sample.rect) is tuple and len(sample.rect) == 4
            and all(type(x) is int for x in sample.rect)
            and sample.rect[0] < sample.rect[2] and sample.rect[1] < sample.rect[3])


def focus_step(ticket, *, allowed_windows, current, modal=False, reader=None, requester=None, now=None):
    """Return (action, diagnostic); only 'refresh' permits fresh enumeration.

    current rechecks the immutable request/owner and deadline immediately before
    the native request. Polling never requests focus again. OS denial is final.
    """
    reader, requester = reader or read_window, requester or request_foreground
    now = time.monotonic() if now is None else now
    if not current():
        return "blocked", "generation_focus_request_changed"
    hwnd, pid = ticket.key[1][:2]
    try:
        sample = reader(hwnd)
        if not _valid(sample, hwnd, pid):
            return "blocked", "generation_focus_target_unavailable"
        if sample.foreground:
            return "refresh", "generation_focus_target_foreground"
        if modal:
            return "wait", "generation_focus_dialog_open"
        foreground = (sample.foreground_hwnd, sample.foreground_process_id)
        # Only actual top-level WhoChat widgets supplied by the UI qualify.
        if (foreground not in allowed_windows or foreground[1] != os.getpid()
                or foreground[0] == hwnd):
            return "wait", "generation_focus_other_application"
        if ticket.attempted:
            return (("blocked", "generation_focus_not_restored") if now - ticket.attempt_at >= 2.0
                    else ("wait", "generation_focus_restoring"))
        own = reader(foreground[0])
        again = reader(hwnd)
        if (not _valid(own, *foreground) or not own.foreground or again != sample or not current()):
            return "wait", "generation_focus_changed_during_check"
        ticket.attempted, ticket.attempt_at = True, now
        requester(hwnd)
        return "wait", "generation_focus_restoring"
    except Exception:
        return "blocked", "generation_focus_native_unavailable"


def refresh_target(runtime, targets, ticket, current, *, finder=None, reader=None):
    """Commit a fresh OS observation of the exact original WeChat, not flags."""
    if finder is None:
        from whochat.platform.window_tracker import find_target_windows
        finder = find_target_windows
    reader = reader or read_window
    hwnd, pid = ticket.key[1][:2]
    if not current():
        return False
    windows = [w for w in finder(targets)
               if w.hwnd == hwnd and w.process_id == pid and w.target_app == "wechat"]
    if len(windows) != 1:
        return False
    window, native = windows[0], reader(hwnd)
    if (not current() or not _valid(native, hwnd, pid) or not native.foreground
            or window.rect != native.rect or window.client_rect != native.client_rect
            or window.foreground is not True or window.visible is not True
            or window.minimized is not False or window.covered is not False):
        return False
    runtime.update_from_window_info(window)
    return current()


def focus_message(reason):
    if reason == "generation_focus_other_application":
        return "读取聊天已暂停：其他软件正在前台，请切回本次选定的微信客户窗口后重试；未调用 AI"
    if reason == "generation_focus_dialog_open":
        return "读取聊天已暂停：请先关闭 WhoChat 弹窗，再回到微信客户窗口重试；未调用 AI"
    return "未能恢复本次选定的微信窗口进行读取，请将该客户聊天置于前台后重试；未调用 AI"
