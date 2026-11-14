"""Explicit screenshot UI. Capturing/choosing/cancelling never calls AI."""
from __future__ import annotations

import time
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from whochat.core.runtime import Rect
from whochat.services.manual_selection import (
    SelectionError, build_selection, capture_selection_frame, selection_error_message,
)


def map_selection(start, end, width, height, source_rect):
    """Qt logical coordinates to frozen screenshot pixels, including negative origins."""
    if width <= 0 or height <= 0:
        raise SelectionError('manual_selection_geometry_invalid')
    x1, x2 = sorted((max(0.0, min(width, start.x())), max(0.0, min(width, end.x()))))
    y1, y2 = sorted((max(0.0, min(height, start.y())), max(0.0, min(height, end.y()))))
    return Rect(source_rect.left+round(x1*source_rect.width/width),
                source_rect.top+round(y1*source_rect.height/height),
                source_rect.left+round(x2*source_rect.width/width),
                source_rect.top+round(y2*source_rect.height/height))


class ScreenshotSelectionOverlay(QWidget):
    selected = Signal(object)
    cancelled = Signal()

    def __init__(self, frame):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('WhoChat · 框选聊天')
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._source_rect = frame.state.window.rect
        rgb = frame.image.convert('RGB')
        self._image = QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width*3,
                             QImage.Format.Format_RGB888).copy()
        self._start = None
        self._end = None
        self._done = False
        self._error = ''
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self.cancel)
        self._timeout.start(120000)

    def show_over_target(self):
        # Move only this application's overlay. Exact physical placement is
        # verified before accepting a user choice; Qt event positions are
        # mapped proportionally, not assumed to equal Windows physical pixels.
        import win32gui
        import win32con
        rect = self._source_rect
        self.resize(rect.width, rect.height)
        self.show()
        win32gui.SetWindowPos(int(self.winId()), win32con.HWND_TOPMOST,
            rect.left, rect.top, rect.width, rect.height, win32con.SWP_SHOWWINDOW)
        actual = tuple(win32gui.GetWindowRect(int(self.winId())))
        if actual != rect.as_tuple():
            self.finish()
            raise SelectionError('manual_selection_geometry_invalid')
        self.activateWindow()
        self.setFocus()

    def paintEvent(self, event):
        painter = QPainter(self)
        bounds = QRectF(self.rect())
        painter.drawImage(bounds, self._image)
        painter.fillRect(bounds, QColor(0, 0, 0, 85))
        if self._start is not None and self._end is not None:
            selection = QRectF(self._start, self._end).normalized().intersected(bounds)
            painter.save()
            painter.setClipRect(selection)
            painter.drawImage(bounds, self._image)
            painter.restore()
            painter.setPen(QPen(QColor('#16b7a3'), 2))
            painter.drawRect(selection)
        # Bottom banner leaves the customer title visible while dragging.
        hint = QRectF(8, max(0, self.height()-74), max(1, self.width()-16), 66)
        painter.fillRect(hint, QColor(15, 35, 43, 235))
        painter.setPen(QColor('#ffffff'))
        painter.drawText(hint.adjusted(10, 5, -10, -5),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            self._error or '框住顶部客户名称和双方完整聊天，左右保留头像。\n下缘停在打字输入框上方；松开鼠标生成，Esc 或右键取消。')

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.cancel()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._start = QPointF(event.position())
            self._end = self._start
            self._error = ''
            self.update()

    def mouseMoveEvent(self, event):
        if self._start is not None:
            self._end = QPointF(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._start is not None:
            self._end = QPointF(event.position())
            selected = map_selection(self._start, self._end, self.width(), self.height(), self._source_rect)
            self.selected.emit(selected)

    def show_error(self, text):
        self._error = text
        self._start = self._end = None
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()

    def cancel(self):
        if not self._done:
            self._done = True
            self._timeout.stop()
            self.hide()
            self.cancelled.emit()
            self.close()

    def finish(self):
        self._done = True
        self._timeout.stop()
        self.close()

    def closeEvent(self, event):
        if not self._done:
            self._done = True
            self.cancelled.emit()
        self._timeout.stop()
        event.accept()


class ScreenshotWorkflow(QObject):
    frame_ready = Signal(object, object)
    frame_failed = Signal(object, str)

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.busy = False
        self.ticket = None
        self.frame = None
        self.overlay = None
        self.future = None
        self._hidden = []
        self.run = None
        self.frame_ready.connect(self._frame_ready, Qt.ConnectionType.QueuedConnection)
        self.frame_failed.connect(self._failed, Qt.ConnectionType.QueuedConnection)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def start(self):
        main = self.main
        if (self.busy or main._manual_generation_phase in {'recognizing', 'generating', 'verifying', 'selecting'}
                or QApplication.activeModalWidget() is not None):
            main._set_manual_generation_phase(main._manual_generation_phase)
            return
        reader = getattr(main, '_local_preread', None)
        if reader is not None:
            reader.pause()
        if main._services.pipeline.is_running or main._services.reply_tasks.is_running:
            main._set_manual_generation_phase('failed')
            main.statusBar().showMessage('上一任务仍在结束，请稍后再点截图生成', 4000)
            return
        state = main._services.runtime.refresh_current()
        if (state.window.hwnd is None or state.window.target.value != 'wechat'
                or state.window.rect is None or state.layout is None or state.paused):
            main._set_manual_generation_phase('failed')
            main.statusBar().showMessage('请先打开要回复的微信客户窗口，再点击截图生成', 5000)
            return
        from whochat.services.generation_focus import GenerationFocus
        if self.run is not None:
            self.run.cancel()
            self.run = None
        self._previous_phase = main._manual_generation_phase
        self._owner = main._manual_window_owner(state)
        self.ticket = object()
        self._focus = GenerationFocus((id(self.ticket), self._owner))
        self._deadline = time.monotonic()+12.0
        self.busy = True
        main._set_manual_generation_phase('selecting')
        main.statusBar().showMessage('正在准备微信截图；框选完成前不会调用 AI', 5000)
        ticket = self.ticket
        QTimer.singleShot(12000, self, lambda: self._capture_timed_out(ticket))
        self._prepare_focus(self.ticket)

    def _capture_timed_out(self, ticket):
        if self.frame is None:
            self._failed(ticket, 'manual_selection_capture_timeout')

    def _current(self, ticket):
        return (self.busy and self.ticket is ticket and time.monotonic() < self._deadline
                and self.main._manual_window_owner(self.main._services.runtime.state) == self._owner)

    def _prepare_focus(self, ticket):
        from whochat.services.generation_focus import focus_step, refresh_target
        if not self._current(ticket):
            self._failed(ticket, 'manual_selection_owner_changed')
            return
        main = self.main
        action, reason = focus_step(self._focus, current=lambda: self._current(ticket),
            allowed_windows=main._manual_copy_handoff_windows(),
            modal=QApplication.activeModalWidget() is not None)
        if action == 'refresh':
            try:
                refreshed = refresh_target(main._services.runtime, main._config.targets,
                    self._focus, lambda: self._current(ticket))
            except Exception:
                refreshed = False
            if not refreshed:
                self._failed(ticket, 'manual_selection_geometry_changed')
                return
            state = main._services.runtime.state
            from threading import Event
            from whochat.services.frozen_screenshot import capture_frozen_pair
            self._capture_cancel = Event()
            self.future = main._services.pipeline.executor.submit(
                capture_frozen_pair, state, main._services.account_scope_resolver,
                cancelled=self._capture_cancel)
            self.future.add_done_callback(lambda future: self._captured(ticket, future))
            return
        if action == 'blocked':
            self._failed(ticket, 'manual_selection_target_unavailable')
            return
        QTimer.singleShot(50, self, lambda: self._prepare_focus(ticket))

    def _captured(self, ticket, future):
        frame = None
        try:
            frame = future.result()
            self.frame_ready.emit(ticket, frame)
        except Exception as exc:
            if frame is not None:
                from whochat.services.frozen_screenshot import close_frame
                close_frame(frame)
            try:
                reason = str(exc) if isinstance(exc, SelectionError) else 'manual_selection_capture_unavailable'
                self.frame_failed.emit(ticket, reason)
            except RuntimeError:
                pass

    def _frame_ready(self, ticket, frame):
        if not self._current(ticket):
            from whochat.services.frozen_screenshot import close_frame
            close_frame(frame)
            self._failed(ticket, 'manual_selection_owner_changed')
            return
        self.frame = frame
        try:
            self._hidden = [widget for widget in (self.main, getattr(self.main, '_floating', None))
                            if widget is not None and widget.isVisible()]
            for widget in self._hidden:
                widget.hide()
            self.overlay = ScreenshotSelectionOverlay(frame)
            self.overlay.selected.connect(self._selected)
            self.overlay.cancelled.connect(self.cancel)
            self.overlay.show_over_target()
        except Exception:
            self._failed(ticket, 'manual_selection_geometry_invalid')

    def _selected(self, rect):
        if not self.busy or self.frame is None:
            return
        try:
            plan = build_selection(self.frame, rect)
        except SelectionError as exc:
            if self.overlay is not None:
                self.overlay.show_error(selection_error_message(str(exc)))
            return
        except Exception:
            self._failed(self.ticket, 'manual_selection_geometry_invalid')
            return
        from whochat.services.frozen_screenshot import freeze_selection
        from whochat.ui.frozen_screenshot import FrozenScreenshotRun
        try:
            request = freeze_selection(self.frame, plan, self.main._manual_generation_request_id+1)
        except SelectionError as exc:
            self._failed(self.ticket, str(exc))
            return
        self._cleanup(restore=True)
        self.busy = True
        self.run = FrozenScreenshotRun(self, request)
        self.run.start()

    def _cleanup(self, *, restore):
        self.busy = False
        self.ticket = None
        cancelled = getattr(self, '_capture_cancel', None)
        if cancelled is not None:
            cancelled.set()
        if self.future is not None:
            self.future.cancel()
            self.future = None
        if self.overlay is not None:
            overlay, self.overlay = self.overlay, None
            overlay.finish()
            overlay.deleteLater()
        if self.frame is not None:
            from whochat.services.frozen_screenshot import close_frame
            close_frame(self.frame)
            self.frame = None
        hidden, self._hidden = self._hidden, []
        if restore:
            for widget in hidden:
                widget.show()

    def _failed(self, ticket, reason):
        if not self.busy or self.ticket is not ticket:
            return
        self._cleanup(restore=True)
        self.main._set_manual_generation_phase(self._previous_phase)
        self.main.append_log('manual_screenshot_failed: '+reason)
        self.main.statusBar().showMessage(selection_error_message(reason), 9000)

    def cancel(self):
        if self.busy:
            self._cleanup(restore=True)
            self.main._set_manual_generation_phase(self._previous_phase)
            self.main.statusBar().showMessage('已取消截图；未调用 AI、未扣额度', 4000)

    def shutdown(self):
        if self.run is not None:
            self.run.cancel()
        self._cleanup(restore=False)
