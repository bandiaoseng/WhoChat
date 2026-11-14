from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from whochat.ai.models import ReplyGenerationResult


@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    width: int
    height: int
    edge: str


class FloatingWidget(QWidget):
    suggestion_copied = Signal(str)
    user_hidden_changed = Signal(bool)
    calibration_requested = Signal()
    generate_requested = Signal()

    GAP = 6
    MIN_WIDTH = 520
    MAX_WIDTH = 880
    BAR_HEIGHT = 40
    SIDE_WIDTH = 520
    SIDE_HEIGHT = 40

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setObjectName("FloatingWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Showing or clicking this mouse-operated tool must leave the chat
        # window in the foreground. WindowDoesNotAcceptFocus prevents window
        # activation; NoFocus on the buttons still permits clicked signals.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._drag_pos = None
        self._user_hidden = False
        self._last_rect: tuple[int, int, int, int] | None = None
        self._last_title = "微信"
        self._last_edge = ""
        self._placement_preference = "auto"
        self._suggestion_count = 2
        self._copy_handler: Callable[[str, object], bool | None] | None = None
        self._copy_pending = False
        self._copy_binding: object | None = None
        self._result_retention_guard: Callable[[], bool] | None = None
        self._generation_state = "idle"

        self.root = QFrame()
        self.root.setObjectName("FloatingRoot")
        shell = QHBoxLayout(self.root)
        shell.setContentsMargins(8, 6, 8, 6)
        shell.setSpacing(6)
        self.contact_label = QLabel("未确认联系人")
        self.contact_label.setObjectName("FloatingContact")
        # The explicit action must remain reachable even for a long customer
        # name. Let labels clip (the full text remains in their tooltips), not
        # overlap the fixed-width action/copy buttons on narrow placements.
        self.contact_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.group_label = QLabel("默认分组")
        self.group_label.setObjectName("FloatingBadge")
        self.group_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.group_label.setVisible(False)
        self.status_label = QLabel("等待")
        self.status_label.setObjectName("FloatingStatus")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.status_label.setMinimumWidth(24)
        self.status_label.setMaximumWidth(84)
        self.action_button = QPushButton("校准")
        self.action_button.setObjectName("FloatingGhostButton")
        self.action_button.setToolTip("打开区域校准")
        self.action_button.setFixedHeight(26)
        self.action_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.action_button.setVisible(False)
        self.action_button.clicked.connect(self.calibration_requested.emit)
        self.generate_button = QPushButton("生成建议")
        self.generate_button.setObjectName("FloatingGenerateButton")
        self.generate_button.setToolTip("点击后识别当前客户和最新聊天，再生成本次建议；不会自动发送")
        self.generate_button.setFixedHeight(28)
        self.generate_button.setMinimumWidth(82)
        self.generate_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.generate_button.clicked.connect(self._request_generation)
        hide_button = QPushButton("×")
        hide_button.setObjectName("FloatingGhostButton")
        hide_button.setToolTip("隐藏悬浮窗")
        hide_button.setFixedSize(26, 26)
        hide_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        hide_button.clicked.connect(self.hide_by_user)
        self.suggestion_buttons: list[QPushButton] = []
        shell.addWidget(self.contact_label, 1)
        shell.addWidget(self.group_label)
        shell.addWidget(self.status_label)
        shell.addWidget(self.action_button)
        shell.addStretch(1)
        shell.addWidget(self.generate_button)
        self._set_suggestions(
            [
                ("回复1", ""),
                ("回复2", ""),
            ]
        )
        for button in self.suggestion_buttons:
            shell.addWidget(button)
        shell.addWidget(hide_button)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.root)

        self._place_on_primary_screen()

    @property
    def user_hidden(self) -> bool:
        return self._user_hidden

    @property
    def placement_preference(self) -> str:
        return self._placement_preference

    @property
    def suggestion_count(self) -> int:
        return self._suggestion_count

    @property
    def placement_edge(self) -> str:
        return self._last_edge

    def apply_preferences(self, *, placement_preference: str = "auto", opacity_percent: int = 96, suggestion_count: int = 2) -> None:
        placement = placement_preference if placement_preference in {"auto", "bottom", "top", "right", "left"} else "auto"
        self._placement_preference = placement
        self._suggestion_count = 2
        self.setWindowOpacity(self._clamp(opacity_percent, 70, 100) / 100)
        for index, button in enumerate(self.suggestion_buttons):
            button.setVisible(index < self._suggestion_count)
        if self._last_rect is not None and not self._user_hidden:
            self.attach_to_window_rect(self._last_rect, self._last_title)

    def update_context(self, *, contact_name: str, group_name: str, status: str, action: str = "", app_label: str = "微信") -> None:
        self.contact_label.setText(_context_label(app_label, contact_name, group_name))
        self.contact_label.setToolTip(_context_label(app_label, contact_name, group_name, max_len=120))
        self.group_label.setText(group_name or "默认分组")
        if not self._copy_pending:
            self.status_label.setText(status or "等待")
            self.status_label.setToolTip(action or status or "等待")
        self._sync_action_button(status=status, action=action)

    def update_reply_result(self, result: ReplyGenerationResult) -> None:
        if result.status.startswith("reply_pending"):
            queued = "waiting_latest" in result.status
            limited = "waiting_rate_limit" in result.status
            self.status_label.setText("排队等待" if queued else "等待间隔" if limited else "生成中")
            self.status_label.setToolTip("上一请求尚未结束，本次尚未发起；仍计入本次总等待时限" if queued else
                                        "尚未发起本次 AI 请求；仍计入本次总等待时限" if limited else
                                        "正在生成；超时会明确结束，不会无限等待")
            self._update_suggestion_buttons([], enabled=False)
            return
        if not result.allowed:
            self.status_label.setText(_short_status(result.status))
            self.status_label.setToolTip(_terminal_status_detail(result.status))
            self._update_suggestion_buttons([], enabled=False)
            return
        if not self._copy_pending:
            self.status_label.setText(_risk_status(result))
        suggestions = [(item.label or f"回复{index + 1}", item.text) for index, item in enumerate(result.suggestions[:2])]
        self._update_suggestion_buttons(suggestions, enabled=True)

    def set_copy_handler(self, handler: Callable[[str, object], bool | None]) -> None:
        self._copy_handler = handler

    def set_copy_status(self, status: str, message: str) -> None:
        self._copy_pending = status == "pending"
        self.status_label.setText({"pending": "复制核验中", "copied": "已复制", "rejected": "未复制"}.get(status, "未复制"))
        self.status_label.setToolTip(message)
        for button in self.suggestion_buttons:
            button.setEnabled(not self._copy_pending and bool(button.property("reply_text"))
                              and self._copy_binding is not None and self._copy_handler is not None)

    def set_result_retention_guard(self, guard: Callable[[], bool]) -> None:
        self._result_retention_guard = guard

    def set_review_status(self, reason: str) -> None:
        """Keep text; a clickable button still requires MainWindow's gate."""
        if self._copy_pending:
            return
        if reason:
            self.status_label.setText("已保留·待核验")
            self.status_label.setToolTip("原客户建议已保留。来源过期时，点击话术会在本地重新核验；聊天更新后需点击生成建议。不会自动请求 AI。")
        elif self._copy_binding is not None:
            self.status_label.setText("可核验复制")
            self.status_label.setToolTip("点击时核验原客户与聊天内容，通过后才复制；不会自动发送。")

    def set_generation_state(self, state: str) -> None:
        """Update the explicit one-click workflow independently of copy state."""
        if state not in {"idle", "selecting", "recognizing", "generating", "verifying", "success", "failed"}:
            raise ValueError("unsupported generation state")
        self._generation_state = state
        busy = state in {"selecting", "recognizing", "generating", "verifying"}
        self.generate_button.setEnabled(not busy)
        self.generate_button.setText({
            "idle": "截图生成" if getattr(self, '_screenshot_mode', False) else "生成建议",
            "selecting": "框选中",
            "recognizing": "识别中",
            "generating": "生成中",
            "verifying": "核验中",
            "success": "重新截图" if getattr(self, '_screenshot_mode', False) else "重新生成",
            "failed": "重新截图" if getattr(self, '_screenshot_mode', False) else "重新生成",
        }[state])

    def set_screenshot_mode(self, enabled: bool) -> None:
        self._screenshot_mode = bool(enabled)
        self.generate_button.setToolTip('框选客户名称和完整聊天后生成建议；不会自动发送' if enabled else
                                       '点击后识别当前客户和最新聊天，再生成本次建议；不会自动发送')
        self.set_generation_state(self._generation_state)

    def _request_generation(self) -> None:
        # Change state before emitting: a second click (including a queued
        # signal/reentrant caller) cannot dispatch another in-flight task.
        if self._generation_state in {"selecting", "recognizing", "generating", "verifying"}:
            return
        self.set_generation_state("recognizing")
        self.generate_requested.emit()

    def set_copy_binding(self, binding: object | None) -> None:
        self._copy_binding = binding
        if binding is None:
            for button in self.suggestion_buttons:
                button.setEnabled(False)

    def disable_suggestions(self) -> None:
        self._update_suggestion_buttons([], enabled=False)

    def show_waiting(self) -> None:
        self._user_hidden = False
        self.user_hidden_changed.emit(False)
        self.contact_label.setText("微信·未确认联系人（默认分组）")
        self.group_label.setText("默认分组")
        self.status_label.setText("等待微信")
        self.action_button.setVisible(False)
        self._place_on_primary_screen()
        self.show()

    def show_by_user(self) -> None:
        self._user_hidden = False
        self.user_hidden_changed.emit(False)
        if self._last_rect is not None:
            self.attach_to_window_rect(self._last_rect, self._last_title)
        else:
            self.show_waiting()

    def hide_by_user(self) -> None:
        self._user_hidden = True
        self.user_hidden_changed.emit(True)
        self.hide()

    def attach_to_window_rect(self, rect: tuple[int, int, int, int], title: str = "微信") -> None:
        self._last_rect = rect
        self._last_title = title
        placement = self._choose_placement(rect)
        if placement is None:
            self.status_label.setText("空间不足")
            if not self._user_hidden:
                self.hide()
            return

        self.setFixedSize(placement.width, placement.height)
        self.move(placement.x, placement.y)
        self._last_edge = placement.edge
        existing_tip = self.status_label.toolTip().strip()
        placement_tip = f"悬浮窗已贴靠目标窗口{placement.edge}"
        self.status_label.setToolTip(f"{existing_tip}\n{placement_tip}".strip() if existing_tip else placement_tip)
        if not self._user_hidden and not self.isVisible():
            self.show()

    def detach_from_window(self, reason: str) -> None:
        self._last_rect = None
        self._last_edge = ""
        self.contact_label.setText("未确认联系人")
        self.group_label.setText("默认分组")
        self.status_label.setText(reason)
        self.action_button.setVisible(False)
        if not self._user_hidden and not self.isVisible():
            self.show()

    def hide_for_window_state(self, reason: str) -> None:
        try:
            retain = (self._last_rect is not None and self._result_retention_guard is not None
                      and self._result_retention_guard() is True)
        except Exception:
            retain = False
        if retain:
            # The follow controller already updated Runtime. Only an existing
            # same-owner completed result may remain as a read-only preview;
            # no callback here is allowed to authorize copy or new work.
            self.set_review_status("window_not_foreground")
            if not self._user_hidden and not self.isVisible():
                self.show()
            return
        self._last_rect = None
        self._last_edge = ""
        self.contact_label.setText("未确认联系人")
        self.group_label.setText("默认分组")
        self.status_label.setText(reason)
        self.action_button.setVisible(False)
        if not self._user_hidden:
            self.hide()

    def apply_window_color(self, rgb: tuple[int, int, int] | None) -> None:
        if rgb is None:
            self.root.setStyleSheet("")
            return
        base = QColor(*rgb)
        dark = _luminance(base) < 0.42
        bg = _blend(base, QColor("#000000"), 0.10) if dark else _blend(base, QColor("#ffffff"), 0.18)
        border = _blend(base, QColor("#ffffff" if dark else "#52606d"), 0.32)
        button_bg = _blend(base, QColor("#ffffff" if dark else "#ffffff"), 0.18 if dark else 0.42)
        hover_bg = _blend(base, QColor("#ffffff" if dark else "#ffffff"), 0.26 if dark else 0.30)
        accent = _readable_accent(base)
        text = "#ffffff" if _luminance(bg) < 0.45 else "#102a43"
        muted = "#e5e7eb" if _luminance(bg) < 0.45 else "#52606d"
        self.root.setStyleSheet(
            f"""
            QFrame#FloatingRoot {{
              background: {bg.name()};
              border: 1px solid {border.name()};
              border-radius: 7px;
            }}
            QLabel#FloatingContact {{
              color: {text};
              background: transparent;
            }}
            QLabel#FloatingStatus {{
              color: {muted};
              background: transparent;
            }}
            QLabel#FloatingBadge {{
              color: {accent.name()};
              background: {button_bg.name()};
            }}
            QPushButton#FloatingGhostButton, QPushButton#FloatingSuggestionButton, QPushButton#FloatingGenerateButton {{
              color: {text};
              background: {button_bg.name()};
              border: 1px solid {border.name()};
            }}
            QPushButton#FloatingSuggestionButton:hover, QPushButton#FloatingGenerateButton:hover {{
              border-color: {accent.name()};
              background: {hover_bg.name()};
            }}
            """
        )

    def _sync_action_button(self, *, status: str, action: str) -> None:
        text = f"{status} {action}"
        should_calibrate = "校准" in text or "区域" in text
        self.action_button.setVisible(should_calibrate)

    def _set_suggestions(self, suggestions: list[tuple[str, str]]) -> None:
        self.suggestion_buttons = [self._suggestion_button(label, text) for label, text in suggestions[:2]]

    def _update_suggestion_buttons(self, suggestions: list[tuple[str, str]], *, enabled: bool) -> None:
        fallback = [(f"回复{index + 1}", "") for index in range(2)]
        values = (suggestions + fallback)[:2]
        for index, (button, (label, text)) in enumerate(zip(self.suggestion_buttons, values)):
            button.setText(_suggestion_preview(text, label))
            button.setToolTip(text if text else "当前没有可复制的建议")
            button.setProperty("reply_text", text)
            button.setEnabled(
                enabled and not self._copy_pending and bool(text) and index < self._suggestion_count
                and self._copy_handler is not None and self._copy_binding is not None
            )
            button.setVisible(index < self._suggestion_count)

    def _suggestion_button(self, label: str, text: str) -> QWidget:
        button = QPushButton(_suggestion_preview(text, label))
        button.setObjectName("FloatingSuggestionButton")
        button.setToolTip(text if text else "当前没有可复制的建议")
        button.setProperty("reply_text", text)
        button.setFixedHeight(28)
        button.setFixedWidth(104)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setEnabled(False)
        button.clicked.connect(lambda checked=False, target=button: self._copy(str(target.property("reply_text") or "")))
        return button

    def _copy(self, text: str) -> bool:
        if not text:
            self.status_label.setText("无建议")
            return False
        # Copy belongs to MainWindow's immutable customer/context binding. A
        # repaint or a stale enabled button is never sufficient authorization.
        if self._copy_handler is None or self._copy_binding is None:
            self.status_label.setText("待确认")
            return False
        outcome = self._copy_handler(text, self._copy_binding)
        if outcome is None:
            self.set_copy_status("pending", "正在切回原微信窗口核验；完成后才复制，不会发送消息")
            return False
        if outcome is not True:
            if self.status_label.text() != "未复制":
                self.status_label.setText("待确认")
            return False
        self.status_label.setText("已复制")
        self.suggestion_copied.emit(text)
        return True

    def _choose_placement(self, rect: tuple[int, int, int, int]) -> Placement | None:
        left, top, right, bottom = rect
        screen = QGuiApplication.screenAt(QPoint((left + right) // 2, (top + bottom) // 2))
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        area = screen.availableGeometry()
        window_width = max(1, right - left)
        bar_width = min(self.MAX_WIDTH, max(self.MIN_WIDTH, min(window_width, self.MAX_WIDTH)))
        centered_x = self._clamp(left + (window_width - bar_width) // 2, area.left(), area.right() - bar_width + 1)
        side_y = self._clamp(top + 72, area.top(), area.bottom() - self.SIDE_HEIGHT + 1)

        def bottom_placement() -> Placement | None:
            bottom_y = bottom + self.GAP
            if bottom_y + self.BAR_HEIGHT <= area.bottom() + 1:
                return Placement(centered_x, bottom_y, bar_width, self.BAR_HEIGHT, "底部")
            return None

        def top_placement() -> Placement | None:
            top_y = top - self.GAP - self.BAR_HEIGHT
            if top_y >= area.top():
                return Placement(centered_x, top_y, bar_width, self.BAR_HEIGHT, "顶部")
            return None

        def right_placement() -> Placement | None:
            right_x = right + self.GAP
            if right_x + self.SIDE_WIDTH <= area.right() + 1:
                return Placement(right_x, side_y, self.SIDE_WIDTH, self.SIDE_HEIGHT, "右侧")
            return None

        def left_placement() -> Placement | None:
            left_x = left - self.GAP - self.SIDE_WIDTH
            if left_x >= area.left():
                return Placement(left_x, side_y, self.SIDE_WIDTH, self.SIDE_HEIGHT, "左侧")
            return None

        candidates = {
            "bottom": bottom_placement,
            "top": top_placement,
            "right": right_placement,
            "left": left_placement,
        }
        order = ["bottom", "top", "right", "left"]
        if self._placement_preference != "auto":
            order = [self._placement_preference, *[item for item in order if item != self._placement_preference]]
        for key in order:
            placement = candidates[key]()
            if placement is not None:
                return placement

        return None

    def _place_on_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        area = screen.availableGeometry()
        self.setFixedSize(min(self.MAX_WIDTH, area.width()), self.BAR_HEIGHT)
        self.move(area.left() + (area.width() - self.width()) // 2, area.bottom() - self.height() + 1)

    @staticmethod
    def _clamp(value: int, low: int, high: int) -> int:
        if high < low:
            return low
        return max(low, min(value, high))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        event.accept()


def _clip_label(value: str) -> str:
    value = value.strip() or "回复"
    return value if len(value) <= 6 else value[:5] + "…"


def _suggestion_preview(text: str, fallback_label: str) -> str:
    value = " ".join(text.split())
    if not value:
        return _clip_label(fallback_label)
    return value if len(value) <= 8 else value[:8] + "..."


def _context_label(app_label: str, contact_name: str, group_name: str, max_len: int = 34) -> str:
    app = (app_label or "聊天").strip()
    contact = (contact_name or "未确认联系人").strip()
    group = (group_name or "默认分组").strip()
    text = f"{app}·{contact}（{group}）"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _short_status(value: str) -> str:
    lowered = value.lower()
    if value == "local_preview":
        return "仅预览"
    if any(marker in lowered for marker in ("timeout", "timed_out", "deadline_exceeded", "超时")):
        return "已超时"
    if "contact_or_manual_protection" in lowered:
        return "生成受限"
    if any(marker in lowered for marker in ("manual_protection", "手动回复保护", "人工回复保护")):
        return "人工保护"
    if any(marker in lowered for marker in ("next_step_noop", "无需推进")):
        return "无需推进"
    if "do_not_contact" in lowered:
        return "暂不联系"
    if any(marker in lowered for marker in ("无需回复", "本次不生成")):
        return "无需回复"
    if value.startswith("blocked:provider_backoff"):
        return "AI退避"
    if value.startswith(("blocked:", "proactive:blocked:")) or any(marker in value for marker in ("已阻断", "被阻断", "暂不可生成")):
        return "已阻断"
    if any(marker in lowered for marker in ("cancelled", "superseded", "任务已取消", "已失效", "任务已结束")):
        return "已结束"
    if "失败" in value:
        return "失败"
    if value.startswith("proactive_notice:") and "正在" not in value:
        return "未生成"
    return "待确认"


def _terminal_status_detail(value: str) -> str:
    """Friendly state text without presenting an empty result as pending."""
    short = _short_status(value)
    return {
        "已超时": "本次等待已超时，未生成可用建议。晚到结果不会自动显示；可手动重新尝试。",
        "人工保护": "当前处于人工回复保护，本次未生成建议；不会自动解除保护或发起 AI 请求。",
        "生成受限": "客户状态尚未确认或处于人工保护，本次未生成；请在主窗口核对具体原因。",
        "无需回复": "本次已判断无需回复或推进，未生成话术，不是在等待 AI。",
        "无需推进": "本轮暂无合适的推进动作，未生成话术，不是在等待 AI。",
        "暂不联系": "本轮已判断暂不主动联系，不是在等待 AI。",
        "AI退避": "AI 暂时不可用，本次已停止生成，不会自动重试。",
        "已阻断": "本次生成被安全条件阻断，任务已结束；请查看主窗口中的具体原因。",
        "已结束": "本次任务已结束，旧结果不会覆盖当前客户的内容。",
        "失败": "本次生成失败，未获得可用建议；不会伪造话术或自动重试。",
        "未生成": "本次未生成建议，请查看主窗口中的明确原因；不表示仍在等待。",
        "仅预览": "未配置可用模型或未接入真实本地后端，本次未生成可复制建议；不能把占位预览作为合格话术。",
    }.get(short, "请查看主窗口中的当前状态。")


def _risk_status(result: ReplyGenerationResult) -> str:
    risks = {item.risk.lower() for item in result.suggestions}
    if "high" in risks:
        return "高风险"
    if "medium" in risks:
        return "中风险"
    return "可复制"


def _blend(left: QColor, right: QColor, right_ratio: float) -> QColor:
    ratio = max(0.0, min(1.0, right_ratio))
    return QColor(
        round(left.red() * (1 - ratio) + right.red() * ratio),
        round(left.green() * (1 - ratio) + right.green() * ratio),
        round(left.blue() * (1 - ratio) + right.blue() * ratio),
    )


def _luminance(color: QColor) -> float:
    return (0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()) / 255


def _readable_accent(base: QColor) -> QColor:
    if _luminance(base) < 0.38:
        return QColor("#a7f3d0")
    return _blend(base, QColor("#0f766e"), 0.52)
