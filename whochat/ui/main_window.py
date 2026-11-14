from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from whochat.ai.models import (
    ReplyContext,
    ReplyGenerationResult,
    ReplyMemoryContext,
    ReplySuggestion,
    reply_generation_result_from_json,
    reply_memory_item_count,
)
from whochat.ai.generator import test_ai_connection
from whochat.ai.current_work_authorization import authorize_current_work_context, cloud_reply_authorized, current_context_source_error
from whochat.ai.prompt import REPLY_POLICY_VERSION, PromptPreview, build_prompt_preview
from whochat.config import AppConfig, ConfigStore, TargetWindowConfig
from whochat.capture.screenshot import capture_rect
from whochat.core.models import AIAnalysisEvent, Contact, ContactStatus, ConversationType, IdentityStatus, Memory, MemoryKind, MemoryStatus, Message, Speaker, Strategy
from whochat.core.models import utc_now_iso
from whochat.core.current_conversation import build_current_message_view
from whochat.core.runtime import LayoutRegions, PageClassification, PageType, Rect, RuntimeState, TargetApp, ThemeMode, WindowState
from whochat.core.paths import app_data_dir
from whochat.diagnostics import diagnostics_log_path
from whochat.ocr.engine import PreviewOcrEngine, create_ocr_engine
from whochat.ocr.models import OcrResult
from whochat.ocr.parser import classify_page_from_ocr, parse_visible_messages
from whochat.services.ingestion import extract_title_candidates, normalize_contact_title, _title_confidence
from whochat.platform.window_tracker import diagnose_target_windows, foreground_window_handle
from whochat.security.redaction import redact_diagnostics_payload, redact_diagnostics_text
from whochat.security.secret_store import BACKEND_UNAVAILABLE as _SECRET_BACKEND_UNAVAILABLE
from whochat.services.bootstrap import AppServices, build_services
from whochat.services.current_identity import CurrentIdentityState, IdentityTransition
from whochat.services.manual_suggestions import (
    MANUAL_SUGGESTIONS_ONLY, check_current_screen_pixels, check_copy_screen_pixels,
    prepare_copy_target_foreground, copy_focus_poll,
)
from whochat.services.account_scope import account_avatar_screen_rect
from whochat.services.instant_activation import normalize_activation_title
from whochat.services.customer_profile_service import CustomerDashboard, DashboardFact, OrderRecord
from whochat.services.sales_followup import SALES_STAGE_LABELS, FollowupWorkbenchItem
from whochat.services.ui_data_tasks import UiDataTaskFailure, UiDataTaskResult, UiDataTaskService
from whochat.services.reply_tasks import ReplyTaskResult, ReplyTaskState
from whochat.services.reply import context_hash as _reply_context_hash
from whochat.services.proactive_contact import (
    ProactiveCloudAuthorizationGate,
    ProactiveContactOutcome,
    ProactiveContactRequest,
    ProactiveTaskResult,
    ProactiveTaskState,
    REJECTION_REASON_LABELS,
)
from whochat.services.status import build_status_chain, primary_action_step
from whochat.ui.calibration_overlay import CalibrationCanvas


_CONTACT_LONG_TERM_CLOUD_LABEL = "联系人级长期云端授权"
_CONTACT_LONG_TERM_CLOUD_HELP = (
    "仅控制此联系人的长期云端许可。关闭后，当前准确识别的工作客户仍可按操作者本次规则使用 AI。"
    "要停止 AI 回复，请进入手动回复保护，或在设置中禁用 AI。"
)


def _identity_evidence_from_runtime(state, account_scope_id: str, ingestion_result=None):
    return SimpleNamespace(
        job_id=getattr(ingestion_result, "job_id", None),
        hwnd=state.window.hwnd,
        process_id=state.window.process_id,
        window_generation=state.window_generation,
        capture_generation=state.capture_generation,
        account_identity_epoch=state.account_identity_epoch,
        account_scope_id=account_scope_id,
        target_instance_id=getattr(ingestion_result, "target_instance_id", "") or "manual-ui-evidence",
    )


@dataclass(frozen=True)
class SuggestionActionBinding:
    """Customer/context that owned a rendered suggestion or an edit dialog.

    Capture generation is deliberately absent: ordinary WhoChat focus changes
    cancel old OCR jobs, but do not change the customer being edited/copied.
    """

    identity_token: tuple[object, ...]
    mode: str
    context_key: str
    local_date: str


_USE_CURRENT_SUGGESTION_BINDING = object()


@dataclass(frozen=True)
class NavItem:
    key: str
    title: str
    subtitle: str


NAV_ITEMS = [
    NavItem("overview", "总览", "当前状态与建议"),
    NavItem("contacts", "聊天对象", "私聊、群聊与画像"),
    NavItem("customer_profile", "客户档案", "长期资料、记录与回访"),
    NavItem("followups", "待跟进", "跨工作微信销售待办"),
    NavItem("strategies", "分组", "目标与语气策略"),
    NavItem("memories", "记忆", "摘要与待确认信息"),
    NavItem("settings", "设置", "AI、隐私与采集"),
]


class StrategyDialog(QDialog):
    def __init__(self, parent: QWidget, strategy: Strategy | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑分组策略" if strategy else "新增分组策略")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        form = QGridLayout()
        self.name = QLineEdit(strategy.name if strategy else "")
        self.goal = QTextEdit(strategy.goal if strategy else "")
        self.goal.setMinimumHeight(88)
        self.mode = QLineEdit(strategy.mode if strategy else "")
        self.tone = QLineEdit(strategy.tone if strategy else "")
        self.avoid = QTextEdit(strategy.avoid if strategy else "")
        self.avoid.setMinimumHeight(68)
        self.reply_variants = QLineEdit(strategy.reply_variants if strategy else "稳妥版,简短版,推进版")
        self.requires_manual_reply = QCheckBox("只整理信息，不生成可直接复制的自动回复")
        self.requires_manual_reply.setChecked(strategy.requires_manual_reply if strategy else False)

        fields: list[tuple[str, QWidget]] = [
            ("分组名称", self.name),
            ("目标", self.goal),
            ("模式", self.mode),
            ("语气", self.tone),
            ("禁忌", self.avoid),
            ("回复变体", self.reply_variants),
            ("保护", self.requires_manual_reply),
        ]
        for row, (label, widget) in enumerate(fields):
            form.addWidget(QLabel(label), row, 0, Qt.AlignmentFlag.AlignTop)
            form.addWidget(widget, row, 1)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        return {
            "name": self.name.text().strip(),
            "goal": self.goal.toPlainText().strip(),
            "mode": self.mode.text().strip() or "自定义",
            "tone": self.tone.text().strip() or "自然、清晰",
            "avoid": self.avoid.toPlainText().strip(),
            "reply_variants": self.reply_variants.text().strip(),
            "requires_manual_reply": self.requires_manual_reply.isChecked(),
        }

    def _accept_if_valid(self) -> None:
        values = self.values()
        if not values["name"] or not values["goal"]:
            self.parentWidget().statusBar().showMessage("分组名称和目标不能为空", 3000)
            return
        self.accept()


class CalibrationDialog(QDialog):
    def __init__(self, parent: QWidget, layout: LayoutRegions, screenshot_path: Path | None = None) -> None:
        super().__init__(parent)
        self._layout = layout
        self._screenshot_path = screenshot_path
        self._ocr_engine = PreviewOcrEngine()
        self._ocr_result: OcrResult | None = None
        self.setWindowTitle("区域校准")
        self.setMinimumSize(960, 820)
        body = QVBoxLayout(self)
        body.setContentsMargins(18, 18, 18, 18)
        body.setSpacing(12)

        hint = QLabel("拖动截图上的区域或微调比例。保存后只记录相对窗口比例，不默认保存截图样本。")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        body.addWidget(hint)

        pixmap = QPixmap(str(screenshot_path)) if screenshot_path and screenshot_path.exists() else QPixmap()
        self.canvas = CalibrationCanvas(layout, pixmap if not pixmap.isNull() else None)
        self.canvas.setFixedHeight(320)
        self.canvas.layout_changed.connect(self._sync_spins_from_canvas)
        body.addWidget(self.canvas, 1)

        form = QGridLayout()
        self.name = QLineEdit("微信默认校准")
        self.nav_right = _ratio_spin(_ratio(layout.window_rect, layout.nav_rect.right, "x"))
        self.chat_list_right = _ratio_spin(_ratio(layout.window_rect, layout.chat_list_rect.right, "x"))
        self.title_bottom = _ratio_spin(_ratio(layout.window_rect, layout.title_rect.bottom, "y"))
        self.input_top = _ratio_spin(_ratio(layout.window_rect, layout.input_rect.top, "y"))
        self.theme = QComboBox()
        self.theme.addItems([ThemeMode.UNKNOWN.value, ThemeMode.LIGHT.value, ThemeMode.DARK.value])

        rows: list[tuple[str, QWidget]] = [
            ("名称", self.name),
            ("左侧导航右边界", self.nav_right),
            ("聊天列表右边界", self.chat_list_right),
            ("标题栏下边界", self.title_bottom),
            ("输入区上边界", self.input_top),
            ("主题", self.theme),
        ]
        for row, (label, widget) in enumerate(rows):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
        body.addLayout(form)

        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setMinimumHeight(110)
        self.preview = preview
        body.addWidget(preview)
        for widget in [self.nav_right, self.chat_list_right, self.title_bottom, self.input_top]:
            widget.valueChanged.connect(self._refresh_preview)
        self._refresh_ocr_preview(layout)
        self._refresh_preview()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        body.addWidget(buttons)

    def calibrated_layout(self) -> LayoutRegions:
        window = self.canvas.layout_regions.window_rect
        nav_right = _absolute_x(window, self.nav_right.value())
        chat_list_right = _absolute_x(window, self.chat_list_right.value())
        title_bottom = _absolute_y(window, self.title_bottom.value())
        input_top = _absolute_y(window, self.input_top.value())
        nav = Rect(window.left, window.top, nav_right, window.bottom)
        chat_list = Rect(nav_right, window.top, chat_list_right, window.bottom)
        content = Rect(chat_list_right, window.top, window.right, window.bottom)
        title = Rect(content.left, content.top, content.right, title_bottom)
        input_rect = Rect(content.left, input_top, content.right, window.bottom)
        message = Rect(content.left, title.bottom, content.right, input_rect.top)
        return LayoutRegions(
            target_app=self._layout.target_app,
            bubble_profile=self._layout.bubble_profile,
            window_rect=window,
            nav_rect=nav,
            chat_list_rect=chat_list,
            content_rect=content,
            title_rect=title,
            message_rect=message,
            input_rect=input_rect,
            confidence=0.92,
            source=self._layout.source,
            reason="用户手动校准预览",
            dpi_scale=getattr(self._layout, "dpi_scale", 1.0),
        )

    def values(self) -> dict:
        return {
            "name": self.name.text().strip() or "微信校准",
            "theme": ThemeMode(self.theme.currentText()),
            "layout": self.calibrated_layout(),
        }

    def _refresh_preview(self) -> None:
        layout = self.calibrated_layout()
        self.canvas.blockSignals(True)
        self.canvas.set_layout_regions(layout)
        self.canvas.blockSignals(False)
        self._refresh_ocr_preview(layout)
        self.preview.setText(
            "\n".join(
                [
                    f"nav={layout.nav_rect.as_tuple()}",
                    f"chat_list={layout.chat_list_rect.as_tuple()}",
                    f"message={layout.message_rect.as_tuple()}",
                    f"input={layout.input_rect.as_tuple()}",
                    "",
                    *self._ocr_summary_lines(),
                ]
            )
        )

    def _sync_spins_from_canvas(self, layout: LayoutRegions) -> None:
        self._set_spin_without_signal(self.nav_right, _ratio(layout.window_rect, layout.nav_rect.right, "x"))
        self._set_spin_without_signal(self.chat_list_right, _ratio(layout.window_rect, layout.chat_list_rect.right, "x"))
        self._set_spin_without_signal(self.title_bottom, _ratio(layout.window_rect, layout.title_rect.bottom, "y"))
        self._set_spin_without_signal(self.input_top, _ratio(layout.window_rect, layout.input_rect.top, "y"))
        self._refresh_ocr_preview(layout)
        self.preview.setText(
            "\n".join(
                [
                    f"nav={layout.nav_rect.as_tuple()}",
                    f"chat_list={layout.chat_list_rect.as_tuple()}",
                    f"message={layout.message_rect.as_tuple()}",
                    f"input={layout.input_rect.as_tuple()}",
                    "",
                    *self._ocr_summary_lines(),
                ]
            )
        )

    def _refresh_ocr_preview(self, layout: LayoutRegions) -> None:
        if not self._screenshot_path:
            self._ocr_result = None
            self.canvas.set_ocr_result(None)
            return
        self._ocr_result = self._ocr_engine.recognize(self._screenshot_path, layout)
        self.canvas.set_ocr_result(self._ocr_result)

    def _ocr_summary_lines(self) -> list[str]:
        if not self._ocr_result:
            return ["ocr_preview=unavailable"]
        page = classify_page_from_ocr(self._ocr_result, self.canvas.layout_regions)
        messages = parse_visible_messages(self._ocr_result, self.canvas.layout_regions)
        return [
            *self._ocr_result.summary_lines(8),
            f"page_preview={page.page_type.value} confidence={page.confidence:.2f}",
            *[f"message_preview={item.speaker.value} {item.confidence:.2f} partial={item.partial} {item.text}" for item in messages[:4]],
        ]

    def _set_spin_without_signal(self, spin: QDoubleSpinBox, value: float) -> None:
        spin.blockSignals(True)
        spin.setValue(max(0.0, min(value, 1.0)))
        spin.blockSignals(False)

    def _accept_if_valid(self) -> None:
        layout = self.calibrated_layout()
        if layout.nav_rect.width < 32 or layout.chat_list_rect.width < 120 or layout.message_rect.width < 240:
            self.parentWidget().statusBar().showMessage("区域过窄，请检查分割线比例", 3000)
            return
        if layout.title_rect.height < 32 or layout.input_rect.height < 60 or layout.message_rect.height < 160:
            self.parentWidget().statusBar().showMessage("标题、消息或输入区域高度不合理", 3000)
            return
        self.accept()


class TargetAppDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("新增目标应用")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        form = QGridLayout()
        self.label = QLineEdit()
        self.process_names = QLineEdit()
        self.title_keywords = QLineEdit()
        self.exclude_title_keywords = QLineEdit()
        for row, (label, widget) in enumerate([
            ("显示名", self.label),
            ("进程名", self.process_names),
            ("标题关键词", self.title_keywords),
            ("排除标题", self.exclude_title_keywords),
        ]):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        label = self.label.text().strip()
        return {
            "app_id": _target_app_id(label),
            "label": label,
            "process_names": _split_match_rules(self.process_names.text()),
            "title_keywords": _split_match_rules(self.title_keywords.text()),
            "exclude_title_keywords": _split_match_rules(self.exclude_title_keywords.text()),
        }

    def _accept_if_valid(self) -> None:
        values = self.values()
        if not values["label"]:
            self.parentWidget().statusBar().showMessage("目标应用显示名不能为空", 3000)
            return
        if not values["process_names"] and not values["title_keywords"]:
            self.parentWidget().statusBar().showMessage("至少填写一个进程名或标题关键词", 3000)
            return
        self.accept()


def _ratio(window: Rect, value: int, axis: str) -> float:
    if axis == "x":
        return (value - window.left) / max(1, window.width)
    return (value - window.top) / max(1, window.height)


def _absolute_x(window: Rect, ratio: float) -> int:
    return window.left + round(window.width * ratio)


def _absolute_y(window: Rect, ratio: float) -> int:
    return window.top + round(window.height * ratio)


def _ratio_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setDecimals(3)
    spin.setSingleStep(0.005)
    spin.setValue(max(0.0, min(value, 1.0)))
    return spin


def _days_spin(value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(1, 365)
    spin.setSuffix(" 天")
    spin.setValue(max(1, min(int(value), 365)))
    return spin


def _split_match_rules(value: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in value.replace("；", ",").replace(";", ",").replace("\n", ",").split(","):
        cleaned = item.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _target_app_id(label: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", label.strip().lower()).strip("_")
    return f"custom_{value or 'chat'}"


def _unique_target_app_id(base: str, existing: list[str]) -> str:
    used = set(existing)
    if base not in used:
        return base
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


def _compact_table(table: QTableWidget) -> None:
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(24)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setAlternatingRowColors(True)


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return ["暂无"]
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:] or ["暂无"]
    except OSError as exc:
        return [f"读取失败：{exc}"]


def _format_app_log(item) -> str:
    return f"[{item.level}] {item.ts} {item.module}.{item.event}: {item.message}"


def _clip_debug_text(value: str, limit: int = 48) -> str:
    text = " ".join(value.strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _avg_int(values: list[int]) -> str:
    return "-" if not values else str(round(sum(values) / len(values)))


def _avg_number(values: list[int]) -> float | None:
    return None if not values else sum(values) / len(values)


def _capture_perf_status(avg_total_ms: float | None) -> tuple[str, str]:
    if avg_total_ms is None:
        return "unknown", "等待更多采集样本"
    if avg_total_ms <= 15000:
        return "ok", "性能正常"
    if avg_total_ms <= 45000:
        return "warning", "偏慢；确认已启用标题/内容区裁剪并避免频繁滚动"
    return "slow", "过慢；建议保持 PaddleOCR worker 常驻、拉长自动 OCR 间隔或重新校准裁剪区域"


def _ingestion_action(accepted: bool, reason: str, has_contact: bool) -> str:
    if accepted and has_contact:
        return "身份已锁定；继续正常聊天，只有基线后的可靠新消息会进入分析"
    if reason.startswith("account_scope_unbound") or reason.startswith("account_target_instance_unavailable"):
        return "保持当前微信在前台完成一次采集，然后在设置中绑定当前微信账号"
    if reason.startswith("title_unavailable") or reason.startswith("contact_title_unavailable"):
        return "检查标题裁剪图和候选文本；必要时重新校准顶部标题区"
    if reason.startswith("title_ocr_warning") or reason.startswith("ocr_warning"):
        return "查看 ocr_worker 日志和 PaddleOCR 耗时；必要时等待熔断恢复或拉长间隔"
    if reason.startswith("page_blocked"):
        return "切回私聊或群聊页面，避免公众号、设置、搜索或文章页"
    return "查看标题 OCR、页面分类和最近日志定位阻断原因"


def _window_match_action(matched_foreground, foreground_excluded, matched_background: list) -> str:
    if matched_foreground is not None:
        return f"action=当前前台已命中：{matched_foreground.app_label}。可运行采集管线或等待自动采集。"
    if foreground_excluded is not None:
        return (
            f"action=当前前台是 {foreground_excluded.app_label} 的非聊天子窗口，"
            "已按排除规则暂停。关闭预览、设置或转发窗口，切回聊天主窗口。"
        )
    if matched_background:
        labels = ", ".join(f"{item.app_label}:{item.title}" for item in matched_background[:3])
        return f"action=发现已启用聊天窗口但不在前台：{labels}。切到目标聊天窗口后才会截图和 OCR。"
    return "action=发现相关窗口但未命中启用目标。检查进程名、标题关键词或排除标题设置。"


def _native_window_matches_evidence(hwnd: int | None, process_id: int | None) -> bool:
    if not hwnd or not process_id:
        return False
    try:
        import win32gui
        import win32process

        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return False
        _thread_id, actual_process_id = win32process.GetWindowThreadProcessId(hwnd)
        return int(actual_process_id) == int(process_id)
    except Exception:
        return False


def _reply_evidence_summary(context: ReplyContext, result: ReplyGenerationResult) -> str:
    contact = f"{context.contact.platform}·{context.contact.display_name}" if context.contact else "未识别对象"
    return (
        f"{contact}｜本次消息 {len(context.effective_messages)} 条｜参考资料 {reply_memory_item_count(context)} 条"
    )


def _reply_evidence_detail(context: ReplyContext, result: ReplyGenerationResult) -> str:
    contact = context.contact
    strategy = context.strategy
    return "\n".join(
        [
            f"allowed={result.allowed}",
            f"status={result.status}",
            f"provider={result.provider}",
            f"contact={contact.platform + '·' + contact.display_name if contact else '-'}",
            f"contact_status={contact.status.value if contact else '-'}",
            f"cloud_reply_authorized={cloud_reply_authorized(context)}",
            f"contact_cloud_ai={contact.allow_cloud_ai if contact else False}",
            f"strategy={strategy.name if strategy else '-'}",
            f"manual_protection={strategy.requires_manual_reply if strategy else False}",
            f"page={context.runtime.page.page_type.value}/{context.runtime.page.confidence:.2f}",
            f"messages={len(context.effective_messages)}",
            f"memories={reply_memory_item_count(context)}",
        ]
    )


def build_reply_context_for_analysis_event(
    services: AppServices,
    event: AIAnalysisEvent,
) -> ReplyContext | None:
    """Load AI input from the event's immutable identity/page binding."""
    contact = services.contacts.get(event.contact_id)
    conversation = services.conversations.get(event.conversation_id or "")
    if (
        contact is None
        or conversation is None
        or contact.account_scope_id != event.account_scope_id
        or conversation.account_scope_id != event.account_scope_id
        or conversation.contact_id != event.contact_id
    ):
        return None
    try:
        snapshot = json.loads(event.trigger_snapshot_json)
    except (TypeError, json.JSONDecodeError):
        return None
    try:
        page_type = PageType(str(snapshot.get("page_type_at_trigger") or ""))
        page_confidence = float(snapshot.get("page_confidence_at_trigger") or 0.0)
    except (TypeError, ValueError):
        return None
    if page_type not in {PageType.CHAT_DM, PageType.CHAT_GROUP, PageType.FILE_HELPER} or page_confidence < 0.65:
        return None
    if (
        str(snapshot.get("source_message_id") or "") != event.message_id
        or str(snapshot.get("account_scope_id") or "") != event.account_scope_id
        or str(snapshot.get("target_instance_id") or "") != event.target_instance_id
        or str(snapshot.get("contact_id") or "") != event.contact_id
        or str(snapshot.get("conversation_id") or "") != event.conversation_id
        or str(snapshot.get("identity_state_at_trigger") or "") != "stable"
    ):
        return None
    frozen_runtime = replace(
        services.runtime.state,
        page=PageClassification(page_type, page_confidence, "immutable_ai_trigger_snapshot"),
    )
    timeline = services.messages.list_timeline(conversation.id, 2000)
    source_index = next((index for index, message in enumerate(timeline) if message.id == event.message_id), None)
    if source_index is None:
        return None
    messages = timeline[max(0, source_index - 11):source_index + 1]
    memory_payload = snapshot.get("reply_memory_context")
    if isinstance(memory_payload, dict):
        try:
            memory_context = ReplyMemoryContext.from_payload(memory_payload)
        except ValueError:
            return None
        if (
            memory_context.account_scope_id != event.account_scope_id
            or memory_context.contact_id != event.contact_id
            or memory_context.conversation_id != event.conversation_id
            or memory_context.current_message_id != event.message_id
            or snapshot.get("reply_memory_hash") != memory_context.memory_hash
        ):
            return None
    else:
        memory_context = services.reply_memory_context.build(
            account_scope_id=event.account_scope_id,
            contact_id=event.contact_id,
            conversation_id=event.conversation_id or "",
            current_message_id=event.message_id,
        )
    source_message = next(
        (message for message in messages if message.id == event.message_id), None
    )
    knowledge_context = services.product_knowledge_context.build(
        memory_context=memory_context,
        current_message=source_message.text if source_message is not None else "",
    )
    return ReplyContext(
        runtime=frozen_runtime,
        contact=contact,
        strategy=services.strategies.get(contact.strategy_id),
        messages=messages,
        memories=services.memories.list_confirmed_for_contact(contact.id),
        memory_context=memory_context,
        current_message_id=event.message_id,
        knowledge_context=knowledge_context,
    )


def analysis_event_matches_binding(
    event: AIAnalysisEvent,
    *,
    account_scope_id: str,
    target_instance_id: str,
    contact_id: str,
    conversation_id: str,
    page_allowed: bool,
) -> bool:
    return bool(
        page_allowed
        and event.account_scope_id == account_scope_id
        and event.target_instance_id == target_instance_id
        and event.contact_id == contact_id
        and event.conversation_id == conversation_id
    )


def analysis_event_matches_display_binding(
    event: AIAnalysisEvent,
    *,
    account_scope_id: str,
    contact_id: str,
    conversation_id: str,
) -> bool:
    """Match a completed result to a durable customer conversation.

    A target instance is a runtime window identity and can legitimately change
    after the same manually bound work-WeChat reconnects.  Display recovery may
    therefore cross target incarnations, but never account, contact or
    conversation boundaries.  Provider execution keeps the stricter exact
    target-instance check in :func:`analysis_event_matches_binding`.
    """
    return bool(
        event.account_scope_id == account_scope_id
        and event.contact_id == contact_id
        and event.conversation_id == conversation_id
    )


def _redacted_config_snapshot(config: AppConfig) -> dict:
    snapshot = asdict(config)
    api_key = snapshot.get("ai", {}).get("api_key", "")
    snapshot["ai"]["api_key"] = "<set>" if api_key else "<empty>"
    return snapshot


def _secret_storage_note(secret_backend: str, has_key: bool) -> str:
    """Human-readable status for where the API key ended up.

    The key is never written to ``config.json``; when no secure backend exists it
    is deliberately dropped instead of being persisted in clear text, and the
    user is told so explicitly.
    """
    if secret_backend == _SECRET_BACKEND_UNAVAILABLE:
        return "安全存储不可用，API Key 未保存（不会以明文写入配置文件）" if has_key else "未设置 API Key"
    if has_key:
        return f"API Key 已加密保存（{secret_backend}），config.json 不含明文"
    return "未设置 API Key"


def _config_changes(before: dict, after: dict) -> dict:
    changes: dict[str, dict[str, object]] = {}

    def walk(prefix: str, left, right) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            keys = sorted(set(left) | set(right))
            for key in keys:
                path = f"{prefix}.{key}" if prefix else str(key)
                walk(path, left.get(key), right.get(key))
            return
        if left == right:
            return
        if prefix == "ai.api_key":
            changes[prefix] = {"old": "<changed>" if left != right else "<unchanged>", "new": "<changed>" if left != right else "<unchanged>"}
            return
        changes[prefix] = {"old": left, "new": right}

    walk("", before, after)
    return changes


def _status_reason_with_action(step) -> str:
    if not getattr(step, "action", ""):
        return step.reason
    if step.action in step.reason:
        return step.reason
    return f"{step.reason}｜建议：{step.action}"


def _message_speaker_label(message, speaker_map: dict[str, str]) -> str:
    base = speaker_map.get(message.speaker.value, message.speaker.value)
    sender_name = getattr(message, "sender_name", "")
    if sender_name and message.speaker != Speaker.ME:
        return f"{base}：{sender_name}"
    return base


def _message_content_key(speaker: Speaker, sender_name: str, text: str) -> tuple[str, str, str]:
    return (
        speaker.value,
        "".join(sender_name.split()).lower(),
        "".join(text.split()).lower(),
    )


def _message_key_counts(messages: list[Message]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for message in messages:
        key = _message_content_key(message.speaker, message.sender_name, message.text)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _primary_status_step(steps):
    return primary_action_step(steps)


def _should_show_overview_capture(state: RuntimeState) -> bool:
    if state.ocr_pending or state.pipeline_status == "running":
        return False
    if not state.capture_decision.should_capture:
        return False
    if not state.page.can_generate_reply:
        return True
    return state.pipeline_status in {"idle", "title_ready"} or state.pipeline_status.startswith("discarded:")


def _strategy_status_label(strategy: Strategy) -> str:
    prefix = "内置" if strategy.id in {"default", "manual_protect"} else "自定义"
    return f"{prefix} / 已归档" if strategy.archived else f"{prefix} / 启用"


def _contact_display_label(contact: Contact) -> str:
    platform = (contact.platform or "chat").strip()
    return f"{platform}·{contact.display_name}"


def _format_profile_time(value: str) -> str:
    if not value:
        return "时间未记录"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.strftime("%m/%d %H:%M")
    except (TypeError, ValueError):
        return str(value)[:16]


def _profile_time_for_edit(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)[:16]


def _elapsed_from_iso_ms(value: str) -> float:
    try:
        started = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds() * 1000.0, 3)
    except (TypeError, ValueError):
        return 0.0


def _iso_delta_ms(start_value: str, end_value: str) -> float:
    try:
        start = datetime.fromisoformat(str(start_value).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return round((end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() * 1000.0, 3)
    except (TypeError, ValueError):
        return 0.0


class MainWindow(QMainWindow):
    floating_visibility_changed = Signal(bool)
    targets_changed = Signal(object)
    capture_mode_changed = Signal(bool)

    def __init__(self, services: AppServices | None = None) -> None:
        super().__init__()
        self.setWindowTitle("WhoChat")
        self.resize(1280, 820)
        self.setMinimumSize(980, 640)
        self._services = services or build_services()
        self._ui_data_tasks = UiDataTaskService(self._services.logs, parent=self)
        self._floating = None
        self._manual_generation_enabled = MANUAL_SUGGESTIONS_ONLY
        self._manual_selection_plan = None
        self._screenshot_workflow = None
        self._local_preread = None
        self._manual_generation_phase = "idle"
        self._manual_generation_request_id = 0
        self._manual_generation_started = 0.0
        self._manual_generation_owner = None
        self._manual_generation_capture_ids: set[int] = set()
        self._manual_generation_completed_ids: set[int] = set()
        self._manual_generation_title_hash = ""
        self._manual_generation_source_result = None
        self._manual_generation_avatar_anchor = None
        self._manual_generation_job_id = None
        self._manual_generation_allow_submit = False
        self._manual_remember_only = False
        self._manual_activation_only = False
        self._manual_review_reason = ""
        self._manual_review_notice: QLabel | None = None
        self._manual_generation_timer = QTimer(self)
        self._manual_generation_timer.setInterval(750)
        self._manual_generation_timer.timeout.connect(self._manual_generation_tick)
        self._manual_generation_followup_pending = None
        if self._manual_generation_enabled:
            self._services.pipeline.capture_request_token_provider = self._manual_capture_request_token
            self._services.pipeline.manual_selection_provider = lambda: self._manual_selection_plan
            self._services.pipeline.activation_purpose_provider = lambda: self._manual_activation_only
        self._config_store = ConfigStore()
        self._config = self._config_store.load()
        self._log_text: QTextEdit | None = None
        self._pending_logs: list[str] = []
        self._last_ui_log_signature = ""
        self._last_duplicate_ui_refresh_monotonic = 0.0
        self._pending_auto_event_ids: list[str] = []
        self._analysis_event_by_job: dict[int, str] = {}
        self._reply_context_by_job: dict[int, ReplyContext] = {}
        self._latest_reply_job_id: int | None = None
        self._latest_proactive_job_id: int | None = None
        self._visible_reply_attempt_key = ""
        self._current_visible_ui_revision = ""
        self._last_chat_render_key = None
        self._nav_buttons: dict[str, QPushButton] = {}
        self._pages = QStackedWidget()
        self._page_titles: dict[str, str] = {}
        self._ai_status_value: QLabel | None = None
        self._ai_health_label: QLabel | None = None
        self._ai_action_status: QLabel | None = None
        self._page_status_value: QLabel | None = None
        self._page_status_subtitle: QLabel | None = None
        self._capture_status_value: QLabel | None = None
        self._capture_status_subtitle: QLabel | None = None
        self._runtime_text: QTextEdit | None = None
        self._environment_text: QTextEdit | None = None
        self._window_match_text: QTextEdit | None = None
        self._diagnostics_file_text: QTextEdit | None = None
        self._generation_log_text: QTextEdit | None = None
        self._log_level_filter: QComboBox | None = None
        self._contact_list: QListWidget | None = None
        self._contact_profile_text: QTextEdit | None = None
        self._contact_identity_text: QTextEdit | None = None
        self._contact_members_text: QTextEdit | None = None
        self._contact_chat_text: QTextEdit | None = None
        self._contact_memory_text: QTextEdit | None = None
        self._contact_feedback_text: QTextEdit | None = None
        self._contact_ai_toggle: QPushButton | None = None
        self._customer_profile_page: QWidget | None = None
        self._customer_profile_scroll: QScrollArea | None = None
        self._customer_profile_back_button: QPushButton | None = None
        self._customer_profile_breadcrumb: QLabel | None = None
        self._customer_profile_state: QLabel | None = None
        self._customer_profile_name: QLabel | None = None
        self._customer_profile_account: QLabel | None = None
        self._customer_profile_summary: QLabel | None = None
        self._customer_profile_stage: QLabel | None = None
        self._customer_profile_stage_combo: QComboBox | None = None
        self._customer_profile_context: QComboBox | None = None
        self._customer_profile_core_table: QTableWidget | None = None
        self._customer_profile_concerns: QLabel | None = None
        self._customer_profile_pending_table: QTableWidget | None = None
        self._customer_profile_timeline_filter: QComboBox | None = None
        self._customer_profile_timeline: QListWidget | None = None
        self._customer_profile_followups: QListWidget | None = None
        self._customer_profile_tracking_advice: QListWidget | None = None
        self._customer_profile_dashboard: CustomerDashboard | None = None
        self._customer_profile_revision = ""
        self._customer_profile_binding: tuple[str, str] | None = None
        self._customer_profile_event_limit = 15
        self._customer_profile_last_error = ""
        self._customer_profile_navigation_contact_id: str | None = None
        self._customer_profile_latest_request_key = ""
        self._customer_profile_inflight_keys: set[str] = set()
        self._profile_return_state: dict[str, object] | None = None
        self._followup_page: QWidget | None = None
        self._followup_table: QTableWidget | None = None
        self._followup_summary_overdue: QLabel | None = None
        self._followup_summary_today: QLabel | None = None
        self._followup_summary_future: QLabel | None = None
        self._followup_revision = ""
        self._followup_items: dict[str, FollowupWorkbenchItem] = {}
        self._followup_latest_request_key = ""
        self._followup_inflight = False
        self._followup_pending_refresh = False
        self._current_page_key = "overview"
        self._overview_status_summary: QLabel | None = None
        self._overview_next_action: QLabel | None = None
        self._overview_next_action_meta: QLabel | None = None
        self._overview_call_budget: QLabel | None = None
        self._overview_capture_button: QPushButton | None = None
        self._overview_status_table: QTableWidget | None = None
        self._overview_chat_table: QTableWidget | None = None
        self._last_ocr_messages = None
        self._last_ocr_result_meta: dict[str, object] | None = None
        self._last_ocr_contact_id: str | None = None
        self._overview_contact_values: dict[str, QLabel] = {}
        self._overview_contact_notes: QTextEdit | None = None
        self._strategy_table: QTableWidget | None = None
        self._strategy_search: QLineEdit | None = None
        self._strategy_show_archived: QCheckBox | None = None
        self._memory_tabs: QTabWidget | None = None
        self._suggestions_panel: QFrame | None = None
        self._suggestions_scroll: QScrollArea | None = None
        self._suggestions_title: QLabel | None = None
        self._suggestion_result: ReplyGenerationResult | None = None
        self._suggestion_context_hash: str | None = None
        self._suggestion_action_binding: SuggestionActionBinding | None = None
        self._suggestion_mode = "empty"
        self._displayed_analysis_event_id: str | None = None
        self._reply_retry_pending = False
        self._proactive_retry_pending = False
        self._proactive_last_request_key = ""
        self._proactive_activation_revisions: dict[int, tuple[str, str, int]] = {}
        self._proactive_cloud_gate = ProactiveCloudAuthorizationGate()
        self._proactive_outcome: ProactiveContactOutcome | None = None
        self._proactive_feedback_status: dict[int, str] = {}
        self._proactive_rejected_feedback: dict[int, tuple[str, str]] = {}
        self._proactive_regeneration_used = False
        self._instant_activation_contact_id: str | None = None
        self._instant_activation_pending_contact_id: str | None = None
        self._instant_activation_binding: tuple[object, ...] | None = None
        self._instant_activation_job_id = 0
        self._instant_activation_observed_sequence = 0
        self._pending_activation_refresh = None
        self._manual_proactive_binding = None
        self._reply_authorization_context = None
        self._identity_suggestions_suspended = False
        self._identity_safe_since_monotonic = 0.0
        self._proactive_stability_retry_pending = False
        self._privacy_long_memory: QCheckBox | None = None
        self._privacy_debug_screenshots: QCheckBox | None = None
        self._privacy_trim_cloud: QCheckBox | None = None
        self._privacy_cloud_review: QCheckBox | None = None
        self._privacy_manual_blocks: QCheckBox | None = None
        self._privacy_log_retention: QSpinBox | None = None
        self._privacy_debug_retention: QSpinBox | None = None
        self._privacy_capture_retention: QSpinBox | None = None
        self._privacy_calibration_retention: QSpinBox | None = None
        self._privacy_feedback_retention: QSpinBox | None = None
        self._ai_request_cooldown: QSpinBox | None = None
        self._ai_dedupe_minutes: QSpinBox | None = None
        self._ai_daily_limit: QSpinBox | None = None
        self._ai_failure_threshold: QSpinBox | None = None
        self._ai_failure_backoff: QSpinBox | None = None
        self._ocr_provider: QComboBox | None = None
        self._ocr_language: QLineEdit | None = None
        self._ocr_min_confidence: QDoubleSpinBox | None = None
        self._ocr_use_gpu: QCheckBox | None = None
        self._capture_debounce: QSlider | None = None
        self._capture_debounce_label: QLabel | None = None
        self._capture_ocr_interval: QSpinBox | None = None
        self._capture_auto_enabled: QCheckBox | None = None
        self._capture_foreground_only: QCheckBox | None = None
        self._capture_pause_unknown: QCheckBox | None = None
        self._capture_block_unconfirmed: QCheckBox | None = None
        self._account_scope_label: QLabel | None = None
        self._floating_placement: QComboBox | None = None
        self._floating_opacity: QSpinBox | None = None
        self._floating_suggestion_count: QSpinBox | None = None
        self._target_checkboxes: dict[str, QCheckBox] = {}
        self._target_process_inputs: dict[str, QLineEdit] = {}
        self._target_title_inputs: dict[str, QLineEdit] = {}
        self._target_exclude_title_inputs: dict[str, QLineEdit] = {}
        self._target_grid: QGridLayout | None = None
        self._provider_health_timer = QTimer(self)
        self._provider_health_timer.setInterval(30000)
        self._provider_health_timer.timeout.connect(self._refresh_provider_health)
        self._customer_profile_refresh_timer = QTimer(self)
        self._customer_profile_refresh_timer.setInterval(2500)
        self._customer_profile_refresh_timer.timeout.connect(self._poll_customer_profile_refresh)
        self._ui_event_loop_timer = QTimer(self)
        self._ui_event_loop_timer.setInterval(250)
        self._ui_event_loop_timer.timeout.connect(self._check_ui_event_loop_latency)
        self._ui_event_loop_expected = time.perf_counter() + 0.25
        self._last_ui_stall_log = 0.0
        self._services.runtime.capture_gate.policy = replace(
            self._services.runtime.capture_gate.policy,
            scroll_debounce_ms=self._config.capture.scroll_debounce_ms,
            ocr_min_interval_ms=self._config.capture.ocr_min_interval_ms,
        )
        self._services.autocapture.set_enabled(False if self._manual_generation_enabled else self._config.capture.auto_capture_enabled)

        root = QWidget()
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_sidebar())
        shell.addWidget(self._build_content(), 1)
        self.setCentralWidget(root)

        self._register_pages()
        self._select_page("overview")
        self._services.runtime.state_changed.connect(self.update_runtime_state)
        self._services.current_identity.transitioned.connect(self._on_current_identity_transition)
        identity_events = getattr(self._services, "identity_events", None)
        pipeline_events = identity_events or self._services.pipeline
        pipeline_events.context_changed.connect(self._on_pipeline_context_changed)
        pipeline_events.title_ready.connect(self._on_pipeline_title_ready)
        pipeline_events.result_ready.connect(self._on_pipeline_result_ready)
        self._services.pipeline.capture_rejected.connect(self._manual_generation_on_pipeline_result)
        if identity_events is not None:
            identity_events.account_revalidation_failed.connect(self._on_account_revalidation_failed)
            identity_events.account_revalidation_progress.connect(self._on_account_revalidation_progress)
            identity_events.fast_context_changed.connect(self._on_fast_probe_context_changed)
            identity_events.fast_title_ready.connect(self._on_fast_probe_title_ready)
        else:
            # Compatibility for standalone legacy UI fixtures. Production
            # AppServices supplies the accepted-event coordinator.
            self._services.fast_activation_probe.context_changed.connect(self._on_fast_probe_context_changed)
            self._services.fast_activation_probe.title_ready.connect(self._on_fast_probe_title_ready)
        self._services.pipeline.result_discarded.connect(self._on_pipeline_discarded)
        self._services.reply_tasks.result_ready.connect(self._on_reply_task_ready, Qt.ConnectionType.QueuedConnection)
        self._services.reply_tasks.result_discarded.connect(self._on_reply_task_discarded)
        self._services.reply_tasks.status_changed.connect(self._on_reply_task_status_changed)
        self._services.reply_tasks.task_state_changed.connect(self._on_reply_task_state_changed, Qt.ConnectionType.QueuedConnection)
        self._services.proactive_tasks.result_ready.connect(self._on_proactive_task_ready, Qt.ConnectionType.QueuedConnection)
        self._services.proactive_tasks.task_state_changed.connect(self._on_proactive_task_state_changed, Qt.ConnectionType.QueuedConnection)
        self._services.proactive_tasks.result_discarded.connect(self._on_proactive_task_discarded)
        self._services.proactive_tasks.status_changed.connect(self._on_proactive_task_status_changed)
        self._services.activation_snapshots.invalidator.invalidated.connect(
            self._on_activation_snapshot_invalidated
        )
        self._services.activation_prewarm.snapshot_ready.connect(self._on_activation_snapshot_ready)
        self._ui_data_tasks.result_ready.connect(self._on_ui_data_task_ready)
        self._ui_data_tasks.failed.connect(self._on_ui_data_task_failed)
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._ui_data_tasks.shutdown)
        self.append_log("app_started: WhoChat UI initialized")
        self.update_runtime_state(self._services.runtime.state)
        self._provider_health_timer.start()
        self._customer_profile_refresh_timer.start()
        self._ui_event_loop_timer.start()
        if self._manual_generation_enabled:
            self._init_local_preread()
            self._render_reply_empty_state("本地预读已启用：打开客户时提前读取可见聊天；只有点击「生成建议」才调用 AI，不自动发送")
            self._manual_generation_timer.start()

    def _init_local_preread(self) -> None:
        # Separate raw-computation channel: no old AutoCapture/IdentityEvent
        # scheduler is enabled and no background result is sent to ingestion.
        from whochat.services.local_preread import LocalPrereader
        self._local_preread = LocalPrereader(self._services.pipeline)
        self._services.pipeline.local_prereader = self._local_preread
        self._services.pipeline.local_preread_cache = self._local_preread.cache
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._local_preread.shutdown)

    def _local_preread_tick(self, state) -> None:
        reader = getattr(self, "_local_preread", None)
        if reader is None:
            return
        if (self._manual_generation_phase in {"selecting", "recognizing", "generating", "verifying"}
                or self._services.reply_tasks.is_running
                or self._services.proactive_tasks.is_running
                or QApplication.activeModalWidget() is not None):
            reader.pause()
            return
        reader.tick(state)

    def append_log(self, message: str, level: str = "info") -> None:
        signature = f"{level}|{message}"
        if signature == self._last_ui_log_signature:
            return
        self._last_ui_log_signature = signature
        line = f"[{level}] {datetime.now().strftime('%H:%M:%S')} {message}"
        if hasattr(self, "_services"):
            try:
                self._services.logs.append(level, "ui", "ui_log", message)
            except Exception:
                pass
        if self._log_text is None:
            # 独立「诊断控制台」页已并入「总览」，UI 不再挂载日志视图。
            # 保留一个有界缓冲（供诊断复制使用），避免长时间运行无界增长。
            self._pending_logs.append(line)
            del self._pending_logs[:-500]
            return
        if self._log_level_filter is not None and self._log_level_filter.currentText() != "全部":
            if level != self._log_level_filter.currentText():
                return
        self._log_text.append(line)

    def _record_ui_performance(
        self,
        operation: str,
        page: str,
        total_ms: float,
        *,
        main_thread_ms: float = 0.0,
        query_ms: float = 0.0,
    ) -> None:
        level = "warning" if main_thread_ms >= 500.0 or total_ms >= 1000.0 else "info"
        try:
            self._services.logs.append(
                level,
                "ui_performance",
                "operation_timing",
                f"{operation} page={page} total={total_ms:.1f}ms main={main_thread_ms:.1f}ms query={query_ms:.1f}ms",
                {
                    "operation": operation,
                    "page": page,
                    "total_ms": round(total_ms, 3),
                    "main_thread_ms": round(main_thread_ms, 3),
                    "query_ms": round(query_ms, 3),
                },
            )
        except Exception:
            pass

    def _check_ui_event_loop_latency(self) -> None:
        now = time.perf_counter()
        lag_ms = max(0.0, (now - self._ui_event_loop_expected) * 1000.0)
        self._ui_event_loop_expected = now + 0.25
        if lag_ms < 750.0 or now - self._last_ui_stall_log < 30.0:
            return
        self._last_ui_stall_log = now
        try:
            self._services.logs.append(
                "warning",
                "ui_performance",
                "event_loop_stall",
                f"Qt event loop delayed by {lag_ms:.1f}ms",
                {"lag_ms": round(lag_ms, 3), "page": self._current_page_key},
            )
        except Exception:
            pass

    def update_runtime_state(self, state: RuntimeState) -> None:
        if self._page_status_value is not None:
            self._page_status_value.setText(state.page.page_type.value)
        if self._page_status_subtitle is not None:
            title = state.window.title or "等待目标窗口"
            self._page_status_subtitle.setText(f"{state.status_label} · {title}")
        if self._capture_status_value is not None:
            self._capture_status_value.setText("允许" if state.capture_decision.should_capture else "暂停")
        if self._capture_status_subtitle is not None:
            self._capture_status_subtitle.setText(state.capture_decision.reason)
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(state))
        if self._window_match_text is not None:
            self._window_match_text.setText(self._format_window_match_diagnostics())
        if self._ai_health_label is not None:
            self._ai_health_label.setText(self._services.reply_generator.provider_health_summary())
        if self._diagnostics_file_text is not None:
            self._diagnostics_file_text.setText(self._format_diagnostics_files())
        self._refresh_account_scope_label()
        self.pause_button.setText("继续采集" if state.paused else "暂停采集")
        if not getattr(self, "_manual_generation_enabled", False):
            self._load_latest_displayable_ai_result()
        self._refresh_overview_data()
        self._sync_floating_content()

    def _refresh_provider_health(self) -> None:
        before = self._services.reply_generator.provider_health_summary()
        after = self._services.reply_generator.refresh_provider_health()
        if self._ai_health_label is not None:
            self._ai_health_label.setText(after)
        if before != after:
            self.append_log(f"ai_provider_health_changed: {after}")
            self._refresh_overview_data()

    def _format_runtime_state(self, state: RuntimeState) -> str:
        window_rect = state.window.rect.as_tuple() if state.window.rect else None
        lines = [
            f"target={state.window.target.value}",
            f"window_state={state.window.state.value}",
            f"title={state.window.title or '-'}",
            f"process={state.window.process_name or '-'}",
            f"process_id={state.window.process_id or '-'}",
            f"window_generation={state.window_generation} capture_generation={state.capture_generation}",
            f"rect={window_rect}",
            f"window_diagnostic={state.window.diagnostic or '-'}",
            f"page={state.page.page_type.value} confidence={state.page.confidence:.2f}",
            f"page_reason={state.page.reason}",
            f"capture={state.capture_decision.should_capture} reason={state.capture_decision.reason}",
            f"paused={state.paused}",
            f"ocr_pending={state.ocr_pending}",
            f"pipeline_status={state.pipeline_status}",
            f"reply_running={self._services.reply_tasks.is_running}",
            f"reply_status={self._services.reply_tasks.last_status}",
            f"provider_health={self._services.reply_generator.provider_health_summary()}",
            f"snapshot_hash={state.last_snapshot_hash or '-'}",
            f"visible_message_count={state.visible_message_count}",
            self._format_title_ocr_state(),
            self._format_ingestion_state(),
        ]
        if state.layout:
            lines.extend(
                [
                    f"layout_source={state.layout.source.value} confidence={state.layout.confidence:.2f}",
                    f"chat_list_rect={state.layout.chat_list_rect.as_tuple()}",
                    f"message_rect={state.layout.message_rect.as_tuple()}",
                    f"input_rect={state.layout.input_rect.as_tuple()}",
                    f"layout_reason={state.layout.reason}",
                ]
            )
        return "\n".join(lines)

    def _format_title_ocr_state(self) -> str:
        title = self._services.pipeline.last_title_result
        if title is None:
            return "title_ocr=none"
        boxes = sorted(title.ocr_result.boxes, key=lambda box: (-box.confidence, box.rect.top, box.rect.left))
        candidates = [
            f"{box.confidence:.2f}:{_clip_debug_text(box.text, 28)}"
            for box in boxes[:5]
            if box.text.strip()
        ]
        warning = title.ocr_result.warning or "-"
        crop = title.title_crop_rect.as_tuple() if title.title_crop_rect else None
        return (
            f"title_ocr=job:{title.job_id} target:{title.target_app}/{title.app_label} "
            f"account:{title.account_scope_name or 'unbound'} instance:{title.target_instance_id or '-'} "
            f"elapsed_ms:{title.elapsed_ms if title.elapsed_ms is not None else '-'} "
            f"warning:{warning} crop:{crop} candidates:{' | '.join(candidates) if candidates else '-'}"
        )

    def _format_ingestion_state(self) -> str:
        ingestion = self._services.ingestion.last_result
        if ingestion is None:
            return "contact_ingestion=none action:等待标题 OCR 或手动运行采集"
        contact = ingestion.contact
        contact_text = f"{contact.platform}/{contact.display_name}/{contact.status.value}" if contact else "-"
        candidates = " | ".join(_clip_debug_text(item, 36) for item in ingestion.title_candidates[:5])
        action = _ingestion_action(ingestion.accepted, ingestion.reason, contact is not None)
        return (
            f"contact_ingestion=accepted:{ingestion.accepted} reason:{ingestion.reason} "
            f"contact:{contact_text} inserted:{ingestion.inserted_messages} duplicate:{ingestion.duplicate_messages} "
            f"observation:{ingestion.observation_status or '-'}:{ingestion.observation_id or '-'} "
            f"title_candidates:{candidates or '-'} action:{action}"
        )

    def _manual_window_owner(self, state) -> tuple:
        return (state.window.hwnd, state.window.process_id, state.window_generation,
                state.account_identity_epoch, state.account_identity_runtime_key)

    def _freeze_manual_avatar_anchor(self, result) -> bool:
        self._manual_generation_avatar_error = "account_checkpoint_invalid"
        try:
            state = self._services.runtime.state
            checkpoint = self._services.account_scope_resolver.account_identity_checkpoint(state.window)
            if (checkpoint is None or checkpoint.identity_state != "trusted"
                    or checkpoint.hwnd != result.hwnd or checkpoint.process_id != result.process_id
                    or checkpoint.identity_epoch != result.account_identity_epoch
                    or checkpoint.runtime_key != result.account_identity_runtime_key
                    or checkpoint.account_scope_id != result.account_scope_id
                    or checkpoint.target_instance_id != result.target_instance_id):
                return False
            row = self._services.account_scopes.get_instance(checkpoint.target_instance_id)
            self._manual_generation_avatar_error = "account_instance_binding_invalid"
            if (row is None or not row.visual_reliable or not row.visual_hash
                    or row.id != result.target_instance_id or row.account_scope_id != result.account_scope_id
                    or row.runtime_key != checkpoint.runtime_key
                    or row.last_hwnd != result.hwnd or row.last_pid != result.process_id):
                return False
            # The database hash is an account matching reference, not the
            # exact pixels of this capture. Freeze the verified same-frame
            # fingerprint, while retaining the live checkpoint/instance gates.
            proof = result.screen_proof
            geometry = account_avatar_screen_rect(result.layout)
            fingerprint = proof.get("account_avatar_fingerprint", "")
            self._manual_generation_avatar_error = "account_capture_anchor_invalid"
            if (proof.get("account_avatar_reliable") is not True
                    or proof.get("account_avatar_geometry") != list(geometry.as_tuple())
                    or not isinstance(fingerprint, str) or len(fingerprint) != 67
                    or not fingerprint.startswith("v2:")
                    or any(char not in "0123456789abcdef" for char in fingerprint[3:])):
                return False
            self._manual_generation_avatar_anchor = {
                "expected_avatar_fingerprint": fingerprint,
                "expected_avatar_geometry": geometry,
            }
            self._manual_generation_avatar_error = ""
            return True
        except Exception:
            self._manual_generation_avatar_error = "account_capture_anchor_unavailable"
            return False

    def _manual_screen_check(self):
        return check_current_screen_pixels(
            self._manual_generation_source_result, self._services.runtime.state,
            account_guard=getattr(getattr(self._services, "account_scope_resolver", None),
                                  "validate_current_native_account", None),
            **(getattr(self, "_manual_generation_avatar_anchor", None) or {}),
        )

    def _manual_copy_handoff_windows(self) -> tuple[tuple[int, int], ...]:
        # Actual visible top-level widgets only, never window names/settings.
        handles = []
        for widget in (self, getattr(self, "_floating", None)):
            try:
                if widget is None or not widget.isWindow() or not widget.isVisible():
                    continue
                handle = int(widget.effectiveWinId())
                if handle > 0 and (handle, os.getpid()) not in handles:
                    handles.append((handle, os.getpid()))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return tuple(handles)

    def _manual_copy_screen_check(self):
        check = check_copy_screen_pixels(
            self._manual_generation_source_result, self._services.runtime.state,
            allowed_handoff_windows=self._manual_copy_handoff_windows(),
            account_guard=getattr(getattr(self._services, "account_scope_resolver", None),
                                  "validate_current_native_account", None),
            **(getattr(self, "_manual_generation_avatar_anchor", None) or {}),
        )
        if not check.unchanged and getattr(check, "diagnostic", None) is not None:
            # Only fixed reasons, numeric geometry and counts; no image/hash/text.
            try:
                self.append_log("manual_copy_pixels: " + json.dumps({
                    "request": self._manual_generation_request_id,
                    "job": self._manual_generation_source_result.job_id,
                    "entry": getattr(self, "_manual_copy_entry", "main"),
                    "reason": check.reason, "difference": check.diagnostic,
                }, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
            except Exception:
                pass  # Missing diagnostics cannot change the authorization verdict.
        return check

    def _manual_result_retention_is_current(self) -> bool:
        snapshot = getattr(self, "_frozen_screenshot_run", lambda: None)()
        if snapshot is not None and snapshot.retained():
            return True  # Presentation only; clipboard still checks live pixels.
        # Pure presentation predicate, not a display/copy/dispatch lease.
        copy_refresh = getattr(self, "_manual_copy_revalidation", None)
        if (not getattr(self, "_manual_generation_enabled", False)
                or (self._manual_generation_phase != "success"
                    and not (self._manual_generation_phase == "verifying" and copy_refresh is not None))):
            return False
        result, source = self._suggestion_result, self._manual_generation_source_result
        if copy_refresh is not None:
            if result is not copy_refresh["result"] or self._suggestion_action_binding != copy_refresh["binding"]:
                return False
            source = copy_refresh["source"]
        saved = getattr(self, "_suggestion_reply_context", None)
        binding = self._suggestion_action_binding
        identity, state = self._services.current_identity.snapshot, self._services.runtime.state
        return bool(result is not None and result.allowed and result.suggestions and source is not None
                    and saved is not None and saved.contact is not None and binding is not None
                    and identity.active and not self._identity_suggestions_suspended
                    and identity.conversation_token == binding.identity_token
                    and identity.contact_id == saved.contact.id
                    and identity.account_scope_id == saved.contact.account_scope_id
                    and self._manual_window_owner(state) == self._manual_generation_owner
                    and state.window.state == WindowState.VISIBLE and state.window.rect is not None)

    def _set_manual_review_status(self, reason: str) -> None:
        # Retaining text is NOT permission to copy, dispatch or renew a proof.
        previous = getattr(self, "_manual_review_reason", "")
        self._manual_review_reason = reason
        notice = getattr(self, "_manual_review_notice", None)
        detail = ("已保留本客户的建议（待核验）。来源过期时，点击复制会在本地重新核验；聊天已更新时，请点击「生成建议」。"
                  if reason else "")
        if notice is not None:
            notice.setText(detail)
            notice.setVisible(bool(reason))
        floating = getattr(self, "_floating", None)
        if floating is not None and hasattr(floating, "set_review_status"):
            floating.set_review_status(reason)
        banner = getattr(self, "_overview_next_action", None)
        if banner is not None and self._manual_generation_phase == "success":
            banner.setText("下一步：建议已保留；复制时核验，不会自动重新生成" if reason else
                           "下一步：本次建议已完成；聊天更新后请点击「生成建议」")
        if reason and reason != previous:
            self.append_log(f"manual_result_retained: request={self._manual_generation_request_id} reason={reason}")

    def _manual_completed_tick(self) -> None:
        # A completed result is not an in-flight capture. Temporary negative
        # evidence parks its original customer-bound text instead of deleting
        # it. Copy still needs the original lease AND a fresh negative check.
        state = self._services.runtime.state
        source = self._manual_generation_source_result
        result = self._suggestion_result
        if source is None or result is None or not result.allowed:
            return
        if self._manual_window_owner(state) != self._manual_generation_owner:
            self._finish_manual_generation("客户窗口或账号已变化，旧建议已撤下")
            return
        identity = self._services.current_identity.snapshot
        binding = self._suggestion_action_binding
        if identity.active and binding is not None and identity.conversation_token != binding.identity_token:
            self._finish_manual_generation("当前客户已变化，旧建议已撤下")
            return
        if not identity.active or self._identity_suggestions_suspended:
            self._set_manual_review_status("identity_pending")
            return
        if (not state.window.foreground or state.window.covered
                or state.window.state != WindowState.VISIBLE):
            self._set_manual_review_status("window_not_foreground")
            return
        check = self._manual_screen_check()
        from whochat.services.reply import screen_disposition, ScreenDisposition
        if screen_disposition(check) == ScreenDisposition.OWNER_CHANGED:
            self._manual_generation_note_blocker(check.reason)
            self._finish_manual_generation("客户或账号来源已变化，旧建议已撤下")
            return
        self._set_manual_review_status("" if check.unchanged else check.reason)

    def _manual_capture_request_token(self):
        # Called from capture workers too: only immutable local request fields,
        # no Qt, native APIs, database access or authority renewal here.
        deadline = (getattr(self, "_manual_generation_review_deadline", 0.0)
                    if self._manual_generation_phase == "verifying"
                    else self._manual_generation_started + 30.0)
        if (self._manual_generation_phase not in {"recognizing", "verifying"}
                or time.monotonic() >= deadline):
            return None
        return (self._manual_generation_request_id, self._manual_generation_owner)

    def _set_manual_generation_phase(self, phase: str) -> None:
        self._manual_generation_phase = phase
        floating = getattr(self, "_floating", None)
        if floating is not None and hasattr(floating, "set_generation_state"):
            floating.set_generation_state(phase)
        button = getattr(self, "_overview_capture_button", None)
        if button is not None:
            button.setEnabled(phase not in {"selecting", "recognizing", "generating", "verifying"})
            button.setText("识别中…" if phase == "recognizing" else "生成中…" if phase == "generating"
                           else "核验中…" if phase == "verifying" else "生成建议")
        remember = getattr(self, "_overview_remember_button", None)
        if remember is not None:
            remember.setEnabled(phase not in {"selecting", "recognizing", "generating", "verifying"})
        activate = getattr(self,'_overview_activation_button',None)
        if activate is not None:
            activate.setEnabled(phase not in {'selecting','recognizing','generating','verifying'})
        screenshot = getattr(self, '_overview_screenshot_button', None)
        if screenshot is not None:
            screenshot.setEnabled(phase not in {'selecting', 'recognizing', 'generating', 'verifying'})
            screenshot.setText('正在框选…' if phase == 'selecting' else '截图生成（手动框选聊天）')

    def _on_screenshot_generate_requested(self) -> None:
        if self._screenshot_workflow is None:
            from whochat.ui.screenshot_selection import ScreenshotWorkflow
            self._screenshot_workflow = ScreenshotWorkflow(self)
        self._screenshot_workflow.start()

    def _frozen_screenshot_run(self):
        from whochat.ui.frozen_screenshot import current_run
        return current_run(self)

    def _on_remember_requested(self) -> None:
        self._on_generate_requested(remember_only=True)

    def _on_generate_requested(self, *, remember_only: bool = False, activation_only: bool = False, _selection_plan=None) -> None:
        if self._manual_generation_phase in {"selecting", "recognizing", "generating", "verifying"}:
            return
        snapshot = getattr(self, "_frozen_screenshot_run", lambda: None)()
        if snapshot is not None:
            snapshot.cancel()
            self._screenshot_workflow.run = None
        self._manual_selection_plan = _selection_plan
        self._manual_remember_only = bool(remember_only)
        self._manual_activation_only = activation_only is True and not remember_only
        # A pre-read is not a manual task and must not make this button look
        # busy. Cancel pending background work; an executing native OCR call
        # cooperatively stops at its next boundary on the shared executor.
        reader = getattr(self, "_local_preread", None)
        if reader is not None:
            reader.pause()
        if self._services.pipeline.is_running or self._services.reply_tasks.is_running:
            self._set_manual_generation_phase("failed")
            self.statusBar().showMessage("上一任务仍在结束，请稍后再点；不会重复发起请求", 4000)
            return
        state = self._services.runtime.refresh_current()
        if state.window.hwnd is None or state.layout is None or state.paused:
            self._set_manual_generation_phase("failed")
            self._render_reply_empty_state("请先打开微信客户窗口，并保持窗口可见，再点「生成建议」")
            return
        self._manual_generation_request_id += 1
        self._manual_generation_followup_pending = None
        self._manual_generation_started = time.monotonic()
        self._manual_generation_owner = self._manual_window_owner(state)
        self._manual_generation_capture_ids = set()
        self._manual_generation_completed_ids = set()
        self._manual_generation_title_hash = ""
        self._manual_generation_last_blocker = ""
        self._manual_generation_body_hash = ""
        self._manual_generation_source_kind = ""
        self._manual_generation_source_result = None
        self._manual_generation_avatar_anchor = None
        self._manual_generation_job_id = None
        self._manual_generation_allow_submit = False
        self._manual_generation_terminal = False
        self._manual_generation_response_context = None
        self._manual_generation_review_deadline = 0.0
        self._set_manual_review_status("")
        self._pending_auto_event_ids.clear()
        self._reply_retry_pending = False
        self._services.proactive_tasks.invalidate("manual_button_request")
        self._services.pipeline.reset_snapshot_hash_cache()
        self._set_manual_generation_phase("recognizing")
        self._render_reply_empty_state(
            "正在核验并记住当前可见聊天；仅保存本地历史，不调用 AI、不扣额度。完成后回到最新聊天再点生成建议。"
            if remember_only else "正在核验当前客户；按你确认的旧话题已结束，只结合已关联历史生成激活建议，不解读旧附件。"
            if self._manual_activation_only else "正在核验当前客户并读取最新聊天；本次只生成一轮建议")
        self.append_log(f"manual_generation_requested: request={self._manual_generation_request_id}")
        self._manual_generation_tick()

    def _manual_generation_tick(self) -> None:
        if not getattr(self, "_manual_generation_enabled", False):
            return
        if getattr(self, '_screenshot_workflow', None) is not None and self._screenshot_workflow.busy:
            return
        phase = self._manual_generation_phase
        state = self._services.runtime.state
        preread_tick = getattr(self, "_local_preread_tick", None)
        if callable(preread_tick):
            preread_tick(state)
        snapshot = getattr(self, "_frozen_screenshot_run", lambda: None)()
        if snapshot is not None and snapshot.retained():
            return
        copy_refresh = getattr(self, "_manual_copy_revalidation", None)
        if copy_refresh is not None and not self._manual_copy_revalidation_current(copy_refresh):
            self._manual_copy_revalidation_end("copy_refresh_changed")
            return
        if phase == "success":
            self._manual_completed_tick()
            return
        if phase in {"recognizing", "generating"}:
            if time.monotonic() >= self._manual_generation_started + 30.0:
                from whochat.services.reply import capture_failure_message
                reason = (capture_failure_message(getattr(self, "_manual_generation_last_blocker", ""))
                          if phase == "recognizing" else "AI 未在本次时限内返回可用结果")
                blocker = getattr(self, "_manual_generation_last_blocker", "")
                if phase == "recognizing" and blocker.startswith("generation_focus_"):
                    from whochat.services.generation_focus import focus_message
                    self._finish_manual_generation(focus_message(blocker))
                    return
                self._finish_manual_generation("本次生成已超时：" + reason + "；未自动重试")
                return
        if phase in {"recognizing", "generating", "verifying"}:
            if self._manual_window_owner(state) != self._manual_generation_owner:
                from whochat.services.reply import account_cancellation_message
                self._finish_manual_generation(account_cancellation_message(
                    getattr(self, "_manual_generation_last_blocker", ""), self._services.runtime,
                    getattr(self._services, "account_scope_resolver", None)))
                return
        if phase == "verifying" and time.monotonic() >= self._manual_generation_review_deadline:
            self._finish_manual_generation("AI 已返回，但未能在本地核验期限内确认原客户；请重新点击生成建议")
            return
        if (phase == "recognizing" or (phase == "verifying" and copy_refresh is not None)) and (not state.window.foreground or state.window.covered
                or getattr(getattr(self, "_manual_generation_focus_ticket", None), "interrupted", False)):
            if not self._manual_generation_prepare_focus(state):
                return
        if phase == "generating" and self._manual_generation_source_result is not None:
            if (not state.window.foreground or state.window.covered
                    or state.window.state != WindowState.VISIBLE):
                return
            check = self._manual_screen_check()
            if not check.unchanged:
                from whochat.services.reply import screen_disposition, ScreenDisposition
                if getattr(self, "_manual_generation_last_blocker", "") != check.reason:
                    self._manual_generation_note_blocker(check.reason)
                if screen_disposition(check) == ScreenDisposition.OWNER_CHANGED:
                    self._finish_manual_generation("客户窗口或账号已变化，本次旧任务已取消")
                # No use of unverified text; keep the HTTP outcome private.
                return
        if phase not in {"recognizing", "verifying"} or self._services.pipeline.is_running:
            return
        # Discards are not verified frames. One in-flight capture plus the
        # absolute deadline bounds this work, not an arbitrary four submissions.
        state = self._services.runtime.refresh_current()
        job_id = self._services.pipeline.submit(state)
        if job_id is not None:
            self._manual_generation_capture_ids.add(job_id)

    def _manual_generation_prepare_focus(self, state) -> bool:
        # A click on our own button may take foreground from WeChat while OCR
        # is running. Discard those frames, acquire foreground once, then let
        # the ordinary two-frame workflow collect NEW evidence. Never renew
        # a source proof or reuse an old customer's result here.
        from whochat.services.generation_focus import GenerationFocus, focus_step, refresh_target, focus_message
        key = (self._manual_generation_request_id, self._manual_generation_owner)
        ticket = getattr(self, "_manual_generation_focus_ticket", None)
        if ticket is None or ticket.key != key:
            ticket = GenerationFocus(key)
            self._manual_generation_focus_ticket = ticket
        if state.window.foreground and not state.window.covered and not ticket.interrupted:
            return True
        if not ticket.interrupted:
            ticket.interrupted = True
            self._manual_generation_capture_ids.clear()
            self._manual_generation_completed_ids.clear()
            self._manual_generation_followup_pending = None
            self._services.pipeline.reset_snapshot_hash_cache()

        def current():
            return (getattr(self, "_manual_generation_focus_ticket", None) is ticket
                    and (self._manual_generation_phase == "recognizing"
                         or (self._manual_generation_phase == "verifying"
                             and getattr(self, "_manual_copy_revalidation", None) is not None))
                    and self._manual_generation_request_id == key[0]
                    and self._manual_window_owner(self._services.runtime.state) == key[1]
                    and time.monotonic() < self._manual_generation_started + 30.0)

        action, reason = focus_step(ticket, current=current,
            allowed_windows=self._manual_copy_handoff_windows(),
            modal=QApplication.activeModalWidget() is not None)
        if action == "refresh":
            try:
                restored = refresh_target(self._services.runtime, self._config.targets, ticket, current)
            except Exception:
                restored = False
            if restored:
                ticket.interrupted = False
                self._manual_generation_last_blocker = ""
                self.append_log(f"manual_generation_focus_restored: request={key[0]}; fresh_frames_required")
                self._manual_generation_schedule_focus_capture(ticket)
                return False  # Fresh capture starts on the next guarded tick.
            action, reason = "blocked", "generation_focus_refresh_unavailable"
        if action == "wait" and reason == "generation_focus_restoring" and ticket.attempted:
            self._manual_generation_schedule_focus_poll(ticket)
        if reason != ticket.reason:
            ticket.reason = reason
            self._manual_generation_note_blocker(reason)
            if action == "wait":
                self.statusBar().showMessage("正在恢复微信读取；若切换到其他软件，请回到原客户窗口继续", 3000)
        if action == "blocked":
            self._finish_manual_generation(focus_message(reason))
        return False

    def _manual_generation_schedule_focus_capture(self, ticket) -> None:
        # Resume at the existing capture gate's deadline, without waiting for
        # the 750ms watchdog. The callback still enters the full guarded tick.
        gate = self._services.runtime.capture_gate
        decision = gate.evaluate(self._services.runtime.state)
        delay = 0
        if not decision.should_capture:
            if decision.reason not in {"截图节流中", "窗口区域刚变化，等待稳定"}:
                return
            interval = (gate.policy.screenshot_min_interval_ms if decision.reason == "截图节流中"
                        else gate.policy.window_stable_delay_ms)
            delay = max(1, interval - (decision.elapsed_ms or 0))
        QTimer.singleShot(delay, self, lambda: self._manual_generation_focus_capture_tick(ticket))

    def _manual_generation_schedule_focus_poll(self, ticket) -> None:
        # Poll only our already-requested handoff. focus_step retains its
        # original single request, 2s bound, foreign-app and modal checks.
        if getattr(self, "_manual_generation_focus_poll_pending", None) is ticket:
            return
        self._manual_generation_focus_poll_pending = ticket
        QTimer.singleShot(50, self, lambda: self._manual_generation_focus_poll_tick(ticket))

    def _manual_generation_focus_poll_tick(self, ticket) -> None:
        if getattr(self, "_manual_generation_focus_poll_pending", None) is not ticket:
            return
        self._manual_generation_focus_poll_pending = None
        self._manual_generation_focus_capture_tick(ticket)

    def _manual_generation_focus_capture_tick(self, ticket) -> None:
        if (getattr(self, "_manual_generation_focus_ticket", None) is not ticket
                or ticket.key != (self._manual_generation_request_id, self._manual_generation_owner)
                or (self._manual_generation_phase != "recognizing"
                    and not (self._manual_generation_phase == "verifying"
                             and getattr(self, "_manual_copy_revalidation", None) is not None))
                or time.monotonic() >= self._manual_generation_started + 30.0):
            return
        self._manual_generation_tick()

    def _manual_generation_schedule_followup(self, result_job_id: int) -> None:
        # Only the first accepted independent frame earns a prompt continuation.
        # Discards/blocked proofs stay on the bounded watchdog, not a busy loop.
        if self._manual_generation_completed_ids != {result_job_id}:
            return
        ticket = (self._manual_generation_request_id, self._manual_generation_owner,
                  result_job_id, frozenset(self._manual_generation_capture_ids))
        if getattr(self, "_manual_generation_followup_pending", None) == ticket:
            return
        gate = self._services.runtime.capture_gate
        decision = gate.evaluate(self._services.runtime.state)
        delay = 0
        if not decision.should_capture:
            if decision.reason not in {"截图节流中", "窗口区域刚变化，等待稳定"}:
                return
            interval = (gate.policy.screenshot_min_interval_ms if decision.reason == "截图节流中"
                        else gate.policy.window_stable_delay_ms)
            delay = max(1, interval - (decision.elapsed_ms or 0))
        self._manual_generation_followup_pending = ticket
        QTimer.singleShot(delay, self, lambda: self._manual_generation_followup_tick(ticket))

    def _manual_generation_followup_tick(self, ticket) -> None:
        if getattr(self, "_manual_generation_followup_pending", None) != ticket:
            return
        self._manual_generation_followup_pending = None
        request_id, owner, job_id, capture_ids = ticket
        if (not self._manual_generation_enabled
                or self._manual_generation_request_id != request_id
                or self._manual_generation_owner != owner
                or self._manual_generation_phase not in {"recognizing", "verifying"}
                or self._manual_generation_completed_ids != {job_id}
                or self._manual_generation_capture_ids != capture_ids):
            return
        # Original tick still checks live ownership, deadline, busy state and a
        # freshly evaluated capture gate. No direct submit/AI call from here.
        self._manual_generation_tick()

    def _manual_generation_note_blocker(self, reason: str) -> None:
        # Diagnostic codes only: never log customer message bodies here.
        self._manual_generation_last_blocker = reason
        self.append_log(f"manual_generation_blocked: request={self._manual_generation_request_id} reason={reason}")

    def _manual_account_boundary_stop(self, result) -> bool:
        # A current, explicit account-selection blocker is not an OCR retry.
        # Never infer ownership from a nearest avatar or an unrelated callback.
        if (not getattr(self, "_manual_generation_enabled", False)
                or self._manual_generation_phase not in {"recognizing", "verifying"}
                or result.job_id not in self._manual_generation_capture_ids
                or self._manual_window_owner(self._services.runtime.state) != self._manual_generation_owner):
            return False
        current = getattr(self._services.runtime, "evidence_is_current", None)
        if not callable(current) or not current(result):
            return False
        from whochat.services.reply import account_confirmation_message
        message = account_confirmation_message(result)
        if not message:
            return False
        self._manual_generation_note_blocker("account_scope_unbound:" + result.account_scope_reason)
        self._finish_manual_generation(message)
        return True

    def _on_account_revalidation_progress(self, result) -> None:
        """A bound first-good recovery frame breaks consecutive bad frames."""
        runtime = self._services.runtime
        gate = getattr(runtime, "account_revalidation_progress_is_current", None)
        applied = getattr(runtime, "account_revalidation_progress_was_applied", None)
        if (not getattr(self, "_manual_generation_enabled", False)
                or self._manual_generation_phase not in {"recognizing", "verifying"}
                or result.job_id not in self._manual_generation_capture_ids
                or self._manual_window_owner(runtime.state) != self._manual_generation_owner
                or not callable(gate) or not gate(result)
                or not callable(applied) or not applied(result)):
            return
        deadline = (self._manual_generation_review_deadline if self._manual_generation_phase == "verifying"
                    else self._manual_generation_started + 30.0)
        if time.monotonic() >= deadline:
            return
        ticket = (self._manual_generation_request_id, self._manual_generation_owner,
                  result.account_identity_epoch)
        previous = getattr(self, "_manual_account_recovery_progress", None)
        if previous and previous[0] == ticket and result.job_id <= previous[1]:
            return
        self._manual_account_recovery_progress = (ticket, result.job_id)
        self._manual_account_failure_frames = None
        self._manual_generation_note_blocker("native_account_confirmation_pending")
        focus_ticket = getattr(self, "_manual_generation_focus_ticket", None)
        if (focus_ticket is not None and getattr(focus_ticket, "key", None)
                == (self._manual_generation_request_id, self._manual_generation_owner)):
            # Progress is not a positive customer frame. Only schedule the
            # existing guarded capture tick at the unchanged capture-gate time.
            self._manual_generation_schedule_focus_capture(focus_ticket)

    def _on_account_revalidation_failed(self, result) -> None:
        """Bound two independent negative frames to this manual request only."""
        gate = getattr(self._services.runtime, "account_revalidation_failure_is_current", None)
        if (not getattr(self, "_manual_generation_enabled", False)
                or self._manual_generation_phase not in {"recognizing", "verifying"}
                or result.job_id not in self._manual_generation_capture_ids
                or self._manual_window_owner(self._services.runtime.state) != self._manual_generation_owner
                or not callable(gate) or not gate(result)):
            return
        progress = getattr(self, "_manual_account_recovery_progress", None)
        progress_ticket = (self._manual_generation_request_id, self._manual_generation_owner,
                           result.account_identity_epoch)
        if progress and progress[0] == progress_ticket and result.job_id <= progress[1]:
            return
        ticket = (self._manual_generation_request_id, self._manual_generation_owner,
                  result.account_identity_epoch, result.account_scope_reason)
        previous = getattr(self, "_manual_account_failure_frames", None)
        jobs = set(previous[1]) if previous and previous[0] == ticket else set()
        jobs.add(result.job_id)
        self._manual_account_failure_frames = (ticket, frozenset(jobs))
        # Show the actual OCR diagnostic without accepting its account/customer
        # attribution or making it usable for replies, memory or copying.
        self._last_ocr_messages = list(result.messages)
        self._last_ocr_result_meta = {
            "job_id": result.job_id, "created_at": result.created_at,
            "page": result.page.page_type.value, "page_confidence": result.page.confidence,
            "warning": "工作微信账号核验未通过；本次消息未用于生成",
        }
        native_reason = str(result.account_scope_reason).removeprefix("account_identity_revalidating:")
        blocker = native_reason if native_reason.startswith("native_account_") else "account_avatar_revalidation_failed"
        self._manual_generation_note_blocker(blocker)
        if len(jobs) >= 2:
            from whochat.services.reply import capture_failure_message
            self._finish_manual_generation(capture_failure_message(blocker))
        self._refresh_overview_data()

    def _manual_generation_on_pipeline_result(self, result) -> None:
        if (self._manual_generation_phase not in {"recognizing", "verifying"}
                or result.job_id not in self._manual_generation_capture_ids
                or result.job_id in self._manual_generation_completed_ids):
            return
        copy_refresh = getattr(self, "_manual_copy_revalidation", None)
        if copy_refresh is not None and not self._manual_copy_revalidation_current(copy_refresh):
            self._manual_copy_revalidation_end("copy_refresh_changed")
            return
        reviewing = self._manual_generation_phase == "verifying"
        deadline = self._manual_generation_review_deadline if reviewing else self._manual_generation_started + 30.0
        if time.monotonic() >= deadline:
            self._finish_manual_generation("本次识别已超时，未自动重试")
            return
        if self._manual_window_owner(self._services.runtime.state) != self._manual_generation_owner:
            from whochat.services.reply import account_cancellation_message
            self._finish_manual_generation(account_cancellation_message(
                getattr(self, "_manual_generation_last_blocker", ""), self._services.runtime,
                getattr(self._services, "account_scope_resolver", None)))
            return
        if self._manual_account_boundary_stop(result):
            return
        title_hash = str(getattr(result, "title_exact_hash", "") or "")
        if not title_hash:
            self._finish_manual_generation("未能核验客户标题，请保持窗口清晰后重试")
            return
        if self._manual_generation_title_hash and title_hash != self._manual_generation_title_hash:
            self._finish_manual_generation("识别期间客户窗口发生变化，请重新点击生成建议")
            return
        proof = getattr(result, "screen_proof", None) or {}
        from whochat.services.reply import manual_capture_boundary_reason, manual_capture_boundary_message
        current = getattr(self._services.runtime, "evidence_is_current", None)
        boundary_reason = (manual_capture_boundary_reason(
            result, self._services.runtime.state, self._manual_generation_request_id)
            if callable(current) and current(result) else "")
        if boundary_reason:
            self._manual_generation_note_blocker(boundary_reason)
            self._finish_manual_generation(manual_capture_boundary_message(boundary_reason, reviewing=reviewing))
            return
        if result.page.page_type != PageType.CHAT_DM:
            self._finish_manual_generation("当前不是可确认的客户私聊窗口，请打开客户对话后重试")
            return
        if not isinstance(proof, dict) or proof.get("valid") is not True:
            # Failed captures are not independent positive frames. Forget
            # dedupe only, so a later valid pair can recover before deadline.
            self._manual_generation_completed_ids.clear()
            self._services.pipeline.forget_snapshot_hash(result.snapshot_hash)
            self._manual_generation_note_blocker(
                proof.get("rejection_code") or "current_view_unavailable"
                if isinstance(proof, dict) else "current_view_unavailable")
            if (not reviewing and isinstance(proof, dict)
                    and proof.get('rejection_code') == 'layout_or_coverage_incomplete'
                    and callable(current) and current(result)):
                # One additional independent capture, not repeated OCR until
                # the 30-second deadline. This grants no positive evidence.
                ticket = (self._manual_generation_request_id, title_hash)
                if getattr(self, '_manual_layout_failure_ticket', None) == ticket:
                    self._finish_manual_generation(
                        '当前客户聊天连续两次未能完整解析，已停止重复读取。'
                        '请核对识别结果，让被截断的聊天、图片或卡片完整显示后重试；'
                        '本次未调用 AI、未扣额度。')
                    return
                self._manual_layout_failure_ticket = ticket
            return
        source_kind = proof.get("source_kind", "visible_observation")
        previous_kind = getattr(self, "_manual_generation_source_kind", "")
        if previous_kind and source_kind != previous_kind:
            self._finish_manual_generation("识别期间聊天内容类型发生变化，请重新点击生成建议")
            return
        if source_kind == "current_window_reactivation":
            body = proof.get("window_body_sha256", "")
            structure = (proof.get("window_structure") or {}).get("structure_sha256", "")
            if (proof.get("valid") is not True or proof.get("window_capture_complete") is not True
                    or proof.get("manual_request_id") != self._manual_generation_request_id
                    or not all(isinstance(value, str) and len(value) == 64
                               and all(char in "0123456789abcdef" for char in value)
                               for value in (body, structure))):
                self._manual_generation_note_blocker("window_reactivation_capture_invalid")
                return
            hashes = (body, structure)
            previous = getattr(self, "_manual_generation_body_hash", "")
            if previous and hashes != previous:
                self._finish_manual_generation("识别期间图片或聊天画面发生变化，请重新点击生成建议")
                return
            self._manual_generation_body_hash = hashes
        self._manual_generation_source_kind = source_kind
        self._manual_generation_title_hash = title_hash
        self._manual_generation_completed_ids.add(result.job_id)
        # Each independent capture still executes live account/title/body
        # verification. Forget only snapshot dedupe, never identity evidence.
        self._services.pipeline.forget_snapshot_hash(result.snapshot_hash)
        if len(self._manual_generation_completed_ids) < 2:
            self._manual_generation_schedule_followup(result.job_id)
            return
        ingestion = self._services.ingestion.last_result
        if not (ingestion is not None and ingestion.accepted and ingestion.identity_stable
                and ingestion.contact is not None and ingestion.observation_id):
            self._manual_generation_note_blocker("identity_pending")
            return
        context = self._build_reply_context()
        source_error = current_context_source_error(context)
        if context.current_message_view is None or source_error:
            proof_error = (getattr(result, "screen_proof", None) or {}).get("rejection_code", "")
            blocker = proof_error or source_error or "current_view_unavailable"
            self._manual_generation_note_blocker(blocker)
            if callable(current) and current(result):
                ticket = (self._manual_generation_request_id, self._manual_generation_owner,
                          title_hash, ingestion.contact.id, blocker)
                previous = getattr(self, "_manual_source_failure_frames", None)
                jobs = set(previous[1]) if previous and previous[0] == ticket else set()
                jobs.add(result.job_id)
                self._manual_source_failure_frames = (ticket, frozenset(jobs))
                if len(jobs) >= 2:
                    message = ('已识别客户并读到聊天，但当前聊天记录的来源核验未通过，'
                               '已停止重复读取。')
                    message += ('本次未调用 AI、未扣额度。' if not reviewing
                                else '未展示未通过复核的建议，未追加 AI 请求。')
                    self._finish_manual_generation(message)
            return
        self._manual_source_failure_frames = None
        validator = self._services.reply_generator.identity_lease_validator
        if not callable(validator) or not validator(context):
            self._manual_generation_note_blocker("identity_lease_invalid")
            return
        self._manual_generation_source_result = result
        if not self._freeze_manual_avatar_anchor(result):
            self._manual_generation_note_blocker(self._manual_generation_avatar_error)
            self._finish_manual_generation("账号或聊天画面尚未通过本次核验，请保持窗口不变后重试")
            return
        check = self._manual_screen_check()
        if not check.unchanged:
            self._manual_generation_note_blocker(check.reason)
            self._finish_manual_generation("当前画面复核未通过，本次未调用 AI；请保持微信客户窗口可见后重试")
            return
        if reviewing:
            if copy_refresh is not None:
                self._manual_copy_revalidation_ready(context)
                return
            original = self._manual_generation_response_context
            if original is None or _reply_context_hash(context) != _reply_context_hash(original):
                self._finish_manual_generation("客户或聊天内容已更新，旧建议不再使用；请点击生成建议处理最新内容")
                return
            # Fresh LOCAL verification clock only. Never call submit/generate
            # from this phase, even on a miss. No second HTTP or quota charge.
            now = time.monotonic()
            local = replace(context, manual_retry_id="", current_work_started_monotonic=now,
                            current_work_deadline_monotonic=min(now + 30.0, deadline))
            recovered = self._services.reply_generator.recover_cached_only(local, self._config)
            if recovered is None:
                self._finish_manual_generation("本地恢复未通过核验或结果已过期；可点击生成建议重新处理")
                return
            self._render_reply_suggestions(recovered, context=local)
            self._suggestion_context_hash = _reply_context_hash(local)
            self._finish_manual_generation("本次建议已完成（已重新核验，未重复调用 AI）", success=True)
            return
        # This branch is reached only after both independent frames, DB source,
        # live customer/account lease, avatar and final native screen check.
        # Recording an old page is not a new incoming message or a cloud grant.
        remember_only = getattr(self, "_manual_remember_only", False)
        history_builder = getattr(self._services, "reply_memory_context", None)
        recorder = getattr(history_builder, "record_current_view", None)
        history_result = {"status": "unavailable", "stored": 0, "duplicates": 0}
        if callable(recorder):
            try:
                history_result = recorder(context.current_message_view)
            except Exception:
                self.append_log("observed_history_record_failed")
        if remember_only:
            if history_result.get("status") == "recorded":
                detail = (f"已核验并保存客户原话 {history_result.get('stored', 0)} 条"
                          f"（已记住的 {history_result.get('duplicates', 0)} 条未重复保存）。")
            elif history_result.get("status") == "no_usable_customer_text":
                detail = "本页没有可保存的完整客户文字；图片、未转文字语音和收货隐私不作为客户事实记忆。"
            elif history_result.get("status") == "disabled":
                detail = "设置中的长期记忆已关闭，本次未保存客户历史。"
            else:
                detail = "本次未能保存历史，请保持完整客户文字可见后重试；未伪报记住。"
            self._finish_manual_generation(detail + "未调用 AI、未扣额度。请回到最新聊天再点「生成建议」。")
            # A local-only completion is not an AI result and cannot become a
            # copyable success. Explicit capture has retired the old binding.
            self._set_manual_generation_phase("idle")
            self._manual_remember_only = False
            return
        if callable(recorder) and history_result.get("status") not in {"recorded", "no_usable_customer_text", "disabled"}:
            self.statusBar().showMessage("本地历史未保存；本次仍按已核验的聊天生成，不代表已完成建档", 8000)
        self._set_manual_generation_phase("generating")
        self._manual_generation_allow_submit = True
        try:
            job_id = self._refresh_reply_suggestions(automatic=False)
        finally:
            self._manual_generation_allow_submit = False
        self._manual_generation_job_id = job_id
        if job_id is None:
            ready = self._suggestion_result
            success = bool(ready is not None and ready.allowed and ready.suggestions)
            self._finish_manual_generation(
                "已恢复本客户相同聊天的已有建议" if success else "本次未生成可用建议，请核对界面提示后重试",
                success=success,
            )

    def _finish_manual_generation(self, reason: str, *, success: bool = False) -> None:
        if getattr(self, "_manual_copy_revalidation", None) is not None:
            self._manual_copy_revalidation_end("copy_refresh_failed")
            return
        if getattr(self, "_manual_generation_terminal", False) and self._manual_generation_phase == "failed":
            return
        # Commit terminal ownership BEFORE emitting cancellation signals. A
        # queued/direct terminal callback must not overwrite the root reason.
        self._manual_generation_terminal = True
        self._manual_generation_followup_pending = None
        self._manual_generation_job_id = None
        self._set_manual_generation_phase("success" if success else "failed")
        if "http_budget_exhausted" in reason:
            reason = "本地生成额度已用完（不是 AI 账户余额不足）；已有建议可在相同聊天核验后复用"
        elif "http_budget_unavailable" in reason:
            reason = "本地调用额度账本暂不可用；本次未发出 AI 请求，也未自动重试"
        elif any(code in reason for code in ("current_work_timeout", "provider_timeout", "task_deadline_exceeded")):
            reason = "本次生成超时：从点击起 30 秒内未完成识别与 AI 回复；未得到可用建议。可点击「重新生成」再试，软件不会自动重试"
        elif "manual_attempt_consumed" in reason:
            reason = "本次点击已处理，不会重复请求；如需重试，请再次点击「重新生成」"
        elif "duplicate_context" in reason:
            reason = "相同聊天近期已有实际 AI 请求，但没有可复用的合格结果；本次未再次调用，请查看原失败原因"
        elif "dosage_requires_explicit_operator_review" in reason:
            reason = "AI 在新建议中写入了未经本次核验的数量或用量（旧聊天记录不等于本次用量授权），已拦截；实际调用已计次。不是缺少确认按钮，也未自动重试"
        elif "unreviewed_quantity_marker_in_result" in reason:
            reason = "AI 未正确处理待核对的历史数值，建议未展示；实际调用已计次，未自动重试"
        elif "agronomy_premise_" in reason:
            from whochat.ai.agronomy_grounding import premise_failure_message
            reason = premise_failure_message(reason) or "AI 草稿存在未经核实的农业判断，已拦截；未自动重试"
        if "blocked:invalid_analysis:" in reason:
            from whochat.services.status import reply_content_failure_message
            reason = reply_content_failure_message(reason) or reason
        self._manual_generation_allow_submit = False
        if not success:
            self._manual_review_reason = ""
            self._services.reply_tasks.invalidate("manual_request_ended")
            self._latest_reply_job_id = None
            self._manual_generation_source_result = None
            self._suggestion_action_binding = None
            self._suggestion_context_hash = None
            self._render_reply_empty_state(reason)
        self.append_log(f"manual_generation_finished: request={self._manual_generation_request_id} success={success} reason={reason}")
        self._refresh_overview_data()

    def _manual_generation_budget_text(self) -> str:
        # The launcher injects an in-memory snapshot; no ledger/HTTP on UI refresh.
        provider = getattr(self._services.reply_generator, "call_budget_status_provider", None)
        if not callable(provider):
            return ""
        try:
            status = provider()
            remaining, limit = status.get("new_remaining"), status.get("grant_limit")
            if (type(remaining) is not int or type(limit) is not int
                    or not 0 <= remaining <= limit or limit <= 0):
                return "本地生成额度暂不可核验；不会自动增加或重置"
            suffix = "；已用完，已有合格缓存仍可复用" if remaining == 0 else "；剩余较少" if remaining <= 20 else ""
            return f"本地人工生成额度：剩余 {remaining} / {limit} 次{suffix}（非服务商账户余额，重启不重置）"
        except Exception:
            return "本地生成额度暂不可核验；不会自动增加或重置"

    def _manual_generation_refresh_budget_status(self) -> None:
        label = getattr(self, "_overview_call_budget", None)
        if label is not None:
            text = self._manual_generation_budget_text()
            label.setText(text)
            label.setVisible(bool(text))
        floating = getattr(self, "_floating", None)
        if floating is not None and hasattr(floating, "generate_button"):
            text = self._manual_generation_budget_text()
            if text:
                floating.generate_button.setToolTip(text + "\n点击后核验当前客户再生成；不会自动发送")

    def _manual_generation_on_reply_ready(self, task) -> None:
        context = self._reply_context_by_job.pop(task.job_id, None)
        if (self._manual_generation_phase != "generating"
                or task.job_id != self._manual_generation_job_id
                or task.job_id != self._latest_reply_job_id or context is None):
            return
        deadline = getattr(task, "current_work_deadline_monotonic", None)
        if (time.monotonic() >= self._manual_generation_started + 30.0
                or deadline is None or time.monotonic() >= deadline):
            self._finish_manual_generation("本次生成已超时，迟到结果未展示")
            return
        # Display the frozen request's answer without another screenshot,
        # native account scan, business-context rebuild or OCR recovery round.
        # These cheap ownership checks do NOT certify the current screen.
        # The ordinary copy callback still performs fresh live validation.
        if (self._manual_window_owner(self._services.runtime.state) != self._manual_generation_owner
                or task.context_digest != _reply_context_hash(context)
                or context.contact is None or task.contact_id != context.contact.id
                or task.hwnd != context.runtime.window.hwnd
                or self._identity_suggestions_suspended
                or self._bind_suggestion_actions("reply", context) is None):
            self._finish_manual_generation("生成任务对应的客户已变化，本次结果未展示")
            return
        result = task.result
        if not result.allowed:
            # A completed, structurally valid answer may have been parked by
            # the worker's stricter live-source gate. Read it only as the
            # original customer's answer, never as a fresh source/AI grant.
            snapshot = self._services.reply_generator.completed_snapshot_for_display(context, self._config)
            if snapshot is not None:
                result = snapshot
        if (time.monotonic() >= self._manual_generation_started + 30.0
                or time.monotonic() >= deadline):
            self._finish_manual_generation("本次生成已超时，迟到结果未展示")
            return
        if result.allowed and result.suggestions:
            # Establish presentation state first: both widgets use retained
            # ownership for button rendering, not fresh clipboard authority.
            self._set_manual_generation_phase("success")
            self._render_reply_suggestions(result, context=context)
            self._suggestion_context_hash = task.context_digest
            self.append_log(f"manual_result_direct_display: request={self._manual_generation_request_id} original_customer=True")
            self._finish_manual_generation("本次建议已完成", success=True)
        else:
            self._finish_manual_generation(f"本次未生成可用建议：{result.status}")

    def _manual_generation_begin_verification(self, original) -> None:
        """An explicit click's completed HTTP response awaits LOCAL checks."""
        self._manual_generation_response_context = deepcopy(original)
        self._manual_generation_review_deadline = time.monotonic() + 90.0
        self._manual_generation_capture_ids = set()
        self._manual_generation_completed_ids = set()
        self._manual_generation_title_hash = ""
        self._manual_generation_body_hash = ""
        self._manual_generation_source_kind = ""
        self._manual_generation_job_id = None
        self._latest_reply_job_id = None
        self._manual_generation_allow_submit = False
        self._services.pipeline.reset_snapshot_hash_cache()
        self._set_manual_generation_phase("verifying")
        self._render_reply_empty_state("AI 已返回；请回到原客户窗口，正在本地核验后恢复，不会再次调用 AI")
        self.append_log(f"manual_generation_review_pending: request={self._manual_generation_request_id}")

    def attach_floating_widget(self, floating: QWidget) -> None:
        self._floating = floating
        if hasattr(floating, "set_result_retention_guard"):
            floating.set_result_retention_guard(self._manual_result_retention_is_current)
        if hasattr(floating, "generate_requested"):
            floating.generate_requested.connect(self._on_screenshot_generate_requested)
            if hasattr(floating, 'set_screenshot_mode'):
                floating.set_screenshot_mode(True)
        if hasattr(floating, "set_generation_state"):
            floating.set_generation_state(getattr(self, "_manual_generation_phase", "idle"))
        if hasattr(floating, "set_copy_handler"):
            floating.set_copy_handler(self._copy_floating_suggestion)
        if hasattr(floating, "apply_preferences"):
            floating.apply_preferences(
                placement_preference=self._config.floating.placement_preference,
                opacity_percent=self._config.floating.opacity_percent,
                suggestion_count=self._config.floating.suggestion_count,
            )
        if hasattr(floating, "suggestion_copied"):
            floating.suggestion_copied.connect(self._on_suggestion_copied)
        if hasattr(floating, "user_hidden_changed"):
            floating.user_hidden_changed.connect(self._on_floating_user_hidden_changed)
        if hasattr(floating, "calibration_requested"):
            floating.calibration_requested.connect(self._open_calibration_dialog)
        self._sync_floating_content()

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(168)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(6)

        brand = QLabel("WhoChat")
        brand.setObjectName("BrandMark")
        sub = QLabel("本地 AI 回复助手")
        sub.setObjectName("BrandSub")
        layout.addWidget(brand)
        layout.addWidget(sub)
        layout.addSpacing(8)

        for item in NAV_ITEMS:
            button = QPushButton(item.title)
            button.setObjectName("NavButton")
            button.setToolTip(item.subtitle)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setProperty("active", False)
            button.clicked.connect(lambda checked=False, key=item.key: self._select_page(key))
            self._nav_buttons[item.key] = button
            layout.addWidget(button)

        layout.addStretch(1)

        status = QLabel("本地优先")
        status.setObjectName("BrandSub")
        layout.addWidget(status)
        return sidebar

    def _build_content(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_topbar())
        layout.addWidget(self._pages, 1)
        return content

    def _build_topbar(self) -> QWidget:
        top = QFrame()
        top.setObjectName("TopBar")
        top.setFixedHeight(50)
        layout = QHBoxLayout(top)
        layout.setContentsMargins(18, 8, 18, 8)

        title_box = QVBoxLayout()
        self.title_label = QLabel("总览")
        self.title_label.setObjectName("PageTitle")
        self.subtitle_label = QLabel("当前仍是原型骨架，用于验证窗口、设置和交互设计。")
        self.subtitle_label.setObjectName("Muted")
        self.subtitle_label.setVisible(False)
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        layout.addLayout(title_box, 1)

        self.pause_button = QPushButton("暂停采集")
        self.pause_button.setVisible(not self._manual_generation_enabled)
        self.pause_button.setObjectName("DangerButton")
        self.pause_button.clicked.connect(self._toggle_capture_pause)
        self.float_button = QPushButton("显示悬浮窗")
        self.float_button.setObjectName("PrimaryButton")
        self.float_button.clicked.connect(self._toggle_floating)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.float_button)
        return top

    def _register_pages(self) -> None:
        pages = {
            "overview": ("总览", "当前会话、建议、风险状态和快速操作。", self._build_overview_page()),
            "contacts": ("聊天对象", "私聊、群聊、身份别名和聊天记录。", self._build_contacts_page()),
            "customer_profile": ("客户档案", "当前客户的长期资料、业务记录、来源与回访。", self._build_customer_profile_page()),
            "followups": ("待跟进", "跨工作微信汇总已逾期、今天和未来7天销售待办。", self._build_followup_page()),
            "strategies": ("分组", "目标、语气、禁忌和手动保护。", self._build_strategies_page()),
            "memories": ("记忆", "长期记忆、临时事项和待确认内容。", self._build_memories_page()),
            "settings": ("设置", "AI Provider、隐私、采集和验证。", self._build_settings_page()),
        }
        for key, (title, subtitle, page) in pages.items():
            self._page_titles[key] = title
            page.setProperty("subtitle", subtitle)
            self._pages.addWidget(page)

    def _select_page(self, key: str) -> None:
        started = time.perf_counter()
        if key == "diagnostics":
            key = "overview"
        previous_key = self._current_page_key
        if key == "customer_profile" and previous_key != "customer_profile":
            if not self._profile_return_state or self._profile_return_state.get("page") != previous_key:
                self._profile_return_state = {"page": previous_key}
            source_title = self._page_titles.get(previous_key, "上一页")
            if self._customer_profile_back_button is not None:
                self._customer_profile_back_button.setEnabled(True)
                self._customer_profile_back_button.setText(f"← 返回{source_title}")
            if self._customer_profile_breadcrumb is not None:
                self._customer_profile_breadcrumb.setText(f"{source_title}  ›  客户档案")
        keys = [item.key for item in NAV_ITEMS]
        index = keys.index(key)
        self._pages.setCurrentIndex(index)
        self._current_page_key = key
        self.title_label.setText(self._page_titles[key])
        self.subtitle_label.setText(self._pages.currentWidget().property("subtitle"))
        for nav_key, button in self._nav_buttons.items():
            button.setProperty("active", nav_key == key)
            button.style().unpolish(button)
            button.style().polish(button)
        if key == "customer_profile":
            self._refresh_customer_profile_page(force=True)
        elif key == "followups":
            self._refresh_followup_workbench(force=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._record_ui_performance("page_switch", key, elapsed_ms, main_thread_ms=elapsed_ms)

    def _toggle_floating(self) -> None:
        if not self._floating:
            return
        if self._floating.isVisible():
            if hasattr(self._floating, "hide_by_user"):
                self._floating.hide_by_user()
            else:
                self._floating.hide()
            self.float_button.setText("显示悬浮窗")
            self.floating_visibility_changed.emit(False)
            return
        state = self._services.runtime.state
        if state.window.state != WindowState.VISIBLE or state.window.rect is None:
            reason = state.window.diagnostic or state.capture_decision.reason or "当前没有可用目标窗口"
            if hasattr(self._floating, "hide_for_window_state"):
                self._floating.hide_for_window_state(reason)
            self.float_button.setText("显示悬浮窗")
            self.statusBar().showMessage(reason, 3000)
            return
        if hasattr(self._floating, "show_by_user"):
            self._floating.show_by_user()
        else:
            self._floating.show()
        self.float_button.setText("隐藏悬浮窗")
        self.floating_visibility_changed.emit(True)

    def _toggle_capture_pause(self) -> None:
        state = self._services.runtime.set_paused(not self._services.runtime.state.paused)
        self.append_log(f"capture_pause_changed: paused={state.paused}")

    def _on_floating_user_hidden_changed(self, hidden: bool) -> None:
        self.float_button.setText("显示悬浮窗" if hidden else "隐藏悬浮窗")

    def _on_suggestion_copied(self, text: str) -> None:
        self.statusBar().showMessage(f"已复制建议：{text[:24]}", 3000)

    def _on_pipeline_title_ready(self, _result) -> None:
        if (getattr(self, "_manual_generation_enabled", False)
                and self._manual_account_boundary_stop(_result)):
            return
        self._services.activation_prewarm.note_user_activity()
        instant_rendered = self._try_instant_activation(_result)
        ingestion = self._services.ingestion.last_result
        if ingestion is None:
            self.append_log("pipeline_title_ready: no title ingestion result", "warning")
            return
        self.append_log(
            f"pipeline_title_ingested: accepted={ingestion.accepted}, reason={ingestion.reason}, "
            f"contact={ingestion.contact.display_name if ingestion.contact else '-'}, "
            f"title_detect_ms={_result.performance_trace.get('title_detect_ms', '-')}, "
            f"title_ocr_ms={_result.performance_trace.get('title_ocr_elapsed_ms', _result.elapsed_ms or 0)}, "
            f"customer_resolve_ms={_result.performance_trace.get('customer_resolve_ms', '-')}, "
            f"identity_confirm_ms={_result.performance_trace.get('identity_confirm_ms', '-')}"
        )
        if ingestion.identity_stable and ingestion.contact is not None:
            QTimer.singleShot(
                0,
                lambda result=_result, evidence=ingestion: self._learn_activation_title_fingerprint(
                    result,
                    evidence,
                ),
            )
        if ingestion.contact:
            self._reload_contact_list(select_contact_id=ingestion.contact.id)
            self._render_contact_detail(ingestion.contact)
            if (
                not self._load_latest_displayable_ai_result(ingestion)
                and not self._has_matching_activation(ingestion.contact.id)
            ):
                self._schedule_proactive_contact(ingestion)
            self._refresh_customer_profile_page(force=True)
        elif not instant_rendered:
            self._services.current_identity.observe_unavailable(
                _result,
                producer="MainWindowTitleConsumer",
                reason=ingestion.reason,
            )
        self._record_capture_state_commit("title", _result, ingestion)
        self._refresh_overview_data()
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(self._services.runtime.state))

    def _try_instant_activation(self, result) -> bool:
        if getattr(self, "_manual_generation_enabled", False):
            return False
        started = time.perf_counter()
        state = self._services.runtime.state
        lifecycle_gate = getattr(self._services.runtime, "evidence_is_current", None)
        if callable(lifecycle_gate) and not lifecycle_gate(result):
            return False
        page_was_chat_dm = bool(result.performance_trace.get("page_was_chat_dm"))
        if (
            state.page.page_type not in {PageType.CHAT_DM, PageType.UNKNOWN}
            or (state.page.page_type == PageType.UNKNOWN and not page_was_chat_dm)
        ) or (
            state.window.state != WindowState.VISIBLE
            or state.window.hwnd is None
            or result.hwnd != state.window.hwnd
            or result.process_id != state.window.process_id
            or result.window_generation != state.window_generation
            or result.capture_generation != state.capture_generation
            or result.account_identity_epoch != state.account_identity_epoch
            or result.account_identity_runtime_key != state.account_identity_runtime_key
            or not result.account_scope_id
            or not result.target_instance_id
        ):
            return False
        candidates = extract_title_candidates(
            result.ocr_result,
            result,
            min_confidence=0.50,
        )
        raw_title = candidates[0] if candidates else ""
        title = normalize_contact_title(raw_title, ConversationType.DM) if raw_title else None
        if not title:
            return False
        confidence = _title_confidence(result.ocr_result, result, title)
        stable_frames = int(result.performance_trace.get("stable_frame_count") or 0)
        ingestion_evidence = self._services.ingestion.last_result
        normal_path_safe = bool(
            ingestion_evidence is not None
            and getattr(self._services.ingestion, "last_evidence", None) is result
            and ingestion_evidence.identity_stable
            and ingestion_evidence.contact is not None
            and ingestion_evidence.contact.conversation_type == ConversationType.DM
            and ingestion_evidence.contact.account_scope_id == result.account_scope_id
            and ingestion_evidence.target_instance_id == result.target_instance_id
        )
        fast_probe_two_frame_safe = bool(
            stable_frames >= 2
            and result.title_source in {"fast_probe_roi_ocr", "fast_probe_fingerprint"}
            and result.performance_trace.get("title_confirmation_path") == "independent_title_two_frame"
            and result.performance_trace.get("page_evidence") in {"runtime_chat_dm", "exact_dm_fingerprint"}
            and result.title_exact_hash
        )
        # A previous runtime CHAT_DM classification alone does not prove that
        # a newly opened title is a DM. The accelerated route must also match a
        # confirmed exact DM fingerprint in this account and target instance.
        fast_hint = None
        if fast_probe_two_frame_safe:
            fast_hint = self._services.compact_activation_index.lookup_fingerprint(
                result.account_scope_id, result.target_instance_id, result.title_exact_hash, ""
            )
            hinted_contact = (
                self._services.compact_activation_index.contact(fast_hint.contact_id)
                if fast_hint is not None else None
            )
            fast_probe_two_frame_safe = bool(
                fast_hint is not None
                and fast_hint.confidence >= 0.95
                and hinted_contact is not None
                and hinted_contact.conversation_type == ConversationType.DM
                and hinted_contact.account_scope_id == result.account_scope_id
                and normalize_activation_title(fast_hint.normalized_title) == normalize_activation_title(title)
            )
        if not normal_path_safe and not fast_probe_two_frame_safe:
            return False
        observed_sequence = int(result.performance_trace.get("observed_sequence") or 0)
        if observed_sequence and observed_sequence < self._instant_activation_observed_sequence:
            return False
        resolution = self._services.fast_customer_resolver.resolve(
            account_scope_id=result.account_scope_id,
            platform=result.target_app,
            target_instance_id=result.target_instance_id,
            normalized_title=title,
            # StableIdentityLock on the normal path, or FastProbe's two equal
            # title frames, has already supplied the required positive frame
            # evidence. Account scope, target instance, current hwnd and
            # capture generation were checked above; the resolver still
            # requires a unique high-confidence local binding.
            title_confidence=max(confidence, 0.95)
            if normal_path_safe or fast_probe_two_frame_safe
            else confidence,
            retained_contact_id=self._services.current_identity.snapshot.contact_id,
        )
        if not resolution.resolved:
            self._services.current_identity.observe_unavailable(
                result,
                producer="FastCustomerResolver",
                reason=f"fast_identity_unavailable:{resolution.reason}",
            )
            return False
        if normal_path_safe and ingestion_evidence.contact.id != resolution.contact_id:
            return False
        if fast_probe_two_frame_safe and fast_hint.contact_id != resolution.contact_id:
            return False
        resolver_completed_at = utc_now_iso()
        resolved_contact = self._services.compact_activation_index.contact(resolution.contact_id)
        if resolved_contact is None:
            self._services.current_identity.observe_unavailable(
                result,
                producer="FastCustomerResolver",
                reason="fast_contact_missing",
            )
            return False
        retained = self._services.current_identity.snapshot
        if retained.retained and retained.contact_id != resolution.contact_id:
            resolved_by_normal_path = bool(
                normal_path_safe
                and ingestion_evidence.contact is not None
                and ingestion_evidence.contact.id == resolution.contact_id
            )
            if stable_frames < 2 and not resolved_by_normal_path:
                self._services.current_identity.observe_possible_switch(
                    result,
                    producer="FastCustomerResolver",
                    reason="unique_contact_waiting_two_frame_lock",
                )
                return False
        light_snapshot_found_at = utc_now_iso()
        token = (
            result.account_scope_id,
            result.target_instance_id,
            result.account_identity_runtime_key,
            result.hwnd,
            result.process_id,
            result.window_generation,
            result.account_identity_epoch,
            resolution.contact_id,
            result.title_exact_hash,
        )
        if (
            self._instant_activation_binding == token
            and self._instant_activation_contact_id == resolution.contact_id
            and self._suggestion_mode == "proactive"
            and self._proactive_outcome is not None
            and self._suggestion_binding_is_current(self._suggestion_action_binding)
        ):
            return True
        if (
            self._services.current_identity.snapshot.contact_id == resolution.contact_id
            and self._services.current_identity.snapshot.account_scope_id == resolution.account_scope_id
            and self._manual_proactive_edit_is_current()
        ):
            return True
        self._instant_activation_pending_contact_id = resolution.contact_id
        try:
            transition = self._services.current_identity.promote(
                resolved_contact,
                result,
                normalized_title=title,
                state=CurrentIdentityState.FAST_SAFE,
                producer="FastCustomerResolver",
                reason=resolution.reason,
            )
        finally:
            self._instant_activation_pending_contact_id = None
        if transition.action == "reject":
            return False
        identity_safe_at = utc_now_iso()
        instant = self._services.activation_snapshots.lookup(resolution)
        snapshot_found_at = utc_now_iso()
        if not instant.available or instant.outcome is None:
            if instant.reason == "known_reply_pending":
                self._pending_activation_refresh = None
                self._render_reply_empty_state("本地已知存在客户待回复消息，等待 Reply 链路")
                QTimer.singleShot(0, lambda: self._complete_instant_activation_transition(resolved_contact))
                return False
            instant = self._services.activation_snapshots.safe_fallback(resolution)
            self._pending_activation_refresh = (
                resolution,
                result,
                self._services.current_identity.snapshot.conversation_token,
                self._services.activation_snapshots.revision(resolution.account_scope_id, resolution.contact_id),
            )
            self._render_reply_empty_state("正在结合客户资料准备建议")
            self._services.activation_prewarm.request_local_build(
                account_scope_id=resolution.account_scope_id,
                contact_id=resolution.contact_id,
                conversation_id=resolution.conversation_id,
            )
            if not instant.available or instant.outcome is None:
                QTimer.singleShot(0, lambda: self._complete_instant_activation_transition(resolved_contact))
                return False
        text_determined_at = utc_now_iso()
        if not self._services.activation_snapshots.revision_is_current(
            resolution.account_scope_id,
            resolution.contact_id,
            instant.activation_revision,
        ):
            self.append_log(
                f"instant_activation_stale_before_render: contact={resolution.contact_id}",
                "warning",
            )
            QTimer.singleShot(0, lambda: self._complete_instant_activation_transition(resolved_contact))
            return False
        render_started = time.perf_counter()
        self._instant_activation_job_id = result.job_id
        self._instant_activation_observed_sequence = max(
            self._instant_activation_observed_sequence,
            observed_sequence,
        )
        self._instant_activation_contact_id = resolution.contact_id
        self._instant_activation_binding = token
        self._pending_activation_refresh = None
        self._render_proactive_outcome(instant.outcome, feedback_preloaded=True)
        self._suggestion_context_hash = instant.context_hash
        if instant.text_source == "edited_and_accepted":
            self._manual_proactive_binding = (
                self._services.current_identity.snapshot.conversation_token, instant.activation_revision
            )
        rendered_at = utc_now_iso()
        render_ms = round((time.perf_counter() - render_started) * 1000.0, 3)
        total_ms = round((time.perf_counter() - started) * 1000.0, 3)
        t0 = str(
            result.performance_trace.get("contact_switch_first_seen_at")
            or result.performance_trace.get("contact_switch_detected_at")
            or result.performance_trace.get("title_capture_started_at")
            or ""
        )
        ttfas_ms = _elapsed_from_iso_ms(t0)
        switch_bound_ms = int(result.performance_trace.get("switch_detection_bound_ms") or 0)
        t1 = str(result.performance_trace.get("contact_switch_confirmed_at") or t0)
        t2 = str(result.performance_trace.get("title_resolved_at") or result.created_at or t1)
        timeline = (
            f"t0_t1_ms={_iso_delta_ms(t0, t1):.3f} "
            f"t1_t2_ms={_iso_delta_ms(t1, t2):.3f} "
            f"t2_t3_ms={_iso_delta_ms(t2, resolver_completed_at):.3f} "
            f"t3_t4_ms={_iso_delta_ms(resolver_completed_at, identity_safe_at):.3f} "
            f"t4_t6_ms={_iso_delta_ms(identity_safe_at, snapshot_found_at):.3f} "
            f"t6_t8_ms={_iso_delta_ms(snapshot_found_at, rendered_at):.3f}"
        )
        QTimer.singleShot(
            0,
            lambda: self.append_log(
                "instant_activation_rendered: "
                f"job={result.job_id} source={instant.source}/{instant.text_source} "
                f"title_source={result.title_source} title_ms={result.elapsed_ms or 0} "
                f"resolve_ms={resolution.elapsed_ms:.3f} lookup_ms={instant.lookup_ms:.3f} "
                f"render_ms={render_ms:.3f} handler_ms={total_ms:.3f} "
                f"switch_bound_ms={switch_bound_ms} ttfas_ms={ttfas_ms:.3f} "
                f"{timeline} t0={t0} t1={t1} t2={t2} "
                f"t3={resolver_completed_at} t4={identity_safe_at} "
                f"t5={light_snapshot_found_at} t6={snapshot_found_at} "
                f"t7={text_determined_at} t8={rendered_at} "
                f"watcher={json.dumps(self._services.fast_activation_probe.metrics_snapshot(), ensure_ascii=False, separators=(',', ':'))}"
            ),
        )
        QTimer.singleShot(0, lambda: self._complete_instant_activation_transition(resolved_contact))
        return True

    def _on_activation_snapshot_ready(self, account_scope_id: str, contact_id: str, revision: int) -> None:
        if getattr(self, "_manual_generation_enabled", False):
            return
        """Upgrade only the still-current loading surface, never a later chat."""

        pending = self._pending_activation_refresh
        if pending is None:
            return
        resolution, evidence, identity_token, expected_revision = pending
        if (resolution.account_scope_id, resolution.contact_id) != (account_scope_id, contact_id):
            return
        identity = self._services.current_identity.snapshot
        lifecycle_gate = getattr(self._services.runtime, "evidence_is_current", None)
        if (
            not identity.active
            or identity.conversation_token != identity_token
            or revision != expected_revision
            or not self._services.activation_snapshots.revision_is_current(account_scope_id, contact_id, revision)
            or (callable(lifecycle_gate) and not lifecycle_gate(evidence))
        ):
            self._pending_activation_refresh = None
            return
        latest_event = self._services.analysis_events.latest_for_conversation_binding(
            account_scope_id, contact_id, resolution.conversation_id
        )
        if latest_event is not None and self._analysis_event_has_reply_priority(
            latest_event, resolution.conversation_id
        ):
            self._pending_activation_refresh = None
            self._load_latest_displayable_ai_result()
            return
        if self._manual_proactive_edit_is_current() or self._has_active_suggestion():
            self._pending_activation_refresh = None
            return
        instant = self._services.activation_snapshots.lookup(resolution)
        if (
            not instant.available
            or instant.outcome is None
            or instant.activation_revision != revision
            or not self._services.activation_snapshots.revision_is_current(account_scope_id, contact_id, revision)
            or not self._services.current_identity.snapshot.active
            or self._services.current_identity.snapshot.conversation_token != identity_token
            or (callable(lifecycle_gate) and not lifecycle_gate(evidence))
        ):
            return
        self._pending_activation_refresh = None
        self._render_proactive_outcome(instant.outcome, feedback_preloaded=True)
        self._suggestion_context_hash = instant.context_hash
        if instant.text_source == "edited_and_accepted":
            self._manual_proactive_binding = (identity_token, instant.activation_revision)
        self._refresh_overview_data()

    def _manual_proactive_edit_is_current(self) -> bool:
        binding = getattr(self, "_manual_proactive_binding", None)
        if binding is None:
            return False
        identity = self._services.current_identity.snapshot
        token, revision = binding
        return bool(
            identity.active
            and identity.conversation_token == token
            and self._services.activation_snapshots.revision_is_current(
                identity.account_scope_id, identity.contact_id, revision
            )
            and self._proactive_outcome is not None
            and self._suggestion_mode == "proactive"
        )

    def _complete_instant_activation_transition(self, contact) -> None:
        """Run SQLite-backed detail refreshes only after the snapshot is visible."""

        current = self._services.current_identity.snapshot
        if not current.active or current.contact_id != contact.id:
            return
        self._reload_contact_list(select_contact_id=contact.id)
        self._render_contact_detail(contact)
        self._refresh_customer_profile_page(force=True)

    def _on_fast_probe_title_ready(self, result) -> None:
        self._services.activation_prewarm.note_user_activity()
        if self._try_instant_activation(result):
            self._refresh_overview_data()

    def _on_fast_probe_context_changed(self, event) -> None:
        self._record_capture_state_commit("fast_context_changed", event, None, event.reason)
        self._refresh_overview_data()

    def _learn_activation_title_fingerprint(self, result, ingestion) -> None:
        """Learn a stable title immediately, after first paint, without content OCR."""

        trace = getattr(result, "performance_trace", None)
        title_source = getattr(result, "title_source", None)
        if not title_source and isinstance(trace, dict):
            title_source = trace.get("title_source")
        # Cache/fingerprint reuse is not a new observation. In particular it
        # must never teach a guessed title back into the persistent exact map.
        if (title_source not in {"roi_ocr", "fast_probe_roi_ocr"}
                or (isinstance(trace, dict) and trace.get("title_cache_hit") is True)):
            return
        if (
            not ingestion.identity_stable
            or ingestion.contact is None
            or ingestion.contact.conversation_type != ConversationType.DM
            or not getattr(result, "title_exact_hash", "")
        ):
            return
        candidates = extract_title_candidates(result.ocr_result, result, min_confidence=0.50)
        raw_title = candidates[0] if candidates else ""
        normalized_title = normalize_contact_title(raw_title, ConversationType.DM) if raw_title else None
        if not normalized_title:
            return
        self._services.compact_activation_index.confirm_fingerprint(
            account_scope_id=ingestion.contact.account_scope_id,
            target_instance_id=result.target_instance_id,
            exact_title_hash=result.title_exact_hash,
            visual_title_hash=result.title_visual_hash,
            normalized_title=normalized_title,
            contact_id=ingestion.contact.id,
            confidence=max(
                0.95,
                _title_confidence(result.ocr_result, result, normalized_title),
            ),
        )

    def _on_pipeline_result_ready(self, _result) -> None:
        if getattr(self, "_manual_generation_enabled", False):
            if self._manual_generation_phase in {"recognizing", "verifying"} and _result.job_id in self._manual_generation_capture_ids:
                self._last_ocr_messages = list(_result.messages)
                self._last_ocr_result_meta = {
                    "job_id": _result.job_id, "created_at": _result.created_at,
                    "page": _result.page.page_type.value, "page_confidence": _result.page.confidence,
                    "warning": _result.ocr_result.warning or "",
                }
                self._manual_generation_on_pipeline_result(_result)
                self._refresh_overview_data()
            return
        # The result-ready slot is the available process-local recognition
        # boundary. Expensive context/profile/UI work below consumes this time.
        from time import monotonic as current_work_clock
        recognition_completed = current_work_clock()
        self._last_ocr_messages = list(_result.messages)
        self._last_ocr_result_meta = {
            "job_id": _result.job_id,
            "created_at": _result.created_at,
            "page": _result.page.page_type.value,
            "page_confidence": _result.page.confidence,
            "warning": _result.ocr_result.warning or "",
        }
        ingestion = self._services.ingestion.last_result
        if ingestion is None:
            self.append_log("pipeline_result_ready: no ingestion result", "warning")
            return
        if ingestion.accepted and ingestion.identity_stable and ingestion.contact and ingestion.observation_id:
            self._current_work_capture_anchor = (
                ingestion.account_scope_id, ingestion.target_instance_id, ingestion.contact.id,
                ingestion.conversation_id, ingestion.session_id, ingestion.observation_id, recognition_completed,
            )
        self.append_log(
            f"pipeline_ingested: accepted={ingestion.accepted}, reason={ingestion.reason}, "
            f"inserted={ingestion.inserted_messages}, duplicate={ingestion.duplicate_messages}"
        )
        if (
            ingestion.accepted
            and ingestion.identity_stable
            and ingestion.contact is not None
            and ingestion.new_incoming_messages
        ):
            # This tuple contains only timeline-verified, non-baseline,
            # non-duplicate OTHER messages.  Suppress the already-rendered
            # Activation before any profile refresh or Provider work begins.
            message_token = ",".join(item.id for item in ingestion.new_incoming_messages)
            self._services.activation_snapshots.invalidate(
                ingestion.contact.account_scope_id,
                ingestion.contact.id,
                "true_new_customer_message",
                mutation_token=message_token,
            )
            self._suppress_activation_for_reply_priority(ingestion.contact.id)
        if (
            ingestion.contact is not None
            and ingestion.contact.id == self._instant_activation_contact_id
        ):
            self.append_log(
                "instant_activation_normal_path_completed: "
                f"activation_job={self._instant_activation_job_id} full_job={_result.job_id} "
                f"t9={_result.performance_trace.get('content_ocr_completed_at', _result.created_at)} "
                f"t10={utc_now_iso()} reply_state={ingestion.reason}"
            )
        if ingestion.contact:
            if ingestion.accepted:
                self._last_ocr_contact_id = ingestion.contact.id
            if _result.page.page_type == PageType.CHAT_DM:
                self._learn_activation_title_fingerprint(_result, ingestion)
        else:
            self._services.current_identity.observe_unavailable(
                _result,
                producer="MainWindowFullOcrConsumer",
                reason=ingestion.reason,
            )
        if (
            ingestion.accepted
            and ingestion.identity_stable
            and ingestion.contact
            and ingestion.inserted_messages
        ):
            # Any persisted incoming or outgoing message changes the proactive
            # decision context.  Expire an in-flight job before Reply/proactive
            # arbitration so a late result can never repaint stale advice.
            self._services.proactive_tasks.invalidate("message_context_changed")
            self._proactive_last_request_key = ""
        # Put a durable Reply event on its dedicated worker before rebuilding
        # contact/profile/overview widgets.  Those local UI refreshes are useful
        # but must never delay the user-facing Provider request.
        if ingestion.accepted and ingestion.identity_stable and ingestion.contact and ingestion.conversation_id:
            self._withdraw_outdated_reply()
            pending_events = self._services.analysis_events.list_pending_for_binding(
                ingestion.contact.account_scope_id,
                ingestion.contact.id,
                ingestion.conversation_id,
            )
            if pending_events:
                self._services.proactive_tasks.invalidate("reply_event_created")
                self._proactive_last_request_key = ""
                if self._suggestion_mode == "proactive":
                    self._render_reply_empty_state("已收到新的客户消息，正在生成 Reply")
            for event in pending_events:
                self._queue_auto_analysis_event(event.id)
        if self._pending_auto_event_ids:
            self._auto_generate_reply_after_ocr()
        else:
            # A baseline is not a new incoming event, but the already visible
            # customer question still deserves a Reply when the window opens.
            self._auto_generate_visible_reply()
        render_key = (
            ingestion.account_scope_id, ingestion.target_instance_id,
            ingestion.contact.id if ingestion.contact else "", ingestion.conversation_id,
            ingestion.session_id, self._current_visible_ui_revision,
        )
        refresh_customer_widgets = render_key != self._last_chat_render_key
        if refresh_customer_widgets:
            self._last_chat_render_key = render_key
            self._reload_contact_list(select_contact_id=ingestion.contact.id if ingestion.contact else None)
        if ingestion.contact and refresh_customer_widgets:
            self._render_contact_detail(ingestion.contact)
            if (
                not self._load_latest_displayable_ai_result(ingestion)
                and not self._has_matching_activation(ingestion.contact.id)
            ):
                self._schedule_proactive_contact(ingestion)
            if self._customer_profile_page is not None and self._pages.currentWidget() is self._customer_profile_page:
                self._refresh_customer_profile_page(force=True)
        self._record_capture_state_commit("result", _result, ingestion)
        self._refresh_overview_data()
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(self._services.runtime.state))

    def _on_pipeline_discarded(self, reason: str) -> None:
        if (getattr(self, '_manual_selection_plan', None) is not None
                and self._manual_generation_phase in {'recognizing', 'verifying'}
                and 'manual_selection_' in reason):
            from whochat.services.manual_selection import selection_error_message
            self._manual_generation_note_blocker(reason)
            self._finish_manual_generation(selection_error_message(reason))
            return
        if reason.startswith("duplicate_snapshot:"):
            now = time.monotonic()
            if now - self._last_duplicate_ui_refresh_monotonic < 10.0:
                return
            self._last_duplicate_ui_refresh_monotonic = now
        else:
            self.append_log(f"pipeline_discarded: {reason}")
        self._refresh_overview_data()

    def _on_pipeline_context_changed(self, event) -> None:
        if getattr(self, "_frozen_screenshot_run", lambda: None)() is not None:
            self._refresh_overview_data()
            return
        state = self._services.runtime.state
        accepted_marker = getattr(self._services.runtime, "capture_context_event_was_applied", None)
        accepted = (
            accepted_marker(event)
            if callable(accepted_marker)
            else self._services.runtime.capture_context_event_is_current(event)
        )
        # Accepted account boundaries deliberately rotate capture generation.
        # They are UI notifications of an already committed boundary, not OCR
        # results that should be tested against its former capture token.
        if not accepted:
            self._record_capture_state_commit("context_change_rejected", event, None, "generation_mismatch")
            return
        self._record_capture_state_commit("context_changed", event, None, event.reason)
        if event.reason == "current_screen_body_changed":
            self._services.reply_tasks.invalidate(event.reason)
            self._services.proactive_tasks.invalidate(event.reason)
            # Keep per-job audit bindings until their terminal signal consumes
            # them. Revocation never refunds an already dispatched request.
            self._latest_reply_job_id = None
            self._latest_proactive_job_id = None
            self._visible_reply_attempt_key = ""
            self._current_visible_ui_revision = ""
            self._proactive_last_request_key = ""
            self._proactive_retry_pending = False
            self._manual_proactive_binding = None
            self._suggestion_action_binding = None
            self._instant_activation_binding = None
            self._displayed_analysis_event_id = None
            if (getattr(self, "_manual_generation_terminal", False)
                    and getattr(self, "_manual_generation_phase", "") == "failed"
                    and getattr(self, "_manual_generation_last_blocker", "")
                    in {"history_navigation_present", "input_boundary_unverified", "untranscribed_voice_present", "message_text_low_confidence"}):
                # Revocation above still applies. A late lifecycle event must
                # not replace this request's actionable terminal diagnosis.
                self._refresh_overview_data()
                return
            self._render_reply_empty_state("聊天画面有变化，正在重新核验当前内容")
            return
        self._refresh_overview_data()
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(state))

    def _build_page_shell(self, left: QWidget, right: QWidget | None = None) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(left, 1)
        if right is not None:
            right.setFixedWidth(344)
            layout.addWidget(right)
        return page

    def _panel(self, title: str, subtitle: str | None = None) -> QFrame:
        panel = QFrame()
        panel.setObjectName("PagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)
        header = QLabel(title)
        header.setObjectName("SectionTitle")
        if subtitle:
            header.setToolTip(subtitle)
        layout.addWidget(header)
        return panel

    def _build_overview_page(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        metrics = QHBoxLayout()
        page_card, self._page_status_value, self._page_status_subtitle = self._metric_card_with_labels("当前页面", "unknown", "等待目标窗口")
        metrics.addWidget(page_card)
        ai_value = "未配置" if not self._config.ai.api_key and self._config.ai.provider != "Local Model" else self._config.ai.model
        ai_card, self._ai_status_value = self._metric_card_with_value("AI 状态", ai_value, self._config.ai.provider)
        metrics.addWidget(ai_card)
        capture_card, self._capture_status_value, self._capture_status_subtitle = self._metric_card_with_labels("截图门控", "暂停", "等待运行态")
        metrics.addWidget(capture_card)
        layout.addLayout(metrics)

        status_chain = self._panel("运行链路", "逐项解释当前窗口、页面、聊天对象、采集、隐私和 AI 是否满足生成条件。")
        action_banner = QFrame()
        action_banner.setObjectName("ActionBanner")
        action_layout = QHBoxLayout(action_banner)
        action_layout.setContentsMargins(10, 8, 10, 8)
        action_layout.setSpacing(10)
        next_action = QLabel("下一步：等待运行态")
        next_action.setObjectName("ActionPrimary")
        next_action.setWordWrap(False)
        next_action_meta = QLabel("未连接目标窗口")
        next_action_meta.setObjectName("TinyMuted")
        next_action_meta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        capture_now = QPushButton("生成建议" if self._manual_generation_enabled else "立即采集")
        capture_now.setObjectName("PrimaryButton")
        capture_now.setToolTip("对当前目标窗口运行一次截图和 OCR，用 OCR 结果确认是否为聊天页。")
        capture_now.clicked.connect(self._run_capture_pipeline)
        capture_now.setVisible(self._manual_generation_enabled)
        if self._manual_generation_enabled:
            capture_now.setToolTip("点击后请保持微信客户窗口在前台；推荐使用微信旁悬浮条的「生成建议」。")
        self._overview_next_action = next_action
        self._overview_next_action_meta = next_action_meta
        self._overview_capture_button = capture_now
        remember = QPushButton("记住当前聊天")
        remember.setToolTip("仅核验并保存当前可见的客户文字到本地历史；不调用 AI、不扣额度。此前未读的历史需要先翻到对应页面。")
        remember.clicked.connect(self._on_remember_requested)
        remember.setVisible(self._manual_generation_enabled)
        self._overview_remember_button = remember
        action_layout.addWidget(next_action, 1)
        action_layout.addWidget(remember)
        action_layout.addWidget(capture_now)
        action_layout.addWidget(next_action_meta)
        status_chain.layout().addWidget(action_banner)
        screenshot = QPushButton('截图生成（手动框选聊天）')
        screenshot.setObjectName('PrimaryButton')
        screenshot.setToolTip('框住顶部客户名称与双方完整聊天，松开鼠标后生成；Esc 取消，不自动发送。')
        screenshot.setVisible(self._manual_generation_enabled)
        screenshot.clicked.connect(self._on_screenshot_generate_requested)
        self._overview_screenshot_button = screenshot
        status_chain.layout().addWidget(screenshot)
        link_history = QPushButton('关联已有历史')
        link_history.setVisible(self._manual_generation_enabled)
        link_history.setToolTip('核对当前账号下的历史客户归属；只做本地关联，不调用 AI。')
        link_history.clicked.connect(self._on_link_imported_history)
        status_chain.layout().addWidget(link_history)
        activate = QPushButton('激活客户（旧话题已结束）')
        activate.setVisible(self._manual_generation_enabled)
        activate.setToolTip('点击表示你确认旧话题已结束。本次只核验客户与当前画面，结合已关联历史生成激活话术；不解读旧语音或物流附件，不自动发送。当前有未处理问题请用「生成建议」。')
        activate.clicked.connect(lambda: self._on_generate_requested(activation_only=True))
        self._overview_activation_button = activate
        status_chain.layout().addWidget(activate)
        if getattr(self._services, "writing_feedback", None) is not None:
            from whochat.ui.writing_feedback import history_dialog
            feedback_history = QPushButton("建议与修改记录")
            feedback_history.clicked.connect(lambda: history_dialog(self))
            status_chain.layout().addWidget(feedback_history)
        call_budget = QLabel()
        call_budget.setObjectName("TinyMuted")
        call_budget.setWordWrap(True)
        self._overview_call_budget = call_budget
        status_chain.layout().addWidget(call_budget)
        status_summary = QLabel("等待运行态")
        status_summary.setObjectName("Muted")
        status_summary.setWordWrap(True)
        status_summary.setMaximumHeight(22)
        status_summary.setVisible(False)
        self._overview_status_summary = status_summary
        status_chain.layout().addWidget(status_summary)
        status_table = QTableWidget(0, 3)
        status_table.setHorizontalHeaderLabels(["环节", "状态", "原因"])
        status_table.setMinimumHeight(178)
        status_table.verticalHeader().setDefaultSectionSize(22)
        _compact_table(status_table)
        status_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        status_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        status_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        status_table.setColumnWidth(0, 86)
        status_table.setColumnWidth(1, 82)
        self._overview_status_table = status_table
        status_chain.layout().addWidget(status_table)
        layout.addWidget(status_chain)

        current_chat = self._panel("最近一次 OCR 结果", "只显示最近一次完整采集解析出的消息，不混入历史记录。")
        self._overview_chat_meta = QLabel("尚未完成 OCR 采集")
        self._overview_chat_meta.setObjectName("TinyMuted")
        current_chat.layout().addWidget(self._overview_chat_meta)
        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["说话人", "发送者", "内容", "时间", "置信度", "状态"])
        _compact_table(table)
        table.setWordWrap(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._overview_chat_table = table
        current_chat.layout().addWidget(table)
        layout.addWidget(current_chat, 1)

        right = self._build_context_rail()
        self._render_reply_suggestions(self._build_reply_result())
        self._refresh_overview_data()
        return self._build_page_shell(left, right)

    def _metric_card(self, title: str, value: str, subtitle: str) -> QWidget:
        card, _value_label = self._metric_card_with_value(title, value, subtitle)
        return card

    def _metric_card_with_value(self, title: str, value: str, subtitle: str) -> tuple[QWidget, QLabel]:
        card, value_label, _subtitle_label = self._metric_card_with_labels(title, value, subtitle)
        return card, value_label

    def _metric_card_with_labels(self, title: str, value: str, subtitle: str) -> tuple[QWidget, QLabel, QLabel]:
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        card.setFixedHeight(72)
        card.setToolTip(subtitle)
        label = QLabel(title)
        label.setObjectName("TinyMuted")
        val = QLabel(value)
        val.setObjectName("MetricValue")
        sub = QLabel(subtitle)
        sub.setObjectName("Muted")
        sub.setWordWrap(True)
        sub.setVisible(False)
        layout.addWidget(label)
        layout.addWidget(val)
        layout.addWidget(sub)
        return card, val, sub

    def _current_contact(self) -> Contact | None:
        selected = self._selected_contact()
        if selected is not None:
            return selected
        recent = self._services.contacts.list_recent(1)
        return recent[0] if recent else None

    def _active_capture_contact(self) -> Contact | None:
        return self._services.current_identity.current_contact()

    def _set_active_capture_contact(
        self,
        contact: Contact,
        hwnd: int | None,
        *,
        evidence=None,
        normalized_title: str = "",
        identity_state: CurrentIdentityState = CurrentIdentityState.CONFIRMED,
        producer: str = "MainWindowEvidence",
        reason: str = "identity_stable",
    ) -> None:
        if hwnd is None or hwnd != self._services.runtime.state.window.hwnd:
            return
        source = evidence or _identity_evidence_from_runtime(
            self._services.runtime.state,
            contact.account_scope_id,
            self._services.ingestion.last_result,
        )
        self._services.current_identity.promote(
            contact,
            source,
            normalized_title=normalized_title or contact.display_name,
            state=identity_state,
            producer=producer,
            reason=reason,
        )

    def _clear_active_capture_context(self, reason: str) -> None:
        strong = reason in {"window_changed", "account_scope_bound", "wechat_window_closed"}
        self._services.current_identity.invalidate(
            self._services.runtime.state,
            producer="MainWindowEvidence",
            reason=reason,
            strong=strong,
        )

    def _on_current_identity_transition(self, transition: IdentityTransition) -> None:
        if getattr(self, "_frozen_screenshot_run", lambda: None)() is not None:
            # Live identity continues to update in its service. These widgets
            # retain the explicitly named screenshot customer, never relabel A as B.
            self._refresh_overview_data()
            return
        from time import monotonic as current_work_clock
        recognition_completed = current_work_clock()
        old = transition.old
        new = transition.new
        old_active = old.active
        old_retained = old.retained
        new_active = new.active
        if new.state == CurrentIdentityState.VERIFYING and new.retained:
            self._identity_suggestions_suspended = True
            self._identity_safe_since_monotonic = 0.0
            self._proactive_stability_retry_pending = False
            if self._services.proactive_tasks.is_running:
                self._services.proactive_tasks.invalidate("identity_verifying")
            self._pending_activation_refresh = None
            self._show_identity_verifying_state()
            if self._suggestions_title is not None:
                self._suggestions_title.setText("正在确认新客户")
            self._refresh_overview_data()
            self.append_log(
                f"identity_verifying: retained_contact={new.contact_id} reason={transition.producer}:{transition.reason}"
            )
            return
        if new_active:
            if not old_active or old.conversation_token != new.conversation_token:
                self._current_work_identity_anchor = (new.conversation_token, recognition_completed)
            was_suspended = self._identity_suggestions_suspended
            self._identity_suggestions_suspended = False
            self._identity_safe_since_monotonic = time.monotonic()
            self._proactive_stability_retry_pending = False
            contact_changed = old.conversation_token != new.conversation_token
            preserve_instant = bool(
                self._instant_activation_pending_contact_id == new.contact_id
                or (
                    self._instant_activation_contact_id == new.contact_id
                    and self._suggestion_mode == "proactive"
                    and self._proactive_outcome is not None
                )
            )
            if contact_changed:
                self._pending_activation_refresh = None
                self._manual_proactive_binding = None
                self._services.proactive_tasks.invalidate("active_contact_changed")
                self._proactive_last_request_key = ""
                self._proactive_retry_pending = False
                self._customer_profile_navigation_contact_id = None
                if self._instant_activation_pending_contact_id == new.contact_id:
                    # The identity signal is synchronous and fires immediately
                    # before the memory snapshot is painted. Avoid an empty
                    # state flash and all profile SQLite reads in that window.
                    return
                if not preserve_instant:
                    self._suggestion_result = None
                    self._suggestion_context_hash = None
                    self._proactive_outcome = None
                    self._proactive_feedback_status = {}
                    self._proactive_rejected_feedback = {}
                    self._proactive_regeneration_used = False
                    self._instant_activation_contact_id = None
                    self._instant_activation_binding = None
                    self._render_reply_empty_state("正在加载当前客户的已完成回复建议")
                self._displayed_analysis_event_id = None
                self._last_ocr_contact_id = None
                self._customer_profile_binding = None
                self._refresh_customer_profile_page(force=True)
            elif was_suspended and self._proactive_outcome is not None:
                self._render_proactive_outcome(self._proactive_outcome, feedback_preloaded=True)
                self._refresh_overview_data()
            return
        if not old_retained:
            return
        reason = f"{transition.producer}:{transition.reason}"
        self._identity_suggestions_suspended = False
        self._identity_safe_since_monotonic = 0.0
        self._proactive_stability_retry_pending = False
        if self._services.proactive_tasks.is_running:
            self._services.proactive_tasks.invalidate(reason)
        if self._services.reply_tasks.is_running:
            self._services.reply_tasks.invalidate(reason)
        self._proactive_last_request_key = ""
        self._proactive_retry_pending = False
        self._instant_activation_contact_id = None
        self._instant_activation_binding = None
        self._pending_activation_refresh = None
        self._manual_proactive_binding = None
        self._suggestion_result = None
        self._suggestion_context_hash = None
        self._proactive_outcome = None
        self._proactive_feedback_status = {}
        self._proactive_rejected_feedback = {}
        self._proactive_regeneration_used = False
        self._displayed_analysis_event_id = None
        self._last_ocr_contact_id = None
        self._customer_profile_binding = None
        self._render_reply_empty_state("正在识别当前客户")
        self.append_log(f"active_capture_context_cleared: {reason}")

    def _refresh_overview_data(self) -> None:
        # The overview is a live capture surface, not the contact manager.
        # Never fall back to a previously selected/recent contact while the
        # current WeChat identity is pending or rejected.
        identity_snapshot = self._services.current_identity.snapshot
        contact = self._active_capture_contact()
        if contact is None and identity_snapshot.state == CurrentIdentityState.VERIFYING:
            contact = self._services.current_identity.retained_contact()
        strategy = self._services.strategies.get(contact.strategy_id) if contact else self._services.strategies.get("default")
        if getattr(self, "_overview_call_budget", None) is not None:
            self._manual_generation_refresh_budget_status()
        if self._overview_status_table is not None:
            steps = build_status_chain(
                runtime=self._services.runtime.state,
                contact=contact,
                strategy=strategy,
                config=self._config,
                reply_running=self._services.reply_tasks.is_running,
                provider_health=self._services.reply_generator.provider_health_summary(),
                has_active_suggestion=self._has_active_suggestion(),
                current_work_cloud_authorized=self._current_work_reply_cloud_allowed(),
            )
            if self._overview_status_summary is not None:
                self._overview_status_summary.setText(" -> ".join(f"{step.stage}:{step.state}" for step in steps))
            if self._overview_next_action is not None and self._overview_next_action_meta is not None:
                primary = _primary_status_step(steps)
                self._overview_next_action.setText(f"下一步：{primary.action or primary.reason}")
                self._overview_next_action.setToolTip(_status_reason_with_action(primary))
                self._overview_next_action_meta.setText(f"{primary.stage} · {primary.state}")
                self._overview_next_action_meta.setToolTip(primary.reason)
                if self._overview_capture_button is not None:
                    self._overview_capture_button.setVisible(
                        getattr(self, "_manual_generation_enabled", False)
                        or _should_show_overview_capture(self._services.runtime.state))
                if getattr(self, "_manual_generation_enabled", False):
                    phase = self._manual_generation_phase
                    manual_action = {
                        "recognizing": "正在核验本次客户和聊天内容",
                        "generating": "正在生成本次建议",
                        "verifying": "AI 已返回，等待原客户窗口本地核验；不会重复请求",
                        "success": "本次建议已完成；聊天更新后请点击「生成建议」",
                        "failed": "本次未完成，请查看下方原因；重试需点击「生成建议」",
                    }.get(phase, "打开客户窗口后点击「生成建议」；未点击不会启动")
                    if phase == "success" and getattr(self, "_manual_review_reason", ""):
                        manual_action = "建议已保留；复制时核验，不会自动重新生成"
                    self._overview_next_action.setText(f"下一步：{manual_action}")
                    self._overview_next_action.setToolTip(manual_action)
            self._fill_table(self._overview_status_table, [(step.stage, step.state, _status_reason_with_action(step)) for step in steps])
            for row in range(self._overview_status_table.rowCount()):
                self._overview_status_table.setRowHeight(row, 22)
            self._overview_status_table.setColumnWidth(0, 86)
            self._overview_status_table.setColumnWidth(1, 82)
        if self._overview_chat_table is not None:
            rows = self._overview_chat_rows(contact)
            self._fill_table(self._overview_chat_table, rows)
            if hasattr(self, "_overview_chat_meta"):
                self._overview_chat_meta.setText(self._overview_chat_summary())
        if not self._overview_contact_values and self._overview_contact_notes is None:
            return
        if contact is None:
            values = {
                "name": "未识别",
                "strategy": "默认",
                "status": "未识别",
                "cloud": "未允许",
            }
            notes = "暂无画像"
        else:
            strategy = self._services.strategies.get(contact.strategy_id)
            aliases = self._services.contacts.list_aliases(contact.id)
            memories = self._services.memories.list_for_contact(contact.id)
            values = {
                "name": _contact_display_label(contact),
                "strategy": strategy.name if strategy else contact.strategy_id,
                "status": (
                    "正在验证"
                    if identity_snapshot.state == CurrentIdentityState.VERIFYING
                    else "已识别"
                ),
                "cloud": (
                    "身份验证中"
                    if identity_snapshot.state == CurrentIdentityState.VERIFYING
                    else self._proactive_ai_permission_label(contact)
                ),
            }
            alias_text = "、".join(alias.alias for alias in aliases[:8]) or "暂无"
            memory_lines = "\n".join(f"- [{memory.status.value}/{memory.kind.value}] {memory.content}" for memory in memories[:8])
            notes = f"备注：{contact.remark or '暂无'}\n别名：{alias_text}\n{memory_lines or '暂无记忆'}"
        for key, value in values.items():
            label = self._overview_contact_values.get(key)
            if label is not None:
                label.setText(value)
        if self._overview_contact_notes is not None:
            self._overview_contact_notes.setText(notes)
        self._sync_floating_content(contact=contact, strategy=strategy)

    def _proactive_ai_permission_label(self, contact: Contact) -> str:
        if contact.allow_cloud_ai:
            return "联系人级允许"
        authorization_context = getattr(self, "_reply_authorization_context", None)
        if (
            authorization_context is not None
            and authorization_context.contact is not None
            and authorization_context.contact.id == contact.id
            and authorization_context.contact.account_scope_id == contact.account_scope_id
            and self._current_work_reply_cloud_allowed()
        ):
            return "当前工作客户AI已授权"
        ingestion = self._services.ingestion.last_result
        active = self._active_capture_contact()
        if (
            ingestion is None
            or active is None
            or active.id != contact.id
            or ingestion.contact is None
            or ingestion.contact.id != contact.id
            or not ingestion.conversation_id
        ):
            return "联系人级未允许"
        latest_event = self._services.analysis_events.latest_for_conversation_binding(
            ingestion.account_scope_id,
            contact.id,
            ingestion.conversation_id,
        )
        reply_pending = bool(
            latest_event is not None
            and self._analysis_event_has_reply_priority(latest_event, ingestion.conversation_id)
        )
        scope = self._proactive_cloud_gate.issue_scope(
            contact=contact,
            page_type=self._services.runtime.state.page.page_type.value,
            identity_stable=bool(ingestion.identity_stable),
            account_scope_id=ingestion.account_scope_id,
            active_contact_id=active.id,
            reply_pending=reply_pending,
        )
        if scope:
            return "主动联系AI已授权"
        if reply_pending:
            return "Reply优先"
        return "联系人级未允许"

    def _record_capture_state_commit(self, source: str, result, ingestion=None, reject_reason: str = "") -> None:
        identity_snapshot = self._services.current_identity.snapshot
        contact = self._services.current_identity.current_contact()
        contact_ref = ""
        if contact is not None:
            contact_ref = hashlib.sha256(contact.id.encode("utf-8")).hexdigest()[:12]
        identity_state = identity_snapshot.state.value
        if ingestion is not None:
            reject_reason = reject_reason or ("" if ingestion.accepted else ingestion.reason)
        try:
            self._services.logs.append(
                "info" if not reject_reason else "warning",
                "ui_state",
                "state_commit",
                source,
                {
                    "state_generation": self._services.runtime.state_generation,
                    "job_id": getattr(result, "job_id", None),
                    "snapshot_id": str(getattr(result, "snapshot_hash", "") or "")[:16],
                    "account_scope_id": getattr(result, "account_scope_id", None),
                    "target_instance_id": getattr(result, "target_instance_id", ""),
                    "window_instance": {
                        "hwnd": getattr(result, "hwnd", None),
                        "process_id": getattr(result, "process_id", None),
                        "window_generation": getattr(result, "window_generation", None),
                        "capture_generation": getattr(result, "capture_generation", None),
                    },
                    "page": getattr(getattr(result, "page", None), "page_type", None).value
                    if getattr(getattr(result, "page", None), "page_type", None) is not None
                    else self._services.runtime.state.page.page_type.value,
                    "contact_safe_ref": contact_ref,
                    "identity_state": identity_state,
                    "reject_reason": reject_reason,
                },
            )
        except Exception:
            pass

    def _sync_floating_content(self, contact: Contact | None = None, strategy: Strategy | None = None) -> None:
        if self._floating is None:
            return
        snapshot = getattr(self, "_frozen_screenshot_run", lambda: None)()
        if snapshot is not None and snapshot.sync_floating():
            return
        if (getattr(self, "_manual_generation_enabled", False)
                and self._manual_generation_phase == "success"
                and self._suggestion_result is not None and self._suggestion_result.allowed):
            saved = getattr(self, "_suggestion_reply_context", None)
            identity = self._services.current_identity.snapshot
            binding = self._suggestion_action_binding
            same_owner = (self._manual_window_owner(self._services.runtime.state) == self._manual_generation_owner
                          and identity.active and binding is not None
                          and identity.conversation_token == binding.identity_token
                          and not self._identity_suggestions_suspended)
            if not same_owner:
                if hasattr(self._floating, "set_copy_binding"):
                    self._floating.set_copy_binding(None)
                if hasattr(self._floating, "disable_suggestions"):
                    self._floating.disable_suggestions()
                return
            # A binding on a button is just a captured callback argument. The
            # actual clipboard callback rechecks DB/lease/profile and pixels.
            if hasattr(self._floating, "set_copy_binding"):
                self._floating.set_copy_binding(binding)
            original_contact = saved.contact if saved is not None else None
            if original_contact is not None and hasattr(self._floating, "update_context"):
                self._floating.update_context(contact_name=original_contact.display_name,
                    group_name="本次已生成", status="已保留", action="复制时重新核验原客户", app_label="微信")
            if hasattr(self._floating, "update_reply_result"):
                self._floating.update_reply_result(self._suggestion_result)
            if hasattr(self._floating, "set_review_status"):
                self._floating.set_review_status(getattr(self, "_manual_review_reason", ""))
            return
        if self._identity_suggestions_suspended:
            if hasattr(self._floating, "set_copy_binding"):
                self._floating.set_copy_binding(None)
            if hasattr(self._floating, "disable_suggestions"):
                self._floating.disable_suggestions()
            return
        if hasattr(self._floating, "set_copy_binding"):
            binding = self._suggestion_action_binding
            self._floating.set_copy_binding(binding if self._suggestion_binding_is_current(binding) else None)
        runtime = self._services.runtime.state
        active_contact = self._active_capture_contact()
        contact = active_contact
        if runtime.window.state != WindowState.VISIBLE or runtime.window.rect is None:
            if hasattr(self._floating, "disable_suggestions"):
                self._floating.disable_suggestions()
            return
        strategy = strategy if strategy is not None else (self._services.strategies.get(contact.strategy_id) if contact else self._services.strategies.get("default"))
        if hasattr(self._floating, "update_context"):
            steps = build_status_chain(
                runtime=self._services.runtime.state,
                contact=contact,
                strategy=strategy,
                config=self._config,
                reply_running=self._services.reply_tasks.is_running,
                provider_health=self._services.reply_generator.provider_health_summary(),
                has_active_suggestion=self._has_active_suggestion(),
                current_work_cloud_authorized=self._current_work_reply_cloud_allowed(),
            )
            ai_step = next((step for step in steps if step.stage == "AI"), steps[-1])
            ocr_step = next((step for step in steps if step.stage == "OCR"), None)
            display_step = ocr_step if ocr_step is not None and ocr_step.state in {"运行中", "读取消息"} else ai_step
            contact_name = contact.display_name if contact else (runtime.window.title or "未确认联系人")
            group_name = strategy.name if strategy else "默认分组"
            self._floating.update_context(
                contact_name=contact_name,
                group_name=group_name,
                status=f"{display_step.stage}:{display_step.state}",
                action=display_step.action or display_step.reason,
                app_label=runtime.window.app_label,
            )
            if display_step.stage == "OCR":
                if self._has_active_suggestion() and hasattr(self._floating, "update_reply_result"):
                    self._floating.update_reply_result(self._suggestion_result)
                elif hasattr(self._floating, "disable_suggestions"):
                    self._floating.disable_suggestions()
                return
            if ai_step.state not in {"就绪", "运行中"}:
                if self._has_active_suggestion() and hasattr(self._floating, "update_reply_result"):
                    self._floating.update_reply_result(self._suggestion_result)
                else:
                    self._floating.update_reply_result(
                        ReplyGenerationResult(False, f"blocked:{ai_step.reason}", [], self._config.ai.provider)
                    )
                return
        if self._suggestion_result is not None and hasattr(self._floating, "update_reply_result"):
            self._floating.update_reply_result(self._suggestion_result)

    def _overview_chat_rows(self, contact: Contact | None) -> list[tuple[str, ...]]:
        messages = self._last_ocr_messages
        if messages is None:
            return [("系统", "-", "尚未完成一次 OCR 采集", "-", "-", "等待")]
        speaker_map = {
            "me": "我",
            "other": "对方",
            "member": "群成员",
            "system": "系统",
            "unknown": "未知",
        }
        if not messages:
            meta = self._last_ocr_result_meta or {}
            warning = str(meta.get("warning") or "无")
            return [("系统", "-", f"本次 OCR 未解析出消息；warning={warning}", "-", "-", "空")]
        rows: list[tuple[str, ...]] = []
        for message in messages:
            speaker = speaker_map.get(message.speaker.value, message.speaker.value)
            sender = message.sender_name or ("我" if message.speaker.value == "me" else "对方")
            # This column is the time printed in the chat UI. Never substitute
            # the OCR completion time when the screenshot has no time anchor.
            time_text = message.time_text or "未识别"
            status = "不完整" if message.partial else "已解析"
            rows.append((speaker, sender, message.text, time_text, f"{message.confidence:.2f}", status))
        return rows

    def _overview_chat_summary(self) -> str:
        meta = self._last_ocr_result_meta
        if meta is None:
            return "尚未完成 OCR 采集"
        warning = str(meta.get("warning") or "")
        warning_text = f"；warning={warning}" if warning else ""
        return (
            f"job={meta.get('job_id', '-')} · page={meta.get('page', '-')} "
            f"· 解析 {len(self._last_ocr_messages or [])} 条 · 采集完成 {meta.get('created_at', '-')}{warning_text}"
        )

    def _build_reply_context(self) -> ReplyContext:
        contact = self._active_capture_contact()
        strategy = self._services.strategies.get(contact.strategy_id) if contact else self._services.strategies.get("default")
        messages = (self._reply_messages_for_contact(contact)[-12:] if contact else [])
        # Only user-confirmed memories may leave the machine: a rejected memory must
        # never reach the prompt, and a pending one waits for review in the 记忆 page.
        memories = self._services.memories.list_confirmed_for_contact(contact.id) if contact else []
        memory_context = None
        current_view = None
        current_message_id = messages[-1].id if messages else ""
        ingestion = self._services.ingestion.last_result
        if (
            contact is not None
            and ingestion is not None
            and ingestion.accepted
            and ingestion.identity_stable
            and ingestion.contact is not None
            and ingestion.contact.id == contact.id
            and ingestion.account_scope_id == contact.account_scope_id
            and ingestion.conversation_id
        ):
            if ingestion.observation_id:
                try:
                    observation = self._services.observations.get(ingestion.observation_id)
                    conversation = self._services.conversations.get(ingestion.conversation_id)
                    evidence = self._services.ingestion.last_evidence
                    current_view = build_current_message_view(
                        observation, conversation=conversation,
                        expected_account_scope_id=ingestion.account_scope_id,
                        expected_contact_id=contact.id,
                        expected_conversation_id=ingestion.conversation_id,
                        expected_target_instance_id=ingestion.target_instance_id,
                        expected_session_id=ingestion.session_id,
                        max_source_job_id=int(getattr(evidence, "job_id", 0) or 0),
                        max_capture_generation=self._services.runtime.state.capture_generation,
                        now=datetime.now(timezone.utc),
                    )
                    memory_context = self._services.reply_memory_context.build_for_current_view(current_view)
                    current_message_id = ""
                except (ValueError, TypeError, AttributeError):
                    # An untrusted current window must not silently fall back
                    # to an older customer's formal-message context.
                    current_view = None
                    current_message_id = ""
                    messages = []
                    memory_context = ReplyMemoryContext(
                        status="fallback_error", account_scope_id=ingestion.account_scope_id,
                        contact_id=contact.id, conversation_id=ingestion.conversation_id,
                        current_message_id="", error="CurrentViewUnavailable",
                        current_view_source_ref=ingestion.observation_id,
                        current_view_source_kind="visible_observation", current_view_verified=False,
                    )
            elif not ingestion.observation_id:
                memory_context = self._services.reply_memory_context.build(
                    account_scope_id=ingestion.account_scope_id,
                    contact_id=contact.id,
                    conversation_id=ingestion.conversation_id,
                    current_message_id=current_message_id,
                )
        if ingestion is not None and ingestion.observation_id and current_view is None:
            # Presence of a visible source forbids formal fallback, including
            # incomplete session/target/conversation/account bindings that did
            # not enter the validation branch above.
            messages = []
            current_message_id = ""
            memory_context = ReplyMemoryContext(
                status="fallback_error", account_scope_id=contact.account_scope_id if contact else "",
                contact_id=contact.id if contact else "", conversation_id=ingestion.conversation_id or "",
                current_message_id="", error="CurrentViewUnavailable",
                current_view_source_ref=ingestion.observation_id,
                current_view_source_kind="visible_observation", current_view_verified=False,
            )
        current_message = next(
            (message for message in messages if message.id == current_message_id), None
        )
        if current_view is not None:
            current_message = current_view.current_message
        knowledge_context = self._services.product_knowledge_context.build(
            memory_context=memory_context,
            current_message=current_message.text if current_message is not None else "",
        )
        from whochat.services.communication_intent import select_communication_intent
        work_intent = select_communication_intent(current_view)
        context = ReplyContext(
            runtime=self._services.runtime.state,
            contact=contact,
            strategy=strategy,
            messages=messages,
            memories=memories,
            memory_context=memory_context,
            current_message_id=current_message_id,
            knowledge_context=knowledge_context,
            current_message_view=current_view,
            current_work_mode=work_intent.mode,
        )
        from whochat.ai.photos import bind_reply_photos
        photos, photo_error = bind_reply_photos(
            current_view, getattr(getattr(self._services, "pipeline", None), "last_result", None), context.runtime,
        )
        context = replace(context, photos=photos, photo_error=photo_error)
        feedback = getattr(self._services, "writing_feedback", None)
        if feedback is not None:
            try:
                context = replace(context, writing_feedback=feedback.learning_for(context))
            except Exception as exc:
                self.statusBar().showMessage(f"本次未能读取写作反馈（{type(exc).__name__}）；客户核验规则不变", 6000)
        self._current_visible_ui_revision = _reply_context_hash(context)
        return self._authorize_current_reply_context(context)

    def _on_link_imported_history(self) -> None:
        """Explicit local mapping; no OCR/AI/send, no nickname auto-merge."""
        if self._manual_generation_phase in {'recognizing','generating','verifying'}:
            self.statusBar().showMessage('请先等待本次生成结束，再关联历史。', 6000)
            return
        contact = self._active_capture_contact()
        ingestion = self._services.ingestion.last_result
        identity = self._services.current_identity.snapshot
        if (contact is None or ingestion is None or not ingestion.accepted or not ingestion.identity_stable
                or ingestion.contact is None or ingestion.contact.id != contact.id
                or not ingestion.conversation_id or not identity.active
                or identity.contact_id != contact.id or identity.account_scope_id != contact.account_scope_id):
            self.statusBar().showMessage('请先打开目标客户，让 WhoChat 确认客户与微信账号，再关联历史。', 8000)
            return
        from whochat.services.imported_history import ImportedHistoryService
        service = ImportedHistoryService(self._services.db)
        binding = (contact.account_scope_id,contact.id,ingestion.conversation_id,
                   self._manual_window_owner(self._services.runtime.state))
        if (not binding[3] or ingestion.account_scope_id != contact.account_scope_id
                or ingestion.target_instance_id != identity.target_instance_id):
            self.statusBar().showMessage('当前微信账号尚未核验，未关联历史。', 8000)
            return
        try:
            choices = service.candidates(contact.account_scope_id,contact.id)
            if not choices:
                QMessageBox.information(self,'关联历史','没有发现本账号下待关联的同名历史。已经关联的历史会在生成时自动核验读取。')
                return
            labels = [f"{i+1}. {c['name']}｜{c['count']} 条｜{c['first'][:10]} 至 {c['last'][:10]}" for i,c in enumerate(choices)]
            selected, ok = QInputDialog.getItem(self,'选择历史记录','同名只用于查找，不能证明是同一客户。请选择要核对的记录：',labels,0,False)
            if not ok:
                return
            chosen = choices[labels.index(selected)]
            preview = '\n'.join(s['date'][:10]+' 客户：'+s['text'] for s in chosen['samples'])
            dialog = QMessageBox(self)
            dialog.setWindowTitle('确认历史客户归属')
            dialog.setTextFormat(Qt.TextFormat.PlainText)
            dialog.setText(f"当前客户：{contact.display_name}\n历史名称：{chosen['name']}\n"
                           f"历史来源标识：{chosen['source_identity']}\n{chosen['count']} 条记录\n\n{preview}\n\n"
                           '请核实这确实是同一个客户。确认后只建立本地归属关联；不调用 AI、不发送消息。'
                           '以后手动生成时，会使用经过筛选、脱敏、带日期的历史文字。')
            dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            dialog.setDefaultButton(QMessageBox.StandardButton.No)
            if dialog.exec() != QMessageBox.StandardButton.Yes:
                return
            fresh_contact = self._active_capture_contact()
            fresh_ingestion = self._services.ingestion.last_result
            fresh_identity = self._services.current_identity.snapshot
            if (fresh_contact is None or fresh_ingestion is None or not fresh_identity.active
                    or not fresh_ingestion.accepted or not fresh_ingestion.identity_stable
                    or fresh_ingestion.contact is None or fresh_ingestion.contact.id != contact.id
                    or fresh_ingestion.account_scope_id != contact.account_scope_id
                    or fresh_ingestion.target_instance_id != identity.target_instance_id
                    or fresh_identity.target_instance_id != identity.target_instance_id
                    or fresh_identity.contact_id != contact.id
                    or fresh_identity.account_scope_id != contact.account_scope_id
                    or (fresh_contact.account_scope_id,fresh_contact.id,fresh_ingestion.conversation_id,
                        self._manual_window_owner(self._services.runtime.state)) != binding):
                raise ValueError('确认期间客户或微信账号变化，未关联')
            count = service.confirm(scope=binding[0],contact=binding[1],conversation=binding[2],
                mapping_id=chosen['id'],ticket=chosen['ticket'],user_confirmed=True)
            self.statusBar().showMessage(f'已关联 {count} 条历史；未调用 AI。下次点击生成时将核验并选取相关记录。',12000)
        except Exception as exc:
            self.statusBar().showMessage('历史未关联：'+str(exc),10000)

    def _build_reply_context_for_event(self, event: AIAnalysisEvent) -> ReplyContext | None:
        if self._analysis_event_matches_active_context(event):
            current = self._build_reply_context()
            if current.current_message_view is not None:
                return current if current.current_message_view.has_unanswered_customer_message else None
            if getattr(self._services.ingestion.last_result, "observation_id", ""):
                return None
        context = build_reply_context_for_analysis_event(self._services, event)
        return self._authorize_current_reply_context(context) if context is not None else None

    def _authorize_current_reply_context(self, context: ReplyContext) -> ReplyContext:
        validator = getattr(getattr(self._services, "reply_generator", None), "identity_lease_validator", None)
        try:
            lease_valid = bool(callable(validator) and validator(context))
        except Exception:
            lease_valid = False
        authorized = authorize_current_work_context(context, identity_lease_valid=lease_valid)
        view = context.current_message_view
        identity = self._services.current_identity.snapshot
        # Attempt timing is local metadata, not a cloud grant. Keep the
        # recognition deadline even while the strict live lease is waiting.
        if (view is not None and context.memory_context is not None
                and context.memory_context.current_view_verified
                and identity.active and identity.contact_id == view.contact_id
                and identity.account_scope_id == view.account_scope_id
                and identity.target_instance_id == view.target_instance_id):
            identity_group = (identity.conversation_token, view.session_id)
            key = (*identity_group, _reply_context_hash(context))
            attempts = getattr(self, "_current_work_attempt_deadlines", None)
            if attempts is None:
                attempts = self._current_work_attempt_deadlines = {}
            if key not in attempts:
                from time import monotonic as current_work_clock
                now = current_work_clock()
                start = now
                anchor = getattr(self, "_current_work_capture_anchor", None)
                if anchor is not None and anchor[:6] == (
                    view.account_scope_id, view.target_instance_id, view.contact_id,
                    view.conversation_id, view.session_id, view.observation_id,
                ):
                    start = anchor[6]
                # First current-customer work includes the time between
                # reliable identity recognition and full message context.
                identity_anchor = getattr(self, "_current_work_identity_anchor", None)
                if not any(old[:2] == identity_group for old in attempts) and (
                    identity_anchor is not None and identity_anchor[0] == identity.conversation_token
                ):
                    start = min(start, identity_anchor[1])
                attempts[key] = (start, start + 30.0)
            start, deadline = attempts[key]
            authorized = replace(authorized, current_work_started_monotonic=start,
                                 current_work_deadline_monotonic=deadline)
        if lease_valid:
            self._reply_authorization_context = authorized
        return authorized

    def _current_work_reply_cloud_allowed(self) -> bool:
        """Read the last bounded Reply grant without rebuilding a profile."""

        context = getattr(self, "_reply_authorization_context", None)
        if context is None:
            return False
        validator = getattr(self._services.reply_generator, "identity_lease_validator", None)
        try:
            return bool(callable(validator) and validator(context) and cloud_reply_authorized(context))
        except Exception:
            return False

    def _queue_auto_analysis_event(self, event_id: str) -> None:
        if not event_id or event_id in self._pending_auto_event_ids or event_id in self._analysis_event_by_job.values():
            return
        self._pending_auto_event_ids.append(event_id)

    def _analysis_event_matches_stable_binding(self, event: AIAnalysisEvent) -> bool:
        contact = self._active_capture_contact()
        ingestion = self._services.ingestion.last_result
        return bool(
            contact is not None
            and ingestion is not None
            and ingestion.accepted
            and ingestion.identity_stable
            and ingestion.contact is not None
            and contact.id == ingestion.contact.id
            and analysis_event_matches_binding(
                event,
                account_scope_id=ingestion.account_scope_id,
                target_instance_id=ingestion.target_instance_id,
                contact_id=contact.id,
                conversation_id=ingestion.conversation_id or "",
                page_allowed=True,
            )
        )

    def _analysis_event_matches_active_context(self, event: AIAnalysisEvent) -> bool:
        return bool(
            self._services.runtime.state.page.can_generate_reply
            and self._analysis_event_matches_stable_binding(event)
        )

    def _load_latest_displayable_ai_result(self, ingestion=None, *, render_context: ReplyContext | None = None) -> bool:
        if getattr(self, "_manual_generation_enabled", False):
            return False
        """Restore the newest durable AI state for the exact stable chat binding."""
        ingestion = ingestion or self._services.ingestion.last_result
        contact = self._active_capture_contact()
        if (
            contact is None
            or ingestion is None
            or not ingestion.accepted
            or not ingestion.identity_stable
            or ingestion.contact is None
            or ingestion.contact.id != contact.id
            or not ingestion.conversation_id
            or not ingestion.account_scope_id
            or not ingestion.target_instance_id
        ):
            return False
        if contact.account_scope_id != ingestion.account_scope_id:
            return False

        event = self._services.analysis_events.latest_for_conversation_binding(
            ingestion.account_scope_id,
            contact.id,
            ingestion.conversation_id,
        )
        # Restoring a completed local result is not a new cloud-AI decision.
        # Once the two-frame identity lock has established the exact account,
        # WeChat instance, contact and conversation binding, a later mutable
        # page=unknown (for example while WhoChat itself has focus) must not
        # hide the durable result.  New event creation and worker completion
        # continue to use `_analysis_event_matches_active_context`, including
        # the chat-page gate.
        if event is None or not analysis_event_matches_display_binding(
            event,
            account_scope_id=ingestion.account_scope_id,
            contact_id=contact.id,
            conversation_id=ingestion.conversation_id,
        ):
            return False
        if not self._analysis_event_has_reply_priority(event, ingestion.conversation_id):
            return False
        live_context = self._build_reply_context()
        if current_context_source_error(live_context):
            return False
        if live_context.current_message_view is not None and (
            not event.context_hash or event.context_hash != _reply_context_hash(live_context)
        ):
            return False

        if event.status == "completed":
            try:
                payload = json.loads(event.result_json)
                trace = payload.get("performance_trace") if isinstance(payload, dict) else None
                profile = trace.get("prompt_profile") if isinstance(trace, dict) else None
                policy_version = profile.get("policy_version") if isinstance(profile, dict) else None
            except (TypeError, json.JSONDecodeError):
                policy_version = None
            if policy_version != REPLY_POLICY_VERSION:
                displayed = self._suggestion_result
                displayed_trace = displayed.performance_trace if displayed is not None else {}
                displayed_profile = displayed_trace.get("prompt_profile") if isinstance(displayed_trace, dict) else None
                if (
                    self._suggestion_mode == "reply"
                    and self._has_active_suggestion()
                    and isinstance(displayed_profile, dict)
                    and displayed_profile.get("policy_version") == REPLY_POLICY_VERSION
                    and self._suggestion_context_hash == _reply_context_hash(render_context or self._build_reply_context())
                ):
                    return True
                self._suggestion_result = None
                self._suggestion_context_hash = None
                self._displayed_analysis_event_id = event.id
                self._render_reply_empty_state("回复建议规则已更新，请点击重新生成")
                return True

        # The same durable event changes state while the async worker runs.  A
        # pending placeholder is not the final result: once SQLite says the
        # event completed, it must be replaced even though the event id did not
        # change.  Conversely, an already-rendered completed result can be
        # reused without rebuilding the panel on every unchanged OCR frame.
        if self._displayed_analysis_event_id == event.id and self._suggestion_result is not None:
            displayed = self._suggestion_result
            if event.status in {"pending", "running"} and displayed.status.startswith("reply_pending"):
                return True
            if (
                event.status == "completed"
                and displayed.allowed
                and len(displayed.suggestions) == 2
                and self._suggestion_context_hash == (event.context_hash or None)
            ):
                self._services.reply_pipeline_metrics.mark(event.id, "ui_ready_at")
                return True

        if event.status in {"pending", "running"}:
            self._suggestion_result = None
            self._suggestion_context_hash = None
            self._displayed_analysis_event_id = event.id
            self._render_reply_suggestions(
                ReplyGenerationResult(
                    False,
                    f"reply_pending:saved_event={event.id}",
                    [],
                    self._config.ai.provider,
                )
            )
            self.append_log(f"saved_ai_result_waiting_for_latest_event: event={event.id} status={event.status}")
            return True

        if event.status != "completed":
            self._suggestion_result = None
            self._suggestion_context_hash = None
            self._displayed_analysis_event_id = event.id
            self._render_reply_suggestions(
                ReplyGenerationResult(False, f"blocked:saved_event_{event.status}", [], self._config.ai.provider)
            )
            return True

        restored, error = reply_generation_result_from_json(event.result_json, context=live_context)
        if restored is None:
            self._suggestion_result = None
            self._suggestion_context_hash = None
            self._displayed_analysis_event_id = event.id
            self._render_reply_suggestions(
                ReplyGenerationResult(
                    False,
                    f"blocked:saved_result_invalid:{error or 'unknown'}",
                    [],
                    self._config.ai.provider,
                )
            )
            self.append_log(f"saved_ai_result_rejected: event={event.id} reason={error or 'unknown'}", "warning")
            return True

        self._services.reply_pipeline_metrics.mark(event.id, "ui_render_started_at")
        self._render_reply_suggestions(restored, context=render_context)
        self._services.reply_pipeline_metrics.mark(event.id, "ui_render_completed_at")
        self._suggestion_context_hash = event.context_hash or None
        self._displayed_analysis_event_id = event.id
        self._services.reply_pipeline_metrics.mark(event.id, "ui_ready_at")
        self.append_log(
            f"saved_ai_result_restored: event={event.id} contact={event.contact_id} conversation={event.conversation_id}"
        )
        return True

    def _analysis_event_has_reply_priority(self, event: AIAnalysisEvent, conversation_id: str) -> bool:
        """Reply owns the panel only while its incoming message is still current."""
        current = self._build_reply_context()
        if current_context_source_error(current):
            return False
        if current.current_message_view is not None:
            if not current.current_message_view.has_unanswered_customer_message:
                return False
            if event.context_hash and event.context_hash != _reply_context_hash(current):
                return False
        source = self._services.messages.get(event.message_id)
        if source is None or source.speaker != Speaker.OTHER:
            return False
        try:
            source_at = datetime.fromisoformat(source.observed_at.replace("Z", "+00:00"))
            if source_at.tzinfo is None:
                source_at = source_at.astimezone()
            age = datetime.now().astimezone() - source_at.astimezone()
        except (AttributeError, TypeError, ValueError):
            return False
        # A durable event from old history must not permanently suppress V1.
        # Fresh pending/running events still have absolute priority over it.
        if age < -timedelta(minutes=5) or age > timedelta(days=1):
            return False
        if event.status in {"pending", "running"}:
            return True
        tail = self._services.messages.list_timeline_tail(conversation_id, 12)
        if not tail:
            return False
        try:
            source_index = next(index for index, item in enumerate(tail) if item.id == source.id)
        except StopIteration:
            return False
        return not any(item.speaker == Speaker.ME for item in tail[source_index + 1 :])

    def _reply_messages_for_contact(self, contact: Contact) -> list[Message]:
        ingestion = self._services.ingestion.last_result
        if ingestion and ingestion.identity_stable and ingestion.contact and ingestion.contact.id == contact.id and ingestion.conversation_id:
            return self._services.messages.list_timeline(ingestion.conversation_id, 2000)[-30:]
        # Never send an uncommitted OCR screen to AI. The fallback is persisted
        # local history only and is used for manual review outside auto capture.
        return list(reversed(self._services.messages.list_for_contact(contact.id, 60)))[-30:]

    def _has_active_suggestion(self) -> bool:
        result = self._suggestion_result
        return bool(
            not (getattr(self, "_manual_generation_enabled", False) and getattr(self, "_manual_review_reason", ""))
            and not self._identity_suggestions_suspended and result and result.allowed and result.suggestions
            and result.status != "local_preview"
        )

    def _has_matching_activation(self, contact_id: str) -> bool:
        outcome = self._proactive_outcome
        return bool(
            self._suggestion_mode == "proactive"
            and outcome is not None
            and outcome.request.contact_id == contact_id
            and outcome.result is not None
            and outcome.result.allowed
            and outcome.result.status != "proactive:local_safe_fallback"
            and len(outcome.result.suggestions) == 1
        )

    def _show_identity_verifying_state(self) -> None:
        """Hide former chat text while retaining its local edit for reacquire."""

        self._suggestion_action_binding = None
        self._sync_floating_content()
        if self._suggestions_panel is None:
            return
        layout = self._suggestions_panel.layout()
        self._manual_review_notice = None
        while layout.count() > 1:
            item = layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        notice = QLabel("正在确认当前客户")
        notice.setObjectName("Muted")
        layout.addWidget(notice)
        self._resize_suggestion_list_to_content()

    def _render_reply_empty_state(self, message: str) -> None:
        if self._suggestions_panel is None:
            return
        self._render_reply_suggestions(
            ReplyGenerationResult(False, f"reply_empty:{message}", [], self._config.ai.provider)
        )
        # Empty/identity-pending is a UI placeholder, not a contact-bound
        # suggestion result.  Keeping it in `_suggestion_result` makes stale
        # state look present to the floating panel and regression gates.
        self._suggestion_result = None
        self._suggestion_context_hash = None
        self._proactive_outcome = None
        self._proactive_feedback_status = {}
        self._proactive_rejected_feedback = {}
        self._proactive_regeneration_used = False
        self._sync_floating_content()

    def _reply_block_is_definitive(self, result: ReplyGenerationResult) -> bool:
        """True when no reply can be produced until the user changes something.

        Derived from the contact / strategy / provider state rather than from the
        status wording, so it cannot drift when a message string is reworded.
        Request-policy and page-state blocks stay transient (``blocked:*``,
        ``reply_pending*`` and an unrecognised page).
        """
        if result.allowed or result.status.startswith(("blocked:", "reply_pending")):
            return False
        context = self._build_reply_context()
        contact = context.contact
        if contact is None:
            return False
        if contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED} or contact.merged_into:
            return True
        if self._config.ai.provider == "Disabled":
            return True
        strategy = self._services.strategies.get(contact.strategy_id) if contact.strategy_id else None
        if strategy is not None and strategy.requires_manual_reply:
            return True
        cloud_configured = self._config.ai.provider in {"OpenAI", "OpenAI Compatible"} and bool(self._config.ai.api_key)
        return bool(cloud_configured and not cloud_reply_authorized(context))

    def _build_reply_result(self) -> ReplyGenerationResult:
        if getattr(self, "_manual_generation_enabled", False):
            return ReplyGenerationResult(False, "blocked:manual_generation_button_required", [], self._config.ai.provider)
        return self._services.reply_generator.generate(self._build_reply_context(), self._config)

    def _render_reply_suggestions(
        self,
        result: ReplyGenerationResult,
        *,
        context: ReplyContext | None = None,
        mode: str = "reply",
        evidence_override: str | None = None,
        title_override: str | None = None,
    ) -> None:
        # Unchanged successful contexts are reused before dispatch. A new
        # waiting/terminal state must never be hidden by an old sales draft.
        self._suggestion_mode = mode
        if self._suggestions_title is not None:
            default_title = "主动联系建议" if mode == "proactive" else "回复建议"
            if mode == "reply" and context is not None and context.current_work_mode == "next_step":
                default_title = "下一步沟通建议"
            elif mode == "reply" and context is not None and context.current_work_mode == "reactivation":
                default_title = "回访／激活建议"
            self._suggestions_title.setText(title_override or default_title)
        context = context or (None if evidence_override is not None else self._build_reply_context())
        if getattr(self._services, "writing_feedback", None) is not None and mode == "reply":
            from whochat.ui.writing_feedback import capture_display
            result = capture_display(self, result, context)
        else:
            self._writing_feedback_ids = ()
        self._suggestion_result = result
        self._suggestion_reply_context = deepcopy(context) if mode == "reply" and result.allowed else None
        profile = getattr(self._services.reply_generator, "display_profile_hash", None)
        self._suggestion_reply_profile = profile(self._config) if callable(profile) else None
        self._suggestion_action_binding = self._bind_suggestion_actions(mode, context) if result.allowed else None
        self._sync_floating_content()
        if self._suggestions_panel is None:
            return
        layout = self._suggestions_panel.layout()
        self._manual_review_notice = None
        while layout.count() > 1:
            item = layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            nested = item.layout()
            if nested is not None:
                while nested.count():
                    nested_item = nested.takeAt(0)
                    nested_widget = nested_item.widget()
                    if nested_widget is not None:
                        nested_widget.deleteLater()
        meta = QLabel(_clip_debug_text(f"{result.provider} · {len(result.suggestions)} 条建议", 28))
        meta.setObjectName("TinyMuted")
        meta.setMaximumHeight(18)
        layout.addWidget(meta)
        # The evidence line stays *visible* rather than tooltip-only: the product
        # requirement is that the user can see which context (contact, group, how
        # many messages and memories, page type, cloud permission) fed the reply.
        # The multi-line diagnostics remain available as the tooltip.
        summary = evidence_override or _reply_evidence_summary(context, result)
        evidence = QLabel(summary)
        evidence.setObjectName("Muted")
        evidence.setWordWrap(True)
        evidence.setToolTip(
            summary if context is None else f"{summary}\n\n{_reply_evidence_detail(context, result)}"
        )
        layout.addWidget(evidence)
        if getattr(self, "_manual_generation_enabled", False) and result.allowed:
            review = QLabel("已保留本客户的建议（待核验）。来源过期时，点击复制会在本地重新核验；聊天已更新时，请点击「生成建议」。")
            review.setObjectName("Muted")
            review.setWordWrap(True)
            review.setVisible(bool(getattr(self, "_manual_review_reason", "")))
            self._manual_review_notice = review
            layout.addWidget(review)
        if result.status.startswith("reply_pending"):
            pending = QLabel(
                "等待请求间隔，尚未发起本次 AI 请求" if "waiting_rate_limit" in result.status else
                "上一请求尚未结束，本次尚未发起" if "waiting_latest" in result.status else "生成中"
            )
            pending.setObjectName("Muted")
            layout.addWidget(pending)
            self._resize_suggestion_list_to_content()
            return
        if result.status.startswith("reply_empty:"):
            empty = QLabel(result.status.partition(":")[2] or "暂无回复建议")
            empty.setObjectName("Muted")
            empty.setWordWrap(True)
            layout.addWidget(empty)
            self._resize_suggestion_list_to_content()
            return
        if result.status.startswith("proactive_notice:"):
            notice = QLabel(result.status.partition(":")[2] or "暂无主动联系建议")
            notice.setObjectName("Muted")
            notice.setWordWrap(True)
            layout.addWidget(notice)
            self._resize_suggestion_list_to_content()
            return
        if result.status == "local_preview":
            preview = QLabel("模型未配置或本地后端未接入，本次未生成可用建议；仅可查看本地资料。")
            preview.setObjectName("Muted")
            preview.setWordWrap(True)
            layout.addWidget(preview)
            self._resize_suggestion_list_to_content()
            return
        if not result.allowed:
            from whochat.services.status import reply_content_failure_message
            blocked = QLabel(reply_content_failure_message(result.status) or f"已阻断：{result.status}")
            blocked.setObjectName("Muted")
            blocked.setWordWrap(True)
            blocked.setToolTip(result.status if "blocked:invalid_analysis:" in result.status
                               else "确认聊天页、聊天对象、分组策略和 AI 设置后再生成。")
            layout.addWidget(blocked)
            self._resize_suggestion_list_to_content()
            return
        analysis_values = [
            ("当前判断", result.problem_assessment or result.customer_summary),
            ("优先追问", result.ask_first),
        ]
        for advice in result.ingredient_advice:
            analysis_values.append((
                {"generic_category": "通用方向", "active_ingredient": "有效成分", "functional_ingredient": "功能成分"}.get(advice.knowledge_kind, "建议主要成分"),
                f"{advice.name}：{advice.role}；{advice.rationale}；条件：{'；'.join(advice.applicable_conditions)}",
            ))
        if result.uncertainties:
            analysis_values.append(("需注意/待确认", "；".join(result.uncertainties)))
        for label, value in analysis_values:
            if not value:
                continue
            item = QLabel(f"{label}：{value}")
            item.setObjectName("Muted")
            item.setWordWrap(True)
            layout.addWidget(item)
        for suggestion_index, suggestion in enumerate(result.suggestions):
            layout.addWidget(self._suggestion_row(suggestion, suggestion_index))
        self._resize_suggestion_list_to_content()

    def _resize_suggestion_list_to_content(self) -> None:
        if self._suggestions_panel is None:
            return
        layout = self._suggestions_panel.layout()
        layout.activate()
        self._suggestions_panel.setMinimumHeight(layout.sizeHint().height())
        self._suggestions_panel.updateGeometry()

    def _refresh_reply_suggestions(self, *, automatic: bool = False) -> int | None:
        if getattr(self, "_manual_generation_enabled", False) and not self._manual_generation_allow_submit:
            return None
        self._services.proactive_tasks.invalidate("reply_requested")
        self._proactive_last_request_key = ""
        context = self._build_reply_context()
        digest = _reply_context_hash(context)
        source_error = current_context_source_error(context)
        if source_error:
            self._render_reply_suggestions(ReplyGenerationResult(
                False, f"blocked:{source_error}", [], self._config.ai.provider
            ), context=context)
            return None
        if not automatic and context.current_message_view is not None:
            # Only this explicit user action creates a fresh attempt for the
            # same content. Automatic repaint/capture/cache recovery never do.
            from time import monotonic as current_work_clock
            from uuid import uuid4
            start = (self._manual_generation_started if getattr(self, "_manual_generation_enabled", False)
                     else current_work_clock())
            view = context.current_message_view
            key = (self._services.current_identity.snapshot.conversation_token, view.session_id, digest)
            attempts = getattr(self, "_current_work_attempt_deadlines", None)
            if attempts is None:
                attempts = self._current_work_attempt_deadlines = {}
            attempts[key] = (start, start + 30.0)
            context = replace(context, current_work_started_monotonic=start,
                              current_work_deadline_monotonic=start + 30.0,
                              manual_retry_id=uuid4().hex)
            self.append_log("current_work_explicit_retry: fresh_attempt_deadline")
        if automatic and self._has_active_suggestion() and self._suggestion_context_hash == digest:
            self.append_log("auto_reply_reused: unchanged_context")
            self.statusBar().showMessage("聊天上下文未变化，继续显示上一轮回复建议", 2500)
            self._refresh_overview_data()
            return None
        recovery = getattr(self._services.reply_generator, "recover_cached_only", None)
        cached = recovery(context, self._config) if callable(recovery) else None
        if cached is not None:
            if (getattr(self, "_manual_generation_enabled", False)
                    and time.monotonic() >= self._manual_generation_started + 30.0):
                return None
            self._render_reply_suggestions(cached, context=context)
            self._suggestion_context_hash = digest
            self._reply_retry_pending = False
            return None
        if not automatic and self._requires_cloud_prompt_review(context) and not self._confirm_cloud_prompt(context):
            self.statusBar().showMessage("已取消云端 AI 请求", 2500)
            self.append_log("reply_suggestions_cancelled: cloud_prompt_review")
            self._refresh_overview_data()
            return None
        job_id = self._services.reply_tasks.submit(context, self._config)
        if job_id is None:
            self._reply_retry_pending = True
            self.statusBar().showMessage("AI 正在生成上一条建议，请稍后", 2500)
            self.append_log("reply_suggestions_skipped: busy", "warning")
            self._refresh_overview_data()
            return None
        self._reply_retry_pending = False
        self._latest_reply_job_id = job_id
        self._reply_context_by_job[job_id] = deepcopy(context)
        if not self._has_active_suggestion() or self._suggestion_context_hash != digest:
            self._suggestion_context_hash = None
            pending = ReplyGenerationResult(False, f"reply_pending: job={job_id}", [], self._config.ai.provider)
            self._render_reply_suggestions(pending)
        else:
            self._sync_floating_content()
        self._refresh_overview_data()
        self.statusBar().showMessage("正在生成回复建议", 2500)
        self.append_log(f"reply_suggestions_submitted: job={job_id}")
        return job_id

    def _auto_generate_visible_reply(self) -> bool:
        if getattr(self, "_manual_generation_enabled", False):
            return False
        context = self._build_reply_context()
        view = context.current_message_view
        if view is None or view.current_message is None or view.current_message.speaker not in {Speaker.OTHER, Speaker.ME}:
            return False
        digest = _reply_context_hash(context)
        # A completed displayed suggestion has a separate display lease. The
        # generation deadline must not disable a later copy of that result.
        if self._has_active_suggestion() and self._suggestion_context_hash == digest:
            return True
        from time import monotonic as current_work_clock
        deadline = context.current_work_deadline_monotonic
        if deadline is not None and current_work_clock() >= deadline:
            self._proactive_stability_retry_pending = False
            self._reply_retry_pending = False
            self._render_reply_suggestions(ReplyGenerationResult(
                False, "blocked:current_work_timeout：识别后等待已超过时限，本次未生成可用建议", [], self._config.ai.provider,
            ), context=context)
            return True
        # This path does not create Message/AIAnalysisEvent/MemoryJob rows and
        # is independent of A/B proactive-contact eligibility.
        source_error = current_context_source_error(context)
        if source_error:
            self._render_reply_empty_state(f"当前消息暂不可用：{source_error}")
            return True
        validator = getattr(self._services.reply_generator, "identity_lease_validator", None)
        if not callable(validator) or not validator(context):
            if not self._proactive_stability_retry_pending:
                self._proactive_stability_retry_pending = True
                QTimer.singleShot(800, self._retry_proactive_after_identity_lease)
            self._render_reply_empty_state("正在确认当前客户，确认后生成回复建议")
            return True
        attempt_key = "|".join((
            digest, view.target_instance_id, view.session_id,
            str(context.runtime.window.hwnd), str(context.runtime.window_generation),
            str(context.runtime.account_identity_epoch), context.runtime.account_identity_runtime_key,
        ))
        pending = getattr(self._services.reply_generator, "has_pending_response", None)
        if self._visible_reply_attempt_key == attempt_key or (
            callable(pending) and pending(context, self._config)
        ):
            # Revalidation is local even with zero remaining HTTP slots.
            # A miss must not retry an unchanged attempted customer turn.
            recovery = getattr(self._services.reply_generator, "recover_cached_only", None)
            cached = recovery(context, self._config) if callable(recovery) else None
            if cached is not None:
                self._render_reply_suggestions(cached, context=context)
                self._suggestion_context_hash = digest
                self._reply_retry_pending = False
                self._visible_reply_attempt_key = attempt_key
                return True
            self._visible_reply_attempt_key = attempt_key
            return True
        self._visible_reply_attempt_key = attempt_key
        self._services.proactive_tasks.invalidate("visible_reply_priority")
        self._proactive_last_request_key = ""
        job_id = self._services.reply_tasks.submit(context, self._config)
        if job_id is None:
            self._render_reply_empty_state("回复任务不可用，请点击重新生成")
            return True
        self._latest_reply_job_id = job_id
        self._reply_context_by_job[job_id] = deepcopy(context)
        self._suggestion_context_hash = None
        self._render_reply_suggestions(
            ReplyGenerationResult(False, f"reply_pending: job={job_id}", [], self._config.ai.provider),
            context=context,
        )
        self.append_log(f"visible_reply_submitted: job={job_id} revision={view.content_revision[:12]}")
        return True

    def _withdraw_outdated_reply(self) -> None:
        latest = self._reply_context_by_job.get(self._latest_reply_job_id)
        if latest is None and not (self._suggestion_mode == "reply" and self._suggestion_context_hash):
            return
        current = self._build_reply_context()
        digest = _reply_context_hash(current)
        old_digest = _reply_context_hash(latest) if latest is not None else self._suggestion_context_hash
        if old_digest == digest:
            return
        self._latest_reply_job_id = None
        self._services.reply_tasks.invalidate("current_message_revision_changed")
        self._suggestion_context_hash = None
        self._render_reply_empty_state("聊天内容已更新，正在判断本次回复")

    def _auto_generate_reply_after_ocr(self) -> None:
        if getattr(self, "_manual_generation_enabled", False):
            return
        if not self._pending_auto_event_ids:
            return
        # Only the newest pending customer turn is useful now. Superseded
        # durable events remain auditable but must not queue old suggestions.
        event_id = self._pending_auto_event_ids[-1]
        superseded_ids = self._pending_auto_event_ids[:-1]
        self._pending_auto_event_ids.clear()
        for old_id in superseded_ids:
            old = self._services.analysis_events.get(old_id)
            if old is not None and old.status == "pending":
                self._services.analysis_events.update_status(
                    old.message_id, "blocked", failure_reason="superseded_by_latest_customer_turn"
                )
        event = self._services.analysis_events.get(event_id)
        if event is None or event.status != "pending":
            return
        source_message = self._services.messages.get(event.message_id)
        if source_message is None:
            self._services.analysis_events.update_status(
                event.message_id,
                "blocked_invalid_trigger_context",
                failure_reason="source_message_missing",
            )
            self.append_log(f"auto_reply_blocked: source_message_missing event={event.id}", "warning")
            return
        self._services.reply_pipeline_metrics.ensure(event, source_message)
        self._services.reply_pipeline_metrics.mark(event.id, "context_started_at")
        context = self._build_reply_context_for_event(event)
        if context is None:
            self._services.analysis_events.update_status(
                event.message_id,
                "blocked_invalid_trigger_context",
                failure_reason="event_binding_or_snapshot_invalid",
            )
            self.append_log(f"auto_reply_blocked: invalid_trigger_context event={event.id}", "warning")
            return
        self._services.reply_pipeline_metrics.mark(event.id, "context_completed_at")
        if context.memory_context is not None:
            try:
                self._services.analysis_events.attach_memory_context_snapshot(
                    event.message_id,
                    payload=context.memory_context.as_payload(),
                    memory_hash=context.memory_context.memory_hash,
                    status=context.memory_context.status,
                    error=context.memory_context.error,
                )
            except Exception as exc:
                self.append_log(
                    f"reply_memory_snapshot_failed: event={event.id} error={type(exc).__name__}",
                    "warning",
                )
        digest = _reply_context_hash(context)
        job_id = self._services.reply_tasks.submit(context, self._config)
        if job_id is None:
            self._services.analysis_events.defer_pending(event.message_id, "reply_worker_busy")
            self._queue_auto_analysis_event(event.id)
            return
        self._services.analysis_events.mark_started(event.message_id, digest)
        self._services.reply_pipeline_metrics.mark(event.id, "reply_started_at")
        # ``ReplyTaskService`` records the queue timestamp immediately before
        # submitting to its executor.  Persist that authoritative value from
        # ``ReplyTaskResult`` instead of writing a slightly later UI timestamp
        # here: ``COALESCE`` intentionally makes trace fields immutable, so the
        # UI placeholder used to win and could make queue -> worker appear
        # negative even though execution order was correct.
        self._analysis_event_by_job[job_id] = event.id
        self._reply_context_by_job[job_id] = deepcopy(context)
        self._latest_reply_job_id = job_id
        if context.current_message_view is not None:
            view = context.current_message_view
            self._visible_reply_attempt_key = "|".join((
                digest, view.target_instance_id, view.session_id,
                str(context.runtime.window.hwnd), str(context.runtime.window_generation),
                str(context.runtime.account_identity_epoch), context.runtime.account_identity_runtime_key,
            ))
        self._suggestion_context_hash = None
        self._render_reply_suggestions(
            ReplyGenerationResult(False, f"reply_pending: job={job_id}", [], self._config.ai.provider),
            context=context,
        )
        try:
            # Coverage analysis is local and durable, but deliberately starts
            # only after the user-facing reply has been submitted.  It records
            # missing knowledge; it never promotes customer chat to knowledge.
            self._services.knowledge_gap_analyzer.analyze(
                source_message=source_message,
                knowledge=context.knowledge_context,
            )
        except Exception as exc:
            self.append_log(
                f"knowledge_gap_analysis_failed: event={event.id} error={type(exc).__name__}",
                "warning",
            )
        self.append_log(
            f"auto_reply_triggered: event={event.id} message={event.message_id} messages={len(context.effective_messages)}"
        )

    def _on_reply_task_ready(self, task: ReplyTaskResult) -> None:
        if getattr(self, "_manual_generation_enabled", False):
            self._manual_generation_on_reply_ready(task)
            return
        task_context = self._reply_context_by_job.pop(task.job_id, None)
        analysis_event_id = self._analysis_event_by_job.pop(task.job_id, "")
        analysis_event = self._services.analysis_events.get(analysis_event_id) if analysis_event_id else None
        if task.job_id != self._latest_reply_job_id:
            self.append_log(f"reply_result_ignored: superseded job={task.job_id}")
            return
        live_context = self._build_reply_context()
        old_view = task_context.current_message_view if task_context is not None else None
        live_view = live_context.current_message_view
        same_owner = bool(task_context is not None
            and task.contact_id == (live_context.contact.id if live_context.contact else None)
            and task.hwnd == live_context.runtime.window.hwnd
            and task_context.runtime.window.process_id == live_context.runtime.window.process_id
            and task_context.runtime.window_generation == live_context.runtime.window_generation
            and task_context.runtime.account_identity_epoch == live_context.runtime.account_identity_epoch
            and task_context.runtime.account_identity_runtime_key == live_context.runtime.account_identity_runtime_key
            and ((old_view is None and live_view is None) or (
                old_view is not None and live_view is not None
                and (old_view.account_scope_id, old_view.contact_id, old_view.conversation_id,
                     old_view.target_instance_id, old_view.session_id)
                    == (live_view.account_scope_id, live_view.contact_id, live_view.conversation_id,
                        live_view.target_instance_id, live_view.session_id))))
        if not same_owner or task.context_digest != _reply_context_hash(live_context):
            self._latest_reply_job_id = None
            if analysis_event is not None:
                self._services.analysis_events.update_status(
                    analysis_event.message_id, "blocked", task.context_digest,
                    failure_reason="superseded_or_invalid_current_source",
                )
            self.append_log(f"reply_result_ignored: no_longer_owns_current_panel job={task.job_id}")
            return
        # Use the frozen worker's effective deadline, not a later config or
        # the original UI context that could have a longer recognition limit.
        completed_deadline = getattr(task, "current_work_deadline_monotonic", None)
        from time import monotonic as current_work_clock
        if completed_deadline is not None and current_work_clock() >= completed_deadline:
            self._latest_reply_job_id = None
            if analysis_event is not None:
                self._services.analysis_events.update_status(
                    analysis_event.message_id, "blocked", task.context_digest,
                    failure_reason="current_work_timeout",
                )
            self._suggestion_context_hash = None
            self._render_reply_suggestions(ReplyGenerationResult(
                False, "blocked:current_work_timeout：本次建议已超过识别后的时限，未展示迟到结果", [], self._config.ai.provider,
            ), context=task_context)
            return
        validator = getattr(getattr(self._services, "reply_generator", None), "identity_lease_validator", None)
        try:
            original_lease_valid = bool(task_context is not None and callable(validator) and validator(task_context))
        except Exception:
            original_lease_valid = False
        if (
            task.context_digest != _reply_context_hash(live_context)
            or current_context_source_error(live_context)
            or not original_lease_valid
            or task.contact_id != (live_context.contact.id if live_context.contact else None)
            or task.hwnd != live_context.runtime.window.hwnd
            or task.window_title != live_context.runtime.window.title
        ):
            if analysis_event is not None:
                self._services.analysis_events.update_status(
                    analysis_event.message_id, "blocked", task.context_digest,
                    failure_reason="superseded_or_invalid_current_source",
                )
            self._latest_reply_job_id = None
            same_turn = bool(task.context_digest == _reply_context_hash(live_context)
                             and task.contact_id == (live_context.contact.id if live_context.contact else None)
                             and task.hwnd == live_context.runtime.window.hwnd)
            pending = getattr(self._services.reply_generator, "has_pending_response", None)
            waiting = bool(same_turn and task_context is not None and callable(pending)
                           and pending(task_context, self._config))
            recovery = getattr(self._services.reply_generator, "recover_cached_only", None)
            recovered = recovery(live_context, self._config) if waiting and callable(recovery) else None
            display_recovery = getattr(self._services.reply_generator, "recover_pending_for_display", None)
            if recovered is None and waiting and callable(display_recovery):
                recovered = display_recovery(task_context, live_context, self._config)
            if recovered is not None:
                recovered_context = (task_context if recovered.status == "ai_cached:display_only_revalidated"
                                     else live_context)
                self._render_reply_suggestions(recovered, context=recovered_context)
                self._suggestion_context_hash = task.context_digest
                if analysis_event is not None:
                    self._services.analysis_events.update_status(
                        analysis_event.message_id, "completed", task.context_digest, failure_reason="",
                        result_json=json.dumps(asdict(recovered), ensure_ascii=False, separators=(",", ":")),
                    )
                self.append_log(f"reply_result_locally_revalidated: job={task.job_id}")
                return
            self._render_reply_empty_state(
                "回复已返回，待当前画面重新核验后本地恢复，不会重复请求 AI" if waiting else
                "当前来源待重新核验，暂不展示旧结果" if same_turn else
                "当前聊天内容已更新，旧结果已撤下，等待最新采集"
            )
            self.append_log(f"reply_result_ignored: current_context_changed job={task.job_id}")
            return
        if analysis_event is not None:
            trace = task.result.performance_trace if isinstance(task.result.performance_trace, dict) else {}
            self._services.reply_pipeline_metrics.mark_many(
                analysis_event.id,
                {
                    "reply_queued_at": task.queued_at,
                    "reply_worker_started_at": task.worker_started_at,
                    "provider_request_started_at": trace.get("provider_request_started_at"),
                    "provider_first_response_at": trace.get("provider_first_response_at"),
                    "provider_response_completed_at": trace.get("provider_response_completed_at"),
                    "reply_parse_completed_at": trace.get("reply_parse_completed_at"),
                },
            )
            self._services.reply_pipeline_metrics.attach_profiles(
                analysis_event.id,
                prompt_profile=trace.get("prompt_profile") if isinstance(trace.get("prompt_profile"), dict) else {},
                provider_profile=trace.get("provider_profile") if isinstance(trace.get("provider_profile"), dict) else {},
            )
            self._services.reply_pipeline_metrics.mark(analysis_event.id, "reply_completed_at")
            result_json = json.dumps(asdict(task.result), ensure_ascii=False, separators=(",", ":"))
            self._services.analysis_events.update_status(
                analysis_event.message_id,
                "completed" if task.result.allowed and task.result.suggestions else "blocked",
                task.context_digest,
                failure_reason="" if task.result.allowed else task.result.status,
                result_json=result_json,
            )
            self._services.reply_pipeline_metrics.mark(analysis_event.id, "result_persisted_at")
            # Treat the durable result as the UI hand-off boundary.  This keeps
            # the persisted T17 -> T18 -> T19 sequence monotonic and guarantees
            # that a crash after the worker signal can still be recovered from
            # SQLite before any contact-specific result is rendered.
            self._services.reply_pipeline_metrics.mark(analysis_event.id, "ui_event_received_at")
            if not self._analysis_event_matches_active_context(analysis_event):
                self.append_log(
                    f"reply_suggestions_saved_for_inactive_context: event={analysis_event.id} contact={analysis_event.contact_id}"
                )
                QTimer.singleShot(0, self._auto_generate_reply_after_ocr)
                self._refresh_overview_data()
                return
            if self._load_latest_displayable_ai_result(render_context=task_context):
                self.append_log(f"reply_suggestions_ready_from_event: event={analysis_event.id}")
                self.statusBar().showMessage("回复建议已更新", 2500)
                self._refresh_generation_log_text()
                self._refresh_overview_data()
                QTimer.singleShot(0, self._auto_generate_reply_after_ocr)
                return
        current = self._build_reply_context()
        current_contact_id = current.contact.id if current.contact else None
        if (
            task.contact_id != current_contact_id
            or task.hwnd != current.runtime.window.hwnd
            or task.window_title != current.runtime.window.title
        ):
            self.append_log(
                (
                    "reply_suggestions_stale: "
                    f"job={task.job_id}, task_contact={task.contact_id or '-'}, "
                    f"current_contact={current_contact_id or '-'}, task_hwnd={task.hwnd or '-'}, "
                    f"current_hwnd={current.runtime.window.hwnd or '-'}, "
                    f"task_title={task.window_title or '-'}, current_title={current.runtime.window.title or '-'}"
                ),
                "warning",
            )
            self._reply_retry_pending = True
            self.statusBar().showMessage("上下文已变化，已忽略旧结果", 2500)
            self._refresh_overview_data()
            return
        current_digest = _reply_context_hash(current)
        if task.context_digest and task.context_digest != current_digest:
            self.append_log(
                f"reply_suggestions_context_changed: job={task.job_id} task={task.context_digest[:12]} current={current_digest[:12]}",
                "warning",
            )
            self._reply_retry_pending = True
            self.statusBar().showMessage("聊天内容已更新，正在等待新一轮回复建议", 2500)
            self._refresh_overview_data()
            return
        self._render_reply_suggestions(task.result, context=task_context)
        if task.result.allowed and task.result.suggestions:
            self._suggestion_context_hash = current_digest
        self.append_log(f"reply_suggestions_ready: job={task.job_id}, allowed={task.result.allowed}, status={task.result.status}")
        self.statusBar().showMessage("回复建议已更新", 2500)
        self._refresh_generation_log_text()
        self._refresh_overview_data()
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(self._services.runtime.state))
        QTimer.singleShot(0, self._auto_generate_reply_after_ocr)

    def _on_reply_task_state_changed(self, state: ReplyTaskState) -> None:
        if (getattr(self, "_manual_generation_enabled", False) and state.terminal
                and state.result_available):
            # result_ready owns the immutable manual answer. Do not rebuild
            # live context just to paint an intermediate terminal placeholder.
            return
        if (getattr(self, "_manual_generation_enabled", False) and state.terminal
                and self._manual_generation_phase == "generating"
                and state.job_id == self._manual_generation_job_id and not state.result_available):
            self._finish_manual_generation(f"本次生成未完成：{state.status}，{state.reason}")
            return
        if not state.terminal:
            if state.job_id == self._latest_reply_job_id and state.status in {"waiting_rate_limit", "waiting_latest", "running"}:
                context = self._reply_context_by_job.get(state.job_id)
                if context is not None and state.context_digest == _reply_context_hash(self._build_reply_context()):
                    self._render_reply_suggestions(ReplyGenerationResult(
                        False, f"reply_pending:{state.status}: job={state.job_id}", [], self._config.ai.provider
                    ), context=context)
            return
        if state.status == "succeeded":
            return
        # A blocked provider result still emits result_ready immediately after
        # this state. Preserve its context/event for durable audit hand-off.
        has_result = state.result_available
        saved = (self._reply_context_by_job.get(state.job_id) if has_result
                 else self._reply_context_by_job.pop(state.job_id, None))
        event_id = (self._analysis_event_by_job.get(state.job_id, "") if has_result
                    else self._analysis_event_by_job.pop(state.job_id, ""))
        if event_id and not has_result:
            event = self._services.analysis_events.get(event_id)
            if event is not None and event.status in {"pending", "running"}:
                self._services.analysis_events.update_status(
                    event.message_id, "blocked", state.context_digest,
                    failure_reason=f"reply_{state.status}:{state.reason}",
                )
        if state.job_id != self._latest_reply_job_id or saved is None:
            return
        current = self._build_reply_context()
        if (
            state.context_digest != _reply_context_hash(current)
            or state.contact_id != (current.contact.id if current.contact else None)
            or state.hwnd != current.runtime.window.hwnd
            or state.window_generation != current.runtime.window_generation
            or state.account_identity_epoch != current.runtime.account_identity_epoch
            or state.account_identity_runtime_key != current.runtime.account_identity_runtime_key
        ):
            return
        labels = {"failed": "生成失败", "timed_out": "生成超时", "blocked": "暂不可生成", "cancelled": "任务已取消", "superseded": "已由最新消息替代"}
        self._suggestion_context_hash = None
        self._render_reply_suggestions(ReplyGenerationResult(
            False, f"blocked:{labels.get(state.status, state.status)}：{state.reason}", [], self._config.ai.provider
        ), context=current)
        self._reply_retry_pending = False

    def _on_reply_task_discarded(self, reason: str) -> None:
        self.append_log(f"reply_task_discarded: {reason}", "warning")
        self._refresh_overview_data()
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(self._services.runtime.state))

    def _on_reply_task_status_changed(self, status: str) -> None:
        self.append_log(f"reply_task_status: {status}")
        self._refresh_overview_data()
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(self._services.runtime.state))

    def _schedule_proactive_contact(self, ingestion=None) -> int | None:
        if getattr(self, "_manual_generation_enabled", False):
            return None
        if self._auto_generate_visible_reply():
            return None
        ingestion = ingestion or self._services.ingestion.last_result
        state = self._services.runtime.state
        contact = self._active_capture_contact()
        identity = self._services.current_identity.snapshot
        if (
            state.page.page_type != PageType.CHAT_DM
            or contact is None
            or ingestion is None
            or not ingestion.accepted
            or not ingestion.identity_stable
            or ingestion.contact is None
            or ingestion.contact.id != contact.id
            or ingestion.account_scope_id != contact.account_scope_id
            or not ingestion.conversation_id
            or not ingestion.target_instance_id
            or state.window.hwnd is None
            or not identity.active
            or identity.contact_id != contact.id
        ):
            return None
        stable_for = time.monotonic() - self._identity_safe_since_monotonic
        stability_lease_seconds = 0.75
        if self._identity_safe_since_monotonic <= 0 or stable_for < stability_lease_seconds:
            if not self._proactive_stability_retry_pending:
                self._proactive_stability_retry_pending = True
                delay_ms = max(1, int((stability_lease_seconds - max(0.0, stable_for)) * 1000) + 1)
                QTimer.singleShot(delay_ms, self._retry_proactive_after_identity_lease)
            return None
        self._proactive_stability_retry_pending = False
        latest_event = self._services.analysis_events.latest_for_conversation_binding(
            ingestion.account_scope_id,
            contact.id,
            ingestion.conversation_id,
        )
        if latest_event is not None and self._analysis_event_has_reply_priority(
            latest_event, ingestion.conversation_id
        ):
            return None
        if self._services.reply_tasks.is_running or self._pending_auto_event_ids:
            return None
        strategy = self._services.strategies.get(contact.strategy_id) if contact.strategy_id else None
        if (
            strategy is not None
            and strategy.requires_manual_reply
            and self._config.privacy.manual_protection_blocks_replies
        ):
            self._render_reply_suggestions(
                ReplyGenerationResult(
                    False,
                    "proactive_notice:当前分组启用了手动回复保护，不生成主动联系文案",
                    [],
                    "Local Reason Engine",
                ),
                mode="proactive",
                evidence_override="本地安全门禁 · 未调用云端 AI",
                title_override="今天暂不建议主动联系",
            )
            return None
        proactive_cloud_authorization = self._proactive_cloud_gate.issue_scope(
            contact=contact,
            page_type=state.page.page_type.value,
            identity_stable=bool(ingestion.identity_stable),
            account_scope_id=ingestion.account_scope_id,
            active_contact_id=contact.id,
            reply_pending=False,
        )
        request = ProactiveContactRequest(
            account_scope_id=ingestion.account_scope_id,
            contact_id=contact.id,
            conversation_id=ingestion.conversation_id,
            target_instance_id=ingestion.target_instance_id,
            hwnd=int(state.window.hwnd),
            window_title=state.window.title,
            capture_generation=state.capture_generation,
            local_date=datetime.now().astimezone().date().isoformat(),
            proactive_cloud_authorization=proactive_cloud_authorization,
            current_observation_id=ingestion.observation_id,
            current_session_id=ingestion.session_id or "",
            source_job_id=int(getattr(self._services.ingestion.last_evidence, "job_id", 0) or 0),
        )
        if self._proactive_last_request_key == request.binding_key:
            return None
        if self._services.proactive_tasks.is_running:
            self._proactive_retry_pending = self._services.proactive_tasks.running_key != request.binding_key
            return None
        job_id = self._services.proactive_tasks.submit(request, self._config)
        if job_id is None:
            self._proactive_retry_pending = False
            self._render_reply_empty_state("本次主动联系任务未启动，可稍后手动重新生成")
            return None
        self._latest_proactive_job_id = job_id
        self._proactive_activation_revisions[job_id] = (
            request.account_scope_id,
            request.contact_id,
            self._services.activation_snapshots.revision(
                request.account_scope_id,
                request.contact_id,
            ),
        )
        if len(self._proactive_activation_revisions) > 32:
            oldest = min(self._proactive_activation_revisions)
            self._proactive_activation_revisions.pop(oldest, None)
        self._proactive_retry_pending = False
        self._proactive_last_request_key = request.binding_key
        if not self._has_active_suggestion():
            self._render_reply_suggestions(
                ReplyGenerationResult(
                    False,
                    f"proactive_notice:正在本地判断是否适合主动联系（job={job_id}）",
                    [],
                    "Local Reason Engine",
                ),
                mode="proactive",
                evidence_override="先做本地 A/B 资格与联系理由判断；通过后才可能调用云端 AI",
                title_override="智能建议",
            )
        self.append_log(f"proactive_contact_submitted: job={job_id}")
        return job_id

    def _retry_proactive_after_identity_lease(self) -> None:
        self._proactive_stability_retry_pending = False
        self._schedule_proactive_contact()

    def _suppress_activation_for_reply_priority(self, contact_id: str) -> None:
        active = self._active_capture_contact()
        if active is None or active.id != contact_id:
            return
        self._services.proactive_tasks.invalidate("true_new_customer_message")
        self._proactive_last_request_key = ""
        self._proactive_retry_pending = False
        self._suggestion_result = None
        self._suggestion_context_hash = None
        self._displayed_analysis_event_id = None
        self._render_reply_empty_state("已收到新的客户消息，正在生成 Reply")

    def _on_activation_snapshot_invalidated(self, event) -> None:
        """Remove only the exact account/contact Activation currently on screen."""

        active = self._active_capture_contact()
        outcome = self._proactive_outcome
        if (
            active is None
            or active.account_scope_id != event.account_scope_id
            or active.id != event.contact_id
            or self._suggestion_mode != "proactive"
            or outcome is None
            or outcome.request.account_scope_id != event.account_scope_id
            or outcome.request.contact_id != event.contact_id
        ):
            return
        self._services.proactive_tasks.invalidate(f"activation_invalidated:{event.reason}")
        self._proactive_last_request_key = ""
        self._proactive_retry_pending = False
        self._suggestion_result = None
        self._suggestion_context_hash = None
        self._displayed_analysis_event_id = None
        self._render_reply_empty_state("客户上下文已变化，主动联系建议已失效")
        self.append_log(
            "activation_snapshot_invalidated: "
            f"contact={event.contact_id} reason={event.reason} revision={event.revision}"
        )

    def _invalidate_proactive_context(self, reason: str, contact_id: str | None = None) -> None:
        """Expire proactive work after a bound local data mutation.

        Mutations for a contact merely selected in the manager must not disturb
        the currently captured chat.  For the active chat, stale worker results
        are rejected and a fresh local decision is queued without a busy loop.
        """
        active = self._active_capture_contact()
        if active is None or (contact_id is not None and active.id != contact_id):
            return
        self._services.proactive_tasks.invalidate(reason)
        self._proactive_last_request_key = ""
        self._proactive_retry_pending = self._services.proactive_tasks.is_running
        self._suggestion_result = None
        self._suggestion_context_hash = None
        self._displayed_analysis_event_id = None
        self._render_reply_empty_state("客户上下文已变化，正在重新评估智能建议")
        QTimer.singleShot(0, self._schedule_proactive_contact)

    def _on_proactive_task_ready(self, task: ProactiveTaskResult) -> None:
        outcome = task.outcome
        request = outcome.request
        if self._auto_generate_visible_reply():
            self._proactive_activation_revisions.pop(task.job_id, None)
            self.append_log(f"proactive_result_ignored: current_visible_reply_priority job={task.job_id}")
            return
        if task.job_id != self._latest_proactive_job_id:
            self._proactive_activation_revisions.pop(task.job_id, None)
            return
        revision_guard = self._proactive_activation_revisions.pop(task.job_id, None)
        if revision_guard is not None and not self._services.activation_snapshots.revision_is_current(
            revision_guard[0], revision_guard[1], revision_guard[2]
        ):
            self.append_log(
                f"proactive_contact_stale_revision: job={task.job_id} contact={request.contact_id}",
                "warning",
            )
            return
        ingestion = self._services.ingestion.last_result
        contact = self._active_capture_contact()
        state = self._services.runtime.state
        if (
            contact is None
            or ingestion is None
            or state.page.page_type != PageType.CHAT_DM
            or state.window.hwnd != request.hwnd
            or contact.id != request.contact_id
            or contact.account_scope_id != request.account_scope_id
            or ingestion.conversation_id != request.conversation_id
            or ingestion.target_instance_id != request.target_instance_id
            or (request.current_session_id and ingestion.session_id != request.current_session_id)
            or state.capture_generation != request.capture_generation
        ):
            self.append_log(
                f"proactive_contact_stale: job={task.job_id} context={outcome.context_hash[:12]}",
                "warning",
            )
            return
        current_context = self._build_reply_context()
        if current_context_source_error(current_context):
            self._render_reply_empty_state("当前聊天来源尚未确认，已撤下旧主动联系结果")
            return
        view = current_context.current_message_view
        if view is not None and outcome.decision.context_payload.get("current_view_revision") != view.content_revision:
            # A later pending job owns the panel; an old result may not clear it.
            if task.job_id == self._latest_proactive_job_id:
                self._render_reply_empty_state("当前聊天内容已变化，已撤下旧主动联系结果")
                self._proactive_last_request_key = ""
                QTimer.singleShot(0, self._schedule_proactive_contact)
            return
        latest_event = self._services.analysis_events.latest_for_conversation_binding(
            request.account_scope_id,
            request.contact_id,
            request.conversation_id,
        )
        if latest_event is not None and self._analysis_event_has_reply_priority(
            latest_event, request.conversation_id
        ):
            self.append_log(f"proactive_contact_suppressed_by_reply: job={task.job_id}")
            self._load_latest_displayable_ai_result(ingestion)
            return
        if task.operation != "regenerate" and self._manual_proactive_edit_is_current():
            self.append_log(f"proactive_background_preserved_manual_edit: job={task.job_id}")
            return
        if task.operation == "regenerate":
            self._manual_proactive_binding = None
            self._proactive_feedback_status = {}
            self._proactive_rejected_feedback = {}
            self._proactive_regeneration_used = True
        expected_revision = revision_guard[2] if revision_guard is not None else None
        self._services.activation_snapshots.cache_on_demand(
            outcome,
            self._config,
            expected_revision=expected_revision,
        )
        if expected_revision is not None and not self._services.activation_snapshots.revision_is_current(
            request.account_scope_id,
            request.contact_id,
            expected_revision,
        ):
            self.append_log(
                f"proactive_contact_cache_stale: job={task.job_id} contact={request.contact_id}",
                "warning",
            )
            return
        self._render_proactive_outcome(outcome)
        self._proactive_last_request_key = request.binding_key
        self._suggestion_context_hash = outcome.context_hash
        self._refresh_generation_log_text()
        self._refresh_overview_data()

    def _render_proactive_outcome(
        self,
        outcome: ProactiveContactOutcome,
        *,
        feedback_preloaded: bool = False,
    ) -> None:
        decision = outcome.decision
        if outcome.result is not None and outcome.result.allowed and len(outcome.result.suggestions) == 1:
            pending_status = {
                index: value
                for index, value in self._proactive_feedback_status.items()
                if value.startswith("正在按反馈重新生成")
            }
            self._proactive_outcome = outcome
            if feedback_preloaded:
                self._proactive_feedback_status = dict(pending_status)
                self._proactive_rejected_feedback = {}
                self._proactive_regeneration_used = bool(pending_status)
            else:
                feedback_view = self._services.proactive_feedback.view_state(outcome)
                self._proactive_feedback_status = dict(feedback_view.statuses)
                self._proactive_feedback_status.update(pending_status)
                self._proactive_rejected_feedback = dict(feedback_view.rejected)
                self._proactive_regeneration_used = bool(
                    feedback_view.regeneration_used or pending_status
                )
        else:
            self._proactive_outcome = None
            self._proactive_feedback_status = {}
            self._proactive_rejected_feedback = {}
            self._proactive_regeneration_used = False
        cache_label = "同日缓存" if outcome.cache_hit else "新评估"
        stage_data = decision.context_payload.get("current_stage") or {}
        need_data = decision.context_payload.get("current_need") or {}
        recency_data = decision.context_payload.get("contact_recency") or {}
        stage_label = str(stage_data.get("stage_label") or "当前阶段待确认")
        active_crop = str(stage_data.get("crop") or "待确认")
        crop_confidence = str(stage_data.get("active_crop_confidence") or "low")
        season_key = str(stage_data.get("season_key") or "当季待确认")
        stage_status = str(stage_data.get("status") or "unknown")
        stage_marker = {
            "confirmed": "已确认",
            "inferred": "推算，待确认",
            "unknown": "待确认",
        }.get(stage_status, "待确认")
        possible_need = str(need_data.get("top_need") or "先确认当前实际情况")
        talking_point = str(need_data.get("best_talking_point") or "先确认当前关键变量")
        days_since = recency_data.get("days_since_last_contact")
        recency_text = "无可用联系时间" if days_since is None else f"距上次联系 {days_since} 天"
        evidence = (
            f"当前作物：{active_crop}｜置信度 {crop_confidence}｜{season_key}\n"
            f"当前阶段：{stage_label}｜{stage_marker}\n"
            f"当前可能需求：{possible_need}\n"
            f"联系切入点：{talking_point}\n"
            f"{recency_text} · {cache_label}"
        )
        style_data = decision.context_payload.get("conversation_style") or {}
        salutation = str(style_data.get("preferred_salutation") or "").strip()
        salutation_confidence = str(style_data.get("salutation_confidence") or "low")
        evidence += (
            f"\n客户称呼：{salutation}｜置信度 {salutation_confidence}"
            if salutation and salutation_confidence in {"medium", "high"}
            else "\n客户称呼：无可靠证据，本次不加称呼"
        )
        if outcome.request.proactive_cloud_authorization:
            evidence += " · 主动联系AI：已授权（不扩展联系人其它云端权限）"
        if outcome.display_mode == "ineligible":
            self._render_reply_suggestions(
                ReplyGenerationResult(
                    False,
                    f"proactive_notice:{decision.do_not_contact_reason or decision.reason_summary}",
                    [],
                    "Local Reason Engine",
                ),
                mode="proactive",
                evidence_override=evidence + " · 未调用云端 AI",
                title_override="智能建议",
            )
            return
        if outcome.display_mode == "do_not_contact":
            next_text = decision.suggested_next_contact_at or "等待客户状态或明确跟进条件变化"
            notice = f"原因：{decision.do_not_contact_reason}；下一条件：{next_text}"
            self._render_reply_suggestions(
                ReplyGenerationResult(False, f"proactive_notice:{notice}", [], "Local Reason Engine"),
                mode="proactive",
                evidence_override=evidence + " · 未调用云端 AI",
                title_override="今天暂不建议主动联系",
            )
            return
        if outcome.display_mode == "reply_priority":
            if not self._auto_generate_visible_reply():
                self._auto_generate_reply_after_ocr()
                if not self._services.reply_tasks.is_running:
                    self._render_reply_empty_state("当前消息来源尚未确认，请重新打开聊天窗口")
            return
        if outcome.result is None:
            self._render_reply_suggestions(
                ReplyGenerationResult(False, "proactive_notice:主动联系评估未生成文案", [], "Local Reason Engine"),
                mode="proactive",
                evidence_override=evidence,
            )
            return
        self._render_reply_suggestions(
            outcome.result,
            mode="proactive",
            evidence_override=evidence,
            title_override="主动联系建议",
        )
        self.statusBar().showMessage("主动联系建议已更新；只复制，不会自动发送", 3000)
        message = (
            f"proactive_contact_ready: reason={decision.reason_type} cache={int(outcome.cache_hit)} "
            f"context={outcome.context_hash[:12]}"
        )
        if feedback_preloaded:
            QTimer.singleShot(0, lambda: self.append_log(message))
        else:
            self.append_log(message)

    def _on_proactive_task_discarded(self, reason: str) -> None:
        self.append_log(f"proactive_task_discarded: {reason}", "warning")
        if reason.startswith(("proactive_stale_result", "proactive_busy")):
            self._proactive_retry_pending = True
            QTimer.singleShot(150, self._retry_proactive_contact)

    def _on_proactive_task_state_changed(self, state: ProactiveTaskState) -> None:
        if not state.terminal:
            return
        if self._proactive_retry_pending:
            QTimer.singleShot(150, self._retry_proactive_contact)
        if state.result_available:
            return
        self._proactive_activation_revisions.pop(state.job_id, None)
        if state.job_id != self._latest_proactive_job_id:
            return
        request = state.request
        ingestion = self._services.ingestion.last_result
        runtime = self._services.runtime.state
        identity = self._services.current_identity.snapshot
        if (
            ingestion is None or not ingestion.accepted or not ingestion.identity_stable
            or not identity.active or identity.contact_id != request.contact_id
            or identity.account_scope_id != request.account_scope_id
            or identity.target_instance_id != request.target_instance_id
            or ingestion.conversation_id != request.conversation_id
            or ingestion.session_id != request.current_session_id
            or runtime.window.hwnd != request.hwnd
            or self._services.reply_tasks.is_running
            or (self._suggestion_mode == "reply" and self._has_active_suggestion())
        ):
            return
        if runtime.capture_generation != request.capture_generation:
            self._render_reply_empty_state("窗口状态已变化，旧主动联系任务已结束，等待重新确认")
            return
        labels = {"failed": "主动联系判断失败", "timed_out": "主动联系判断超时", "blocked": "本次主动联系被阻断", "cancelled": "主动联系任务已取消", "superseded": "主动联系上下文已更新"}
        self._render_reply_suggestions(ReplyGenerationResult(
            False, f"proactive_notice:{labels.get(state.status, state.status)}：{state.reason}", [], "Local Reason Engine"
        ), mode="proactive", evidence_override="本次任务已结束；可手动重新生成")

    def _retry_proactive_contact(self) -> None:
        if not self._proactive_retry_pending or self._services.proactive_tasks.is_running:
            return
        self._proactive_retry_pending = False
        self._schedule_proactive_contact()

    def _on_proactive_task_status_changed(self, status: str) -> None:
        self.append_log(f"proactive_task_status: {status}")

    def _resume_reply_flow_if_unblocked(self, reason: str) -> None:
        if getattr(self, "_manual_generation_enabled", False):
            self._refresh_overview_data()
            return
        previous = self._suggestion_result
        was_blocked = previous is not None and not previous.allowed and not previous.status.startswith("reply_pending")
        if self._services.reply_tasks.is_running:
            self.append_log(f"reply_flow_resume_skipped: running reason={reason}")
            self._refresh_overview_data()
            return
        context = self._build_reply_context()
        steps = build_status_chain(
            runtime=context.runtime,
            contact=context.contact,
            strategy=context.strategy,
            config=self._config,
            reply_running=False,
            provider_health=self._services.reply_generator.provider_health_summary(),
            has_active_suggestion=self._has_active_suggestion(),
            current_work_cloud_authorized=cloud_reply_authorized(context),
        )
        ai_step = next((step for step in steps if step.stage == "AI"), steps[-1])
        if ai_step.state == "就绪":
            if was_blocked:
                self.append_log(f"reply_flow_resuming: reason={reason}")
                QTimer.singleShot(0, self._refresh_reply_suggestions)
            else:
                self.append_log(f"reply_flow_ready: reason={reason}")
                self._refresh_overview_data()
            return
        if was_blocked:
            self._render_reply_suggestions(
                ReplyGenerationResult(False, f"blocked:{ai_step.reason}", [], self._config.ai.provider)
            )
        self.append_log(f"reply_flow_still_blocked: reason={reason} block={ai_step.reason}")
        self._refresh_overview_data()

    def _requires_cloud_prompt_review(self, context: ReplyContext) -> bool:
        return (
            self._config.privacy.require_cloud_prompt_review
            and self._config.ai.provider in {"OpenAI", "OpenAI Compatible"}
            and bool(self._config.ai.api_key)
            and context.contact is not None
            and cloud_reply_authorized(context)
        )

    def _confirm_cloud_prompt(self, context: ReplyContext) -> bool:
        preview = build_prompt_preview(context, self._config)
        dialog = QDialog(self)
        dialog.setWindowTitle("确认云端 AI 请求")
        dialog.setMinimumSize(720, 560)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        summary = QLabel(self._format_prompt_preview_summary(preview))
        summary.setWordWrap(True)
        summary.setObjectName("Muted")
        layout.addWidget(summary)
        body = QTextEdit()
        body.setReadOnly(True)
        body.setText(preview.combined_text)
        layout.addWidget(body, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认发送给 AI")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def _format_prompt_preview_summary(self, preview: PromptPreview) -> str:
        return (
            f"Provider：{preview.provider} · 模型：{preview.model}\n"
            f"上下文：消息 {preview.message_count} 条，记忆 {preview.memory_count} 条\n"
            f"照片：{preview.photo_count} 张（仅聊天图片区域；图片内文字未自动脱敏）\n"
            f"脱敏：{preview.redaction_summary}"
        )

    def _suggestion_row(self, suggestion: ReplySuggestion, suggestion_index: int) -> QWidget:
        row = QFrame()
        row.setObjectName("MetricCard")
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        header = QHBoxLayout()
        tag = QLabel(suggestion.label)
        tag.setObjectName("Badge")
        tag.setMinimumWidth(58)
        tag.setMaximumWidth(76)
        tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tag.setText(_clip_debug_text(suggestion.label, 8))
        tag.setToolTip(suggestion.label)
        body = QLabel(suggestion.text)
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setWordWrap(True)
        body.setMinimumHeight(34)
        body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        body.setToolTip(suggestion.text)
        risk_label = QLabel(f"risk: {suggestion.risk}")
        risk_label.setObjectName("TinyMuted")
        risk_label.setToolTip(suggestion.rationale)
        # Why this suggestion is phrased the way it is must be readable at a glance,
        # not hidden in a tooltip: the label shows "low · 目标：稳妥" and the tooltip
        # keeps the full rationale.
        risk_label.setText(_clip_debug_text(f"{suggestion.risk} · {suggestion.rationale}", 40))
        copy = QPushButton("复制")
        copy.setObjectName("PrimaryButton")
        copy.setFixedSize(48, 24)
        copy.setToolTip("复制完整回复")
        copy.setEnabled((self._suggestion_mode == "reply" and self._manual_result_retention_is_current())
                        or self._suggestion_binding_is_current(self._suggestion_action_binding))
        if not copy.isEnabled():
            copy.setToolTip("正在确认新客户，暂时禁止复制旧客户建议")
        copy.clicked.connect(
            lambda _checked=False, item=suggestion, index=suggestion_index, binding=self._suggestion_action_binding: self._copy_reply_suggestion(
                item, index, binding=binding
            )
        )
        header.addWidget(tag)
        header.addWidget(risk_label)
        header.addStretch(1)
        proactive_actions: QHBoxLayout | None = None
        if self._suggestion_mode == "proactive":
            proactive_actions = QHBoxLayout()
            proactive_actions.setSpacing(6)
            proactive_actions.addStretch(1)
            proactive_actions.addWidget(copy)
            modify = QPushButton("修改")
            modify.setFixedSize(48, 24)
            modify.setToolTip("修改这条建议并保存为本次采用文本")
            modify.setEnabled(not self._identity_suggestions_suspended)
            modify.clicked.connect(
                lambda _checked=False, index=suggestion_index: self._modify_proactive_suggestion(index)
            )
            reject = QPushButton("不满意")
            reject.setFixedSize(58, 24)
            reject.setToolTip("记录本地话术反馈；不会自动调用 AI")
            reject.setEnabled(not self._identity_suggestions_suspended)
            reject.clicked.connect(
                lambda _checked=False, index=suggestion_index: self._reject_proactive_suggestion(index)
            )
            proactive_actions.addWidget(modify)
            proactive_actions.addWidget(reject)
            if suggestion_index in self._proactive_rejected_feedback:
                regenerate = QPushButton("重新生成")
                regenerate.setFixedHeight(24)
                regenerate.setEnabled(
                    not self._proactive_regeneration_used
                    and not self._identity_suggestions_suspended
                )
                regenerate.setToolTip(
                    "按本次不满意原因重新生成；同一客户、上下文和日期最多一次"
                )
                regenerate.clicked.connect(
                    lambda _checked=False, index=suggestion_index: self._regenerate_proactive_suggestions(index)
                )
                proactive_actions.addWidget(regenerate)
        else:
            header.addWidget(copy)
        layout.addLayout(header)
        layout.addWidget(body)
        if self._suggestion_mode == "reply" and getattr(self._services, "writing_feedback", None) is not None:
            from whochat.ui.writing_feedback import add_edit_action
            add_edit_action(self, row, layout, suggestion_index)
        if proactive_actions is not None:
            layout.addLayout(proactive_actions)
        feedback_status = self._proactive_feedback_status.get(suggestion_index, "")
        if self._suggestion_mode == "proactive" and feedback_status:
            status = QLabel(feedback_status)
            status.setObjectName("TinyMuted")
            status.setWordWrap(True)
            layout.addWidget(status)
        return row

    def _bind_suggestion_actions(self, mode: str, context=None) -> SuggestionActionBinding | None:
        snapshot = getattr(self, "_frozen_screenshot_run", lambda: None)()
        if snapshot is not None:
            return snapshot.binding(mode, context)
        identity = self._services.current_identity.snapshot
        if not identity.active:
            return None
        if mode == "proactive":
            outcome = self._proactive_outcome
            if outcome is None:
                return None
            request = outcome.request
            if (
                request.account_scope_id != identity.account_scope_id
                or request.contact_id != identity.contact_id
                or request.target_instance_id != identity.target_instance_id
                or (request.hwnd and request.hwnd != identity.hwnd)
            ):
                return None
            context_key = outcome.context_hash + "|" + request.binding_key
            local_date = request.local_date
        else:
            if context is not None:
                contact = context.contact
                if contact is None or contact.id != identity.contact_id or contact.account_scope_id != identity.account_scope_id:
                    return None
                source = context.runtime
                if (
                    source.window.hwnd != identity.hwnd
                    or source.window.process_id != identity.process_id
                    or source.window_generation != identity.window_generation
                    or source.account_identity_epoch != identity.account_identity_epoch
                ):
                    return None
            context_key = _reply_context_hash(context) if context is not None else ""
            local_date = ""
        return SuggestionActionBinding(identity.conversation_token, mode, context_key, local_date)

    def _suggestion_binding_is_current(self, binding: SuggestionActionBinding | None) -> bool:
        snapshot = getattr(self, "_frozen_screenshot_run", lambda: None)()
        if snapshot is not None:
            return snapshot.binding_current(binding)
        if binding is None or binding != self._suggestion_action_binding or self._identity_suggestions_suspended:
            return False
        identity = self._services.current_identity.snapshot
        state = self._services.runtime.state
        if (
            not identity.active
            or self._services.current_identity.current_contact() is None
            or identity.conversation_token != binding.identity_token
            or state.window.state != WindowState.VISIBLE
            or identity.hwnd != state.window.hwnd
            or identity.process_id != state.window.process_id
            or identity.window_generation != state.window_generation
            or identity.account_identity_epoch != state.account_identity_epoch
            or binding.mode != self._suggestion_mode
        ):
            return False
        lifecycle_gate = getattr(self._services.runtime, "evidence_is_current", None)
        if callable(lifecycle_gate) and not lifecycle_gate(SimpleNamespace(
            hwnd=identity.hwnd,
            process_id=identity.process_id,
            window_generation=identity.window_generation,
            capture_generation=state.capture_generation,
            account_identity_epoch=identity.account_identity_epoch,
            account_identity_runtime_key=state.account_identity_runtime_key,
            account_scope_id=identity.account_scope_id,
            target_instance_id=identity.target_instance_id,
        )):
            return False
        if binding.mode == "proactive":
            outcome = self._proactive_outcome
            return bool(
                outcome is not None
                and binding.local_date == datetime.now().astimezone().date().isoformat()
                and binding.context_key == outcome.context_hash + "|" + outcome.request.binding_key
            )
        saved = getattr(self, "_suggestion_reply_context", None)
        display = getattr(self._services.reply_generator, "display_lease_validator", None)
        if saved is not None and saved.current_message_view is not None and callable(display):
            profile = getattr(self._services.reply_generator, "display_profile_hash", None)
            if (not callable(profile) or profile(self._config) != getattr(self, "_suggestion_reply_profile", None)
                    or binding.context_key != _reply_context_hash(saved) or not display(saved)):
                return False
            current = self._build_reply_context()
            # This exception is DISPLAY ONLY. The unchanged old proof and new
            # context cannot receive a cloud grant through this path.
            error = current_context_source_error(current)
            matches = binding.context_key == _reply_context_hash(current)
            if not matches and saved.photos:
                from whochat.services.manual_suggestions import retained_photo_business_matches
                matches = retained_photo_business_matches(
                    saved, current, getattr(self, "_manual_generation_source_result", None),
                    getattr(getattr(self._services, "pipeline", None), "last_result", None),
                    binding.context_key,
                )
            return bool(error in {None, "current_view_screen_runtime_mismatch"}
                        and matches and display(saved))
        current = self._build_reply_context()
        if current_context_source_error(current) or binding.context_key != _reply_context_hash(current):
            return False
        if current.current_message_view is not None:
            validator = getattr(self._services.reply_generator, "identity_lease_validator", None)
            return bool(callable(validator) and validator(current))
        return True

    def _manual_copy_revalidation_begin(self, suggestion, index, binding) -> bool:
        """An explicit copy may collect NEW local evidence; never generate/cache-recover.

        The old proof remains immutable and invalid. No copy permission is
        granted here. Both fresh captures, DB/identity leases, full business
        equality and the normal final clipboard guards are required below.
        """
        if getattr(self, "_frozen_screenshot_run", lambda: None)() is not None:
            self.statusBar().showMessage('截图来源已过期，请重新截图生成；不会自动调用 AI。', 5000)
            return False
        saved = getattr(self, "_suggestion_reply_context", None)
        result = self._suggestion_result
        profile = getattr(self._services.reply_generator, "display_profile_hash", None)
        try:
            expired = (saved is not None and saved.current_message_view is not None
                       and datetime.fromisoformat(json.loads(saved.current_message_view.screen_evidence_json)["expires_at"]) <= datetime.now(timezone.utc))
        except (ValueError, TypeError):
            expired = False
        if (not expired or not self._manual_result_retention_is_current()
                or binding is None or binding != self._suggestion_action_binding or binding.mode != "reply"
                or result is None or not 0 <= index < len(result.suggestions) or result.suggestions[index] != suggestion
                or _reply_context_hash(saved) != binding.context_key
                or not callable(profile) or profile(self._config) != self._suggestion_reply_profile
                or not self.isVisible() or QApplication.activeModalWidget() is not None
                or self._services.reply_tasks.is_running):
            return False
        reader = getattr(self, "_local_preread", None)
        if reader is not None:
            reader.pause()
        if self._services.pipeline.is_running:
            return False
        self._manual_generation_request_id += 1
        now = time.monotonic()
        ticket = dict(result=result, suggestion=suggestion, index=index, text=suggestion.text,
            binding=binding, context=saved, profile=self._suggestion_reply_profile,
            source=self._manual_generation_source_result,
            avatar=getattr(self, "_manual_generation_avatar_anchor", None),
            owner=self._manual_generation_owner, request_id=self._manual_generation_request_id,
            deadline=now + 30.0)
        self._manual_copy_revalidation = ticket
        self._manual_generation_started = now
        self._manual_generation_review_deadline = ticket["deadline"]
        self._manual_generation_capture_ids = set()
        self._manual_generation_completed_ids = set()
        self._manual_generation_followup_pending = None
        self._manual_generation_focus_ticket = None
        self._manual_generation_title_hash = ""
        self._manual_generation_body_hash = ""
        self._manual_generation_source_kind = ""
        self._manual_generation_last_blocker = ""
        self._manual_generation_source_result = None
        self._manual_generation_avatar_anchor = None
        self._manual_generation_job_id = None
        self._manual_generation_allow_submit = False
        self._manual_generation_terminal = False
        self._manual_generation_response_context = None
        self._services.pipeline.reset_snapshot_hash_cache()
        self._set_manual_generation_phase("verifying")
        self._manual_copy_feedback("copy_refresh_pending", pending=True)
        self._manual_generation_tick()
        return getattr(self, "_manual_copy_revalidation", None) is ticket

    def _manual_copy_revalidation_current(self, ticket) -> bool:
        try:
            identity = self._services.current_identity.snapshot
            return bool(getattr(self, "_manual_copy_revalidation", None) is ticket
                and self._manual_generation_phase == "verifying"
                and self._manual_generation_request_id == ticket["request_id"]
                and self._manual_window_owner(self._services.runtime.state) == ticket["owner"]
                and self._suggestion_result is ticket["result"]
                and self._suggestion_action_binding == ticket["binding"]
                and self._suggestion_reply_context is ticket["context"]
                and self._suggestion_mode == "reply" and identity.active
                and not self._identity_suggestions_suspended
                and identity.conversation_token == ticket["binding"].identity_token
                and self._services.current_identity.current_contact() is not None
                and self._services.reply_generator.display_profile_hash(self._config) == ticket["profile"]
                and self._suggestion_result.suggestions[ticket["index"]] == ticket["suggestion"]
                and ticket["suggestion"].text == ticket["text"]
                and self.isVisible() and QApplication.activeModalWidget() is None
                and time.monotonic() < ticket["deadline"])
        except (AttributeError, IndexError, TypeError):
            return False

    def _manual_copy_revalidation_end(self, reason) -> None:
        ticket = getattr(self, "_manual_copy_revalidation", None)
        if ticket is None:
            return
        self._manual_copy_revalidation = None
        self._manual_generation_capture_ids = set()
        self._manual_generation_completed_ids = set()
        self._manual_generation_followup_pending = None
        self._manual_generation_allow_submit = False
        self._manual_generation_terminal = True
        self._manual_generation_response_context = None
        # Restore presentation only when it still belongs to the original
        # result. A customer transition must never resurrect A as B's result.
        retained = (self._suggestion_result is not None and self._suggestion_result.allowed
                    and self._suggestion_reply_context is ticket["context"]
                    and self._suggestion_action_binding == ticket["binding"])
        if retained:
            self._manual_generation_source_result = ticket["source"]
            self._manual_generation_avatar_anchor = ticket["avatar"]
        self._set_manual_generation_phase("success" if retained else "failed")
        if retained:
            self._set_manual_review_status(reason)
        self.append_log(f"manual_copy_refresh_stopped: request={ticket['request_id']} reason={reason}"
                        f" blocker={self._manual_generation_last_blocker}")
        self._manual_copy_feedback(reason)

    def _manual_copy_revalidation_ready(self, context) -> None:
        ticket = getattr(self, "_manual_copy_revalidation", None)
        if ticket is None or not self._manual_copy_revalidation_current(ticket):
            self._manual_copy_revalidation_end("copy_refresh_changed")
            return
        validator = self._services.reply_generator.identity_lease_validator
        if (current_context_source_error(context) or not callable(validator) or not validator(context)
                or _reply_context_hash(context) != ticket["binding"].context_key
                or _reply_context_hash(ticket["context"]) != ticket["binding"].context_key):
            self._manual_copy_revalidation_end("copy_refresh_context_changed")
            return
        # Bind the SAME retained (possibly edited) result to genuinely new
        # evidence. Never read an AI cache, repaint/register feedback, or
        # modify the expired proof. Ordinary display and copy checks run next.
        self._suggestion_reply_context = deepcopy(context)
        self._manual_copy_revalidation = None
        self._manual_generation_followup_pending = None
        self._manual_generation_capture_ids = set()
        self._manual_generation_completed_ids = set()
        self._manual_generation_allow_submit = False
        self._manual_generation_terminal = True
        self._set_manual_generation_phase("success")
        self._set_manual_review_status("")
        copied = self._copy_reply_suggestion(ticket["suggestion"], ticket["index"],
            binding=ticket["binding"], _entry=getattr(self, "_manual_copy_entry", "main"))
        if copied is not None:
            self._manual_copy_feedback("copied" if copied else "copy_refresh_final_check_failed")

    def _copy_floating_suggestion(self, text: str, binding: SuggestionActionBinding | None) -> bool | None:
        if (getattr(self, "_manual_copy_ticket", None) is not None
                or getattr(self, "_manual_copy_revalidation", None) is not None):
            return None
        self._manual_copy_entry = "floating"
        if self._suggestion_result is None:
            if (getattr(self, "_manual_generation_enabled", False)
                    and self._manual_generation_phase == "success" and self._suggestion_result is not None):
                self._set_manual_review_status("binding_or_source_pending")
            self._manual_copy_feedback("binding_or_source_pending")
            return False
        for index, suggestion in enumerate(self._suggestion_result.suggestions):
            if suggestion.text == text:
                return self._copy_reply_suggestion(suggestion, index, binding=binding, _entry="floating")
        return False

    def _manual_copy_feedback(self, reason: str, *, pending: bool = False) -> None:
        if pending:
            message = ("正在本地重新核验原客户与聊天，通过后复制当前话术；不调用 AI、不扣额度"
                       if reason == "copy_refresh_pending" else "正在切回原微信窗口核验，核验通过后复制；不会发送消息")
        elif reason.startswith("copy_refresh_"):
            message = ("聊天或参考资料已变化，未复制旧建议；请核对原客户，处理新内容需点击生成建议。本次未调用 AI。"
                       if reason == "copy_refresh_context_changed" else
                       "本地复制核验未完成，建议仍保留；请保持原客户聊天完整可见，再点复制。本次未调用 AI、未扣额度。")
        elif reason == "copied":
            message = "已复制完整建议，可自行粘贴；未发送微信消息"
        elif reason.startswith("native_account_"):
            from whochat.services.reply import capture_failure_message
            message = "未复制：" + capture_failure_message(reason)
        elif "expired" in reason:
            message = "建议仍保留，聊天来源已过期；请再点复制进行本地核验（本次未调用 AI）"
        elif reason in {"account_avatar_pixels_changed", "account_avatar_sample_unreliable"}:
            message = "未复制：微信账号头像被遮挡或已变化；请保持原微信窗口完整可见"
        elif reason == "message_pixels_changed":
            message = "未复制：聊天区画面与生成时不同，暂不能确认是否仍匹配；建议已保留。请回到原聊天位置后再复制"
        elif reason == "title_pixels_changed":
            message = "未复制：客户标题画面已变化；请确认回到原客户后再复制，建议已保留"
        elif reason == "copy_focus_timeout":
            message = "未复制：切回原微信窗口的核验未就绪；请点击原微信聊天窗口后再复制，不用重新生成"
        elif reason == "copy_target_covered":
            message = "未复制：原微信聊天窗口仍被遮挡；请让原聊天可见后再复制，建议已保留"
        elif reason == "copy_clipboard_failed":
            message = "未能写入剪贴板，请稍后再次点击复制；不会重新调用 AI"
        elif reason == "binding_or_source_pending":
            message = "未复制：客户、来源期限或资料核验未通过；建议已保留"
        else:
            message = "未复制：原微信窗口尚未就绪或已切换；建议已保留，请回到原客户后再试"
        self.statusBar().showMessage(message, 5000)
        floating = getattr(self, "_floating", None)
        if floating is not None and hasattr(floating, "set_copy_status"):
            floating.set_copy_status("pending" if pending else "copied" if reason == "copied" else "rejected", message)
        if not pending:
            # Reasons only: never log suggestion text, pixels, clipboard contents or credentials.
            self.append_log(f"manual_copy_result: reason={reason} entry={getattr(self, '_manual_copy_entry', 'main')}"
                            f" request={getattr(self, '_manual_generation_request_id', 0)}")

    def _manual_copy_ticket_current(self, ticket) -> bool:
        return bool(getattr(self, "_manual_copy_ticket", None) is ticket
                    and self._manual_generation_phase == "success"
                    and self._manual_generation_request_id == ticket["request_id"]
                    and self._manual_generation_source_result is ticket["source"]
                    and self._suggestion_result is ticket["result"]
                    and self._suggestion_action_binding == ticket["binding"]
                    and self._manual_window_owner(self._services.runtime.state) == ticket["owner"]
                    and self.isVisible() and QApplication.activeModalWidget() is None
                    and 0 <= ticket["index"] < len(ticket["result"].suggestions)
                    and ticket["result"].suggestions[ticket["index"]] == ticket["suggestion"]
                    and ticket["suggestion"].text == ticket["text"])

    def _manual_copy_cancel(self, reason: str) -> None:
        ticket = getattr(self, "_manual_copy_ticket", None)
        if ticket is not None:
            try:
                window = self._services.runtime.state.window
                self.append_log("manual_copy_wait: " + json.dumps({
                    "request": ticket["request_id"], "reason": reason,
                    "stage": ticket.get("last_focus_reason", "not_polled"),
                    "refresh_attempted": bool(ticket.get("refresh_attempted", False)),
                    "refresh_result": ticket.get("refresh_result", "not_attempted"),
                    "runtime_foreground": bool(window.foreground), "runtime_covered": bool(window.covered),
                }, ensure_ascii=True, separators=(",", ":")))
            except Exception:
                pass
        self._manual_copy_ticket = None
        if self._manual_generation_phase == "success":
            self._set_manual_review_status(reason)
        self._manual_copy_feedback(reason)

    def _manual_copy_refresh_target(self, ticket) -> str:
        """One fresh native observation, not permission to copy or new OCR/AI.

        Only used after the native poll has seen the ORIGINAL WeChat foreground.
        Never fabricate Runtime visibility or pick another logged-in account.
        All ticket, source, account, geometry and pixel checks still run afterward.
        """
        if ticket.get("refresh_attempted", False):
            return ticket.get("refresh_result", "already_attempted")
        ticket["refresh_attempted"] = True
        result = "copy_refresh_unavailable"
        try:
            from whochat.platform.window_tracker import find_target_windows
            if not self._manual_copy_ticket_current(ticket) or time.monotonic() >= ticket["deadline"]:
                result = "copy_refresh_expired"
            else:
                windows = find_target_windows(self._config.targets)
                source = ticket["source"]
                matches = [window for window in windows
                           if window.hwnd == source.hwnd and window.process_id == source.process_id
                           and window.target_app == "wechat"]
                if (len(matches) != 1 or not self._manual_copy_ticket_current(ticket)
                        or time.monotonic() >= ticket["deadline"]):
                    result = "copy_refresh_target_unavailable"
                else:
                    window = matches[0]
                    anchor = ticket["native"]
                    if (window.rect != anchor.rect or window.client_rect != anchor.client_rect
                            or window.foreground is not True or window.visible is not True
                            or window.minimized is not False):
                        result = "copy_refresh_target_unavailable"
                    elif window.covered is not False:
                        result = "copy_target_covered"
                    else:
                        self._services.runtime.update_from_window_info(window)
                        result = "copy_runtime_refreshed"
        except Exception:
            # Preserve the bounded normal polling path if discovery is unavailable.
            result = "copy_refresh_unavailable"
        ticket["refresh_result"] = result
        return result

    def _manual_copy_focus_tick(self, ticket) -> None:
        if getattr(self, "_manual_copy_ticket", None) is not ticket:
            return
        if not self._manual_copy_ticket_current(ticket):
            self._manual_copy_cancel("copy_ticket_changed")
            return
        if time.monotonic() >= ticket["deadline"]:
            self._manual_copy_cancel("copy_focus_timeout")
            return
        poll = copy_focus_poll(ticket["source"], self._services.runtime.state, ticket["native"])
        ticket["last_focus_reason"] = poll.reason
        if not poll.allowed:
            self._manual_copy_cancel(poll.reason)
            return
        if poll.reason == "copy_focus_runtime_pending" and not ticket.get("refresh_attempted", False):
            refreshed = self._manual_copy_refresh_target(ticket)
            if not self._manual_copy_ticket_current(ticket):
                self._manual_copy_cancel("copy_ticket_changed")
                return
            if time.monotonic() >= ticket["deadline"]:
                self._manual_copy_cancel("copy_focus_timeout")
                return
            if refreshed == "copy_target_covered":
                self._manual_copy_cancel(refreshed)
                return
            poll = copy_focus_poll(ticket["source"], self._services.runtime.state, ticket["native"])
            ticket["last_focus_reason"] = poll.reason
            if not poll.allowed:
                self._manual_copy_cancel(poll.reason)
                return
        stable = not poll.needs_wait and poll.native == ticket["last_native"]
        ticket["last_native"] = None if poll.needs_wait else poll.native
        if not stable:
            QTimer.singleShot(50, self, lambda: self._manual_copy_focus_tick(ticket))
            return
        # Re-enter the complete DB/binding + fresh pixel gate. The ticket never
        # grants permission, renews the original proof, or triggers generation.
        try:
            copied = self._copy_reply_suggestion(ticket["suggestion"], ticket["index"],
                                                 binding=ticket["binding"], _copy_ticket=ticket)
        except Exception:
            copied = False
            self._manual_copy_feedback("copy_verification_failed")
        finally:
            if getattr(self, "_manual_copy_ticket", None) is ticket:
                self._manual_copy_ticket = None
        if copied is True:
            self._manual_copy_feedback("copied")

    def _copy_reply_suggestion(
        self, suggestion: ReplySuggestion, suggestion_index: int, *, binding=_USE_CURRENT_SUGGESTION_BINDING,
        _copy_ticket=None, _entry="main",
    ) -> bool | None:
        # None means pending, never success. One gesture/one ticket; a queued
        # second click cannot switch the pending text or perform another focus request.
        if ((getattr(self, "_manual_copy_ticket", None) is not None and _copy_ticket is None)
                or getattr(self, "_manual_copy_revalidation", None) is not None):
            return None
        if _copy_ticket is None:
            self._manual_copy_entry = "floating" if _entry == "floating" else "main"
        if _copy_ticket is not None and not self._manual_copy_ticket_current(_copy_ticket):
            self._manual_copy_feedback("copy_ticket_changed")
            return False
        if getattr(self, "_manual_generation_enabled", False):
            source = self._manual_generation_source_result
            if self._manual_generation_phase != "success" or source is None:
                self.statusBar().showMessage("请先点击「生成建议」核验当前客户，暂不能复制旧建议", 4000)
                return False
        # Explicit None belongs to a button painted without a safe identity;
        # never replace it with a later customer's authorization.
        if binding is _USE_CURRENT_SUGGESTION_BINDING:
            binding = self._suggestion_action_binding
        result = self._suggestion_result
        if (
            not self._suggestion_binding_is_current(binding)
            or result is None
            or not result.allowed
            or not 0 <= suggestion_index < len(result.suggestions)
            or result.suggestions[suggestion_index] != suggestion
        ):
            if (getattr(self, "_manual_generation_enabled", False)
                    and self._manual_generation_phase == "success" and result is not None and result.allowed):
                if (_copy_ticket is None and self._manual_copy_revalidation_begin(
                        suggestion, suggestion_index, binding)):
                    return None
                self._set_manual_review_status("binding_or_source_pending")
                self.statusBar().showMessage("建议已保留，客户或来源尚未通过复制核验；请确认原客户并点击「生成建议」读取最新内容", 4000)
                self._manual_copy_feedback("binding_or_source_pending")
            else:
                self.statusBar().showMessage("正在确认新客户，暂时不能复制建议", 3000)
            return False
        if getattr(self, "_manual_generation_enabled", False):
            if _copy_ticket is None:
                prepared = prepare_copy_target_foreground(source, self._services.runtime.state,
                    allowed_handoff_windows=self._manual_copy_handoff_windows())
                if not prepared.allowed:
                    self._set_manual_review_status(prepared.reason)
                    self._manual_copy_feedback(prepared.reason)
                    return False
                if prepared.needs_wait:
                    ticket = dict(source=source, result=result, suggestion=suggestion, index=suggestion_index,
                        text=suggestion.text, binding=binding, owner=self._manual_window_owner(self._services.runtime.state),
                        request_id=self._manual_generation_request_id, native=prepared.native,
                        last_native=None, deadline=time.monotonic() + 1.5)
                    self._manual_copy_ticket = ticket
                    self._manual_copy_feedback("copy_focus_pending", pending=True)
                    QTimer.singleShot(50, self, lambda: self._manual_copy_focus_tick(ticket))
                    return None
            check = self._manual_copy_screen_check()
            if not check.unchanged:
                self._set_manual_review_status(check.reason)
                self.statusBar().showMessage("建议已保留，但当前画面未通过复制核验；请打开原客户窗口，必要时点击「生成建议」读取最新内容", 5000)
                self._manual_copy_feedback(check.reason)
                return False
            if _copy_ticket is not None:
                final = copy_focus_poll(source, self._services.runtime.state, _copy_ticket["native"])
                if (not self._manual_copy_ticket_current(_copy_ticket) or not final.allowed or final.needs_wait
                        or final.native != _copy_ticket["last_native"]
                        or time.monotonic() >= _copy_ticket["deadline"]):
                    self._manual_copy_feedback(final.reason if not final.allowed else "copy_final_state_changed")
                    return False
            if not self._suggestion_binding_is_current(binding):
                self._manual_copy_feedback("binding_or_source_pending")
                return False
        try:
            clipboard = QApplication.clipboard()
            clipboard.setText(suggestion.text)
            if clipboard.text() != suggestion.text:
                raise RuntimeError("clipboard_readback_mismatch")
        except Exception:
            self._manual_copy_feedback("copy_clipboard_failed")
            return False
        if getattr(self, "_manual_generation_enabled", False):
            self._set_manual_review_status("")
        self.statusBar().showMessage(f"已复制建议：{suggestion.text[:24]}", 3000)
        self.append_log("reply_suggestion_copied")
        outcome = self._proactive_outcome
        if self._suggestion_mode != "proactive" or outcome is None:
            return True
        try:
            record = self._services.proactive_feedback.record_accepted(
                outcome,
                suggestion_index,
                model=self._config.ai.model,
            )
            self._services.activation_snapshots.warm_l1()
            self._proactive_feedback_status[suggestion_index] = "已记录采用"
            self.append_log(
                f"proactive_feedback_recorded: {record.feedback_id} type=accepted index={suggestion_index}"
            )
            # Repaint only local widgets.  This does not invalidate identity,
            # capture context, cache, conversation or Provider state.
            self._render_proactive_outcome(outcome)
        except Exception as exc:
            self.append_log(f"proactive_feedback_failed: accepted {type(exc).__name__}", "warning")
            self.statusBar().showMessage("建议已复制；采用记录保存失败", 3000)
        return True

    def _modify_proactive_suggestion(self, suggestion_index: int) -> None:
        opening_binding = self._suggestion_action_binding
        if not self._suggestion_binding_is_current(opening_binding):
            self.statusBar().showMessage("正在确认新客户，暂时不能修改建议", 3000)
            return
        outcome = self._proactive_outcome
        if outcome is None or outcome.result is None or suggestion_index >= len(outcome.result.suggestions):
            self.statusBar().showMessage("当前没有可修改的主动联系建议", 2500)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"修改建议 {suggestion_index + 1}")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        hint = QLabel("可修改表达或沟通方向；保存后用于学习，不会自动改变客户事实。")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        editor = QTextEdit()
        editor.setPlainText(outcome.result.suggestions[suggestion_index].text)
        editor.setMinimumHeight(120)
        layout.addWidget(editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存并采用")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._save_proactive_edit(suggestion_index, editor.toPlainText(), expected_binding=opening_binding)

    def _save_proactive_edit(
        self, suggestion_index: int, final_text: str, *, expected_binding: SuggestionActionBinding | None = None
    ) -> None:
        outcome = self._proactive_outcome
        binding = expected_binding or self._suggestion_action_binding
        if outcome is None or binding is None or binding.mode != "proactive" or not self._suggestion_binding_is_current(binding):
            self.statusBar().showMessage("当前主动联系上下文已变化，未保存", 3000)
            return
        try:
            record, updated = self._services.proactive_feedback.record_edited_and_accepted(
                outcome,
                suggestion_index,
                final_text,
                model=self._config.ai.model,
            )
        except ValueError:
            self.statusBar().showMessage("修改后的建议不能为空且不能超过500字", 3000)
            return
        except Exception as exc:
            self.append_log(f"proactive_feedback_failed: edit {type(exc).__name__}", "warning")
            self.statusBar().showMessage("修改保存失败", 3000)
            return
        self._proactive_outcome = updated
        self._services.activation_snapshots.warm_l1()
        identity = self._services.current_identity.snapshot
        self._manual_proactive_binding = (
            identity.conversation_token,
            self._services.activation_snapshots.revision(identity.account_scope_id, identity.contact_id),
        )
        self._pending_activation_refresh = None
        self._suggestion_result = updated.result
        self._proactive_feedback_status[suggestion_index] = "已按你的修改保存"
        self._render_proactive_outcome(updated)
        self.statusBar().showMessage("已按你的修改保存；同日缓存将显示此版本", 3000)
        self.append_log(
            f"proactive_feedback_recorded: {record.feedback_id} type=edited_and_accepted "
            f"index={suggestion_index}"
        )

    def _reject_proactive_suggestion(self, suggestion_index: int) -> None:
        opening_binding = self._suggestion_action_binding
        if (
            opening_binding is None
            or opening_binding.mode != "proactive"
            or not self._suggestion_binding_is_current(opening_binding)
        ):
            self.statusBar().showMessage("正在确认新客户，暂时不能评价建议", 3000)
            return
        outcome = self._proactive_outcome
        if outcome is None:
            self.statusBar().showMessage("当前没有可评价的主动联系建议", 2500)
            return
        labels = list(REJECTION_REASON_LABELS.values())
        selected, accepted = QInputDialog.getItem(
            self,
            "不满意原因",
            "请选择一个原因（只保存在本地）：",
            labels,
            0,
            False,
        )
        if not accepted:
            return
        if not self._suggestion_binding_is_current(opening_binding):
            self.statusBar().showMessage("当前建议上下文已变化，未保存评价", 3000)
            return
        reason_code = next(
            (code for code, label in REJECTION_REASON_LABELS.items() if label == selected),
            "other",
        )
        try:
            record = self._services.proactive_feedback.record_rejected(
                outcome,
                suggestion_index,
                reason_code,
                model=self._config.ai.model,
            )
        except Exception as exc:
            self.append_log(f"proactive_feedback_failed: rejected {type(exc).__name__}", "warning")
            self.statusBar().showMessage("不满意反馈保存失败", 3000)
            return
        self._proactive_feedback_status[suggestion_index] = f"已记录不满意：{selected}"
        self._proactive_rejected_feedback[suggestion_index] = (record.feedback_id, reason_code)
        self._render_proactive_outcome(outcome)
        self.statusBar().showMessage("已记录；只有点击重新生成才会调用 AI", 3000)
        self.append_log(
            f"proactive_feedback_recorded: {record.feedback_id} type=rejected "
            f"reason={reason_code} index={suggestion_index}"
        )

    def _regenerate_proactive_suggestions(self, suggestion_index: int) -> None:
        if getattr(self, "_manual_generation_enabled", False):
            self._on_generate_requested()
            return
        if self._identity_suggestions_suspended or not self._services.current_identity.snapshot.active:
            self.statusBar().showMessage("正在确认新客户，暂时不能重新生成", 3000)
            return
        outcome = self._proactive_outcome
        feedback = self._proactive_rejected_feedback.get(suggestion_index)
        if outcome is None or feedback is None:
            self.statusBar().showMessage("请先选择不满意原因", 2500)
            return
        if self._proactive_regeneration_used:
            self.statusBar().showMessage("本客户当前上下文今天已使用过一次重新生成", 3000)
            return
        job_id = self._services.proactive_tasks.submit_regeneration(
            outcome.request,
            self._config,
            feedback[0],
        )
        if job_id is None:
            self.statusBar().showMessage("主动联系任务正在运行，请稍后", 2500)
            return
        self._latest_proactive_job_id = job_id
        self._proactive_activation_revisions[job_id] = (
            outcome.request.account_scope_id,
            outcome.request.contact_id,
            self._services.activation_snapshots.revision(
                outcome.request.account_scope_id,
                outcome.request.contact_id,
            ),
        )
        self._proactive_regeneration_used = True
        self._proactive_feedback_status[suggestion_index] = "正在按反馈重新生成（本上下文仅一次）"
        self._render_proactive_outcome(outcome)
        self.statusBar().showMessage("正在按不满意原因重新生成1条建议", 3000)
        self.append_log(
            f"proactive_regeneration_submitted: job={job_id} reason={feedback[1]}"
        )

    def _record_reply_feedback(self, suggestion: ReplySuggestion, feedback: str) -> None:
        result = self._suggestion_result
        if result is None or not result.allowed:
            self.statusBar().showMessage("当前没有可评价的回复建议", 2500)
            return
        context = self._build_reply_context()
        digest = _reply_context_hash(context)
        record = self._services.reply_feedback.append(
            contact_id=context.contact.id if context.contact else None,
            strategy_id=context.strategy.id if context.strategy else None,
            provider=result.provider,
            status=result.status,
            suggestion_label=suggestion.label,
            suggestion_text=suggestion.text,
            risk=suggestion.risk,
            feedback=feedback,
            context_hash=digest,
            page_type=context.runtime.page.page_type.value,
            message_count=len(context.effective_messages),
            memory_count=reply_memory_item_count(context),
        )
        label = "好用" if feedback == "useful" else "不合适"
        self.statusBar().showMessage(f"已记录反馈：{label}", 2500)
        self.append_log(f"reply_feedback_recorded: {record.id} feedback={feedback}")
        if context.contact is not None:
            self._render_contact_detail(context.contact)
        self._refresh_generation_log_text()

    def _build_context_rail(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("RightRail")
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title = QLabel("当前聊天对象画像")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self._overview_contact_values = {}
        for label, value, key in [
            ("聊天对象", "未识别", "name"),
            ("分组", "默认", "strategy"),
            ("当前会话", "未识别", "status"),
            ("云端 AI", "未允许", "cloud"),
        ]:
            row, value_label = self._kv(label, value)
            self._overview_contact_values[key] = value_label
            layout.addWidget(row)

        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setText("暂无画像")
        notes.setMinimumHeight(72)
        notes.setMaximumHeight(104)
        self._overview_contact_notes = notes
        layout.addWidget(notes)

        layout.addWidget(QLabel("操作", objectName="SectionTitle"))
        confirm = QPushButton("确认当前对象")
        confirm.clicked.connect(lambda: self._update_selected_contact_status(ContactStatus.CONFIRMED))
        merge = QPushButton("合并对象")
        merge.clicked.connect(self._merge_selected_contact)
        protect = QPushButton("进入手动回复保护")
        protect.clicked.connect(self._protect_selected_contact)
        action_grid = QGridLayout()
        action_grid.setSpacing(6)
        action_grid.addWidget(confirm, 0, 0)
        action_grid.addWidget(merge, 0, 1)
        action_grid.addWidget(protect, 1, 0, 1, 2)
        layout.addLayout(action_grid)

        suggestions_title = QLabel("回复建议")
        suggestions_title.setObjectName("SectionTitle")
        self._suggestions_title = suggestions_title
        layout.addWidget(suggestions_title)
        suggestions_scroll = QScrollArea()
        suggestions_scroll.setObjectName("SuggestionScroll")
        suggestions_scroll.setWidgetResizable(True)
        suggestions_scroll.setFrameShape(QFrame.Shape.NoFrame)
        suggestions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        suggestions_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        suggestions_scroll.setMinimumHeight(156)
        suggestions_scroll.setMaximumHeight(336)
        suggestions = self._panel("", "仅在聊天页、聊天对象与策略通过安全门控后生成；默认只复制，不自动发送。")
        suggestions.layout().itemAt(0).widget().hide()
        suggestions.setObjectName("SuggestionList")
        suggestions.setMinimumWidth(300)
        suggestions_scroll.setWidget(suggestions)
        self._suggestions_panel = suggestions
        self._suggestions_scroll = suggestions_scroll
        layout.addWidget(suggestions_scroll, 1)
        return rail

    def _kv(self, key: str, value: str) -> tuple[QWidget, QLabel]:
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        k = QLabel(key)
        k.setObjectName("Muted")
        v = QLabel(value)
        v.setObjectName("Badge")
        v.setMaximumWidth(190)
        v.setWordWrap(True)
        layout.addWidget(k)
        layout.addStretch(1)
        layout.addWidget(v)
        return box, v

    def _build_contacts_page(self) -> QWidget:
        splitter = QWidget()
        layout = QHBoxLayout(splitter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        list_panel = self._panel("聊天对象", "私聊和群聊都可以查看画像、别名、分组和聊天记录。")
        contact_list = QListWidget()
        self._contact_list = contact_list
        contacts = self._reload_contact_list()
        list_panel.layout().addWidget(contact_list)
        layout.addWidget(list_panel, 1)

        detail = self._panel("对象详情", "此处展示选中私聊或群聊的画像、身份线索和聊天摘要。")
        tabs = QTabWidget()
        self._contact_profile_text = self._readonly_text("画像", "暂无聊天对象。需要先从当前会话识别或手动创建。")
        self._contact_identity_text = self._readonly_text("身份", "暂无身份链接。")
        self._contact_members_text = self._readonly_text("群成员", "仅群聊可用。")
        self._contact_chat_text = self._readonly_text("聊天记录", "暂无聊天记录。")
        self._contact_memory_text = self._readonly_text("记忆", "暂无记忆。")
        # `_render_contact_detail` has always populated this pane, but the pane itself
        # was never created, so the reply-feedback evidence was invisible in the UI.
        self._contact_feedback_text = self._readonly_text("回复反馈", "暂无回复反馈。")
        tabs.addTab(self._contact_profile_text, "画像")
        tabs.addTab(self._contact_identity_text, "身份")
        tabs.addTab(self._contact_members_text, "群成员")
        tabs.addTab(self._contact_chat_text, "聊天记录")
        tabs.addTab(self._contact_memory_text, "记忆")
        tabs.addTab(self._contact_feedback_text, "回复反馈")
        detail.layout().addWidget(tabs)
        actions = QGridLayout()
        actions.setSpacing(6)
        confirm = QPushButton("确认")
        confirm.setToolTip("把自动识别到的聊天对象标记为已确认；已确认对象更适合长期画像和回复建议。")
        confirm.clicked.connect(lambda: self._update_selected_contact_status(ContactStatus.CONFIRMED))
        edit = QPushButton("编辑")
        edit.setToolTip("修改显示名称、分组策略、云端 AI 授权和备注。")
        edit.clicked.connect(self._edit_selected_contact_profile)
        alias = QPushButton("别名")
        alias.setToolTip("给当前聊天对象增加别名；用于昵称变化或 OCR 识别到不同写法时继续归到同一对象。")
        alias.clicked.connect(self._add_alias_to_selected_contact)
        merge = QPushButton("合并")
        merge.setToolTip("把当前聊天对象合并到另一个对象；适合重复识别或同一个人有多个窗口对象的情况。")
        merge.clicked.connect(self._merge_selected_contact)
        link_identity = QPushButton("链接身份")
        link_identity.setToolTip("把当前平台聊天对象链接到一个真实身份；例如 微信·小红 和 QQ·大红 可以链接为同一个人。")
        link_identity.clicked.connect(self._link_selected_contact_identity)
        new_identity = QPushButton("新建身份")
        new_identity.setToolTip("为当前聊天对象创建一个真实身份；身份用于跨应用合并同一个人。")
        new_identity.clicked.connect(self._create_identity_for_selected_contact)
        person_alias = QPushButton("身份别名")
        person_alias.setToolTip("给已链接的真实身份增加别名；适合一个人使用多个昵称的情况。")
        person_alias.clicked.connect(self._add_alias_to_selected_person)
        add_member = QPushButton("添加成员")
        add_member.setToolTip("给群聊手动添加群成员候选；仅群聊有意义。")
        add_member.clicked.connect(self._add_member_to_selected_group)
        member_identity = QPushButton("成员身份")
        member_identity.setToolTip("把选中的群成员昵称链接到真实身份；用于识别群里具体是谁在说话。")
        member_identity.clicked.connect(self._link_selected_group_member_identity)
        member_contact = QPushButton("成员对象")
        member_contact.setToolTip("把群成员昵称链接到已有私聊对象；适合群成员和好友列表中的同一人重叠。")
        member_contact.clicked.connect(self._link_selected_group_member_contact)
        protect = QPushButton("保护")
        protect.setToolTip("把当前聊天对象切到手动回复保护分组；用于亲友或敏感关系，避免 AI 代替你处理感情。")
        protect.clicked.connect(self._protect_selected_contact)
        export = QPushButton("导出")
        export.setToolTip("导出当前聊天对象的数据摘要，便于备份或排查。")
        export.clicked.connect(self._export_selected_contact_data)
        clear = QPushButton("清空")
        clear.setToolTip("清空当前聊天对象的聊天记录、记忆、AI 审计和反馈；不会删除对象本身。")
        clear.clicked.connect(self._clear_selected_contact_data)
        ai_toggle = QPushButton("单客云端关")
        ai_toggle.setCheckable(True)
        ai_toggle.setToolTip(_CONTACT_LONG_TERM_CLOUD_HELP)
        ai_toggle.clicked.connect(self._toggle_selected_contact_cloud_ai)
        self._contact_ai_toggle = ai_toggle
        buttons = [
            confirm,
            edit,
            alias,
            merge,
            link_identity,
            new_identity,
            person_alias,
            add_member,
            member_identity,
            member_contact,
            protect,
            export,
            clear,
            ai_toggle,
        ]
        for index, button in enumerate(buttons):
            actions.addWidget(button, index // 7, index % 7)
        detail.layout().addLayout(actions)
        layout.addWidget(detail, 2)
        contact_list.currentItemChanged.connect(self._on_contact_selected)
        if contacts:
            contact_list.setCurrentRow(0)
        else:
            self._sync_contact_ai_toggle(None)
        return self._build_page_shell(splitter)

    def _build_customer_profile_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        navigation = QHBoxLayout()
        back = QPushButton("← 返回")
        back.setObjectName("CustomerProfileBackButton")
        back.setEnabled(False)
        back.clicked.connect(self._return_from_customer_profile)
        breadcrumb = QLabel("客户档案")
        breadcrumb.setObjectName("Muted")
        navigation.addWidget(back)
        navigation.addWidget(breadcrumb)
        navigation.addStretch(1)
        layout.addLayout(navigation)
        self._customer_profile_scroll = scroll
        self._customer_profile_back_button = back
        self._customer_profile_breadcrumb = breadcrumb

        customer = self._panel("当前客户", "档案跟随当前已稳定识别的微信联系人，并按工作微信账号隔离。")
        customer.setObjectName("CustomerProfileHeader")
        header = QHBoxLayout()
        identity = QVBoxLayout()
        name = QLabel("等待识别当前客户")
        name.setObjectName("MetricValue")
        account = QLabel("工作微信：—")
        account.setObjectName("Muted")
        summary = QLabel("暂无客户档案")
        summary.setObjectName("ActionPrimary")
        summary.setWordWrap(True)
        identity.addWidget(name)
        identity.addWidget(account)
        identity.addWidget(summary)
        header.addLayout(identity, 1)
        context_box = QVBoxLayout()
        stage = QLabel("销售状态：新客户")
        stage.setObjectName("Badge")
        stage_row = QHBoxLayout()
        stage_combo = QComboBox()
        stage_combo.setObjectName("CustomerSalesStageCombo")
        for stage_key, stage_label in SALES_STAGE_LABELS.items():
            stage_combo.addItem(stage_label, stage_key)
        set_stage = QPushButton("人工设置")
        set_stage.clicked.connect(self._set_customer_sales_stage)
        stage_row.addWidget(stage_combo, 1)
        stage_row.addWidget(set_stage)
        context = QComboBox()
        context.setMinimumWidth(230)
        context.setToolTip("这里只切换档案查看的种植项目，不会改变微信联系人或 Reply AI 的上下文。")
        context.currentIndexChanged.connect(self._on_customer_profile_context_changed)
        context_box.addWidget(stage, 0, Qt.AlignmentFlag.AlignRight)
        context_box.addLayout(stage_row)
        context_box.addWidget(context)
        header.addLayout(context_box)
        customer.layout().addLayout(header)
        state = QLabel("当前联系人稳定后自动加载；档案展示不会调用云端 AI。")
        state.setObjectName("TinyMuted")
        state.setWordWrap(True)
        customer.layout().addWidget(state)
        imported_archive=QPushButton('查看已关联聊天历史（本地）')
        imported_archive.clicked.connect(self._show_customer_imported_archive)
        customer.layout().addWidget(imported_archive)
        self._customer_profile_name = name
        self._customer_profile_account = account
        self._customer_profile_summary = summary
        self._customer_profile_stage = stage
        self._customer_profile_stage_combo = stage_combo
        self._customer_profile_context = context
        self._customer_profile_state = state
        layout.addWidget(customer)

        facts = self._panel("客户档案", "按编号汇总本地已确认资料与关键业务状态；缺失项显示“暂未确认”。")
        core = QTableWidget(0, 3)
        core.setObjectName("CustomerProfileCoreTable")
        core.setHorizontalHeaderLabels(["编号资料", "当前值", "操作"])
        _compact_table(core)
        core.verticalHeader().setDefaultSectionSize(34)
        core.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        core.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        core.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        core.setColumnWidth(2, 170)
        core.setMinimumHeight(455)
        self._customer_profile_core_table = core
        facts.layout().addWidget(core)
        layout.addWidget(facts)

        tracking_panel = self._panel(
            "跟踪建议",
            "完全根据本地档案、事件、销售阶段与待办生成；不会调用 DeepSeek。",
        )
        tracking = QListWidget()
        tracking.setObjectName("CustomerTrackingAdvice")
        tracking.setMinimumHeight(125)
        self._customer_profile_tracking_advice = tracking
        tracking_panel.layout().addWidget(tracking)
        layout.addWidget(tracking_panel)

        concerns_panel = self._panel("客户关注", "明确表达的顾虑、偏好和价格关注。")
        concerns = QLabel("暂无明确关注点")
        concerns.setObjectName("CustomerConcernTags")
        concerns.setWordWrap(True)
        self._customer_profile_concerns = concerns
        concerns_panel.layout().addWidget(concerns)
        layout.addWidget(concerns_panel)

        pending_panel = self._panel("待确认资料", "uncertain / conflicting 与已确认资料分开；不要求日常逐条审核。")
        pending = QTableWidget(0, 4)
        pending.setObjectName("CustomerProfilePendingTable")
        pending.setHorizontalHeaderLabels(["资料", "候选值", "状态", "操作"])
        _compact_table(pending)
        pending.verticalHeader().setDefaultSectionSize(36)
        pending.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        pending.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        pending.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        pending.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        pending.setColumnWidth(3, 220)
        pending.setMinimumHeight(130)
        self._customer_profile_pending_table = pending
        pending_panel.layout().addWidget(pending)
        layout.addWidget(pending_panel)

        timeline_panel = self._panel("最近记录", "按时间倒序显示问题、方案、产品、购买、使用、效果与回访。")
        filters = QHBoxLayout()
        filters.addWidget(QLabel("筛选"))
        timeline_filter = QComboBox()
        for title, key in (
            ("全部", "all"),
            ("问题", "problem"),
            ("方案", "solution"),
            ("产品/报价", "product_quote"),
            ("购买/使用", "purchase_usage"),
            ("效果", "effect"),
            ("回访", "followup"),
        ):
            timeline_filter.addItem(title, key)
        timeline_filter.currentIndexChanged.connect(self._on_customer_profile_timeline_filter_changed)
        filters.addWidget(timeline_filter)
        filters.addStretch(1)
        more = QPushButton("查看更多")
        more.clicked.connect(self._load_more_customer_profile_events)
        filters.addWidget(more)
        timeline_panel.layout().addLayout(filters)
        timeline = QListWidget()
        timeline.setObjectName("CustomerProfileTimeline")
        timeline.setMinimumHeight(230)
        timeline_panel.layout().addWidget(timeline)
        self._customer_profile_timeline_filter = timeline_filter
        self._customer_profile_timeline = timeline
        layout.addWidget(timeline_panel)

        followup_panel = self._panel("待回访", "当前客户的 pending 回访；所有操作均为本地 SQLite，不会自动发送微信。")
        followups = QListWidget()
        followups.setObjectName("CustomerProfileFollowups")
        followups.setMinimumHeight(100)
        self._customer_profile_followups = followups
        followup_panel.layout().addWidget(followups)
        followup_actions = QHBoxLayout()
        add_followup = QPushButton("新增回访")
        add_followup.clicked.connect(self._create_followup_for_current_customer)
        manage_followups = QPushButton("打开待跟进工作台")
        manage_followups.clicked.connect(lambda: self._select_page("followups"))
        followup_actions.addWidget(add_followup)
        followup_actions.addWidget(manage_followups)
        followup_actions.addStretch(1)
        followup_panel.layout().addLayout(followup_actions)
        layout.addWidget(followup_panel)
        layout.addStretch(1)

        scroll.setWidget(body)
        self._customer_profile_page = self._build_page_shell(scroll)
        return self._customer_profile_page

    def _build_followup_page(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        metrics = QHBoxLayout()
        overdue_card, overdue = self._metric_card_with_value("已逾期", "0", "最先处理")
        today_card, today = self._metric_card_with_value("今天到期", "0", "今日联系")
        future_card, future = self._metric_card_with_value("未来7天", "0", "提前安排")
        self._followup_summary_overdue = overdue
        self._followup_summary_today = today
        self._followup_summary_future = future
        metrics.addWidget(overdue_card)
        metrics.addWidget(today_card)
        metrics.addWidget(future_card)
        layout.addLayout(metrics)

        panel = self._panel(
            "销售待办",
            "汇总所有已绑定工作微信；点击客户只切换 WhoChat 档案，不会控制或切换微信窗口。",
        )
        table = QTableWidget(0, 9)
        table.setObjectName("FollowupWorkbenchTable")
        table.setHorizontalHeaderLabels(
            ["分类", "客户", "工作微信", "作物", "回访事项", "到期时间", "销售阶段", "优先级", "来源"]
        )
        _compact_table(table)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for column in (0, 1, 2, 3, 5, 6, 7, 8):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumHeight(430)
        table.cellDoubleClicked.connect(lambda _row, _column: self._open_selected_followup_customer())
        self._followup_table = table
        panel.layout().addWidget(table)

        actions = QHBoxLayout()
        open_customer = QPushButton("查看客户档案")
        open_customer.clicked.connect(self._open_selected_followup_customer)
        add = QPushButton("新增回访")
        add.clicked.connect(self._create_followup_for_current_customer)
        edit = QPushButton("修改任务")
        edit.clicked.connect(self._edit_selected_followup)
        complete = QPushButton("标记完成")
        complete.clicked.connect(lambda: self._resolve_selected_followup("completed"))
        cancel = QPushButton("取消任务")
        cancel.clicked.connect(lambda: self._resolve_selected_followup("cancelled"))
        refresh = QPushButton("刷新")
        refresh.clicked.connect(lambda: self._refresh_followup_workbench(force=True))
        for button in (open_customer, add, edit, complete, cancel, refresh):
            actions.addWidget(button)
        actions.addStretch(1)
        panel.layout().addLayout(actions)

        note = QLabel("所有阶段判断与回访操作均在本地 SQLite 完成；本页面不会调用 DeepSeek，也不会自动发送微信。")
        note.setObjectName("TinyMuted")
        note.setWordWrap(True)
        panel.layout().addWidget(note)
        layout.addWidget(panel, 1)
        self._followup_page = self._build_page_shell(body)
        return self._followup_page

    def _refresh_followup_workbench(self, *, force: bool = False) -> None:
        if self._followup_table is None:
            return
        if self._followup_inflight:
            self._followup_pending_refresh = self._followup_pending_refresh or force
            return
        self._followup_inflight = True
        self._followup_latest_request_key = f"followups:{time.monotonic_ns()}"
        if force:
            self.statusBar().showMessage("正在后台加载待跟进，不影响微信识别与回复建议", 1800)
        self._ui_data_tasks.submit(
            "followup_workbench",
            self._followup_latest_request_key,
            self._services.sales_followup.get_workbench,
        )

    def _render_followup_workbench(self, workbench) -> None:
        if workbench.revision == self._followup_revision:
            return
        self._followup_revision = workbench.revision
        self._followup_items = {item.id: item for item in workbench.all_items}
        if self._followup_summary_overdue is not None:
            self._followup_summary_overdue.setText(str(len(workbench.overdue)))
        if self._followup_summary_today is not None:
            self._followup_summary_today.setText(str(len(workbench.today)))
        if self._followup_summary_future is not None:
            self._followup_summary_future.setText(str(len(workbench.future7)))
        rows = [
            ("已逾期", item) for item in workbench.overdue
        ] + [
            ("今天", item) for item in workbench.today
        ] + [
            ("未来7天", item) for item in workbench.future7
        ]
        table = self._followup_table
        table.setUpdatesEnabled(False)
        table.setRowCount(len(rows))
        source_labels = {"explicit": "明确约定", "system_rule": "规则建议", "user_created": "人工创建"}
        for row, (bucket, item) in enumerate(rows):
            values = (
                bucket,
                item.customer_name,
                item.account_name,
                item.crop,
                item.title,
                _format_profile_time(item.due_at),
                item.sales_stage_label,
                str(item.priority),
                source_labels.get(item.source, item.source),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, item.id)
                additional = (
                    f"\n另有{item.additional_followup_count}条次要待办，可在客户档案中查看"
                    if item.additional_followup_count else ""
                )
                cell.setToolTip(
                    f"{item.reason or item.title}\n作物来源：{item.crop_source_type}{additional}"
                )
                table.setItem(row, column, cell)
        table.setUpdatesEnabled(True)
        self._restore_followup_selection()
        if not rows:
            self.statusBar().showMessage("目前没有已逾期、今天或未来7天的待跟进任务", 2500)

    def _selected_followup_item(self) -> FollowupWorkbenchItem | None:
        table = self._followup_table
        if table is None or table.currentRow() < 0:
            return None
        cell = table.item(table.currentRow(), 0)
        followup_id = cell.data(Qt.ItemDataRole.UserRole) if cell else None
        return self._followup_items.get(followup_id) if followup_id else None

    def _open_selected_followup_customer(self) -> None:
        item = self._selected_followup_item()
        if item is None:
            self.statusBar().showMessage("请先选择一条待跟进任务", 2500)
            return
        table = self._followup_table
        self._profile_return_state = {
            "page": "followups",
            "followup_id": item.id,
            "row": table.currentRow() if table is not None else -1,
            "scroll": table.verticalScrollBar().value() if table is not None else 0,
        }
        if self._customer_profile_back_button is not None:
            self._customer_profile_back_button.setEnabled(True)
            self._customer_profile_back_button.setText("← 返回待跟进")
        if self._customer_profile_breadcrumb is not None:
            self._customer_profile_breadcrumb.setText(f"待跟进  ›  {item.customer_name}")
        self._customer_profile_navigation_contact_id = item.contact_id
        self._reload_contact_list(select_contact_id=item.contact_id)
        self._customer_profile_binding = None
        self._select_page("customer_profile")

    def _return_from_customer_profile(self) -> None:
        state = self._profile_return_state
        if not state:
            self._select_page("overview")
            return
        self._select_page(str(state.get("page") or "followups"))

    def _restore_followup_selection(self) -> None:
        state = self._profile_return_state
        table = self._followup_table
        if not state or table is None or state.get("page") != "followups":
            return
        followup_id = str(state.get("followup_id") or "")
        selected_row = -1
        for row in range(table.rowCount()):
            cell = table.item(row, 0)
            if cell and str(cell.data(Qt.ItemDataRole.UserRole) or "") == followup_id:
                selected_row = row
                break
        if selected_row < 0:
            selected_row = min(max(0, int(state.get("row") or 0)), max(0, table.rowCount() - 1))
        if table.rowCount():
            table.selectRow(selected_row)
        table.verticalScrollBar().setValue(int(state.get("scroll") or 0))

    def _create_followup_for_current_customer(self) -> None:
        selected = self._selected_followup_item()
        profile_id = selected.customer_profile_id if selected else ""
        context_id = None
        if not profile_id and self._customer_profile_dashboard and self._customer_profile_dashboard.available:
            profile_id = self._customer_profile_dashboard.profile_id
            context_id = self._customer_profile_dashboard.selected_context_id
        if not profile_id:
            contact = self._customer_profile_contact()
            profile = self._services.customer_profiles.get_customer_profile(
                account_scope_id=contact.account_scope_id,
                contact_id=contact.id,
            ) if contact else None
            profile_id = profile.id if profile else ""
        if not profile_id:
            QMessageBox.information(self, "新增回访", "请先选择已有客户档案或待跟进任务。")
            return
        values = self._prompt_followup_values("新增回访任务")
        if values is None:
            return
        title, due_at, priority = values
        try:
            self._services.sales_followup.create_user_followup(
                profile_id,
                growing_context_id=context_id,
                title=title,
                due_at=due_at,
                priority=priority,
            )
        except Exception as exc:
            QMessageBox.warning(self, "新增失败", f"没有写入任何更改：\n{exc}")
            return
        self._after_followup_change("回访任务已创建")

    def _edit_selected_followup(self) -> None:
        item = self._selected_followup_item()
        if item is None:
            self.statusBar().showMessage("请先选择一条待跟进任务", 2500)
            return
        values = self._prompt_followup_values(
            "修改回访任务",
            title=item.title,
            due_at=_profile_time_for_edit(item.due_at),
            priority=item.priority,
        )
        if values is None:
            return
        title, due_at, priority = values
        try:
            self._services.sales_followup.update_followup(
                item.id,
                title=title,
                due_at=due_at,
                priority=priority,
            )
        except Exception as exc:
            QMessageBox.warning(self, "修改失败", f"没有写入任何更改：\n{exc}")
            return
        self._after_followup_change("回访任务已更新")

    def _resolve_selected_followup(self, status: str) -> None:
        item = self._selected_followup_item()
        if item is None:
            self.statusBar().showMessage("请先选择一条待跟进任务", 2500)
            return
        verb = "完成" if status == "completed" else "取消"
        answer = QMessageBox.question(
            self,
            f"{verb}任务",
            f"确认{verb}“{item.title}”？任务历史会保留。",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            if status == "completed":
                self._services.sales_followup.complete_followup(item.id)
            else:
                self._services.sales_followup.cancel_followup(item.id)
        except Exception as exc:
            QMessageBox.warning(self, f"{verb}失败", f"没有写入任何更改：\n{exc}")
            return
        self._after_followup_change(f"回访任务已{verb}")

    def _prompt_followup_values(
        self,
        dialog_title: str,
        *,
        title: str = "",
        due_at: str = "",
        priority: int = 60,
    ) -> tuple[str, str, int] | None:
        default_due = due_at or (datetime.now().astimezone() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
        task_title, ok = QInputDialog.getText(self, dialog_title, "回访事项：", text=title)
        if not ok or not task_title.strip():
            return None
        task_due, ok = QInputDialog.getText(
            self,
            dialog_title,
            "到期时间（YYYY-MM-DD HH:MM）：",
            text=default_due,
        )
        if not ok or not task_due.strip():
            return None
        task_priority, ok = QInputDialog.getInt(
            self,
            dialog_title,
            "优先级（0-100）：",
            value=int(priority),
            minValue=0,
            maxValue=100,
        )
        if not ok:
            return None
        return task_title.strip(), task_due.strip(), task_priority

    def _after_followup_change(self, message: str) -> None:
        dashboard = self._customer_profile_dashboard
        self._followup_revision = ""
        self._customer_profile_revision = ""
        self._invalidate_proactive_context(
            "followup_context_changed",
            dashboard.contact_id if dashboard is not None else None,
        )
        self._refresh_followup_workbench(force=True)
        if self._customer_profile_page is not None:
            self._refresh_customer_profile_page(force=True)
        self.statusBar().showMessage(message, 3000)

    def _set_customer_sales_stage(self) -> None:
        dashboard = self._customer_profile_dashboard
        combo = self._customer_profile_stage_combo
        if dashboard is None or not dashboard.available or combo is None:
            self.statusBar().showMessage("当前没有可编辑的客户档案", 2500)
            return
        stage = combo.currentData(Qt.ItemDataRole.UserRole)
        reason, ok = QInputDialog.getText(self, "人工设置销售阶段", "设置原因（可留空）：")
        if not ok:
            return
        try:
            self._services.customer_profile_service.set_sales_stage(
                dashboard.profile_id,
                str(stage),
                reason=reason,
            )
        except Exception as exc:
            QMessageBox.warning(self, "阶段设置失败", f"没有写入任何更改：\n{exc}")
            return
        self._customer_profile_revision = ""
        self._followup_revision = ""
        self._invalidate_proactive_context("sales_stage_changed", dashboard.contact_id)
        self._refresh_customer_profile_page(force=True)
        self.statusBar().showMessage("销售阶段已人工设置并保留审计记录", 3500)

    def _customer_profile_contact(self) -> Contact | None:
        if self._customer_profile_navigation_contact_id:
            contact = self._services.contacts.get(self._customer_profile_navigation_contact_id)
            if contact is not None:
                return contact
        return self._active_capture_contact() or self._selected_contact()

    def _poll_customer_profile_refresh(self) -> None:
        if self._customer_profile_page is not None and self._pages.currentWidget() is self._customer_profile_page:
            self._refresh_customer_profile_page(force=False)
        elif self._followup_page is not None and self._pages.currentWidget() is self._followup_page:
            self._refresh_followup_workbench(force=False)

    def _refresh_customer_profile_page(self, *, force: bool = False) -> None:
        if self._customer_profile_state is None:
            return
        contact = self._customer_profile_contact()
        if contact is None:
            self._render_customer_profile_empty("等待稳定识别当前微信联系人")
            return
        binding = (contact.account_scope_id, contact.id)
        selected_context_id = None
        if self._customer_profile_context is not None and self._customer_profile_binding == binding:
            selected_context_id = self._customer_profile_context.currentData(Qt.ItemDataRole.UserRole)
        if binding != self._customer_profile_binding:
            self._customer_profile_event_limit = 15
            self._customer_profile_revision = ""
        filter_key = "all"
        if self._customer_profile_timeline_filter is not None:
            filter_key = self._customer_profile_timeline_filter.currentData(Qt.ItemDataRole.UserRole) or "all"
        request_key = "|".join(
            (
                "customer_profile",
                contact.account_scope_id,
                contact.id,
                str(selected_context_id or ""),
                str(filter_key),
                str(self._customer_profile_event_limit),
            )
        )
        self._customer_profile_latest_request_key = request_key
        if request_key in self._customer_profile_inflight_keys:
            return
        self._customer_profile_inflight_keys.add(request_key)
        if force and self._customer_profile_state is not None:
            self._customer_profile_state.setText("正在后台加载本地客户档案；OCR 与 Reply AI 不受影响。")
        self._ui_data_tasks.submit(
            "customer_profile",
            request_key,
            lambda: self._services.customer_profile_service.get_customer_dashboard(
                account_scope_id=contact.account_scope_id,
                contact_id=contact.id,
                selected_context_id=selected_context_id,
                timeline_filter=filter_key,
                event_limit=self._customer_profile_event_limit,
            ),
        )

    def _on_ui_data_task_ready(self, result: UiDataTaskResult) -> None:
        if result.kind == "followup_workbench":
            self._followup_inflight = False
            if result.request_key != self._followup_latest_request_key:
                return
            render_started = time.perf_counter()
            changed = result.value.revision != self._followup_revision
            self._render_followup_workbench(result.value)
            render_ms = (time.perf_counter() - render_started) * 1000.0
            if changed or result.query_ms + render_ms >= 500.0:
                self._record_ui_performance(
                    "followup_workbench_load",
                    "followups",
                    result.query_ms + render_ms,
                    main_thread_ms=render_ms,
                    query_ms=result.query_ms,
                )
            if self._followup_pending_refresh:
                self._followup_pending_refresh = False
                QTimer.singleShot(0, lambda: self._refresh_followup_workbench(force=False))
            return
        if result.kind != "customer_profile":
            return
        self._customer_profile_inflight_keys.discard(result.request_key)
        if result.request_key != self._customer_profile_latest_request_key:
            return
        dashboard = result.value
        self._customer_profile_last_error = ""
        self._customer_profile_binding = (dashboard.account_scope_id, dashboard.contact_id)
        if dashboard.revision == self._customer_profile_revision:
            return
        render_started = time.perf_counter()
        self._customer_profile_revision = dashboard.revision
        self._customer_profile_dashboard = dashboard
        self._render_customer_profile_dashboard(dashboard)
        render_ms = (time.perf_counter() - render_started) * 1000.0
        self._record_ui_performance(
            "customer_profile_load",
            "customer_profile",
            result.query_ms + render_ms,
            main_thread_ms=render_ms,
            query_ms=result.query_ms,
        )

    def _on_ui_data_task_failed(self, failure: UiDataTaskFailure) -> None:
        if failure.kind == "followup_workbench":
            self._followup_inflight = False
            self._followup_pending_refresh = False
            if failure.request_key == self._followup_latest_request_key:
                self.statusBar().showMessage("待跟进暂时无法加载；OCR、Memory 和 Reply AI 不受影响", 4000)
        elif failure.kind == "customer_profile":
            self._customer_profile_inflight_keys.discard(failure.request_key)
            if failure.request_key == self._customer_profile_latest_request_key:
                signature = f"{failure.error_type}:{failure.error_message}"
                self._render_customer_profile_error()
                if signature != self._customer_profile_last_error:
                    self._customer_profile_last_error = signature
                    self.append_log(f"customer_profile_load_failed: {failure.error_type}", "warning")
        try:
            self._services.logs.append(
                "warning",
                "ui_performance",
                "local_data_task_failed",
                f"{failure.kind} failed: {failure.error_type}",
                {"request_key": failure.request_key, "error_type": failure.error_type},
            )
        except Exception:
            pass

    def _render_customer_profile_dashboard(self, dashboard: CustomerDashboard) -> None:
        if self._customer_profile_name is not None:
            self._customer_profile_name.setText(dashboard.display_name)
        if self._customer_profile_account is not None:
            self._customer_profile_account.setText(f"工作微信：{dashboard.account_name}")
        if self._customer_profile_stage is not None:
            source_label = "人工" if dashboard.sales_stage_source == "user_explicit" else "自动"
            self._customer_profile_stage.setText(f"销售状态：{dashboard.sales_stage_label}（{source_label}）")
        if self._customer_profile_stage_combo is not None:
            index = self._customer_profile_stage_combo.findData(dashboard.sales_stage)
            if index >= 0:
                self._customer_profile_stage_combo.setCurrentIndex(index)
        selected = next((item for item in dashboard.contexts if item.id == dashboard.selected_context_id), None)
        if self._customer_profile_summary is not None:
            if selected is None:
                self._customer_profile_summary.setText("暂无种植项目")
            else:
                second = " · ".join(item for item in (selected.region, selected.growth_stage) if item)
                suffix = f"　其他种植项目（{dashboard.other_context_count}）" if dashboard.other_context_count else ""
                self._customer_profile_summary.setText(f"{selected.short_label}{suffix}\n{second}".strip())
        if self._customer_profile_context is not None:
            combo = self._customer_profile_context
            combo.blockSignals(True)
            combo.clear()
            if dashboard.contexts:
                for item in dashboard.contexts:
                    prefix = "当前相关：" if item.id == dashboard.selected_context_id else "其他项目："
                    combo.addItem(f"{prefix}{item.short_label}", item.id)
                    if item.id == dashboard.selected_context_id:
                        combo.setCurrentIndex(combo.count() - 1)
                combo.setEnabled(True)
            else:
                combo.addItem("暂无种植项目", None)
                combo.setEnabled(False)
            combo.blockSignals(False)
        if self._customer_profile_state is not None:
            self._customer_profile_state.setText(
                f"{dashboard.history_status_label}；档案和跟踪建议已从本地 SQLite 加载，不调用云端 AI。"
            )

        self._render_customer_profile_core(dashboard)
        if self._customer_profile_tracking_advice is not None:
            self._customer_profile_tracking_advice.clear()
            for advice in dashboard.tracking_advice:
                QListWidgetItem(f"{advice.number}. {advice.text}", self._customer_profile_tracking_advice)
        if self._customer_profile_concerns is not None:
            self._customer_profile_concerns.setText(
                "　".join(f"「{item.value}」" for item in dashboard.concerns) or "暂无明确关注点"
            )
        self._render_customer_profile_pending(dashboard.pending_facts)
        if self._customer_profile_timeline is not None:
            self._customer_profile_timeline.clear()
            if dashboard.events:
                for event in dashboard.events:
                    when = _format_profile_time(event.event_at)
                    text = f"{when}　{event.icon} {event.label}\n{event.title}"
                    item = QListWidgetItem(text)
                    item.setToolTip(event.summary or event.title)
                    self._customer_profile_timeline.addItem(item)
            else:
                QListWidgetItem("暂无相关记录", self._customer_profile_timeline)
        if self._customer_profile_followups is not None:
            self._customer_profile_followups.clear()
            if dashboard.followups:
                for followup in dashboard.followups:
                    due = _format_profile_time(followup.due_at)
                    priority = "高" if followup.priority >= 70 else "普通"
                    source = {"explicit": "明确约定", "system_rule": "规则建议", "user_created": "人工创建"}.get(followup.source, followup.source)
                    item = QListWidgetItem(f"{due}　{followup.title}　优先级：{priority}　{source}", self._customer_profile_followups)
                    item.setToolTip(followup.reason or "本地回访任务")
            else:
                QListWidgetItem("暂无待回访", self._customer_profile_followups)

    def _render_customer_profile_core(self, dashboard: CustomerDashboard) -> None:
        table = self._customer_profile_core_table
        if table is None:
            return
        fields_by_key = {item.key: item for item in dashboard.core_fields}
        table.setRowCount(len(dashboard.archive_items))
        for row, archive in enumerate(dashboard.archive_items):
            field = fields_by_key.get(archive.key)
            table.setItem(row, 0, QTableWidgetItem(f"{archive.number}. {archive.label}"))
            table.setItem(row, 1, QTableWidgetItem(archive.value or "暂未确认"))
            actions = QWidget()
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(4)
            is_order = archive.key == "purchase_order"
            source = QPushButton("查看订单记录" if is_order else "查看来源")
            source.setEnabled(
                bool(dashboard.order_records)
                if is_order
                else bool(field and (field.fact_id or field.source_type == "remark"))
            )
            if is_order:
                source.clicked.connect(self._show_customer_order_history)
            elif field and field.fact_id:
                source.clicked.connect(lambda _checked=False, fact_id=field.fact_id: self._show_customer_fact_sources(fact_id))
            elif field and field.source_type == "remark":
                source.clicked.connect(
                    lambda _checked=False, label=field.label, value=field.value, raw=field.source_text:
                    self._show_customer_remark_source(label=label, value=value, source_text=raw)
                )
            edit = QPushButton("修改")
            edit.setEnabled(bool(field and field.fact_id) and not is_order)
            if field and field.fact_id:
                edit.clicked.connect(
                    lambda _checked=False, fact_id=field.fact_id, value=field.value: self._correct_customer_fact(fact_id, value)
                )
            action_layout.addWidget(source)
            action_layout.addWidget(edit)
            table.setCellWidget(row, 2, actions)

    def _show_customer_imported_archive(self) -> None:
        import sqlite3
        dashboard=self._customer_profile_dashboard
        if dashboard is None:
            QMessageBox.information(self,'聊天历史','请先打开已识别客户的档案。')
            return
        from whochat.services.imported_history import ImportedHistoryService
        service=ImportedHistoryService(self._services.db)
        try:
            archive=service.profile_archive(dashboard.account_scope_id,dashboard.contact_id)
        except ValueError:
            QMessageBox.warning(self,'聊天历史','客户归属发生变化，请重新打开客户档案。')
            return
        except sqlite3.DatabaseError:
            QMessageBox.warning(self,'聊天历史','本地历史数据库读取失败，未改动客户资料。请稍后重试；若仍失败，请保留当前窗口并反馈。')
            return
        if not archive['total']:
            QMessageBox.information(self,'聊天历史','尚未关联这位客户的导入历史；不代表没有聊过。可在总览使用“关联已有历史”核对归属。')
            return
        dialog=QDialog(self);dialog.setWindowTitle(f'本地聊天历史 · {dashboard.account_name} · {archive["name"]}')
        dialog.resize(920,600);layout=QVBoxLayout(dialog)
        note=QLabel(f'已关联 {archive["total"]} 条历史，已有结构化事实 {archive["structured_fact_count"]} 条。\n'
            +archive['status']+'。只查本地，不调用 AI。')
        note.setWordWrap(True);layout.addWidget(note)
        text=QTextEdit();text.setReadOnly(True);layout.addWidget(text)
        next_page=QPushButton('下一页（更早100条）');layout.addWidget(next_page)
        def render(page):
            lines=[]
            for record in page['records']:
                role={'me':'我','other':'客户'}.get(record['sender'],'系统/未知')
                body=record['text'] if record['message_type']=='text' else f'[{record["message_type"]}附件，未解读内容]'
                lines.append(f'{record["sent_at"]} · {role}\n{body}\n来源：{record["historical_message_id"]}')
            text.setPlainText('\n\n'.join(lines))
            next_page.setProperty('offset',page['offset']+100)
            next_page.setEnabled(page['offset']+len(page['records'])<page['total'])
        def advance():
            try:
                render(service.profile_archive(dashboard.account_scope_id,dashboard.contact_id,
                    offset=next_page.property('offset')))
            except ValueError:
                dialog.reject()
                QMessageBox.warning(self,'聊天历史','客户归属发生变化，请重新打开客户档案。')
            except sqlite3.DatabaseError:
                QMessageBox.warning(self,'聊天历史','本地历史读取失败，保留当前页；未改动客户资料。请稍后重试。')
        next_page.clicked.connect(advance);render(archive)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject);layout.addWidget(buttons);dialog.exec()

    def _show_customer_order_history(self) -> None:
        dashboard = self._customer_profile_dashboard
        if dashboard is None or not dashboard.order_records:
            QMessageBox.information(self, "订单记录", "当前没有可追溯的订单或报价记录。")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"订单记录 · {dashboard.display_name}")
        dialog.resize(1220, 560)
        layout = QVBoxLayout(dialog)
        note = QLabel("每行对应一条独立、可追溯的业务事件；缺失字段显示“暂未确认”，系统不会跨消息推算或拼成一笔订单。")
        note.setWordWrap(True)
        note.setObjectName("Muted")
        layout.addWidget(note)
        table = QTableWidget(len(dashboard.order_records), 10)
        table.setHorizontalHeaderLabels(
            ["日期", "产品", "规格", "数量", "单价", "金额", "状态", "付款", "发货", "来源"]
        )
        _compact_table(table)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for row, record in enumerate(dashboard.order_records):
            values = (
                _format_profile_time(record.event_at), record.product, record.specification,
                record.quantity, record.unit_price, record.amount, record.status,
                record.payment, record.shipping,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value or "暂未确认")
                cell.setToolTip(value or "暂未确认")
                table.setItem(row, column, cell)
            source = QPushButton("查看来源")
            source.setEnabled(bool(record.sources))
            source.clicked.connect(
                lambda _checked=False, selected=record: self._show_order_record_sources(selected)
            )
            table.setCellWidget(row, 9, source)
        layout.addWidget(table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_order_record_sources(self, record: OrderRecord) -> None:
        lines: list[str] = []
        for index, source in enumerate(record.sources, 1):
            sender = {"other": "客户", "me": "我"}.get(source.sender, source.sender)
            source_id = source.historical_message_id or source.message_id or "来源ID缺失"
            lines.append(
                f"{index}. {_format_profile_time(source.observed_at)} · {sender}\n"
                f"{source.source_text or '原文暂缺'}\n来源ID：{source_id}"
            )
        QMessageBox.information(
            self,
            "订单记录来源",
            "\n\n".join(lines) if lines else "当前事件没有合法来源证据。",
        )

    def _render_customer_profile_pending(self, facts: tuple[DashboardFact, ...]) -> None:
        table = self._customer_profile_pending_table
        if table is None:
            return
        table.setRowCount(len(facts))
        for row, fact in enumerate(facts):
            table.setItem(row, 0, QTableWidgetItem(fact.label))
            table.setItem(row, 1, QTableWidgetItem(fact.value))
            table.setItem(row, 2, QTableWidgetItem("冲突" if fact.status == "conflicting" else "待确认"))
            actions = QWidget()
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(3)
            confirm = QPushButton("确认")
            confirm.clicked.connect(lambda _checked=False, fact_id=fact.id: self._confirm_customer_fact(fact_id))
            edit = QPushButton("修改")
            edit.clicked.connect(lambda _checked=False, fact_id=fact.id, value=fact.value: self._correct_customer_fact(fact_id, value))
            invalid = QPushButton("无效")
            invalid.clicked.connect(lambda _checked=False, fact_id=fact.id: self._invalidate_customer_fact(fact_id))
            source = QPushButton("来源")
            source.clicked.connect(lambda _checked=False, fact_id=fact.id: self._show_customer_fact_sources(fact_id))
            for button in (confirm, edit, invalid, source):
                action_layout.addWidget(button)
            table.setCellWidget(row, 3, actions)

    def _render_customer_profile_empty(self, message: str, *, keep_identity: bool = False) -> None:
        self._customer_profile_dashboard = None
        self._customer_profile_revision = ""
        if not keep_identity:
            if self._customer_profile_name is not None:
                self._customer_profile_name.setText("等待识别当前客户")
            if self._customer_profile_account is not None:
                self._customer_profile_account.setText("工作微信：—")
        if self._customer_profile_summary is not None:
            self._customer_profile_summary.setText("暂无客户档案")
        if self._customer_profile_state is not None:
            self._customer_profile_state.setText(message)
        if self._customer_profile_context is not None:
            self._customer_profile_context.blockSignals(True)
            self._customer_profile_context.clear()
            self._customer_profile_context.addItem("暂无种植项目", None)
            self._customer_profile_context.setEnabled(False)
            self._customer_profile_context.blockSignals(False)
        for table in (self._customer_profile_core_table, self._customer_profile_pending_table):
            if table is not None:
                table.setRowCount(0)
        if self._customer_profile_concerns is not None:
            self._customer_profile_concerns.setText("暂无明确关注点")
        if self._customer_profile_tracking_advice is not None:
            self._customer_profile_tracking_advice.clear()
            QListWidgetItem("1. 等待稳定识别客户后加载本地跟踪建议", self._customer_profile_tracking_advice)
        for listing, empty in (
            (self._customer_profile_timeline, "暂无相关记录"),
            (self._customer_profile_followups, "暂无待回访"),
        ):
            if listing is not None:
                listing.clear()
                QListWidgetItem(empty, listing)

    def _render_customer_profile_error(self) -> None:
        self._render_customer_profile_empty("客户档案暂时无法加载；OCR、新消息和 Reply AI 不受影响。", keep_identity=True)

    def _on_customer_profile_context_changed(self, _index: int) -> None:
        self._customer_profile_revision = ""
        self._customer_profile_event_limit = 15
        self._refresh_customer_profile_page(force=True)

    def _on_customer_profile_timeline_filter_changed(self, _index: int) -> None:
        self._customer_profile_event_limit = 15
        self._customer_profile_revision = ""
        self._refresh_customer_profile_page(force=True)

    def _load_more_customer_profile_events(self) -> None:
        self._customer_profile_event_limit += 15
        self._customer_profile_revision = ""
        self._refresh_customer_profile_page(force=True)

    def _show_customer_fact_sources(self, fact_id: str) -> None:
        try:
            sources = self._services.customer_profile_service.get_fact_sources(fact_id)
        except Exception as exc:
            QMessageBox.warning(self, "来源读取失败", f"暂时无法读取这条资料的来源：\n{type(exc).__name__}")
            return
        sender_labels = {"other": "客户", "me": "我", "manual": "人工操作"}
        lines = []
        for source in sources:
            lines.append(
                f"{_format_profile_time(source.observed_at)}\n"
                f"{sender_labels.get(source.sender, source.sender)}：\n{source.source_text}"
            )
        dialog = QDialog(self)
        dialog.setWindowTitle("资料来源")
        dialog.resize(620, 420)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n\n".join(lines) if lines else "暂无可追溯来源")
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_customer_remark_source(self, *, label: str, value: str, source_text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("资料来源")
        dialog.resize(620, 300)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(
            "来源类型：微信备注\n"
            f"资料：{label}\n"
            f"当前值：{value}\n\n"
            f"微信备注原文：\n{source_text or '原文暂缺'}\n\n"
            "说明：该值仅作为低优先级本地展示兜底，不会自动写成高置信客户事实；"
            "人工修改或已有确认资料始终优先。"
        )
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _confirm_customer_fact(self, fact_id: str) -> None:
        reason, ok = QInputDialog.getText(self, "确认资料", "确认说明（可留空）：")
        if not ok:
            return
        self._apply_customer_fact_edit(
            "confirm",
            lambda: self._services.customer_profile_service.confirm_fact(fact_id, reason=reason),
        )

    def _correct_customer_fact(self, fact_id: str, current_value: str) -> None:
        value, ok = QInputDialog.getText(self, "修改/纠正资料", "新的正确值：", text=current_value)
        if not ok or not value.strip():
            return
        reason, ok = QInputDialog.getText(self, "修改原因", "为什么修改（可留空）：")
        if not ok:
            return
        self._apply_customer_fact_edit(
            "correct",
            lambda: self._services.customer_profile_service.correct_fact(
                fact_id,
                value_text=value,
                reason=reason,
            ),
        )

    def _invalidate_customer_fact(self, fact_id: str) -> None:
        answer = QMessageBox.question(
            self,
            "标记无效",
            "这会把当前资料标记为无效，但会保留原值、消息来源和审计记录。是否继续？",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        reason, ok = QInputDialog.getText(self, "无效原因", "为什么无效（可留空）：")
        if not ok:
            return
        self._apply_customer_fact_edit(
            "invalidate",
            lambda: self._services.customer_profile_service.invalidate_fact(fact_id, reason=reason),
        )

    def _apply_customer_fact_edit(self, operation: str, callback) -> None:
        dashboard = self._customer_profile_dashboard
        try:
            callback()
        except Exception as exc:
            self.append_log(f"customer_memory_{operation}_failed: {type(exc).__name__}", "warning")
            QMessageBox.warning(self, "档案修改失败", f"没有写入任何更改：\n{exc}")
            return
        self._customer_profile_revision = ""
        self._invalidate_proactive_context(
            f"customer_fact_{operation}",
            dashboard.contact_id if dashboard is not None else None,
        )
        self._refresh_customer_profile_page(force=True)
        self.statusBar().showMessage("客户档案已更新，历史值和来源已保留", 3500)
        self.append_log(f"customer_memory_{operation}_completed")

    def _reload_contact_list(self, select_contact_id: str | None = None) -> list[Contact]:
        if self._contact_list is None:
            return []
        self._contact_list.blockSignals(True)
        self._contact_list.clear()
        contacts = self._services.contacts.list_recent()
        selected_row = -1
        if contacts:
            for index, contact in enumerate(contacts):
                item = QListWidgetItem(f"{_contact_display_label(contact)} · {contact.status.value} · {contact.conversation_type.value}")
                item.setData(Qt.ItemDataRole.UserRole, contact.id)
                item.setToolTip(f"{contact.platform} · {contact.display_name}")
                self._contact_list.addItem(item)
                if contact.id == select_contact_id:
                    selected_row = index
        else:
            QListWidgetItem("暂无聊天对象\n打开私聊或群聊并确认后会显示在这里", self._contact_list)
        self._contact_list.blockSignals(False)
        if selected_row >= 0:
            self._contact_list.setCurrentRow(selected_row)
        return contacts

    def _build_strategies_page(self) -> QWidget:
        page = self._panel("分组策略", "预设只是模板，用户可以为每个分组定义自己的目标、语气和禁忌。")
        filters = QHBoxLayout()
        search = QLineEdit()
        search.setPlaceholderText("搜索分组、目标、语气或禁忌")
        search.textChanged.connect(lambda _value: self._reload_strategy_table())
        show_archived = QCheckBox("显示归档")
        show_archived.stateChanged.connect(lambda _value: self._reload_strategy_table())
        self._strategy_search = search
        self._strategy_show_archived = show_archived
        filters.addWidget(search, 1)
        filters.addWidget(show_archived)
        page.layout().addLayout(filters)

        table = QTableWidget(0, 6)
        self._strategy_table = table
        table.setHorizontalHeaderLabels(["分组", "目标", "语气", "保护", "状态", "使用"])
        _compact_table(table)
        table.itemDoubleClicked.connect(lambda _item: self._edit_selected_strategy())
        page.layout().addWidget(table)
        actions = QHBoxLayout()
        add = QPushButton("新增分组")
        add.clicked.connect(self._add_strategy)
        duplicate = QPushButton("复制为新策略")
        duplicate.clicked.connect(self._duplicate_selected_strategy)
        edit = QPushButton("编辑目标")
        edit.clicked.connect(self._edit_selected_strategy)
        archive = QPushButton("归档/恢复")
        archive.clicked.connect(self._toggle_selected_strategy_archive)
        actions.addWidget(add)
        actions.addWidget(duplicate)
        actions.addWidget(edit)
        actions.addWidget(archive)
        actions.addStretch(1)
        page.layout().addLayout(actions)
        self._reload_strategy_table()
        return self._build_page_shell(page)

    def _build_memories_page(self) -> QWidget:
        page = self._panel("记忆整理", "AI 提取的新记忆会先进入待确认区，避免 OCR 错误污染长期画像。")
        tabs = QTabWidget()
        self._memory_tabs = tabs
        tabs.addTab(self._build_memory_review_tab(MemoryStatus.PENDING), "待确认")
        tabs.addTab(self._build_memory_review_tab(MemoryStatus.CONFIRMED), "长期画像")
        tabs.addTab(self._build_memory_review_tab(MemoryStatus.REJECTED), "已拒绝")
        page.layout().addWidget(tabs)
        return self._build_page_shell(page)

    def _build_settings_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        ai = self._panel("自定义 AI", "支持 OpenAI 和第三方 GPT 兼容接口。API Key 只用于本地配置，不应写入日志。")
        form = QGridLayout()
        provider = QComboBox()
        provider.addItems(["OpenAI Compatible", "OpenAI", "Local Model", "Disabled"])
        provider.setCurrentText(self._config.ai.provider)
        base_url = QLineEdit(self._config.ai.base_url)
        base_url.setPlaceholderText("例如：https://api.openai.com/v1")
        base_url.setToolTip(
            "填服务商给的 API 根地址即可，程序会自动拼接 /chat/completions。"
            "OpenAI 一般填到 /v1；DeepSeek 直接填 https://api.deepseek.com（末尾不要加 /v1）。"
        )
        model = QLineEdit(self._config.ai.model)
        api_key = QLineEdit()
        api_key.setText(self._config.ai.api_key)
        api_key.setEchoMode(QLineEdit.EchoMode.Password)
        temperature = QDoubleSpinBox()
        temperature.setRange(0.0, 2.0)
        temperature.setSingleStep(0.1)
        temperature.setValue(self._config.ai.temperature)
        max_context = QSpinBox()
        max_context.setRange(1000, 200000)
        max_context.setValue(self._config.ai.context_tokens)
        timeout = QSpinBox()
        timeout.setRange(5, 300)
        timeout.setValue(self._config.ai.timeout_seconds)
        cooldown = QSpinBox()
        cooldown.setRange(0, 300)
        cooldown.setValue(self._config.ai.request_cooldown_seconds)
        dedupe = QSpinBox()
        dedupe.setRange(0, 1440)
        dedupe.setValue(self._config.ai.dedupe_context_minutes)
        daily_limit = QSpinBox()
        daily_limit.setRange(0, 10000)
        daily_limit.setValue(self._config.ai.max_daily_cloud_requests)
        failure_threshold = QSpinBox()
        failure_threshold.setRange(1, 20)
        failure_threshold.setValue(self._config.ai.failure_backoff_threshold)
        failure_backoff = QSpinBox()
        failure_backoff.setRange(0, 240)
        failure_backoff.setValue(self._config.ai.failure_backoff_minutes)
        health_label = QLabel(self._services.reply_generator.provider_health_summary())
        health_label.setObjectName("TinyMuted")
        self._ai_health_label = health_label
        action_status = QLabel("未测试")
        action_status.setObjectName("TinyMuted")
        self._ai_action_status = action_status
        self._ai_provider = provider
        self._ai_base_url = base_url
        self._ai_model = model
        self._ai_api_key = api_key
        self._ai_temperature = temperature
        self._ai_context_tokens = max_context
        self._ai_timeout = timeout
        self._ai_request_cooldown = cooldown
        self._ai_dedupe_minutes = dedupe
        self._ai_daily_limit = daily_limit
        self._ai_failure_threshold = failure_threshold
        self._ai_failure_backoff = failure_backoff
        for row, (label, widget) in enumerate([
            ("服务类型", provider),
            ("接口地址", base_url),
            ("模型名称", model),
            ("API Key", api_key),
            ("温度", temperature),
            ("上下文上限", max_context),
            ("超时时间（秒）", timeout),
            ("云端冷却（秒）", cooldown),
            ("重复上下文（分钟）", dedupe),
            ("每日云端上限", daily_limit),
            ("失败退避阈值", failure_threshold),
            ("失败退避（分钟）", failure_backoff),
            ("健康状态", health_label),
        ]):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
        ai.layout().addLayout(form)
        ai_actions = QHBoxLayout()
        test_ai = QPushButton("测试连接")
        test_ai.clicked.connect(self._test_ai_settings)
        ai_actions.addWidget(test_ai)
        reset_health = QPushButton("恢复健康状态")
        reset_health.clicked.connect(self._reset_ai_provider_health)
        ai_actions.addWidget(reset_health)
        save_ai = QPushButton("保存设置")
        save_ai.setObjectName("PrimaryButton")
        save_ai.clicked.connect(self._save_ai_settings)
        ai_actions.addWidget(save_ai)
        ai_actions.addWidget(action_status)
        ai_actions.addStretch(1)
        ai.layout().addLayout(ai_actions)
        layout.addWidget(ai)

        privacy = self._panel("隐私与数据", "默认本地保存，用户可以关闭截图保存和长期记忆。")
        privacy_items = [
            ("启用长期记忆", self._config.privacy.enable_long_term_memory, "_privacy_long_memory"),
            ("保存截图样本用于调试", self._config.privacy.save_debug_screenshots, "_privacy_debug_screenshots"),
            ("云端请求前裁剪上下文", self._config.privacy.trim_context_for_cloud, "_privacy_trim_cloud"),
            ("云端请求前预览并确认", self._config.privacy.require_cloud_prompt_review, "_privacy_cloud_review"),
            ("手动回复保护默认不生成可复制回复", self._config.privacy.manual_protection_blocks_replies, "_privacy_manual_blocks"),
        ]
        for text, checked, attr in privacy_items:
            checkbox = QCheckBox(text)
            checkbox.setChecked(checked)
            setattr(self, attr, checkbox)
            privacy.layout().addWidget(checkbox)
        retention_form = QGridLayout()
        self._privacy_log_retention = _days_spin(self._config.privacy.diagnostic_log_retention_days)
        self._privacy_debug_retention = _days_spin(self._config.privacy.debug_sample_retention_days)
        self._privacy_capture_retention = _days_spin(self._config.privacy.capture_retention_days)
        self._privacy_calibration_retention = _days_spin(self._config.privacy.calibration_retention_days)
        for row, (label, widget) in enumerate([
            ("日志保留", self._privacy_log_retention),
            ("调试样本", self._privacy_debug_retention),
            ("截图缓存", self._privacy_capture_retention),
            ("校准样本", self._privacy_calibration_retention),
        ]):
            retention_form.addWidget(QLabel(label), row // 2, (row % 2) * 2)
            retention_form.addWidget(widget, row // 2, (row % 2) * 2 + 1)
        privacy.layout().addLayout(retention_form)
        governance_actions = QHBoxLayout()
        export_all = QPushButton("全局导出")
        export_all.clicked.connect(self._export_all_data)
        cleanup_local = QPushButton("立即清理")
        cleanup_local.clicked.connect(self._run_retention_cleanup)
        clear_all = QPushButton("清空全部内容")
        clear_all.setObjectName("DangerButton")
        clear_all.clicked.connect(self._clear_all_content)
        governance_actions.addWidget(export_all)
        governance_actions.addWidget(cleanup_local)
        governance_actions.addWidget(clear_all)
        governance_actions.addStretch(1)
        privacy.layout().addLayout(governance_actions)
        layout.addWidget(privacy)

        ocr = self._panel("OCR 识别", "默认预览引擎只用于验证坐标流；真实本地识别可选择 RapidOCR 或 PaddleOCR。")
        ocr_form = QGridLayout()
        ocr_provider = QComboBox()
        ocr_provider.addItems(["Preview Fixture", "RapidOCR", "PaddleOCR"])
        ocr_provider.setCurrentText(self._config.ocr.provider)
        ocr_language = QLineEdit(self._config.ocr.language)
        ocr_min_confidence = QDoubleSpinBox()
        ocr_min_confidence.setRange(0.0, 1.0)
        ocr_min_confidence.setSingleStep(0.05)
        ocr_min_confidence.setValue(self._config.ocr.min_confidence)
        ocr_use_gpu = QCheckBox("启用 GPU（取决于 OCR 库和本机环境）")
        ocr_use_gpu.setChecked(self._config.ocr.use_gpu)
        self._ocr_provider = ocr_provider
        self._ocr_language = ocr_language
        self._ocr_min_confidence = ocr_min_confidence
        self._ocr_use_gpu = ocr_use_gpu
        for row, (label, widget) in enumerate([
            ("识别引擎", ocr_provider),
            ("语言", ocr_language),
            ("最低置信度", ocr_min_confidence),
            ("硬件", ocr_use_gpu),
        ]):
            ocr_form.addWidget(QLabel(label), row, 0)
            ocr_form.addWidget(widget, row, 1)
        ocr.layout().addLayout(ocr_form)
        layout.addWidget(ocr)

        capture = self._panel("采集与防抖", "控制 OCR 和 AI 请求频率，避免滚动时产生大量请求。")
        self._capture_auto_enabled = QCheckBox("自动 OCR 采集")
        self._capture_auto_enabled.setChecked(False if self._manual_generation_enabled else self._config.capture.auto_capture_enabled)
        self._capture_auto_enabled.setToolTip("开启后，目标窗口命中并稳定时会自动截图、OCR 并确认聊天页。")
        if self._manual_generation_enabled:
            self._capture_auto_enabled.setText("手动按钮模式：自动 OCR 已关闭")
            self._capture_auto_enabled.setEnabled(False)
            self._capture_auto_enabled.setToolTip("只有点击「生成建议」才读取本次聊天；窗口移动和新消息不会自动启动。")
        capture.layout().addWidget(self._capture_auto_enabled)
        self._capture_foreground_only = QCheckBox("仅采集当前前台目标窗口")
        self._capture_foreground_only.setChecked(self._config.capture.foreground_only)
        self._capture_foreground_only.setToolTip("关闭后进入测试模式：允许采集后台未最小化目标窗口，但截图可能被其他窗口遮挡。")
        capture.layout().addWidget(self._capture_foreground_only)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(100, 3000)
        slider.setValue(self._config.capture.scroll_debounce_ms)
        self._capture_debounce = slider
        self._capture_debounce_label = QLabel(f"滚动停止防抖：{slider.value()} ms")
        slider.valueChanged.connect(lambda value: self._capture_debounce_label.setText(f"滚动停止防抖：{value} ms"))
        capture.layout().addWidget(self._capture_debounce_label)
        capture.layout().addWidget(slider)
        ocr_interval = QSpinBox()
        ocr_interval.setRange(2500, 60000)
        ocr_interval.setSingleStep(1000)
        ocr_interval.setSuffix(" ms")
        ocr_interval.setValue(self._config.capture.ocr_min_interval_ms)
        self._capture_ocr_interval = ocr_interval
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("自动 OCR 最小间隔"))
        interval_row.addWidget(ocr_interval)
        interval_row.addStretch(1)
        capture.layout().addLayout(interval_row)
        self._capture_pause_unknown = QCheckBox("页面类型未知时暂停 AI 请求")
        self._capture_pause_unknown.setChecked(self._config.capture.pause_ai_on_unknown_page)
        self._capture_block_unconfirmed = QCheckBox("聊天对象识别后立即可用；忽略对象会阻断")
        self._capture_block_unconfirmed.setChecked(self._config.capture.block_memory_for_unconfirmed_contact)
        self._capture_block_unconfirmed.setToolTip("自动 OCR 识别到标题后会立即建立聊天对象；用户可后续确认、合并或设为忽略。")
        capture.layout().addWidget(self._capture_pause_unknown)
        capture.layout().addWidget(self._capture_block_unconfirmed)
        layout.addWidget(capture)

        account_boundary = self._panel(
            "微信账号边界",
            "每个微信登录账号使用独立本地空间。首次识别到新的微信窗口后，只需为该窗口绑定一次本地名称。",
        )
        self._account_scope_label = QLabel(self._format_account_scope_state())
        self._account_scope_label.setObjectName("TinyMuted")
        self._account_scope_label.setWordWrap(True)
        account_boundary.layout().addWidget(self._account_scope_label)
        account_actions = QHBoxLayout()
        bind_account = QPushButton("绑定当前微信账号")
        bind_account.clicked.connect(self._bind_current_account_scope)
        account_actions.addWidget(bind_account)
        account_actions.addStretch(1)
        account_boundary.layout().addLayout(account_actions)
        layout.addWidget(account_boundary)

        floating = self._panel("悬浮窗", "控制贴边位置、透明度和可见回复数量。")
        floating_form = QGridLayout()
        placement = QComboBox()
        placement.addItems(["auto", "bottom", "top", "right", "left"])
        placement.setCurrentText(self._config.floating.placement_preference)
        opacity = QSpinBox()
        opacity.setRange(70, 100)
        opacity.setSuffix("%")
        opacity.setValue(self._config.floating.opacity_percent)
        suggestion_count = QSpinBox()
        suggestion_count.setRange(2, 2)
        suggestion_count.setValue(2)
        suggestion_count.setEnabled(False)
        self._floating_placement = placement
        self._floating_opacity = opacity
        self._floating_suggestion_count = suggestion_count
        for row, (label, widget) in enumerate([
            ("贴靠优先", placement),
            ("透明度", opacity),
            ("回复数量（固定）", suggestion_count),
        ]):
            floating_form.addWidget(QLabel(label), row, 0)
            floating_form.addWidget(widget, row, 1)
        floating.layout().addLayout(floating_form)
        layout.addWidget(floating)

        targets = self._panel("目标应用", None)
        target_grid = QGridLayout()
        self._target_grid = target_grid
        self._refresh_target_grid()
        target_grid.setColumnStretch(2, 1)
        target_grid.setColumnStretch(3, 1)
        target_grid.setColumnStretch(4, 1)
        targets.layout().addLayout(target_grid)
        target_actions = QHBoxLayout()
        add_target = QPushButton("新增应用")
        add_target.clicked.connect(self._add_target_app)
        target_actions.addWidget(add_target)
        target_actions.addStretch(1)
        targets.layout().addLayout(target_actions)
        layout.addWidget(targets)

        tools = self._panel("排障工具", "总览显示当前运行链路；这里保留主动执行的维护操作。")
        tool_actions = QHBoxLayout()
        copy_logs = QPushButton("复制运行日志")
        copy_logs.setToolTip("复制已脱敏的运行链路、OCR、AI 和窗口信息")
        copy_logs.clicked.connect(self._copy_diagnostics_bundle)
        save_sample = QPushButton("保存调试样本")
        save_sample.setToolTip("保存当前采集结果和脱敏诊断信息，便于复现识别问题")
        save_sample.clicked.connect(self._save_debug_sample)
        run_pipeline = QPushButton("立即采集")
        run_pipeline.setObjectName("PrimaryButton")
        run_pipeline.clicked.connect(self._run_capture_pipeline)
        recalibrate = QPushButton("重新校准区域")
        recalibrate.clicked.connect(self._open_calibration_dialog)
        for button in (copy_logs, save_sample, run_pipeline, recalibrate):
            tool_actions.addWidget(button)
        tool_actions.addStretch(1)
        tools.layout().addLayout(tool_actions)
        layout.addWidget(tools)

        layout.addStretch(1)
        scroll.setWidget(body)
        return self._build_page_shell(scroll)

    def _build_diagnostics_page(self) -> QWidget:
        page = self._panel("诊断控制台", "日志保持简洁、精准，可用于定位窗口识别、OCR 和 AI 请求问题。")
        runtime = QTextEdit()
        runtime.setReadOnly(True)
        runtime.setMinimumHeight(112)
        self._runtime_text = runtime
        runtime.setText(self._format_runtime_state(self._services.runtime.state))
        page.layout().addWidget(runtime)

        environment = QTextEdit()
        environment.setReadOnly(True)
        environment.setMinimumHeight(86)
        environment.setMaximumHeight(130)
        self._environment_text = environment
        environment.setText(self._format_environment_checks())
        page.layout().addWidget(environment)

        window_match = QTextEdit()
        window_match.setReadOnly(True)
        window_match.setMinimumHeight(76)
        window_match.setMaximumHeight(120)
        self._window_match_text = window_match
        window_match.setText(self._format_window_match_diagnostics())
        window_match_toolbar = QHBoxLayout()
        window_match_toolbar.addWidget(QLabel("窗口诊断"))
        window_match_toolbar.addStretch(1)
        refresh_window = QPushButton("刷新窗口")
        refresh_window.clicked.connect(self._refresh_window_match_diagnostics)
        window_match_toolbar.addWidget(refresh_window)
        copy_window = QPushButton("复制窗口")
        copy_window.clicked.connect(self._copy_window_match_diagnostics)
        window_match_toolbar.addWidget(copy_window)
        page.layout().addLayout(window_match_toolbar)
        page.layout().addWidget(window_match)

        log = QTextEdit()
        log.setReadOnly(True)
        self._log_text = log
        log_toolbar = QHBoxLayout()
        level_filter = QComboBox()
        level_filter.addItems(["全部", "debug", "info", "warning", "error"])
        level_filter.currentTextChanged.connect(lambda _value: self._refresh_ui_log_text())
        self._log_level_filter = level_filter
        log_toolbar.addWidget(QLabel("日志级别"))
        log_toolbar.addWidget(level_filter)
        log_toolbar.addStretch(1)
        page.layout().addLayout(log_toolbar)
        self._refresh_ui_log_text()
        self._pending_logs.clear()
        page.layout().addWidget(log)

        diagnostics_files = QTextEdit()
        diagnostics_files.setReadOnly(True)
        diagnostics_files.setMinimumHeight(70)
        diagnostics_files.setMaximumHeight(120)
        self._diagnostics_file_text = diagnostics_files
        diagnostics_files.setText(self._format_diagnostics_files())
        page.layout().addWidget(diagnostics_files)

        generation_log = QTextEdit()
        generation_log.setReadOnly(True)
        generation_log.setMinimumHeight(78)
        self._generation_log_text = generation_log
        self._refresh_generation_log_text()
        page.layout().addWidget(generation_log)

        actions = QHBoxLayout()
        copy_logs = QPushButton("复制日志")
        copy_logs.clicked.connect(self._copy_diagnostics_bundle)
        actions.addWidget(copy_logs)
        save_sample = QPushButton("保存调试样本")
        save_sample.clicked.connect(self._save_debug_sample)
        actions.addWidget(save_sample)
        run_pipeline = QPushButton("运行采集管线")
        run_pipeline.clicked.connect(self._run_capture_pipeline)
        actions.addWidget(run_pipeline)
        recalibrate = QPushButton("重新校准区域")
        recalibrate.clicked.connect(self._open_calibration_dialog)
        actions.addWidget(recalibrate)
        actions.addStretch(1)
        page.layout().addLayout(actions)
        return self._build_page_shell(page)

    def _format_environment_checks(self) -> str:
        return self._services.environment.format_text()

    def _format_window_match_diagnostics(self) -> str:
        targets = [target for target in self._config.targets if target.enabled]
        target_line = "enabled_targets=" + ",".join(
            f"{target.app_id}:{target.label}"
            f"(process={','.join(target.process_names) or '-'};"
            f"title={','.join(target.title_keywords) or '-'};"
            f"exclude={','.join(target.exclude_title_keywords) or '-'})"
            for target in targets
        )
        candidates = diagnose_target_windows(self._config.targets, limit=12)
        foreground = foreground_window_handle()
        if not candidates:
            return "\n".join(
                [
                    target_line,
                    f"foreground_hwnd={foreground or '-'}",
                    "related_windows=0",
                    "action=未发现相关窗口。请打开已启用的聊天应用，或在设置中补充进程名/标题关键词。",
                ]
            )
        matched_foreground = next((item for item in candidates if item.foreground and item.matched), None)
        foreground_excluded = next((item for item in candidates if item.foreground and item.reason == "excluded_by_title"), None)
        matched_background = [item for item in candidates if item.matched and not item.foreground]
        lines = [
            target_line,
            f"foreground_hwnd={foreground or '-'}",
            f"related_windows={len(candidates)}",
            _window_match_action(matched_foreground, foreground_excluded, matched_background),
        ]
        for item in candidates:
            target = item.target_app or "-"
            label = item.app_label or "-"
            process = item.process_name or "-"
            lines.append(
                f"hwnd={item.hwnd} foreground={item.foreground} matched={item.matched} "
                f"process={process} target={target}/{label} reason={item.reason} title={item.title}"
            )
        return "\n".join(lines)

    def _refresh_window_match_diagnostics(self) -> None:
        if self._window_match_text is not None:
            self._window_match_text.setText(self._format_window_match_diagnostics())
        if self._runtime_text is not None:
            self._runtime_text.setText(self._format_runtime_state(self._services.runtime.state))
        self.statusBar().showMessage("窗口诊断已刷新", 2500)
        self.append_log("window_match_diagnostics_refreshed")

    def _copy_window_match_diagnostics(self) -> None:
        text = redact_diagnostics_text(self._format_window_match_diagnostics())
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("窗口诊断已复制", 2500)
        self.append_log("window_match_diagnostics_copied")

    def _refresh_ui_log_text(self) -> None:
        if self._log_text is None:
            return
        selected = self._log_level_filter.currentText() if self._log_level_filter else "全部"
        logs = list(reversed(self._services.logs.tail(120)))
        if selected != "全部":
            logs = [item for item in logs if item.level == selected]
        text = "\n".join(_format_app_log(item) for item in logs)
        if selected == "全部" and self._pending_logs:
            text = "\n".join([text, *self._pending_logs]).strip()
        self._log_text.setText(text or f"[{selected}] 暂无")

    def _format_diagnostics_files(self) -> str:
        crash_log = diagnostics_log_path("crash")
        ocr_log = diagnostics_log_path("ocr_worker")
        ai_log = diagnostics_log_path("ai_provider")
        lines = [
            f"crash_log={crash_log}",
            f"ocr_worker_log={ocr_log}",
            f"ai_provider_log={ai_log}",
            "ocr_worker_tail:",
            *_tail_lines(ocr_log, 5),
            "ai_provider_tail:",
            *_tail_lines(ai_log, 5),
            "capture_samples:",
            *self._format_capture_performance_summary(20),
            *self._format_capture_sample_lines(5),
        ]
        return "\n".join(lines)

    def _format_capture_performance_summary(self, limit: int = 20) -> list[str]:
        samples = [
            item for item in self._services.capture_samples.tail(limit)
            if item.total_elapsed_ms is not None
        ]
        if not samples:
            return ["capture_perf=暂无"]
        title_values = [item.title_ocr_elapsed_ms for item in samples if item.title_ocr_elapsed_ms is not None]
        content_values = [item.content_ocr_elapsed_ms for item in samples if item.content_ocr_elapsed_ms is not None]
        total_values = [item.total_elapsed_ms for item in samples if item.total_elapsed_ms is not None]
        slowest = max(samples, key=lambda item: item.total_elapsed_ms or 0)
        avg_total = _avg_number(total_values)
        status, action = _capture_perf_status(avg_total)
        return [
            (
                f"capture_perf=status:{status} count:{len(samples)} "
                f"avg_title_ms:{_avg_int(title_values)} "
                f"avg_content_ms:{_avg_int(content_values)} "
                f"avg_total_ms:{_avg_int(total_values)} "
                f"slowest_job:{slowest.job_id} slowest_total_ms:{slowest.total_elapsed_ms} "
                f"action:{action}"
            )
        ]

    def _format_capture_sample_lines(self, limit: int = 5) -> list[str]:
        samples = self._services.capture_samples.tail(limit)
        if not samples:
            return ["暂无"]
        return [
            (
                f"job={item.job_id} app={item.target_app}/{item.app_label} page={item.page_type} "
                f"messages={item.message_count} title_ms={item.title_ocr_elapsed_ms if item.title_ocr_elapsed_ms is not None else '-'} "
                f"content_ms={item.content_ocr_elapsed_ms if item.content_ocr_elapsed_ms is not None else '-'} "
                f"total_ms={item.total_elapsed_ms if item.total_elapsed_ms is not None else '-'}"
            )
            for item in samples
        ]

    def _diagnostics_bundle_text(self) -> str:
        state = self._services.runtime.state
        sections = [
            "# runtime",
            self._format_runtime_state(state),
            "",
            "# ui_logs",
            "\n".join(f"[{item.level}] {item.ts} {item.module}.{item.event}: {item.message}" for item in reversed(self._services.logs.tail(80))) or "暂无",
            "",
            "# environment",
            self._format_environment_checks(),
            "",
            "# window_matching",
            self._format_window_match_diagnostics(),
            "",
            "# generation_logs",
            self._format_generation_log_lines(20),
            "",
            "# reply_feedback",
            self._format_reply_feedback_lines(20),
            "",
            "# capture_samples",
            "\n".join(self._format_capture_sample_lines(10)),
            "",
            "# files",
            self._format_diagnostics_files(),
        ]
        return "\n".join(sections)

    def _copy_diagnostics_bundle(self) -> None:
        text = redact_diagnostics_text(self._diagnostics_bundle_text())
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("诊断日志已复制", 3000)
        self.append_log("diagnostics_bundle_copied")

    def _save_debug_sample(self) -> None:
        output_dir = app_data_dir() / "debug_samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now_iso().replace(":", "-")
        sample_dir = output_dir / f"sample-{stamp}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        state = self._services.runtime.state
        last_result = self._services.pipeline.last_result
        last_title = self._services.pipeline.last_title_result
        payload = {
            "export": {
                "app": "WhoChat",
                "schema_version": 1,
                "exported_at": utc_now_iso(),
                "scope": "diagnostics",
            },
            "runtime": self._format_runtime_state(state),
            "environment": [asdict(item) for item in self._services.environment.checks()],
            "window_matching": [asdict(item) for item in diagnose_target_windows(self._config.targets, limit=20)],
            "pipeline": {
                "last_status": self._services.pipeline.last_status,
                "last_discard_reason": self._services.pipeline.last_discard_reason,
                "last_title_job_id": last_title.job_id if last_title else None,
                "last_title_target_app": last_title.target_app if last_title else None,
                "last_title_app_label": last_title.app_label if last_title else None,
                "last_title_image": str(last_title.title_ocr_image_path) if last_title else None,
                "last_title_crop_rect": last_title.title_crop_rect.as_tuple() if last_title and last_title.title_crop_rect else None,
                "last_title_elapsed_ms": last_title.elapsed_ms if last_title else None,
                "last_title_ocr_boxes": [asdict(box) for box in last_title.ocr_result.boxes] if last_title else [],
                "last_result_job_id": last_result.job_id if last_result else None,
                "last_result_target_app": last_result.target_app if last_result else None,
                "last_result_app_label": last_result.app_label if last_result else None,
                "last_result_image": str(last_result.image_path) if last_result else None,
                "last_result_ocr_image": str(last_result.ocr_image_path) if last_result else None,
                "last_result_crop_rect": last_result.crop_rect.as_tuple() if last_result and last_result.crop_rect else None,
                "last_result_title_elapsed_ms": last_result.title_ocr_elapsed_ms if last_result else None,
                "last_result_content_elapsed_ms": last_result.content_ocr_elapsed_ms if last_result else None,
                "last_result_total_elapsed_ms": last_result.total_elapsed_ms if last_result else None,
                "last_result_page": last_result.page.page_type.value if last_result else None,
                "last_result_page_confidence": last_result.page.confidence if last_result else None,
                "last_result_messages": len(last_result.messages) if last_result else 0,
                "last_result_layout": asdict(last_result.layout) if last_result else None,
                "last_result_ocr_boxes": [asdict(box) for box in last_result.ocr_result.boxes] if last_result else [],
                "last_result_parsed_messages": [asdict(message) for message in last_result.messages] if last_result else [],
            },
            "logs": [asdict(item) for item in self._services.logs.tail(120)],
            "generation_logs": [asdict(item) for item in self._services.generation_logs.tail(40)],
            "reply_feedback": [asdict(item) for item in self._services.reply_feedback.tail(40)],
            "settings_audit": [asdict(item) for item in self._services.settings_audit.tail(40)],
            "diagnostics_files": self._diagnostics_file_payload(),
        }
        redacted_payload = redact_diagnostics_payload(payload)
        (sample_dir / "diagnostics.json").write_text(json.dumps(redacted_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._config.privacy.save_debug_screenshots and last_result and last_result.image_path.exists():
            shutil.copy2(last_result.image_path, sample_dir / last_result.image_path.name)
        if self._config.privacy.save_debug_screenshots and last_title and last_title.title_ocr_image_path.exists():
            shutil.copy2(last_title.title_ocr_image_path, sample_dir / last_title.title_ocr_image_path.name)
        self.statusBar().showMessage(f"调试样本已保存：{sample_dir}", 6000)
        self.append_log(f"debug_sample_saved: {sample_dir}")

    def _diagnostics_file_payload(self) -> dict:
        crash_log = diagnostics_log_path("crash")
        ocr_log = diagnostics_log_path("ocr_worker")
        ai_log = diagnostics_log_path("ai_provider")
        return {
            "crash_log": str(crash_log),
            "crash_tail": _tail_lines(crash_log, 40),
            "ocr_worker_log": str(ocr_log),
            "ocr_worker_tail": _tail_lines(ocr_log, 40),
            "ai_provider_log": str(ai_log),
            "ai_provider_tail": _tail_lines(ai_log, 40),
        }

    def _refresh_generation_log_text(self) -> None:
        if self._generation_log_text is None:
            return
        text = "\n\n".join([self._format_generation_log_lines(12), self._format_reply_feedback_lines(12)])
        self._generation_log_text.setText(text)

    def _format_generation_log_lines(self, limit: int) -> str:
        rows = self._services.generation_logs.tail(limit)
        if not rows:
            return "暂无 AI 生成审计。"
        lines = ["AI 生成审计"]
        for row in rows:
            lines.append(
                f"{row.ts} allowed={row.allowed} provider={row.provider} model={row.model or '-'} "
                f"status={row.status} suggestions={row.suggestion_count} risks={row.risk_summary or '-'} "
                f"context={row.context_hash[:12]} page={row.page_type}/{row.page_confidence:.2f} "
                f"messages={row.message_count} memories={row.memory_count}"
            )
        return "\n".join(lines)

    def _format_reply_feedback_lines(self, limit: int) -> str:
        rows = self._services.reply_feedback.tail(limit)
        if not rows:
            return "暂无回复反馈。"
        useful = sum(1 for row in rows if row.feedback == "useful")
        bad = sum(1 for row in rows if row.feedback == "bad")
        lines = [f"回复反馈：count={len(rows)} useful={useful} bad={bad}"]
        for row in rows:
            label = "好用" if row.feedback == "useful" else "不合适"
            lines.append(
                f"{row.ts} feedback={label} provider={row.provider} status={row.status} "
                f"label={row.suggestion_label} risk={row.risk} context={row.context_hash[:12]} "
                f"page={row.page_type} messages={row.message_count} memories={row.memory_count} "
                f"preview={row.suggestion_text_preview}"
            )
        return "\n".join(lines)

    def _readonly_text(self, title: str, content: str) -> QTextEdit:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(content)
        return text

    def _fill_table(self, table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
        table.setRowCount(len(rows))
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    def _reload_strategy_table(self) -> None:
        if self._strategy_table is None:
            return
        show_archived = self._strategy_show_archived.isChecked() if self._strategy_show_archived else False
        query = (self._strategy_search.text() if self._strategy_search else "").strip().lower()
        strategies = self._services.strategies.list_all(include_archived=show_archived)
        if query:
            strategies = [
                item for item in strategies
                if query in " ".join([item.name, item.goal, item.mode, item.tone, item.avoid]).lower()
            ]
        self._strategy_table.setRowCount(len(strategies))
        self._strategy_table.setAlternatingRowColors(True)
        self._strategy_table.verticalHeader().setVisible(False)
        for row_index, strategy in enumerate(strategies):
            assigned = self._services.strategies.count_assigned_contacts(strategy.id)
            values = [
                strategy.name,
                strategy.goal,
                strategy.tone,
                "是" if strategy.requires_manual_reply else "否",
                _strategy_status_label(strategy),
                str(assigned),
            ]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, strategy.id)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._strategy_table.setItem(row_index, col_index, item)
        self._strategy_table.resizeColumnsToContents()
        self._strategy_table.horizontalHeader().setStretchLastSection(True)

    def _selected_strategy(self) -> Strategy | None:
        if self._strategy_table is None or self._strategy_table.currentRow() < 0:
            return None
        item = self._strategy_table.item(self._strategy_table.currentRow(), 0)
        if item is None:
            return None
        return self._services.strategies.get(item.data(Qt.ItemDataRole.UserRole))

    def _add_strategy(self) -> None:
        dialog = StrategyDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        self._services.strategies.create(**values)
        self._reload_strategy_table()
        self.append_log(f"strategy_created: {values['name']}")

    def _duplicate_selected_strategy(self) -> None:
        source = self._selected_strategy()
        if source is None:
            self.statusBar().showMessage("请先选择一个分组", 2500)
            return
        dialog = StrategyDialog(self, Strategy(**{**asdict(source), "id": "", "name": f"{source.name} 副本"}))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        self._services.strategies.create(**values)
        self._reload_strategy_table()
        self.append_log(f"strategy_duplicated: source={source.id}, name={values['name']}")

    def _edit_selected_strategy(self) -> None:
        strategy = self._selected_strategy()
        if strategy is None:
            self.statusBar().showMessage("请先选择一个分组", 2500)
            return
        dialog = StrategyDialog(self, strategy)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        updated = Strategy(
            id=strategy.id,
            created_at=strategy.created_at,
            updated_at=strategy.updated_at,
            archived=strategy.archived,
            **values,
        )
        self._services.strategies.update(updated)
        self._reload_strategy_table()
        self.append_log(f"strategy_updated: {strategy.id}")

    def _toggle_selected_strategy_archive(self) -> None:
        strategy = self._selected_strategy()
        if strategy is None:
            self.statusBar().showMessage("请先选择一个分组", 2500)
            return
        if strategy.id in {"default", "manual_protect"} and not strategy.archived:
            self.statusBar().showMessage("内置安全分组不能归档", 3000)
            return
        assigned = self._services.strategies.count_assigned_contacts(strategy.id)
        archived = not strategy.archived
        if archived and assigned > 0:
            confirmed = QMessageBox.question(
                self,
                "归档分组",
                f"有 {assigned} 个聊天对象仍在使用该分组。归档不会删除这些关系，只会从默认列表中收起。是否继续？",
            )
            if confirmed != QMessageBox.StandardButton.Yes:
                return
        self._services.strategies.set_archived(strategy.id, archived)
        if not archived and self._strategy_show_archived:
            self._strategy_show_archived.setChecked(True)
        self._reload_strategy_table()
        self.append_log(f"strategy_archived_changed: {strategy.id} archived={archived}")

    def _on_contact_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        if current is None:
            return
        contact_id = current.data(Qt.ItemDataRole.UserRole)
        if not contact_id:
            return
        contact = self._services.contacts.get(contact_id)
        if contact is None:
            return
        self._customer_profile_navigation_contact_id = None
        self._render_contact_detail(contact)
        if self._active_capture_contact() is None:
            self._customer_profile_binding = None
            self._refresh_customer_profile_page(force=True)

    def _render_contact_detail(self, contact: Contact) -> None:
        messages = list(reversed(self._services.messages.list_for_contact(contact.id, 50)))
        memories = self._services.memories.list_for_contact(contact.id)
        aliases = self._services.contacts.list_aliases(contact.id)
        linked_people = self._services.identities.list_people_for_contact(contact.id)
        people_candidates = self._services.identities.find_people_by_alias(contact.display_name)
        strategy = self._services.strategies.get(contact.strategy_id)
        profile = (
            f"聊天对象：{_contact_display_label(contact)}\n"
            f"平台：{contact.platform}\n"
            f"会话类型：{contact.conversation_type.value}\n"
            f"分组：{strategy.name if strategy else contact.strategy_id}\n"
            f"确认等级：{contact.status.value}\n"
            f"云端 AI：{'联系人级允许' if contact.allow_cloud_ai else '联系人级未允许'}\n"
            f"主动联系 AI：{self._proactive_ai_permission_label(contact)}\n"
            f"别名：{', '.join(alias.alias for alias in aliases) or '暂无'}\n"
            f"备注：{contact.remark or '暂无'}\n"
            f"记忆数量：{len(memories)}"
        )
        identity_text = self._format_contact_identity_detail(contact, linked_people, people_candidates)
        chat = "\n".join(f"{message.speaker.value}: {message.text}" for message in messages) or "暂无聊天记录。"
        memory_text = "\n".join(f"[{memory.status.value}/{memory.kind.value}] {memory.content}" for memory in memories) or "暂无记忆。"
        if self._contact_profile_text:
            self._contact_profile_text.setText(profile)
        if self._contact_identity_text:
            self._contact_identity_text.setText(identity_text)
        if self._contact_members_text:
            self._contact_members_text.setText(self._format_group_members_detail(contact))
        if self._contact_chat_text:
            self._contact_chat_text.setText(chat)
        if self._contact_memory_text:
            self._contact_memory_text.setText(memory_text)
        if self._contact_feedback_text:
            self._contact_feedback_text.setText(self._format_contact_feedback_detail(contact))
        self._sync_contact_ai_toggle(contact)
        self._refresh_overview_data()

    def _show_updated_contact(self, contact: Contact) -> None:
        self._reload_contact_list(contact.id)
        self._render_contact_detail(contact)
        self._sync_floating_content(contact=contact)

    def _format_contact_identity_detail(self, contact, linked_people, people_candidates) -> str:
        lines = [
            f"平台对象：{contact.platform}·{contact.display_name}",
            "",
            "已链接真实身份：",
        ]
        if linked_people:
            for person, link in linked_people:
                aliases = self._services.identities.list_person_aliases(person.id)
                lines.extend(
                    [
                        f"- {person.display_name} · {person.status.value} · verified={link.verified} · confidence={link.confidence:.2f}",
                        f"  aliases={', '.join(alias.alias for alias in aliases) or '暂无'}",
                        f"  id={person.id}",
                    ]
                )
        else:
            lines.append("- 暂无")
        candidate_ids = {person.id for person, _link in linked_people}
        candidates = [person for person in people_candidates if person.id not in candidate_ids]
        lines.extend(["", "同名候选："])
        if candidates:
            for person in candidates[:8]:
                aliases = self._services.identities.list_person_aliases(person.id)
                lines.append(f"- {person.display_name} · {person.status.value} · aliases={', '.join(alias.alias for alias in aliases) or '暂无'} · {person.id}")
        else:
            lines.append("- 暂无")
        return "\n".join(lines)

    def _format_group_members_detail(self, contact: Contact) -> str:
        if contact.conversation_type != ConversationType.GROUP:
            return "仅群聊可用。"
        members = self._services.identities.list_group_members(contact.id)
        if not members:
            return "暂无群成员。可手动添加成员候选，后续 OCR 会逐步补全。"
        lines = [f"群聊：{contact.platform}·{contact.display_name}", "", "成员："]
        for member in members:
            person = self._services.identities.get_person(member.person_id) if member.person_id else None
            platform_contact = self._services.contacts.get(member.platform_contact_id) if member.platform_contact_id else None
            lines.append(
                "- "
                f"{member.member_display_name} · confidence={member.confidence:.2f} · source={member.source} · "
                f"person={person.display_name if person else '未链接'} · "
                f"对象={platform_contact.platform + '·' + platform_contact.display_name if platform_contact else '未链接'}"
            )
        return "\n".join(lines)

    def _format_contact_feedback_detail(self, contact: Contact) -> str:
        rows = self._services.reply_feedback.list_for_contact(contact.id, 20)
        if not rows:
            return "暂无回复反馈。"
        useful = sum(1 for row in rows if row.feedback == "useful")
        bad = sum(1 for row in rows if row.feedback == "bad")
        lines = [f"最近反馈：{len(rows)} 条 · 好用 {useful} · 不合适 {bad}", ""]
        for row in rows:
            label = "好用" if row.feedback == "useful" else "不合适"
            lines.append(
                f"{row.ts} [{label}] {row.suggestion_label} · risk={row.risk} · "
                f"{row.provider}/{row.status} · {row.page_type} · "
                f"消息{row.message_count}/记忆{row.memory_count} · context={row.context_hash[:12]}"
            )
            lines.append(f"  {row.suggestion_text_preview}")
        return "\n".join(lines)

    def _selected_contact(self) -> Contact | None:
        if self._contact_list is None or self._contact_list.currentItem() is None:
            return None
        contact_id = self._contact_list.currentItem().data(Qt.ItemDataRole.UserRole)
        return self._services.contacts.get(contact_id) if contact_id else None

    def _update_selected_contact_status(self, status: ContactStatus) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        updated = self._services.contacts.update_profile(contact.id, status=status)
        if status == ContactStatus.CONFIRMED:
            self._services.identity_bindings.mark_contact_verified(updated.id)
        self._show_updated_contact(updated)
        self._invalidate_proactive_context("contact_status_changed", updated.id)
        self.append_log(f"contact_status_updated: {updated.id} -> {status.value}")
        self.statusBar().showMessage(f"聊天对象已更新为：{status.value}", 2500)
        self._resume_reply_flow_if_unblocked("contact_status")

    def _protect_selected_contact(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        updated = self._services.contacts.update_profile(contact.id, strategy_id="manual_protect", status=ContactStatus.CONFIRMED)
        self._services.identity_bindings.mark_contact_verified(updated.id)
        self._show_updated_contact(updated)
        self._invalidate_proactive_context("contact_strategy_changed", updated.id)
        self.append_log(f"contact_manual_protect_enabled: {updated.id}")
        self.statusBar().showMessage("已进入手动回复保护", 2500)
        self._resume_reply_flow_if_unblocked("manual_protect")

    def _toggle_selected_contact_cloud_ai(self, checked: bool) -> None:
        contact = self._current_contact()
        if contact is None:
            self._sync_contact_ai_toggle(None)
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        updated = self._services.contacts.update_profile(contact.id, allow_cloud_ai=checked)
        self._show_updated_contact(updated)
        self._invalidate_proactive_context("contact_cloud_policy_changed", updated.id)
        self.append_log(f"contact_cloud_ai_updated: {updated.id} -> {updated.allow_cloud_ai}")
        self.statusBar().showMessage(f"联系人级云端授权已{'开启' if updated.allow_cloud_ai else '关闭'}", 2500)
        self._resume_reply_flow_if_unblocked("contact_cloud_ai")

    def _sync_contact_ai_toggle(self, contact: Contact | None) -> None:
        if self._contact_ai_toggle is None:
            return
        self._contact_ai_toggle.blockSignals(True)
        enabled = contact is not None
        checked = bool(contact and contact.allow_cloud_ai)
        self._contact_ai_toggle.setEnabled(enabled)
        self._contact_ai_toggle.setChecked(checked)
        self._contact_ai_toggle.setText("单客云端开" if checked else "单客云端关")
        self._contact_ai_toggle.setToolTip(_CONTACT_LONG_TERM_CLOUD_HELP)
        self._contact_ai_toggle.blockSignals(False)

    def _edit_selected_contact_profile(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑聊天对象资料")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        form = QGridLayout()
        name = QLineEdit(contact.display_name)
        remark = QTextEdit(contact.remark)
        remark.setMinimumHeight(82)
        status = QComboBox()
        status.addItems([item.value for item in ContactStatus if item != ContactStatus.MERGED])
        status.setCurrentText(contact.status.value)
        strategy = QComboBox()
        strategies = self._services.strategies.list_active()
        if contact.strategy_id not in {item.id for item in strategies}:
            current_strategy = self._services.strategies.get(contact.strategy_id)
            if current_strategy is not None:
                strategies.append(current_strategy)
        for item in strategies:
            label = f"{item.name}（已归档）" if item.archived else item.name
            strategy.addItem(label, item.id)
        index = next((i for i, item in enumerate(strategies) if item.id == contact.strategy_id), 0)
        strategy.setCurrentIndex(index)
        allow_cloud_ai = QCheckBox(_CONTACT_LONG_TERM_CLOUD_LABEL)
        allow_cloud_ai.setToolTip(_CONTACT_LONG_TERM_CLOUD_HELP)
        allow_cloud_ai.setChecked(contact.allow_cloud_ai)
        for row, (label, widget) in enumerate([
            ("显示名称", name),
            ("确认等级", status),
            ("分组策略", strategy),
            ("云端 AI", allow_cloud_ai),
            ("备注", remark),
        ]):
            form.addWidget(QLabel(label), row, 0, Qt.AlignmentFlag.AlignTop)
            form.addWidget(widget, row, 1)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated = self._services.contacts.update_profile(
            contact.id,
            display_name=name.text().strip() or contact.display_name,
            status=ContactStatus(status.currentText()),
            strategy_id=strategy.currentData(),
            allow_cloud_ai=allow_cloud_ai.isChecked(),
            remark=remark.toPlainText().strip(),
        )
        self._show_updated_contact(updated)
        self._invalidate_proactive_context("contact_profile_changed", updated.id)
        self.append_log(f"contact_profile_updated: {updated.id}")
        self._resume_reply_flow_if_unblocked("contact_profile")

    def _add_alias_to_selected_contact(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        value, ok = QInputDialog.getText(self, "添加对象别名", "别名")
        if not ok:
            return
        alias = self._services.contacts.add_alias(contact.id, value, "manual")
        if alias is None:
            self.statusBar().showMessage("别名不能为空", 2500)
            return
        refreshed = self._services.contacts.get(contact.id)
        if refreshed:
            self._render_contact_detail(refreshed)
        self.append_log(f"contact_alias_added: {contact.id} alias={alias.alias}")

    def _merge_selected_contact(self) -> None:
        source = self._current_contact()
        if source is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        candidates = [item for item in self._services.contacts.list_recent(500) if item.id != source.id and item.platform == source.platform]
        if not candidates:
            self.statusBar().showMessage("没有可合并的目标对象", 3000)
            return
        labels = [f"{item.display_name} · {item.status.value} · {item.id}" for item in candidates]
        selected, ok = QInputDialog.getItem(self, "合并聊天对象", "合并到", labels, 0, False)
        if not ok or not selected:
            return
        target = candidates[labels.index(selected)]
        updated = self._services.contacts.merge_contacts(source.id, target.id)
        self._reload_contact_list(updated.id)
        self._render_contact_detail(updated)
        self.append_log(f"contact_merged: source={source.id}, target={target.id}")
        self.statusBar().showMessage(f"已合并到：{updated.display_name}", 3000)

    def _create_identity_for_selected_contact(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        name, ok = QInputDialog.getText(self, "新建真实身份", "身份名称", text=contact.display_name)
        if not ok:
            return
        name = name.strip()
        if not name:
            self.statusBar().showMessage("身份名称不能为空", 2500)
            return
        person = self._services.identities.create_person(name, source=f"{contact.platform}:{contact.id}", status=IdentityStatus.CONFIRMED)
        self._services.identities.add_person_alias(person.id, contact.display_name, source="contact_display_name")
        self._services.identities.link_contact_to_person(contact.id, person.id, confidence=1.0, source="manual", verified=True)
        self._render_contact_detail(contact)
        self.append_log(f"identity_created_and_linked: contact={contact.id}, person={person.id}")
        self.statusBar().showMessage(f"已链接真实身份：{person.display_name}", 3000)

    def _link_selected_contact_identity(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        candidates = self._services.identities.find_people_by_alias(contact.display_name)
        if not candidates:
            self.statusBar().showMessage("没有同名身份候选，可先新建身份", 3000)
            return
        labels = [f"{person.display_name} · {person.status.value} · {person.id}" for person in candidates]
        selected, ok = QInputDialog.getItem(self, "链接真实身份", "链接到", labels, 0, False)
        if not ok or not selected:
            return
        person = candidates[labels.index(selected)]
        self._services.identities.link_contact_to_person(contact.id, person.id, confidence=1.0, source="manual", verified=True)
        self._render_contact_detail(contact)
        self.append_log(f"identity_linked: contact={contact.id}, person={person.id}")
        self.statusBar().showMessage(f"已链接：{person.display_name}", 3000)

    def _add_alias_to_selected_person(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        linked_people = self._services.identities.list_people_for_contact(contact.id)
        if not linked_people:
            self.statusBar().showMessage("当前对象还没有链接真实身份", 3000)
            return
        labels = [f"{person.display_name} · {person.status.value} · {person.id}" for person, _link in linked_people]
        selected, ok = QInputDialog.getItem(self, "选择真实身份", "身份", labels, 0, False)
        if not ok or not selected:
            return
        person = linked_people[labels.index(selected)][0]
        alias, ok = QInputDialog.getText(self, "添加身份别名", "别名")
        if not ok:
            return
        item = self._services.identities.add_person_alias(person.id, alias, "manual")
        if item is None:
            self.statusBar().showMessage("身份别名不能为空", 2500)
            return
        self._render_contact_detail(contact)
        self.append_log(f"identity_alias_added: person={person.id}, alias={item.alias}")
        self.statusBar().showMessage("身份别名已添加", 2500)

    def _add_member_to_selected_group(self) -> None:
        group = self._current_contact()
        if group is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        if group.conversation_type != ConversationType.GROUP:
            self.statusBar().showMessage("仅群聊支持群成员", 2500)
            return
        name, ok = QInputDialog.getText(self, "添加群成员", "成员昵称")
        if not ok:
            return
        name = name.strip()
        if not name:
            self.statusBar().showMessage("成员昵称不能为空", 2500)
            return
        member = self._services.identities.add_group_member(
            group_contact_id=group.id,
            member_display_name=name,
            confidence=1.0,
            source="manual",
        )
        self._render_contact_detail(group)
        self.append_log(f"group_member_added: group={group.id}, member={member.member_display_name}")
        self.statusBar().showMessage("群成员已添加", 2500)

    def _link_selected_group_member_identity(self) -> None:
        group = self._current_group_or_warn()
        if group is None:
            return
        member = self._choose_group_member(group)
        if member is None:
            return
        candidates = self._services.identities.find_people_by_alias(member.member_display_name)
        if not candidates:
            self.statusBar().showMessage("没有同名身份候选，可先在身份页新建真实身份", 3500)
            return
        labels = [f"{person.display_name} · {person.status.value} · {person.id}" for person in candidates]
        selected, ok = QInputDialog.getItem(self, "链接成员身份", "真实身份", labels, 0, False)
        if not ok or not selected:
            return
        person = candidates[labels.index(selected)]
        updated = self._services.identities.add_group_member(
            group_contact_id=group.id,
            member_display_name=member.member_display_name,
            person_id=person.id,
            platform_contact_id=member.platform_contact_id,
            confidence=1.0,
            source="manual",
        )
        self._render_contact_detail(group)
        self.append_log(f"group_member_person_linked: group={group.id}, member={updated.member_display_name}, person={person.id}")
        self.statusBar().showMessage(f"成员已链接身份：{person.display_name}", 3000)

    def _link_selected_group_member_contact(self) -> None:
        group = self._current_group_or_warn()
        if group is None:
            return
        member = self._choose_group_member(group)
        if member is None:
            return
        candidates = [item for item in self._services.contacts.list_recent(500) if item.id != group.id]
        if not candidates:
            self.statusBar().showMessage("暂无可链接的平台聊天对象", 3000)
            return
        labels = [f"{item.platform}·{item.display_name} · {item.conversation_type.value} · {item.id}" for item in candidates]
        selected, ok = QInputDialog.getItem(self, "链接成员对象", "平台聊天对象", labels, 0, False)
        if not ok or not selected:
            return
        platform_contact = candidates[labels.index(selected)]
        updated = self._services.identities.add_group_member(
            group_contact_id=group.id,
            member_display_name=member.member_display_name,
            person_id=member.person_id,
            platform_contact_id=platform_contact.id,
            confidence=max(member.confidence, 0.9),
            source="manual",
        )
        self._render_contact_detail(group)
        self.append_log(
            f"group_member_contact_linked: group={group.id}, member={updated.member_display_name}, contact={platform_contact.id}"
        )
        self.statusBar().showMessage(f"成员已链接对象：{platform_contact.display_name}", 3000)

    def _current_group_or_warn(self) -> Contact | None:
        group = self._current_contact()
        if group is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return None
        if group.conversation_type != ConversationType.GROUP:
            self.statusBar().showMessage("仅群聊支持群成员", 2500)
            return None
        return group

    def _choose_group_member(self, group: Contact):
        members = self._services.identities.list_group_members(group.id)
        if not members:
            self.statusBar().showMessage("请先添加群成员", 2500)
            return None
        labels = [f"{member.member_display_name} · {member.source} · {member.id}" for member in members]
        selected, ok = QInputDialog.getItem(self, "选择群成员", "成员", labels, 0, False)
        if not ok or not selected:
            return None
        return members[labels.index(selected)]

    def _export_selected_contact_data(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        try:
            path = self._services.governance.export_contact(contact.id)
        except Exception as exc:
            self.statusBar().showMessage("导出失败", 3000)
            self.append_log(f"contact_export_failed: {contact.id} error={exc}")
            QMessageBox.warning(self, "导出失败", f"无法导出聊天对象数据：\n{exc}")
            return
        self.append_log(f"contact_exported: {contact.id} -> {path}")
        self.statusBar().showMessage(f"已导出：{path}", 6000)
        QMessageBox.information(self, "导出完成", f"聊天对象数据已导出到：\n{path}")

    def _clear_selected_contact_data(self) -> None:
        contact = self._current_contact()
        if contact is None:
            self.statusBar().showMessage("请先选择聊天对象", 2500)
            return
        answer = QMessageBox.question(
            self,
            "清空聊天对象记录",
            (
                f"将清空「{contact.display_name}」的聊天记录、记忆、AI 生成审计和回复反馈。\n\n"
                "对象名称、别名、备注和分组会保留。此操作无法撤销。是否继续？"
            ),
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._services.governance.clear_contact_data(contact.id)
        except Exception as exc:
            self.statusBar().showMessage("清空失败", 3000)
            self.append_log(f"contact_clear_failed: {contact.id} error={exc}")
            QMessageBox.warning(self, "清空失败", f"无法清空聊天对象记录：\n{exc}")
            return
        self._render_contact_detail(result.contact)
        self.append_log(
            "contact_content_cleared: "
            f"{contact.id} messages={result.messages_deleted} "
            f"memories={result.memories_deleted} generations={result.generation_logs_deleted} "
            f"feedback={result.reply_feedback_deleted}"
        )
        self.statusBar().showMessage("已清空聊天对象记录，对象资料和别名已保留", 4000)

    def _export_all_data(self) -> None:
        try:
            path = self._services.governance.export_all()
        except Exception as exc:
            self.statusBar().showMessage("全局导出失败", 3000)
            self.append_log(f"global_export_failed: {exc}", "warning")
            QMessageBox.warning(self, "全局导出失败", f"无法导出全部数据：\n{exc}")
            return
        self.append_log(f"global_exported: {path}")
        self.statusBar().showMessage(f"已导出：{path}", 6000)
        QMessageBox.information(self, "导出完成", f"全部本地数据已导出到：\n{path}")

    def _clear_all_content(self) -> None:
        answer = QMessageBox.question(
            self,
            "清空全部内容",
            (
                "将删除所有聊天对象、身份、群成员、聊天记录、记忆、AI 生成审计和回复反馈。\n\n"
                "分组策略、区域校准、设置审计和配置文件会保留。此操作无法撤销。是否继续？"
            ),
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._services.governance.clear_all_content()
        except Exception as exc:
            self.statusBar().showMessage("清空失败", 3000)
            self.append_log(f"global_clear_failed: {exc}", "warning")
            QMessageBox.warning(self, "清空失败", f"无法清空全部内容：\n{exc}")
            return
        self._reload_contact_list()
        self._refresh_overview_data()
        self.append_log(
            "global_content_cleared: "
            f"contacts={result.contacts_deleted} messages={result.messages_deleted} "
            f"memories={result.memories_deleted} generations={result.generation_logs_deleted} "
            f"feedback={result.reply_feedback_deleted}"
        )
        self.statusBar().showMessage("全部内容已清空，设置和审计已保留", 5000)

    def _run_retention_cleanup(self) -> None:
        before = _redacted_config_snapshot(self._config)
        self._sync_retention_config_from_ui()
        secret_backend = self._config_store.save(self._config)
        changes = _config_changes(before, _redacted_config_snapshot(self._config))
        if changes:
            self._services.settings_audit.append(
                actor="local_user",
                scope="retention",
                changes=changes,
                secret_backend=secret_backend,
            )
        try:
            result = self._services.retention.cleanup(self._config.privacy)
        except Exception as exc:
            self.statusBar().showMessage("本地清理失败", 3000)
            self.append_log(f"retention_cleanup_failed: {exc}", "warning")
            QMessageBox.warning(self, "本地清理失败", f"无法清理过期文件：\n{exc}")
            return
        self.append_log(
            "retention_cleanup_completed: "
            f"files={result.files_deleted} dirs={result.dirs_deleted} bytes={result.bytes_deleted} "
            f"records={result.records_deleted}"
        )
        self.statusBar().showMessage(
            f"已清理 {result.files_deleted} 个文件、{result.records_deleted} 条记录，释放 {result.bytes_deleted // 1024} KB",
            4000,
        )
        if self._diagnostics_file_text is not None:
            self._diagnostics_file_text.setText(self._format_diagnostics_files())
        self._refresh_generation_log_text()
        contact = self._current_contact()
        if contact is not None:
            self._render_contact_detail(contact)

    def _build_memory_review_tab(self, status: MemoryStatus) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        memories = self._services.memories.list_by_status(status)
        if not memories:
            layout.addWidget(self._readonly_text(status.value, "暂无数据。"))
            return container
        for memory in memories:
            layout.addWidget(self._memory_row(memory, status))
        layout.addStretch(1)
        return container

    def _memory_row(self, memory: Memory, status: MemoryStatus) -> QWidget:
        row = QFrame()
        row.setObjectName("MetricCard")
        layout = QGridLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)
        contact = self._services.contacts.get(memory.contact_id)
        title = QLabel(f"{contact.display_name if contact else '未知联系人'} · {memory.kind.value}")
        title.setObjectName("SectionTitle")
        content = QLabel(memory.content)
        content.setWordWrap(True)
        meta = QLabel(f"confidence={memory.confidence if memory.confidence is not None else '-'} · {memory.created_at}")
        meta.setObjectName("TinyMuted")
        layout.addWidget(title, 0, 0, 1, 2)
        layout.addWidget(content, 1, 0, 1, 2)
        layout.addWidget(meta, 2, 0)
        if status == MemoryStatus.PENDING:
            confirm = QPushButton("确认")
            confirm.clicked.connect(lambda checked=False, memory_id=memory.id: self._set_memory_status(memory_id, MemoryStatus.CONFIRMED))
            reject = QPushButton("拒绝")
            reject.clicked.connect(lambda checked=False, memory_id=memory.id: self._set_memory_status(memory_id, MemoryStatus.REJECTED))
            layout.addWidget(confirm, 0, 2)
            layout.addWidget(reject, 1, 2)
        layout.setColumnStretch(1, 1)
        return row

    def _set_memory_status(self, memory_id: str, status: MemoryStatus) -> None:
        memory = self._services.memories.update_status(memory_id, status)
        if self._memory_tabs is not None:
            current = self._memory_tabs.currentIndex()
            self._memory_tabs.clear()
            self._memory_tabs.addTab(self._build_memory_review_tab(MemoryStatus.PENDING), "待确认")
            self._memory_tabs.addTab(self._build_memory_review_tab(MemoryStatus.CONFIRMED), "长期画像")
            self._memory_tabs.addTab(self._build_memory_review_tab(MemoryStatus.REJECTED), "已拒绝")
            self._memory_tabs.setCurrentIndex(min(current, self._memory_tabs.count() - 1))
        contact = self._services.contacts.get(memory.contact_id)
        if contact is not None:
            self._show_updated_contact(contact)
        self.append_log(f"memory_status_updated: {memory_id} -> {status.value}")
        self.statusBar().showMessage(f"记忆已更新为：{status.value}", 2500)

    def _open_calibration_dialog(self) -> None:
        state = self._services.runtime.state
        if state.layout is None or state.window.rect is None:
            self.statusBar().showMessage("当前没有可校准的微信窗口区域", 3000)
            self.append_log("calibration_open_failed: no layout", "warning")
            return
        screenshot_path = self._capture_calibration_screenshot(state)
        dialog = CalibrationDialog(self, state.layout, screenshot_path)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        calibration = self._services.calibrations.create_from_layout(
            name=values["name"],
            target=TargetApp.WECHAT,
            window_rect=state.window.rect,
            layout=values["layout"],
            theme=values["theme"],
            active=True,
        )
        refreshed = self._services.runtime.refresh_current()
        self.update_runtime_state(refreshed)
        self.append_log(f"layout_calibration_saved: {calibration.id} name={calibration.name}")
        self.statusBar().showMessage("区域校准已保存并启用", 3000)

    def _capture_calibration_screenshot(self, state: RuntimeState) -> Path | None:
        if state.window.rect is None:
            return None
        output = app_data_dir() / "calibration" / "current_window.png"
        try:
            return capture_rect(state.window.rect.as_tuple(), output)
        except Exception as exc:
            self.append_log(f"calibration_screenshot_unavailable: {exc}", "warning")
            return None

    def _run_capture_pipeline(self) -> None:
        if getattr(self, "_manual_generation_enabled", False):
            self._on_generate_requested()
            return
        state = self._services.runtime.state
        job_id = self._services.pipeline.submit(self._services.runtime.state)
        if job_id is None:
            summary = self._manual_capture_block_summary(state)
            self.statusBar().showMessage(summary, 6000)
            self.append_log(f"capture_pipeline_submit_skipped: {summary}", "warning")
            if self._runtime_text is not None:
                self._runtime_text.setText(self._format_runtime_state(self._services.runtime.state))
            if self._window_match_text is not None:
                self._window_match_text.setText(self._format_window_match_diagnostics())
            return
        self.statusBar().showMessage(f"采集管线已启动：job={job_id}", 2500)
        self.append_log(f"capture_pipeline_submitted: job={job_id}")

    def _format_account_scope_state(self) -> str:
        result = self._latest_account_evidence()
        if result is None or not getattr(result, "target_instance_id", ""):
            return "尚未取得当前微信窗口的账号证据。请保持微信在前台，完成一次采集。"
        bindable, reason = self._account_evidence_bindability(result)
        if not bindable:
            return f"最近一次账号证据不可用：{reason}。请回到要绑定的微信窗口重新采集。"
        if result.account_scope_id:
            return f"当前已绑定：{result.account_scope_name}（本地隔离空间）"
        return "刚刚采集到一个未绑定的微信窗口；可点击下方按钮绑定。绑定前已阻止消息入库和 AI。"

    def _refresh_account_scope_label(self) -> None:
        if self._account_scope_label is not None:
            self._account_scope_label.setText(self._format_account_scope_state())

    def _bind_current_account_scope(self) -> None:
        result = self._latest_account_evidence()
        if result is None or not getattr(result, "target_instance_id", ""):
            self.statusBar().showMessage("请先保持微信在前台并完成一次采集", 4000)
            return
        bindable, reason = self._account_evidence_bindability(result)
        if not bindable:
            self.statusBar().showMessage(f"无法绑定：{reason}；请回到要绑定的微信窗口重新采集", 6000)
            return
        initial = result.account_scope_name or "工作号1"
        local_name, ok = QInputDialog.getText(
            self,
            "绑定当前微信账号",
            "给这个微信登录账号设置一个仅保存在本机的名称：",
            text=initial,
        )
        if not ok or not local_name.strip():
            return
        try:
            context = self._services.account_scope_resolver.bind(result.target_instance_id, local_name.strip())
        except Exception as exc:
            self.append_log(f"account_scope_bind_failed: {exc}", "warning")
            self.statusBar().showMessage("账号绑定失败，请查看诊断信息", 5000)
            return
        self._services.ingestion.reset_context("account_scope_bound")
        self._services.pipeline.reset_snapshot_hash_cache()
        self._services.runtime.capture_gate.last_capture_ms = 0
        self._services.runtime.capture_gate.last_hash = None
        if self._account_scope_label is not None:
            self._account_scope_label.setText(f"当前已绑定：{context.account_scope_name}（本地隔离空间；等待下一帧确认）")
        self.append_log(f"account_scope_bound: scope={context.account_scope_id} instance={context.target_instance_id}")
        self.statusBar().showMessage(f"已绑定当前微信账号：{context.account_scope_name}，请保持微信前台等待两帧确认", 6000)

    def _latest_account_evidence(self):
        candidates = [
            item for item in (self._services.pipeline.last_title_result, self._services.pipeline.last_result)
            if item is not None
        ]
        return max(candidates, key=lambda item: item.job_id, default=None)

    def _account_evidence_bindability(self, result) -> tuple[bool, str]:
        state = self._services.runtime.state
        exact_current = (
            result.hwnd == state.window.hwnd
            and result.process_id == state.window.process_id
            and result.window_generation == state.window_generation
            and result.capture_generation == state.capture_generation
        )
        if exact_current:
            return True, "current_target"
        # In foreground-only mode clicking WhoChat necessarily removes WeChat
        # from RuntimeState.  Permit a short, fail-closed handoff only when the
        # exact HWND/PID from the latest capture still exists and the persisted
        # target-instance evidence has not changed.
        if state.window.hwnd is not None:
            return False, "另一个微信窗口已经成为当前目标"
        try:
            observed = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
            age_seconds = (datetime.now(observed.tzinfo) - observed).total_seconds()
        except (TypeError, ValueError):
            return False, "采集时间无效"
        if age_seconds < 0 or age_seconds > 120:
            return False, "采集证据已超过 120 秒"
        instance = self._services.account_scopes.get_instance(result.target_instance_id)
        if instance is None:
            return False, "本地窗口实例不存在"
        if instance.last_hwnd != result.hwnd or instance.last_pid != result.process_id:
            return False, "窗口实例证据已经变化"
        if not _native_window_matches_evidence(result.hwnd, result.process_id):
            return False, "原微信窗口已经关闭或进程已变化"
        return True, "recent_verified_target"

    def _manual_capture_block_summary(self, state: RuntimeState) -> str:
        discard = self._services.pipeline.last_discard_reason or "-"
        if state.window.state.value != "visible":
            return f"采集未启动：窗口 {state.window.state.value}；{state.window.diagnostic or state.capture_decision.reason}"
        if state.layout is None:
            return f"采集未启动：区域不可用；{state.page.reason}"
        if not state.capture_decision.should_capture:
            return f"采集未启动：{state.capture_decision.reason}；discard={discard}"
        if self._services.pipeline.is_running:
            return f"采集未启动：上一轮采集仍在运行；discard={discard}"
        return f"采集未启动：{discard}"

    def _refresh_target_grid(self) -> None:
        if self._target_grid is None:
            return
        pending = {
            target.app_id: (
                self._target_checkboxes[target.app_id].isChecked() if target.app_id in self._target_checkboxes else target.enabled,
                self._target_process_inputs[target.app_id].text() if target.app_id in self._target_process_inputs else ", ".join(target.process_names),
                self._target_title_inputs[target.app_id].text() if target.app_id in self._target_title_inputs else ", ".join(target.title_keywords),
                self._target_exclude_title_inputs[target.app_id].text()
                if target.app_id in self._target_exclude_title_inputs
                else ", ".join(target.exclude_title_keywords),
            )
            for target in self._config.targets
        }
        while self._target_grid.count():
            item = self._target_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._target_checkboxes = {}
        self._target_process_inputs = {}
        self._target_title_inputs = {}
        self._target_exclude_title_inputs = {}
        for col, label in enumerate(["启用", "应用", "进程名", "标题关键词", "排除标题", ""]):
            header = QLabel(label)
            header.setObjectName("TinyMuted")
            self._target_grid.addWidget(header, 0, col)
        for row, target in enumerate(self._config.targets):
            checked, process_text, title_text, exclude_title_text = pending.get(
                target.app_id,
                (
                    target.enabled,
                    ", ".join(target.process_names),
                    ", ".join(target.title_keywords),
                    ", ".join(target.exclude_title_keywords),
                ),
            )
            checkbox = QCheckBox(target.label)
            checkbox.setChecked(checked)
            process_input = QLineEdit(process_text)
            title_input = QLineEdit(title_text)
            exclude_title_input = QLineEdit(exclude_title_text)
            self._target_checkboxes[target.app_id] = checkbox
            self._target_process_inputs[target.app_id] = process_input
            self._target_title_inputs[target.app_id] = title_input
            self._target_exclude_title_inputs[target.app_id] = exclude_title_input
            self._target_grid.addWidget(checkbox, row + 1, 0)
            self._target_grid.addWidget(QLabel(target.label), row + 1, 1)
            self._target_grid.addWidget(process_input, row + 1, 2)
            self._target_grid.addWidget(title_input, row + 1, 3)
            self._target_grid.addWidget(exclude_title_input, row + 1, 4)
            if target.app_id.startswith("custom_"):
                remove = QPushButton("删除")
                remove.setObjectName("DangerButton")
                remove.clicked.connect(lambda _checked=False, app_id=target.app_id: self._remove_target_app(app_id))
                self._target_grid.addWidget(remove, row + 1, 5)

    def _add_target_app(self) -> None:
        dialog = TargetAppDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        app_id = _unique_target_app_id(values["app_id"], [target.app_id for target in self._config.targets])
        self._config.targets.append(
            TargetWindowConfig(
                app_id=app_id,
                label=values["label"],
                enabled=True,
                process_names=values["process_names"],
                title_keywords=values["title_keywords"],
                exclude_title_keywords=values["exclude_title_keywords"],
            )
        )
        self._refresh_target_grid()
        self.append_log(f"target_app_added: {app_id}")

    def _remove_target_app(self, app_id: str) -> None:
        self._config.targets = [target for target in self._config.targets if target.app_id != app_id or not target.app_id.startswith("custom_")]
        self._refresh_target_grid()
        self.append_log(f"target_app_removed: {app_id}")

    def _save_ai_settings(self) -> None:
        prereader = getattr(self, "_local_preread", None)
        if prereader is not None:
            prereader.pause()
            if prereader.is_running:
                self.statusBar().showMessage("本地聊天读取正在结束，请稍后再次保存设置", 5000)
                return
        parsed_targets = self._target_configs_from_ui()
        if parsed_targets is None:
            return
        validation_error = self._settings_validation_error(parsed_targets)
        if validation_error:
            self.statusBar().showMessage(validation_error, 5000)
            self._set_ai_action_status(validation_error)
            self.append_log(f"settings_save_failed: {validation_error}", "warning")
            return
        before = _redacted_config_snapshot(self._config)
        previous_connection = (
            self._config.ai.provider,
            self._config.ai.base_url,
            self._config.ai.model,
            self._config.ai.api_key,
        )
        previous_api_key = self._config.ai.api_key
        submitted_api_key = self._ai_api_key.text()
        self._config.ai.provider = self._ai_provider.currentText()
        self._config.ai.base_url = self._ai_base_url.text().strip()
        self._config.ai.model = self._ai_model.text().strip()
        self._config.ai.api_key = submitted_api_key
        self._config.ai.temperature = self._ai_temperature.value()
        self._config.ai.context_tokens = self._ai_context_tokens.value()
        self._config.ai.timeout_seconds = self._ai_timeout.value()
        self._config.ai.request_cooldown_seconds = self._ai_request_cooldown.value() if self._ai_request_cooldown else 8
        self._config.ai.dedupe_context_minutes = self._ai_dedupe_minutes.value() if self._ai_dedupe_minutes else 30
        self._config.ai.max_daily_cloud_requests = self._ai_daily_limit.value() if self._ai_daily_limit else 100
        self._config.ai.failure_backoff_threshold = self._ai_failure_threshold.value() if self._ai_failure_threshold else 3
        self._config.ai.failure_backoff_minutes = self._ai_failure_backoff.value() if self._ai_failure_backoff else 5
        self._config.privacy.enable_long_term_memory = self._privacy_long_memory.isChecked() if self._privacy_long_memory else True
        self._config.privacy.save_debug_screenshots = self._privacy_debug_screenshots.isChecked() if self._privacy_debug_screenshots else False
        self._config.privacy.trim_context_for_cloud = self._privacy_trim_cloud.isChecked() if self._privacy_trim_cloud else True
        self._config.privacy.require_cloud_prompt_review = self._privacy_cloud_review.isChecked() if self._privacy_cloud_review else False
        self._config.privacy.manual_protection_blocks_replies = self._privacy_manual_blocks.isChecked() if self._privacy_manual_blocks else True
        self._sync_retention_config_from_ui()
        # A missing widget must never silently downgrade OCR: keep whatever the
        # loaded config says instead of writing the fixture engine to disk.
        if self._ocr_provider is not None:
            self._config.ocr.provider = self._ocr_provider.currentText()
        else:
            self.append_log("settings_save_ocr_widget_missing: 保留原识别引擎设置", "warning")
        self._config.ocr.language = self._ocr_language.text().strip() if self._ocr_language else "ch"
        self._config.ocr.min_confidence = self._ocr_min_confidence.value() if self._ocr_min_confidence else 0.5
        self._config.ocr.use_gpu = self._ocr_use_gpu.isChecked() if self._ocr_use_gpu else False
        self._config.capture.scroll_debounce_ms = self._capture_debounce.value() if self._capture_debounce else 500
        self._config.capture.ocr_min_interval_ms = self._capture_ocr_interval.value() if self._capture_ocr_interval else 5000
        self._config.capture.auto_capture_enabled = self._capture_auto_enabled.isChecked() if self._capture_auto_enabled else True
        self._config.capture.foreground_only = self._capture_foreground_only.isChecked() if self._capture_foreground_only else True
        self._config.capture.pause_ai_on_unknown_page = self._capture_pause_unknown.isChecked() if self._capture_pause_unknown else True
        self._config.capture.block_memory_for_unconfirmed_contact = (
            self._capture_block_unconfirmed.isChecked() if self._capture_block_unconfirmed else True
        )
        self._config.floating.placement_preference = self._floating_placement.currentText() if self._floating_placement else "auto"
        self._config.floating.opacity_percent = self._floating_opacity.value() if self._floating_opacity else 96
        self._config.floating.suggestion_count = 2
        self._config.targets = parsed_targets
        current_connection = (
            self._config.ai.provider,
            self._config.ai.base_url,
            self._config.ai.model,
            self._config.ai.api_key,
        )
        self._services.autocapture.set_enabled(False if self._manual_generation_enabled else self._config.capture.auto_capture_enabled)
        self.capture_mode_changed.emit(self._config.capture.foreground_only)
        self._services.runtime.capture_gate.policy = replace(
            self._services.runtime.capture_gate.policy,
            scroll_debounce_ms=self._config.capture.scroll_debounce_ms,
            ocr_min_interval_ms=self._config.capture.ocr_min_interval_ms,
        )
        old_ocr_shutdown = getattr(self._services.pipeline.ocr_engine, "shutdown", None)
        if callable(old_ocr_shutdown):
            try:
                old_ocr_shutdown()
            except Exception as exc:
                self.append_log(f"ocr_engine_shutdown_failed_before_switch: {exc}", "warning")
        self._services.pipeline.ocr_engine = create_ocr_engine(self._config.ocr)
        if self._floating is not None and hasattr(self._floating, "apply_preferences"):
            self._floating.apply_preferences(
                placement_preference=self._config.floating.placement_preference,
                opacity_percent=self._config.floating.opacity_percent,
                suggestion_count=self._config.floating.suggestion_count,
            )
        self.targets_changed.emit(self._config.targets)
        secret_backend = self._config_store.save(self._config)
        provider_changed = previous_connection != current_connection
        if provider_changed:
            health_summary = self._services.reply_generator.reset_provider_health("provider_settings_changed")
            self.append_log(f"ai_provider_health_reset_after_settings_change: {health_summary}", "info")
        changes = _config_changes(before, _redacted_config_snapshot(self._config))
        if previous_api_key != self._config.ai.api_key or submitted_api_key:
            changes["ai.api_key"] = {
                "old": "<set>" if previous_api_key else "<empty>",
                "new": "<set>" if self._config.ai.api_key else "<empty>",
                "changed": previous_api_key != self._config.ai.api_key,
                "submitted": bool(submitted_api_key),
            }
        if changes:
            self._services.settings_audit.append(
                actor="local_user",
                scope="settings",
                changes=changes,
                secret_backend=secret_backend,
            )
        if self._ai_status_value is not None:
            value = self._config.ai.model if self._config.ai.provider != "Disabled" else "已禁用"
            self._ai_status_value.setText(value)
        if self._ai_health_label is not None:
            self._ai_health_label.setText(self._services.reply_generator.provider_health_summary())
        self.append_log(
            f"ai_settings_saved: provider={self._config.ai.provider}, model={self._config.ai.model}, "
            f"changes={len(changes)}, secret_backend={secret_backend}"
        )
        secret_note = _secret_storage_note(secret_backend, bool(self._config.ai.api_key))
        self.statusBar().showMessage(secret_note, 5000)
        self._set_ai_action_status(
            ("设置已保存，已恢复 Provider 健康状态" if provider_changed else "设置已保存") + f"；{secret_note}"
        )
        self._resume_reply_flow_if_unblocked("settings_saved")

    def _settings_validation_error(self, parsed_targets: list[TargetWindowConfig]) -> str:
        provider = self._ai_provider.currentText() if self._ai_provider else self._config.ai.provider
        base_url = self._ai_base_url.text().strip() if self._ai_base_url else ""
        model = self._ai_model.text().strip() if self._ai_model else ""
        if provider in {"OpenAI", "OpenAI Compatible"}:
            if not base_url.startswith(("http://", "https://")):
                return "云端 AI 接口地址必须以 http:// 或 https:// 开头"
            if not model:
                return "云端 AI 模型名称不能为空"
        elif provider == "Local Model" and not model:
            return "本地模型名称不能为空"
        ocr_language = self._ocr_language.text().strip() if self._ocr_language else "ch"
        if not ocr_language:
            return "OCR 语言不能为空"
        if not any(target.enabled for target in parsed_targets):
            return "至少启用一个目标聊天应用"
        return ""

    def _reset_ai_provider_health(self) -> None:
        summary = self._services.reply_generator.reset_provider_health("manual_ui")
        if self._ai_health_label is not None:
            self._ai_health_label.setText(summary)
        self._services.logs.append(
            "warning",
            "ai",
            "provider_health_reset",
            "AI Provider health state reset by user",
            {"summary": summary},
        )
        self.append_log(f"ai_provider_health_reset: {summary}", "warning")
        self.statusBar().showMessage("AI Provider 健康状态已恢复", 3000)
        self._resume_reply_flow_if_unblocked("provider_health_reset")

    def _sync_retention_config_from_ui(self) -> None:
        self._config.privacy.diagnostic_log_retention_days = self._privacy_log_retention.value() if self._privacy_log_retention else 14
        self._config.privacy.debug_sample_retention_days = self._privacy_debug_retention.value() if self._privacy_debug_retention else 14
        self._config.privacy.capture_retention_days = self._privacy_capture_retention.value() if self._privacy_capture_retention else 7
        self._config.privacy.calibration_retention_days = self._privacy_calibration_retention.value() if self._privacy_calibration_retention else 30
        self._config.privacy.reply_feedback_retention_days = self._privacy_feedback_retention.value() if self._privacy_feedback_retention else 180

    def _target_configs_from_ui(self) -> list[TargetWindowConfig] | None:
        result: list[TargetWindowConfig] = []
        for target in self._config.targets:
            checkbox = self._target_checkboxes.get(target.app_id)
            process_input = self._target_process_inputs.get(target.app_id)
            title_input = self._target_title_inputs.get(target.app_id)
            exclude_title_input = self._target_exclude_title_inputs.get(target.app_id)
            enabled = checkbox.isChecked() if checkbox is not None else target.enabled
            process_names = _split_match_rules(process_input.text()) if process_input is not None else list(target.process_names)
            title_keywords = _split_match_rules(title_input.text()) if title_input is not None else list(target.title_keywords)
            exclude_title_keywords = (
                _split_match_rules(exclude_title_input.text())
                if exclude_title_input is not None
                else list(target.exclude_title_keywords)
            )
            if enabled and not process_names and not title_keywords:
                self.statusBar().showMessage(f"{target.label} 已启用但缺少匹配规则", 4000)
                self.append_log(f"settings_save_failed: target_without_rules app={target.app_id}", "warning")
                return None
            result.append(
                TargetWindowConfig(
                    app_id=target.app_id,
                    label=target.label,
                    # ``bubble_profile`` is not editable in the settings grid, so it
                    # must be carried over from the loaded config.  Omitting it here
                    # silently rewrote every target back to "auto" on each save,
                    # which dropped WeChat's tuned ``wechat_green`` bubble profile.
                    bubble_profile=target.bubble_profile,
                    enabled=enabled,
                    process_names=process_names,
                    title_keywords=title_keywords,
                    exclude_title_keywords=exclude_title_keywords,
                )
            )
        return result

    def _test_ai_settings(self) -> None:
        self._set_ai_action_status("正在测试连接...")
        QApplication.processEvents()
        config = replace(
            self._config,
            ai=replace(
                self._config.ai,
                provider=self._ai_provider.currentText(),
                base_url=self._ai_base_url.text().strip(),
                model=self._ai_model.text().strip(),
                api_key=self._ai_api_key.text(),
                temperature=self._ai_temperature.value(),
                context_tokens=self._ai_context_tokens.value(),
                timeout_seconds=self._ai_timeout.value(),
            ),
        )
        result = test_ai_connection(config)
        level = "info" if result.ok else "warning"
        self.statusBar().showMessage(result.detail, 5000)
        self._set_ai_action_status(result.detail)
        self.append_log(
            f"ai_settings_test_{'passed' if result.ok else 'failed'}: "
            f"provider={result.provider} status={result.status} elapsed_ms={result.elapsed_ms}",
            level,
        )

    def _set_ai_action_status(self, text: str) -> None:
        if self._ai_action_status is not None:
            self._ai_action_status.setText(text)
