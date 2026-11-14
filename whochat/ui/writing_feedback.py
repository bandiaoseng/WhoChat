"""Feedback UI. Editing is local; it never acquires new copy/source rights."""
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QTextEdit, QVBoxLayout,
)


def _plain(text):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def capture_display(host, result, context):
    service = getattr(host._services, "writing_feedback", None)
    host._writing_feedback_ids = ()
    if service is None or context is None or not result.allowed or not result.suggestions:
        return result
    try:
        # Rerendering the adopted text must not register it as a new AI original.
        previous = getattr(host, "_writing_feedback_display", None)
        from whochat.services.reply import context_hash
        source = context_hash(context)
        if previous and result is previous[0] and source == previous[1]:
            ids = previous[2]
        else:
            ids = service.capture(context, result)
        adopted = service.overlay(result, ids)
        host._writing_feedback_ids = ids
        host._writing_feedback_display = (adopted, source, ids)
        return adopted
    except Exception as exc:
        host.statusBar().showMessage(f"建议可继续使用，但反馈历史未保存（{type(exc).__name__}）", 8000)
        return result


def replace_visible_text(host, key, text):
    """Update only an immutable matching original; don't bind to current customer."""
    ids = getattr(host, "_writing_feedback_ids", ())
    result = getattr(host, "_suggestion_result", None)
    if key not in ids or result is None or not result.allowed:
        return False
    index = ids.index(key)
    if index >= len(result.suggestions):
        return False
    # An in-flight focus-handoff copy must not later copy the pre-edit version.
    pending = getattr(host, "_manual_copy_ticket", None)
    host._manual_copy_ticket = None
    floating = getattr(host, "_floating", None)
    if pending is not None and floating is not None and hasattr(floating, "set_copy_status"):
        floating.set_copy_status("rejected", "话术已修改；请重新点击复制最终版本")
    suggestions = list(result.suggestions)
    suggestions[index] = replace(suggestions[index], text=text)
    adopted = replace(result, suggestions=suggestions)
    host._suggestion_result = adopted
    previous = getattr(host, "_writing_feedback_display", None)
    if previous:
        host._writing_feedback_display = (adopted, previous[1], ids)
    host._sync_floating_content()
    # Replace the card in place. A full render would renew customer bindings.
    panel = getattr(host, "_suggestions_panel", None)
    if panel is not None:
        layout = panel.layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None and widget.property("feedback_record") == key:
                fresh = host._suggestion_row(suggestions[index], index)
                layout.replaceWidget(widget, fresh)
                widget.hide()
                widget.deleteLater()
                break
        host._resize_suggestion_list_to_content()
    return True


def save_edit(host, key, text, revision):
    service = host._services.writing_feedback
    row = service.save(key, text, revision)
    visible = replace_visible_text(host, key, row["final"])
    update = "主界面和悬浮条已同步" if visible else "原客户的历史建议已更新，当前其他建议未改动"
    host.statusBar().showMessage(f"修改已保存到本地；{update}。下次手动生成时参考，不会自动发送。", 8000)
    return row


def edit_dialog(host, key, parent=None):
    try:
        row = host._services.writing_feedback.get(key)
    except Exception as exc:
        QMessageBox.warning(parent or host, "无法读取建议", f"读取失败（{type(exc).__name__}），未修改任何话术。")
        return
    dialog = QDialog(parent or host)
    dialog.setWindowTitle("修改反馈 · " + row["name"])
    dialog.resize(620, 660)
    layout = QVBoxLayout(dialog)
    layout.addWidget(_plain(f"客户：{row['name']}\n工作微信：{row['owner']}\n生成时间：{row['created']}"))
    layout.addWidget(_plain("此修改始终属于上面这位客户；切换微信客户不会改变反馈归属。"))
    original = QTextEdit()
    original.setReadOnly(True)
    original.setPlainText(row["original"])
    original.setMaximumHeight(120)
    layout.addWidget(_plain("AI 原建议（保留供对照）"))
    layout.addWidget(original)
    layout.addWidget(_plain("你的最终话术（可直接修改或粘贴；不需要解释原因）"))
    editor = QTextEdit()
    editor.setAcceptRichText(False)
    editor.setPlainText(row["final"] if row["final"] is not None else row["original"])
    layout.addWidget(editor)
    layout.addWidget(_plain("保存只在本地替换这一条建议，不调用 AI、不发送消息。四个工作微信共用表达和沟通偏好，客户事实不共用。"))
    error = _plain("")
    layout.addWidget(error)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
    buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存修改")
    def save():
        try:
            save_edit(host, key, editor.toPlainText(), row["revision"])
        except ValueError as exc:
            error.setText(str(exc))
        except Exception as exc:
            error.setText(f"保存失败（{type(exc).__name__}），未宣布成功；请稍后重试。")
        else:
            dialog.accept()
    buttons.accepted.connect(save)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def add_edit_action(host, card, layout, index):
    ids = getattr(host, "_writing_feedback_ids", ())
    if index >= len(ids):
        return
    key = ids[index]
    card.setProperty("feedback_record", key)
    line = QHBoxLayout()
    line.addStretch(1)
    button = QPushButton("修改反馈")
    button.setToolTip("编辑或粘贴你的最终话术；本地保存，不调用 AI、不自动发送")
    button.clicked.connect(lambda _checked=False, frozen_key=key: edit_dialog(host, frozen_key))
    line.addWidget(button)
    layout.addLayout(line)


def history_dialog(host):
    dialog = QDialog(host)
    dialog.setWindowTitle("建议与修改记录（本地）")
    dialog.resize(780, 720)
    layout = QVBoxLayout(dialog)
    layout.addWidget(_plain("保留最近 200 条建议及已保存的反馈。修改旧建议不会覆盖当前其他客户的建议。"))
    only = QCheckBox("只看已保存反馈")
    layout.addWidget(only)
    listing = QListWidget()
    layout.addWidget(listing, 1)
    detail = QTextEdit()
    detail.setReadOnly(True)
    layout.addWidget(detail, 1)
    error = _plain("")
    layout.addWidget(error)
    actions = QHBoxLayout()
    edit = QPushButton("修改反馈")
    delete = QPushButton("删除学习反馈")
    refresh = QPushButton("刷新")
    for widget in (edit, delete, refresh):
        actions.addWidget(widget)
    layout.addLayout(actions)
    service = host._services.writing_feedback
    def selected():
        item = listing.currentItem()
        return service.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
    def show():
        try:
            row = selected()
            edit.setEnabled(row is not None)
            delete.setEnabled(bool(row and row["active"]))
            if row is None:
                detail.clear()
                return
            detail.setPlainText(f"客户：{row['name']}\n工作微信：{row['owner']}\n生成时间：{row['created']}\n"
                f"AI 原建议：\n{row['original']}\n\n你的最终话术：\n{row['final'] or '尚未修改'}\n\n"
                f"生成时可见聊天（本地留存，不是当前聊天）：\n{row['context_text']}\n\n"
                f"生成时参考的反馈编号：{row['learning_refs']}")
        except Exception as exc:
            error.setText(f"读取失败（{type(exc).__name__}）")
            edit.setEnabled(False)
            delete.setEnabled(False)
    def load():
        listing.clear()
        try:
            for row in service.recent(feedback_only=only.isChecked()):
                item = QListWidgetItem(f"{row['name']} · {row['label']} · {row['created'][:16]} · " + ("已反馈" if row['active'] else "历史建议"))
                item.setData(Qt.ItemDataRole.UserRole, row["id"])
                item.setToolTip("工作微信：" + row["owner"])
                listing.addItem(item)
            error.clear()
            if listing.count():
                listing.setCurrentRow(0)
            else:
                show()
        except Exception as exc:
            error.setText(f"历史读取失败（{type(exc).__name__}）")
    def do_edit():
        if listing.currentItem():
            edit_dialog(host, listing.currentItem().data(Qt.ItemDataRole.UserRole), dialog)
            load()
    def do_delete():
        try:
            row = selected()
            if row and QMessageBox.question(dialog, "删除学习反馈", "删除后不再用于后续建议；已采用的话术和原建议仍保留。确定吗？") == QMessageBox.StandardButton.Yes:
                service.delete(row["id"], row["revision"])
                load()
        except Exception as exc:
            error.setText(f"删除失败（{type(exc).__name__}）")
    listing.currentItemChanged.connect(lambda *_: show())
    only.toggled.connect(lambda *_: load())
    refresh.clicked.connect(load)
    edit.clicked.connect(do_edit)
    delete.clicked.connect(do_delete)
    close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    close.rejected.connect(dialog.reject)
    layout.addWidget(close)
    load()
    dialog.exec()
