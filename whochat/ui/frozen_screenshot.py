"""UI ownership for a frozen screenshot request, separate from live polling."""
from copy import deepcopy
import json
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal

from whochat.services.frozen_screenshot import prepare_context, failure_message
from whochat.services.manual_selection import SelectionError
from whochat.services.reply import context_hash


def current_run(main):
    workflow = getattr(main, '_screenshot_workflow', None)
    run = getattr(workflow, 'run', None)
    return run if run is not None and (run.busy or run.retained()) else None


class FrozenScreenshotRun(QObject):
    ready = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, workflow, request):
        super().__init__(workflow)
        self.workflow, self.main, self.request = workflow, workflow.main, request
        self.busy = True
        self.context = self.source = self.result = None
        self.config = deepcopy(self.main._config)
        self.ready.connect(self._ready, Qt.ConnectionType.QueuedConnection)
        self.failed.connect(self._failed, Qt.ConnectionType.QueuedConnection)

    def start(self):
        main = self.main
        main._manual_generation_request_id = self.request.request_id
        main._manual_generation_started = self.request.started
        main._manual_generation_owner = main._manual_window_owner(self.request.state)
        main._manual_generation_source_result = None
        main._manual_generation_terminal = False
        main._manual_generation_job_id = None
        main._manual_selection_plan = None
        main._manual_review_reason = ''
        main._manual_copy_ticket = None
        main._manual_generation_capture_ids.clear()
        main._set_manual_generation_phase('generating')
        main._render_reply_empty_state('正在读取已框选的固定截图；移动微信不会改变本次输入。')
        main.append_log('frozen_screenshot_started: '+json.dumps({
            'request': self.request.request_id, 'selection': self.request.selection.selection_id,
            'captures': [frame.capture_id for frame in self.request.frames]}, separators=(',', ':')))
        self.future = main._services.pipeline.executor.submit(self._work)
        self.future.add_done_callback(self._done)
        QTimer.singleShot(30000, self, self._timeout)

    def _work(self):
        services = self.main._services
        context, source = prepare_context(self.request, services, services.pipeline.ocr_engine)
        self.request.check()
        # The existing guarded generator remains the ONLY provider entry.
        # It uses the screenshot seal instead of changing live window state.
        result = services.reply_generator.generate(context, self.config)
        self.request.check()
        return context, source, result

    def _done(self, future):
        try:
            context, source, result = future.result()
            self.ready.emit(context, source, result)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, (SelectionError, TimeoutError)) else type(exc).__name__
            # No exception payload from a provider is written into diagnostics.
            try:
                self.failed.emit(reason)
            except RuntimeError:
                pass

    def _ready(self, context, source, result):
        if not self.busy or self.workflow.run is not self:
            return
        if (time.monotonic() >= self.request.deadline
                or not self.main._services.reply_generator.snapshot_authority.valid(context)):
            self._failed('snapshot_source_expired')
            return
        self.context, self.source = context, source
        if not result.allowed or not result.suggestions:
            self._failed(result.status)
            return
        main = self.main
        self.busy = self.workflow.busy = False
        main._manual_generation_source_result = source
        main._last_ocr_messages = list(source.messages)
        main._last_ocr_result_meta = {'job_id': source.job_id, 'created_at': source.created_at,
            'page': source.page.page_type.value, 'page_confidence': source.page.confidence,
            'warning': '本次固定截图 · '+context.contact.display_name}
        # The anchor is the same-frame account evidence already checked before
        # the overlay. Copy still obtains fresh native account and pixel evidence.
        from whochat.services.account_scope import account_avatar_screen_rect
        self.result = result
        main._manual_generation_avatar_anchor = {
            'expected_avatar_fingerprint': source.screen_proof['account_avatar_fingerprint'],
            'expected_avatar_geometry': account_avatar_screen_rect(source.layout),
        }
        main._set_manual_generation_phase('success')
        main._render_reply_suggestions(result, context=context,
            title_override='截图建议 · '+context.contact.display_name)
        self.result = main._suggestion_result
        main._suggestion_context_hash = context_hash(context)
        main._finish_manual_generation('截图建议已完成，属于「'+context.contact.display_name+'」；复制时核对当前微信。', success=True)
        self._log('success')

    def _failed(self, reason):
        if not self.busy:
            return
        self.busy = self.workflow.busy = False
        self.request.cancelled.set()
        self.main._services.reply_generator.snapshot_authority.revoke()
        from whochat.services.status import reply_content_failure_message
        detail = reply_content_failure_message(reason) or failure_message(reason)
        self.main._finish_manual_generation(detail+'（'+reason+'）')
        self._log(reason)

    def _timeout(self):
        if self.busy:
            self._failed('snapshot_request_timeout')

    def _log(self, status):
        self.main.append_log('frozen_screenshot_finished: '+json.dumps({
            'request': self.request.request_id, 'status': status,
            'seconds': round(time.monotonic()-self.request.started, 6)}, separators=(',', ':')))

    def retained(self):
        saved = getattr(self.main, '_suggestion_reply_context', None)
        return bool(self.context is not None and saved is not None
                    and saved.frozen_screenshot_id == self.request.selection.selection_id
                    and self.main._suggestion_result is not None and self.main._suggestion_result.allowed)

    def binding(self, mode, context):
        if (mode != 'reply' or context is None or self.context is None
                or context.frozen_screenshot_id != self.request.selection.selection_id
                or context_hash(context) != context_hash(self.context)):
            return None
        from whochat.ui.main_window import SuggestionActionBinding
        return SuggestionActionBinding(('frozen_screenshot', context.frozen_screenshot_id,
            context.contact.account_scope_id, context.contact.id), mode, context_hash(context), '')

    def binding_current(self, binding):
        saved = getattr(self.main, '_suggestion_reply_context', None)
        profile = self.main._services.reply_generator.display_profile_hash
        return bool(self.retained() and binding is not None
                    and binding == self.main._suggestion_action_binding == self.binding('reply', saved)
                    and profile(self.main._config) == self.main._suggestion_reply_profile
                    and self.main._services.reply_generator.snapshot_authority.valid(saved, display=True))

    def sync_floating(self):
        floating = getattr(self.main, '_floating', None)
        if floating is None or not self.retained():
            return False
        floating.set_copy_binding(self.main._suggestion_action_binding)
        floating.update_context(contact_name=self.context.contact.display_name, group_name='截图时客户',
            status='建议已保留', action='复制时核对当前微信', app_label='微信')
        floating.update_reply_result(self.main._suggestion_result)
        return True

    def cancel(self):
        self.request.cancelled.set()
        self.busy = False
        self.main._services.reply_generator.snapshot_authority.revoke()
