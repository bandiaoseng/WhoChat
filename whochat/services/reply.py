from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from copy import deepcopy
from enum import Enum
from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING
from threading import RLock
from time import monotonic

from whochat.ai.generator import ReplyGenerator, _current_work_remaining
from whochat.ai.dispatch_audit import ProviderPreDispatchBlocked
from whochat.ai.current_work_authorization import cloud_reply_authorized, current_context_source_error
from whochat.ai.prompt import REPLY_POLICY_VERSION
from whochat.ai.photos import photo_source_error, photo_status_message
from whochat.ai.models import (
    ReplyContext,
    ReplyGenerationResult,
    reply_memory_item_count,
    structured_reply_result_error,
)
from whochat.config import AppConfig
from whochat.core.models import ContactStatus
from whochat.core.runtime import PageType
from whochat.diagnostics import append_diagnostics_log
from whochat.services.provider_health import ProviderHealthStore, provider_health_snapshot
from whochat.services.reply_result_cache import reply_cache_profile_hash
from whochat.storage.repositories import GenerationLogRepository

if TYPE_CHECKING:
    from whochat.services.reply_result_cache import ReplyResultCache


class ScreenDisposition(str, Enum):
    CURRENT = "current"
    UNVERIFIED = "unverified"
    OWNER_CHANGED = "owner_changed"


def screen_disposition(check) -> ScreenDisposition:
    """Observation loss is not a customer change or a grant to use old text."""
    if check.unchanged:
        return ScreenDisposition.CURRENT
    if check.reason in {"source_owner_changed", "source_generation_invalid",
                        "account_avatar_pixels_changed"}:
        return ScreenDisposition.OWNER_CHANGED
    return ScreenDisposition.UNVERIFIED


def account_confirmation_message(result) -> str:
    """Actionable unbound-account state, never authority to pick an account.

    Avatar similarity can narrow candidates, not select among four logins.
    Transient second-frame revalidation must retain its normal verification path.
    """
    if getattr(result, "account_scope_id", None) or not getattr(result, "target_instance_id", ""):
        return ""
    reason = getattr(result, "account_scope_reason", "")
    labels = {
        "visual_match_ambiguous": "当前微信头像匹配到多个账号，无法确定本窗口属于哪个账号",
        "visual_binding_not_found": "当前微信窗口尚未关联到本机保存的账号",
        "runtime_instance_unbound": "当前微信窗口尚未绑定本地账号",
        "unbound": "当前微信窗口尚未绑定本地账号",
    }
    label = labels.get(reason)
    if label:
        return label + "；请在「设置→绑定当前微信账号」确认本窗口所属账号，再点击生成建议。本次未调用 AI，也未扣额度"
    if reason == "visual_match_active_on_other_window":
        return "当前账号候选已关联到另一个活动微信窗口；请先核对各窗口的账号归属，本次未调用 AI，也未扣额度"
    return ""


def account_cancellation_message(last_blocker, runtime, authority=None) -> str:
    """Keep a current account rejection visible when its epoch cancels a job."""
    reason = str(last_blocker or "")
    try:
        checkpoint = authority.account_identity_checkpoint(runtime.state.window) if authority is not None else None
        if checkpoint is not None and checkpoint.identity_state == "revalidating":
            native_reason = str(getattr(checkpoint, "reason", ""))
            if native_reason.startswith("native_account_"):
                reason = native_reason
    except Exception:
        pass
    if reason == "account_avatar_revalidation_failed" or reason.startswith("native_account_"):
        return capture_failure_message(reason)
    return "微信窗口或账号身份已变化，本次已取消"


def capture_failure_message(reason: str) -> str:
    labels = {
        "native_account_evidence_unavailable": "未能读取当前工作微信的账号身份证据，已停止本次操作；请保持该账号正常登录后重试",
        "native_account_evidence_ambiguous": "当前微信进程的账号证据存在冲突，无法唯一确认账号；已停止本次操作，请核对登录状态",
        "native_account_mapping_unverified": "当前微信账号尚无经过核实的资料映射，已停止本次操作；请先核对账号设置",
        "native_account_owner_changed": "微信窗口所属进程或登录状态已变化，本次操作已取消；请重新核对当前客户后生成",
        "native_account_binding_conflict": "当前登录账号与已有窗口绑定不一致，已停止本次操作；请核对账号设置，系统未自动更改绑定",
        "native_account_avatar_unavailable": "账号已核对，但当前头像画面不清晰，无法建立本次画面核验依据；请保持微信窗口完整可见",
        "native_account_confirmation_pending": "工作微信账号正在重新核验，本次操作已停止；请核对当前客户后重新生成",
        "account_avatar_revalidation_failed": "聊天已读取，但工作微信账号头像连续核验不一致；已停止重复读取。请核对当前微信账号及设置中的绑定；未调用 AI、未扣额度",
        "history_navigation_present": "当前停留在历史消息位置；请点击微信「跳转到最新消息」，回到底部后重新生成",
        "input_boundary_unverified": "无法确认聊天区与输入框的边界；请让微信聊天区和输入框完整显示后重新生成",
        "untranscribed_voice_present": "聊天中有语音尚未读到转写文字；请先在微信中转为文字，并让转写内容完整显示后，再手动点击生成建议",
        "message_text_low_confidence": "聊天中有文字未能可靠识别，本次已停止重复读取；请调整微信窗口大小，让文字清晰显示后手动重试",
        "middle_or_tail_incomplete": "聊天中间或最后一条未完整读取，请让相关内容完整显示后重试",
        "layout_or_coverage_incomplete": "聊天或语音转文字未完整读取，请让气泡和窗口边界完整显示后重试",
        "current_view_unavailable": "聊天已读取，但来源核验尚未完成",
        "identity_pending": "客户标题或账号尚未稳定确认",
        "identity_lease_invalid": "客户或账号的本次窗口核验未完成",
    }
    return labels.get(reason, "尚未取得两次有效的客户与聊天核验")


def manual_capture_boundary_reason(result, state, request_id: int) -> str:
    """Read bound negative evidence, never grant a source or infer from text.

    The caller additionally checks the current request's capture job and the
    runtime lifecycle. This proof cannot authorize generation or copying.
    """
    from whochat.core.current_conversation import SCREEN_CAPTURE_SCHEMA, SCREEN_TTL_SECONDS

    proof = getattr(result, "screen_proof", None)
    if not isinstance(proof, dict):
        return ""
    code = proof.get("rejection_code")
    if code not in {"history_navigation_present", "input_boundary_unverified", "untranscribed_voice_present", "message_text_low_confidence"}:
        return ""
    title_boundary_only = (code == "input_boundary_unverified"
                           and proof.get("rejection_scope") == "title_verified_boundary")
    if (proof.get("schema") != SCREEN_CAPTURE_SCHEMA or proof.get("valid") is not False
            or type(request_id) is not int or request_id <= 0
            or type(proof.get("manual_request_id")) is not int
            or proof["manual_request_id"] != request_id
            or any(proof.get(key) is not True for key in (
                "negative_capture_evidence", "title_verified", "account_verified"))
            or proof.get("capture_verified") is not (False if title_boundary_only else True)
            or proof.get("rejection_scope") != ("title_verified_boundary" if title_boundary_only else "full_capture")):
        return ""
    try:
        result_bindings = {
            "source_job_id": result.job_id, "snapshot_hash": result.snapshot_hash,
            "title_exact_hash": result.title_exact_hash,
            "account_scope_id": result.account_scope_id, "target_instance_id": result.target_instance_id,
            "hwnd": result.hwnd, "process_id": result.process_id,
            "window_generation": result.window_generation, "capture_generation": result.capture_generation,
            "account_identity_epoch": result.account_identity_epoch,
            "account_identity_runtime_key": result.account_identity_runtime_key,
        }
        if (any(value is None or value == "" or proof.get(key) != value
                for key, value in result_bindings.items())
                or any(proof[key] != value for key, value in {
                    "hwnd": state.window.hwnd, "process_id": state.window.process_id,
                    "window_generation": state.window_generation, "capture_generation": state.capture_generation,
                    "account_identity_epoch": state.account_identity_epoch,
                    "account_identity_runtime_key": state.account_identity_runtime_key,
                }.items())
                or (not title_boundary_only and result.page.page_type != PageType.CHAT_DM)):
            return ""
        observed = datetime.fromisoformat(proof["observed_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(proof["expires_at"].replace("Z", "+00:00"))
        captured = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if (observed.tzinfo is None or expires.tzinfo is None or captured.tzinfo is None
                or observed != captured or not observed <= now < expires
                or not 0 < (expires - observed).total_seconds() <= SCREEN_TTL_SECONDS):
            return ""
        if code == "history_navigation_present":
            if (proof.get("history_navigation_present") is not True
                    or not isinstance(proof.get("history_navigation_evidence"), dict)
                    or not proof["history_navigation_evidence"]):
                return ""
        elif code == "untranscribed_voice_present":
            if (type(proof.get("untranscribed_voice_count")) is not int
                    or proof["untranscribed_voice_count"] <= 0
                    or proof.get("input_boundary_verified") is not True
                    or proof.get("history_navigation_present") is not False):
                return ""
        elif code == "message_text_low_confidence":
            from whochat.ocr.low_confidence import low_confidence_messages
            count = proof.get("low_confidence_count")
            attempts = proof.get("low_confidence_recovery_attempts")
            if (type(count) is not int or count <= 0 or type(attempts) is not int or attempts not in {0, 2}
                    or proof.get("input_boundary_verified") is not True
                    or proof.get("history_navigation_present") is not False
                    or proof.get("parsed_coverage_complete") is not True
                    or type(proof.get("unaccounted_box_count")) is not int or proof["unaccounted_box_count"] != 0
                    or type(proof.get("untranscribed_voice_count")) is not int or proof["untranscribed_voice_count"] != 0
                    or len(low_confidence_messages(result.messages, result.layout)) != count):
                return ""
        elif proof.get("input_boundary_verified") is not False:
            return ""
    except (AttributeError, KeyError, TypeError, ValueError):
        return ""
    return code


def manual_capture_boundary_message(reason: str, *, reviewing: bool = False) -> str:
    if reason not in {"history_navigation_present", "input_boundary_unverified", "untranscribed_voice_present", "message_text_low_confidence"}:
        return ""
    suffix = ("；已有结果未恢复展示，本次核验未新增 AI 调用" if reviewing
              else "；本次未调用 AI，也未扣额度")
    return capture_failure_message(reason) + suffix


class ReplyGenerationService:
    def __init__(
        self,
        audit_logs: GenerationLogRepository,
        generator: ReplyGenerator | None = None,
        health_store: ProviderHealthStore | None = None,
        identity_lease_validator: Callable[[ReplyContext], bool] | None = None,
        result_cache: ReplyResultCache | None = None,
        display_lease_validator: Callable[[ReplyContext], bool] | None = None,
    ) -> None:
        self.audit_logs = audit_logs
        self.generator = generator or ReplyGenerator()
        self.health_store = health_store or ProviderHealthStore()
        self.identity_lease_validator = identity_lease_validator
        self.result_cache = result_cache
        self.display_lease_validator = display_lease_validator
        # Private responses do not grant current-screen or clipboard authority.
        # Live recovery needs new evidence; the exact manual request may also
        # expose its immutable answer as an original-customer display snapshot.
        self._pending_responses: dict[tuple, tuple[datetime, ReplyGenerationResult]] = {}
        self._manual_display_responses: dict[str, tuple[datetime, tuple, ReplyGenerationResult]] = {}
        self._pending_lock = RLock()
        self._manual_attempts: dict[str, float] = {}
        self.last_audit_id: str | None = None
        self._last_cloud_attempt_at: datetime | None = None
        self._suggestion_cache: dict[tuple[str, str, str, str, str], ReplyGenerationResult] = {}
        self._suggestion_cache_expires_at: dict[tuple[str, str, str, str, str], datetime] = {}
        snapshot = self.health_store.load()
        self.consecutive_cloud_failures = snapshot.consecutive_failures
        self.provider_backoff_until = _parse_ts(snapshot.backoff_until) if snapshot.backoff_until else None
        self.provider_health_status = snapshot.status or "healthy"

    def generate(self, context: ReplyContext, config: AppConfig) -> ReplyGenerationResult:
        if _current_work_deadline_expired(context, config):
            return ReplyGenerationResult(False, "blocked:current_work_timeout", [], config.ai.provider)
        source_error = current_context_source_error(context)
        if source_error:
            return ReplyGenerationResult(False, f"blocked:{source_error}", [], config.ai.provider)
        photo_error = photo_source_error(context)
        if photo_error:
            return ReplyGenerationResult(False, "blocked:photo_source: " + photo_status_message(photo_error), [], config.ai.provider)
        if context.current_message_view is not None and self.identity_lease_validator is None:
            return ReplyGenerationResult(False, "blocked:current_view_live_lease_required", [], config.ai.provider)
        if self.identity_lease_validator is not None and not self.identity_lease_validator(context):
            return ReplyGenerationResult(False, "blocked:identity_stability_lease", [], config.ai.provider)
        digest = request_context_hash(context)
        cloud_may_attempt = _cloud_request_may_be_attempted(context, config)
        result = self._blocked_by_request_policy(digest, config, context=context) if cloud_may_attempt else None
        did_dispatch = False
        cache_key = _suggestion_cache_key(digest, config)
        # Freeze this identity while the pre-dispatch lease is still valid.
        # Never authorize storage/recovery from a mutated post-response context.
        pending_key = self._pending_response_key(context, config, digest)
        pending_deadline = datetime.now(timezone.utc) + timedelta(seconds=120)
        display_request_id = context.manual_retry_id if self._valid_manual_attempt(context, config) else ""
        # Private response retention is not a live source lease or a new HTTP
        # deadline. Recovery ALWAYS requires fresh source+identity validation.
        # Expiring this together with the old click made successful responses
        # disappear just before a fresh explicit retry could recover them.
        if cloud_may_attempt and (result is not None or context.manual_retry_id):
            # Pending recovery belongs to a verified manual click (or the
            # explicit local-verification phase), not legacy/background cache
            # lookup. Preserve normal cache invalidation and expiry there.
            cached = (self.recover_cached_only(context, config)
                      if self._valid_manual_attempt(context, config) else None)
            if cached is None:
                cached = self._read_cached_result(digest, context, config)
            if cached is not None:
                cache_error = structured_reply_result_error(cached, context=context)
                if cache_error:
                    self._suggestion_cache.pop(cache_key, None)
                    self._suggestion_cache_expires_at.pop(cache_key, None)
                    result = ReplyGenerationResult(
                        False,
                        f"blocked:cache_invalid: 缓存结果不完整（{cache_error}）",
                        [],
                        cached.provider,
                    )
                else:
                    result = replace(cached, status="ai_cached:unchanged_context", performance_trace={
                        "prompt_profile": {"policy_version": REPLY_POLICY_VERSION},
                        "cache_profile": {"hit": True},
                    })
        if result is None and cloud_may_attempt and context.manual_retry_id:
            if not self._claim_manual_attempt(context, config):
                result = ReplyGenerationResult(False, "blocked:manual_attempt_consumed", [], config.ai.provider)
        if result is None:
            previous_attempt_at = self._last_cloud_attempt_at
            dispatch_started_at = None
            if cloud_may_attempt:
                dispatch_started_at = datetime.now(timezone.utc)
                self._last_cloud_attempt_at = dispatch_started_at
                did_dispatch = True
            try:
                result = self.generator.generate(context, config)
            except ProviderPreDispatchBlocked as exc:
                # Only the trusted transport's typed refusal proves no HTTP
                # began. Do not poison cooldown, health or same-context retry.
                # The durable ledger remains authoritative and is never reset
                # or refunded here, even if it failed after reserving a slot.
                did_dispatch = False
                if self._last_cloud_attempt_at == dispatch_started_at:
                    self._last_cloud_attempt_at = previous_attempt_at
                result = ReplyGenerationResult(False, f"blocked:{exc.reason}", [], config.ai.provider)
            except Exception as exc:
                # Invalidating display rights never refunds an attempted
                # request. Record only the exception type, not private text.
                if did_dispatch:
                    self._append_audit(context, config, digest, ReplyGenerationResult(
                        False, f"attempted:exception:v2:{type(exc).__name__}", [], config.ai.provider
                    ))
                raise
            if cloud_may_attempt and result.allowed:
                validation_error = structured_reply_result_error(result, context=context)
                if validation_error:
                    result = ReplyGenerationResult(
                        False,
                        f"AI 返回缺少完整的结构化分析字段（{validation_error}）",
                        [],
                        result.provider,
                    )
                elif did_dispatch and pending_key is not None and not _current_work_deadline_expired(context, config):
                    self._remember_pending_response(pending_key, pending_deadline, result,
                                                    manual_request_id=display_request_id)
            if cloud_may_attempt and did_dispatch:
                # Provider health describes the actual parsed provider verdict,
                # never a later display/capture-lifecycle downgrade.
                self._apply_provider_health(result, config)
            if self.identity_lease_validator is not None and not self.identity_lease_validator(context):
                result = ReplyGenerationResult(False, "blocked:stale_identity_lease", [], config.ai.provider)
        if _current_work_deadline_expired(context, config):
            result = ReplyGenerationResult(False, "blocked:current_work_timeout", [], config.ai.provider,
                                           performance_trace=result.performance_trace)
        source_error = current_context_source_error(context)
        if not _current_work_deadline_expired(context, config) and (source_error or (
            self.identity_lease_validator is not None and not self.identity_lease_validator(context)
        )):
            result = ReplyGenerationResult(False, f"blocked:{source_error or 'stale_identity_lease'}", [], config.ai.provider)
            if not did_dispatch:
                return result
        if result.allowed:
            trace = dict(result.performance_trace)
            profile = dict(trace.get("prompt_profile") or {})
            profile["policy_version"] = REPLY_POLICY_VERSION
            trace["prompt_profile"] = profile
            result = replace(result, performance_trace=trace)
        if cloud_may_attempt and result.allowed and result.suggestions and not _current_work_deadline_expired(context, config):
            if did_dispatch:
                ttl = min(24 * 60 * 60, max(0.0, float(config.ai.dedupe_context_minutes) * 60))
                now = datetime.now(timezone.utc)
                for old_key, deadline in tuple(self._suggestion_cache_expires_at.items()):
                    if deadline <= now:
                        self._suggestion_cache.pop(old_key, None)
                        self._suggestion_cache_expires_at.pop(old_key, None)
                if (ttl > 0 and not _current_work_deadline_expired(context, config)
                        and (cache_key in self._suggestion_cache or len(self._suggestion_cache) < 512)):
                    self._suggestion_cache[cache_key] = result
                    self._suggestion_cache_expires_at[cache_key] = now + timedelta(seconds=ttl)
            if did_dispatch and self.result_cache is not None and not _current_work_deadline_expired(context, config):
                try:
                    self.result_cache.put(digest, config, context, result)
                except Exception as exc:
                    append_diagnostics_log("reply_cache", f"cache_write_failed:{type(exc).__name__}")
        if _current_work_deadline_expired(context, config):
            result = ReplyGenerationResult(False, "blocked:current_work_timeout", [], config.ai.provider,
                                           performance_trace=result.performance_trace)
        audit_result = result
        if did_dispatch and result.status.startswith("blocked:"):
            audit_result = replace(result, status="attempted:" + result.status)
        self._append_audit(context, config, digest, audit_result)
        return result

    @staticmethod
    def _pending_response_key(context: ReplyContext, config: AppConfig, digest: str) -> tuple | None:
        view, runtime = context.current_message_view, context.runtime
        if view is None or context.contact is None:
            return None
        return (*_suggestion_cache_key(digest, config), view.account_scope_id, view.contact_id,
                view.conversation_id, view.target_instance_id, view.session_id,
                runtime.window.hwnd, runtime.window.process_id, runtime.window_generation,
                runtime.account_identity_epoch, runtime.account_identity_runtime_key)

    def _remember_pending_response(self, key: tuple, deadline: datetime, result: ReplyGenerationResult,
                                   *, manual_request_id: str = "") -> None:
        now = datetime.now(timezone.utc)
        with self._pending_lock:
            self._pending_responses = {k: value for k, value in self._pending_responses.items() if value[0] > now}
            if deadline > now and (key in self._pending_responses or len(self._pending_responses) < 32):
                # Never extend an earlier response's lifetime on cache recovery.
                self._pending_responses[key] = (deadline, deepcopy(result))
            self._manual_display_responses = {k: v for k, v in self._manual_display_responses.items() if v[0] > now}
            if (manual_request_id in self._manual_attempts and deadline > now
                    and (manual_request_id in self._manual_display_responses or len(self._manual_display_responses) < 32)):
                self._manual_display_responses[manual_request_id] = (deadline, key, deepcopy(result))

    @staticmethod
    def display_profile_hash(config: AppConfig) -> str:
        return reply_cache_profile_hash(config)

    def has_pending_response(self, context: ReplyContext, config: AppConfig) -> bool:
        """Presence only, never plaintext or authority to display/request."""
        key = self._pending_response_key(context, config, request_context_hash(context))
        with self._pending_lock:
            entry = self._pending_responses.get(key)
            return bool(entry is not None and entry[0] > datetime.now(timezone.utc))

    def completed_snapshot_for_display(
        self, context: ReplyContext, config: AppConfig,
    ) -> ReplyGenerationResult | None:
        """Read an already validated response for its ORIGINAL manual request.

        Presentation only: no current-screen assertion, source renewal, cache
        write, grant, HTTP or clipboard permission. The UI must keep the frozen
        customer label and reject superseded jobs. Copy retains its independent
        live account/customer/content checks. Legacy and background callers do
        not acquire this manual-display path.
        """
        if not self._valid_manual_attempt(context, config):
            return None
        key = self._pending_response_key(context, config, request_context_hash(context))
        with self._pending_lock:
            entry = self._manual_display_responses.get(context.manual_retry_id)
            result = deepcopy(entry[2]) if (entry is not None and entry[1] == key
                                           and entry[0] > datetime.now(timezone.utc)) else None
        if (result is None or not result.allowed or result.provider != config.ai.provider
                or structured_reply_result_error(result, context=context)
                or _current_work_deadline_expired(context, config)):
            return None
        trace = dict(result.performance_trace)
        trace['display_snapshot'] = {'original_request_only': True, 'copy_requires_live_validation': True}
        return replace(result, status='ai_completed:original_customer_snapshot', performance_trace=trace)

    def recover_cached_only(self, context: ReplyContext, config: AppConfig) -> ReplyGenerationResult | None:
        """Fresh source+DB/live lease on BOTH sides; never dispatch or reserve HTTP.

        In particular this works with an exhausted HTTP ledger. Neither a hit
        nor a miss consults request policy, changes budget, or retries Provider.
        """
        def fresh() -> bool:
            try:
                return bool(current_context_source_error(context) is None
                            and not _current_work_deadline_expired(context, config)
                            and _cloud_request_may_be_attempted(context, config)
                            and callable(self.identity_lease_validator)
                            and self.identity_lease_validator(context))
            except Exception:
                return False
        if not fresh():
            return None
        digest = request_context_hash(context)
        key = self._pending_response_key(context, config, digest)
        with self._pending_lock:
            entry = self._pending_responses.get(key)
            result = deepcopy(entry[1]) if entry is not None and entry[0] > datetime.now(timezone.utc) else None
        if result is None:
            result = self._read_cached_result(digest, context, config)
        if (result is None or not result.allowed or result.provider != config.ai.provider
                or structured_reply_result_error(result, context=context) or not fresh()):
            return None
        return replace(result, status="ai_cached:locally_revalidated", performance_trace={
            "prompt_profile": {"policy_version": REPLY_POLICY_VERSION},
            "cache_profile": {"hit": True, "locally_revalidated": True},
        })

    def recover_pending_for_display(
        self, original: ReplyContext, current: ReplyContext, config: AppConfig,
    ) -> ReplyGenerationResult | None:
        """An already dispatched response, NOT a new cloud/capture capability.

        Only the exact still-latest original source may survive focus/coverage.
        A fresh rebuilt local business/memory/knowledge hash must match. Any
        negative source, geometry, checkpoint, session or TTL boundary fails.
        """
        def display_valid() -> bool:
            try:
                return bool(callable(self.display_lease_validator)
                            and not _current_work_deadline_expired(original, config)
                            and not _current_work_deadline_expired(current, config)
                            and current_context_source_error(original) is None
                            and current_context_source_error(current) in {None, "current_view_screen_runtime_mismatch"}
                            and context_hash(original) == context_hash(current)
                            and self.display_lease_validator(original))
            except Exception:
                return False
        if not display_valid():
            return None
        key = self._pending_response_key(original, config, request_context_hash(original))
        with self._pending_lock:
            entry = self._pending_responses.get(key)
            result = deepcopy(entry[1]) if entry is not None and entry[0] > datetime.now(timezone.utc) else None
        if (result is None or not result.allowed or result.provider != config.ai.provider
                or structured_reply_result_error(result, context=original) or not display_valid()):
            return None
        return replace(result, status="ai_cached:display_only_revalidated", performance_trace={
            "prompt_profile": {"policy_version": REPLY_POLICY_VERSION},
            "cache_profile": {"hit": True, "display_only": True},
        })

    def _read_cached_result(self, digest: str, context: ReplyContext, config: AppConfig) -> ReplyGenerationResult | None:
        if _current_work_deadline_expired(context, config):
            return None
        if current_context_source_error(context):
            return None
        if context.current_message_view is not None and self.identity_lease_validator is None:
            return None
        if self.identity_lease_validator is not None:
            try:
                if not self.identity_lease_validator(context):
                    return None
            except Exception:
                return None
        key = _suggestion_cache_key(digest, config)
        cached = self._suggestion_cache.get(key)
        expires_at = self._suggestion_cache_expires_at.get(key)
        if cached is not None and (expires_at is None or datetime.now(timezone.utc) >= expires_at):
            self._suggestion_cache.pop(key, None)
            self._suggestion_cache_expires_at.pop(key, None)
            cached = None
        if cached is not None:
            # Keep the existing explicit invalid-cache terminal, not a hidden
            # provider retry, when an in-process entry fails the new policy.
            return cached
        if self.result_cache is None:
            return None
        try:
            cached = self.result_cache.get(digest, config, context)
        except Exception as exc:
            append_diagnostics_log("reply_cache", f"cache_read_failed:{type(exc).__name__}")
            return None
        # Keep persisted entries on their original database TTL. Copying them
        # into a new process-local TTL would silently extend their lifetime.
        return cached

    def _append_audit(self, context: ReplyContext, config: AppConfig, digest: str, result: ReplyGenerationResult) -> None:
        audit = self.audit_logs.append(
            contact_id=context.contact.id if context.contact else None,
            strategy_id=context.strategy.id if context.strategy else None,
            provider=result.provider,
            model=config.ai.model if config.ai.provider != "Disabled" else "",
            allowed=result.allowed,
            status=result.status,
            suggestion_count=len(result.suggestions),
            risk_summary=_risk_summary(result),
            context_hash=digest,
            page_type=context.runtime.page.page_type.value,
            page_confidence=context.runtime.page.confidence,
            message_count=len(context.effective_messages),
            memory_count=reply_memory_item_count(context),
        )
        self.last_audit_id = audit.id

    @staticmethod
    def _valid_manual_attempt(context: ReplyContext | None, config: AppConfig) -> bool:
        if context is None or context.current_message_view is None:
            return False
        token = context.manual_retry_id
        return bool(isinstance(token, str) and len(token) == 32
                    and all(c in "0123456789abcdef" for c in token)
                    and context.current_work_started_monotonic is not None
                    and context.current_work_deadline_monotonic is not None
                    and not _current_work_deadline_expired(context, config))

    def _claim_manual_attempt(self, context: ReplyContext, config: AppConfig) -> bool:
        # Single-flight per click, including the first request. Cache recovery
        # never claims a token. Expired tokens cannot dispatch with old clocks.
        if not self._valid_manual_attempt(context, config):
            return False
        with self._pending_lock:
            now = monotonic()
            self._manual_attempts = {k: expiry for k, expiry in self._manual_attempts.items() if expiry > now}
            if context.manual_retry_id in self._manual_attempts or len(self._manual_attempts) >= 512:
                return False
            self._manual_attempts[context.manual_retry_id] = context.current_work_deadline_monotonic
            return True

    def _blocked_by_request_policy(self, digest: str, config: AppConfig,
                                   *, context: ReplyContext | None = None) -> ReplyGenerationResult | None:
        now = datetime.now(timezone.utc)
        self.refresh_provider_health(now)
        if self.provider_backoff_until and now < self.provider_backoff_until:
            return ReplyGenerationResult(False, "blocked:provider_backoff: AI Provider 连续失败，退避中", [], config.ai.provider)

        cooldown = max(0, config.ai.request_cooldown_seconds)
        if self._last_cloud_attempt_at and (now - self._last_cloud_attempt_at).total_seconds() < cooldown:
            return ReplyGenerationResult(False, "blocked:cooldown: 云端 AI 请求过于频繁，已暂缓", [], config.ai.provider)

        dedupe_minutes = max(0, config.ai.dedupe_context_minutes)
        if dedupe_minutes:
            latest_attempt = getattr(self.audit_logs, "latest_cloud_attempt_for_context", None)
            latest = latest_attempt(digest) if callable(latest_attempt) else self.audit_logs.latest_for_context(digest)
            if latest and latest.provider in {"OpenAI", "OpenAI Compatible"} and not latest.status.startswith("blocked:"):
                latest_at = _parse_ts(latest.ts)
                if latest_at and now - latest_at < timedelta(minutes=dedupe_minutes):
                    # Dedupe suppresses automatic/background repeats, not a new
                    # explicit verified click. A cache hit still wins before
                    # dispatch; the one-shot click token, all live source gates,
                    # cooldown/backoff/daily limit and transport ledger remain.
                    if not self._valid_manual_attempt(context, config):
                        return ReplyGenerationResult(False, "blocked:duplicate_context: 相同上下文近期已经请求过云端 AI", [], config.ai.provider)

        daily_limit = max(0, config.ai.max_daily_cloud_requests)
        if daily_limit:
            day_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc).isoformat()
            used = self.audit_logs.count_cloud_attempts_since(day_start)
            if used >= daily_limit:
                return ReplyGenerationResult(False, "blocked:daily_limit: 今日云端 AI 请求已达到上限", [], config.ai.provider)
        return None

    def dispatch_delay_seconds(self, context: ReplyContext, config: AppConfig) -> float:
        """Delay an unstarted latest request without disabling rate limits.

        No worker, model request or retry is started here. The scheduler keeps
        one replaceable pending item, then generate rechecks every safety gate.
        """
        if current_context_source_error(context) or not _cloud_request_may_be_attempted(context, config):
            return 0.0
        if self.identity_lease_validator is not None:
            try:
                if not self.identity_lease_validator(context):
                    return 0.0
            except Exception:
                return 0.0
        digest = request_context_hash(context)
        cached = self._read_cached_result(digest, context, config)
        if cached is not None and structured_reply_result_error(cached, context=context) is None:
            return 0.0
        if self._last_cloud_attempt_at is None:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - self._last_cloud_attempt_at).total_seconds()
        return max(0.0, float(max(0, config.ai.request_cooldown_seconds)) - elapsed)

    def _apply_provider_health(self, result: ReplyGenerationResult, config: AppConfig) -> None:
        if result.allowed:
            self.consecutive_cloud_failures = 0
            self.provider_backoff_until = None
            self.provider_health_status = "healthy"
            self._persist_provider_health()
            return
        if result.status.startswith("blocked:"):
            return
        self.consecutive_cloud_failures += 1
        threshold = max(1, config.ai.failure_backoff_threshold)
        if self.consecutive_cloud_failures >= threshold:
            minutes = max(0, config.ai.failure_backoff_minutes)
            self.provider_backoff_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            self.provider_health_status = "backoff"
        else:
            self.provider_health_status = f"degraded:{self.consecutive_cloud_failures}/{threshold}"
        self._persist_provider_health()

    def provider_health_summary(self) -> str:
        until = self.provider_backoff_until.isoformat() if self.provider_backoff_until else "-"
        return (
            f"status={self.provider_health_status} "
            f"failures={self.consecutive_cloud_failures} "
            f"backoff_until={until}"
        )

    def refresh_provider_health(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        if self.provider_backoff_until and now >= self.provider_backoff_until:
            before = self.provider_health_summary()
            self.provider_backoff_until = None
            self.provider_health_status = "recovering"
            self._persist_provider_health()
            after = self.provider_health_summary()
            append_diagnostics_log("ai_provider", f"provider_health_recovering before={before} after={after}")
        return self.provider_health_summary()

    def reset_provider_health(self, reason: str = "manual") -> str:
        before = self.provider_health_summary()
        self.consecutive_cloud_failures = 0
        self.provider_backoff_until = None
        self.provider_health_status = "healthy"
        self._persist_provider_health()
        after = self.provider_health_summary()
        append_diagnostics_log("ai_provider", f"provider_health_reset reason={reason} before={before} after={after}")
        return after

    def _persist_provider_health(self) -> None:
        snapshot = provider_health_snapshot(
            self.provider_health_status,
            self.consecutive_cloud_failures,
            self.provider_backoff_until,
        )
        self.health_store.save(snapshot)


def request_context_hash(context: ReplyContext) -> str:
    base = context_hash(context)
    if not context.writing_feedback:
        return base
    data = json.dumps(context.writing_feedback, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((base + "|writing_feedback_v1|" + data).encode("utf-8")).hexdigest()


def context_hash(context: ReplyContext) -> str:
    payload = {
        "reply_policy_version": REPLY_POLICY_VERSION,
        "photo_input_policy": "visible_photo_v1",
        "photo_input": [(p.ordinal, p.content_hash, p.speaker) for p in context.photos],
        "photo_error": context.photo_error,
        "current_work_mode": context.current_work_mode,
        "local_date": datetime.now().astimezone().date().isoformat(),
        "account_scope_id": context.contact.account_scope_id if context.contact else None,
        "contact_id": context.contact.id if context.contact else None,
        "strategy_id": context.strategy.id if context.strategy else None,
        "strategy_behavior": ({
            "name": context.strategy.name,
            "goal": context.strategy.goal,
            "mode": context.strategy.mode,
            "tone": context.strategy.tone,
            "avoid": context.strategy.avoid,
            "reply_variants": context.strategy.reply_variants,
            "requires_manual_reply": context.strategy.requires_manual_reply,
            "archived": context.strategy.archived,
        } if context.strategy is not None else None),
        "messages": [
            {
                "speaker": message.speaker.value,
                "fingerprint": message.fingerprint,
                "text": " ".join(message.text.split()),
                "sender_name": message.sender_name,
                "message_time": message.message_time,
                "observed_at": message.observed_at,
                "partial": message.partial,
            }
            for message in context.messages[-30:]
        ],
        "memories": [
            {
                "id": memory.id,
                "status": memory.status.value,
                "kind": memory.kind.value,
                "updated_at": memory.updated_at,
            }
            for memory in context.memories[:30]
        ],
        "reply_memory_context": (
            context.memory_context.as_payload() if context.memory_context is not None else None
        ),
        "verified_general_agronomy_knowledge": (
            context.knowledge_context.as_payload() if context.knowledge_context is not None else None
        ),
        "current_message_id": context.current_message_id,
    }
    if context.current_message_view is not None:
        # Capture/job/observation IDs prove provenance, not a new business
        # conversation. Repaints must not change the request/cache revision.
        payload["messages"] = context.current_message_view.reply_business_payload()
        payload["current_message_id"] = ""
        payload["reply_memory_context"] = (
            context.memory_context.business_payload() if context.memory_context is not None else None
        )
        if payload["reply_memory_context"] is not None:
            # Hash-only projection: never mutate the source/lease revision.
            memory_payload = deepcopy(payload["reply_memory_context"])
            current_view = memory_payload.get("current_view")
            if isinstance(current_view, dict):
                current_view["content_revision"] = context.current_message_view.reply_business_revision
            payload["reply_memory_context"] = memory_payload
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _current_work_deadline_expired(context: ReplyContext, config: AppConfig) -> bool:
    # Legacy contexts acquire the bounded attempt in ReplyTaskService. Actual
    # current-window contexts carry the UI recognition clock from the outset.
    if context.current_work_started_monotonic is None and context.current_work_deadline_monotonic is None:
        return False
    return _current_work_remaining(context, config)[1] is not None


def _suggestion_cache_key(digest: str, config: AppConfig) -> tuple[str, str, str, str, str]:
    return (
        digest,
        config.ai.provider,
        config.ai.base_url.strip().rstrip("/"),
        config.ai.model.strip(),
        reply_cache_profile_hash(config),
    )


def _cloud_request_may_be_attempted(context: ReplyContext, config: AppConfig) -> bool:
    if config.ai.provider not in {"OpenAI", "OpenAI Compatible"} or not config.ai.api_key:
        return False
    if config.privacy.manual_protection_blocks_replies and context.strategy and context.strategy.requires_manual_reply:
        return False
    if config.capture.pause_ai_on_unknown_page and context.runtime.page.page_type not in {PageType.CHAT_DM, PageType.CHAT_GROUP}:
        return False
    if context.contact is None or not cloud_reply_authorized(context):
        return False
    if context.contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED} or context.contact.merged_into:
        return False
    return True


def _parse_ts(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _risk_summary(result: ReplyGenerationResult) -> str:
    if not result.suggestions:
        return ""
    risks: dict[str, int] = {}
    for suggestion in result.suggestions:
        risks[suggestion.risk] = risks.get(suggestion.risk, 0) + 1
    return ",".join(f"{risk}:{count}" for risk, count in sorted(risks.items()))
