from __future__ import annotations

from dataclasses import replace
import json
from math import isfinite
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from whochat.ai.models import (
    AIConnectionTestResult,
    ReplyContext,
    ReplyGenerationResult,
    ReplySuggestion,
    reply_generation_result_from_json,
    project_own_message_for_drafting,
    UNREVIEWED_QUANTITY_MARKER,
    reply_mode_policy,
)
from whochat.ai.prompt import REPLY_POLICY_VERSION, build_prompt_preview
from whochat.ai.current_work_authorization import cloud_reply_authorized, current_context_source_error
from whochat.ai.photos import multimodal_user_content, photo_source_error, photo_status_message
from whochat.config import AppConfig
from whochat.core.models import ContactStatus, Speaker, utc_now_iso
from whochat.core.runtime import PageType
from whochat.diagnostics import append_diagnostics_log
from whochat.security.redaction import redact_diagnostics_text


class ReplyGenerator:
    def generate(self, context: ReplyContext, config: AppConfig) -> ReplyGenerationResult:
        blocked = _blocking_reason(context, config)
        if blocked:
            return ReplyGenerationResult(False, blocked, [], config.ai.provider)

        if config.ai.provider in {"OpenAI", "OpenAI Compatible"} and config.ai.api_key:
            return _generate_openai_compatible(context, config)

        return _generate_local_preview(context, config)


def test_ai_connection(config: AppConfig) -> AIConnectionTestResult:
    provider = config.ai.provider
    if provider == "Disabled":
        return AIConnectionTestResult(True, "disabled", provider, "AI 已禁用；不会发起网络请求")
    if provider not in {"OpenAI", "OpenAI Compatible"}:
        return AIConnectionTestResult(True, "local_preview", provider, "本地预览模式可用；不会发起网络请求")
    if not config.ai.api_key:
        return AIConnectionTestResult(False, "missing_api_key", provider, "缺少 API Key")
    if not config.ai.model.strip():
        return AIConnectionTestResult(False, "missing_model", provider, "缺少模型名称")
    if not config.ai.base_url.startswith(("http://", "https://")):
        return AIConnectionTestResult(False, "invalid_base_url", provider, "Base URL 需要以 http:// 或 https:// 开头")

    endpoint = _chat_completions_endpoint(config.ai.base_url)
    payload = {
        "model": config.ai.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a connection test endpoint. Return compact json."},
            {"role": "user", "content": "Return json: {\"ok\":true}."},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_provider_headers(config.ai.api_key),
        method="POST",
    )
    timeout = max(3, min(config.ai.timeout_seconds, 20))
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
            elapsed = _elapsed_ms(started)
            _log_ai_provider(endpoint, elapsed, "connection_test_ok", f"status={getattr(response, 'status', '-')}")
            data = json.loads(raw_body)
    except urllib.error.HTTPError as exc:
        elapsed = _elapsed_ms(started)
        body = _read_http_error_body(exc)
        _log_ai_provider(endpoint, elapsed, f"connection_test_http_error:{exc.code}", f"reason={exc.reason} body={_clip(body, 500)}")
        return AIConnectionTestResult(
            False,
            f"http_error:{exc.code}",
            provider,
            _http_error_message(exc.code, exc.reason, body),
            elapsed,
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        elapsed = _elapsed_ms(started)
        _log_ai_provider(endpoint, elapsed, "connection_test_transport_error", str(exc))
        return AIConnectionTestResult(False, "transport_error", provider, str(exc), elapsed)
    except json.JSONDecodeError as exc:
        elapsed = _elapsed_ms(started)
        _log_ai_provider(endpoint, elapsed, "connection_test_bad_json", str(exc))
        return AIConnectionTestResult(False, "bad_response_json", provider, f"返回不是合法 JSON：{exc}", elapsed)

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        elapsed = _elapsed_ms(started)
        _log_ai_provider(endpoint, elapsed, "connection_test_bad_shape", _clip(str(data), 160))
        return AIConnectionTestResult(False, "bad_response_shape", provider, "返回结构缺少 choices", elapsed)
    return AIConnectionTestResult(True, "ok", provider, "连接测试通过", _elapsed_ms(started))


def _blocking_reason(context: ReplyContext, config: AppConfig) -> str | None:
    _, deadline_error = _current_work_remaining(context, config)
    if deadline_error:
        return deadline_error
    if config.ai.provider == "Disabled":
        return "AI 已在设置中禁用"
    if config.privacy.manual_protection_blocks_replies and context.strategy and context.strategy.requires_manual_reply:
        return "当前分组启用了手动回复保护"
    if config.capture.pause_ai_on_unknown_page and context.runtime.page.page_type not in {PageType.CHAT_DM, PageType.CHAT_GROUP}:
        return f"当前页面未确认是聊天页：{context.runtime.page.page_type.value}"
    if context.contact is None:
        return "聊天对象尚未识别或选择，不能生成回复建议"
    if context.contact.status == ContactStatus.IGNORED:
        return "聊天对象已被忽略，不能生成回复建议"
    if context.contact.status == ContactStatus.MERGED or context.contact.merged_into:
        return "聊天对象已合并到其他对象，请切换到合并后的对象"
    source_error = current_context_source_error(context)
    if source_error:
        return "blocked:current_context_source:" + source_error
    if _uses_cloud_provider(config) and not cloud_reply_authorized(context):
        return "联系人未允许发送上下文到第三方 AI"
    photo_error = photo_source_error(context)
    if photo_error:
        return "blocked:photo_source: " + photo_status_message(photo_error)
    return None


def _current_work_remaining(context: ReplyContext, config: AppConfig) -> tuple[float, str | None]:
    """Return the remaining absolute attempt budget, never extend a clock.

    Task submission fills both fields for legacy contexts. Direct legacy
    generator callers remain transport-bounded, without fabricating recognition
    evidence; the real UI always supplies the original recognized-at clock.
    """
    started = getattr(context, "current_work_started_monotonic", None)
    deadline = getattr(context, "current_work_deadline_monotonic", None)
    configured = config.ai.timeout_seconds
    if type(configured) not in {int, float} or not isfinite(configured) or configured <= 0:
        return 0.0, "blocked:invalid_current_work_deadline"
    maximum = min(30.0, float(configured))
    if started is None and deadline is None:
        return maximum, None
    now = time.monotonic()
    if (type(started) not in {int, float} or type(deadline) not in {int, float}
            or not isfinite(started) or not isfinite(deadline) or started < 0 or started > now
            or deadline <= started or deadline - started > 30.0 + 1e-9):
        return 0.0, "blocked:invalid_current_work_deadline"
    remaining = min(float(deadline), float(started) + maximum) - now
    return (remaining, None) if remaining > 0 else (0.0, "blocked:current_work_timeout")


def _uses_cloud_provider(config: AppConfig) -> bool:
    return config.ai.provider in {"OpenAI", "OpenAI Compatible"} and bool(config.ai.api_key)


def _reply_request_options(endpoint: str, model: str) -> dict:
    """Provider-specific latency policy, never leak vendor flags to a proxy.

    DeepSeek Flash defaults to thinking; a manual short reply uses the official
    non-thinking mode. Keep the full multimodal prompt and strict reply schema.
    A length-limited response is a failure, not a partially usable suggestion.
    """
    parsed = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme == "https" and parsed.netloc.lower() in {"api.deepseek.com", "api.deepseek.com:443"}
            and parsed.path in {"/chat/completions", "/v1/chat/completions"}
            and not parsed.query and not parsed.fragment and model.strip() == "deepseek-flash"):
        return {"thinking": {"type": "disabled"}, "max_tokens": 4096}
    return {}


def _provider_usage_counts(data: object) -> dict[str, int]:
    """Keep only optional provider counters, never response text or identifiers.

    Missing/invalid counters stay absent; zero must mean a measured zero.
    These diagnostics have no role in authorization, billing or acceptance.
    """
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        value = usage.get(key)
        if type(value) is int and 0 <= value <= 2**31 - 1:
            result[key] = value
    for section, key, output in (
        ("prompt_tokens_details", "cached_tokens", "cached_input_tokens"),
        ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
    ):
        detail = usage.get(section)
        value = detail.get(key) if isinstance(detail, dict) else None
        if type(value) is int and 0 <= value <= 2**31 - 1:
            result[output] = value
    return result


def _log_response_metrics(endpoint: str, elapsed_ms: int, performance: dict, *, rejected: bool = False) -> None:
    """Bounded numerical diagnostics; a failed metric write cannot reject a reply."""
    try:
        provider = performance.get("provider_profile") or {}
        prompt = performance.get("prompt_profile") or {}
        metrics = {key: provider[key] for key in (
            "request_bytes", "response_bytes", "request_to_headers_ms",
            "headers_to_body_complete_ms", "total_ms",
        ) if type(provider.get(key)) is int and 0 <= provider[key] <= 2**31 - 1}
        count = performance.get("completion_content_chars")
        if type(count) is int and 0 <= count <= 2**31 - 1:
            metrics["completion_content_chars"] = count
        metrics["usage"] = _provider_usage_counts({"usage": performance.get("provider_usage")})
        # _provider_usage_counts has already flattened these two detail counters.
        for key in ("cached_input_tokens", "reasoning_tokens"):
            value = (performance.get("provider_usage") or {}).get(key)
            if type(value) is int and 0 <= value <= 2**31 - 1:
                metrics["usage"][key] = value
        inputs = {key: prompt[key] for key in (
            "system_chars", "user_chars", "estimated_tokens", "recent_message_count",
            "recent_chat_chars", "memory_context_chars", "knowledge_context_chars",
            "prompt_build_ms", "photo_count", "photo_bytes",
        ) if type(prompt.get(key)) is int and 0 <= prompt[key] <= 2**31 - 1}
        # Keep the two records below the existing per-log 800-character bound.
        # Headers describe transport progress, not the first usable AI answer.
        prefix = "failed_" if rejected else ""
        _log_ai_provider(endpoint, elapsed_ms, prefix + "provider_metrics",
                         json.dumps(metrics, ensure_ascii=True, separators=(",", ":")))
        _log_ai_provider(endpoint, elapsed_ms, prefix + "prompt_metrics",
                         json.dumps(inputs, ensure_ascii=True, separators=(",", ":")))
    except Exception:
        pass


def _log_reply_contract_metadata(endpoint: str, elapsed_ms: int, parsed: dict) -> None:
    """Classify a rejected policy field without recording any model-supplied value."""
    try:
        if "policy_version" not in parsed:
            state = "missing"
        else:
            value = parsed["policy_version"]
            if type(value) is not str:
                state = "invalid_type"
            elif value == "":
                state = "empty"
            elif value == REPLY_POLICY_VERSION:
                state = "current"
            else:
                state = "different"
        _log_ai_provider(endpoint, elapsed_ms, "reply_contract_metadata",
                         json.dumps({"policy_field_state": state}, separators=(",", ":")))
    except Exception:
        pass


def _generate_local_preview(context: ReplyContext, config: AppConfig) -> ReplyGenerationResult:
    current = context.effective_current_message
    recent_other = (
        current.text if current is not None and current.speaker == Speaker.OTHER and current.text.strip()
        else next((m.text for m in reversed(context.effective_messages) if m.speaker == Speaker.OTHER and m.text.strip()), "")
    )
    base = recent_other or "我看到了你的消息"
    return ReplyGenerationResult(
        False,
        "local_preview",
        [],
        "Local Preview",
        customer_summary=context.contact.display_name if context.contact else "未知客户",
        intent=f"需结合上下文确认：{_clip(base, 48)}",
        relevant_history=tuple(memory.content for memory in context.memories[:2]) or ("暂无可用历史",),
        ask_first="仅本地预览，尚未生成专业判断；请核对当前问题和已有资料。",
        sales_judgement="信息不足时先诊断，不急于推荐产品或报价。",
        next_action="补齐关键信息后，再给处理方案和回访节点。",
        policy_version=REPLY_POLICY_VERSION,
        recommendation_mode="clarify",
        problem_assessment="本地占位预览，不代表已完成农业问题判断。",
        uncertainties=("未调用专业判断模型，不能据此配药或承诺效果。",),
    )


def _generate_openai_compatible(context: ReplyContext, config: AppConfig) -> ReplyGenerationResult:
    if (getattr(context, "current_work_started_monotonic", None) is None
            and getattr(context, "current_work_deadline_monotonic", None) is None):
        remaining, deadline_error = _current_work_remaining(context, config)
        if deadline_error:
            return ReplyGenerationResult(False, deadline_error, [], config.ai.provider)
        began = time.monotonic()
        context = replace(context, current_work_started_monotonic=began,
                          current_work_deadline_monotonic=began + remaining)
    blocked = _blocking_reason(context, config)
    if blocked:
        return ReplyGenerationResult(False, blocked, [], config.ai.provider)
    endpoint = _chat_completions_endpoint(config.ai.base_url)
    prompt_started = time.monotonic()
    preview = build_prompt_preview(context, config)
    prompt_build_ms = _elapsed_ms(prompt_started)
    memory_payload = context.memory_context.as_payload() if context.memory_context is not None else {}
    knowledge_payload = context.knowledge_context.as_payload() if context.knowledge_context is not None else {}
    prompt_profile = {
        "policy_version": REPLY_POLICY_VERSION,
        "system_chars": len(preview.system_prompt),
        "user_chars": len(preview.user_prompt),
        "estimated_tokens": max(1, (len(preview.system_prompt) + len(preview.user_prompt) + 2) // 3),
        "recent_message_count": len(context.effective_messages),
        "recent_chat_chars": sum(len(message.text) for message in context.effective_messages),
        "memory_context_chars": len(json.dumps(memory_payload, ensure_ascii=False, separators=(",", ":"))),
        "knowledge_context_chars": len(json.dumps(knowledge_payload, ensure_ascii=False, separators=(",", ":"))),
        "prompt_build_ms": prompt_build_ms,
        "photo_count": len(context.photos),
        "photo_bytes": sum(len(photo.png) for photo in context.photos),
    }
    payload = {
        "model": config.ai.model,
        "temperature": config.ai.temperature,
        "messages": [
            {"role": "system", "content": preview.system_prompt},
            {"role": "user", "content": multimodal_user_content(context, preview.user_prompt)},
        ],
        "response_format": {"type": "json_object"},
    }
    request_options = _reply_request_options(endpoint, config.ai.model)
    payload.update(request_options)
    encoded_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded_payload,
        headers=_provider_headers(config.ai.api_key),
        method="POST",
    )
    started = time.monotonic()
    provider_request_started_at = utc_now_iso()
    provider_first_response_at = ""
    provider_response_completed_at = ""
    response_bytes = 0
    provider_usage: dict[str, int] = {}
    completion_content_chars: int | None = None
    remaining, deadline_error = _current_work_remaining(context, config)
    if deadline_error:
        return ReplyGenerationResult(False, deadline_error, [], config.ai.provider)
    phase = "headers"
    mode_policy = reply_mode_policy(context)
    returned_mode = "not_parsed"

    def trace(failure_stage: str = "") -> dict:
        return {
            "provider_request_started_at": provider_request_started_at,
            # Headers are evidence of connection only, not a usable AI answer.
            "provider_first_response_at": provider_first_response_at,
            "provider_response_completed_at": provider_response_completed_at,
            "prompt_profile": prompt_profile,
            "provider_usage": dict(provider_usage),
            "completion_content_chars": completion_content_chars,
            "provider_profile": {
                "transport": "urllib.request", "connection_reused": False,
                "request_bytes": len(encoded_payload), "response_bytes": response_bytes,
                "request_to_headers_ms": _duration_ms(provider_request_started_at, provider_first_response_at),
                "headers_to_body_complete_ms": _duration_ms(provider_first_response_at, provider_response_completed_at),
                "total_ms": _elapsed_ms(started), "timeout_seconds": remaining,
                "request_mode": "deepseek_flash_non_thinking" if request_options else "provider_default",
                "failure_stage": failure_stage,
                "reply_mode_policy": mode_policy,
                "returned_recommendation_mode": returned_mode,
            },
        }

    def failed(status: str, stage: str) -> ReplyGenerationResult:
        performance = trace(stage)
        # This metadata intentionally excludes request/response text, photos,
        # authorization headers and customer identifiers, even on exceptions.
        _log_ai_provider(endpoint, _elapsed_ms(started), "provider_phase",
                         json.dumps(performance["provider_profile"], ensure_ascii=False))
        return ReplyGenerationResult(False, status, [], config.ai.provider,
                                     performance_trace=performance)

    _log_ai_provider(endpoint, 0, "request_policy",
                     json.dumps(trace()["provider_profile"], ensure_ascii=False))
    try:
        with urllib.request.urlopen(request, timeout=remaining) as response:
            provider_first_response_at = utc_now_iso()
            phase = "body"
            raw_bytes = response.read()
            provider_response_completed_at = utc_now_iso()
            response_bytes = len(raw_bytes)
            raw_body = raw_bytes.decode("utf-8")
            _log_ai_provider(endpoint, _elapsed_ms(started), "http_ok", f"status={getattr(response, 'status', '-')}")
            phase = "response_json"
            data = json.loads(raw_body)
            provider_usage = _provider_usage_counts(data)
    except urllib.error.HTTPError as exc:
        body = _read_http_error_body(exc)
        _log_ai_provider(endpoint, _elapsed_ms(started), f"http_error:{exc.code}", f"reason={exc.reason} body={_clip(body, 500)}")
        return failed(f"AI 请求失败：{_http_error_message(exc.code, exc.reason, body)}", "http_error")
    except (urllib.error.URLError, TimeoutError) as exc:
        _log_ai_provider(endpoint, _elapsed_ms(started), "transport_error", str(exc))
        if isinstance(exc, TimeoutError) or (isinstance(exc, urllib.error.URLError)
                                             and isinstance(exc.reason, TimeoutError)):
            return failed("provider_timeout: AI 请求读取超时，可手动重试", phase)
        return failed(f"AI 请求失败：{exc}", phase)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _log_ai_provider(endpoint, _elapsed_ms(started), "bad_response_json", str(exc))
        return failed(f"AI 返回不是合法 JSON：{exc}", "response_json")

    _, deadline_error = _current_work_remaining(context, config)
    if deadline_error:
        return failed(deadline_error, "deadline_after_body")

    try:
        choice = data.get("choices", [{}])[0]
        if choice.get("finish_reason") == "length":
            return failed("provider_output_truncated: AI 返回达到长度上限，内容不完整；本次未展示，也未自动重试", "output_truncated")
        content = choice.get("message", {}).get("content", "")
        completion_content_chars = len(content) if isinstance(content, str) else None
    except (AttributeError, IndexError, TypeError):
        _log_ai_provider(endpoint, _elapsed_ms(started), "bad_response_shape", _clip(str(data), 240))
        return failed("AI 返回结构无法解析", "response_shape")
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        _log_ai_provider(endpoint, _elapsed_ms(started), "bad_content_json", _clip(str(content), 240))
        return failed("AI 返回格式无法解析", "content_json")
    if not isinstance(parsed, dict):
        _log_ai_provider(endpoint, _elapsed_ms(started), "bad_content_type", type(parsed).__name__)
        return failed("AI 返回格式无法解析", "content_type")

    returned_mode = parsed.get("recommendation_mode")
    if not isinstance(returned_mode, str) or returned_mode not in {"ingredients", "clarify", "not_applicable"}:
        returned_mode = "invalid_or_missing"
    reply_parse_completed_at = utc_now_iso()
    result = _parse_reply_payload(
        parsed,
        config.ai.provider,
        context=context,
        performance_trace={
            **trace(),
            "reply_parse_completed_at": reply_parse_completed_at,
        },
    )
    _, deadline_error = _current_work_remaining(context, config)
    if deadline_error:
        return failed(deadline_error, "deadline_after_parse")
    if not result.allowed:
        _log_reply_contract_metadata(endpoint, _elapsed_ms(started), parsed)
        result = failed(result.status, "reply_contract")
    _log_ai_provider(
        endpoint, _elapsed_ms(started), "parsed" if result.allowed else "invalid_analysis_result",
        f"suggestions={len(result.suggestions)}" if result.allowed else result.status,
    )
    _log_response_metrics(endpoint, _elapsed_ms(started), result.performance_trace,
                          rejected=not result.allowed)
    return result


def _parse_reply_payload(
    parsed: object,
    provider: str,
    *,
    context: ReplyContext,
    performance_trace: dict | None = None,
) -> ReplyGenerationResult:
    """Pure strict parser; it never calls a model, opens files or retries.

    Reuse the durable schema reader so a fresh response, cache and restored
    result cannot disagree on whether the ingredient contract is complete.
    Model-supplied traces/status/provider are not treated as runtime evidence.
    """
    if not isinstance(parsed, dict):
        return ReplyGenerationResult(False, "blocked:invalid_analysis:result_not_object", [], provider)
    items = parsed.get("reply_suggestions")
    if not isinstance(items, list) or len(items) != 2 or any(not isinstance(item, dict) for item in items):
        return ReplyGenerationResult(False, "blocked:invalid_analysis:reply_suggestions_count", [], provider)
    data = dict(parsed)
    # Only a literal quote of the latest actual outgoing message can be
    # resolved back to its original evidence. No suggestion/action is ever
    # unmasked. Durable results retain the original quote and pass the same
    # strict provenance/dosage checks as a fresh response.
    evidence = data.get("product_introduction_evidence")
    latest_own = next((m for m in reversed(context.effective_messages)
                       if m.speaker.value == "me"), None)
    if isinstance(evidence, list) and latest_own is not None:
        projected = project_own_message_for_drafting(latest_own.text).strip()
        data["product_introduction_evidence"] = [
            latest_own.text.strip()
            if isinstance(item, str) and UNREVIEWED_QUANTITY_MARKER in item
            and item.strip() == projected else item
            for item in evidence
        ]
    data.update({
        "allowed": True, "status": "ai_generated", "provider": provider,
        "performance_trace": performance_trace or {},
        "suggestions": [
            {**item, "label": item.get("label", f"回复{index + 1}")}
            for index, item in enumerate(items)
        ],
    })
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        return ReplyGenerationResult(False, "blocked:invalid_analysis:invalid_json_values", [], provider)
    result, error = reply_generation_result_from_json(payload, context=context)
    if error or result is None:
        return ReplyGenerationResult(False, "blocked:invalid_analysis:" + (error or "unknown"), [], provider)
    return result


def _clip(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _duration_ms(start: str, end: str) -> int | None:
    if not start or not end:
        return None
    from datetime import datetime

    try:
        return max(0, round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000))
    except ValueError:
        return None


def _chat_completions_endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def _http_error_message(code: int, reason: str, body: str = "") -> str:
    detail = _provider_error_detail(body)
    if code == 401:
        return "HTTP 401：认证失败，请检查 API Key 是否属于当前服务商，接口地址是否填到对应 Provider 的 /v1"
    if code == 403:
        return "HTTP 403：服务商拒绝访问；若其他软件可用，请核对 Base URL、模型名、账号/模型权限，或服务商是否限制请求头/IP"
    if code == 404:
        return "HTTP 404：接口或模型不存在，请检查接口地址和模型名称"
    if code == 429:
        return "HTTP 429：请求过于频繁或额度不足，请稍后重试或检查服务商额度"
    if detail:
        return f"HTTP {code}: {detail}"
    return f"HTTP {code}: {reason}"


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _provider_error_detail(body: str) -> str:
    if not body.strip():
        return ""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return _clip(" ".join(body.split()), 240)
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or "").strip()
        if message and code:
            return f"{message} ({code})"
        if message:
            return message
    if isinstance(data, dict):
        message = str(data.get("message") or data.get("detail") or "").strip()
        if message:
            return message
    return _clip(str(data), 240)


def _provider_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "WhoChat/0.1 OpenAI-Compatible-Client",
    }


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)[\"']?\b(?:api[_-]?key|apikey|authorization|access[_-]?token|refresh[_-]?token"
    r"|client[_-]?secret|secret|password|token)\b[\"']?\s*[:=]\s*[\"']?[^\s,;\"'}\]\]]+"
)
_SECRET_KEY_TOKEN = re.compile(
    r"(?i)(?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|client[_-]?secret)"
)


def _sanitize_provider_log_detail(detail: str) -> str:
    """Strip credentials before they reach the AI provider diagnostics log.

    Providers occasionally echo the submitted ``api_key`` field, a fragment of
    the Authorization header, or a ``invalid_api_key``-style error code inside
    an error body. ``ai_provider.log`` is rotated and shared when diagnosing
    connectivity problems, so no log line may ever contain credential material
    or even a credential field name.
    """
    text = " ".join(str(detail).split())
    text = _SECRET_ASSIGNMENT.sub("[已脱敏:密钥]", text)
    text = redact_diagnostics_text(text)
    return _SECRET_KEY_TOKEN.sub("[已脱敏:密钥]", text)


def _log_ai_provider(endpoint: str, elapsed_ms: int, status: str, detail: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    target = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    clipped = _sanitize_provider_log_detail(detail)[:800]
    append_diagnostics_log("ai_provider", f"target={target} elapsed_ms={elapsed_ms} status={status} {clipped}")
