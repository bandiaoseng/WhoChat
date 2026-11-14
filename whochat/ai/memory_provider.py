from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from whochat.ai.generator import (
    _chat_completions_endpoint,
    _http_error_message,
    _log_ai_provider,
    _provider_headers,
    _read_http_error_body,
)
from whochat.config import AppConfig


class MemoryProviderError(RuntimeError):
    pass


class MemoryProviderBlocked(MemoryProviderError):
    pass


@dataclass(frozen=True)
class MemoryProviderResult:
    raw_output: str
    elapsed_ms: int


class OpenAICompatibleMemoryProvider:
    """One-shot JSON extraction using the already DPAPI-backed AI config."""

    def extract(self, input_context_json: str, config: AppConfig) -> MemoryProviderResult:
        if config.ai.provider not in {"OpenAI", "OpenAI Compatible"}:
            raise MemoryProviderBlocked("cloud_memory_provider_disabled")
        if not config.ai.api_key:
            raise MemoryProviderBlocked("memory_provider_api_key_missing")
        if not config.ai.model.strip():
            raise MemoryProviderBlocked("memory_provider_model_missing")
        if not config.ai.base_url.startswith(("http://", "https://")):
            raise MemoryProviderBlocked("memory_provider_base_url_invalid")
        json.loads(input_context_json)
        endpoint = _chat_completions_endpoint(config.ai.base_url)
        payload = {
            "model": config.ai.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": input_context_json},
            ],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=_provider_headers(config.ai.api_key),
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=config.ai.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                elapsed = int((time.monotonic() - started) * 1000)
                outer = json.loads(body)
                content = outer.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not isinstance(content, str) or not content.strip():
                    raise MemoryProviderError("memory_provider_empty_content")
                _log_ai_provider(endpoint, elapsed, "memory_extraction_http_ok", "content_received=true")
                return MemoryProviderResult(content.strip(), elapsed)
        except urllib.error.HTTPError as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            body = _read_http_error_body(exc)
            _log_ai_provider(endpoint, elapsed, f"memory_extraction_http_error:{exc.code}", body)
            raise MemoryProviderError(_http_error_message(exc.code, exc.reason, body)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            _log_ai_provider(endpoint, elapsed, "memory_extraction_transport_error", str(exc))
            raise MemoryProviderError(f"memory_provider_transport_error:{exc}") from exc
        except (json.JSONDecodeError, AttributeError, IndexError) as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            _log_ai_provider(endpoint, elapsed, "memory_extraction_bad_response", type(exc).__name__)
            raise MemoryProviderError("memory_provider_bad_response_shape") from exc


_SYSTEM_PROMPT = r"""
You extract durable agricultural-sales customer memory from ONE designated source message.
Return exactly one JSON object. Never add markdown or explanations outside JSON.

Use recent messages only to resolve references. Every candidate MUST be asserted by source_message,
must use source_message.id as source_message_id, and must copy source_message.text exactly as source_text.
Never extract a new memory solely from a recent_context message.

Sender rules:
- sender=other: explicit customer statements may use provenance=customer_explicit.
- sender=me: use provenance=user_explicit. A salesperson's question/diagnosis is not a confirmed customer fact.
- Anything inferred rather than explicit must use provenance=ai_inferred, status=uncertain, confidence <= 0.69.

Business rules:
- Symptoms/problems become events (problem_reported), not permanent disease facts.
- Sender=me may create diagnosis, solution_given, product_recommended, dosage_given, quote_given,
  management_plan, or followup events.
- Buying/price/order/use/effect statements map to the supplied event enum without inventing states.
- Bind a crop event only when the source explicitly identifies that crop or exactly one existing context is unambiguous.
  Otherwise growing_context_reference must be null.
- A followup requires explicit time evidence AND explicit action evidence in the source. Do not invent a due date.
- Do not guess acreage, date, disease, product, purchase, use, or effect.

Top-level schema (all five keys are required):
{
  "growing_context_updates": [],
  "facts": [],
  "events": [],
  "followups": [],
  "nothing_to_store": false
}

Every candidate requires:
source_message_id, source_text, confidence (0..1), provenance, reason.

growing_context_updates item:
reference, crop, variety, region, acreage_value(number|null), acreage_unit, growth_stage,
cultivation_type, season_or_year, status(active|uncertain), plus common fields.
Use an existing context id from active_growing_contexts when it is the same crop/context.
For a new context use a stable local reference beginning with "new:".

facts item:
fact_category, fact_key, value_text, normalized_value, value_json(object),
growing_context_reference(string|null), status(active|uncertain),
resolution(conflict|supersede|coexist), plus common fields.

events item:
event_type, title, summary, structured_data(object),
growing_context_reference(string|null), plus common fields.
Allowed event_type values: problem_reported, diagnosis, solution_given, product_recommended,
dosage_given, quote_given, order_intent, order_placed, purchased, usage_started,
effect_feedback, complaint, followup, customer_concern, management_plan, other.

followups item:
followup_type, title, due_at(ISO-8601), priority(0..100),
time_evidence, action_evidence, growing_context_reference(string|null), plus common fields.

If the source has nothing durable, return empty arrays and nothing_to_store=true.
""".strip()
