from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from whochat.core.models import CustomerEventType, CustomerFactProvenance


class MemoryOutputError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedMemoryExtraction:
    growing_context_updates: tuple[dict[str, Any], ...]
    facts: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    followups: tuple[dict[str, Any], ...]
    nothing_to_store: bool
    rejected_items: tuple[str, ...] = ()


_TOP_LEVEL_KEYS = {
    "growing_context_updates",
    "facts",
    "events",
    "followups",
    "nothing_to_store",
}
_PROVENANCE = {item.value for item in CustomerFactProvenance}
_EVENT_TYPES = {item.value for item in CustomerEventType}


def parse_memory_extraction_json(payload: str | dict[str, Any]) -> ValidatedMemoryExtraction:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MemoryOutputError(f"invalid_json:{exc.msg}") from exc
    else:
        data = payload
    if not isinstance(data, dict):
        raise MemoryOutputError("top_level_must_be_object")
    missing = sorted(_TOP_LEVEL_KEYS - set(data))
    if missing:
        raise MemoryOutputError(f"missing_top_level_fields:{','.join(missing)}")
    if not isinstance(data["nothing_to_store"], bool):
        raise MemoryOutputError("nothing_to_store_must_be_boolean")
    for key in _TOP_LEVEL_KEYS - {"nothing_to_store"}:
        if not isinstance(data[key], list):
            raise MemoryOutputError(f"{key}_must_be_array")

    rejected: list[str] = []
    contexts = _valid_items(data["growing_context_updates"], "growing_context", _validate_context, rejected)
    facts = _valid_items(data["facts"], "fact", _validate_fact, rejected)
    events = _valid_items(data["events"], "event", _validate_event, rejected)
    followups = _valid_items(data["followups"], "followup", _validate_followup, rejected)
    if data["nothing_to_store"] and any((contexts, facts, events, followups)):
        raise MemoryOutputError("nothing_to_store_conflicts_with_candidates")
    return ValidatedMemoryExtraction(
        tuple(contexts), tuple(facts), tuple(events), tuple(followups),
        bool(data["nothing_to_store"]), tuple(rejected),
    )


def _valid_items(items, label, validator, rejected):
    valid = []
    for index, item in enumerate(items):
        try:
            valid.append(validator(item))
        except MemoryOutputError as exc:
            rejected.append(f"{label}[{index}]:{exc}")
    return valid


def _base(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise MemoryOutputError("must_be_object")
    required = ("source_message_id", "source_text", "confidence", "provenance", "reason")
    for key in required:
        if key not in item:
            raise MemoryOutputError(f"missing_{key}")
    source_message_id = _string(item, "source_message_id", 200)
    source_text = _string(item, "source_text", 4000)
    reason = _string(item, "reason", 1000)
    provenance = _string(item, "provenance", 50)
    if provenance not in _PROVENANCE:
        raise MemoryOutputError("invalid_provenance")
    confidence = item["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise MemoryOutputError("confidence_must_be_number")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise MemoryOutputError("confidence_out_of_range")
    return {
        **item,
        "source_message_id": source_message_id,
        "source_text": source_text,
        "reason": reason,
        "provenance": provenance,
        "confidence": confidence,
    }


def _validate_context(item: Any) -> dict[str, Any]:
    result = _base(item)
    result["reference"] = _string(result, "reference", 200)
    for key in ("crop", "variety", "region", "acreage_unit", "growth_stage", "cultivation_type", "season_or_year"):
        value = result.get(key)
        if value is not None and not isinstance(value, str):
            raise MemoryOutputError(f"{key}_must_be_string_or_null")
        result[key] = value.strip() if isinstance(value, str) else None
    acreage = result.get("acreage_value")
    if acreage is not None and (isinstance(acreage, bool) or not isinstance(acreage, (int, float)) or float(acreage) < 0):
        raise MemoryOutputError("invalid_acreage_value")
    result["acreage_value"] = float(acreage) if acreage is not None else None
    status = str(result.get("status") or "uncertain")
    if status not in {"active", "uncertain"}:
        raise MemoryOutputError("invalid_context_status")
    result["status"] = status
    return result


def _validate_fact(item: Any) -> dict[str, Any]:
    result = _base(item)
    for key in ("fact_category", "fact_key", "value_text"):
        result[key] = _string(result, key, 2000 if key == "value_text" else 200)
    normalized = result.get("normalized_value")
    if normalized is not None and not isinstance(normalized, (str, int, float, bool)):
        raise MemoryOutputError("normalized_value_must_be_json_scalar_or_null")
    result["normalized_value"] = normalized
    reference = result.get("growing_context_reference")
    if reference is not None and not isinstance(reference, str):
        raise MemoryOutputError("growing_context_reference_must_be_string_or_null")
    result["growing_context_reference"] = reference.strip() if isinstance(reference, str) else None
    status = str(result.get("status") or "uncertain")
    if status not in {"active", "uncertain"}:
        raise MemoryOutputError("invalid_fact_status")
    result["status"] = status
    resolution = str(result.get("resolution") or "conflict")
    if resolution not in {"conflict", "supersede", "coexist"}:
        raise MemoryOutputError("invalid_fact_resolution")
    result["resolution"] = resolution
    value_json = result.get("value_json", {})
    if not isinstance(value_json, (dict, list)):
        raise MemoryOutputError("value_json_must_be_object_or_array")
    result["value_json"] = value_json
    return result


def _validate_event(item: Any) -> dict[str, Any]:
    result = _base(item)
    event_type = _string(result, "event_type", 100)
    if event_type not in _EVENT_TYPES:
        raise MemoryOutputError("invalid_event_type")
    result["event_type"] = event_type
    result["title"] = _string(result, "title", 300)
    summary = result.get("summary", "")
    if not isinstance(summary, str):
        raise MemoryOutputError("summary_must_be_string")
    result["summary"] = summary.strip()
    structured = result.get("structured_data", {})
    if not isinstance(structured, (dict, list)):
        raise MemoryOutputError("structured_data_must_be_object_or_array")
    result["structured_data"] = structured
    reference = result.get("growing_context_reference")
    if reference is not None and not isinstance(reference, str):
        raise MemoryOutputError("growing_context_reference_must_be_string_or_null")
    result["growing_context_reference"] = reference.strip() if isinstance(reference, str) else None
    return result


def _validate_followup(item: Any) -> dict[str, Any]:
    result = _base(item)
    for key, limit in (("followup_type", 100), ("title", 300), ("due_at", 80), ("time_evidence", 200), ("action_evidence", 200)):
        result[key] = _string(result, key, limit)
    reference = result.get("growing_context_reference")
    if reference is not None and not isinstance(reference, str):
        raise MemoryOutputError("growing_context_reference_must_be_string_or_null")
    result["growing_context_reference"] = reference.strip() if isinstance(reference, str) else None
    priority = result.get("priority", 50)
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
        raise MemoryOutputError("invalid_priority")
    result["priority"] = priority
    return result


def _string(item: dict[str, Any], key: str, limit: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MemoryOutputError(f"{key}_must_be_nonempty_string")
    return value.strip()[:limit]
