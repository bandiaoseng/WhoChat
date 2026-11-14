from __future__ import annotations

import json
import unicodedata

from whochat.core.models import (
    CustomerFact,
    CustomerGrowingContext,
    Message,
    Speaker,
)
from whochat.services.memory_canonicalization import (
    canonicalize_acreage,
    canonicalize_concern,
    canonicalize_crop,
    canonicalize_fact_identity,
    canonicalize_growth_stage,
    canonicalize_region,
    canonicalize_variety,
    crop_terms,
    extract_acreages,
    growth_stage_relation,
)
from whochat.storage.memory_writer import PlannedFact


SUPPORTED_CONFIRMATION_KEYS = frozenset(
    {
        "region",
        "crop",
        "variety",
        "acreage",
        "growth_stage",
        "cultivation_type",
        "season_or_year",
        "customer_concern",
        "customer_preference",
        "price_sensitivity",
    }
)


class LocalExplicitFactReconciler:
    """Conservatively attach explicit customer evidence to existing active facts only."""

    def reconcile(
        self,
        *,
        source: Message,
        customer_profile_id: str,
        contexts: list[CustomerGrowingContext],
        facts: list[CustomerFact],
    ) -> list[PlannedFact]:
        if source.speaker != Speaker.OTHER:
            return []
        context_by_id = {item.id: item for item in contexts}
        mentioned_context_ids = self._mentioned_context_ids(source.text, contexts)
        multiple_contexts = len(contexts) > 1
        result: list[PlannedFact] = []
        for fact in facts:
            identity = canonicalize_fact_identity(
                customer_profile_id=customer_profile_id,
                growing_context_id=fact.growing_context_id,
                fact_category=fact.fact_category,
                fact_key=fact.fact_key,
                value_text=fact.value_text,
                normalized_value=fact.normalized_value,
                value_json=fact.value_json,
            )
            if identity.canonical_key not in SUPPORTED_CONFIRMATION_KEYS:
                continue
            if fact.growing_context_id:
                if fact.growing_context_id not in context_by_id:
                    continue
                if multiple_contexts:
                    if not mentioned_context_ids or fact.growing_context_id not in mentioned_context_ids:
                        continue
            if not self._explicitly_confirms(identity.canonical_key, identity.canonical_value, source.text):
                continue
            result.append(
                PlannedFact(
                    growing_context_reference=fact.growing_context_id,
                    fact_category=identity.canonical_category,
                    fact_key=identity.canonical_key,
                    value_text=fact.value_text,
                    value_json=_json_object(fact.value_json),
                    normalized_value=identity.canonical_value,
                    confidence=fact.confidence,
                    provenance="customer_explicit",
                    status="active",
                    resolution="coexist",
                    confirmation_only=True,
                    existing_fact_id=fact.id,
                    candidate_origins=("local",),
                )
            )
        return result

    @staticmethod
    def _mentioned_context_ids(
        source_text: str,
        contexts: list[CustomerGrowingContext],
    ) -> set[str]:
        source = _normal_text(source_text)
        matches: set[str] = set()
        for context in contexts:
            crop = canonicalize_crop(context.crop)
            if crop and any(_normal_text(term) in source for term in crop_terms(crop)):
                matches.add(context.id)
        return matches

    @staticmethod
    def _explicitly_confirms(key: str, canonical_value: str, source_text: str) -> bool:
        source = _normal_text(source_text)
        if not source or not canonical_value:
            return False
        if key == "crop":
            crop = canonicalize_crop(canonical_value)
            return any(_normal_text(term) in source for term in crop_terms(crop))
        if key == "variety":
            return _normal_text(canonicalize_variety(canonical_value)) in source
        if key == "region":
            region = _normal_text(canonicalize_region(canonical_value))
            source_without_suffixes = source.replace("省", "").replace("市", "")
            return bool(region) and (region in source or region in source_without_suffixes)
        if key == "acreage":
            parsed = {
                canonicalize_acreage(
                    value_text=f"{number}{unit}",
                    value_json={"value": number, "unit": unit},
                )
                for number, unit in extract_acreages(source_text)
            }
            return canonical_value in parsed
        if key == "growth_stage":
            source_stage = canonicalize_growth_stage(source_text, source_text=source_text)
            return growth_stage_relation(canonical_value, source_stage, source_text=source_text) in {
                "same",
                "compatible",
            }
        if key == "customer_concern":
            has_concern_cue = any(term in source for term in ("担心", "怕", "顾虑", "关注"))
            return has_concern_cue and canonicalize_concern(source_text) == canonical_value
        if key == "price_sensitivity":
            return any(term in source for term in ("价格", "贵", "成本"))
        if key in {"cultivation_type", "season_or_year", "customer_preference"}:
            return _normal_text(canonical_value) in source
        return False


def _json_object(value: str) -> dict | list:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, (dict, list)) else {}


def _normal_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()
