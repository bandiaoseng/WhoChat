from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from whochat.ai.memory_models import (
    MemoryOutputError,
    ValidatedMemoryExtraction,
    parse_memory_extraction_json,
)
from whochat.core.models import Message, Speaker
from whochat.services.structured_memory import (
    StructuredMemoryExtractionService,
    _crop_terms,
    _normal_text,
    _normalize_crop,
    _validate_followup_evidence,
)
from whochat.storage.customer_profiles import CustomerProfileRepository
from whochat.storage.historical import HistoricalArchiveRepository, HistoricalMessageRow
from whochat.storage.memory_writer import AtomicMemoryWriter
from whochat.storage.repositories import ContactRepository, LogRepository


class HistoricalBackfillProvider(Protocol):
    name: str
    model: str

    def extract(self, input_context_json: str) -> str: ...


class HistoricalProviderTimeoutError(RuntimeError):
    """A provider request reached its bounded historical-import timeout."""


class FakeHistoricalBackfillProvider:
    """Deterministic test provider. It never performs network I/O."""

    name = "fake"
    model = "fixture-memory-v1"

    def __init__(self, responder: Callable[[dict[str, Any]], dict[str, Any]] | dict[str, Any]) -> None:
        self.responder = responder
        self.call_count = 0
        self.inputs: list[dict[str, Any]] = []

    def extract(self, input_context_json: str) -> str:
        payload = json.loads(input_context_json)
        self.inputs.append(payload)
        self.call_count += 1
        result = self.responder(payload) if callable(self.responder) else self.responder
        return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DisabledHistoricalBackfillProvider:
    """Production-safe default until a backfill run receives explicit authorization."""

    name = "disabled"
    model = ""

    def extract(self, input_context_json: str) -> str:
        del input_context_json
        raise RuntimeError("historical_backfill_provider_not_authorized")


@dataclass(frozen=True)
class BackfillChunkPolicy:
    max_messages: int = 75
    max_characters: int = 10_000
    break_after_days: int = 14


class HistoricalMemoryBackfillService:
    """Low-priority, explicit history-to-memory pipeline.

    This service never starts itself.  A caller must preview/import, confirm a
    resolved mapping, prepare a selected-customer job, and explicitly invoke
    ``run_next``.  Consequently startup and live Reply AI can never be blocked
    by historical work.
    """

    def __init__(
        self,
        archive: HistoricalArchiveRepository,
        profiles: CustomerProfileRepository,
        contacts: ContactRepository,
        live_memory_service: StructuredMemoryExtractionService,
        *,
        provider: HistoricalBackfillProvider,
        logs: LogRepository | None = None,
        postprocessor: Callable[[str, tuple[str, ...]], None] | None = None,
    ) -> None:
        self.archive = archive
        self.profiles = profiles
        self.contacts = contacts
        self.live_memory_service = live_memory_service
        self.provider = provider
        self.logs = logs
        self.postprocessor = postprocessor
        self.writer = AtomicMemoryWriter(profiles.db)
        self.archive.recover_interrupted_backfills()

    def prepare(
        self,
        *,
        import_job_id: str,
        mapping_id: str,
        policy: BackfillChunkPolicy = BackfillChunkPolicy(),
    ) -> dict[str, Any]:
        mapping = self.archive.get_mapping(mapping_id)
        if mapping is None or mapping["status"] != "resolved":
            raise ValueError("backfill_requires_resolved_mapping")
        rows = self.archive.list_mapping_messages(mapping_id)
        eligible = [row for row in rows if is_business_message(row)]
        job = self.archive.create_backfill_job(import_job_id=import_job_id, mapping_id=mapping_id)
        profile = self.profiles.get_customer_profile(
            account_scope_id=mapping["account_scope_id"], contact_id=mapping["contact_id"]
        )
        # An explicitly authorised, resolved historical mapping is itself
        # enough to initialise the local archive.  Previously the profile was
        # only created as a side effect of the first accepted memory write.
        # A customer whose every chunk legitimately returned ``no_memory``
        # therefore stayed at history_not_initialized even though the job had
        # completed.  Create and link the empty profile before chunking so the
        # completion state, numbered archive and local Tracking Advisor remain
        # available without manufacturing any Fact/Event/Followup.
        if profile is None:
            contact = self.contacts.get(str(mapping["contact_id"]))
            if contact is None or contact.account_scope_id != mapping["account_scope_id"]:
                raise ValueError("historical_backfill_contact_scope_mismatch")
            profile = self.profiles.get_or_create_customer_profile(
                account_scope_id=str(mapping["account_scope_id"]),
                contact_id=str(mapping["contact_id"]),
                conversation_id=str(mapping["conversation_id"]),
                display_name=contact.display_name,
            )
        self.archive.link_backfill_profile(str(job["id"]), profile.id)
        chunks = self._chunks(
            eligible,
            mapping=mapping,
            profile=profile,
            policy=policy,
        )
        stored = self.archive.replace_pending_chunks(
            job["id"], chunks, eligible_messages=len(eligible)
        )
        if not stored:
            self.archive.finish_empty_backfill(job["id"])
            self.archive.set_profile_history_status(
                account_scope_id=mapping["account_scope_id"],
                contact_id=mapping["contact_id"],
                status="memory_backfilled",
                detail="没有需要提取的历史片段",
            )
        else:
            self.archive.set_profile_history_status(
                account_scope_id=mapping["account_scope_id"],
                contact_id=mapping["contact_id"],
                status="memory_backfill_pending",
                detail=f"待处理 {len(stored)} 个历史片段",
            )
        return {
            "job_id": job["id"],
            "total_messages": len(rows),
            "eligible_messages": len(eligible),
            "filtered_messages": len(rows) - len(eligible),
            "total_chunks": len(stored),
        }

    def run_next(self, job_id: str) -> dict[str, Any]:
        chunk = self.archive.claim_next_chunk(
            job_id, provider=self.provider.name, model=self.provider.model
        )
        if chunk is None:
            job = self.archive.get_backfill_job(job_id)
            return {"job_id": job_id, "status": job["status"] if job else "missing", "processed": 0}
        raw_output = ""
        try:
            raw_output = self.provider.extract(chunk["input_context_json"])
            return self._apply_output(job_id, chunk, raw_output)
        except HistoricalProviderTimeoutError as exc:
            self.archive.finish_chunk(
                chunk["id"], status="provider_timeout_pending", error=str(exc)
            )
            return {
                "job_id": job_id,
                "chunk_index": chunk["chunk_index"],
                "status": "provider_timeout_pending",
            }
        except json.JSONDecodeError as exc:
            self.archive.finish_chunk(
                chunk["id"], status="invalid_output",
                output_json=_safe_failed_output(raw_output), error=f"invalid_json:{exc.msg}",
            )
            return {"job_id": job_id, "chunk_index": chunk["chunk_index"], "status": "invalid_output"}
        except MemoryOutputError as exc:
            self.archive.finish_chunk(
                chunk["id"], status="invalid_output",
                output_json=_safe_failed_output(raw_output), error=str(exc),
            )
            return {"job_id": job_id, "chunk_index": chunk["chunk_index"], "status": "invalid_output"}
        except Exception as exc:
            self.archive.finish_chunk(chunk["id"], status="failed", error=f"{type(exc).__name__}:{exc}")
            self._log("error", "historical_backfill_failed", f"job={job_id} error={type(exc).__name__}")
            return {"job_id": job_id, "chunk_index": chunk["chunk_index"], "status": "failed"}

    def replay_invalid_chunk(self, job_id: str, chunk_index: int) -> dict[str, Any]:
        """Locally salvage valid candidates from a saved provider response.

        No provider is contacted and ``ai_call_count`` is immutable.  This is
        intentionally limited to a stored invalid output with zero committed
        writes. Invalid results/candidates are audited and discarded without
        weakening source-message traceability.
        """
        chunk = self.archive.get_backfill_chunk(job_id, chunk_index)
        if chunk is None:
            raise ValueError("historical_backfill_chunk_not_found")
        if chunk["status"] != "invalid_output":
            raise ValueError("historical_replay_requires_invalid_output")
        if str(chunk["write_result_json"] or "{}") not in {"", "{}"}:
            raise ValueError("historical_replay_requires_zero_writes")
        raw_output = str(chunk["output_json"] or "")
        if not raw_output or raw_output == "{}":
            raise ValueError("historical_replay_output_missing")
        calls_before = int(chunk["ai_call_count"])
        try:
            result = self._apply_output(job_id, chunk, raw_output)
        except Exception as exc:
            self.archive.finish_chunk(
                chunk["id"], status="invalid_output", output_json=raw_output,
                error=f"offline_replay_failed:{type(exc).__name__}:{exc}",
            )
            raise
        current = self.archive.get_backfill_chunk(job_id, chunk_index)
        if current is None or int(current["ai_call_count"]) != calls_before:
            raise RuntimeError("historical_local_replay_changed_provider_call_count")
        return {**result, "replayed": True, "provider_calls": calls_before}

    def replay_saved_chunk(self, job_id: str, chunk_index: int) -> dict[str, Any]:
        """Re-apply one terminal chunk from its saved provider response only.

        This path is deliberately provider-free.  It exists for audited local
        rule improvements such as context binding, and keeps ``ai_call_count``
        immutable.  Atomic writer dedupe makes already accepted candidates
        idempotent while newly safe candidates can be recovered.
        """
        chunk = self.archive.get_backfill_chunk(job_id, chunk_index)
        if chunk is None:
            raise ValueError("historical_backfill_chunk_not_found")
        if chunk["status"] not in {"completed", "no_memory"}:
            raise ValueError("historical_saved_replay_requires_terminal_chunk")
        raw_output = str(chunk["output_json"] or "")
        if not raw_output or raw_output == "{}":
            raise ValueError("historical_replay_output_missing")
        calls_before = int(chunk["ai_call_count"])
        result = self._apply_output(job_id, chunk, raw_output)
        current = self.archive.get_backfill_chunk(job_id, chunk_index)
        if current is None or int(current["ai_call_count"]) != calls_before:
            raise RuntimeError("historical_local_replay_changed_provider_call_count")
        return {
            **result,
            "replayed": True,
            "provider_calls_before": calls_before,
            "provider_calls_after": int(current["ai_call_count"]),
        }

    def _apply_output(
        self,
        job_id: str,
        chunk: dict[str, Any],
        raw_output: str,
    ) -> dict[str, Any]:
        payload = json.loads(raw_output)
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise MemoryOutputError("historical_output_results_missing")
        context = json.loads(chunk["input_context_json"])
        allowed = {str(item["message_id"]): item for item in context["messages"]}
        reference_map = {
            str(item.get("source_ref") or f"m{index:03d}"): str(item["message_id"])
            for index, item in enumerate(context["messages"])
        }
        seen: set[str] = set()
        prepared: list[tuple[int, Message, ValidatedMemoryExtraction]] = []
        rejections: list[dict[str, Any]] = []
        job = self.archive.get_backfill_job(job_id)
        if job is None:
            raise ValueError("historical_backfill_job_not_found")
        contact = self.contacts.get(job["contact_id"])
        if contact is None or contact.account_scope_id != job["account_scope_id"]:
            raise ValueError("historical_backfill_contact_scope_mismatch")

        def reject(
            *, result_index: int, reason: str, collection: str = "result",
            candidate_index: int | None = None,
        ) -> None:
            entry: dict[str, Any] = {
                "result_index": int(result_index),
                "collection": collection,
                "reason": str(reason)[:300],
            }
            if candidate_index is not None:
                entry["candidate_index"] = int(candidate_index)
            rejections.append(entry)

        collection_attributes = {
            "growing_context_updates": "growing_context_updates",
            "facts": "facts",
            "events": "events",
            "followups": "followups",
        }
        for index, result in enumerate(results):
            if not isinstance(result, dict):
                reject(result_index=index, reason="result_must_be_object")
                continue
            supplied_id = str(result.get("source_message_id") or "")
            source_ref = str(result.get("source_ref") or "")
            mapped_ref = reference_map.get(source_ref, "") if source_ref else ""
            if supplied_id and mapped_ref and supplied_id != mapped_ref:
                reject(result_index=index, reason="source_id_and_ref_disagree")
                continue
            source_id = supplied_id or mapped_ref
            if source_id not in allowed or source_id in seen:
                reject(
                    result_index=index,
                    reason=("duplicate_source" if source_id in seen else "source_not_allowed"),
                )
                continue
            seen.add(source_id)
            extraction = result.get("extraction")
            if not isinstance(extraction, dict):
                reject(result_index=index, reason="extraction_must_be_object")
                continue

            parsed_items: dict[str, list[dict[str, Any]]] = {
                attribute: [] for attribute in collection_attributes.values()
            }
            for collection, attribute in collection_attributes.items():
                values = extraction.get(collection, [])
                if not isinstance(values, list):
                    reject(
                        result_index=index, collection=collection,
                        reason="collection_must_be_array",
                    )
                    continue
                for candidate_index, candidate in enumerate(values):
                    if not isinstance(candidate, dict):
                        reject(
                            result_index=index, collection=collection,
                            candidate_index=candidate_index, reason="candidate_must_be_object",
                        )
                        continue
                    candidate = dict(candidate)
                    nested_id = str(candidate.get("source_message_id") or "")
                    nested_text = candidate.get("source_text")
                    if nested_id and nested_id != source_id:
                        reject(
                            result_index=index, collection=collection,
                            candidate_index=candidate_index,
                            reason="nested_source_id_mismatch",
                        )
                        continue
                    if nested_text not in (None, "") and str(nested_text) != str(
                        allowed[source_id]["text"]
                    ):
                        reject(
                            result_index=index, collection=collection,
                            candidate_index=candidate_index,
                            reason="nested_source_text_mismatch",
                        )
                        continue
                    # The provider selects a locally generated source_ref (or a
                    # legacy exact id). The authoritative evidence is attached
                    # locally, never copied or repaired from provider prose.
                    candidate["source_message_id"] = source_id
                    candidate["source_text"] = str(allowed[source_id]["text"])
                    single = {
                        "growing_context_updates": [], "facts": [], "events": [],
                        "followups": [], "nothing_to_store": False,
                    }
                    single[collection] = [candidate]
                    try:
                        validated = parse_memory_extraction_json(single)
                    except MemoryOutputError as exc:
                        reject(
                            result_index=index, collection=collection,
                            candidate_index=candidate_index,
                            reason=f"schema:{exc}",
                        )
                        continue
                    if validated.rejected_items:
                        reject(
                            result_index=index, collection=collection,
                            candidate_index=candidate_index,
                            reason=f"schema:{validated.rejected_items[0]}",
                        )
                        continue
                    parsed_items[attribute].append(getattr(validated, attribute)[0])

            parsed = ValidatedMemoryExtraction(
                growing_context_updates=tuple(parsed_items["growing_context_updates"]),
                facts=tuple(parsed_items["facts"]),
                events=tuple(parsed_items["events"]),
                followups=tuple(parsed_items["followups"]),
                nothing_to_store=not any(parsed_items.values()),
            )
            source = _history_as_message(allowed[source_id], job)
            valid_followups = []
            for candidate_index, followup in enumerate(parsed.followups):
                try:
                    _validate_followup_evidence(followup, source)
                except (ValueError, KeyError) as exc:
                    reject(
                        result_index=index, collection="followups",
                        candidate_index=candidate_index,
                        reason=f"business:{exc}",
                    )
                    continue
                valid_followups.append(followup)
            parsed = replace(
                parsed,
                followups=tuple(valid_followups),
                nothing_to_store=not any((
                    parsed.growing_context_updates, parsed.facts,
                    parsed.events, valid_followups,
                )),
            )
            prepared.append((index, source, parsed))

        writes: list[dict[str, Any]] = []
        written_plans = 0
        for result_index, source, parsed in prepared:
            profile = self.profiles.get_customer_profile(
                account_scope_id=job["account_scope_id"], contact_id=job["contact_id"]
            )
            parsed = self._remap_references_for_current_profile(parsed, profile, source)
            fake_job = SimpleNamespace(
                account_scope_id=job["account_scope_id"],
                target_instance_id=None,
                contact_id=job["contact_id"],
                conversation_id=job["conversation_id"],
            )

            def build(value):
                return replace(
                    self.live_memory_service._build_write_plan(
                        fake_job,
                        source,
                        contact.display_name,
                        profile,
                        value,
                        allow_empty=value.nothing_to_store,
                    ),
                    source_kind="historical",
                )

            plan = None
            while True:
                try:
                    plan = build(parsed)
                    break
                except MemoryOutputError as exc:
                    dropped = _drop_rejected_plan_candidate(parsed, str(exc))
                    if dropped is None:
                        raise
                    parsed, collection, candidate_index, reason = dropped
                    reject(
                        result_index=result_index, collection=collection,
                        candidate_index=candidate_index, reason=f"business:{reason}",
                    )
            if plan.item_count:
                writes.append(self.writer.apply(plan))
                written_plans += 1
            elif not parsed.nothing_to_store:
                raise MemoryOutputError("validated_historical_plan_is_empty")

        rejected_results = sum(1 for item in rejections if item["collection"] == "result")
        rejected_followups = sum(1 for item in rejections if item["collection"] == "followups")
        rejected_evidence_items = sum(
            1 for item in rejections
            if str(item["reason"]) in {
                "nested_source_id_mismatch", "nested_source_text_mismatch"
            }
        )
        result_payload = {
            "messages_with_output": len(prepared),
            "plans_written": written_plans,
            "contexts": sum(int(item.get("contexts", 0)) for item in writes),
            "facts": sum(int(item.get("facts", 0)) for item in writes),
            "events": sum(int(item.get("events", 0)) for item in writes),
            "followups": sum(int(item.get("followups", 0)) for item in writes),
            "rejected_followups": rejected_followups,
            "rejected_evidence_items": rejected_evidence_items,
            "rejected_results": rejected_results,
            "rejected_candidates": len(rejections) - rejected_results,
            "rejections": rejections[:200],
        }
        status = "completed" if (written_plans or rejections) else "no_memory"
        profile = self.profiles.get_customer_profile(
            account_scope_id=job["account_scope_id"], contact_id=job["contact_id"]
        )
        if profile is not None:
            self.archive.link_backfill_profile(job_id, profile.id)
        self.archive.finish_chunk(
            chunk["id"],
            status=status,
            output_json=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            write_result_json=json.dumps(result_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        current = self.archive.get_backfill_job(job_id)
        if current and current["status"] == "completed":
            self.archive.set_profile_history_status(
                account_scope_id=current["account_scope_id"],
                contact_id=current["contact_id"],
                status="memory_backfilled",
                detail=f"已完成 {current['completed_chunks']} 个历史片段",
            )
        elif current:
            self.archive.set_profile_history_status(
                account_scope_id=current["account_scope_id"],
                contact_id=current["contact_id"],
                status="partial",
                detail=f"已完成 {current['completed_chunks']}/{current['total_chunks']} 个历史片段",
            )
        if self.postprocessor:
            for item in writes:
                profile_id = str(item.get("profile_id") or "")
                event_ids = tuple(str(value) for value in item.get("created_event_ids", ()) if value)
                if profile_id:
                    self.postprocessor(profile_id, event_ids)
        return {
            "job_id": job_id,
            "chunk_index": chunk["chunk_index"],
            "status": status,
            **result_payload,
        }

    def pause(self, job_id: str) -> None:
        self.archive.pause_backfill(job_id)

    def resume(self, job_id: str) -> None:
        self.archive.resume_backfill(job_id)

    def _remap_references_for_current_profile(self, parsed, profile, source: Message):
        """Resolve a later chunk message against contexts created earlier.

        Provider input is a snapshot taken when the chunk is prepared. A prior
        message in the same chunk may create a context before a later explicit
        update (for example "今年扩大到30亩") is written. We bind only when an
        existing context is unique by crop, or when a change cue exists and the
        customer has exactly one active context. Ambiguity remains rejected.
        """
        if profile is None:
            return parsed
        contexts = self.profiles.list_active_growing_contexts(profile.id)
        if not contexts:
            return _drop_customer_level_event_contexts(parsed, source)
        known_context_ids = {context.id for context in contexts}
        remap: dict[str, str] = {}
        updated_contexts: list[dict[str, Any]] = []
        has_change_cue = any(
            cue in source.text for cue in ("现在", "今年", "扩大", "增加", "减少", "改成", "变成", "更新为", "已经")
        )
        for raw in parsed.growing_context_updates:
            item = dict(raw)
            reference = str(item.get("reference") or "")
            if reference not in known_context_ids:
                crop = re.sub(r"\s+", "", str(item.get("crop") or "")).casefold()
                crop_is_explicit = bool(
                    crop
                    and any(
                        _normal_text(term) in _normal_text(source.text)
                        for term in _crop_terms(_normalize_crop(crop))
                    )
                )
                matches = [
                    context for context in contexts
                    if crop_is_explicit
                    and re.sub(r"\s+", "", _normalize_crop(context.crop)).casefold()
                    == _normalize_crop(crop)
                ]
                if len(matches) == 1:
                    remap[reference] = matches[0].id
                    item["reference"] = matches[0].id
                elif has_change_cue and len(contexts) == 1:
                    # The local source itself states a change and this profile
                    # has only one active context.  This is a Level 2
                    # continuation bind; any ungrounded provider crop is still
                    # stripped later by ``_ground_context_fields``.
                    remap[reference] = contexts[0].id
                    item["reference"] = contexts[0].id
            updated_contexts.append(item)

        unique_context_id, _binding_level, _match_count = _unique_historical_context(
            contexts, source.text
        )

        def mapped(items, *, collection: str):
            result = []
            for raw in items:
                item = dict(raw)
                reference = item.get("growing_context_reference")
                if reference in remap:
                    item["growing_context_reference"] = remap[reference]
                elif (
                    reference
                    and reference not in known_context_ids
                    and unique_context_id
                ):
                    # Binding is based only on exact crop/variety evidence in
                    # the local source text.  The provider's invented
                    # ``new:*`` label is never treated as evidence.
                    item["growing_context_reference"] = unique_context_id
                elif (
                    collection == "events"
                    and reference
                    and reference not in known_context_ids
                    and _is_customer_level_historical_event(item, source.text)
                ):
                    # Customer-level sales/fulfilment history is valid without
                    # a crop context.  Preserve customer, conversation and
                    # exact source evidence; do not manufacture a crop link.
                    item["growing_context_reference"] = None
                result.append(item)
            return tuple(result)

        return replace(
            parsed,
            growing_context_updates=tuple(updated_contexts),
            facts=mapped(parsed.facts, collection="facts"),
            events=mapped(parsed.events, collection="events"),
            followups=mapped(parsed.followups, collection="followups"),
        )

    def _chunks(
        self,
        messages: list[HistoricalMessageRow],
        *,
        mapping: dict[str, Any],
        profile,
        policy: BackfillChunkPolicy,
    ) -> list[dict[str, Any]]:
        contexts = self.profiles.list_active_growing_contexts(profile.id)[:20] if profile else []
        facts = self.profiles.list_active_facts(profile.id)[:50] if profile else []
        result: list[dict[str, Any]] = []
        bucket: list[HistoricalMessageRow] = []
        characters = 0
        previous_at: datetime | None = None

        def flush() -> None:
            nonlocal bucket, characters
            if not bucket:
                return
            input_payload = {
                "purpose": "historical_customer_memory_backfill",
                "identity": {
                    "account_scope_id": mapping["account_scope_id"],
                    "contact_id": mapping["contact_id"],
                    "conversation_id": mapping["conversation_id"],
                },
                "existing_profile": {
                    "profile_id": profile.id if profile else None,
                    "display_name": profile.display_name if profile else mapping["source_display_name"],
                    "sales_stage": profile.sales_stage.value if profile else "new",
                },
                "active_growing_contexts": [
                    {
                        "id": item.id, "crop": item.crop, "variety": item.variety,
                        "region": item.region, "acreage_value": item.acreage_value,
                        "acreage_unit": item.acreage_unit, "growth_stage": item.growth_stage,
                        "cultivation_type": item.cultivation_type,
                        "season_or_year": item.season_or_year,
                    }
                    for item in contexts
                ],
                "active_facts": [
                    {
                        "id": item.id, "growing_context_id": item.growing_context_id,
                        "category": item.fact_category, "key": item.fact_key,
                        "value_text": item.value_text, "normalized_value": item.normalized_value,
                        "last_confirmed_at": item.last_confirmed_at,
                    }
                    for item in facts
                ],
                "messages": [
                    {
                        "source_ref": f"m{index:03d}",
                        "message_id": item.historical_message_id,
                        "sender": item.sender,
                        "text": item.text,
                        "sent_at": item.sent_at,
                        "message_type": item.message_type,
                    }
                    for index, item in enumerate(bucket)
                ],
                "output_contract": {
                    "results": [
                        {
                            "source_ref": "must exactly match a supplied source_ref",
                            "extraction": {
                                "growing_context_updates": [], "facts": [], "events": [],
                                "followups": [], "nothing_to_store": False,
                            },
                        }
                    ]
                },
            }
            serialized = json.dumps(input_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            result.append(
                {
                    "chunk_index": len(result),
                    "first_message_id": bucket[0].historical_message_id,
                    "last_message_id": bucket[-1].historical_message_id,
                    "message_count": len(bucket),
                    "character_count": characters,
                    "input_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                    "input_context_json": serialized,
                }
            )
            bucket = []
            characters = 0

        for message in messages:
            current_at = _parse_time(message.sent_at)
            gap_break = bool(
                bucket and previous_at and current_at
                and abs((current_at - previous_at).total_seconds()) > policy.break_after_days * 86400
            )
            size_break = bool(
                bucket and (
                    len(bucket) >= policy.max_messages
                    or characters + len(message.text) > policy.max_characters
                )
            )
            if gap_break or size_break:
                flush()
            bucket.append(message)
            characters += len(message.text)
            previous_at = current_at or previous_at
        flush()
        return result

    def _log(self, level: str, event: str, message: str) -> None:
        if self.logs:
            self.logs.append(level, "historical_backfill", event, message[:500])


def _drop_rejected_plan_candidate(
    parsed: ValidatedMemoryExtraction,
    error: str,
) -> tuple[ValidatedMemoryExtraction, str, int, str] | None:
    match = re.match(r"^(growing_context|fact|event|followup)\[(\d+)\]:(.+)$", error)
    if match is None:
        return None
    label, raw_index, reason = match.groups()
    attributes = {
        "growing_context": ("growing_context_updates", "growing_context_updates"),
        "fact": ("facts", "facts"),
        "event": ("events", "events"),
        "followup": ("followups", "followups"),
    }
    attribute, collection = attributes[label]
    values = list(getattr(parsed, attribute))
    index = int(raw_index)
    if not 0 <= index < len(values):
        return None
    values.pop(index)
    updated = replace(parsed, **{attribute: tuple(values)})
    updated = replace(
        updated,
        nothing_to_store=not any((
            updated.growing_context_updates, updated.facts,
            updated.events, updated.followups,
        )),
    )
    return updated, collection, index, reason


_CUSTOMER_LEVEL_HISTORICAL_EVENT_TYPES = frozenset({
    "order_intent",
    "quote_given",
    "order_placed",
    "purchased",
    "usage_started",
    "effect_feedback",
    "customer_concern",
})

_CUSTOMER_LEVEL_OTHER_TERMS = (
    "发货",
    "寄出",
    "物流",
    "快递",
    "收货",
    "收到货",
    "货收",
    "产品收到",
    "药收到",
    "肥收到",
    "付款",
    "转账",
    "打款",
    "已付",
    "回访",
    "跟进",
    "再联系",
)


def _is_customer_level_historical_event(item: dict[str, Any], source_text: str) -> bool:
    """Return true only for events whose meaning does not require a crop link."""
    event_type = str(item.get("event_type") or "")
    if event_type in _CUSTOMER_LEVEL_HISTORICAL_EVENT_TYPES:
        return True
    return event_type == "other" and any(term in source_text for term in _CUSTOMER_LEVEL_OTHER_TERMS)


def _unique_historical_context(contexts, source_text: str) -> tuple[str | None, str | None, int]:
    """Bind exact local evidence to one existing context, never fuzzy labels.

    A unique explicit variety is Level 2 evidence.  A unique exact crop/alias
    mention is Level 3 evidence.  Region, acreage, stage and provider-created
    reference names are intentionally insufficient on their own.
    """
    normalized_source = _normal_text(source_text)
    crop_matches: list[str] = []
    variety_matches: list[str] = []
    for context in contexts:
        crop = _normalize_crop(context.crop)
        if crop and any(
            _normal_text(term) in normalized_source for term in _crop_terms(crop)
        ):
            crop_matches.append(str(context.id))
        variety = _normal_text(context.variety or "")
        if len(variety) >= 2 and variety in normalized_source:
            variety_matches.append(str(context.id))
    variety_matches = list(dict.fromkeys(variety_matches))
    crop_matches = list(dict.fromkeys(crop_matches))
    if len(variety_matches) == 1:
        return variety_matches[0], "level_2_unique_variety", 1
    if len(crop_matches) == 1:
        return crop_matches[0], "level_3_unique_crop", 1
    return None, None, max(len(variety_matches), len(crop_matches))


def _drop_customer_level_event_contexts(parsed, source: Message):
    events = []
    for raw in parsed.events:
        item = dict(raw)
        if item.get("growing_context_reference") and _is_customer_level_historical_event(
            item, source.text
        ):
            item["growing_context_reference"] = None
        events.append(item)
    return replace(parsed, events=tuple(events))


def is_business_message(message: HistoricalMessageRow) -> bool:
    if message.mapping_status != "resolved" or message.sender not in {"other", "me"}:
        return False
    if message.message_type != "text":
        return False
    text = re.sub(r"\s+", "", message.text)
    if not text or text in {"[图片]", "[表情]", "[语音]", "图片", "表情", "收到", "好的", "嗯", "哦"}:
        return False
    if re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
        return False
    return True


def _history_as_message(item: dict[str, Any], job: dict[str, Any]) -> Message:
    speaker = Speaker.OTHER if item["sender"] == "other" else Speaker.ME
    return Message(
        id=str(item["message_id"]),
        contact_id=str(job["contact_id"]),
        speaker=speaker,
        text=str(item["text"]),
        content_type=str(item.get("message_type") or "text"),
        ocr_confidence=None,
        observed_at=str(item["sent_at"]),
        message_time=str(item["sent_at"]),
        time_source="historical_import",
        partial=False,
        fingerprint="",
        source="historical_import",
        conversation_id=str(job["conversation_id"]),
        account_scope_id=str(job["account_scope_id"]),
    )


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _safe_failed_output(raw_output: str) -> str:
    """Keep invalid provider output local for auditable, zero-call diagnosis."""
    if not raw_output:
        return "{}"
    try:
        json.loads(raw_output)
        return raw_output
    except json.JSONDecodeError:
        return json.dumps(
            {"invalid_raw_output": raw_output},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

