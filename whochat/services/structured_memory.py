from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

from whochat.ai.memory_models import (
    MemoryOutputError,
    ValidatedMemoryExtraction,
    parse_memory_extraction_json,
)
from whochat.ai.memory_provider import (
    MemoryProviderBlocked,
    MemoryProviderResult,
    OpenAICompatibleMemoryProvider,
)
from whochat.config import AppConfig, ConfigStore
from whochat.core.models import (
    CustomerEventStatus,
    CustomerFactProvenance,
    CustomerFactStatus,
    GrowingContextStatus,
    MemoryExtractionJob,
    MemoryExtractionJobStatus,
    Message,
    Speaker,
)
from whochat.storage.customer_profiles import CustomerProfileRepository
from whochat.storage.memory_jobs import MemoryExtractionJobRepository
from whochat.storage.memory_writer import (
    AtomicMemoryWriter,
    MemoryWritePlan,
    PlannedContext,
    PlannedEvent,
    PlannedFact,
    PlannedFollowup,
)
from whochat.storage.repositories import ContactRepository, LogRepository
from whochat.services.memory_canonicalization import (
    canonicalize_acreage,
    canonicalize_crop,
    canonicalize_fact_identity,
    canonicalize_growth_stage,
    crop_terms,
    extract_acreage,
    format_number,
    growth_stage_relation,
)
from whochat.services.memory_reconciliation import LocalExplicitFactReconciler


@dataclass(frozen=True)
class StructuredMemoryRunResult:
    discovered: int
    processed: int
    terminal_status: str = ""
    job_id: str = ""


class StructuredMemoryExtractionService:
    """Independent durable worker; it never calls or blocks the reply worker."""

    def __init__(
        self,
        jobs: MemoryExtractionJobRepository,
        profiles: CustomerProfileRepository,
        contacts: ContactRepository,
        *,
        provider=None,
        config_provider: Callable[[], AppConfig] | None = None,
        logs: LogRepository | None = None,
        poll_interval_seconds: float = 1.0,
        local_postprocessor: Callable[[str, tuple[str, ...]], None] | None = None,
    ) -> None:
        self.jobs = jobs
        self.profiles = profiles
        self.contacts = contacts
        self.writer = AtomicMemoryWriter(profiles.db)
        self.reconciler = LocalExplicitFactReconciler()
        self.provider = provider or OpenAICompatibleMemoryProvider()
        self.config_provider = config_provider or ConfigStore().load
        self.logs = logs
        self.poll_interval_seconds = max(0.2, float(poll_interval_seconds))
        self.local_postprocessor = local_postprocessor
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self.jobs.ensure_activation_cursor()
        self.jobs.recover_interrupted()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="whochat-memory", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def wake(self) -> None:
        self._wake.set()

    def run_once(self, *, max_jobs: int = 20) -> StructuredMemoryRunResult:
        if not self._lock.acquire(blocking=False):
            return StructuredMemoryRunResult(0, 0, "busy")
        try:
            config = self.config_provider()
            if not config.privacy.enable_long_term_memory:
                return StructuredMemoryRunResult(0, 0, "disabled")
            discovered = self.jobs.discover_new_messages(
                provider=config.ai.provider,
                model=config.ai.model,
                limit=max_jobs * 4,
            )
            processed = 0
            status = ""
            last_job_id = ""
            while processed < max_jobs:
                claimed = self.jobs.claim_next()
                if claimed is None:
                    break
                job, provider_required = claimed
                last_job_id = job.id
                status = self._process_job(job, provider_required, config)
                processed += 1
            return StructuredMemoryRunResult(len(discovered), processed, status, last_job_id)
        finally:
            self._lock.release()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self._log("error", "memory_worker_loop_failed", type(exc).__name__)
            self._wake.wait(self.poll_interval_seconds)
            self._wake.clear()

    def _process_job(self, job: MemoryExtractionJob, provider_required: bool, config: AppConfig) -> str:
        try:
            source = self.jobs.load_source_message(job)
            contact = self.contacts.get(job.contact_id)
            if contact is None:
                raise ValueError("contact_missing")
            profile = self.profiles.get_customer_profile(
                account_scope_id=job.account_scope_id,
                contact_id=job.contact_id,
            )
            input_json, input_hash = self._build_input(job, source, contact, profile)
            self.jobs.save_input(
                job.id,
                customer_profile_id=profile.id if profile else None,
                input_hash=input_hash,
                input_context_json=input_json,
            )
            if provider_required:
                if not self.jobs.begin_provider_call(job.id):
                    raise RuntimeError("provider_call_allowance_unavailable")
                self.jobs.mark_performance(job.id, "provider_started_at")
                provider_result: MemoryProviderResult = self.provider.extract(input_json, config)
                self.jobs.mark_performance(job.id, "provider_completed_at")
                try:
                    canonical_output = json.dumps(
                        json.loads(provider_result.raw_output),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                except json.JSONDecodeError as exc:
                    self.jobs.store_invalid_output(job.id, provider_result.raw_output, f"invalid_json:{exc.msg}")
                    self._log("warning", "memory_invalid_output", f"job={job.id} invalid_json")
                    return MemoryExtractionJobStatus.INVALID_OUTPUT.value
                self.jobs.save_provider_output(job.id, canonical_output, provider_result.elapsed_ms)
                output_payload = canonical_output
            else:
                current = self.jobs.get(job.id)
                output_payload = current.output_json if current else "{}"
            parsed = parse_memory_extraction_json(output_payload)
            if parsed.rejected_items:
                raise MemoryOutputError(";".join(parsed.rejected_items))
            plan = self._build_write_plan(
                job,
                source,
                contact.display_name,
                profile,
                parsed,
                allow_empty=parsed.nothing_to_store,
            )
            self.jobs.mark_performance(job.id, "write_started_at")
            if plan.item_count > 0:
                write_result = self.writer.apply(plan)
                self.jobs.attach_customer_profile(job.id, write_result["profile_id"])
                final = MemoryExtractionJobStatus.COMPLETED
            else:
                final = MemoryExtractionJobStatus.NO_MEMORY
                write_result = {
                    "profile_created": False,
                    "contexts": 0,
                    "facts": 0,
                    "events": 0,
                    "created_event_ids": [],
                    "followups": 0,
                    "ai_fact_candidates": 0,
                    "local_reconfirmations": 0,
                    "confirmed_fact_ids": [],
                    "added_source_count": 0,
                }
            self.jobs.mark_performance(job.id, "write_completed_at")
            self.jobs.finish(
                job.id,
                final,
                write_result_json=json.dumps(write_result, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                error="",
            )
            if final == MemoryExtractionJobStatus.COMPLETED and self.local_postprocessor:
                profile_id = str(write_result.get("profile_id") or "")
                event_ids = tuple(str(item) for item in write_result.get("created_event_ids", []) if item)
                if profile_id:
                    try:
                        self.local_postprocessor(profile_id, event_ids)
                    except Exception as exc:
                        # Sales/follow-up maintenance is intentionally isolated:
                        # it can be retried locally and must never turn a valid
                        # memory extraction into a failed AI job.
                        self._log(
                            "error",
                            "sales_postprocess_failed",
                            f"job={job.id} error={type(exc).__name__}",
                        )
            written = sum(int(write_result.get(key, 0)) for key in ("contexts", "facts", "events", "followups"))
            self._log("info", "memory_job_finished", f"job={job.id} status={final.value} written={written}")
            return final.value
        except MemoryOutputError as exc:
            self.jobs.finish(job.id, MemoryExtractionJobStatus.INVALID_OUTPUT, error=str(exc))
            return MemoryExtractionJobStatus.INVALID_OUTPUT.value
        except MemoryProviderBlocked as exc:
            self.jobs.finish(job.id, MemoryExtractionJobStatus.BLOCKED, error=str(exc))
            return MemoryExtractionJobStatus.BLOCKED.value
        except Exception as exc:
            self.jobs.finish(job.id, MemoryExtractionJobStatus.FAILED, error=f"{type(exc).__name__}:{exc}")
            self._log("error", "memory_job_failed", f"job={job.id} error={type(exc).__name__}")
            return MemoryExtractionJobStatus.FAILED.value

    def _build_input(self, job: MemoryExtractionJob, source: Message, contact, profile) -> tuple[str, str]:
        contexts = self.profiles.list_active_growing_contexts(profile.id)[:20] if profile else []
        facts = self.profiles.list_active_facts(profile.id)[:40] if profile else []
        recent = self.jobs.list_context_messages(job, limit=10)
        payload = {
            "schema": "whochat.memory_extraction.v1",
            "identity": {
                "account_scope_id": job.account_scope_id,
                "contact_id": job.contact_id,
                "conversation_id": job.conversation_id,
            },
            "customer_profile": {
                "id": profile.id if profile else None,
                "display_name": profile.display_name if profile else contact.display_name,
                "sales_stage": profile.sales_stage.value if profile else "new",
                "latest_contact_at": profile.latest_contact_at if profile else None,
            },
            "active_growing_contexts": [
                {
                    "id": item.id,
                    "crop": item.crop,
                    "variety": item.variety,
                    "region": item.region,
                    "acreage_value": item.acreage_value,
                    "acreage_unit": item.acreage_unit,
                    "growth_stage": item.growth_stage,
                    "cultivation_type": item.cultivation_type,
                    "season_or_year": item.season_or_year,
                }
                for item in contexts
            ],
            "active_facts": [
                {
                    "id": item.id,
                    "growing_context_id": item.growing_context_id,
                    "category": item.fact_category,
                    "key": item.fact_key,
                    "value_text": item.value_text,
                    "normalized_value": item.normalized_value,
                    "last_confirmed_at": item.last_confirmed_at,
                }
                for item in facts
            ],
            "recent_messages": [_message_payload(item) for item in recent],
            "source_message": _message_payload(source),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return serialized, hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_write_plan(
        self,
        job: MemoryExtractionJob,
        source: Message,
        display_name: str,
        profile,
        parsed: ValidatedMemoryExtraction,
        *,
        allow_empty: bool = False,
    ) -> MemoryWritePlan:
        if not job.conversation_id:
            raise MemoryOutputError("conversation_id_is_required")
        if parsed.rejected_items:
            raise MemoryOutputError(";".join(parsed.rejected_items))
        existing_contexts = self.profiles.list_active_growing_contexts(profile.id) if profile else []
        existing_facts = self.profiles.list_active_facts(profile.id) if profile else []
        existing_by_id = {item.id: item for item in existing_contexts}
        valid_references = set(existing_by_id)
        crop_by_reference = {item.id: _normalize_crop(item.crop) for item in existing_contexts}
        context_plans: list[PlannedContext] = []
        candidate_references: set[str] = set()

        for index, raw in enumerate(parsed.growing_context_updates):
            try:
                item = _ground_context_fields(_enforce_source_policy(raw, source), source)
                reference = item["reference"]
                if reference in candidate_references:
                    raise ValueError("duplicate_context_reference")
                candidate_references.add(reference)
                known_context_id = reference if reference in existing_by_id else None
                crop = _normalize_crop(item.get("crop"))
                if known_context_id is None and crop:
                    matches = [
                        existing for existing in existing_contexts
                        if _normalize_crop(existing.crop) == crop
                        and (
                            not item.get("season_or_year")
                            or not existing.season_or_year
                            or existing.season_or_year == item.get("season_or_year")
                        )
                    ]
                    if len(matches) == 1:
                        known_context_id = matches[0].id
                if known_context_id is None and not crop:
                    raise ValueError("new_context_requires_explicit_crop")
                stage = _normalize_stage(item.get("growth_stage"), source.text)
                acreage_value = item.get("acreage_value")
                acreage_unit = item.get("acreage_unit")
                if acreage_value is None:
                    parsed_area = _extract_acreage(source.text)
                    if parsed_area is not None:
                        acreage_value, acreage_unit = parsed_area
                current = existing_by_id.get(known_context_id) if known_context_id else None
                values = _safe_context_values(
                    item,
                    source,
                    current,
                    crop=crop or None,
                    stage=stage or None,
                    acreage_value=acreage_value,
                    acreage_unit=acreage_unit,
                )
                context_plans.append(
                    PlannedContext(
                        reference=reference,
                        existing_context_id=known_context_id,
                        status=_effective_context_status(item, source).value,
                        **values,
                    )
                )
                valid_references.add(reference)
                crop_by_reference[reference] = crop or (current.crop if current else "")
            except (ValueError, KeyError) as exc:
                raise MemoryOutputError(f"growing_context[{index}]:{exc}") from exc

        explicit_keys = {
            (item.get("growing_context_reference"), item.get("fact_key"))
            for item in parsed.facts
            if isinstance(item, dict)
        }
        fact_plans: list[PlannedFact] = []
        for context_raw, context_plan in zip(parsed.growing_context_updates, context_plans):
            item = _ground_context_fields(_enforce_source_policy(context_raw, source), source)
            derived = _context_fact_candidates(
                item,
                crop=context_plan.crop or "",
                stage=context_plan.growth_stage or "",
                acreage_value=context_plan.acreage_value,
                acreage_unit=context_plan.acreage_unit or "",
            )
            for fact in derived:
                if (context_plan.reference, fact["fact_key"]) in explicit_keys:
                    continue
                fact_plans.append(
                    PlannedFact(
                        growing_context_reference=context_plan.reference,
                        fact_category=fact["fact_category"],
                        fact_key=fact["fact_key"],
                        value_text=fact["value_text"],
                        value_json=fact["value_json"],
                        normalized_value=fact["normalized_value"],
                        confidence=float(item["confidence"]),
                        provenance=item["provenance"],
                        status=(
                            CustomerFactStatus.ACTIVE.value
                            if context_plan.status == GrowingContextStatus.ACTIVE.value
                            else CustomerFactStatus.UNCERTAIN.value
                        ),
                        resolution=_safe_resolution(
                            fact["fact_key"], item.get("resolution", "conflict"), source
                        ),
                    )
                )

        for index, raw in enumerate(parsed.facts):
            try:
                item = _enforce_source_policy(raw, source)
                _require_explicit_fact_grounding(item, source)
                reference = item.get("growing_context_reference")
                _require_known_reference(reference, valid_references)
                value_text = item["value_text"]
                normalized = _canonical_scalar(item.get("normalized_value"), fallback=value_text)
                if item["fact_key"] == "crop":
                    value_text = _normalize_crop(value_text)
                    normalized = value_text
                identity = canonicalize_fact_identity(
                    customer_profile_id=profile.id if profile else "",
                    growing_context_id=reference,
                    fact_category=item["fact_category"],
                    fact_key=item["fact_key"],
                    value_text=value_text,
                    normalized_value=normalized,
                    value_json=item.get("value_json", {}),
                    source_text=source.text,
                )
                fact_plans.append(
                    PlannedFact(
                        growing_context_reference=reference,
                        fact_category=identity.canonical_category,
                        fact_key=identity.canonical_key,
                        value_text=value_text,
                        value_json=item.get("value_json", {}),
                        normalized_value=identity.canonical_value,
                        confidence=float(item["confidence"]),
                        provenance=item["provenance"],
                        status=_effective_fact_status(item, source).value,
                        resolution=_safe_resolution(
                            identity.canonical_key, item.get("resolution", "conflict"), source
                        ),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise MemoryOutputError(f"fact[{index}]:{exc}") from exc

        event_plans: list[PlannedEvent] = []
        for index, raw in enumerate(parsed.events):
            try:
                item = _enforce_source_policy(raw, source)
                reference = item.get("growing_context_reference")
                if reference is not None:
                    _require_known_reference(reference, valid_references)
                else:
                    reference = _infer_event_reference(crop_by_reference, source.text)
                provenance = CustomerFactProvenance(item["provenance"])
                event_plans.append(
                    PlannedEvent(
                        growing_context_reference=reference,
                        event_type=item["event_type"],
                        title=item["title"],
                        summary=item.get("summary") or "",
                        structured_data=item.get("structured_data", {}),
                        confidence=float(item["confidence"]),
                        provenance=provenance.value,
                        status=(
                            CustomerEventStatus.UNCERTAIN.value
                            if provenance == CustomerFactProvenance.AI_INFERRED
                            else CustomerEventStatus.ACTIVE.value
                        ),
                    )
                )
                if item["event_type"] == "customer_concern":
                    concern = item.get("structured_data", {}).get("concern")
                    if isinstance(concern, str) and concern.strip():
                        concern = concern.strip()
                        value_text = f"担心{concern}" if _normal_text(f"担心{concern}") in _normal_text(source.text) else concern
                        if _normal_text(value_text) in _normal_text(source.text):
                            fact_plans.append(
                                PlannedFact(
                                    growing_context_reference=reference,
                                    fact_category="customer_preference",
                                    fact_key="customer_concern",
                                    value_text=value_text,
                                    value_json={"concern": concern},
                                    normalized_value=canonicalize_fact_identity(
                                        customer_profile_id=profile.id if profile else "",
                                        growing_context_id=reference,
                                        fact_category="customer_preference",
                                        fact_key="customer_concern",
                                        value_text=value_text,
                                        normalized_value=concern,
                                        value_json={"concern": concern},
                                        source_text=source.text,
                                    ).canonical_value,
                                    confidence=float(item["confidence"]),
                                    provenance=provenance.value,
                                    status=(
                                        CustomerFactStatus.ACTIVE.value
                                        if (
                                            source.speaker == Speaker.OTHER
                                            and provenance == CustomerFactProvenance.CUSTOMER_EXPLICIT
                                            and float(item["confidence"]) >= 0.75
                                        )
                                        else CustomerFactStatus.UNCERTAIN.value
                                    ),
                                    resolution="coexist",
                                )
                            )
            except (ValueError, KeyError) as exc:
                raise MemoryOutputError(f"event[{index}]:{exc}") from exc

        followup_plans: list[PlannedFollowup] = []
        for index, raw in enumerate(parsed.followups):
            try:
                item = _enforce_source_policy(raw, source)
                _validate_followup_evidence(item, source)
                reference = item.get("growing_context_reference")
                _require_known_reference(reference, valid_references)
                followup_plans.append(
                    PlannedFollowup(
                        growing_context_reference=reference,
                        followup_type=item["followup_type"],
                        title=item["title"],
                        due_at=item["due_at"],
                        priority=int(item["priority"]),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise MemoryOutputError(f"followup[{index}]:{exc}") from exc

        if profile is not None:
            fact_plans.extend(
                self.reconciler.reconcile(
                    source=source,
                    customer_profile_id=profile.id,
                    contexts=existing_contexts,
                    facts=existing_facts,
                )
            )

        reference_map = {
            item.reference: item.existing_context_id or item.reference
            for item in context_plans
        }
        reference_map.update({context_id: context_id for context_id in existing_by_id})

        plan = MemoryWritePlan(
            account_scope_id=job.account_scope_id,
            target_instance_id=job.target_instance_id,
            contact_id=job.contact_id,
            conversation_id=job.conversation_id,
            display_name=display_name,
            source_message_id=source.id,
            source_text=source.text,
            source_observed_at=source.observed_at,
            existing_context_ids=tuple(existing_by_id),
            contexts=tuple(context_plans),
            facts=tuple(_dedupe_planned_facts(fact_plans, reference_map=reference_map)),
            events=tuple(event_plans),
            followups=tuple(followup_plans),
        )
        if plan.item_count <= 0 and not allow_empty:
            raise MemoryOutputError("validated_write_plan_is_empty")
        return plan

    def replay_saved_output(
        self,
        job_id: str,
        *,
        failure_injector=None,
    ) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError("memory_job_not_found")
        if not job.output_json or job.output_json == "{}":
            raise ValueError("memory_job_has_no_saved_output")
        source = self.jobs.load_source_message(job)
        contact = self.contacts.get(job.contact_id)
        if contact is None:
            raise ValueError("contact_missing")
        profile = self.profiles.get_customer_profile(
            account_scope_id=job.account_scope_id,
            contact_id=job.contact_id,
        )
        calls_before = job.ai_call_count
        parsed = parse_memory_extraction_json(job.output_json)
        plan = self._build_write_plan(
            job,
            source,
            contact.display_name,
            profile,
            parsed,
            allow_empty=parsed.nothing_to_store,
        )
        if plan.item_count > 0:
            result = self.writer.apply(plan, failure_injector=failure_injector)
        else:
            result = {
                "profile_id": profile.id if profile else None,
                "profile_created": False,
                "contexts": 0,
                "contexts_created": 0,
                "context_ids": [],
                "facts": 0,
                "facts_created": 0,
                "fact_ids": [],
                "ai_fact_candidates": 0,
                "local_reconfirmations": 0,
                "confirmed_fact_ids": [],
                "added_source_count": 0,
                "events": 0,
                "events_created": 0,
                "event_ids": [],
                "followups": 0,
                "followups_created": 0,
                "followup_ids": [],
            }
        current = self.jobs.get(job_id)
        if current is None or current.ai_call_count != calls_before:
            raise RuntimeError("local_replay_changed_provider_call_count")
        self._log(
            "info",
            "memory_local_replay_completed",
            f"job={job_id} original_status={job.status.value} writes={plan.item_count}",
        )
        return {
            **result,
            "job_id": job_id,
            "original_job_status": job.status.value,
            "provider_calls_before": calls_before,
            "provider_calls_after": current.ai_call_count,
        }

    def _validate_and_write(
        self,
        job: MemoryExtractionJob,
        source: Message,
        profile_id: str,
        parsed: ValidatedMemoryExtraction,
    ) -> dict:
        raise RuntimeError("non_atomic_memory_write_path_disabled")
        result = {
            "contexts": 0,
            "contexts_created": 0,
            "facts": 0,
            "facts_created": 0,
            "events": 0,
            "followups": 0,
            "rejected": list(parsed.rejected_items),
        }
        existing_contexts = self.profiles.list_active_growing_contexts(profile_id)
        context_map = {item.id: item.id for item in existing_contexts}
        context_objects = {item.id: item for item in existing_contexts}

        for index, raw in enumerate(parsed.growing_context_updates):
            try:
                item = _enforce_source_policy(raw, source)
                item = _ground_context_fields(item, source)
                reference = item["reference"]
                known_context_id = context_map.get(reference)
                if known_context_id is None and not reference.startswith("new:"):
                    raise ValueError("unknown_context_reference")
                crop = _normalize_crop(item.get("crop", ""))
                if known_context_id is None and crop:
                    crop_matches = [
                        existing for existing in existing_contexts
                        if _normalize_crop(existing.crop) == crop
                        and (
                            not item.get("season_or_year")
                            or not existing.season_or_year
                            or existing.season_or_year == item.get("season_or_year")
                        )
                    ]
                    if len(crop_matches) == 1:
                        known_context_id = crop_matches[0].id
                if known_context_id is None and not crop:
                    raise ValueError("new_context_requires_explicit_crop")
                stage = _normalize_stage(item.get("growth_stage", ""), source.text)
                acreage_value = item.get("acreage_value")
                acreage_unit = item.get("acreage_unit", "")
                if acreage_value is None:
                    parsed_area = _extract_acreage(source.text)
                    if parsed_area is not None:
                        acreage_value, acreage_unit = parsed_area
                context_status = _effective_context_status(item, source)
                current = context_objects.get(known_context_id) if known_context_id else None
                context_values = _safe_context_values(
                    item,
                    source,
                    current,
                    crop=crop,
                    stage=stage,
                    acreage_value=acreage_value,
                    acreage_unit=acreage_unit,
                )
                context, created = self.profiles.add_or_confirm_growing_context(
                    profile_id,
                    context_id=known_context_id,
                    **context_values,
                    status=context_status,
                    confirmed_at=source.observed_at,
                )
                context_map[reference] = context.id
                context_map[context.id] = context.id
                context_objects[context.id] = context
                result["contexts"] += 1
                result["contexts_created"] += int(created)
                derived = _context_fact_candidates(
                    item,
                    crop=crop,
                    stage=stage,
                    acreage_value=acreage_value,
                    acreage_unit=acreage_unit,
                )
                for fact in derived:
                    stored, fact_created = self.profiles.add_or_confirm_fact(
                        profile_id,
                        growing_context_id=context.id,
                        fact_category=fact["fact_category"],
                        fact_key=fact["fact_key"],
                        value_text=fact["value_text"],
                        value_json=fact["value_json"],
                        normalized_value=fact["normalized_value"],
                        confidence=item["confidence"],
                        provenance=item["provenance"],
                        status=CustomerFactStatus.ACTIVE if context_status == GrowingContextStatus.ACTIVE else CustomerFactStatus.UNCERTAIN,
                        resolution=_safe_resolution(fact["fact_key"], item.get("resolution", "conflict"), source),
                        source_message_id=source.id,
                        source_conversation_id=job.conversation_id,
                        source_text=source.text,
                        source_observed_at=source.observed_at,
                    )
                    result["facts"] += 1
                    result["facts_created"] += int(fact_created)
            except (ValueError, KeyError) as exc:
                result["rejected"].append(f"growing_context[{index}]:{exc}")

        for index, raw in enumerate(parsed.facts):
            try:
                item = _enforce_source_policy(raw, source)
                _require_explicit_fact_grounding(item, source)
                context_id = _context_reference(item.get("growing_context_reference"), context_map)
                status = _effective_fact_status(item, source)
                resolution = _safe_resolution(item["fact_key"], item.get("resolution", "conflict"), source)
                value_text = item["value_text"]
                normalized = item["normalized_value"]
                if item["fact_key"] == "crop":
                    value_text = _normalize_crop(value_text)
                    normalized = value_text
                stored, created = self.profiles.add_or_confirm_fact(
                    profile_id,
                    growing_context_id=context_id,
                    fact_category=item["fact_category"],
                    fact_key=item["fact_key"],
                    value_text=value_text,
                    value_json=item.get("value_json", {}),
                    normalized_value=normalized,
                    confidence=item["confidence"],
                    provenance=item["provenance"],
                    status=status,
                    resolution=resolution,
                    source_message_id=source.id,
                    source_conversation_id=job.conversation_id,
                    source_text=source.text,
                    source_observed_at=source.observed_at,
                )
                result["facts"] += 1
                result["facts_created"] += int(created)
            except (ValueError, KeyError) as exc:
                result["rejected"].append(f"fact[{index}]:{exc}")

        for index, raw in enumerate(parsed.events):
            try:
                item = _enforce_source_policy(raw, source)
                context_id = _event_context_reference(
                    item.get("growing_context_reference"), context_map, context_objects, source.text
                )
                provenance = CustomerFactProvenance(item["provenance"])
                status = CustomerEventStatus.UNCERTAIN if provenance == CustomerFactProvenance.AI_INFERRED else CustomerEventStatus.ACTIVE
                event = self.profiles.create_customer_event(
                    profile_id,
                    growing_context_id=context_id,
                    event_type=item["event_type"],
                    title=item["title"],
                    summary=item.get("summary", ""),
                    structured_data_json=item.get("structured_data", {}),
                    event_at=source.observed_at,
                    confidence=item["confidence"],
                    provenance=provenance,
                    status=status,
                )
                self.profiles.add_event_source(
                    event.id,
                    message_id=source.id,
                    conversation_id=job.conversation_id,
                    source_text=source.text,
                    observed_at=source.observed_at,
                )
                result["events"] += 1
            except (ValueError, KeyError) as exc:
                result["rejected"].append(f"event[{index}]:{exc}")

        for index, raw in enumerate(parsed.followups):
            try:
                item = _enforce_source_policy(raw, source)
                _validate_followup_evidence(item, source)
                context_id = _context_reference(item.get("growing_context_reference"), context_map)
                self.profiles.create_followup(
                    profile_id,
                    growing_context_id=context_id,
                    followup_type=item["followup_type"],
                    title=item["title"],
                    due_at=item["due_at"],
                    priority=item["priority"],
                )
                result["followups"] += 1
            except (ValueError, KeyError) as exc:
                result["rejected"].append(f"followup[{index}]:{exc}")
        return result

    def _log(self, level: str, event: str, message: str) -> None:
        if self.logs is not None:
            self.logs.append(level, "structured_memory", event, message[:500])


def _message_payload(message: Message) -> dict:
    return {
        "message_id": message.id,
        "sender": message.speaker.value,
        "text": message.text,
        "observed_at": message.observed_at,
    }


def _enforce_source_policy(item: dict, source: Message) -> dict:
    if item["source_message_id"] != source.id:
        raise ValueError("source_message_id_not_allowed")
    if _normal_text(item["source_text"]) != _normal_text(source.text):
        raise ValueError("source_text_does_not_match_message")
    provenance = CustomerFactProvenance(item["provenance"])
    if source.speaker == Speaker.OTHER and provenance == CustomerFactProvenance.USER_EXPLICIT:
        raise ValueError("other_message_cannot_be_user_explicit")
    if source.speaker == Speaker.ME and provenance == CustomerFactProvenance.CUSTOMER_EXPLICIT:
        raise ValueError("me_message_cannot_be_customer_explicit")
    if provenance == CustomerFactProvenance.AI_INFERRED:
        item = dict(item)
        item["confidence"] = min(float(item["confidence"]), 0.69)
        item["status"] = "uncertain"
    return item


def _effective_context_status(item: dict, source: Message) -> GrowingContextStatus:
    provenance = CustomerFactProvenance(item["provenance"])
    if (
        source.speaker == Speaker.OTHER
        and provenance == CustomerFactProvenance.CUSTOMER_EXPLICIT
        and float(item["confidence"]) >= 0.75
        and item.get("status") == "active"
    ):
        return GrowingContextStatus.ACTIVE
    return GrowingContextStatus.UNCERTAIN


def _effective_fact_status(item: dict, source: Message) -> CustomerFactStatus:
    provenance = CustomerFactProvenance(item["provenance"])
    if (
        source.speaker == Speaker.OTHER
        and provenance == CustomerFactProvenance.CUSTOMER_EXPLICIT
        and float(item["confidence"]) >= 0.75
        and item.get("status") == "active"
    ):
        return CustomerFactStatus.ACTIVE
    return CustomerFactStatus.UNCERTAIN


def _ground_context_fields(item: dict, source: Message) -> dict:
    provenance = CustomerFactProvenance(item["provenance"])
    if provenance == CustomerFactProvenance.AI_INFERRED:
        return item
    grounded = dict(item)
    source_text = _normal_text(source.text)
    crop = _normalize_crop(grounded.get("crop", ""))
    if crop and not any(_normal_text(term) in source_text for term in _crop_terms(crop)):
        grounded["crop"] = ""
    for field in ("variety", "region", "cultivation_type", "season_or_year"):
        value = grounded.get(field, "")
        if value and _normal_text(value) not in source_text:
            grounded[field] = ""
    stage = _normalize_stage(grounded.get("growth_stage", ""), source.text)
    if stage:
        stage_root = stage[:-1] if stage.endswith("期") else stage
        source_stage = _normalize_stage(source.text, source.text)
        relation = growth_stage_relation(stage, source_stage, source_text=source.text)
        if (
            _normal_text(stage) not in source_text
            and _normal_text(stage_root) not in source_text
            and relation not in {"same", "compatible"}
        ):
            grounded["growth_stage"] = ""
    acreage_value = grounded.get("acreage_value")
    acreage_unit = grounded.get("acreage_unit", "")
    if acreage_value is not None or acreage_unit:
        parsed = _extract_acreage(source.text)
        if (
            parsed is None
            or acreage_value is None
            or float(parsed[0]) != float(acreage_value)
            or parsed[1] != acreage_unit
        ):
            grounded["acreage_value"] = None
            grounded["acreage_unit"] = ""
    if not any(
        grounded.get(field) not in ("", None)
        for field in (
            "crop", "variety", "region", "acreage_value", "growth_stage",
            "cultivation_type", "season_or_year",
        )
    ):
        raise ValueError("context_has_no_locally_grounded_field")
    return grounded


def _require_explicit_fact_grounding(item: dict, source: Message) -> None:
    provenance = CustomerFactProvenance(item["provenance"])
    if provenance == CustomerFactProvenance.AI_INFERRED:
        return
    source_text = _normal_text(source.text)
    values = (
        _normal_text(item.get("value_text", "")),
        _normal_text(item.get("normalized_value", "")),
    )
    if any(value and value in source_text for value in values):
        return
    if item.get("fact_key") == "crop":
        crop = _normalize_crop(item.get("value_text", ""))
        if crop and any(_normal_text(term) in source_text for term in _crop_terms(crop)):
            return
    if item.get("fact_key") == "growth_stage":
        stage = _normalize_stage(item.get("value_text", ""), source.text)
        stage_root = stage[:-1] if stage.endswith("期") else stage
        source_stage = _normalize_stage(source.text, source.text)
        relation = growth_stage_relation(stage, source_stage, source_text=source.text)
        if (stage_root and _normal_text(stage_root) in source_text) or relation in {"same", "compatible"}:
            return
    raise ValueError("explicit_fact_value_not_grounded_in_source")


def _context_reference(reference: str | None, context_map: dict[str, str]) -> str | None:
    if not reference:
        return None
    context_id = context_map.get(reference)
    if context_id is None:
        raise ValueError("unknown_context_reference")
    return context_id


def _require_known_reference(reference: str | None, valid_references: set[str]) -> None:
    if reference is not None and reference not in valid_references:
        raise ValueError(f"unknown_context_reference:{reference}")


def _infer_event_reference(crop_by_reference: dict[str, str], source_text: str) -> str | None:
    matches = []
    for reference, crop in crop_by_reference.items():
        if crop and any(term in source_text for term in _crop_terms(crop)):
            matches.append(reference)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    unique = list(dict.fromkeys(crop_by_reference))
    return unique[0] if not matches and len(unique) == 1 else None


def _canonical_scalar(value, *, fallback: str) -> str:
    if value is None:
        return _normal_text(fallback)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        return cleaned.casefold() if cleaned else _normal_text(fallback)
    raise ValueError("normalized_value_is_not_json_scalar")


def _dedupe_planned_facts(
    items: list[PlannedFact],
    *,
    reference_map: dict[str, str] | None = None,
) -> list[PlannedFact]:
    result: list[PlannedFact] = []
    positions: dict[tuple[str | None, str, str, str], int] = {}
    reference_map = reference_map or {}
    for item in items:
        key = (
            reference_map.get(
                item.growing_context_reference or "",
                item.growing_context_reference,
            ),
            item.fact_category,
            item.fact_key,
            item.normalized_value,
        )
        position = positions.get(key)
        if position is not None:
            existing = result[position]
            result[position] = replace(
                existing,
                confirmation_only=(
                    existing.confirmation_only and item.confirmation_only
                ),
                existing_fact_id=(
                    existing.existing_fact_id or item.existing_fact_id
                ),
                candidate_origins=tuple(
                    dict.fromkeys((*existing.candidate_origins, *item.candidate_origins))
                ),
            )
            continue
        positions[key] = len(result)
        result.append(item)
    return result


def _event_context_reference(reference, context_map, context_objects, source_text) -> str | None:
    if reference:
        return _context_reference(reference, context_map)
    matches = []
    for context_id, context in context_objects.items():
        crop = _normalize_crop(context.crop)
        if crop and any(alias in source_text for alias in _crop_terms(crop)):
            matches.append(context_id)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    unique_contexts = list(dict.fromkeys(context_map.values()))
    return unique_contexts[0] if not matches and len(unique_contexts) == 1 else None


def _safe_context_values(
    item: dict,
    source: Message,
    current,
    *,
    crop: str,
    stage: str,
    acreage_value: float | None,
    acreage_unit: str,
) -> dict:
    values = {
        "crop": crop,
        "variety": item.get("variety", ""),
        "region": item.get("region", ""),
        "acreage_value": acreage_value,
        "acreage_unit": acreage_unit,
        "growth_stage": stage,
        "cultivation_type": item.get("cultivation_type", ""),
        "season_or_year": item.get("season_or_year", ""),
    }
    if current is None:
        return values
    current_values = {
        "crop": current.crop,
        "variety": current.variety,
        "region": current.region,
        "acreage_value": current.acreage_value,
        "acreage_unit": current.acreage_unit,
        "growth_stage": current.growth_stage,
        "cultivation_type": current.cultivation_type,
        "season_or_year": current.season_or_year,
    }
    fact_keys = {
        "crop": "crop",
        "variety": "variety",
        "region": "region",
        "acreage_value": "acreage",
        "acreage_unit": "acreage",
        "growth_stage": "growth_stage",
        "cultivation_type": "cultivation_type",
        "season_or_year": "season_or_year",
    }
    requested = item.get("resolution", "conflict")
    for field, value in list(values.items()):
        old = current_values[field]
        if value in ("", None) or old in ("", None) or _normal_text(str(value)) == _normal_text(str(old)):
            continue
        if _safe_resolution(fact_keys[field], requested, source) != "supersede":
            values[field] = None if field == "acreage_value" else ""
    return values


def _context_fact_candidates(
    item: dict,
    *,
    crop: str,
    stage: str,
    acreage_value: float | None,
    acreage_unit: str,
) -> list[dict]:
    entries: list[tuple[str, str, object, str]] = []
    if crop:
        entries.append(("crop", crop, {}, crop))
    if item.get("variety"):
        entries.append(("variety", item["variety"], {}, _normal_text(item["variety"])))
    if item.get("region"):
        entries.append(("region", item["region"], {}, _normal_text(item["region"])))
    if acreage_value is not None and acreage_unit:
        display = f"{_format_number(acreage_value)}{acreage_unit}"
        entries.append((
            "acreage", display,
            {"value": acreage_value, "unit": acreage_unit},
            canonicalize_acreage(
                value_text=display,
                value_json={"value": acreage_value, "unit": acreage_unit},
            ),
        ))
    if stage:
        entries.append(("growth_stage", stage, {}, stage))
    if item.get("cultivation_type"):
        entries.append(("cultivation_type", item["cultivation_type"], {}, _normal_text(item["cultivation_type"])))
    if item.get("season_or_year"):
        entries.append(("season_or_year", item["season_or_year"], {}, _normal_text(item["season_or_year"])))
    return [
        {
            "fact_category": "growing_context",
            "fact_key": key,
            "value_text": value,
            "value_json": value_json,
            "normalized_value": normalized,
        }
        for key, value, value_json, normalized in entries
    ]


def _safe_resolution(fact_key: str, requested: str, source: Message) -> str:
    dynamic_keys = {"acreage", "growth_stage", "variety", "region", "planting_date", "expected_harvest", "harvest_status"}
    has_change_cue = any(cue in source.text for cue in ("现在", "今年", "扩大", "增加", "减少", "改成", "变成", "更新为", "已经"))
    if fact_key in dynamic_keys and has_change_cue and source.speaker == Speaker.OTHER:
        return "supersede"
    if requested == "supersede" and not has_change_cue:
        return "conflict"
    return requested


def _validate_followup_evidence(item: dict, source: Message) -> None:
    if _normal_text(item["time_evidence"]) not in _normal_text(source.text):
        raise ValueError("time_evidence_not_in_source")
    if _normal_text(item["action_evidence"]) not in _normal_text(source.text):
        raise ValueError("action_evidence_not_in_source")
    if not _EXPLICIT_TIME_RE.search(source.text):
        raise ValueError("followup_has_no_explicit_time")
    try:
        datetime.fromisoformat(item["due_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("followup_due_at_not_iso8601") from exc


def _normalize_crop(value: str) -> str:
    return canonicalize_crop(value)


def _crop_terms(canonical: str) -> tuple[str, ...]:
    return crop_terms(canonical)


def _normalize_stage(value: str, source_text: str = "") -> str:
    return canonicalize_growth_stage(value, source_text=source_text)


def _extract_acreage(text: str) -> tuple[float, str] | None:
    return extract_acreage(text)


def _format_number(value: float) -> str:
    return format_number(value)


def _normal_text(value: str) -> str:
    return "".join(str(value or "").split()).casefold()


_EXPLICIT_TIME_RE = re.compile(r"(?:\d+\s*(?:天|日|周|个月|月)后|明天|后天|\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?|\d{1,2}月\d{1,2}日)")
