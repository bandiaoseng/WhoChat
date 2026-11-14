from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from whochat.config import ConfigStore
from whochat.ai.current_work_authorization import current_context_source_error
from whochat.ocr.engine import create_ocr_engine
from whochat.storage.database import Database
from whochat.storage.customer_profiles import CustomerProfileRepository
from whochat.storage.memory_jobs import MemoryExtractionJobRepository
from whochat.storage.product_knowledge import ProductKnowledgeRepository
from whochat.storage.knowledge_gaps import KnowledgeGapRepository, ReplyPipelineMetricRepository
from whochat.storage.historical import HistoricalArchiveRepository
from whochat.storage.repositories import (
    CalibrationRepository,
    AccountScopeRepository,
    AIAnalysisEventRepository,
    CaptureSampleRepository,
    ContactRepository,
    ConversationRepository,
    IdentityBindingRepository,
    LogRepository,
    MemoryRepository,
    MessageRepository,
    ObservationRepository,
    GenerationLogRepository,
    IdentityRepository,
    ReplyFeedbackRepository,
    StrategyRepository,
    SettingsAuditRepository,
)
from whochat.services.ingestion import (
    ChatIngestionService,
    _title_confidence,
    extract_title_candidates,
    normalize_contact_title,
)
from whochat.core.models import ConversationType
from whochat.services.conversation_timeline import PersistentConversationTimeline
from whochat.services.customer_memory_context import CustomerMemoryContextBuilder
from whochat.services.customer_profile_service import CustomerProfileService
from whochat.services.sales_followup import SalesFollowupService
from whochat.services.identity_lock import StableIdentityLock
from whochat.services.current_identity import CurrentConversationIdentityController
from whochat.services.account_scope import AccountScopeResolver
from whochat.services.autocapture import AutoCaptureController, ContactIdentityRecoveryController
from whochat.services.environment import EnvironmentDiagnosticsService
from whochat.services.governance import DataGovernanceService
from whochat.services.reply import ReplyGenerationService
from whochat.services.reply_result_cache import ReplyResultCache
from whochat.services.writing_feedback import WritingFeedbackService
from whochat.services.reply_tasks import ReplyTaskService
from whochat.services.proactive_contact import (
    ProactiveContactCacheRepository,
    ProactiveContactContextBuilder,
    ProactiveContactService,
    ProactiveContactTaskService,
    ProactiveSuggestionFeedbackService,
    SalesWritingPreferenceService,
    SuggestionFeedbackRepository,
)
from whochat.services.instant_activation import (
    ActivationPrewarmScheduler,
    ActivationSnapshotService,
    CompactActivationIndex,
    FastCustomerResolver,
)
from whochat.services.fast_activation_probe import FastActivationProbeService
from whochat.services.manual_suggestions import MANUAL_SUGGESTIONS_ONLY
from whochat.services.conversation_style import CustomerSalutationResolver
from whochat.services.structured_memory import StructuredMemoryExtractionService
from whochat.services.historical_import import HistoricalImportService
from whochat.services.historical_backfill import (
    DisabledHistoricalBackfillProvider,
    HistoricalMemoryBackfillService,
)
from whochat.services.tracking_advisor import CustomerTrackingAdvisor
from whochat.services.knowledge_import import KnowledgeImportService
from whochat.services.product_knowledge_context import (
    GENERIC_KNOWLEDGE_KINDS,
    ProductKnowledgeContextBuilder,
)
from whochat.services.knowledge_gap import KnowledgeGapAnalyzer, KnowledgeCoverageReportService
from whochat.services.retention import RetentionCleanupService
from whochat.services.runtime import RuntimeStateService
from whochat.services.identity_events import IdentityEventCoordinator
from whochat.services.pipeline import CapturePipelineService
from whochat.services.transcript_stitcher import TranscriptStitcher
from dataclasses import replace


@dataclass(frozen=True)
class AppServices:
    db: Database
    strategies: StrategyRepository
    contacts: ContactRepository
    customer_profiles: CustomerProfileRepository
    customer_profile_service: CustomerProfileService
    customer_tracking_advisor: CustomerTrackingAdvisor
    sales_followup: SalesFollowupService
    reply_memory_context: CustomerMemoryContextBuilder
    identity_bindings: IdentityBindingRepository
    account_scopes: AccountScopeRepository
    identities: IdentityRepository
    conversations: ConversationRepository
    analysis_events: AIAnalysisEventRepository
    messages: MessageRepository
    observations: ObservationRepository
    memories: MemoryRepository
    logs: LogRepository
    generation_logs: GenerationLogRepository
    reply_feedback: ReplyFeedbackRepository
    capture_samples: CaptureSampleRepository
    settings_audit: SettingsAuditRepository
    runtime: RuntimeStateService
    calibrations: CalibrationRepository
    pipeline: CapturePipelineService
    reply_generator: ReplyGenerationService
    reply_tasks: ReplyTaskService
    proactive_contact: ProactiveContactService
    proactive_tasks: ProactiveContactTaskService
    proactive_feedback: ProactiveSuggestionFeedbackService
    sales_writing_preferences: SalesWritingPreferenceService
    conversation_styles: CustomerSalutationResolver
    compact_activation_index: CompactActivationIndex
    fast_customer_resolver: FastCustomerResolver
    activation_snapshots: ActivationSnapshotService
    activation_prewarm: ActivationPrewarmScheduler
    fast_activation_probe: FastActivationProbeService
    ingestion: ChatIngestionService
    memory_jobs: MemoryExtractionJobRepository
    structured_memory: StructuredMemoryExtractionService
    historical_archive: HistoricalArchiveRepository
    historical_import: HistoricalImportService
    historical_backfill: HistoricalMemoryBackfillService
    product_knowledge: ProductKnowledgeRepository
    knowledge_import: KnowledgeImportService
    product_knowledge_context: ProductKnowledgeContextBuilder
    knowledge_gaps: KnowledgeGapRepository
    knowledge_gap_analyzer: KnowledgeGapAnalyzer
    knowledge_coverage_report: KnowledgeCoverageReportService
    reply_pipeline_metrics: ReplyPipelineMetricRepository
    autocapture: AutoCaptureController
    identity_recovery: ContactIdentityRecoveryController
    governance: DataGovernanceService
    retention: RetentionCleanupService
    environment: EnvironmentDiagnosticsService
    transcript_stitcher: TranscriptStitcher
    identity_lock: StableIdentityLock
    current_identity: CurrentConversationIdentityController
    account_scope_resolver: AccountScopeResolver
    identity_events: IdentityEventCoordinator
    timeline: PersistentConversationTimeline
    writing_feedback: WritingFeedbackService | None = None

    def shutdown(self) -> None:
        try:
            self.activation_prewarm.stop()
        except Exception:
            pass
        try:
            self.fast_activation_probe.stop()
        except Exception:
            pass
        try:
            self.identity_recovery.stop()
        except Exception:
            pass
        try:
            self.autocapture.stop()
            self.autocapture.set_enabled(False)
        except Exception:
            pass
        try:
            self.structured_memory.shutdown()
        except Exception:
            pass
        try:
            self.pipeline.shutdown()
        except Exception:
            pass
        try:
            self.reply_tasks.shutdown()
        except Exception:
            pass
        try:
            self.proactive_tasks.shutdown()
        except Exception:
            pass
        try:
            self.account_scope_resolver.native_identity.close()
        except Exception:
            pass
        try:
            self.logs.append("info", "bootstrap", "services_shutdown", "Application services shut down")
        except Exception:
            pass


def build_services(db_path: Path | None = None) -> AppServices:
    db = Database(db_path)
    db.migrate()
    config = ConfigStore().load()
    calibrations = CalibrationRepository(db)
    strategies = StrategyRepository(db)
    contacts = ContactRepository(db)
    customer_profiles = CustomerProfileRepository(db)
    historical_archive = HistoricalArchiveRepository(db)
    historical_import = HistoricalImportService(historical_archive)
    memory_jobs = MemoryExtractionJobRepository(db)
    product_knowledge = ProductKnowledgeRepository(db)
    knowledge_import = KnowledgeImportService(product_knowledge)
    product_knowledge_context = ProductKnowledgeContextBuilder(
        product_knowledge, allowed_kinds=GENERIC_KNOWLEDGE_KINDS
    )
    knowledge_gaps = KnowledgeGapRepository(db)
    reply_pipeline_metrics = ReplyPipelineMetricRepository(db)
    knowledge_gap_analyzer = KnowledgeGapAnalyzer(knowledge_gaps)
    knowledge_coverage_report = KnowledgeCoverageReportService(
        knowledge_gaps, product_knowledge, reply_pipeline_metrics
    )
    identity_bindings = IdentityBindingRepository(db)
    account_scopes = AccountScopeRepository(db)
    identities = IdentityRepository(db)
    conversations = ConversationRepository(db)
    analysis_events = AIAnalysisEventRepository(db)
    analysis_events.recover_interrupted()
    messages = MessageRepository(db)
    observations = ObservationRepository(db)
    memories = MemoryRepository(db)
    logs = LogRepository(db)
    sales_followup = SalesFollowupService(customer_profiles, logs=logs)
    customer_tracking_advisor = CustomerTrackingAdvisor()
    customer_profile_service = CustomerProfileService(
        customer_profiles,
        account_scopes,
        contacts,
        logs=logs,
        sales_followup=sales_followup,
        tracking_advisor=customer_tracking_advisor,
    )
    log_cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, config.privacy.diagnostic_log_retention_days))
    logs.enforce_retention(log_cutoff.isoformat(), max_rows=20000)
    generation_logs = GenerationLogRepository(db)
    reply_feedback = ReplyFeedbackRepository(db)
    capture_samples = CaptureSampleRepository(db)
    settings_audit = SettingsAuditRepository(db)
    governance = DataGovernanceService(contacts, messages, memories, generation_logs, reply_feedback, logs, db)
    retention = RetentionCleanupService(logs, reply_feedback=reply_feedback)
    environment = EnvironmentDiagnosticsService()
    runtime = RuntimeStateService(calibrations=calibrations, logs=logs)
    current_identity = CurrentConversationIdentityController(contacts, logs=logs)
    runtime.state_changed.connect(current_identity.observe_runtime_state)
    from whochat.services.native_account_identity import NativeAccountIdentity
    from whochat.services.native_open_files import NativeFileReader
    account_scope_resolver = AccountScopeResolver(
        account_scopes, native_identity=NativeAccountIdentity(
            account_scopes.native_account_mappings, file_reader=NativeFileReader()))
    runtime.bind_account_identity_authority(account_scope_resolver)
    runtime.capture_gate.policy = replace(
        runtime.capture_gate.policy,
        scroll_debounce_ms=config.capture.scroll_debounce_ms,
        ocr_min_interval_ms=config.capture.ocr_min_interval_ms,
    )
    pipeline = CapturePipelineService(
        ocr_engine=create_ocr_engine(config.ocr),
        capture_samples=capture_samples,
        retain_capture_images=config.privacy.save_debug_screenshots,
        capture_retention_days=config.privacy.capture_retention_days,
        account_scope_resolver=account_scope_resolver,
        current_state_provider=lambda: runtime.state,
        ocr_engine_factory=lambda: create_ocr_engine(ConfigStore().load().ocr),
        capture_gate=runtime.capture_gate,
    )
    transcript_stitcher = TranscriptStitcher()
    identity_lock = StableIdentityLock(contacts, identity_bindings)
    timeline = PersistentConversationTimeline(conversations, messages, observations)
    # Structured customer memory is intentionally not part of the ingestion
    # transaction. Formal messages are discovered through a durable v20 cursor
    # and processed by an independent worker, so provider failures cannot delay
    # message ingestion or the existing reply-analysis flow.
    structured_memory = StructuredMemoryExtractionService(
        memory_jobs,
        customer_profiles,
        contacts,
        config_provider=ConfigStore().load,
        logs=logs,
        local_postprocessor=sales_followup.on_memory_written,
    )
    historical_backfill = HistoricalMemoryBackfillService(
        historical_archive,
        customer_profiles,
        contacts,
        structured_memory,
        provider=DisabledHistoricalBackfillProvider(),
        logs=logs,
        postprocessor=sales_followup.on_memory_written,
    )
    reply_memory_context = CustomerMemoryContextBuilder(db, logs=logs)
    # Separate, local-only history. Constructing this service performs no
    # I/O, import/backfill, extraction request or provider call.
    from whochat.core.paths import app_data_dir
    from whochat.services.observed_history import ObservedHistoryService
    observed_history = ObservedHistoryService(
        app_data_dir() / "state" / "observed_history.sqlite3",
        verify_view=reply_memory_context.validate_current_view,
        verify_source=reply_memory_context.validate_historical_source,
    )
    reply_memory_context.observed_history = observed_history
    reply_memory_context.history_enabled = lambda: ConfigStore().load().privacy.enable_long_term_memory
    governance.observed_history = observed_history
    ingestion = ChatIngestionService(
        contacts,
        messages,
        logs,
        transcript_stitcher,
        None,
        identity_lock,
        conversations,
        timeline,
        analysis_events,
        current_state_provider=lambda: runtime.state,
    )
    ingestion.on_identity_unstable = pipeline.forget_snapshot_hash
    autocapture = AutoCaptureController(
        runtime, pipeline,
        enabled=bool(config.capture.auto_capture_enabled and not MANUAL_SUGGESTIONS_ONLY),
        interval_ms=5000,
    )
    identity_recovery = ContactIdentityRecoveryController(
        runtime,
        pipeline,
        ingestion,
        autocapture,
        current_identity=current_identity,
        logs=logs,
    )
    from whochat.services.frozen_screenshot import SnapshotAuthority
    snapshot_authority = SnapshotAuthority(reply_memory_context)
    reply_generator = ReplyGenerationService(
        generation_logs,
        result_cache=ReplyResultCache(db.path.parent / "cache" / "reply_results.sqlite3"),
        identity_lease_validator=lambda context: snapshot_authority.valid(context)
        if snapshot_authority.recognizes(context) else _reply_identity_lease_valid(
            current_identity,
            runtime,
            context,
            ingestion=ingestion,
            current_view_validator=reply_memory_context.validate_current_view,
        ) and (context.current_message_view is None or reply_memory_context.validate_history_context(
            context.memory_context, context.current_message_view)),
        display_lease_validator=lambda context: snapshot_authority.valid(context, display=True)
        if snapshot_authority.recognizes(context) else _reply_display_lease_valid(
            current_identity, runtime, context, ingestion=ingestion,
            current_view_validator=reply_memory_context.validate_current_view,
            current_memory_builder=reply_memory_context.build_for_current_view,
        ),
    )
    reply_generator.snapshot_authority = snapshot_authority
    reply_tasks = ReplyTaskService(reply_generator, task_timeout_seconds=30.0)
    suggestion_feedback = SuggestionFeedbackRepository(db)
    sales_writing_preferences = SalesWritingPreferenceService(suggestion_feedback)
    proactive_feedback = ProactiveSuggestionFeedbackService(
        suggestion_feedback,
        sales_writing_preferences,
    )
    conversation_styles = CustomerSalutationResolver(db)
    proactive_contact = ProactiveContactService(
        ProactiveContactContextBuilder(
            contacts,
            messages,
            memories,
            customer_profile_service,
            strategies,
            observations,
            conversation_styles,
        ),
        ProactiveContactCacheRepository(db),
        generation_logs,
        logs,
        feedback=proactive_feedback,
        writing_preferences=sales_writing_preferences,
        current_runtime_provider=lambda: runtime.state,
        identity_lease_validator=lambda request: _proactive_identity_lease_valid(
            current_identity,
            runtime,
            request,
        ),
    )
    proactive_tasks = ProactiveContactTaskService(proactive_contact)
    compact_activation_index = CompactActivationIndex(db)
    fast_customer_resolver = FastCustomerResolver(compact_activation_index)
    activation_snapshots = ActivationSnapshotService(
        db,
        contacts,
        proactive_contact,
        ConfigStore().load,
        l1_capacity=config.proactive.activation_l1_capacity,
    )
    pipeline.fast_title_lookup = compact_activation_index.lookup_fingerprint
    fast_activation_probe = FastActivationProbeService(
        current_state_provider=lambda: runtime.state,
        last_title_provider=lambda: pipeline.last_title_result,
        fast_title_lookup=compact_activation_index.lookup_fingerprint,
        same_contact_title_validator=lambda result: _fast_probe_matches_retained_contact(
            current_identity,
            fast_customer_resolver,
            result,
        ),
        trusted_dm_anchor_validator=lambda result, hint: _fast_probe_trusted_dm_anchor(
            compact_activation_index,
            fast_customer_resolver,
            result,
            hint,
        ),
        ocr_engine_factory=lambda: create_ocr_engine(ConfigStore().load().ocr),
        account_identity_guard=account_scope_resolver.guard_fingerprint,
    )
    activation_prewarm = ActivationPrewarmScheduler(
        activation_snapshots,
        ConfigStore().load,
        busy_provider=lambda: bool(
            reply_tasks.is_running or proactive_tasks.is_running or pipeline.is_running
        ),
    )
    pipeline.status_changed.connect(
        lambda message: runtime.apply_pipeline_started(_pipeline_started_job_id(message))
        if message.startswith("pipeline_started:")
        else None
    )
    identity_events = IdentityEventCoordinator(
        runtime, ingestion, current_identity, fast_activation_probe, identity_recovery,
        submit_identity=_submit_ingestion_identity,
        submit_context=_submit_context_identity_evidence,
        logs=logs,
    )
    pipeline.context_changed.connect(identity_events.on_context_changed)
    pipeline.title_ready.connect(identity_events.on_title_ready)
    pipeline.result_ready.connect(identity_events.on_result_ready)
    pipeline.result_discarded_detail.connect(runtime.apply_pipeline_discarded)
    # FastProbe's changed-pixel event is deliberately weak evidence: it may
    # suspend the current live identity while the title is reacquired, but it
    # must not erase StableIdentityLock's first positive title observation.
    # Repeated ROI repaint events used to reset the lock between ordinary
    # pipeline jobs and could strand a real contact at identity_pending:1/2.
    # A real title/window change still resets naturally through the candidate
    # key/window boundary in StableIdentityLock and RuntimeStateService.
    fast_activation_probe.context_changed.connect(identity_events.on_fast_context_changed)
    fast_activation_probe.title_ready.connect(identity_events.on_fast_title_ready)
    services = AppServices(
        db=db,
        strategies=strategies,
        contacts=contacts,
        customer_profiles=customer_profiles,
        customer_profile_service=customer_profile_service,
        customer_tracking_advisor=customer_tracking_advisor,
        sales_followup=sales_followup,
        reply_memory_context=reply_memory_context,
        writing_feedback=WritingFeedbackService(app_data_dir() / "state" / "reply_feedback.sqlite3"),
        identity_bindings=identity_bindings,
        account_scopes=account_scopes,
        identities=identities,
        conversations=conversations,
        analysis_events=analysis_events,
        messages=messages,
        observations=observations,
        memories=memories,
        logs=logs,
        generation_logs=generation_logs,
        reply_feedback=reply_feedback,
        capture_samples=capture_samples,
        settings_audit=settings_audit,
        runtime=runtime,
        calibrations=calibrations,
        pipeline=pipeline,
        reply_generator=reply_generator,
        reply_tasks=reply_tasks,
        proactive_contact=proactive_contact,
        proactive_tasks=proactive_tasks,
        proactive_feedback=proactive_feedback,
        sales_writing_preferences=sales_writing_preferences,
        conversation_styles=conversation_styles,
        compact_activation_index=compact_activation_index,
        fast_customer_resolver=fast_customer_resolver,
        activation_snapshots=activation_snapshots,
        activation_prewarm=activation_prewarm,
        fast_activation_probe=fast_activation_probe,
        ingestion=ingestion,
        memory_jobs=memory_jobs,
        structured_memory=structured_memory,
        historical_archive=historical_archive,
        historical_import=historical_import,
        historical_backfill=historical_backfill,
        product_knowledge=product_knowledge,
        knowledge_import=knowledge_import,
        product_knowledge_context=product_knowledge_context,
        knowledge_gaps=knowledge_gaps,
        knowledge_gap_analyzer=knowledge_gap_analyzer,
        knowledge_coverage_report=knowledge_coverage_report,
        reply_pipeline_metrics=reply_pipeline_metrics,
        autocapture=autocapture,
        identity_recovery=identity_recovery,
        governance=governance,
        retention=retention,
        environment=environment,
        transcript_stitcher=transcript_stitcher,
        identity_lock=identity_lock,
        current_identity=current_identity,
        account_scope_resolver=account_scope_resolver,
        identity_events=identity_events,
        timeline=timeline,
    )
    services.strategies.ensure_defaults()
    try:
        services.retention.cleanup(config.privacy)
    except Exception:
        # Retention is best-effort at startup; capture-level cleanup remains in
        # the pipeline and can retry on the next successful frame.
        pass
    services.structured_memory.start()
    if not MANUAL_SUGGESTIONS_ONLY:
        services.activation_prewarm.start()
        services.fast_activation_probe.start()
    services.logs.append("info", "bootstrap", "services_ready", "Application services initialized")
    return services


def _submit_ingestion_identity(controller, ingestion_result, evidence, *, producer: str) -> None:
    if ingestion_result is None:
        controller.observe_unavailable(evidence, producer=producer, reason="ingestion_result_unavailable")
        return
    if ingestion_result.reason in {"title_group_requires_full_page", "identity_contact_type_conflict"}:
        # Positive non-DM header evidence must suspend an old DM even when a
        # weak pixel-change notification was absent or arrived later.
        controller.observe_possible_switch(evidence, producer=producer, reason=ingestion_result.reason)
        return
    if ingestion_result.identity_stable and ingestion_result.contact is not None:
        controller.promote(
            ingestion_result.contact,
            evidence,
            normalized_title=ingestion_result.contact.display_name,
            producer=producer,
            reason=ingestion_result.reason,
        )
        return
    if ingestion_result.reason.startswith("identity_pending:") or ingestion_result.reason == "identity_waiting_page_type":
        controller.observe_pending(evidence, producer=producer, reason=ingestion_result.reason)
        return
    controller.observe_unavailable(evidence, producer=producer, reason=ingestion_result.reason)


def _submit_context_identity_evidence(controller, event, *, producer: str, runtime=None) -> None:
    """Route ambiguous title changes through the single identity authority.

    Pixel changes are positive evidence that the title surface may have
    changed, but are not positive evidence of a different customer.  The
    controller retains the last contact and suspends actions until a unique
    resolver result promotes or switches it.
    """

    if str(event.reason).startswith("account_identity_revalidation:"):
        if runtime is not None and not runtime.account_boundary_event_was_applied(event):
            return
        evidence = runtime.state if runtime is not None else event
        controller.invalidate(evidence, producer=producer, reason=event.reason, strong=True)
        return
    if event.reason in {"title_pixels_changed", "fast_title_probe_changed_stable"}:
        controller.observe_possible_switch(event, producer=producer, reason=event.reason)
        return
    controller.invalidate(event, producer=producer, reason=event.reason, strong=False)


def _reset_ingestion_for_pipeline_context(ingestion, event, *, runtime=None) -> None:
    if runtime is not None:
        if not runtime.capture_context_event_was_applied(event):
            return
    if event.reason in {"title_pixels_changed", "fast_title_probe_changed_stable"}:
        return
    ingestion.reset_context(f"capture_context_changed:{event.reason}")


def _reset_ingestion_for_account_boundary(ingestion, event, *, runtime=None) -> None:
    if str(event.reason).startswith("account_identity_revalidation:"):
        if runtime is not None and not runtime.account_boundary_event_was_applied(event):
            return
        ingestion.reset_context(f"capture_context_changed:{event.reason}")


def _fast_probe_matches_retained_contact(controller, resolver, result) -> bool:
    """Return true only when variant title evidence uniquely retains Contact."""

    snapshot = controller.snapshot
    if not snapshot.retained or not snapshot.contact_id:
        return False
    if (
        snapshot.account_scope_id != (result.account_scope_id or "")
        or snapshot.target_instance_id != result.target_instance_id
    ):
        return False
    candidates = extract_title_candidates(result.ocr_result, result, min_confidence=0.50)
    raw_title = candidates[0] if candidates else ""
    title = normalize_contact_title(raw_title, ConversationType.DM) if raw_title else None
    if not title:
        return False
    resolution = resolver.resolve(
        account_scope_id=result.account_scope_id or "",
        platform=result.target_app,
        target_instance_id=result.target_instance_id,
        normalized_title=title,
        # The callback runs only after FastProbe observed two identical title
        # frames. Raw OCR confidence must not defeat a unique account-scoped
        # match back to the already-retained Contact; title uniqueness and the
        # original high-confidence binding remain mandatory in the resolver.
        title_confidence=max(0.95, _title_confidence(result.ocr_result, result, title)),
        retained_contact_id=snapshot.contact_id,
    )
    return bool(resolution.resolved and resolution.contact_id == snapshot.contact_id)


def _fast_probe_trusted_dm_anchor(index, resolver, result, hint) -> bool:
    """Validate exact UNKNOWN-page evidence against the confirmed DM index.

    Fingerprints in ``CompactActivationIndex`` are learned only after a stable
    DM identity.  This additional check keeps UNKNOWN-page admission scoped to
    the same account/target and requires the title to remain a unique exact DM
    binding. Similar/visual fingerprints and title variants are not accepted.
    """

    contact_id = str(getattr(hint, "contact_id", "") or "")
    contact = index.contact(contact_id) if contact_id else None
    if (
        contact is None
        or contact.conversation_type != ConversationType.DM
        or contact.account_scope_id != (result.account_scope_id or "")
    ):
        return False
    candidates = extract_title_candidates(result.ocr_result, result, min_confidence=0.50)
    raw_title = candidates[0] if candidates else ""
    title = normalize_contact_title(raw_title, ConversationType.DM) if raw_title else None
    hint_title = normalize_contact_title(
        str(getattr(hint, "normalized_title", "") or ""),
        ConversationType.DM,
    )
    if not title or not hint_title or title.casefold() != hint_title.casefold():
        return False
    resolution = resolver.resolve(
        account_scope_id=result.account_scope_id or "",
        platform=result.target_app,
        target_instance_id=result.target_instance_id,
        normalized_title=title,
        title_confidence=max(0.95, float(getattr(hint, "confidence", 0.0) or 0.0)),
    )
    return bool(
        resolution.resolved
        and resolution.contact_id == contact_id
        and resolution.reason == "unique_exact_account_title"
    )


def _proactive_identity_lease_valid(controller, runtime, request) -> bool:
    snapshot = controller.snapshot
    state = runtime.state
    return bool(
        snapshot.active
        and snapshot.contact_id == request.contact_id
        and snapshot.account_scope_id == request.account_scope_id
        and snapshot.target_instance_id == request.target_instance_id
        and snapshot.hwnd == request.hwnd
        and state.window.hwnd == request.hwnd
        and state.capture_generation == request.capture_generation
        and _current_identity_lifecycle_valid(snapshot, runtime)
    )


def _reply_identity_lease_valid(
    controller, runtime, context, *, ingestion=None, current_view_validator=None,
) -> bool:
    snapshot = controller.snapshot
    state = runtime.state
    contact = getattr(context, "contact", None)
    submitted = getattr(context, "runtime", None)
    if contact is None or submitted is None:
        return False
    valid = bool(
        snapshot.active
        and snapshot.contact_id == contact.id
        and snapshot.account_scope_id == contact.account_scope_id
        and snapshot.hwnd == submitted.window.hwnd
        and state.window.hwnd == submitted.window.hwnd
        and state.window.process_id == submitted.window.process_id
        and state.window_generation == submitted.window_generation
        and state.account_identity_runtime_key == submitted.account_identity_runtime_key
        and state.capture_generation == submitted.capture_generation
        and state.account_identity_epoch == submitted.account_identity_epoch
        and _current_identity_lifecycle_valid(snapshot, runtime)
    )
    if not valid or current_context_source_error(context) is not None:
        return False
    view = getattr(context, "current_message_view", None)
    if view is None:
        return True
    if not view.current_screen_runtime_matches(state):
        return False
    observed = getattr(ingestion, "last_result", None)
    evidence = getattr(ingestion, "last_evidence", None)
    if (
        snapshot.target_instance_id != view.target_instance_id
        or observed is None or not observed.accepted or not observed.identity_stable
        or observed.contact is None or observed.contact.id != contact.id
        or observed.account_scope_id != view.account_scope_id
        or observed.target_instance_id != view.target_instance_id
        or observed.conversation_id != view.conversation_id
        or observed.session_id != view.session_id
        or observed.observation_id != view.observation_id
        or evidence is None or not callable(current_view_validator)
    ):
        return False
    if not view.validate_binding(
        expected_account_scope_id=snapshot.account_scope_id, expected_contact_id=snapshot.contact_id,
        expected_conversation_id=observed.conversation_id,
        expected_target_instance_id=snapshot.target_instance_id,
        expected_session_id=observed.session_id, max_source_job_id=evidence.job_id,
        max_capture_generation=state.capture_generation, now=datetime.now(timezone.utc),
    ):
        return False
    try:
        return bool(current_view_validator(view))
    except Exception:
        return False


def _reply_display_lease_valid(
    controller, runtime, context, *, ingestion, current_view_validator, current_memory_builder,
) -> bool:
    """Existing rendered result only: focus is not a new capture/cloud grant.

    Keep the submitted proof unchanged. Reject resize/minimization, every
    identity/account checkpoint boundary, pending title/body negative evidence,
    ended session, DB source replacement/expiry, and changed ingestion owner.
    The UI additionally rebuilds memory/knowledge and compares its business hash.
    """
    snapshot, state, source = controller.snapshot, runtime.state, context.runtime
    contact, view = context.contact, context.current_message_view
    if (contact is None or view is None or current_context_source_error(context)
            or not snapshot.active or snapshot.contact_id != contact.id
            or snapshot.account_scope_id != contact.account_scope_id
            or snapshot.target_instance_id != view.target_instance_id
            or state.window.state != source.window.state or state.window.state.value != "visible"
            or state.window.rect is None or source.window.rect is None
            or (state.window.rect.width, state.window.rect.height)
               != (source.window.rect.width, source.window.rect.height)
            or state.page.page_type.value != "chat_dm" or state.page.confidence < .65
            or state.window.hwnd != source.window.hwnd or state.window.process_id != source.window.process_id
            or snapshot.hwnd != source.window.hwnd or snapshot.process_id != source.window.process_id
            or state.window_generation != source.window_generation
            or state.account_identity_epoch != source.account_identity_epoch
            or state.account_identity_runtime_key != source.account_identity_runtime_key
            or state.capture_generation < source.capture_generation
            or not _current_identity_lifecycle_valid(snapshot, runtime)):
        return False
    observed = getattr(ingestion, "last_result", None)
    if (observed is None or not observed.accepted or not observed.identity_stable
            or observed.contact is None or observed.contact.id != contact.id
            or observed.account_scope_id != view.account_scope_id
            or observed.target_instance_id != view.target_instance_id
            or observed.conversation_id != view.conversation_id or observed.session_id != view.session_id
            or observed.observation_id != view.observation_id):
        return False
    try:
        if not callable(current_view_validator) or not current_view_validator(view) or not callable(current_memory_builder):
            return False
        fresh_memory = current_memory_builder(view)
        return bool(context.memory_context is not None and fresh_memory.current_view_verified
                    and fresh_memory.business_payload() == context.memory_context.business_payload())
    except Exception:
        return False


def _current_identity_lifecycle_valid(snapshot, runtime) -> bool:
    """Check the live account checkpoint even before its queued event arrives.

    A retained customer is not authorization if AccountScope already froze
    its lifecycle on another thread. Ordinary focus capture generations are
    intentionally taken from Runtime, not from the customer identity token.
    """
    state = runtime.state
    if (
        snapshot.hwnd != state.window.hwnd
        or snapshot.process_id != state.window.process_id
        or snapshot.window_generation != state.window_generation
        or snapshot.account_identity_epoch != state.account_identity_epoch
    ):
        return False
    validator = getattr(runtime, "evidence_is_current", None)
    if validator is None:
        return True  # Compatibility for minimal offline Runtime fixtures.
    return validator(SimpleNamespace(
        hwnd=snapshot.hwnd,
        process_id=snapshot.process_id,
        window_generation=snapshot.window_generation,
        capture_generation=state.capture_generation,
        account_identity_epoch=snapshot.account_identity_epoch,
        account_identity_runtime_key=state.account_identity_runtime_key,
        account_scope_id=snapshot.account_scope_id,
        target_instance_id=snapshot.target_instance_id,
    ))


def _pipeline_started_job_id(message: str) -> int | None:
    try:
        return int(message.partition("job=")[2].split(",", 1)[0].strip())
    except (TypeError, ValueError):
        return None
