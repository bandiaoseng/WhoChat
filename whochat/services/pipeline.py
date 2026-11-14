from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock, RLock
import time
import unicodedata

from PIL import Image, ImageChops
from PySide6.QtCore import QObject, Signal

from whochat.capture.policy import CaptureGate, image_hash, image_hash_region
from whochat.capture.screenshot import capture_rect
from whochat.core.models import utc_now_iso
from whochat.core.current_conversation import (
    SCREEN_CAPTURE_SCHEMA, SCREEN_TTL_SECONDS, screen_message_payload,
    screen_rows_digest, screen_business_rows_digest, screen_top_partial_prefix, _validate_capture_fields,
)
from whochat.core.paths import app_data_dir
from whochat.core.runtime import LayoutRegions, PageClassification, PageType, Rect, RuntimeState, constrain_layout_to_client
from whochat.diagnostics import append_diagnostics_log
from whochat.ocr.engine import OcrEngine, PreviewOcrEngine
from whochat.ocr.models import OcrRegion, OcrResult, OcrTextBox, ParsedOcrMessage
from whochat.ocr.parser import account_identity_boundary_reason, classify_page_from_ocr, parse_visible_messages, parse_visible_messages_with_evidence
from whochat.storage.repositories import CaptureSampleRepository
from whochat.services.account_scope import (
    AccountImageEvidenceUnavailable, AccountScopeResolver,
    _visual_fingerprint, account_avatar_screen_rect,
)
from whochat.services.frame_diagnostics import build_signature
from whochat.vision.layout_refine import refine_layout_from_image
from whochat.ai.photos import PhotoCrop, collect_photo_crops


@dataclass(frozen=True)
class PipelineResult:
    job_id: int
    hwnd: int | None
    target_app: str
    app_label: str
    snapshot_hash: str
    image_path: Path
    ocr_image_path: Path
    crop_rect: Rect | None
    layout: LayoutRegions
    ocr_result: OcrResult
    page: PageClassification
    messages: list[ParsedOcrMessage]
    created_at: str
    title_ocr_image_path: Path | None = None
    title_crop_rect: Rect | None = None
    title_ocr_elapsed_ms: int | None = None
    content_ocr_elapsed_ms: int | None = None
    total_elapsed_ms: int | None = None
    capture_elapsed_ms: int | None = None
    verify_capture_elapsed_ms: int | None = None
    page_classification_elapsed_ms: int | None = None
    layout_elapsed_ms: int | None = None
    title_input_elapsed_ms: int | None = None
    content_input_elapsed_ms: int | None = None
    title_preprocess_elapsed_ms: int | None = None
    title_inference_elapsed_ms: int | None = None
    title_postprocess_elapsed_ms: int | None = None
    content_preprocess_elapsed_ms: int | None = None
    content_inference_elapsed_ms: int | None = None
    content_postprocess_elapsed_ms: int | None = None
    title_cache_hit: bool = False
    performance_trace: dict[str, object] = field(default_factory=dict)
    process_id: int | None = None
    window_generation: int = 0
    capture_generation: int = 0
    account_scope_id: str | None = None
    account_scope_name: str = ""
    target_instance_id: str = ""
    account_scope_reason: str = ""
    account_identity_epoch: int = 0
    account_identity_runtime_key: str = ""
    title_visual_hash: str = ""
    title_exact_hash: str = ""
    screen_proof: dict[str, object] | None = None
    photo_crops: tuple[PhotoCrop, ...] = field(default=(), repr=False)
    # Ephemeral diagnostics only: never part of source proof, ingestion or authorization.
    frame_diagnostic: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class TitleOcrResult:
    job_id: int
    hwnd: int | None
    target_app: str
    app_label: str
    snapshot_hash: str
    image_path: Path
    title_ocr_image_path: Path
    title_crop_rect: Rect | None
    layout: LayoutRegions
    ocr_result: OcrResult
    created_at: str
    elapsed_ms: int | None = None
    process_id: int | None = None
    window_generation: int = 0
    capture_generation: int = 0
    account_scope_id: str | None = None
    account_scope_name: str = ""
    target_instance_id: str = ""
    account_scope_reason: str = ""
    account_identity_epoch: int = 0
    account_identity_runtime_key: str = ""
    title_visual_hash: str = ""
    title_exact_hash: str = ""
    title_source: str = "roi_ocr"
    performance_trace: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureContextChange:
    job_id: int
    hwnd: int | None
    process_id: int | None
    window_generation: int
    capture_generation: int
    snapshot_hash: str
    reason: str
    account_scope_id: str | None = None
    target_instance_id: str = ""
    account_identity_epoch: int = 0
    account_identity_runtime_key: str = ""


@dataclass(frozen=True)
class PipelineDiscardEvent:
    reason: str
    job_id: int
    hwnd: int | None
    process_id: int | None
    window_generation: int
    capture_generation: int
    snapshot_hash: str = ""
    account_scope_id: str | None = None
    target_instance_id: str = ""
    account_identity_epoch: int = 0
    account_identity_runtime_key: str = ""


def _same_frame_account_avatar_anchor(image_path, layout, account_context):
    """Freeze this trusted capture's avatar, never a later database sample.

    The resolver is the only account authority. This helper merely verifies
    that its reliable current fingerprint really came from the same original
    image/layout and binds those pixels to the already-authorized capture.
    A persisted instance can be refreshed by another observation, so it is not
    the immutable pixel anchor for a completed request.
    """
    try:
        if (account_context is None or account_context.actions_allowed is not True
                or account_context.observed_visual_reliable is not True):
            return None
        fingerprint, reliable = _visual_fingerprint(image_path, layout)
        geometry = account_avatar_screen_rect(layout)
        if (reliable is not True or not fingerprint.startswith("v2:")
                or len(fingerprint) != 67
                or any(char not in "0123456789abcdef" for char in fingerprint[3:])
                or fingerprint != account_context.observed_visual_hash):
            return None
        return {
            "account_avatar_fingerprint": fingerprint,
            "account_avatar_geometry": list(geometry.as_tuple()),
            "account_avatar_sampling_rule": "fixed_grid_96dpi_18_46_40_v1",
            "account_avatar_dpi_scale": layout.dpi_scale,
            "account_avatar_reliable": True,
        }
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _build_current_screen_capture_proof(
    *, job_id, state, account_context, snapshot_hash, created_at, layout,
    ocr_result, page, messages, parser_report, title_exact_hash,
    capture_verified=False, title_verified=False, manual_request_token=None,
    account_avatar_anchor=None, title_only_rejection=False, manual_activation=False,
):
    """Issue only from verified full OCR; uncertainty is a conservative miss."""
    from whochat.services.window_reactivation_capture import build_window_reactivation_capture_proof

    def invalid(code):
        return {"schema": SCREEN_CAPTURE_SCHEMA, "valid": False, "rejection_code": code,
                "source_job_id": job_id, "snapshot_hash": snapshot_hash}

    selection_id = getattr(layout, "manual_selection_id", "")
    if selection_id:
        owner = (state.window.hwnd, state.window.process_id, state.window_generation,
                 getattr(account_context, "identity_epoch", None),
                 getattr(account_context, "runtime_key", None))
        if (not isinstance(manual_request_token, tuple) or len(manual_request_token) != 2
                or type(manual_request_token[0]) is not int or manual_request_token[0] <= 0
                or manual_request_token[1] != owner):
            return invalid("manual_selection_request_missing")

    # A distinct explicit operator task, never a fallback after reply failure.
    # It creates a zero-message window capability, retaining body/owner checks.
    if manual_activation is True and capture_verified is True:
        return build_window_reactivation_capture_proof(
            job_id=job_id,state=state,account_context=account_context,snapshot_hash=snapshot_hash,
            created_at=created_at,layout=layout,ocr_result=ocr_result,page=page,messages=messages,
            parser_report=parser_report,title_exact_hash=title_exact_hash,capture_verified=capture_verified,
            title_verified=title_verified,manual_request_token=manual_request_token,
            account_avatar_anchor=account_avatar_anchor,manual_activation=True)

    # A proven navigation control, unproven input boundary or unread voice is not a
    # source capability. Keep its *negative* evidence tied to this button and
    # frame so the UI can explain the remedy without spending the whole
    # recognition deadline. Never let the textless/reactivation fallback turn
    # an incomplete historical viewport into permission to generate.
    try:
        from whochat.ocr.low_confidence import low_confidence_messages
        low_text_count = len(low_confidence_messages(messages, layout, parser_report))
        local_read_attempts = parser_report.get("low_confidence_recovery_attempts", 0) if isinstance(parser_report, dict) else 0
        unread_voice_count = (parser_report.get("untranscribed_voice_count")
                              if isinstance(parser_report, dict) else None)
        blocker = ("history_navigation_present"
                   if isinstance(parser_report, dict)
                   and parser_report.get("history_navigation_present") is True
                   else "input_boundary_unverified"
                   if getattr(layout, "input_boundary_verified", None) is False
                   else "untranscribed_voice_present"
                   if (type(unread_voice_count) is int and unread_voice_count > 0
                       and getattr(layout, "input_boundary_verified", None) is True)
                   else "message_text_low_confidence" if low_text_count > 0
                   and type(local_read_attempts) is int and local_read_attempts in {0, 2} else "")
        if blocker:
            title_boundary_only = (title_only_rejection is True
                                   and blocker == "input_boundary_unverified"
                                   and capture_verified is False)
            owner = (state.window.hwnd, state.window.process_id, state.window_generation,
                     account_context.identity_epoch if account_context else None,
                     account_context.runtime_key if account_context else None)
            if ((capture_verified is not True and not title_boundary_only) or title_verified is not True
                    or account_context is None or account_context.actions_allowed is not True
                    or not account_context.account_scope_id or not account_context.target_instance_id
                    or state.window.target.value != "wechat"
                    or (not title_boundary_only and (page.page_type.value != "chat_dm" or page.confidence < .65))
                    or ocr_result.warning
                    or (title_boundary_only and not any(box.region == OcrRegion.TITLE
                        and box.text.strip() and box.confidence >= .65 for box in ocr_result.boxes))
                    or not isinstance(title_exact_hash, str) or not title_exact_hash
                    or not isinstance(manual_request_token, tuple) or len(manual_request_token) != 2
                    or type(manual_request_token[0]) is not int or manual_request_token[0] <= 0
                    or manual_request_token[1] != owner):
                return invalid(blocker)
            observed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if observed.tzinfo is None:
                return invalid(blocker)
            return {**invalid(blocker), "negative_capture_evidence": True,
                    "capture_verified": not title_boundary_only, "title_verified": True, "account_verified": True,
                    "rejection_scope": "title_verified_boundary" if title_boundary_only else "full_capture",
                    "manual_request_id": manual_request_token[0],
                    "title_exact_hash": title_exact_hash,
                    "account_scope_id": account_context.account_scope_id,
                    "target_instance_id": account_context.target_instance_id,
                    "hwnd": state.window.hwnd, "process_id": state.window.process_id,
                    "window_generation": state.window_generation,
                    "capture_generation": state.capture_generation,
                    "account_identity_epoch": account_context.identity_epoch,
                    "account_identity_runtime_key": account_context.runtime_key,
                    "input_boundary_verified": getattr(layout, "input_boundary_verified", False),
                    "history_navigation_present": (isinstance(parser_report, dict)
                                                   and parser_report.get("history_navigation_present") is True),
                    "history_navigation_evidence": (parser_report.get("history_navigation_evidence", {})
                                                    if isinstance(parser_report, dict) else {}),
                    **({"untranscribed_voice_count": unread_voice_count}
                       if blocker == "untranscribed_voice_present" else {}),
                    **({"low_confidence_count": low_text_count,
                        "low_confidence_recovery_attempts": local_read_attempts,
                        "parsed_coverage_complete": True, "unaccounted_box_count": 0,
                        "untranscribed_voice_count": 0}
                       if blocker == "message_text_low_confidence" else {}),
                    "observed_at": observed.isoformat(),
                    "expires_at": (observed + timedelta(seconds=SCREEN_TTL_SECONDS)).isoformat()}
    except (ValueError, TypeError, AttributeError, KeyError):
        return invalid("capture_proof_invalid")

    window_proof = build_window_reactivation_capture_proof(
        job_id=job_id, state=state, account_context=account_context,
        snapshot_hash=snapshot_hash, created_at=created_at, layout=layout,
        ocr_result=ocr_result, page=page, messages=messages, parser_report=parser_report,
        title_exact_hash=title_exact_hash, capture_verified=capture_verified,
        title_verified=title_verified, manual_request_token=manual_request_token,
        account_avatar_anchor=account_avatar_anchor,
    )
    if window_proof is not None:
        return window_proof

    try:
        if (capture_verified is not True or title_verified is not True
                or account_context is None or account_context.actions_allowed is not True
                or state.window.target.value != "wechat" or page.page_type.value != "chat_dm"
                or page.confidence < .65 or ocr_result.warning
                or getattr(layout, "input_boundary_verified", False) is not True
                or not isinstance(parser_report, dict)):
            return invalid("capture_binding_incomplete")
        prefix = screen_top_partial_prefix(messages, layout)
        if (prefix is None or any(parser_report.get(key) is not True for key in (
                "tail_complete", "input_boundary_verified", "parsed_coverage_complete"))
                or (parser_report.get("complete") is not True and prefix == 0)):
            return invalid("layout_or_coverage_incomplete")
        box_count, parsed_boxes, metadata_boxes = (parser_report.get(key) for key in (
            "message_box_count", "parsed_box_count", "ignored_metadata_box_count"))
        media_count = parser_report.get("image_occurrence_count", 0)
        media_only = (type(media_count) is int and media_count > 0 and len(messages) == media_count
                      and all(item.content_type == "image" and item.media_refs for item in messages))
        manual_owner = (state.window.hwnd, state.window.process_id, state.window_generation,
                        account_context.identity_epoch, account_context.runtime_key)
        media_only = bool(media_only and isinstance(manual_request_token, tuple)
                          and len(manual_request_token) == 2
                          and type(manual_request_token[0]) is int and manual_request_token[0] > 0
                          and manual_request_token[1] == manual_owner)
        if (any(type(value) is not int or value < 0 for value in (box_count, parsed_boxes, metadata_boxes))
                or ((box_count < 1 or parsed_boxes < 1) and not media_only)
                or parsed_boxes + metadata_boxes != box_count
                or parser_report.get("unaccounted_box_count") != 0):
            return invalid("unaccounted_message_boxes")
        raw = [screen_message_payload(item) for item in messages]
        rows = raw[prefix:]
        if (not rows or any(row["order_uncertain"] is not False or row["speaker"] not in {"me", "other"}
                for row in raw) or any(row["partial"] is not False
                or row["speaker"] not in {"me", "other"} or not row["text"].strip()
                or isinstance(row["confidence"], bool) or not isinstance(row["confidence"], (int, float))
                or not .65 <= row["confidence"] <= 1 for row in rows)):
            return invalid("middle_or_tail_incomplete")
        observed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        proof = {
            "schema": SCREEN_CAPTURE_SCHEMA, "valid": True, "capture_verified": True, "title_verified": True,
            "account_verified": True, "parser_complete": True, "tail_complete": True,
            "input_boundary_verified": True, "page_type": "chat_dm", "page_confidence": page.confidence,
            "raw_count": len(raw), "kept_count": len(rows), "omitted_count": prefix,
            "omitted_top_prefix_count": prefix, "partial_count": prefix, "kept_partial_count": 0,
            "order_uncertain_count": 0, "unaccounted_box_count": 0,
            "message_box_count": box_count, "parsed_box_count": parsed_boxes,
            "ignored_metadata_box_count": metadata_boxes,
            "min_confidence": min(row["confidence"] for row in rows),
            "rows_sha256": screen_rows_digest(rows), "business_rows_sha256": screen_business_rows_digest(rows),
            "raw_rows_sha256": screen_rows_digest(raw), "omitted_prefix_sha256": screen_rows_digest(raw[:prefix]),
            "source_job_id": job_id, "snapshot_hash": snapshot_hash,
            "account_scope_id": account_context.account_scope_id, "target_instance_id": account_context.target_instance_id,
            "hwnd": state.window.hwnd, "process_id": state.window.process_id,
            "window_generation": state.window_generation, "capture_generation": state.capture_generation,
            "account_identity_epoch": account_context.identity_epoch,
            "account_identity_runtime_key": account_context.runtime_key, "title_exact_hash": title_exact_hash,
            "observed_at": observed.isoformat(), "expires_at": (observed + timedelta(seconds=SCREEN_TTL_SECONDS)).isoformat(),
        }
        if isinstance(account_avatar_anchor, dict) and account_avatar_anchor.get("account_avatar_reliable") is True:
            fingerprint = account_avatar_anchor.get("account_avatar_fingerprint")
            if (isinstance(fingerprint, str) and len(fingerprint) == 67 and fingerprint.startswith("v2:")
                    and all(char in "0123456789abcdef" for char in fingerprint[3:])
                    and account_avatar_anchor.get("account_avatar_geometry") == list(account_avatar_screen_rect(layout).as_tuple())):
                proof.update({key: account_avatar_anchor[key] for key in (
                    "account_avatar_fingerprint", "account_avatar_geometry", "account_avatar_reliable",
                )})
                for key in ("account_avatar_sampling_rule", "account_avatar_dpi_scale"):
                    if key in account_avatar_anchor:
                        proof[key] = account_avatar_anchor[key]
        if selection_id:
            proof.update(capture_method="manual_selection", manual_selection_id=selection_id,
                manual_request_id=manual_request_token[0],
                selected_window_rect=list(layout.window_rect.as_tuple()),
                selected_title_rect=list(layout.title_rect.as_tuple()),
                selected_message_rect=list(layout.message_rect.as_tuple()))
        _validate_capture_fields(proof, datetime.now(timezone.utc))
        return proof
    except (ValueError, TypeError, AttributeError, KeyError):
        return invalid("capture_proof_invalid")


def _current_screen_geometry_metadata(job_id, layout, crop_rect, ocr_result, messages, report):
    """Bounded geometry and parser flags; no OCR text, sender names or paths."""
    def rect(value):
        return list(value.as_tuple()) if value is not None else None
    return {
        "job_id": job_id, "message_rect": rect(layout.message_rect),
        "client_rect": rect(getattr(layout, "client_rect", None)),
        "input_rect": rect(layout.input_rect), "crop_rect": rect(crop_rect),
        "input_boundary_verified": getattr(layout, "input_boundary_verified", False) is True,
        "history_navigation_present": report.get("history_navigation_present") is True,
        "parser_complete": report.get("complete") is True,
        "tail_complete": report.get("tail_complete") is True,
        "ignored_unsupported_notice_count": report.get("ignored_unsupported_notice_count", 0),
        "parsed_coverage_complete": report.get("parsed_coverage_complete") is True,
        "untranscribed_voice_count": (report.get("untranscribed_voice_count")
            if type(report.get("untranscribed_voice_count")) is int
            and 0 <= report["untranscribed_voice_count"] <= len(ocr_result.boxes) else None),
        "ocr_warning_present": bool(ocr_result.warning),
        "box_count": len(ocr_result.boxes), "parsed_count": len(messages),
        "unaccounted_box_count": report.get("unaccounted_box_count"),
        "boxes": [{"region": box.region.value, "rect": rect(box.rect), "confidence": float(box.confidence)}
                  for box in ocr_result.boxes[:64]],
        "messages": [{"speaker": item.speaker.value, "rect": rect(item.rect),
                      "confidence": float(item.confidence), "partial": item.partial,
                      "reason_code": "partial" if item.partial else "parsed_" + item.speaker.value}
                     for item in messages[:32]],
    }


class CapturePipelineService(QObject):
    context_changed = Signal(object)
    title_ready = Signal(object)
    result_ready = Signal(object)
    capture_rejected = Signal(object)
    result_discarded = Signal(str)
    result_discarded_detail = Signal(object)
    status_changed = Signal(str)

    def __init__(
        self,
        ocr_engine: OcrEngine | None = None,
        capture_func=None,
        executor: ThreadPoolExecutor | None = None,
        capture_samples: CaptureSampleRepository | None = None,
        retain_capture_images: bool = False,
        capture_retention_days: int = 7,
        account_scope_resolver: AccountScopeResolver | None = None,
        current_state_provider=None,
        ocr_engine_factory: Callable[[], OcrEngine] | None = None,
        fast_title_lookup: Callable[[str, str, str, str], object | None] | None = None,
        parent: QObject | None = None,
        capture_gate: CaptureGate | None = None,
    ) -> None:
        super().__init__(parent)
        self.ocr_engine = ocr_engine or PreviewOcrEngine()
        self.capture_func = capture_func or capture_rect
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="whochat-capture")
        self.capture_samples = capture_samples
        self.retain_capture_images = retain_capture_images
        self.capture_retention_days = max(1, int(capture_retention_days))
        self.account_scope_resolver = account_scope_resolver
        self.current_state_provider = current_state_provider
        self.capture_gate = capture_gate
        self._ocr_engine_factory = ocr_engine_factory
        self.fast_title_lookup = fast_title_lookup
        self._last_capture_cleanup_monotonic = 0.0
        self._last_duplicate_log_monotonic = 0.0
        self._job_id = 0
        self._latest_job_id = 0
        self._running: Future | None = None
        self._running_job_id: int | None = None
        self._running_started_monotonic = 0.0
        self._running_stage = "idle"
        self._expired_job_ids: set[int] = set()
        self._retired_engines: dict[int, OcrEngine] = {}
        self._lifecycle_lock = RLock()
        self._last_snapshot_hash: str | None = None
        self._snapshot_hash_cache: OrderedDict[str, None] = OrderedDict()
        self._screen_proof_cache: OrderedDict[str, tuple[float | None, int]] = OrderedDict()
        self._snapshot_hash_owners: dict[str, int] = {}
        self._job_snapshot_hashes: dict[int, str] = {}
        self._snapshot_hash_cache_lock = Lock()
        self._snapshot_hash_cache_size = _snapshot_hash_cache_size()
        self._title_cache: OrderedDict[tuple[object, ...], OcrResult] = OrderedDict()
        self._title_cache_size = 64
        # Raw OCR only. A second distinct capture still proves its own live
        # account/title/body and receives a new job/time/proof. Never store a
        # Contact, identity resolution, cloud grant or parsed business result.
        self._verified_content_ocr_cache: OrderedDict[tuple, tuple[float, OcrResult]] = OrderedDict()
        self._last_verified_title_context: tuple[object, ...] | None = None
        self._submitted_states: dict[int, RuntimeState] = {}
        # UI manual mode binds every capture to one explicit button request.
        # None preserves standalone service/test usage; production injects it
        # before any capture scheduler is started.
        self.capture_request_token_provider = None
        self._capture_request_tokens: dict[int, object] = {}
        self.manual_selection_provider = None
        self._manual_selections: dict[int, object] = {}
        self.activation_purpose_provider = None
        self._activation_request_tokens: dict[int, object] = {}
        self._job_account_owners: dict[int, tuple[str, str]] = {}
        self.last_result: PipelineResult | None = None
        self.last_title_result: TitleOcrResult | None = None
        self.last_discard_reason: str | None = None
        self.last_status: str = "idle"

    @property
    def is_running(self) -> bool:
        return self._running is not None and not self._running.done()

    @property
    def running_job_id(self) -> int | None:
        return self._running_job_id if self.is_running else None

    @property
    def running_stage(self) -> str:
        return self._running_stage if self.is_running else "idle"

    @property
    def running_elapsed_seconds(self) -> float:
        if not self.is_running or self._running_started_monotonic <= 0:
            return 0.0
        return max(0.0, time.monotonic() - self._running_started_monotonic)

    @property
    def can_restart_after_timeout(self) -> bool:
        return self._ocr_engine_factory is not None

    def shutdown(self) -> None:
        prereader = getattr(self, "local_prereader", None)
        preread_future = prereader.shutdown() if prereader is not None else None
        if self._running is not None and not self._running.done():
            self._running.cancel()
        shutdown = getattr(self.ocr_engine, "shutdown", None)
        if callable(shutdown):
            def close_engine(_completed=None):
                try:
                    shutdown()
                except Exception as exc:
                    append_diagnostics_log("capture_pipeline", f"ocr_engine_shutdown_failed error={exc}")
            # Closing a daemon in the middle of a background recognize call
            # can race its lazy startup. Finish/cancel that one call first,
            # then close the captured engine without blocking the Qt thread.
            if preread_future is not None and not preread_future.done():
                preread_future.add_done_callback(close_engine)
            else:
                close_engine()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)
        except Exception as exc:
            append_diagnostics_log("capture_pipeline", f"executor_shutdown_failed error={exc}")
        self._set_status("pipeline_shutdown")

    def reset_snapshot_hash_cache(self) -> None:
        with self._snapshot_hash_cache_lock:
            self._last_snapshot_hash = None
            self._snapshot_hash_cache.clear()
            self._screen_proof_cache.clear()
            self._snapshot_hash_owners.clear()
        self._title_cache.clear()
        self._verified_content_ocr_cache.clear()
        self._last_verified_title_context = None

    def prepare_identity_reacquire(self, snapshot_hash: str = "", reason: str = "contact_unknown") -> bool:
        """Allow one fresh verified frame without weakening identity gates."""
        if self.is_running:
            return False
        if snapshot_hash:
            self.forget_snapshot_hash(snapshot_hash)
        self._title_cache.clear()
        self._last_verified_title_context = None
        self._set_status(f"identity_reacquire_prepared:{reason}")
        return True

    def expire_running_job(self, reason: str) -> bool:
        """Quarantine a wedged native OCR call and rotate to a fresh worker.

        Python cannot safely kill a running native call.  The expired job id is
        therefore prevented from emitting any late title/page/contact result,
        while a new engine and executor serve the next capture.
        """
        with self._lifecycle_lock:
            future = self._running
            job_id = self._running_job_id
            if future is None or job_id is None or future.done() or self._ocr_engine_factory is None:
                return False
            try:
                replacement_engine = self._ocr_engine_factory()
            except Exception as exc:
                append_diagnostics_log("capture_pipeline", f"watchdog_engine_restart_failed job={job_id} error={exc}")
                return False
            submitted_state = self._submitted_states.get(job_id)
            submitted_account_owner = self._job_account_owners.get(job_id, ("", ""))
            old_executor = self.executor
            old_engine = self.ocr_engine
            self._expired_job_ids.add(job_id)
            self._retired_engines[job_id] = old_engine
            self.ocr_engine = replacement_engine
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whochat-capture-recovery")
            self._running = None
            self._running_job_id = None
            self._running_started_monotonic = 0.0
            self._running_stage = "idle"
            self._forget_job_snapshot_hash(job_id)
        future.cancel()
        try:
            old_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            old_executor.shutdown(wait=False)
        except Exception:
            pass
        timeout_reason = f"pipeline_timeout:{reason}"
        self._discard(timeout_reason, job_id=job_id, submitted_state=submitted_state,
                      submitted_account_owner=submitted_account_owner)
        self._set_status(timeout_reason)
        append_diagnostics_log("capture_pipeline", f"watchdog_expired job={job_id} reason={reason}")
        return True

    def forget_snapshot_hash(self, snapshot_hash: str) -> None:
        """Allow one more frame when identity has not reached two-frame stability."""
        with self._snapshot_hash_cache_lock:
            self._snapshot_hash_cache.pop(snapshot_hash, None)
            self._snapshot_hash_owners.pop(snapshot_hash, None)
            if self._last_snapshot_hash == snapshot_hash:
                self._last_snapshot_hash = None

    def submit(self, state: RuntimeState) -> int | None:
        with self._lifecycle_lock:
            return self._submit(state)

    def _submit(self, state: RuntimeState) -> int | None:
        provider = getattr(self, "capture_request_token_provider", None)
        request_token = provider() if callable(provider) else None
        if callable(provider) and request_token is None:
            self._discard("manual_generation_button_required")
            return None
        selection_provider = getattr(self, "manual_selection_provider", None)
        selection = selection_provider() if callable(selection_provider) else None
        if selection is not None:
            from whochat.services.manual_selection import ManualSelection
            if not isinstance(selection, ManualSelection) or request_token is None:
                self._discard("manual_selection_request_missing")
                return None
        if state.layout is None or state.window.rect is None:
            self._discard("layout_unavailable")
            return None
        gate = getattr(self, "capture_gate", None)
        decision = gate.evaluate(state) if gate is not None else state.capture_decision
        if not decision.should_capture:
            self._discard(decision.reason)
            return None
        heavy_ocr = getattr(self.ocr_engine, "name", "") == "paddleocr"
        if self._running and not self._running.done():
            if heavy_ocr:
                self._discard(f"pipeline_busy:job={self._running_job_id or '-'}")
                return None
            self._running.cancel()
            self._discard("superseded_running_job")
        self._job_id += 1
        self._latest_job_id = self._job_id
        job_id = self._job_id
        self._running_job_id = job_id
        self._capture_request_tokens[job_id] = request_token
        self._manual_selections[job_id] = selection
        while len(self._manual_selections) > 32:
            self._manual_selections.pop(next(iter(self._manual_selections)))
        activation = getattr(self,'activation_purpose_provider',None)
        if callable(activation) and activation() is True and request_token is not None:
            self._activation_request_tokens[job_id] = request_token
        while len(getattr(self,'_activation_request_tokens',{}))>32:
            self._activation_request_tokens.pop(next(iter(self._activation_request_tokens)))
        while len(self._capture_request_tokens) > 32:
            self._capture_request_tokens.pop(next(iter(self._capture_request_tokens)))
        self._running_started_monotonic = time.monotonic()
        self._running_stage = "queued"
        self._submitted_states[job_id] = state
        self._job_account_owners[job_id] = self._submission_account_owner(state)
        self._set_status(f"pipeline_started: job={job_id}")
        if _verbose_capture_diagnostics():
            append_diagnostics_log("capture_pipeline", f"started job={job_id} hwnd={state.window.hwnd} ocr={self.ocr_engine.name}")
        engine = self.ocr_engine
        self._running = self.executor.submit(self._run_job, job_id, state, engine)
        self._running.add_done_callback(
            lambda completed, completed_job_id=job_id: self._on_done(completed, completed_job_id)
        )
        return job_id

    def _submission_account_owner(self, state: RuntimeState) -> tuple[str, str]:
        """Read-only, immutable ownership; never relabel a late job as current."""
        checkpoint_reader = getattr(self.account_scope_resolver, "account_identity_checkpoint", None)
        if not callable(checkpoint_reader):
            return "", ""
        checkpoint = checkpoint_reader(state.window)
        if checkpoint is None or (
            checkpoint.runtime_key != state.account_identity_runtime_key
            or checkpoint.hwnd != state.window.hwnd
            or checkpoint.process_id != state.window.process_id
            or checkpoint.identity_epoch != state.account_identity_epoch
            or checkpoint.identity_state != "trusted"
            or not checkpoint.account_scope_id
            or not checkpoint.target_instance_id
        ):
            return "", ""
        return checkpoint.account_scope_id, checkpoint.target_instance_id

    def run_sync(self, state: RuntimeState) -> PipelineResult | None:
        job_id = self.submit(state)
        if job_id is None or self._running is None:
            return None
        try:
            result = self._running.result(timeout=30)
        except DuplicateSnapshotError as exc:
            self._discard(f"duplicate_snapshot:{exc.snapshot_hash}", job_id=job_id)
            if not self.retain_capture_images:
                self._cleanup_job_files(job_id)
            self._log_duplicate_snapshot(exc.snapshot_hash, synchronous=True)
            return None
        except StaleWindowContentError as exc:
            self._discard(f"{getattr(exc, 'discard_reason', 'stale_window_content')}:{exc.snapshot_hash}", job_id=job_id)
            self._cleanup_job_files(job_id)
            append_diagnostics_log("capture_pipeline", f"sync_discarded stale_window_content={exc.snapshot_hash}")
            return None
        except StaleCaptureContextError as exc:
            self._discard(f"stale_capture_context:{exc.reason}", job_id=job_id)
            self._cleanup_job_files(job_id)
            append_diagnostics_log("capture_pipeline", f"sync_discarded stale_capture_context={exc.reason}")
            return None
        except Exception as exc:
            self._discard(f"pipeline_failed:{exc}", job_id=job_id)
            append_diagnostics_log("capture_pipeline", f"sync_failed error={exc}")
            return None
        if not self.retain_capture_images:
            self._cleanup_job_files(result.job_id)
        return result

    def _assert_native_account_capture(self, job_id, state, snapshot_hash, account_context, token):
        reader = getattr(self.account_scope_resolver, "capture_identity_reason", None)
        if not callable(reader) or account_context is None or not account_context.actions_allowed:
            return
        reason = reader(state.window, token)
        if reason:
            context = self.account_scope_resolver.freeze_for_explicit_boundary(state.window, reason)
            if context.epoch_changed:
                self._emit_account_identity_boundary(state, job_id, snapshot_hash, context)
            self.forget_snapshot_hash(snapshot_hash)
            raise StaleCaptureContextError(context.reason)

    def _run_job(self, job_id: int, state: RuntimeState, ocr_engine: OcrEngine) -> PipelineResult:
        self._assert_job_current(job_id, state, "before_capture")
        job_started = time.monotonic()
        capture_cycle_started_at = utc_now_iso()
        if state.layout is None or state.window.rect is None:
            raise RuntimeError("pipeline requires layout and window rect")
        if (getattr(state.layout, "client_rect", None) is not None
                and state.layout.client_rect != getattr(state.window, "client_rect", None)):
            raise StaleCaptureContextError("layout_client_geometry_owner_mismatch")
        output = app_data_dir() / "capture" / f"job_{job_id}.png"
        self._set_running_stage(job_id, "capture")
        capture_started_at = utc_now_iso()
        capture_started = time.monotonic()
        # Runtime's foreground bit is poll-based. Recheck native evidence
        # before screen pixels are read, not only after the OCR job returns.
        evidence_gate = getattr(self.account_scope_resolver, "require_avatar_evidence_current", None)
        if callable(evidence_gate):
            try:
                evidence_gate(state.window)
            except AccountImageEvidenceUnavailable as exc:
                raise StaleCaptureContextError(str(exc)) from exc
        if getattr(state.window, "client_rect", None) is not None:
            _assert_native_client_geometry(state, "before_capture")
        native_reader = getattr(self.account_scope_resolver, "capture_identity_token", None)
        native_capture_token = native_reader(state.window) if callable(native_reader) else None
        native_options = ({"native_capture_token": native_capture_token}
                          if getattr(self.account_scope_resolver, "native_identity", None) is not None else {})
        image_path = self.capture_func(state.window.rect.as_tuple(), output)
        if getattr(state.window, "client_rect", None) is not None:
            _assert_native_client_geometry(state, "after_capture")
        captured_at_ms = int(time.monotonic() * 1000)
        capture_elapsed_ms = _elapsed_ms(capture_started)
        capture_completed_at = utc_now_iso()
        with self._lifecycle_lock:
            self._assert_job_current(job_id, state, "after_capture")
            gate = getattr(self, "capture_gate", None)
            if gate is not None:
                gate.record_capture(state, captured_at_ms=captured_at_ms)
        # The ratio-based layout prior drifts on real WeChat 4.x frames; measure
        # the three boundaries that matter from the screenshot itself.  Falls
        # back to the prior when the pixels are not conclusive.
        layout_started = time.monotonic()
        selection = getattr(self, "_manual_selections", {}).get(job_id)
        if selection is not None:
            from whochat.services.manual_selection import SelectionError
            try:
                layout = selection.for_capture(state, image_path, native_capture_token)
            except SelectionError as exc:
                raise StaleCaptureContextError(str(exc)) from exc
        else:
            layout = refine_layout_from_image(image_path, state.layout)
        if getattr(state.window, "client_rect", None) is not None:
            # Refinement/calibration may adjust interior splits but may never
            # reintroduce the OS-proven non-client frame into conversation OCR.
            layout = constrain_layout_to_client(layout, state.window.client_rect)
        layout_elapsed_ms = _elapsed_ms(layout_started)
        # This object belongs only to this captured job, never to a path cache.
        # Closing the decoder here releases the file; the independent pixels
        # remain available to hashing, crops and parsing until this job exits.
        captured_image = Image.open(image_path)
        try:
            primary_image = captured_image.copy()
        finally:
            captured_image.close()
        primary_rgb = None
        try:
            primary_rgb = primary_image if primary_image.mode == "RGB" else primary_image.convert("RGB")
            # Exact title + message pixels are the earliest safe unchanged-screen
            # gate.  Run it before account-instance persistence, crop file writes,
            # and OCR so a stable 2 s foreground probe remains read-only and cheap.
            # Identity-pending ingestion explicitly forgets this digest, preserving
            # the mandatory second distinct observation.
            conversation_digest = _exact_conversation_digest(image_path, layout, frame_image=primary_rgb)
            snapshot_hash = hashlib.sha256(
                (
                    f"hwnd={state.window.hwnd}|pid={state.window.process_id}|"
                    f"window_generation={state.window_generation}|capture_generation={state.capture_generation}|"
                    f"pixels={conversation_digest}"
                ).encode("ascii")
            ).hexdigest()[:32]
            with self._lifecycle_lock:
                self._assert_job_current(job_id, state, "before_account_observation")
                if self._has_seen_snapshot_hash(snapshot_hash):
                    raise DuplicateSnapshotError(snapshot_hash)
                self._job_snapshot_hashes[job_id] = snapshot_hash
                try:
                    account_context = (
                        self.account_scope_resolver.observe(image_path, layout, state.window, **native_options)
                        if self.account_scope_resolver is not None
                        else None
                    )
                except AccountImageEvidenceUnavailable as exc:
                    # This is a discarded capture, not an account boundary: retain
                    # the existing epoch/Contact and let the next valid frame retry.
                    raise StaleCaptureContextError(str(exc)) from exc
            if account_context is not None and account_context.epoch_changed:
                self._emit_account_identity_boundary(state, job_id, snapshot_hash, account_context)
                self.forget_snapshot_hash(snapshot_hash)
                raise StaleCaptureContextError(account_context.reason)
            if (native_capture_token is not None
                    and getattr(self.account_scope_resolver, "native_identity", None) is not None
                    and account_context is not None
                    and account_context.actions_allowed is False
                    and not account_context.account_scope_id and not account_context.target_instance_id
                    and account_context.reason ==
                        "account_identity_revalidating:native_account_confirmation_pending"):
                # This full-window/avatar observation has fresh matching native
                # reads around capture, but the resolver still requires a second
                # independent account observation. Nothing on this frame can be
                # ingested or authorize an action. Publish only recovery progress
                # instead of doing title/body OCR that the coordinator must drop.
                # The trusted path below still performs both fresh customer-title
                # observations, all message verification and all action guards.
                self._assert_job_current(job_id, state, "before_account_recovery_progress")
                self.forget_snapshot_hash(snapshot_hash)
                return PipelineResult(
                    job_id=job_id, hwnd=state.window.hwnd, target_app=state.window.target.value,
                    app_label=state.window.app_label, snapshot_hash=snapshot_hash,
                    image_path=image_path, ocr_image_path=image_path, crop_rect=None,
                    layout=layout, ocr_result=OcrResult([], str(image_path), ocr_engine.name,
                        "工作微信账号等待第二次独立采集确认"),
                    page=state.page, messages=[], created_at=utc_now_iso(),
                    title_ocr_elapsed_ms=0, content_ocr_elapsed_ms=0,
                    capture_elapsed_ms=capture_elapsed_ms, layout_elapsed_ms=layout_elapsed_ms,
                    total_elapsed_ms=_elapsed_ms(job_started),
                    process_id=state.window.process_id, window_generation=state.window_generation,
                    capture_generation=state.capture_generation,
                    account_scope_reason=account_context.reason,
                    account_identity_epoch=account_context.identity_epoch,
                    account_identity_runtime_key=account_context.runtime_key,
                    screen_proof={"valid": False, "reason": "native_account_confirmation_pending"},
                    performance_trace={"account_recovery_only": True},
                )
            self._set_running_stage(job_id, "title_input")
            title_input_started = time.monotonic()
            title_ocr_image_path, title_layout, title_offset, title_crop_rect = _prepare_title_ocr_input(
                image_path, layout, materialize=False,
            )
            title_input_elapsed_ms = _elapsed_ms(title_input_started)
            content_input_started = time.monotonic()
            ocr_image_path, ocr_layout, offset, crop_rect = _prepare_ocr_input(image_path, layout, materialize=False)
            content_input_elapsed_ms = _elapsed_ms(content_input_started)
            # ``image_path`` is the window crop and therefore 0-based, while
            # ``layout.title_rect`` lives in absolute screen coordinates.  Hashing the
            # band with absolute coordinates collapses the region to zero width as
            # soon as the window does not sit at the screen origin, and PIL then
            # raises "height and width must be > 0", killing the whole job (no title
            # OCR -> no contact -> no messages -> no memories).  Convert to image
            # space first.
            # Identity fingerprints intentionally cover only the stable text side
            # of the title bar.  The right side contains WeChat action/search
            # controls whose hover/repaint state can change while the conversation
            # itself has not.  Using the full title bar here caused a correct,
            # freshly confirmed contact to be invalidated by a harmless repaint.
            title_identity_rect = title_identity_screen_rect(layout)
            title_band = _shift_rect(title_identity_rect, -layout.window_rect.left, -layout.window_rect.top)
            initial_title_hash = _title_visual_digest(image_path, title_band, frame_image=primary_image)
            exact_title_hash = _exact_region_digest(image_path, title_band, frame_image=primary_rgb)
            title_context = (
                state.window.hwnd,
                state.window.process_id,
                state.window_generation,
                state.capture_generation,
                account_context.account_scope_id if account_context else None,
                account_context.target_instance_id if account_context else "",
                account_context.identity_epoch if account_context else state.account_identity_epoch,
            )
            previous_title_context = self._last_verified_title_context
            title_pixels_changed = bool(
                previous_title_context is not None
                and previous_title_context[:-1] == title_context
                and previous_title_context[-1] != initial_title_hash
            )
            title_cache_key = (
                state.window_generation,
                account_context.account_scope_id if account_context else None,
                account_context.target_instance_id if account_context else "",
                account_context.identity_epoch if account_context else state.account_identity_epoch,
                layout.title_rect.as_tuple(),
                exact_title_hash,
            )
            title_started_at = utc_now_iso()
            title_started = time.monotonic()
            title_source = "roi_ocr"
            # Each button-owned capture must independently read the live title.
            # This also isolates old persistent fingerprints polluted by an older
            # perceptual-hash fallback: they cannot label a manual request, even
            # when the old entry happens to match its exact pixels.
            manual_title_fresh_required = self._capture_request_tokens.get(job_id) is not None
            cached_title = None if manual_title_fresh_required else self._title_cache.get(title_cache_key)
            fingerprint_hint = None
            if (not manual_title_fresh_required and cached_title is None
                    and callable(self.fast_title_lookup) and account_context is not None):
                fingerprint_hint = self.fast_title_lookup(
                    account_context.account_scope_id or "",
                    account_context.target_instance_id or "",
                    exact_title_hash,
                    str(initial_title_hash),
                )
            title_cache_hit = cached_title is not None or fingerprint_hint is not None
            if cached_title is not None:
                self._title_cache.move_to_end(title_cache_key)
                title_ocr_result = OcrResult(
                    boxes=list(cached_title.boxes),
                    source_image=str(image_path),
                    engine=cached_title.engine,
                    warning=cached_title.warning,
                )
                title_source = "roi_lru"
                title_timings: dict[str, int] = {}
            elif fingerprint_hint is not None:
                hint_text = str(getattr(fingerprint_hint, "normalized_title", "") or "").strip()
                hint_confidence = float(getattr(fingerprint_hint, "confidence", 0.0) or 0.0)
                if not hint_text or hint_confidence < 0.95:
                    fingerprint_hint = None
                else:
                    box = Rect(
                        layout.title_rect.left + max(2, layout.title_rect.width // 20),
                        layout.title_rect.top + max(2, layout.title_rect.height // 8),
                        layout.title_rect.right - max(2, layout.title_rect.width // 4),
                        layout.title_rect.bottom - max(2, layout.title_rect.height // 8),
                    )
                    title_ocr_result = OcrResult(
                        boxes=[OcrTextBox(hint_text, box, hint_confidence, OcrRegion.TITLE, "confirmed_title_fingerprint")],
                        source_image=str(image_path),
                        engine="Confirmed Title Fingerprint",
                        warning=None,
                    )
                    title_source = "confirmed_fingerprint"
                    title_timings = {}
            if (cached_title is None and fingerprint_hint is None) or self.retain_capture_images:
                title_input_started = time.monotonic()
                title_ocr_image_path, title_layout, title_offset, title_crop_rect = _prepare_title_ocr_input(
                    image_path, layout, frame_image=primary_image,
                )
                title_input_elapsed_ms += _elapsed_ms(title_input_started)
                title_started = time.monotonic()
            if cached_title is None and fingerprint_hint is None:
                self._set_running_stage(job_id, "title_ocr")
                title_ocr_result = _offset_ocr_result(
                    getattr(ocr_engine, "recognize_title", ocr_engine.recognize)(title_ocr_image_path, title_layout),
                    title_offset,
                    image_path,
                )
                title_timings = dict(getattr(ocr_engine, "last_timings", {}) or {})
            self._assert_job_current(job_id, state, "after_title_ocr")
            title_elapsed_ms = _elapsed_ms(title_started)
            title_completed_at = utc_now_iso()
            created_at = utc_now_iso()
            previous_title_text = _normalized_title_observation(
                self.last_title_result.ocr_result if self.last_title_result is not None else None
            )
            current_title_text = _normalized_title_observation(title_ocr_result)
            title_identity_changed = bool(
                title_pixels_changed
                and previous_title_text
                and current_title_text
                and previous_title_text != current_title_text
            )
            # Verify only the title pixels immediately.  This is the safety boundary
            # that lets the activation path publish identity evidence while the
            # full message OCR continues independently in the same worker.
            fast_verify_output = output.with_name(f"{output.stem}_title_verify{output.suffix}")
            fast_verify_started = time.monotonic()
            if getattr(state.window, "client_rect", None) is not None:
                _assert_native_client_geometry(state, "before_title_verify")
            fast_verify_path = self.capture_func(title_identity_rect.as_tuple(), fast_verify_output)
            try:
                if getattr(state.window, "client_rect", None) is not None:
                    _assert_native_client_geometry(state, "after_title_verify")
                # Even when a transport reuses a filename, verification reads NEW
                # pixels. It never borrows primary_image or an earlier job's frame.
                verify_image = Image.open(fast_verify_path)
                try:
                    expected_size = (title_identity_rect.width, title_identity_rect.height)
                    if verify_image.size == expected_size:
                        verify_rect = Rect(0, 0, verify_image.width, verify_image.height)
                    else:
                        verify_rect = title_band
                    fast_verify_title_hash = _title_visual_digest(fast_verify_path, verify_rect, frame_image=verify_image)
                finally:
                    verify_image.close()
            finally:
                if Path(fast_verify_path).resolve() == fast_verify_output.resolve():
                    _safe_unlink_capture(fast_verify_output)
            fast_verify_elapsed_ms = _elapsed_ms(fast_verify_started)
            if fast_verify_title_hash != initial_title_hash:
                self.forget_snapshot_hash(snapshot_hash)
                # One mismatched verification frame is ambiguous (hover repaint,
                # transient overlay, or a real switch).  Discard it without
                # clearing a confirmed identity.  A real switch is invalidated by
                # the next two-frame-stable title observation below or by the
                # independent fast title probe.
                raise StaleWindowContentError(snapshot_hash)
            self._assert_job_current(job_id, state, "after_title_verify")
            if title_identity_changed:
                self._emit_current_context_change(job_id, state,
                    CaptureContextChange(
                        job_id=job_id, hwnd=state.window.hwnd, process_id=state.window.process_id,
                        window_generation=state.window_generation, capture_generation=state.capture_generation,
                        snapshot_hash=snapshot_hash, reason="title_pixels_changed",
                        account_scope_id=account_context.account_scope_id if account_context else None,
                        target_instance_id=account_context.target_instance_id if account_context else "",
                        account_identity_epoch=account_context.identity_epoch if account_context else state.account_identity_epoch,
                        account_identity_runtime_key=(
                            account_context.runtime_key if account_context else state.account_identity_runtime_key
                        ),
                    )
                )
            title_result = TitleOcrResult(
                job_id=job_id,
                hwnd=state.window.hwnd,
                target_app=state.window.target.value,
                app_label=state.window.app_label,
                snapshot_hash=snapshot_hash,
                image_path=image_path,
                title_ocr_image_path=title_ocr_image_path,
                title_crop_rect=title_crop_rect,
                layout=layout,
                ocr_result=title_ocr_result,
                created_at=created_at,
                elapsed_ms=title_elapsed_ms,
                process_id=state.window.process_id,
                window_generation=state.window_generation,
                capture_generation=state.capture_generation,
                account_scope_id=account_context.account_scope_id if account_context else None,
                account_scope_name=account_context.account_scope_name if account_context else "",
                target_instance_id=account_context.target_instance_id if account_context else "",
                account_scope_reason=account_context.reason if account_context else "resolver_unavailable",
                account_identity_epoch=account_context.identity_epoch if account_context else state.account_identity_epoch,
                account_identity_runtime_key=(
                    account_context.runtime_key if account_context else state.account_identity_runtime_key
                ),
                title_visual_hash=str(initial_title_hash),
                title_exact_hash=exact_title_hash,
                title_source=title_source,
                performance_trace={
                    "contact_switch_detected_at": capture_started_at if title_identity_changed else "",
                    "title_capture_started_at": capture_started_at,
                    "title_resolved_at": title_completed_at,
                    "title_detect_ms": _elapsed_ms(job_started),
                    "title_verify_elapsed_ms": fast_verify_elapsed_ms,
                    "title_ocr_elapsed_ms": title_elapsed_ms,
                    "title_cache_hit": title_cache_hit,
                    "title_source": title_source,
                    "manual_title_fresh_required": manual_title_fresh_required,
                    "page_was_chat_dm": state.page.page_type.value == "chat_dm",
                    "observed_sequence": time.monotonic_ns(),
                },
            )
            with self._lifecycle_lock:
                self._assert_job_current(job_id, state, "before_title_publish")
                self._assert_native_account_capture(job_id, state, snapshot_hash, account_context, native_capture_token)
                # Pending/unbound account evidence must be observed again even
                # when all pixels are unchanged. Otherwise dedup runs before the
                # avatar's second recovery frame and can freeze the account forever.
                if account_context is None or account_context.actions_allowed:
                    self._remember_snapshot_hash(snapshot_hash, job_id=job_id)
                if not title_ocr_result.warning and title_ocr_result.boxes:
                    self._title_cache[title_cache_key] = title_ocr_result
                    self._title_cache.move_to_end(title_cache_key)
                    while len(self._title_cache) > self._title_cache_size:
                        self._title_cache.popitem(last=False)
                self._last_verified_title_context = title_context + (initial_title_hash,)
                self.last_title_result = title_result
                self.title_ready.emit(title_result)
            if layout.input_boundary_verified is False:
                # Geometry failure is a negative result, never a body/source
                # capability. Once this request's live title and account are
                # checked, stop before editor/caret repaint can trigger endless
                # body-difference retries. Normal positive frames below retain
                # every full OCR, fresh body and independent-frame check.
                boundary_proof = _build_current_screen_capture_proof(
                    job_id=job_id, state=state, account_context=account_context,
                    snapshot_hash=snapshot_hash, created_at=created_at, layout=layout,
                    ocr_result=title_ocr_result, page=state.page, messages=[], parser_report={},
                    title_exact_hash=exact_title_hash, capture_verified=False, title_verified=True,
                    manual_request_token=self._capture_request_tokens.get(job_id),
                    title_only_rejection=True,
                )
                if boundary_proof.get("negative_capture_evidence") is True:
                    self._assert_job_current(job_id, state, "before_boundary_rejection")
                    self.forget_snapshot_hash(snapshot_hash)
                    return PipelineResult(
                        job_id=job_id, hwnd=state.window.hwnd, target_app=state.window.target.value,
                        app_label=state.window.app_label, snapshot_hash=snapshot_hash,
                        image_path=image_path, ocr_image_path=title_ocr_image_path, crop_rect=title_crop_rect,
                        layout=layout, ocr_result=title_ocr_result, page=state.page, messages=[],
                        created_at=created_at, title_ocr_image_path=title_ocr_image_path,
                        title_crop_rect=title_crop_rect, title_ocr_elapsed_ms=title_elapsed_ms,
                        total_elapsed_ms=_elapsed_ms(job_started), process_id=state.window.process_id,
                        window_generation=state.window_generation, capture_generation=state.capture_generation,
                        account_scope_id=account_context.account_scope_id,
                        account_scope_name=account_context.account_scope_name,
                        target_instance_id=account_context.target_instance_id,
                        account_scope_reason=account_context.reason,
                        account_identity_epoch=account_context.identity_epoch,
                        account_identity_runtime_key=account_context.runtime_key,
                        title_visual_hash=str(initial_title_hash), title_exact_hash=exact_title_hash,
                        screen_proof=boundary_proof,
                    )
            content_started_at = utc_now_iso()
            content_started = time.monotonic()
            self._set_running_stage(job_id, "content_ocr")
            content_reuse_key = _content_ocr_reuse_key(
                state, account_context, layout, exact_title_hash, conversation_digest, ocr_engine,
            )
            cached_content = self._cached_verified_content_ocr(content_reuse_key, image_path)
            if cached_content is None and manual_title_fresh_required:
                cached_content = _local_preread_ocr(
                    self, "content", state, account_context, layout, primary_rgb, ocr_engine, image_path,
                )
            content_ocr_cache_hit = cached_content is not None
            if not content_ocr_cache_hit or self.retain_capture_images:
                content_input_started = time.monotonic()
                ocr_image_path, ocr_layout, offset, crop_rect = _prepare_ocr_input(
                    image_path, layout, frame_image=primary_image,
                )
                content_input_elapsed_ms += _elapsed_ms(content_input_started)
                content_started = time.monotonic()
            content_ocr_result = cached_content if cached_content is not None else _offset_ocr_result(
                ocr_engine.recognize(ocr_image_path, ocr_layout), offset, image_path,
            )
            # Only this exact input's raw inference can enter the pre-read
            # cache. Recovery below may use additional crops/parameters.
            raw_content_ocr_result = content_ocr_result
            content_timings = {} if content_ocr_cache_hit else dict(getattr(ocr_engine, "last_timings", {}) or {})
            self._assert_job_current(job_id, state, "after_content_ocr")
            local_read_trace = {"attempts": 0, "recovered": False}
            if manual_title_fresh_required:
                from whochat.ocr.low_confidence import recover_low_confidence_content
                content_ocr_result, local_read_trace = recover_low_confidence_content(
                    content_ocr_result, title_ocr_result, layout, primary_rgb, ocr_engine, output,
                    lambda: self._assert_job_current(job_id, state, "local_text_reread"),
                )
            content_elapsed_ms = _elapsed_ms(content_started)
            content_completed_at = utc_now_iso()
            verify_output = output.with_name(f"{output.stem}_verify{output.suffix}")
            self._set_running_stage(job_id, "verify_capture")
            verify_capture_started = time.monotonic()
            if getattr(state.window, "client_rect", None) is not None:
                _assert_native_client_geometry(state, "before_full_verify")
            verify_image_path = self.capture_func(state.window.rect.as_tuple(), verify_output)
            verify_capture_elapsed_ms = _elapsed_ms(verify_capture_started)
            try:
                if getattr(state.window, "client_rect", None) is not None:
                    _assert_native_client_geometry(state, "after_full_verify")
                verified_image = Image.open(verify_image_path)
                try:
                    verify_title_hash = _title_visual_digest(verify_image_path, title_band, frame_image=verified_image)
                    verified_rgb = verified_image.convert("RGB")
                    try:
                        verify_conversation_digest = _exact_conversation_digest(
                            verify_image_path, layout, frame_image=verified_rgb,
                        )
                        if (verify_title_hash != initial_title_hash
                                or verify_conversation_digest != conversation_digest):
                            self._record_verification_difference(
                                job_id, state, layout, primary_rgb, verified_rgb,
                                title_hash_matches=verify_title_hash == initial_title_hash,
                            )
                    finally:
                        verified_rgb.close()
                finally:
                    verified_image.close()
            finally:
                if Path(verify_image_path).resolve() == verify_output.resolve():
                    _safe_unlink_capture(verify_output)
            if verify_title_hash != initial_title_hash:
                self.forget_snapshot_hash(snapshot_hash)
                # Content OCR finishing after a single changed title sample must
                # not revoke title identity that was already published safely.
                # The frame is stale for message ingestion, so discard only the
                # full result; the two-frame title paths own identity invalidation.
                raise StaleWindowContentError(snapshot_hash)
            if verify_conversation_digest != conversation_digest:
                self.forget_snapshot_hash(snapshot_hash)
                self._emit_current_context_change(job_id, state, CaptureContextChange(
                    job_id=job_id, hwnd=state.window.hwnd, process_id=state.window.process_id,
                    window_generation=state.window_generation, capture_generation=state.capture_generation,
                    snapshot_hash=snapshot_hash, reason="current_screen_body_changed",
                    account_scope_id=account_context.account_scope_id if account_context else None,
                    target_instance_id=account_context.target_instance_id if account_context else "",
                    account_identity_epoch=account_context.identity_epoch if account_context else state.account_identity_epoch,
                    account_identity_runtime_key=account_context.runtime_key if account_context else state.account_identity_runtime_key,
                ))
                raise CurrentScreenBodyChangedError(snapshot_hash)
            self._assert_job_current(job_id, state, "after_verify_capture")
            ocr_result = _merge_ocr_results(title_ocr_result, content_ocr_result)
            classification_started = time.monotonic()
            page = classify_page_from_ocr(ocr_result, layout)
            page_classification_elapsed_ms = _elapsed_ms(classification_started)
            explicit_boundary = account_identity_boundary_reason(ocr_result, layout, page)
            if explicit_boundary and self.account_scope_resolver is not None:
                with self._lifecycle_lock:
                    self._assert_job_current(job_id, state, "before_explicit_account_boundary")
                    account_context = self.account_scope_resolver.freeze_for_explicit_boundary(
                        state.window,
                        explicit_boundary,
                    )
                if account_context.epoch_changed:
                    self._emit_account_identity_boundary(state, job_id, snapshot_hash, account_context)
                self.forget_snapshot_hash(snapshot_hash)
                raise StaleCaptureContextError(account_context.reason)
            messages, parser_report = parse_visible_messages_with_evidence(
                ocr_result, layout, direct_chat=page.page_type == PageType.CHAT_DM,
                frame_image=primary_rgb,
            )
            parser_report["low_confidence_recovery_attempts"] = local_read_trace["attempts"]
            if local_read_trace["attempts"]:
                append_diagnostics_log("capture_pipeline", "local_text_reread " + json.dumps(
                    {"job_id": job_id, **local_read_trace}, separators=(",", ":"), allow_nan=False))
            try:
                append_diagnostics_log("capture_pipeline", "current_screen_geometry " + json.dumps(
                    _current_screen_geometry_metadata(job_id, layout, crop_rect, ocr_result, messages, parser_report),
                    ensure_ascii=True, separators=(",", ":"), allow_nan=False,
                ))
            except (ValueError, TypeError, AttributeError):
                append_diagnostics_log("capture_pipeline", "current_screen_geometry unavailable")
            self._assert_job_current(job_id, state, "before_result_return")
            self._assert_native_account_capture(job_id, state, snapshot_hash, account_context, native_capture_token)
            screen_proof = _build_current_screen_capture_proof(
                job_id=job_id, state=state, account_context=account_context, snapshot_hash=snapshot_hash,
                created_at=created_at, layout=layout, ocr_result=ocr_result, page=page, messages=messages,
                parser_report=parser_report, title_exact_hash=exact_title_hash,
                capture_verified=True, title_verified=True,
                manual_request_token=self._capture_request_tokens.get(job_id),
                account_avatar_anchor=_same_frame_account_avatar_anchor(image_path, layout, account_context),
                manual_activation=(self._activation_request_tokens.get(job_id) is not None
                                   and self._activation_request_tokens[job_id]==self._capture_request_tokens.get(job_id)),
            )
            with self._lifecycle_lock:
                self._assert_job_current(job_id, state, "before_screen_proof_cache")
                self._record_screen_proof_cache(snapshot_hash, screen_proof, job_id=job_id)
                if (screen_proof.get("valid") is True and parser_report.get("complete") is True
                        and not content_ocr_cache_hit):
                    self._remember_verified_content_ocr(content_reuse_key, content_ocr_result)
                if manual_title_fresh_required and not content_ocr_cache_hit:
                    _remember_local_preread_ocr(
                        self, state, account_context, layout, primary_rgb, ocr_engine,
                        raw_content_ocr_result,
                    )
            return PipelineResult(
                job_id=job_id,
                hwnd=state.window.hwnd,
                target_app=state.window.target.value,
                app_label=state.window.app_label,
                snapshot_hash=snapshot_hash,
                image_path=image_path,
                ocr_image_path=ocr_image_path,
                crop_rect=crop_rect,
                layout=layout,
                ocr_result=ocr_result,
                page=page,
                messages=messages,
                created_at=created_at,
                title_ocr_image_path=title_ocr_image_path,
                title_crop_rect=title_crop_rect,
                title_ocr_elapsed_ms=title_elapsed_ms,
                content_ocr_elapsed_ms=content_elapsed_ms,
                total_elapsed_ms=_elapsed_ms(job_started),
                capture_elapsed_ms=capture_elapsed_ms,
                verify_capture_elapsed_ms=verify_capture_elapsed_ms,
                page_classification_elapsed_ms=page_classification_elapsed_ms,
                layout_elapsed_ms=layout_elapsed_ms,
                title_input_elapsed_ms=title_input_elapsed_ms,
                content_input_elapsed_ms=content_input_elapsed_ms,
                title_preprocess_elapsed_ms=title_timings.get("preprocess_ms", 0),
                title_inference_elapsed_ms=title_timings.get("inference_ms", 0),
                title_postprocess_elapsed_ms=title_timings.get("postprocess_ms", 0),
                content_preprocess_elapsed_ms=content_timings.get("preprocess_ms"),
                content_inference_elapsed_ms=content_timings.get("inference_ms"),
                content_postprocess_elapsed_ms=content_timings.get("postprocess_ms"),
                title_cache_hit=title_cache_hit,
                performance_trace={
                    "capture_cycle_started_at": capture_cycle_started_at,
                    "capture_started_at": capture_started_at,
                    "capture_completed_at": capture_completed_at,
                    "title_ocr_started_at": title_started_at,
                    "title_ocr_completed_at": title_completed_at,
                    "fast_title_published_at": title_completed_at,
                    "content_ocr_started_at": content_started_at,
                    "content_ocr_completed_at": content_completed_at,
                    "title_cache_hit": title_cache_hit,
                    "title_source": title_source,
                    "manual_title_fresh_required": manual_title_fresh_required,
                    "capture_elapsed_ms": capture_elapsed_ms,
                    "layout_elapsed_ms": layout_elapsed_ms,
                    "title_input_elapsed_ms": title_input_elapsed_ms,
                    "content_input_elapsed_ms": content_input_elapsed_ms,
                    "title_ocr_elapsed_ms": title_elapsed_ms,
                    "title_verify_elapsed_ms": fast_verify_elapsed_ms,
                    "content_ocr_elapsed_ms": content_elapsed_ms,
                    "content_ocr_cache_hit": content_ocr_cache_hit,
                    "verify_capture_elapsed_ms": verify_capture_elapsed_ms,
                    "page_classification_elapsed_ms": page_classification_elapsed_ms,
                },
                process_id=state.window.process_id,
                window_generation=state.window_generation,
                capture_generation=state.capture_generation,
                account_scope_id=account_context.account_scope_id if account_context else None,
                account_scope_name=account_context.account_scope_name if account_context else "",
                target_instance_id=account_context.target_instance_id if account_context else "",
                account_scope_reason=account_context.reason if account_context else "resolver_unavailable",
                account_identity_epoch=account_context.identity_epoch if account_context else state.account_identity_epoch,
                account_identity_runtime_key=(
                    account_context.runtime_key if account_context else state.account_identity_runtime_key
                ),
                title_visual_hash=str(initial_title_hash),
                title_exact_hash=exact_title_hash,
                screen_proof=screen_proof,
                frame_diagnostic=build_signature(primary_rgb, layout, snapshot_hash, job_id,
                                                 title_rect=title_identity_screen_rect(layout)),
                photo_crops=collect_photo_crops(image_path, layout, messages, permitted=(
                    screen_proof.get("valid") is True
                    and bool(self._capture_request_tokens.get(job_id))
                )),
            )
        finally:
            # Failed Futures retain traceback locals. Release owned pixel
            # buffers even when a job is discarded or the caller keeps it.
            try:
                if primary_rgb is not None and primary_rgb is not primary_image:
                    primary_rgb.close()
            finally:
                primary_image.close()

    def _assert_job_active(self, job_id: int, stage: str) -> None:
        with self._lifecycle_lock:
            selection = getattr(self, "_manual_selections", {}).get(job_id)
            if selection is not None:
                selection_provider = getattr(self, "manual_selection_provider", None)
                if not callable(selection_provider) or selection_provider() is not selection:
                    raise StaleCaptureContextError("manual_selection_request_changed")
            provider = getattr(self, "capture_request_token_provider", None)
            if callable(provider) and (
                self._capture_request_tokens.get(job_id) is None
                or provider() != self._capture_request_tokens.get(job_id)
            ):
                raise StaleCaptureContextError(f"{stage}:manual_request_expired:{job_id}")
            if job_id in self._expired_job_ids:
                raise StaleCaptureContextError(f"{stage}:expired_job:{job_id}")
            if job_id != self._latest_job_id:
                raise StaleCaptureContextError(f"{stage}:superseded_job:{job_id}:{self._latest_job_id}")

    def _assert_job_current(self, job_id: int, submitted: RuntimeState, stage: str) -> None:
        self._assert_job_active(job_id, stage)
        self._assert_context_current(submitted, stage)

    def _emit_current_context_change(self, job_id: int, state: RuntimeState, event: CaptureContextChange) -> None:
        with self._lifecycle_lock:
            self._assert_job_current(job_id, state, "before_context_publish")
            self.context_changed.emit(event)

    def _assert_context_current(self, submitted: RuntimeState, stage: str) -> None:
        if self.current_state_provider is None:
            return
        current = self.current_state_provider()
        if current is None:
            raise StaleCaptureContextError(f"{stage}:runtime_unavailable")
        expected = (
            submitted.window.hwnd,
            submitted.window.process_id,
            submitted.window_generation,
            submitted.capture_generation,
            submitted.account_identity_epoch,
            submitted.account_identity_runtime_key,
        )
        actual = (
            current.window.hwnd,
            current.window.process_id,
            current.window_generation,
            current.capture_generation,
            current.account_identity_epoch,
            current.account_identity_runtime_key,
        )
        if expected != actual:
            raise StaleCaptureContextError(f"{stage}:expected={expected}:actual={actual}")

    def _emit_account_identity_boundary(self, state, job_id: int, snapshot_hash: str, context) -> None:
        with self._lifecycle_lock:
            self._assert_job_active(job_id, "before_account_boundary_publish")
            self.context_changed.emit(
                CaptureContextChange(
                    job_id=job_id,
                    hwnd=state.window.hwnd,
                    process_id=state.window.process_id,
                    window_generation=state.window_generation,
                    capture_generation=state.capture_generation,
                    snapshot_hash=snapshot_hash,
                    reason=f"account_identity_revalidation:{context.reason}",
                    account_scope_id=None,
                    target_instance_id="",
                    account_identity_epoch=context.identity_epoch,
                    account_identity_runtime_key=context.runtime_key,
                )
            )

    def _on_done(self, future: Future, completed_job_id: int) -> None:
        if completed_job_id in self._expired_job_ids:
            self._forget_job_snapshot_hash(completed_job_id)
            self._expired_job_ids.discard(completed_job_id)
            with self._lifecycle_lock:
                self._submitted_states.pop(completed_job_id, None)
                self._job_account_owners.pop(completed_job_id, None)
            retired_engine = self._retired_engines.pop(completed_job_id, None)
            shutdown = getattr(retired_engine, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
            self._cleanup_job_files(completed_job_id)
            append_diagnostics_log("capture_pipeline", f"discarded expired_late_result={completed_job_id}")
            return
        self._job_snapshot_hashes.pop(completed_job_id, None)
        # A cancelled light-weight OCR future may finish after its replacement
        # has already started.  Only the current job may clear the running id;
        # otherwise an old callback makes the new job look idle and can clean
        # files belonging to the wrong capture.
        if self._running_job_id == completed_job_id:
            self._running_job_id = None
            self._running_started_monotonic = 0.0
            self._running_stage = "idle"
        with self._lifecycle_lock:
            submitted_state = self._submitted_states.pop(completed_job_id, None)
            submitted_account_owner = self._job_account_owners.pop(completed_job_id, ("", ""))
        try:
            result = future.result()
        except DuplicateSnapshotError as exc:
            self._discard(
                f"duplicate_snapshot:{exc.snapshot_hash}",
                job_id=completed_job_id,
                submitted_state=submitted_state,
                submitted_account_owner=submitted_account_owner,
            )
            if not self.retain_capture_images:
                self._cleanup_job_files(completed_job_id)
            self._log_duplicate_snapshot(exc.snapshot_hash, synchronous=False)
            return
        except StaleWindowContentError as exc:
            # A frame that changed between its first and verification capture
            # is ambiguous: the user may have changed chats, but WhoChat or a
            # transient WeChat surface may also have covered the title.  It is
            # not valid identity evidence and must never clear a confirmed
            # contact.  A genuine switch is detected by the next internally
            # stable frame through the verified title_pixels_changed path.
            self._discard(
                f"{getattr(exc, 'discard_reason', 'stale_window_content')}:{exc.snapshot_hash}",
                job_id=completed_job_id,
                submitted_state=submitted_state,
                submitted_account_owner=submitted_account_owner,
            )
            self._cleanup_job_files(completed_job_id)
            append_diagnostics_log("capture_pipeline", f"discarded job={completed_job_id} reason={getattr(exc, 'discard_reason', 'stale_window_content')} snapshot={exc.snapshot_hash}")
            return
        except StaleCaptureContextError as exc:
            self._discard(
                f"stale_capture_context:{exc.reason}",
                job_id=completed_job_id,
                submitted_state=submitted_state,
                submitted_account_owner=submitted_account_owner,
            )
            self._cleanup_job_files(completed_job_id)
            append_diagnostics_log("capture_pipeline", f"discarded stale_capture_context={exc.reason}")
            return
        except Exception as exc:
            self._discard(
                f"pipeline_failed:{exc}",
                job_id=completed_job_id,
                submitted_state=submitted_state,
                submitted_account_owner=submitted_account_owner,
            )
            self._cleanup_job_files(completed_job_id)
            append_diagnostics_log("capture_pipeline", f"failed error={exc}")
            return
        if result.job_id != self._latest_job_id:
            self._discard(
                f"stale_result:{result.job_id}",
                job_id=completed_job_id,
                submitted_state=submitted_state,
                submitted_account_owner=submitted_account_owner,
            )
            self._cleanup_job_files(result.job_id)
            append_diagnostics_log("capture_pipeline", f"discarded stale_result={result.job_id}")
            return
        try:
            self._publish_full_result(result, completed_job_id, submitted_state)
        except StaleCaptureContextError as exc:
            self._discard(f"stale_capture_context:{exc.reason}", job_id=completed_job_id,
                          submitted_state=submitted_state, submitted_account_owner=submitted_account_owner)
            self._cleanup_job_files(result.job_id)
            return
        if self.retain_capture_images:
            self._cleanup_retained_captures()
        else:
            self._cleanup_job_files(result.job_id)

    def _publish_full_result(self, result: PipelineResult, job_id: int, submitted_state: RuntimeState | None) -> None:
        with self._lifecycle_lock:
            self._assert_job_active(job_id, "before_full_publish")
            if submitted_state is not None:
                self._assert_context_current(submitted_state, "before_full_publish")
            if (isinstance(result.screen_proof, dict)
                    and result.screen_proof.get("negative_capture_evidence") is True
                    and result.screen_proof.get("valid") is False):
                # Negative geometry evidence is UI-only, not an observation.
                # result_ready is wired to identity/ingestion before the UI;
                # sending an empty rejected frame there would persist a false
                # conversation observation. Do not update last_result or sample
                # repositories on this separate, request-bound rejection path.
                reason = result.screen_proof.get("rejection_code", "capture_rejected")
                self._set_status(f"pipeline_rejected: job={job_id}, reason={reason}")
                append_diagnostics_log("capture_pipeline", f"rejected job={job_id} reason={reason}")
                self.capture_rejected.emit(result)
                return
            self.last_result = result
            self._record_capture_sample(result)
            self._set_status(f"pipeline_finished: job={result.job_id}, page={result.page.page_type.value}, messages={len(result.messages)}")
            append_diagnostics_log(
                "capture_pipeline",
                f"finished job={result.job_id} page={result.page.page_type.value} messages={len(result.messages)} "
                f"account_scope={result.account_scope_id or 'unbound'} instance={result.target_instance_id or '-'} "
                f"scope_reason={result.account_scope_reason or '-'} generation={result.window_generation}/{result.capture_generation} "
                f"title_ms={result.title_ocr_elapsed_ms if result.title_ocr_elapsed_ms is not None else '-'} "
                f"content_ms={result.content_ocr_elapsed_ms if result.content_ocr_elapsed_ms is not None else '-'} "
                f"capture_ms={result.capture_elapsed_ms if result.capture_elapsed_ms is not None else '-'} "
                f"verify_capture_ms={result.verify_capture_elapsed_ms if result.verify_capture_elapsed_ms is not None else '-'} "
                f"classify_ms={result.page_classification_elapsed_ms if result.page_classification_elapsed_ms is not None else '-'} "
                f"total_ms={result.total_elapsed_ms if result.total_elapsed_ms is not None else '-'} "
                f"warning={result.ocr_result.warning or '-'}",
            )
            self.result_ready.emit(result)

    def _log_duplicate_snapshot(self, snapshot_hash: str, *, synchronous: bool) -> None:
        """Rate-limit the expected steady-screen diagnostic.

        Exact duplicate frames are the normal state while a foreground chat is
        idle.  Persisting one line every adaptive poll would turn a safe 2 s
        change detector into a noisy disk writer, so retain one heartbeat per
        minute while still exposing every state transition through runtime
        status/signals.
        """
        now = time.monotonic()
        if now - self._last_duplicate_log_monotonic < 60.0:
            return
        self._last_duplicate_log_monotonic = now
        prefix = "sync_discarded" if synchronous else "discarded"
        append_diagnostics_log("capture_pipeline", f"{prefix} duplicate_snapshot={snapshot_hash}")

    def _record_verification_difference(
        self, job_id, state, layout, primary_rgb, verified_rgb, *, title_hash_matches,
    ) -> None:
        """Bounded diagnostic only; never grants or changes a capture decision.

        No screenshot, text, avatar, pixel payload or customer name is retained.
        A new explicit request gets at most three metadata samples. Exceptions
        here must not replace the original rejection or cause an AI request.
        """
        try:
            token = self._capture_request_tokens.get(job_id)
            if token is None:
                return
            with self._lifecycle_lock:
                # A retired worker may return after a new button request. It
                # must not reset the new request's bounded diagnostic counter.
                self._assert_job_current(job_id, state, "before_verification_diagnostic")
                if getattr(self, "_verification_diagnostic_token", None) != token:
                    self._verification_diagnostic_token = token
                    self._verification_diagnostic_count = 0
                if self._verification_diagnostic_count >= 3:
                    return
                self._verification_diagnostic_count += 1
                sample = self._verification_diagnostic_count
            metadata = _verification_difference_metadata(primary_rgb, verified_rgb, layout)
            metadata.update({
                "schema": 1, "job_id": job_id, "request_id": token[0], "sample": sample,
                "window_generation": state.window_generation,
                "capture_generation": state.capture_generation,
                "window_rect": list(layout.window_rect.as_tuple()),
                "input_boundary_verified": layout.input_boundary_verified,
                "title_ahash_matches": bool(title_hash_matches),
                "rejection": "current_screen_body_changed" if title_hash_matches else "stale_window_content",
            })
            append_diagnostics_log("capture_pipeline", "verification_difference " + json.dumps(
                metadata, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
            ))
        except Exception:
            # Observability failure must never alter the existing fail-closed path.
            pass

    def _record_capture_sample(self, result: PipelineResult) -> None:
        if self.capture_samples is None:
            return
        try:
            self.capture_samples.append(
                job_id=result.job_id,
                hwnd=result.hwnd,
                target_app=result.target_app,
                app_label=result.app_label,
                snapshot_hash=result.snapshot_hash,
                image_path=str(result.image_path),
                ocr_image_path=str(result.ocr_image_path),
                crop_rect_json=json.dumps(result.crop_rect.as_tuple() if result.crop_rect else None, separators=(",", ":")),
                title_ocr_image_path=str(result.title_ocr_image_path) if result.title_ocr_image_path else "",
                title_crop_rect_json=json.dumps(result.title_crop_rect.as_tuple() if result.title_crop_rect else None, separators=(",", ":")),
                title_ocr_elapsed_ms=result.title_ocr_elapsed_ms,
                content_ocr_elapsed_ms=result.content_ocr_elapsed_ms,
                total_elapsed_ms=result.total_elapsed_ms,
                capture_elapsed_ms=result.capture_elapsed_ms,
                layout_elapsed_ms=result.layout_elapsed_ms,
                title_input_elapsed_ms=result.title_input_elapsed_ms,
                content_input_elapsed_ms=result.content_input_elapsed_ms,
                title_preprocess_elapsed_ms=result.title_preprocess_elapsed_ms,
                title_inference_elapsed_ms=result.title_inference_elapsed_ms,
                title_postprocess_elapsed_ms=result.title_postprocess_elapsed_ms,
                content_preprocess_elapsed_ms=result.content_preprocess_elapsed_ms,
                content_inference_elapsed_ms=result.content_inference_elapsed_ms,
                content_postprocess_elapsed_ms=result.content_postprocess_elapsed_ms,
                verify_capture_elapsed_ms=result.verify_capture_elapsed_ms,
                page_classification_elapsed_ms=result.page_classification_elapsed_ms,
                title_cache_hit=result.title_cache_hit,
                ocr_engine=result.ocr_result.engine,
                ocr_warning=result.ocr_result.warning or "",
                page_type=result.page.page_type.value,
                page_confidence=result.page.confidence,
                message_count=len(result.messages),
                retained_image=self.retain_capture_images,
            )
        except Exception as exc:
            append_diagnostics_log("capture_pipeline", f"capture_sample_record_failed job={result.job_id} error={exc}")

    def _cleanup_job_files(self, job_id: int | None) -> None:
        if job_id is None:
            return
        directory = (app_data_dir() / "capture").resolve()
        for suffix in ("", "_title", "_content", "_verify"):
            _safe_unlink_capture(directory / f"job_{job_id}{suffix}.png")

    def _cleanup_retained_captures(self) -> None:
        now = time.monotonic()
        if now - self._last_capture_cleanup_monotonic < 3600:
            return
        self._last_capture_cleanup_monotonic = now
        directory = (app_data_dir() / "capture").resolve()
        if not directory.exists():
            return
        cutoff = time.time() - self.capture_retention_days * 86400
        for path in directory.glob("job_*.png"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    _safe_unlink_capture(path)
            except OSError:
                continue

    def _discard(
        self,
        reason: str,
        *,
        job_id: int | None = None,
        submitted_state: RuntimeState | None = None,
        submitted_account_owner: tuple[str, str] | None = None,
    ) -> None:
        self.last_discard_reason = reason
        self.result_discarded.emit(reason)
        state = submitted_state or (self._submitted_states.get(job_id) if job_id is not None else None)
        if job_id is not None and state is not None:
            owner = submitted_account_owner if submitted_account_owner is not None else self._job_account_owners.get(job_id, ("", ""))
            snapshot_hash = reason.partition(":")[2] if ":" in reason else ""
            self.result_discarded_detail.emit(
                PipelineDiscardEvent(
                    reason=reason,
                    job_id=job_id,
                    hwnd=state.window.hwnd,
                    process_id=state.window.process_id,
                    window_generation=state.window_generation,
                    capture_generation=state.capture_generation,
                    account_scope_id=owner[0] or None,
                    target_instance_id=owner[1],
                    account_identity_epoch=state.account_identity_epoch,
                    account_identity_runtime_key=state.account_identity_runtime_key,
                    snapshot_hash=snapshot_hash,
                )
            )

    def _set_status(self, status: str) -> None:
        self.last_status = status
        self.status_changed.emit(status)

    def _set_running_stage(self, job_id: int, stage: str) -> None:
        if self._running_job_id == job_id and job_id not in self._expired_job_ids:
            self._running_stage = stage

    def _has_seen_snapshot_hash(self, snapshot_hash: str) -> bool:
        with self._snapshot_hash_cache_lock:
            proof_state = self._screen_proof_cache.get(snapshot_hash)
            if proof_state is not None and proof_state[0] is not None and time.monotonic() >= proof_state[0]:
                self._snapshot_hash_cache.pop(snapshot_hash, None)
                self._snapshot_hash_owners.pop(snapshot_hash, None)
                if self._last_snapshot_hash == snapshot_hash:
                    self._last_snapshot_hash = None
                return False
            if self._last_snapshot_hash == snapshot_hash:
                return True
            if snapshot_hash not in self._snapshot_hash_cache:
                return False
            self._snapshot_hash_cache.move_to_end(snapshot_hash)
            return True

    def _cached_verified_content_ocr(self, key, image_path) -> OcrResult | None:
        if key is None:
            return None
        now = time.monotonic()
        for old_key, (deadline, _result) in tuple(self._verified_content_ocr_cache.items()):
            if deadline <= now:
                self._verified_content_ocr_cache.pop(old_key, None)
        cached = self._verified_content_ocr_cache.get(key)
        if cached is None:
            return None
        self._verified_content_ocr_cache.move_to_end(key)
        return replace(cached[1], source_image=str(image_path), boxes=list(cached[1].boxes))

    def _remember_verified_content_ocr(self, key, result: OcrResult) -> None:
        if key is None or result.warning or not result.boxes or len(result.boxes) > 512:
            return
        self._verified_content_ocr_cache[key] = (
            time.monotonic() + 8.0, replace(result, boxes=list(result.boxes)),
        )
        self._verified_content_ocr_cache.move_to_end(key)
        while len(self._verified_content_ocr_cache) > 2:
            self._verified_content_ocr_cache.popitem(last=False)

    def _record_screen_proof_cache(self, snapshot_hash: str, proof: dict, *, job_id: int) -> None:
        """Expire valid captures at their real deadline; one bounded invalid recapture.

        This only grants another OCR read, never an HTTP retry or timeline
        promotion. The lifecycle is part of snapshot_hash. A manual cache
        reset explicitly permits another bounded attempt.
        """
        with self._snapshot_hash_cache_lock:
            if self._snapshot_hash_owners.get(snapshot_hash) != job_id:
                return
            old = self._screen_proof_cache.get(snapshot_hash, (None, 0))
            if proof.get("valid") is True:
                try:
                    expires = datetime.fromisoformat(str(proof["expires_at"]).replace("Z", "+00:00"))
                    remaining = max(0.0, (expires - datetime.now(timezone.utc)).total_seconds())
                    deadline, attempts = time.monotonic() + remaining, old[1]
                except (ValueError, TypeError, KeyError):
                    deadline, attempts = None, min(2, old[1] + 1)
            else:
                deadline, attempts = None, min(2, old[1] + 1)
            self._screen_proof_cache[snapshot_hash] = (deadline, attempts)
            self._screen_proof_cache.move_to_end(snapshot_hash)
            while len(self._screen_proof_cache) > min(128, self._snapshot_hash_cache_size):
                self._screen_proof_cache.popitem(last=False)
            if proof.get("valid") is not True and attempts == 1:
                self._snapshot_hash_cache.pop(snapshot_hash, None)
                self._snapshot_hash_owners.pop(snapshot_hash, None)
                if self._last_snapshot_hash == snapshot_hash:
                    self._last_snapshot_hash = None

    def _remember_snapshot_hash(self, snapshot_hash: str, *, job_id: int = 0) -> None:
        with self._snapshot_hash_cache_lock:
            self._last_snapshot_hash = snapshot_hash
            self._snapshot_hash_cache[snapshot_hash] = None
            self._snapshot_hash_owners[snapshot_hash] = job_id
            self._snapshot_hash_cache.move_to_end(snapshot_hash)
            while len(self._snapshot_hash_cache) > self._snapshot_hash_cache_size:
                removed_hash, _ = self._snapshot_hash_cache.popitem(last=False)
                self._snapshot_hash_owners.pop(removed_hash, None)

    def _forget_job_snapshot_hash(self, job_id: int) -> None:
        snapshot_hash = self._job_snapshot_hashes.pop(job_id, "")
        if not snapshot_hash:
            return
        with self._snapshot_hash_cache_lock:
            if self._snapshot_hash_owners.get(snapshot_hash) != job_id:
                return
            self._snapshot_hash_cache.pop(snapshot_hash, None)
            self._snapshot_hash_owners.pop(snapshot_hash, None)
            if self._last_snapshot_hash == snapshot_hash:
                self._last_snapshot_hash = None


class DuplicateSnapshotError(RuntimeError):
    def __init__(self, snapshot_hash: str) -> None:
        super().__init__(snapshot_hash)
        self.snapshot_hash = snapshot_hash


class StaleWindowContentError(RuntimeError):
    def __init__(self, snapshot_hash: str) -> None:
        super().__init__(snapshot_hash)
        self.snapshot_hash = snapshot_hash


class CurrentScreenBodyChangedError(StaleWindowContentError):
    discard_reason = "current_screen_body_changed"


class StaleCaptureContextError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _heavy_ocr_min_interval_ms() -> int:
    try:
        return max(5000, int(os.environ.get("WHOCHAT_HEAVY_OCR_MIN_INTERVAL_MS", "5000")))
    except ValueError:
        return 5000


def _snapshot_hash_cache_size() -> int:
    try:
        return min(512, max(1, int(os.environ.get("WHOCHAT_SNAPSHOT_HASH_CACHE_SIZE", "64"))))
    except ValueError:
        return 64


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _safe_unlink_capture(path: Path) -> None:
    root = (app_data_dir() / "capture").resolve()
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return
    try:
        resolved.unlink(missing_ok=True)
    except OSError:
        return


def _local_preread_ocr(pipeline, kind, state, account, layout, frame, engine, image_path):
    """Reuse computation only; never confer customer/source/dispatch authority."""
    cache = getattr(pipeline, "local_preread_cache", None)
    if cache is None:
        return None
    try:
        result = cache.lookup(kind, state, account, layout, frame, engine)
        if result is None:
            return None
        # The cache owns independent copies; source paths never refer to the
        # background temporary image and no timestamp/proof is copied here.
        return replace(result, source_image=str(image_path))
    except Exception as exc:
        append_diagnostics_log("capture_pipeline", f"local_preread_miss kind={kind} error={type(exc).__name__}")
        return None


def _remember_local_preread_ocr(pipeline, state, account, layout, frame, engine, result):
    """Seed only raw OCR after the worker's unchanged native/pixel guards."""
    cache = getattr(pipeline, "local_preread_cache", None)
    remember = getattr(cache, "remember_content", None)
    if not callable(remember):
        return
    try:
        remember(state, account, layout, frame, engine, result)
    except Exception:
        # Optional computation reuse cannot change the captured result.
        return


def _prepare_ocr_input(image_path: Path, layout: LayoutRegions, *, materialize: bool = True,
                       frame_image: Image.Image | None = None) -> tuple[Path, LayoutRegions, tuple[int, int], Rect | None]:
    selected = bool(getattr(layout, "manual_selection_id", ""))
    if not selected and os.environ.get("WHOCHAT_OCR_CROP_CONTENT", "1") == "0":
        return image_path, _layout_relative_to_image(layout, layout.window_rect.left, layout.window_rect.top), (
            layout.window_rect.left,
            layout.window_rect.top,
        ), None
    crop_rect = (_clip_rect(layout.message_rect, layout.window_rect) if selected else
                 _clip_rect(Rect(layout.content_rect.left, layout.title_rect.bottom, layout.content_rect.right, layout.content_rect.bottom), layout.window_rect))
    if not selected and (crop_rect.width < 120 or crop_rect.height < 160):
        return image_path, _layout_relative_to_image(layout, layout.window_rect.left, layout.window_rect.top), (
            layout.window_rect.left,
            layout.window_rect.top,
        ), None
    relative_crop = (
        crop_rect.left - layout.window_rect.left,
        crop_rect.top - layout.window_rect.top,
        crop_rect.right - layout.window_rect.left,
        crop_rect.bottom - layout.window_rect.top,
    )
    crop_path = image_path.with_name(f"{image_path.stem}_content{image_path.suffix}")
    if materialize:
        if frame_image is not None:
            frame_image.crop(relative_crop).save(crop_path)
        else:
            with Image.open(image_path) as image:
                image.crop(relative_crop).save(crop_path)
    crop_layout = _layout_relative_to_image(layout, crop_rect.left, crop_rect.top)
    if materialize and _verbose_capture_diagnostics():
        append_diagnostics_log(
            "capture_pipeline",
            f"ocr_crop source={image_path.name} crop={crop_rect.as_tuple()} size={crop_rect.width}x{crop_rect.height}",
        )
    return crop_path, crop_layout, (crop_rect.left, crop_rect.top), crop_rect


def _prepare_title_ocr_input(image_path: Path, layout: LayoutRegions, *, materialize: bool = True,
                             frame_image: Image.Image | None = None) -> tuple[Path, LayoutRegions, tuple[int, int], Rect | None]:
    if os.environ.get("WHOCHAT_OCR_CROP_CONTENT", "1") == "0":
        return image_path, _layout_relative_to_image(layout, layout.window_rect.left, layout.window_rect.top), (
            layout.window_rect.left,
            layout.window_rect.top,
        ), None
    crop_rect = _clip_rect(_title_crop_rect(layout), layout.window_rect)
    if crop_rect.width < 120 or crop_rect.height < 28:
        return image_path, _layout_relative_to_image(layout, layout.window_rect.left, layout.window_rect.top), (
            layout.window_rect.left,
            layout.window_rect.top,
        ), None
    relative_crop = (
        crop_rect.left - layout.window_rect.left,
        crop_rect.top - layout.window_rect.top,
        crop_rect.right - layout.window_rect.left,
        crop_rect.bottom - layout.window_rect.top,
    )
    crop_path = image_path.with_name(f"{image_path.stem}_title{image_path.suffix}")
    if materialize:
        if frame_image is not None:
            frame_image.crop(relative_crop).save(crop_path)
        else:
            with Image.open(image_path) as image:
                image.crop(relative_crop).save(crop_path)
    crop_layout = _layout_relative_to_image(layout, crop_rect.left, crop_rect.top)
    if materialize and _verbose_capture_diagnostics():
        append_diagnostics_log(
            "capture_pipeline",
            f"ocr_title_crop source={image_path.name} crop={crop_rect.as_tuple()} size={crop_rect.width}x{crop_rect.height}",
        )
    return crop_path, crop_layout, (crop_rect.left, crop_rect.top), crop_rect


def _title_crop_rect(layout: LayoutRegions) -> Rect:
    """Title band safe for OCR.

    Keep a small, proportional left safety margin for antialiased leading
    strokes, while excluding the right-most window/search/action controls.  The
    true conversation boundary is measured by ``refine_layout_from_image``;
    this crop no longer compensates for a bad boundary by swallowing a large
    part of the chat list.
    """
    window_width = layout.window_rect.width
    pad = max(6, round(window_width * 0.012))
    left = (layout.title_rect.left if getattr(layout, "manual_selection_id", "") else
            max(layout.window_rect.left, layout.title_rect.left - pad))
    usable_width = max(160, round(layout.title_rect.width * 0.72))
    right = min(layout.title_rect.right, layout.title_rect.left + usable_width)
    if right - left < 120:
        left = max(layout.window_rect.left, layout.title_rect.left)
        right = min(layout.title_rect.right, left + 120)
    return Rect(left, layout.title_rect.top, right, layout.title_rect.bottom)


def title_identity_screen_rect(layout: LayoutRegions) -> Rect:
    """Stable screen-space ROI used for title identity fingerprints.

    It matches the text-bearing width of the title OCR crop, but clips its
    left padding back to the true conversation column so chat-list repainting
    cannot look like a contact switch.  The right action/control area is
    deliberately excluded for the same reason.
    """
    ocr_rect = _clip_rect(_title_crop_rect(layout), layout.window_rect)
    left = max(layout.title_rect.left, ocr_rect.left)
    right = min(layout.title_rect.right, ocr_rect.right)
    if right - left < 120:
        left = layout.title_rect.left
        right = min(layout.title_rect.right, left + max(120, layout.title_rect.width))
    return Rect(left, layout.title_rect.top, right, layout.title_rect.bottom)


def _ocr_input_hash(
    title_image_path: Path,
    content_image_path: Path,
    source_image_path: Path | None = None,
    title_band: Rect | None = None,
) -> str:
    # Fingerprint the conversation column only.  The title OCR crop is padded
    # leftwards (see ``_title_crop_rect``) and that padding overlaps the chat
    # list, so hashing it would treat a chat-list repaint as a new conversation.
    if source_image_path is not None and title_band is not None:
        title_hash = image_hash_region(source_image_path, title_band.as_tuple(), size=8)
    else:
        title_hash = image_hash(title_image_path, size=8)
    content_hash = image_hash(content_image_path, size=16)
    payload = f"title:{title_hash}|content:{content_hash}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:32]


def _title_visual_digest(image_path: Path, rect: Rect, *, frame_image: Image.Image | None = None) -> str:
    """Same 8x8 aHash as image_hash_region, preserving the borrowed frame's mode."""
    if frame_image is None:
        return image_hash_region(image_path, rect.as_tuple(), size=8)
    width, height = frame_image.size
    left, top = max(0, min(rect.left, width)), max(0, min(rect.top, height))
    right, bottom = max(left, min(rect.right, width)), max(top, min(rect.bottom, height))
    gray = frame_image.convert("L")
    if right > left and bottom > top:
        gray = gray.crop((left, top, right, bottom))
    pixels = list(gray.resize((8, 8)).getdata())
    average = sum(pixels) / max(1, len(pixels))
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"


def _exact_region_digest(image_path: Path, rect: Rect, *, frame_image: Image.Image | None = None) -> str:
    """Cryptographic pixel digest used only for safe title OCR reuse.

    Perceptual hashes remain appropriate for unchanged-screen suppression, but
    identity reuse requires exact pixels plus the full window/account/generation
    key assembled by the pipeline.
    """
    if frame_image is None:
        with Image.open(image_path) as image:
            frame_image = image.convert("RGB")
    rgb = frame_image if frame_image.mode == "RGB" else frame_image.convert("RGB")
    bounds = (0, 0, rgb.width, rgb.height)
    clipped = (
        max(bounds[0], rect.left), max(bounds[1], rect.top),
        min(bounds[2], rect.right), min(bounds[3], rect.bottom),
    )
    region = rgb.crop(clipped)
    payload = region.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _normalized_title_observation(result: OcrResult | None) -> str:
    if result is None:
        return ""
    boxes = sorted(
        (
            box
            for box in result.boxes
            if box.region == OcrRegion.TITLE and str(box.text or "").strip()
        ),
        key=lambda box: (box.rect.top, box.rect.left),
    )
    text = " ".join(str(box.text).strip() for box in boxes)
    text = unicodedata.normalize("NFKC", " ".join(text.split())).casefold()
    return re.sub(r"\s*[（(]\s*\d+\s*[）)]\s*$", "", text).strip()


def _content_ocr_reuse_key(state, account, layout, title_digest, conversation_digest, engine):
    """Exact frame equivalence within one owner/lifecycle, never identity proof."""
    if (account is None or account.actions_allowed is not True
            or not account.account_scope_id or not account.target_instance_id
            or not title_digest or not conversation_digest or not layout.input_boundary_verified):
        return None
    return (
        state.window.hwnd, state.window.process_id, state.window_generation, state.capture_generation,
        account.account_scope_id, account.target_instance_id, account.identity_epoch, account.runtime_key,
        state.account_identity_epoch, state.account_identity_runtime_key,
        layout.window_rect.as_tuple(), layout.title_rect.as_tuple(), layout.message_rect.as_tuple(),
        layout.input_rect.as_tuple(), layout.bubble_profile, title_digest, conversation_digest,
        id(engine), str(getattr(engine, "name", "")),
    )


def _assert_native_client_geometry(state: RuntimeState, stage: str) -> None:
    """Keep the OS exclusion bound to the exact captured HWND/PID/frame.

    No pixel-derived mask or tolerance grants permission. Missing native
    evidence or a changed client frame discards the job, before publication.
    """
    from whochat.platform.window_tracker import read_client_screen_rect

    try:
        window = state.window
        expected = window.client_rect.as_tuple()
        actual = read_client_screen_rect(
            window.hwnd, window.rect.as_tuple(), expected_process_id=window.process_id,
        )
        if actual != expected:
            raise ValueError("client_geometry_changed")
    except Exception as exc:
        raise StaleCaptureContextError(f"{stage}:native_client_geometry_unavailable_or_changed") from exc


def _noncontent_scrollbar_rail(rgb: Image.Image, layout: LayoutRegions, message: Rect):
    """Prove a tiny native WeChat scrollbar gutter, not a generic side mask.

    We never mask avatar columns, bubble/image/emoji rectangles, arbitrary
    changed-pixel bounding boxes, or a percentage of the conversation. The
    Legacy layouts retain their original final-eight-pixel window-edge proof.
    Native-client layouts use a separate fixed client-edge slot, surrounded
    by exact blank guards; native geometry must already have been checked by
    the caller. A client rectangle alone is never a screen/identity lease.
    A painted rail must consist solely of one narrow neutral vertical thumb;
    coloured content, multiple shapes, horizontal strokes or an occupied
    guard preserve every pixel. Unknown layouts keep the original digest.
    """
    try:
        client = getattr(layout, "client_rect", None)
        if client is not None:
            outer, absolute = layout.window_rect, layout.message_rect
            if (layout.target_app.value != "wechat"
                    or layout.bubble_profile not in {"auto", "wechat_green"}
                    or layout.input_boundary_verified is not True
                    or not all(isinstance(rect, Rect) for rect in (outer, client, absolute, message))
                    or any(type(value) is not int for rect in (outer, client, absolute, message)
                           for value in rect.as_tuple())
                    or rgb.mode != "RGB" or rgb.size != (outer.width, outer.height)
                    or outer.width <= 0 or outer.height <= 0
                    or not (outer.left <= client.left < client.right <= outer.right
                            and outer.top <= client.top < client.bottom <= outer.bottom)
                    or not (client.left <= absolute.left < absolute.right == client.right
                            and client.top <= absolute.top < absolute.bottom <= client.bottom)
                    or message.as_tuple() != (absolute.left - outer.left, absolute.top - outer.top,
                                              absolute.right - outer.left, absolute.bottom - outer.top)
                    or message.width < 240 or message.height < 100
                    or message.left < 0 or message.top < 0
                    or message.right > rgb.width or message.bottom > rgb.height):
                return None
            if absolute.right == outer.right:
                # Native client/outer geometry has already passed the strict
                # checks above. With no right frame inset, keep the pre-existing
                # edge-renderer contract exactly, not the inset renderer below.
                # This local comparison-only projection cannot alter a source
                # proof or the caller's native geometry/account validation.
                return _noncontent_scrollbar_rail(rgb, replace(layout, client_rect=None), message)
            # A fixed, narrow native-client slot, NOT a changed-pixel bbox or
            # a proportional side mask. Both blank and painted states normalize
            # exactly this same ROI. Everything outside it remains byte exact.
            slot = (message.right - 16, message.top, message.right - 4, message.bottom)
            left_guard = rgb.crop((message.right - 20, message.top,
                                   message.right - 16, message.bottom))
            right_guard = rgb.crop((message.right - 4, message.top,
                                    message.right, message.bottom))
            rail = rgb.crop(slot)
            try:
                # The verified editor boundary can include the top few rows of
                # its rounded border. They are NOT scrollbar pixels. Prove the
                # long blank gutter first, then trim only this tiny bottom tail
                # from the normalization ROI. Guards and the entire tail stay
                # byte-exact in the conversation digest, even when static.
                # Never mask arbitrary occupied rows or move the chat boundary.
                safe_height = left_guard.height
                background_rows = left_guard.crop((0, 0, left_guard.width, safe_height - 8))
                reference = background_rows.getcolors(1)
                background_rows.close()
                if reference and len(reference) == 1:
                    reference_color = reference[0][1]
                    occupied = [y for y in range(safe_height - 8, safe_height)
                                if any(left_guard.getpixel((x, y)) != reference_color
                                       for x in range(left_guard.width))
                                or any(right_guard.getpixel((x, y)) != reference_color
                                       for x in range(right_guard.width))]
                    if occupied:
                        tail_start = min(occupied)
                        # A neutral editor rim is allowed to terminate the
                        # gutter, not authorize erasing text or coloured media.
                        tail_values = [guard.getpixel((x, y))
                                       for guard in (left_guard, right_guard)
                                       for y in range(tail_start, safe_height)
                                       for x in range(guard.width)]
                        if any(max(value) - min(value) > 4
                               or any(not 0 <= bg - channel <= 48
                                      for bg, channel in zip(reference_color, value))
                               for value in tail_values):
                            return None
                        trimmed = [part.crop((0, 0, part.width, tail_start))
                                   for part in (left_guard, right_guard, rail)]
                        for part in (left_guard, right_guard, rail):
                            part.close()
                        left_guard, right_guard, rail = trimmed
                        slot = (slot[0], slot[1], slot[2], slot[1] + tail_start)
                colors = left_guard.getcolors(1)
                if not colors or len(colors) != 1:
                    return None
                background = colors[0][1]
                if (max(background) - min(background) > 4
                        or min(background) < 192
                        or right_guard.getcolors(1) != [(right_guard.width * right_guard.height, background)]):
                    return None
                foreground = []
                pixels = rail.load()
                for y in range(rail.height):
                    for x in range(rail.width):
                        value = pixels[x, y]
                        if value == background:
                            continue
                        # This light-theme contract deliberately does not infer
                        # a dark-theme rail or permit coloured/near-background
                        # marks. Unknown rendering is left fully in the digest.
                        if (max(value) - min(value) > 4
                                or any(not 8 <= bg - channel <= 144
                                       for bg, channel in zip(background, value))):
                            return None
                        foreground.append((x, y))
                if not foreground:
                    return slot, background
                xs, ys = zip(*foreground)
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                width, height = right - left + 1, bottom - top + 1
                if (not 4 <= width <= 8 or height < max(32, width * 8)
                        or height > int(message.height * .85)
                        or left < 1 or right >= rail.width - 1
                        or top < 3 or bottom >= message.height - 3):
                    return None
                rows = {y: set() for y in range(top, bottom + 1)}
                for x, y in foreground:
                    rows[y].add(x)
                stem = set(range(left, right + 1))
                # A single uninterrupted solid stem, with at most four rows of
                # symmetric rounded end caps. Letters, multiple shapes, holes,
                # horizontal strokes and repeated vertical text are not a rail.
                stem_color = pixels[left + width // 2, top + 4]
                if not (rows[top] != stem and rows[bottom] != stem):
                    return None
                for y, columns in rows.items():
                    if not columns:
                        return None
                    lo, hi = min(columns), max(columns)
                    if columns != set(range(lo, hi + 1)) or lo - left != right - hi:
                        return None
                    if top + 4 <= y <= bottom - 4:
                        if columns != stem or any(pixels[x, y] != stem_color for x in columns):
                            return None
                    elif any(pixels[x, y] != stem_color for x in columns):
                        # Antialiasing is not guessed away. A different renderer
                        # remains unmatched until it has its own proven fixture.
                        return None
                upper = [len(rows[y]) for y in range(top, top + 5)]
                lower = [len(rows[y]) for y in range(bottom - 4, bottom + 1)]
                if upper != sorted(upper) or lower != sorted(lower, reverse=True):
                    return None
                return slot, background
            finally:
                left_guard.close()
                right_guard.close()
                rail.close()
        if (layout.target_app.value != "wechat" or layout.bubble_profile not in {"auto", "wechat_green"}
                or not layout.input_boundary_verified or message.width < 240 or message.height < 100
                or layout.message_rect.right != layout.window_rect.right
                or message.right != rgb.width or message.left < 0 or message.top < 0 or message.bottom > rgb.height):
            return None
        rail_left = message.right - 8
        guard = rgb.crop((rail_left - 4, message.top, rail_left, message.bottom))
        colors = guard.getcolors(1)
        if not colors or len(colors) != 1:
            return None
        background = colors[0][1]
        if max(background) - min(background) > 4:
            return None
        # The outermost pixel is a window border, not the scroll thumb.
        rail = rgb.crop((rail_left, message.top, message.right - 1, message.bottom))
        foreground = []
        for y in range(rail.height):
            for x in range(rail.width):
                value = rail.getpixel((x, y))
                if value == background:
                    continue
                if max(value) - min(value) > 4 or not 8 <= abs(value[0] - background[0]) <= 144:
                    return None
                foreground.append((x, y))
        if foreground:
            xs, ys = zip(*foreground)
            left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
            if (right - left + 1 > 6 or bottom - top + 1 < 24
                    or bottom - top + 1 > round(message.height * .85)
                    or len(set(ys)) != bottom - top + 1):
                return None
            # A continuous substantial stem rules out punctuation/emoji dots.
            rows = {y: set() for y in range(top, bottom + 1)}
            for x, y in foreground:
                rows[y].add(x)
            stem = set(range(left, right + 1))
            if sum(columns == stem for columns in rows.values()) < (bottom - top + 1) * .80:
                return None
        border = rgb.crop((message.right - 1, message.top, message.right, message.bottom)).getcolors(1)
        if not border or max(border[0][1]) - min(border[0][1]) > 4:
            return None
        return (rail_left, message.top, message.right, message.bottom), background
    except (ValueError, TypeError, AttributeError, IndexError):
        return None


def _verification_difference_metadata(primary: Image.Image, verified: Image.Image, layout: LayoutRegions) -> dict:
    """Locate rejected pixel differences without persisting image content.

    Inputs are borrowed immutable RGB frames already captured by the job. The
    production digest and its native-rail proof are not replaced by this helper.
    """
    metadata = {"primary_size": list(primary.size), "verified_size": list(verified.size)}
    if getattr(layout, "client_rect", None) is not None:
        metadata["client_rect"] = list(layout.client_rect.as_tuple())
    if primary.size != verified.size or primary.mode != "RGB" or verified.mode != "RGB":
        metadata["comparable"] = False
        return metadata
    metadata["comparable"] = True
    title = _shift_rect(title_identity_screen_rect(layout), -layout.window_rect.left, -layout.window_rect.top)
    messages = _shift_rect(layout.message_rect, -layout.window_rect.left, -layout.window_rect.top)
    rails = (_noncontent_scrollbar_rail(primary, layout, messages),
             _noncontent_scrollbar_rail(verified, layout, messages))
    metadata["primary_rail"] = list(rails[0][0]) if rails[0] else None
    metadata["verified_rail"] = list(rails[1][0]) if rails[1] else None
    for name, rect in (("title", title), ("messages", messages)):
        bounds = (max(0, rect.left), max(0, rect.top), min(primary.width, rect.right), min(primary.height, rect.bottom))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            metadata[name] = {"comparable": False, "bounds": list(bounds)}
            continue
        owned = []
        try:
            first, second = primary.crop(bounds), verified.crop(bounds)
            owned.extend((first, second))
            difference = ImageChops.difference(first, second)
            channels = difference.split()
            owned.extend((difference, *channels))
            rg = ImageChops.lighter(channels[0], channels[1])
            mask = ImageChops.lighter(rg, channels[2])
            owned.extend((rg, mask))
            box = mask.getbbox()
            summary = {
                "bounds": list(bounds), "raw_equal": box is None,
                "changed_pixels": mask.width * mask.height - mask.histogram()[0],
                "changed_bbox": [box[0] + bounds[0], box[1] + bounds[1],
                                 box[2] + bounds[0], box[3] + bounds[1]] if box else None,
                "max_channel_delta": mask.getextrema()[1],
            }
            if name == "messages":
                # Only local crops are painted; the parser still sees raw RGB.
                for crop, rail in zip((first, second), rails):
                    if rail:
                        left, top, right, bottom = rail[0]
                        crop.paste(rail[1], (left - bounds[0], top - bounds[1], right - bounds[0], bottom - bounds[1]))
                summary["normalized_equal"] = first.tobytes() == second.tobytes()
            else:
                summary["normalized_equal"] = box is None
            metadata[name] = summary
        finally:
            for image in reversed(owned):
                image.close()
    return metadata


def _exact_conversation_digest(image_path: Path, layout: LayoutRegions, *, frame_image: Image.Image | None = None) -> str:
    """Exact visible message/identity digest except a proved native rail.

    The input editor is intentionally excluded: a blinking caret or unsent
    draft must not trigger OCR or AI. Text, media, emoji and avatar pixels stay
    exact. No perceptual-distance threshold can swallow a short message. The
    narrow native scrollbar exemption requires a separate geometry proof.
    """
    title = _shift_rect(title_identity_screen_rect(layout), -layout.window_rect.left, -layout.window_rect.top)
    messages = _shift_rect(layout.message_rect, -layout.window_rect.left, -layout.window_rect.top)
    if frame_image is None:
        with Image.open(image_path) as image:
            frame_image = image.convert("RGB")
    rgb = frame_image if frame_image.mode == "RGB" else frame_image.convert("RGB")
    rail = _noncontent_scrollbar_rail(rgb, layout, messages)
    if rail is not None:
        # Scrollbar normalization must never overwrite the shared raw pixels
        # later used to parse messages, photos or account/avatar evidence.
        rgb = rgb.copy()
        rgb.paste(rail[1], rail[0])
    payload = b""
    for rect in (title, messages):
        clipped = (
            max(0, rect.left), max(0, rect.top),
            min(rgb.width, rect.right), min(rgb.height, rect.bottom),
        )
        region = rgb.crop(clipped)
        payload += f"{region.width}x{region.height}|".encode("ascii") + region.tobytes()
    return hashlib.sha256(payload).hexdigest()[:32]


def _merge_ocr_results(title: OcrResult, content: OcrResult) -> OcrResult:
    warnings = [value for value in [title.warning, content.warning] if value]
    boxes = _dedupe_ocr_boxes([*title.boxes, *content.boxes])
    return OcrResult(
        boxes=boxes,
        source_image=content.source_image,
        engine=content.engine if content.engine == title.engine else f"{title.engine}+{content.engine}",
        warning="; ".join(warnings) if warnings else None,
    )


def _dedupe_ocr_boxes(boxes: list[OcrTextBox]) -> list[OcrTextBox]:
    result: list[OcrTextBox] = []
    seen: set[tuple[str, tuple[int, int, int, int], int]] = set()
    for box in boxes:
        key = (" ".join(box.text.split()), box.rect.as_tuple(), round(box.confidence * 100))
        if key in seen:
            continue
        seen.add(key)
        result.append(box)
    return result


def _layout_relative_to_image(layout: LayoutRegions, origin_left: int, origin_top: int) -> LayoutRegions:
    return LayoutRegions(
        target_app=layout.target_app,
        bubble_profile=layout.bubble_profile,
        window_rect=_shift_rect(layout.window_rect, -origin_left, -origin_top),
        nav_rect=_shift_rect(layout.nav_rect, -origin_left, -origin_top),
        chat_list_rect=_shift_rect(layout.chat_list_rect, -origin_left, -origin_top),
        content_rect=_shift_rect(layout.content_rect, -origin_left, -origin_top),
        title_rect=_shift_rect(layout.title_rect, -origin_left, -origin_top),
        message_rect=_shift_rect(layout.message_rect, -origin_left, -origin_top),
        input_rect=_shift_rect(layout.input_rect, -origin_left, -origin_top),
        confidence=layout.confidence,
        source=layout.source,
        reason=layout.reason,
        client_rect=(_shift_rect(layout.client_rect, -origin_left, -origin_top)
                     if getattr(layout, "client_rect", None) is not None else None),
        dpi_scale=getattr(layout, "dpi_scale", 1.0),
        input_boundary_verified=bool(getattr(layout, "manual_selection_id", "")) and getattr(layout, "input_boundary_verified", False),
        manual_selection_id=getattr(layout, "manual_selection_id", ""),
    )


def _offset_ocr_result(
    result: OcrResult,
    offset: tuple[int, int],
    source_image: Path | str | None = None,
) -> OcrResult:
    left, top = offset
    if left == 0 and top == 0:
        if source_image is None:
            return result
        return OcrResult(
            boxes=result.boxes,
            source_image=str(source_image),
            engine=result.engine,
            warning=result.warning,
        )
    return OcrResult(
        boxes=[
            OcrTextBox(
                text=box.text,
                rect=_shift_rect(box.rect, left, top),
                confidence=box.confidence,
                region=box.region,
                source=box.source,
            )
            for box in result.boxes
        ],
        source_image=str(source_image) if source_image is not None else result.source_image,
        engine=result.engine,
        warning=result.warning,
    )


def _clip_rect(rect: Rect, bounds: Rect) -> Rect:
    return Rect(
        max(rect.left, bounds.left),
        max(rect.top, bounds.top),
        min(rect.right, bounds.right),
        min(rect.bottom, bounds.bottom),
    )


def _shift_rect(rect: Rect, left: int, top: int) -> Rect:
    return Rect(rect.left + left, rect.top + top, rect.right + left, rect.bottom + top)


def _verbose_capture_diagnostics() -> bool:
    return os.environ.get("WHOCHAT_VERBOSE_CAPTURE_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "on"}
