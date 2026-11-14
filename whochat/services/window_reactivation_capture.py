"""Explicit-button, zero-message capture capability for a limited revisit.

It proves a current chat surface, not a message, photo meaning, or incoming
event. The text-message proof contract deliberately remains unchanged.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from whochat.core.current_conversation import (
    SCREEN_CAPTURE_SCHEMA, SCREEN_TTL_SECONDS, WINDOW_REACTIVATION_SOURCE,
    _validate_capture_fields, screen_rows_digest, screen_business_rows_digest, screen_message_payload,
)
from whochat.core.runtime import PageType, TargetApp


def build_window_reactivation_capture_proof(
    *, job_id, state, account_context, snapshot_hash, created_at, layout,
    ocr_result, page, messages, parser_report, title_exact_hash,
    capture_verified=False, title_verified=False, manual_request_token=None,
    account_avatar_anchor=None, manual_activation=False,
) -> dict | None:
    if manual_activation is not True and (messages or page.reason != "verified_textless_picture_chat_structure"):
        return None

    def invalid(code):
        return {"schema": SCREEN_CAPTURE_SCHEMA, "source_kind": WINDOW_REACTIVATION_SOURCE,
                "valid": False, "rejection_code": code, "source_job_id": job_id, "snapshot_hash": snapshot_hash}

    try:
        if (capture_verified is not True or title_verified is not True or account_context is None
                or account_context.actions_allowed is not True or state.window.target != TargetApp.WECHAT
                or page.page_type != PageType.CHAT_DM or page.confidence < .65 or ocr_result.warning
                or layout.input_boundary_verified is not True or not isinstance(parser_report,dict)
                or (manual_activation is not True and any(parser_report.get(key) != 0 for key in ("message_box_count","parsed_box_count","unaccounted_box_count")))
                or (manual_activation is True and parser_report.get('history_navigation_present') is True)):
            return invalid("current_window_capture_binding_incomplete")
        if (not isinstance(manual_request_token,tuple) or len(manual_request_token)!=2
                or type(manual_request_token[0]) is not int or manual_request_token[0] <= 0
                or not isinstance(manual_request_token[1],tuple) or not manual_request_token[1]):
            return invalid("current_window_manual_button_required")
        expected_owner=(state.window.hwnd,state.window.process_id,state.window_generation,
                        account_context.identity_epoch,account_context.runtime_key)
        if manual_request_token[1] != expected_owner:
            return invalid("current_window_manual_owner_changed")
        if not isinstance(account_avatar_anchor,dict):
            return invalid("current_window_avatar_anchor_missing")
        if manual_activation is True:
            from whochat.vision.manual_activation_structure import manual_activation_structure
            structure=manual_activation_structure(ocr_result.source_image,layout)
        else:
            from whochat.vision.current_window_structure import textless_chat_structure
            structure=textless_chat_structure(ocr_result.source_image,layout)
        if structure is None:
            return invalid("current_window_positive_structure_missing")
        observed=datetime.fromisoformat(created_at.replace("Z","+00:00"))
        proof={"schema":SCREEN_CAPTURE_SCHEMA,"source_kind":WINDOW_REACTIVATION_SOURCE,"valid":True,
            "purpose":"reactivation","manual_request_id":manual_request_token[0],"window_capture_complete":True,
            "capture_verified":True,"title_verified":True,"account_verified":True,"parser_complete":False,
            "tail_complete":False,"input_boundary_verified":True,"page_type":"chat_dm","page_confidence":page.confidence,
            "raw_count":0,"kept_count":0,"omitted_count":0,"omitted_top_prefix_count":0,"partial_count":0,
            "kept_partial_count":0,"order_uncertain_count":0,"unaccounted_box_count":0,
            "message_box_count":0,"parsed_box_count":0,"ignored_metadata_box_count":0,"min_confidence":page.confidence,
            "rows_sha256":screen_rows_digest([]),"business_rows_sha256":screen_business_rows_digest([]),
            "raw_rows_sha256":screen_rows_digest([]),"omitted_prefix_sha256":screen_rows_digest([]),
            "window_body_sha256":structure["body_sha256"],"window_structure":structure,
            "source_job_id":job_id,"snapshot_hash":snapshot_hash,"account_scope_id":account_context.account_scope_id,
            "target_instance_id":account_context.target_instance_id,"hwnd":state.window.hwnd,"process_id":state.window.process_id,
            "window_generation":state.window_generation,"capture_generation":state.capture_generation,
            "account_identity_epoch":account_context.identity_epoch,"account_identity_runtime_key":account_context.runtime_key,
            "title_exact_hash":title_exact_hash,"observed_at":observed.isoformat(),
            "expires_at":(observed+timedelta(seconds=SCREEN_TTL_SECONDS)).isoformat()}
        for key in ("account_avatar_fingerprint","account_avatar_geometry","account_avatar_reliable"):
            proof[key]=account_avatar_anchor.get(key)
        if manual_activation is True:
            proof['operator_selected_activation']=True
            proof['unread_current_rows_sha256']=screen_rows_digest([screen_message_payload(m) for m in messages])
        _validate_capture_fields(proof,datetime.now(timezone.utc))
        return proof
    except (ValueError,TypeError,AttributeError,KeyError):
        return invalid("current_window_capture_proof_invalid")
