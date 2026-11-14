"""Bounded, process-local chat photo crops. Never persist pixels or trust paths in messages.

Capture owns the screenshot; OCR only identifies complete media rectangles.
The UI binds those crops to an independently validated current customer view.
The provider rechecks this binding before building the multimodal request.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json

from PIL import Image

from whochat.core.models import Speaker
from whochat.core.runtime import Rect

MAX_PHOTOS = 6
MAX_PHOTO_BYTES = 1_000_000
MAX_TOTAL_BYTES = 4_000_000
MAX_SIDE = 1280


@dataclass(frozen=True)
class PhotoCrop:
    content_hash: str
    encoded_hash: str
    speaker: str
    width: int
    height: int
    png: bytes = field(repr=False)


@dataclass(frozen=True)
class ReplyPhoto:
    ordinal: int
    content_hash: str
    encoded_hash: str
    speaker: str
    width: int
    height: int
    account_scope_id: str
    contact_id: str
    conversation_id: str
    target_instance_id: str
    session_id: str
    content_revision: str
    png: bytes = field(repr=False)


def _hashes(message):
    found = []
    for ref in message.media_refs:
        try:
            value = json.loads(ref) if isinstance(ref, str) else ref
            digest = value.get("content_hash", "")
            if (value.get("content_type") == "image" and isinstance(digest, str)
                    and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
                found.append(digest)
        except (ValueError, TypeError, AttributeError):
            continue
    return tuple(found)


def photo_messages(view):
    if view is None:
        return ()
    # Same bounded text window as the prompt. Quoted thumbnails are not originals.
    return tuple(m for m in view.messages[-12:]
                 if m.content_type == "image" and m.speaker in (Speaker.OTHER, Speaker.ME)
                 and "引用图片" not in m.text)


def collect_photo_crops(image_path, layout, messages, *, permitted):
    """Called before screenshot cleanup, only on a proved manual capture.

    Parser rectangles are screen-relative; translate them to window pixels.
    Matching raw RGB digest proves that we crop
    exactly the surface the parser recognized, not the surrounding chat UI.
    """
    if not permitted:
        return ()
    selected = [m for m in messages[-12:] if m.content_type == "image"
                and not m.partial and "引用图片" not in m.text][-MAX_PHOTOS:]
    if not selected:
        return ()
    try:
        with Image.open(image_path) as original:
            if original.size != (layout.window_rect.width, layout.window_rect.height):
                return ()
            frame = original.convert("RGB")
        ox, oy = layout.window_rect.left, layout.window_rect.top
        def local(rect):
            return Rect(rect.left - ox, rect.top - oy, rect.right - ox, rect.bottom - oy)
        band = local(layout.message_rect)
        output, total = [], 0
        for message in selected:
            rect = local(message.rect)
            hashes = _hashes(message)
            if (len(hashes) != 1 or message.speaker not in (Speaker.OTHER, Speaker.ME)
                    or rect.width < 70 or rect.height < 72
                    or not (0 <= band.left <= rect.left < rect.right <= band.right <= frame.width)
                    or not (0 <= band.top <= rect.top < rect.bottom <= band.bottom <= frame.height)):
                continue
            crop = frame.crop(rect.as_tuple())
            if hashlib.sha256(crop.tobytes()).hexdigest() != hashes[0]:
                continue
            # Preserve native pixels; never upscale a thumbnail into fake detail.
            crop.thumbnail((MAX_SIDE, MAX_SIDE))
            buffer = BytesIO()
            crop.save(buffer, format="PNG")  # Fresh encoding discards EXIF and file metadata.
            pixels = buffer.getvalue()
            if len(pixels) > MAX_PHOTO_BYTES or total + len(pixels) > MAX_TOTAL_BYTES:
                continue
            output.append(PhotoCrop(hashes[0], hashlib.sha256(pixels).hexdigest(),
                                    message.speaker.value, crop.width, crop.height, pixels))
            total += len(pixels)
        return tuple(output)
    except (OSError, ValueError, TypeError, AttributeError):
        return ()


def bind_reply_photos(view, result, runtime):
    """No IO; never use another customer's last result as a fallback."""
    expected = photo_messages(view)
    if not expected:
        return (), ""
    if len(expected) > MAX_PHOTOS:
        return (), "too_many_photos"
    try:
        proof = result.screen_proof or {}
        if (proof.get("valid") is not True or result.target_app != "wechat"
                or result.account_scope_id != view.account_scope_id
                or result.target_instance_id != view.target_instance_id
                or result.job_id < view.source_job_id
                or result.capture_generation != runtime.capture_generation
                or result.window_generation != runtime.window_generation
                or result.hwnd != runtime.window.hwnd or result.process_id != runtime.window.process_id
                or result.account_identity_epoch != runtime.account_identity_epoch
                or result.account_identity_runtime_key != runtime.account_identity_runtime_key):
            return (), "photo_source_changed"
        if view.screen_evidence_json:
            view_proof = json.loads(view.screen_evidence_json)
            if result.title_exact_hash != view_proof.get("title_exact_hash"):
                return (), "photo_customer_changed"
        output = []
        for message in expected:
            hashes = _hashes(message)
            if len(hashes) != 1:
                return (), "photo_reference_missing"
            matches = [p for p in result.photo_crops
                       if p.content_hash == hashes[0] and p.speaker == message.speaker.value]
            if not matches or any(p != matches[0] for p in matches):
                return (), "photo_pixels_unavailable"
            crop = matches[0]
            output.append(ReplyPhoto(message.ordinal, crop.content_hash, crop.encoded_hash, crop.speaker, crop.width, crop.height,
                view.account_scope_id, view.contact_id, view.conversation_id, view.target_instance_id,
                view.session_id, view.content_revision, crop.png))
        return tuple(output), ""
    except (ValueError, TypeError, AttributeError):
        return (), "photo_pixels_unavailable"


def photo_source_error(context):
    expected = photo_messages(context.current_message_view)
    if context.photo_error:
        return context.photo_error
    if not expected:
        return "unexpected_photos" if context.photos else None
    if len(expected) > MAX_PHOTOS or len(context.photos) != len(expected):
        return "photo_pixels_unavailable"
    view = context.current_message_view
    total = 0
    for message, photo in zip(expected, context.photos):
        if (type(photo) is not ReplyPhoto or _hashes(message) != (photo.content_hash,)
                or photo.ordinal != message.ordinal or photo.speaker != message.speaker.value
                or any(getattr(photo, key) != getattr(view, key) for key in (
                    "account_scope_id", "contact_id", "conversation_id", "target_instance_id",
                    "session_id", "content_revision"))):
            return "photo_binding_invalid"
        if not isinstance(photo.png, bytes) or not 0 < len(photo.png) <= MAX_PHOTO_BYTES:
            return "photo_bytes_invalid"
        if hashlib.sha256(photo.png).hexdigest() != photo.encoded_hash:
            return "photo_bytes_invalid"
        total += len(photo.png)
        try:
            with Image.open(BytesIO(photo.png)) as decoded:
                if (decoded.format != "PNG" or decoded.mode != "RGB"
                        or decoded.size != (photo.width, photo.height)
                        or not 1 <= photo.width <= MAX_SIDE or not 1 <= photo.height <= MAX_SIDE):
                    return "photo_bytes_invalid"
                decoded.verify()
        except (OSError, ValueError, SyntaxError):
            return "photo_bytes_invalid"
    return "photo_bytes_limit" if total > MAX_TOTAL_BYTES else None


def multimodal_user_content(context, text):
    error = photo_source_error(context)
    if error:
        raise ValueError(error)
    if not context.photos:
        return text  # Preserve the existing text-only transport.
    parts = [{"type": "text", "text": text}]
    for photo in context.photos:
        message = context.current_message_view.messages[photo.ordinal]
        who = "客户发来的照片" if photo.speaker == "other" else "我方发出的图片（可能是产品图，不是客户症状证据）"
        when = message.message_time or message.time_text or "日期未知，不等于今天"
        parts.append({"type": "text", "text":
            f"聊天第{photo.ordinal + 1}条：{who}；显示时间：{when}；"
            f"当前窗口可见图 {photo.width}×{photo.height} 像素，非原始大图。"})
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(photo.png).decode("ascii"), "detail": "high"}})
    return parts


def photo_status_message(error):
    if error == "too_many_photos":
        return "本次可见图片超过6张，请将相关的最多6张照片放在当前窗口后重新生成。"
    return "本次照片尚未取得完整、匹配当前客户的图像；请让相关照片完整显示在聊天窗口后重新生成。未调用AI。"
