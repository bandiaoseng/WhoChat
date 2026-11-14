"""Bounded re-reading of one fully visible, low-confidence text bubble.

No confidence is invented and no failing row is dropped. Two fixed local OCR
views must agree at a stricter threshold. The caller still verifies a fresh
full capture and builds the ordinary account/title/message proof afterwards.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from PIL import Image

from whochat.core.current_conversation import screen_top_partial_prefix
from whochat.core.models import Speaker
from whochat.core.runtime import Rect
from whochat.ocr.models import OcrRegion, OcrResult


def low_confidence_messages(messages, layout, report=None, *, threshold=.65):
    """Only real, retained, otherwise complete rows; never a top partial badge."""
    if report is not None and (not isinstance(report, dict)
            or any(report.get(key) is not True for key in (
                "input_boundary_verified", "parsed_coverage_complete"))
            or report.get("history_navigation_present") is not False
            or type(report.get("untranscribed_voice_count")) is not int
            or report["untranscribed_voice_count"] != 0
            or type(report.get("unaccounted_box_count")) is not int
            or report["unaccounted_box_count"] != 0):
        return []
    prefix = screen_top_partial_prefix(messages, layout)
    if prefix is None or not getattr(layout, "input_boundary_verified", False):
        return []
    rows = messages[prefix:]
    for row in rows:
        if (row.partial is not False or row.speaker not in {Speaker.ME, Speaker.OTHER}
                or type(row.confidence) not in {int, float}
                or not math.isfinite(row.confidence) or not 0 <= row.confidence <= 1
                or not row.text.strip()):
            return []
    return [row for row in rows if row.confidence < threshold]


def _inside(inner, outer, margin=0):
    return (outer.left - margin <= inner.left < inner.right <= outer.right + margin
            and outer.top - margin <= inner.top < inner.bottom <= outer.bottom + margin)


def _intersection(a, b):
    return max(0, min(a.right, b.right)-max(a.left, b.left)) * max(0, min(a.bottom, b.bottom)-max(a.top, b.top))


def _compact(text):
    return re.sub(r"\s+", "", text)


def _has_quantity(text):
    # Protect the whole statement, including a dose denominator or unit before
    # the number. Comparing numeric substrings alone would miss those changes.
    return re.search(r"[\d零〇一二两三四五六七八九十百千万亿半壹贰叁肆伍陆柒捌玖拾佰仟萬億]", text) is not None


def _line_consensus(outputs, original):
    """The recognizer reads the complete crop; it does not produce new boxes."""
    if len(outputs) != 2:
        return None
    for output in outputs:
        if not isinstance(output, tuple) or len(output) != 2:
            return None
        text, confidence = output
        if (not isinstance(text, str) or not text.strip()
                or type(confidence) not in {int, float} or not math.isfinite(confidence)
                or not .90 <= confidence <= 1
                or ((_has_quantity(text) or _has_quantity(original.text))
                    and _compact(text) != _compact(original.text))):
            return None
    if _compact(outputs[0][0]) != _compact(outputs[1][0]):
        return None
    return replace(original, text=outputs[0][0], confidence=min(item[1] for item in outputs),
                   source=original.source + ":native_line_consensus")


def _blank_bubble_outline(crop):
    """All four outer edges must be a blank uniform surround, not cut ink."""
    width, height = crop.size
    pixels = crop.load()
    background = pixels[0, 0]
    edge = ([pixels[x, 0] for x in range(width)] + [pixels[x, height-1] for x in range(width)]
            + [pixels[0, y] for y in range(height)] + [pixels[width-1, y] for y in range(height)])
    return all(max(abs(a-b) for a, b in zip(pixel, background)) <= 12 for pixel in edge)


def _dark_surface_outline(crop):
    """Only blank canvas and native dark-grey antialiasing, never clipped ink."""
    width,height=crop.size
    pixels=crop.load();background=pixels[0,0]
    if max(background)-min(background)>3 or not 20 <= min(background) <= 40:
        return False
    edges=([pixels[x,0] for x in range(width)]+[pixels[x,height-1] for x in range(width)]
           +[pixels[0,y] for y in range(height)]+[pixels[width-1,y] for y in range(height)])
    return all(max(p)-min(p)<=3 and min(p)>=min(background)-2 and max(p)<=52 for p in edges)


def recover_low_confidence_content(content, title, layout, frame, engine, work_path, assert_current):
    """At most two local calls per frame, no shared cache or external service.

    Only one isolated single-line bubble is recoverable here. Multiple failing
    rows, media/voice metadata, clipped bubbles, ambiguous numeric text and
    conflicting local OCR never gain a recovery authorization.
    """
    trace = {"attempts": 0, "recovered": False}
    recognize_line = getattr(engine, "recognize_line", None)
    if not callable(recognize_line) or content.warning or frame is None or not layout.input_boundary_verified:
        return content, trace
    from whochat.ocr.parser import classify_box_region, classify_page_from_ocr, parse_visible_messages_with_evidence
    from whochat.vision.bubbles import detect_bubbles, bubble_for_box
    from whochat.core.runtime import PageType

    if not any(type(b.confidence) in {int, float} and math.isfinite(b.confidence)
               and 0 <= b.confidence < .90
               and classify_box_region(b.rect, layout) == OcrRegion.MESSAGE for b in content.boxes):
        return content, trace
    combined = OcrResult(title.boxes + content.boxes, content.source_image, content.engine, content.warning)
    page = classify_page_from_ocr(combined, layout)
    if page.page_type != PageType.CHAT_DM or page.confidence < .65:
        return content, trace
    messages, report = parse_visible_messages_with_evidence(combined, layout, direct_chat=True, frame_image=frame)
    # Also allow improvement of one otherwise accepted weak row. This changes
    # no acceptance threshold: on an unsuccessful read its original evidence
    # remains untouched and the existing .65 proof rules still decide.
    rows = low_confidence_messages(messages, layout, report, threshold=.90)
    failing = [row for row in rows if row.confidence < .65]
    if not rows or len(failing) > 1:
        return content, trace
    # Pick once, before re-reading: the single blocking row, otherwise the
    # unique weakest already-acceptable row. Never try another candidate after
    # disagreement and never remove/ignore other original rows.
    lowest = min(row.confidence for row in rows)
    targets = failing or [row for row in rows if row.confidence == lowest]
    if len(targets) != 1 or targets[0].content_type != "text":
        return content, trace
    row = targets[0]
    originals = [box for box in content.boxes if _intersection(box.rect, row.rect) > 0]
    if len(originals) != 1 or originals[0].rect != row.rect:
        return content, trace
    original = originals[0]
    ox, oy = layout.window_rect.left, layout.window_rect.top
    def local(rect):
        return Rect(rect.left-ox, rect.top-oy, rect.right-ox, rect.bottom-oy)
    local_layout = replace(layout, **{name: local(getattr(layout, name)) for name in (
        "window_rect", "nav_rect", "chat_list_rect", "content_rect", "title_rect", "message_rect", "input_rect")})
    bubble = bubble_for_box(local(row.rect), detect_bubbles(frame, local_layout))
    if (bubble is None or bubble.partial or bubble.speaker != row.speaker
            or not _inside(bubble.rect, local_layout.message_rect)
            or not _inside(local(row.rect), bubble.rect, 2)
            or bubble.rect.height > 64 or bubble.rect.width > 600
            or row.rect.height > 40 or bubble.rect.height > row.rect.height + 24
            or bubble.rect.width < 16 or bubble.rect.height < 16
            or any(box is not original and _intersection(local(box.rect), bubble.rect) > 0 for box in content.boxes)):
        return content, trace
    outputs = []
    # Native UI text has an upright, single-line layout. The generic OCR
    # direction classifier can mistake short text beside emoji for 180-degree
    # print. Read the COMPLETE, unmasked bubble with the text recognizer only;
    # leave the shared detector/classifier and photo orientation untouched.
    # Two fixed scales are a consistency check, NOT two identity observations.
    crop = frame.crop(bubble.rect.as_tuple())
    try:
        if not _blank_bubble_outline(crop):
            if bubble.speaker != Speaker.OTHER or not _dark_surface_outline(crop):
                return content, trace
            # Neutral connected components can stop on the antialiased edge
            # of a dark native bubble. Add only a two-pixel actual surround,
            # never fabricated padding. All geometry/other-text/blank-edge
            # checks still apply before either recognition call.
            padded = Rect(bubble.rect.left-2, bubble.rect.top-2,
                          bubble.rect.right+2, bubble.rect.bottom+2)
            if (not _inside(padded, local_layout.message_rect)
                    or padded.height > 64 or padded.width > 600
                    or padded.height > row.rect.height + 24
                    or any(box is not original and _intersection(local(box.rect), padded) > 0
                           for box in content.boxes)):
                return content, trace
            crop.close()
            crop = frame.crop(padded.as_tuple())
            if not _blank_bubble_outline(crop):
                return content, trace
        for scale in (1, 2):
            assert_current()
            image = crop.copy() if scale == 1 else crop.resize(
                (crop.width*scale, crop.height*scale), Image.Resampling.LANCZOS)
            try:
                output = recognize_line(image)
                trace["attempts"] += 1
            finally:
                image.close()
            assert_current()
            outputs.append(output)
        replacement = _line_consensus(outputs, original)
        if replacement is not None:
            trace["recovered"] = True
            return replace(content, boxes=[replacement if box is original else box for box in content.boxes]), trace
        return content, trace
    finally:
        crop.close()
