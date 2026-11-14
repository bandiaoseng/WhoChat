from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from PIL import Image

from whochat.core.models import Speaker
from whochat.core.runtime import LayoutRegions, Rect, TargetApp
from whochat.vision.visual import load_rgb_image


@dataclass(frozen=True)
class BubbleRegion:
    rect: Rect
    speaker: Speaker
    confidence: float
    partial: bool
    profile: str


def detect_bubbles(image_path: str | Path | Image.Image, layout: LayoutRegions) -> list[BubbleRegion]:
    profile = bubble_profile_for_layout(layout)
    if profile == "wechat_green":
        return _detect_wechat_bubbles(image_path, layout)
    return []


def bubble_profile_for_layout(layout: LayoutRegions) -> str:
    if layout.bubble_profile and layout.bubble_profile != "auto":
        return layout.bubble_profile
    if layout.target_app == TargetApp.WECHAT:
        return "wechat_green"
    return "geometry"


def outgoing_avatar_text_regions(
    image_path: str | Path | Image.Image,
    boxes: list[Rect],
    layout: LayoutRegions,
) -> set[Rect]:
    """Recognize tiny right-avatar ink, not arbitrary short/small messages.

    A WeChat outgoing avatar sits outside the green message surface. Its logo
    may itself be green, so connected-component/nearby bubble matching can
    attach its OCR to a real message. Require all three independent cues:
    outer avatar-rail geometry, a taller inboard green-bubble text box on the
    same row, and a visible neutral vertical gutter separating the two.
    A small character *inside* a bubble has no such gutter and stays content.
    Ambiguous pictures, missing pixels and non-WeChat layouts are left alone.

    Coordinates are window-local, just like the source screenshot pixels.
    """
    if bubble_profile_for_layout(layout) != "wechat_green" or len(boxes) < 2:
        return set()
    message = layout.message_rect
    if message.width < 160:
        return set()
    heights = sorted(box.height for box in boxes if 12 <= box.height <= 64)
    font_height = heights[len(heights) // 2] if heights else 18
    candidates: dict[Rect, list[Rect]] = {}
    for outer in boxes:
        center_x = (outer.left + outer.right) / 2
        if (center_x < message.left + message.width * 0.82
                or outer.height > max(9, font_height * 0.62)
                or outer.width > max(20, font_height * 2)
                or outer.width < 2 or outer.height < 2):
            continue
        peers = []
        for inner in boxes:
            gap = outer.left - inner.right
            overlap = min(outer.bottom, inner.bottom) - max(outer.top, inner.top)
            if (inner == outer or inner.width < max(16, outer.width * 1.5)
                    or inner.height < max(12, outer.height * 1.4)
                    or (inner.left + inner.right) / 2 >= message.left + message.width * 0.82
                    or not 4 <= gap <= max(48, min(96, message.width * 0.20))
                    or overlap / max(1, min(inner.height, outer.height)) < 0.10):
                continue
            peers.append(inner)
        if peers:
            candidates[outer] = peers
    if not candidates:
        return set()
    try:
        rgb = load_rgb_image(image_path)
        if rgb is None:
            return set()
    except (OSError, ValueError, IndexError):
        return set()
    ignored: set[Rect] = set()
    for outer, peers in candidates.items():
        for inner in peers:
            bubble = bubble_for_box_in_image(rgb, inner, layout)
            if bubble is None or bubble.speaker != Speaker.ME:
                continue
            if _neutral_gutter_between(rgb, inner, outer, message):
                ignored.add(outer)
                break
    return ignored


def incoming_bubble_avatar_text_regions(
    image_path: str | Path | Image.Image, boxes: list[Rect], layout: LayoutRegions,
    *, image_cards: tuple[Rect, ...] = (),
) -> set[Rect]:
    """Remove ink *inside* a proved left avatar, never a short customer word.

    Require a complete neutral message card on the same row, a textured
    square avatar on the outer rail and a visible neutral gutter. This is
    independent of the recognised character (an avatar may read 英 or 美).
    """
    if (bubble_profile_for_layout(layout) != "wechat_green"
            or layout.input_boundary_verified is not True or (len(boxes) < 2 and not image_cards)):
        return set()
    band = layout.message_rect
    candidates = [box for box in boxes if 2 <= box.width <= 60 and 2 <= box.height <= 60
                  and band.left <= box.left and box.right <= band.left+band.width*.20]
    if not candidates:
        return set()
    try:
        rgb = load_rgb_image(image_path)
        if rgb is None:
            return set()
        if rgb.size != (layout.window_rect.width, layout.window_rect.height):
            return set()
        ignored = set()
        for outer in candidates:
            # These rectangles come only from this same frame's existing
            # complete chat-picture detector (texture, avatar and gutter).
            # Re-prove the avatar pixels; a short word in a bubble has no
            # authority just because it is near a photograph.
            for card in image_cards:
                avatar = _incoming_voice_avatar(rgb, card, band)
                if (avatar is not None and _voice_avatar_ocr_box(outer, avatar, card)
                        and _neutral_gutter_between(rgb, avatar, card, band)):
                    ignored.add(outer)
                    break
            if outer in ignored:
                continue
            peers = [inner for inner in boxes if 4 <= inner.left-outer.right <= 75
                     and min(inner.bottom,outer.bottom) > max(inner.top,outer.top)]
            for inner in peers:
                card = _flat_neutral_rectangle(rgb, inner, band, max_width=band.width*.82,
                    max_height=160, horizontal_padding=24, vertical_padding=20)
                if card is None:
                    continue
                avatar = _incoming_voice_avatar(rgb, card, band)
                if (avatar is not None and _voice_avatar_ocr_box(outer, avatar, card)
                        and _neutral_gutter_between(rgb, avatar, card, band)):
                    ignored.add(outer)
                    break
        return ignored
    except (OSError, ValueError, IndexError):
        return set()


def incoming_transcribed_voice_metadata(
    image_path: str | Path | Image.Image, boxes: list[tuple[Rect, str]], layout: LayoutRegions,
    *, unresolved_voice_regions: set[Rect] | None = None,
    complete_transcript_regions: set[Rect] | None = None,
) -> tuple[set[Rect], set[Rect]]:
    """Prove native transcribed-voice chrome, never infer spoken content.

    A quote/number is insufficient: require a bounded flat card, independent
    waveform strokes, a textured incoming avatar and gutter, and a separate
    transcription bubble without a new avatar. The caller enables this only
    for confirmed direct chats. Missing/ambiguous pixels remain unclassified.
    """
    if bubble_profile_for_layout(layout) != "wechat_green" or not boxes:
        return set(), set()
    try:
        rgb = load_rgb_image(image_path)
        if rgb is None:
            return set(), set()
    except (OSError, ValueError, IndexError):
        return set(), set()
    band = layout.message_rect
    voice_ink: set[Rect] = set()
    avatar_ink: set[Rect] = set()
    for badge, text in boxes:
        compact = re.sub(r"\s+", "", text)
        # The waveform is not text: OCR may render it as punctuation, a Latin
        # letter or a Chinese glyph. Only the trailing duration is lexical
        # evidence. Never whitelist waveform spellings; the independent card,
        # waveform pixels, avatar, gutter and transcript below must all agree.
        match = re.fullmatch(r"\D{0,3}(\d{1,2})[\"″”'’]", compact)
        if match is None:
            # Native short audio ink can be turned upside down by OCR's line
            # classifier: the reported 6-second badge became '.9 ('. This is
            # only a bounded candidate, never proof or a correction of speech.
            # Do not admit words, decimals, units or a naked number. The same
            # independent waveform/card/avatar/gutter/transcript checks apply.
            match = re.fullmatch(r"[.·()（）»«]{0,3}(\d{1,2})[.·()（）»«]{1,3}", compact)
        if (match is None or not 1 <= int(match[1]) <= 60
                or badge.width > 90 or badge.height > 40):
            continue
        card = _flat_neutral_rectangle(rgb, badge, band, max_width=min(240, band.width * .55),
                                       max_height=64, horizontal_padding=240, vertical_padding=24)
        if (card is None or card.width < 65 or card.height < 28
                or card.left > band.left + band.width * .30
                or not card.left + 5 <= badge.left <= card.left + 38
                or badge.right > card.left + 90
                or not _voice_waveform_pixels(rgb, card)):
            continue
        avatar = (_incoming_voice_avatar(rgb, card, band)
                  or _incoming_voice_textured_avatar(rgb, card, band))
        # Grey photographic avatars still require a complete square, texture,
        # exact native lane/alignment and clean gutter. Transcript and waveform
        # checks below are unchanged; avatar colour is not customer identity.
        if avatar is None or not _neutral_gutter_between(rgb, avatar, card, band):
            continue
        transcript = None
        for body, body_text in boxes:
            if (not body_text.strip() or body == badge
                    or not 0 <= body.top - card.bottom <= 35
                    or abs(body.left - card.left) > 24 or body.width < 8
                    or re.fullmatch(r"[()（）>»]{0,3}\d{1,2}[\"″”'’]", re.sub(r"\s+", "", body_text))):
                continue
            surface = _flat_neutral_rectangle(
                rgb, body, band, max_width=band.width, max_height=180,
                horizontal_padding=band.width, vertical_padding=180,
            )
            if (surface is None or not 2 <= surface.top - card.bottom <= 20
                    or abs(surface.left - card.left) > 8
                    or _incoming_voice_avatar(rgb, surface, band) is not None
                    or _left_rail_has_message_ink(rgb, surface, band)):
                continue
            transcript = surface
            break
        if transcript is None:
            if unresolved_voice_regions is not None:
                unresolved_voice_regions.add(badge)
            continue
        card_boxes = [rect for rect, _ in boxes if _inside_with_margin(rect, card, 2)]
        # Extra unexplained words inside a candidate are not disposable.
        if any(rect != badge and not (rect.right <= card.left + 36
                                     and rect.width <= 28 and rect.height <= 28)
               for rect in card_boxes):
            continue
        voice_ink.update(card_boxes)
        avatar_ink.update(rect for rect, _ in boxes if _voice_avatar_ocr_box(rect, avatar, card))
        if (complete_transcript_regions is not None
                and rgb.size == (layout.window_rect.width, layout.window_rect.height)):
            complete_transcript_regions.add(transcript)
    return voice_ink, avatar_ink


def _incoming_voice_textured_avatar(rgb, card: Rect, band: Rect):
    """Independent complete photographic square, including grey photographs."""
    left, right = max(0, band.left + 4), card.left - 4
    top, bottom = max(band.top, card.top - 8), min(band.bottom, card.bottom + 8)
    if not 28 <= right-left <= 110 or bottom-top < 28:
        return None
    crop = rgb.crop((left, top, right, bottom))
    mask = [max(p) < 240 or max(p)-min(p) >= 18 for p in crop.getdata()]
    matches = []
    for x1,y1,x2,y2,area in _connected_components(mask, crop.width, crop.height):
        rect = Rect(left+x1, top+y1, left+x2, top+y2)
        if (not 28 <= rect.width <= 60 or not 28 <= rect.height <= 60
                or abs(rect.width-rect.height) > 6 or abs(rect.top-card.top) > 8
                or not 4 <= card.left-rect.right <= 35
                or rect.right > band.left+band.width*.19
                or rect.left <= left or rect.right >= right
                or rect.top <= top or rect.bottom >= bottom
                or area/(rect.width*rect.height) < .70):
            continue
        pixels = list(rgb.crop(rect.as_tuple()).getdata())
        tones = sorted(sum(p)/3 for p in pixels)
        colours = {tuple(v//16 for v in p) for p in pixels}
        if len(colours) >= 12 and tones[-len(tones)//10]-tones[len(tones)//10] >= 45:
            matches.append(rect)
    return matches[0] if len(matches) == 1 else None


def _voice_avatar_ocr_box(ink: Rect, avatar: Rect, card: Rect) -> bool:
    # OCR detection boxes can include the avatar's rounded white corners and
    # several background pixels. The colour component is smaller than that
    # box. Accept bounded detector padding only on this independently proved
    # avatar, never text extending into the adjacent message card.
    padding = max(4, min(12, round(max(avatar.width, avatar.height) * .25)))
    return (
        _inside_with_margin(ink, avatar, padding)
        and ink.right < card.left
        and avatar.left <= (ink.left + ink.right) / 2 <= avatar.right
        and avatar.top <= (ink.top + ink.bottom) / 2 <= avatar.bottom
        and _overlap_ratio(ink, avatar) >= .55
    )


def _left_rail_has_message_ink(rgb, surface: Rect, band: Rect) -> bool:
    # Absence of a second *coloured* avatar is not enough: a grey avatar or
    # ambiguous rail ink also makes a following ordinary message possible.
    # Such a frame must not turn an untranscribed voice into disposable chrome.
    left, right = max(0, band.left + 4), surface.left - 4
    top, bottom = surface.top, min(surface.bottom, surface.top + 28)
    if right <= left or bottom <= top:
        return True
    ink = sum(max(pixel) < 245 or max(pixel) - min(pixel) >= 18
              for pixel in rgb.crop((left, top, right, bottom)).getdata())
    return ink >= 24


def _inside_with_margin(inner: Rect, outer: Rect, margin: int = 0) -> bool:
    return (outer.left - margin <= inner.left and inner.right <= outer.right + margin
            and outer.top - margin <= inner.top and inner.bottom <= outer.bottom + margin)


def _flat_neutral_rectangle(rgb, ink: Rect, band: Rect, *, max_width: float,
                            max_height: int, horizontal_padding: int, vertical_padding: int):
    """Bounded native-card component, distinct from the white canvas."""
    left, top = max(0, band.left, ink.left - horizontal_padding), max(0, band.top, ink.top - vertical_padding)
    right = min(rgb.width, band.right, ink.right + horizontal_padding)
    bottom = min(rgb.height, band.bottom, ink.bottom + vertical_padding)
    if right <= left or bottom <= top:
        return None
    crop = rgb.crop((left, top, right, bottom))
    mask = [max(pixel) - min(pixel) <= 5 and 205 <= sum(pixel) / 3 <= 245 for pixel in crop.getdata()]
    matches = []
    for x1, y1, x2, y2, area in _connected_components(mask, crop.width, crop.height):
        rect = Rect(left + x1, top + y1, left + x2, top + y2)
        if (not 20 <= rect.width <= max_width or not 22 <= rect.height <= max_height
                or not _inside_with_margin(ink, rect, 1)
                or rect.left <= band.left + 2 or rect.right >= band.right - 2
                or rect.top <= band.top + 2 or rect.bottom >= band.bottom - 2
                or area / max(1, rect.width * rect.height) < .72):
            continue
        # A crop edge is not evidence of a complete card boundary.
        if rect.left <= left or rect.top <= top or rect.right >= right or rect.bottom >= bottom:
            continue
        matches.append(rect)
    return min(matches, key=lambda rect: rect.width * rect.height) if matches else None


def _voice_waveform_pixels(rgb, card: Rect) -> bool:
    """Two thin curved audio strokes, independently of the OCR label."""
    left, top = card.left + 10, card.top + 5
    right, bottom = min(card.left + 34, card.right - 8), card.bottom - 5
    crop = rgb.crop((left, top, right, bottom))
    mask = [max(pixel) < 125 for pixel in crop.getdata()]
    strokes = []
    for x1, y1, x2, y2, area in _connected_components(mask, crop.width, crop.height):
        width, height = x2 - x1, y2 - y1
        if (2 <= width <= 8 and 6 <= height <= 23 and height >= width * 1.4
                and area <= width * height * .78
                and abs((top + y1 + top + y2) / 2 - (card.top + card.bottom) / 2) <= 4):
            strokes.append((x1, y1, x2, y2))
    strokes.sort()
    # Native curved strokes are separate components but their bounding boxes
    # can overlap by one antialiased column; they must still grow outwards.
    return any(-2 <= later[0] - earlier[2] <= 8
               and later[3] - later[1] >= earlier[3] - earlier[1] + 2
               for earlier, later in zip(strokes, strokes[1:]))


def _incoming_voice_avatar(rgb, card: Rect, band: Rect):
    left, right = max(0, band.left + 4), card.left - 4
    top, bottom = max(0, band.top, card.top - 9), min(rgb.height, band.bottom, card.top + 65)
    if right - left < 28 or right - left > 110 or bottom - top < 28:
        return None
    crop = rgb.crop((left, top, right, bottom))
    points = [(x, y) for y in range(crop.height) for x in range(crop.width)
              if max(crop.getpixel((x, y))) - min(crop.getpixel((x, y))) >= 18]
    if not points:
        return None
    xs, ys = zip(*points)
    avatar = Rect(left + min(xs), top + min(ys), left + max(xs) + 1, top + max(ys) + 1)
    if (not 28 <= avatar.width <= 60 or not 28 <= avatar.height <= 60
            or abs(avatar.width - avatar.height) > 10 or abs(avatar.top - card.top) > 10
            or not 4 <= card.left - avatar.right <= 35
            or avatar.right > band.left + band.width * .19):
        return None
    pixels = list(rgb.crop(avatar.as_tuple()).getdata())
    colours = {tuple(value // 16 for value in pixel) for pixel in pixels}
    if len(colours) < 14 or len(points) < avatar.width * avatar.height * .18:
        return None
    return avatar


def _neutral_gutter_between(rgb: Image.Image, inner: Rect, outer: Rect, message: Rect) -> bool:
    left = max(0, message.left, inner.right + 2)
    right = min(rgb.width, message.right, outer.left)
    top = max(0, message.top, min(inner.top, outer.top) - 2)
    bottom = min(rgb.height, message.bottom, max(inner.bottom, outer.bottom) + 2)
    if right - left < 2 or bottom - top < 8:
        return False
    pixels = rgb.load()
    consecutive = 0
    for x in range(left, right):
        samples = [pixels[x, y] for y in range(top, bottom)]
        neutral = [pixel for pixel in samples if max(pixel) - min(pixel) <= 12]
        flat = bool(neutral and max(max(pixel) for pixel in neutral)
                    - min(min(pixel) for pixel in neutral) <= 12)
        # Require the entire padded vertical band to be neutral. A tall ink
        # stroke inside a green message can occupy >90% of a column, but its
        # green padding still proves that it is NOT a page-background gutter.
        if len(neutral) == len(samples) and flat:
            consecutive += 1
            if consecutive >= 2:
                return True
        else:
            consecutive = 0
    return False


def bubble_for_box(box: Rect, bubbles: list[BubbleRegion]) -> BubbleRegion | None:
    if not bubbles:
        return None
    center_x = (box.left + box.right) // 2
    center_y = (box.top + box.bottom) // 2
    containing = [
        bubble
        for bubble in bubbles
        if bubble.rect.left <= center_x <= bubble.rect.right and bubble.rect.top <= center_y <= bubble.rect.bottom
    ]
    if containing:
        return max(containing, key=lambda bubble: bubble.confidence)
    overlaps = [(bubble, _overlap_ratio(box, bubble.rect)) for bubble in bubbles]
    bubble, score = max(overlaps, key=lambda item: item[1])
    if score >= 0.25:
        return bubble
    nearby = [(bubble, _nearby_score(box, bubble.rect)) for bubble in bubbles]
    bubble, score = max(nearby, key=lambda item: item[1])
    return bubble if score >= 0.42 else None


def bubble_for_box_from_image(
    image_path: str | Path | Image.Image,
    box: Rect,
    layout: LayoutRegions,
) -> BubbleRegion | None:
    """Classify a text box from the color ring around it.

    OCR boxes usually cover glyphs, not the complete chat bubble. Sampling the
    surrounding ring is therefore more stable than requiring a connected
    component to contain the OCR box.

    ``image_path`` accepts an already loaded :class:`PIL.Image.Image` so callers
    that classify many boxes can decode the screenshot once.
    """
    try:
        rgb = load_rgb_image(image_path)
        return bubble_for_box_in_image(rgb, box, layout) if rgb is not None else None
    except (OSError, ValueError, IndexError):
        return None


def bubble_for_box_in_image(
    rgb: Image.Image,
    box: Rect,
    layout: LayoutRegions,
) -> BubbleRegion | None:
    """Ring-sampling bubble classifier working on an in-memory RGB screenshot."""
    image_width, image_height = rgb.size
    message = layout.message_rect
    left = max(0, message.left, box.left - 18)
    top = max(0, message.top, box.top - 14)
    right = min(image_width, message.right, box.right + 18)
    bottom = min(image_height, message.bottom, box.bottom + 14)
    if right <= left or bottom <= top:
        return None
    samples: list[tuple[int, int, int]] = []
    for y in range(top, bottom, 2):
        for x in range(left, right, 2):
            if box.left <= x <= box.right and box.top <= y <= box.bottom:
                continue
            if 0 <= x < image_width and 0 <= y < image_height:
                samples.append(rgb.getpixel((x, y)))
    if len(samples) < 12:
        return None

    green_ratio = sum(_is_green_pixel(pixel) for pixel in samples) / len(samples)
    neutral_ratio = sum(_is_neutral_bubble_pixel(pixel) for pixel in samples) / len(samples)
    if green_ratio >= 0.18:
        confidence = min(0.96, 0.70 + green_ratio * 0.45)
        return BubbleRegion(Rect(left, top, right, bottom), Speaker.ME, confidence, _is_partial_rect(Rect(left, top, right, bottom), message), "wechat_green")

    center_ratio = ((box.left + box.right) / 2 - message.left) / max(1, message.width)
    if neutral_ratio >= 0.28 and center_ratio <= 0.70:
        confidence = min(0.90, 0.58 + neutral_ratio * 0.35)
        return BubbleRegion(Rect(left, top, right, bottom), Speaker.OTHER, confidence, _is_partial_rect(Rect(left, top, right, bottom), message), "wechat_left")
    return None


def _detect_wechat_bubbles(image_path: str | Path | Image.Image, layout: LayoutRegions) -> list[BubbleRegion]:
    try:
        rgb = load_rgb_image(image_path)
        if rgb is None:
            return []
        crop = rgb.crop(layout.message_rect.as_tuple())
    except OSError:
        return []
    width, height = crop.size
    if width < 80 or height < 80:
        return []
    # Text and thin outlines in WeChat's dark theme are often the only green
    # pixels left; dilate them slightly before finding message-level regions.
    mask = _dilate_mask(_green_mask(crop), width, height)
    components = _group_green_components(_connected_components(mask, width, height), width, height)
    bubbles: list[BubbleRegion] = []
    min_area = max(48, round(width * height * 0.00018))
    for left, top, right, bottom, area in components:
        rect = Rect(
            layout.message_rect.left + left,
            layout.message_rect.top + top,
            layout.message_rect.left + right,
            layout.message_rect.top + bottom,
        )
        if area < min_area or rect.width < 18 or rect.height < 8:
            continue
        center_ratio = ((rect.left + rect.right) / 2 - layout.message_rect.left) / max(1, layout.message_rect.width)
        if center_ratio < 0.45:
            continue
        density = area / max(1, rect.width * rect.height)
        partial = rect.top <= layout.message_rect.top + 4 or rect.bottom >= layout.message_rect.bottom - 4
        confidence = min(0.95, 0.58 + density * 0.5 + max(0.0, center_ratio - 0.55) * 0.35)
        bubbles.append(BubbleRegion(rect, Speaker.ME, confidence, partial, "wechat_green"))
    bubbles.extend(_detect_left_bubbles(crop, layout))
    bubbles.sort(key=lambda bubble: (bubble.rect.top, bubble.rect.left))
    return _merge_close_bubbles(bubbles, layout)


def _detect_left_bubbles(crop: Image.Image, layout: LayoutRegions) -> list[BubbleRegion]:
    width, height = crop.size
    # Reject a neutral canvas from a tiny probe first.  Building the full mask
    # in Python is unnecessary when the conversation background itself already
    # satisfies the neutral-bubble colour range.
    probe_scale = min(1.0, 96 / max(1, width), 96 / max(1, height))
    probe = crop.resize(
        (max(1, round(width * probe_scale)), max(1, round(height * probe_scale))),
        Image.Resampling.BILINEAR,
    )
    probe_mask = _left_bubble_mask(probe)
    if probe_mask and sum(probe_mask) / len(probe_mask) >= 0.55:
        return []
    raw_mask = _left_bubble_mask(crop)
    # A light/dark conversation canvas itself is neutral.  On such frames the
    # old threshold marked almost the entire message pane as one "left bubble"
    # and then walked hundreds of thousands of pixels in Python.  It added no
    # usable component (the giant rectangle is filtered later), while costing
    # 1.5--4 s after every OCR.  Per-OCR-box ring sampling remains active and is
    # the more precise fallback, so skip only this unusable global mask when it
    # covers most of the pane.
    if raw_mask and sum(raw_mask) / len(raw_mask) >= 0.55:
        return []
    mask = _dilate_mask(raw_mask, width, height)
    components = _group_left_components(_connected_components(mask, width, height), width, height)
    bubbles: list[BubbleRegion] = []
    min_area = max(120, round(width * height * 0.00045))
    for left, top, right, bottom, area in components:
        rect = Rect(
            layout.message_rect.left + left,
            layout.message_rect.top + top,
            layout.message_rect.left + right,
            layout.message_rect.top + bottom,
        )
        if area < min_area or rect.width < 36 or rect.height < 18:
            continue
        center_ratio = ((rect.left + rect.right) / 2 - layout.message_rect.left) / max(1, layout.message_rect.width)
        left_ratio = (rect.left - layout.message_rect.left) / max(1, layout.message_rect.width)
        if center_ratio > 0.66 or left_ratio > 0.35:
            continue
        density = area / max(1, rect.width * rect.height)
        partial = rect.top <= layout.message_rect.top + 4 or rect.bottom >= layout.message_rect.bottom - 4
        confidence = min(0.9, 0.54 + density * 0.45 + max(0.0, 0.45 - left_ratio) * 0.25)
        bubbles.append(BubbleRegion(rect, Speaker.OTHER, confidence, partial, "wechat_left"))
    return bubbles


def _green_mask(image: Image.Image) -> list[bool]:
    try:
        import numpy as np

        pixels = np.asarray(image if image.mode == "RGB" else image.convert("RGB"), dtype=np.int16)
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        light = (green >= 125) & (red <= 220) & (blue <= 205) & (green - red >= 18) & (green - blue >= 18)
        dark = (green >= 80) & (red <= 150) & (blue <= 175) & (green - red >= 20) & (green - blue >= 15)
        return (light | dark).ravel().tolist()
    except (ImportError, ValueError, TypeError):
        pass
    pixels = list(image.getdata())
    mask: list[bool] = []
    for red, green, blue in pixels:
        # The dark theme's accent is commonly close to #00a361.  Do not
        # require a non-zero red channel: that was the reason real screenshots
        # produced no detections.
        mask.append(_is_green_pixel((red, green, blue)))
    return mask


def _is_green_pixel(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    light_wechat = green >= 125 and red <= 220 and blue <= 205 and green - red >= 18 and green - blue >= 18
    dark_wechat = green >= 80 and red <= 150 and blue <= 175 and green - red >= 20 and green - blue >= 15
    return light_wechat or dark_wechat


def _is_neutral_bubble_pixel(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    maximum = max(pixel)
    minimum = min(pixel)
    grayish = maximum - minimum <= 16
    # Native dark WeChat uses (47,47,48); the chat canvas remains near (30,30,31).
    return grayish and ((210 <= red <= 255) or (47 <= red <= 105))


def _is_partial_rect(rect: Rect, message: Rect) -> bool:
    return rect.top <= message.top + 4 or rect.bottom >= message.bottom - 4


def _left_bubble_mask(image: Image.Image) -> list[bool]:
    try:
        import numpy as np

        pixels = np.asarray(image if image.mode == "RGB" else image.convert("RGB"), dtype=np.int16)
        maximum = pixels.max(axis=2)
        minimum = pixels.min(axis=2)
        grayish = maximum - minimum <= 18
        light = grayish & (pixels[..., 0] >= 218) & (pixels[..., 1] >= 218) & (pixels[..., 2] >= 218)
        dark = grayish & (pixels[..., 0] >= 47) & (pixels[..., 0] <= 92) & (pixels[..., 1] >= 47) & (pixels[..., 1] <= 92) & (pixels[..., 2] >= 47) & (pixels[..., 2] <= 96)
        return (light | dark).ravel().tolist()
    except (ImportError, ValueError, TypeError):
        pass
    pixels = list(image.getdata())
    mask: list[bool] = []
    for red, green, blue in pixels:
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        grayish = maximum - minimum <= 18
        light_bubble = grayish and 218 <= red <= 255 and 218 <= green <= 255 and 218 <= blue <= 255
        dark_bubble = grayish and 47 <= red <= 92 and 47 <= green <= 92 and 47 <= blue <= 96
        mask.append(light_bubble or dark_bubble)
    return mask


def _dilate_mask(mask: list[bool], width: int, height: int) -> list[bool]:
    try:
        import numpy as np

        source = np.asarray(mask, dtype=np.bool_).reshape((height, width))
        padded = np.pad(source, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(source)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy:dy + height, dx:dx + width]
        return expanded.ravel().tolist()
    except (ImportError, ValueError, TypeError):
        pass
    expanded = bytearray(len(mask))
    for index, value in enumerate(mask):
        if not value:
            continue
        x = index % width
        y = index // width
        for dy in (-1, 0, 1):
            next_y = y + dy
            if next_y < 0 or next_y >= height:
                continue
            for dx in (-1, 0, 1):
                next_x = x + dx
                if 0 <= next_x < width:
                    expanded[next_y * width + next_x] = 1
    return [bool(value) for value in expanded]


def _native_connected_components(mask, width: int, height: int) -> list[tuple[int, int, int, int, int]] | None:
    """Equivalent four-neighbour statistics, or None for the original DFS.

    WU/SAUF labels components in first-pixel row-major order, just like the
    original scan. Do not sort by bounding-box corners: those can have a
    different order. This borrows the mask without changing it and retains no
    image, component, customer or frame state between calls.
    """
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or len(mask) != width * height):
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        # Boolean conversion precedes uint8: nonzero values such as 256 must
        # remain foreground, never wrap to zero during a narrowing conversion.
        bitmap = np.asarray(mask, dtype=np.bool_).astype(np.uint8).reshape((height, width))
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
            bitmap, 4, cv2.CV_32S, cv2.CCL_WU,
        )
        if stats.shape != (count, 5):
            return None
        return [(int(left), int(top), int(left) + int(w), int(top) + int(h), int(area))
                for left, top, w, h, area in stats[1:]]
    except (AttributeError, TypeError, ValueError, OverflowError, cv2.error):
        return None


def _connected_components(mask: list[bool], width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    native = _native_connected_components(mask, width, height)
    if native is not None:
        return native
    return _connected_components_dfs(mask, width, height)


def _connected_components_dfs(mask: list[bool], width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    seen = bytearray(width * height)
    components: list[tuple[int, int, int, int, int]] = []
    for index, value in enumerate(mask):
        if not value or seen[index]:
            continue
        stack = [index]
        seen[index] = 1
        left = right = index % width
        top = bottom = index // width
        area = 0
        while stack:
            current = stack.pop()
            area += 1
            x = current % width
            y = current // width
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
            for neighbor in _neighbors(x, y, width, height):
                if mask[neighbor] and not seen[neighbor]:
                    seen[neighbor] = 1
                    stack.append(neighbor)
        components.append((left, top, right + 1, bottom + 1, area))
    return components


def _group_green_components(
    components: list[tuple[int, int, int, int, int]], width: int, height: int
) -> list[tuple[int, int, int, int, int]]:
    """Join anti-aliased characters/edges into a message-level green region."""
    pending = sorted(components, key=lambda item: (item[1], item[0]))
    grouped: list[tuple[int, int, int, int, int]] = []
    max_gap = max(12, round(height * 0.045))
    max_horizontal_gap = max(24, round(width * 0.06))
    for component in pending:
        if not grouped:
            grouped.append(component)
            continue
        previous = grouped[-1]
        left, top, right, bottom, area = component
        p_left, p_top, p_right, p_bottom, p_area = previous
        vertical_gap = max(0, max(p_top, top) - min(p_bottom, bottom))
        horizontal_gap = max(0, max(p_left, left) - min(p_right, right))
        vertical_overlap = min(p_bottom, bottom) - max(p_top, top)
        close_on_same_message = (
            vertical_gap <= max_gap
            and horizontal_gap <= max_horizontal_gap
            and (vertical_overlap > 0 or abs(top - p_top) <= max_gap)
            and max(right, p_right) - min(left, p_left) <= round(width * 0.78)
        )
        if close_on_same_message:
            grouped[-1] = (
                min(p_left, left),
                min(p_top, top),
                max(p_right, right),
                max(p_bottom, bottom),
                p_area + area,
            )
        else:
            grouped.append(component)
    return grouped


def _group_left_components(
    components: list[tuple[int, int, int, int, int]], width: int, height: int
) -> list[tuple[int, int, int, int, int]]:
    candidates = [
        component
        for component in components
        if component[2] - component[0] >= 8
        and component[3] - component[1] >= 8
        and component[0] <= round(width * 0.42)
        and component[2] <= round(width * 0.78)
    ]
    pending = sorted(candidates, key=lambda item: (item[1], item[0]))
    grouped: list[tuple[int, int, int, int, int]] = []
    max_vertical_gap = max(10, round(height * 0.035))
    max_horizontal_gap = max(18, round(width * 0.05))
    for component in pending:
        if not grouped:
            grouped.append(component)
            continue
        previous = grouped[-1]
        left, top, right, bottom, area = component
        p_left, p_top, p_right, p_bottom, p_area = previous
        vertical_gap = top - p_bottom
        horizontal_gap = max(0, max(p_left, left) - min(p_right, right))
        same_lane = abs(left - p_left) <= max_horizontal_gap or horizontal_gap <= max_horizontal_gap
        close = 0 <= vertical_gap <= max_vertical_gap and same_lane and max(right, p_right) - min(left, p_left) <= round(width * 0.72)
        if close:
            grouped[-1] = (
                min(p_left, left),
                min(p_top, top),
                max(p_right, right),
                max(p_bottom, bottom),
                p_area + area,
            )
        else:
            grouped.append(component)
    return grouped


def _neighbors(x: int, y: int, width: int, height: int):
    if x > 0:
        yield y * width + x - 1
    if x + 1 < width:
        yield y * width + x + 1
    if y > 0:
        yield (y - 1) * width + x
    if y + 1 < height:
        yield (y + 1) * width + x


def _merge_close_bubbles(bubbles: list[BubbleRegion], layout: LayoutRegions) -> list[BubbleRegion]:
    if len(bubbles) < 2:
        return bubbles
    merged: list[BubbleRegion] = []
    current = bubbles[0]
    for bubble in bubbles[1:]:
        vertical_gap = bubble.rect.top - current.rect.bottom
        horizontal_overlap = min(current.rect.right, bubble.rect.right) - max(current.rect.left, bubble.rect.left)
        if 0 <= vertical_gap <= max(5, round(layout.message_rect.height * 0.012)) and horizontal_overlap > 0:
            rect = Rect(
                min(current.rect.left, bubble.rect.left),
                min(current.rect.top, bubble.rect.top),
                max(current.rect.right, bubble.rect.right),
                max(current.rect.bottom, bubble.rect.bottom),
            )
            current = BubbleRegion(
                rect=rect,
                speaker=current.speaker,
                confidence=max(current.confidence, bubble.confidence),
                partial=current.partial or bubble.partial,
                profile=current.profile,
            )
        else:
            merged.append(current)
            current = bubble
    merged.append(current)
    return merged


def _overlap_ratio(left: Rect, right: Rect) -> float:
    overlap_left = max(left.left, right.left)
    overlap_top = max(left.top, right.top)
    overlap_right = min(left.right, right.right)
    overlap_bottom = min(left.bottom, right.bottom)
    area = max(0, overlap_right - overlap_left) * max(0, overlap_bottom - overlap_top)
    return area / max(1, left.width * left.height)


def _nearby_score(left: Rect, right: Rect) -> float:
    vertical_overlap = max(0, min(left.bottom, right.bottom) - max(left.top, right.top))
    vertical_score = vertical_overlap / max(1, min(left.height, right.height))
    if vertical_score < 0.35:
        return 0.0
    center_gap = abs((left.left + left.right) / 2 - (right.left + right.right) / 2)
    allowed_gap = max(36, right.width * 0.65)
    horizontal_score = max(0.0, 1.0 - center_gap / allowed_gap)
    edge_gap = max(0, max(left.left, right.left) - min(left.right, right.right))
    edge_score = max(0.0, 1.0 - edge_gap / max(24, right.width * 0.25))
    return max(horizontal_score, edge_score) * vertical_score
