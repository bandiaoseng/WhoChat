"""Recover the true WeChat layout from the captured screenshot.

The geometry prior in :mod:`whochat.platform.adapters` is a ratio profile.  It
is close enough for medium windows but drifts on real WeChat 4.x frames: on a
974x1047 window the estimated conversation surface started ~74px too far right,
which silently clipped the first characters of every contact / group name
(for example, a conversation title lost its leading characters).

Rather than tuning more ratios, this module measures the three boundaries that
matter directly from the pixels:

* the chat-list / conversation divider (title band, bright surface to the right)
* the dead band that some window frames expose on the right edge
* the top edge of the input box (long bright run anchored to the bottom)

Every measurement is validated and falls back to the adapter estimate when the
evidence is weak, so a misdetection can never make the layout worse than today.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from whochat.core.runtime import LayoutRegions, Rect
from whochat.vision.visual import BRIGHT_LUMINANCE, VisualContext

# Dead (never painted) columns are this dark on average.
DEAD_COLUMN_MEAN = 26.0
DEAD_BAND_MIN_COLUMNS = 8
DEAD_BAND_MAX_SHARE = 0.40
CONTENT_REFERENCE_MIN = 120.0

# Divider detection.
HEADER_REFERENCE_SHARE = 0.55
HEADER_ROW_BRIGHT_FRACTION = 0.85
DIVIDER_BAND_INSET = 0.10
DIVIDER_BRIGHT_FRACTION = 0.50
DIVIDER_PROBE_RUN = 80
DIVIDER_MAX_SHIFT_SHARE = 0.30
MIN_CHAT_LIST_WIDTH = 96

# Dark theme uses two dark surfaces instead of a bright conversation surface,
# so the light-theme ``bright pixel`` walk above has no signal.  In that case
# look for a sustained luminance step between two locally flat runs.  The run
# length is derived from the window width and deliberately spans more pixels
# than a normal title glyph, which keeps nickname strokes from becoming a fake
# sidebar divider.
DARK_DIVIDER_MIN_CONTRAST = 12.0
DARK_DIVIDER_RUN_MIN = 8
DARK_DIVIDER_RUN_MAX = 18
DARK_DIVIDER_MAX_MAD = 8.0
DARK_DIVIDER_EDGE_SHARE = 0.55
DARK_DIVIDER_TITLE_TOP_SHARE = 0.42
DARK_DIVIDER_TITLE_BOTTOM_SHARE = 0.68

# Input box detection. A real WeChat input divider is a thin, nearly full-width
# line with locally flat surfaces immediately above and below it. The previous
# bottom-anchored bright-run detector could start inside a large light bubble
# and move the boundary upward through the final lines of a message.
INPUT_SEARCH_FROM_BOTTOM_SHARE = 0.45
INPUT_MIN_HEIGHT_SHARE = 0.10
INPUT_MAX_SHIFT_SHARE = 0.30
INPUT_SEPARATOR_SIDE_ROWS = 7
INPUT_SEPARATOR_MAX_ROWS = 3
INPUT_LIGHT_SIDE_BRIGHT = 0.88
INPUT_LIGHT_LINE_BRIGHT = 0.18
INPUT_DARK_SIDE_MAX_DIFFERENCE = 14.0
INPUT_DARK_SIDE_MAX_MAD = 5.0
INPUT_DARK_LINE_MIN_CONTRAST = 7.0


def refine_layout_from_image(
    image_path: str | Path | None,
    layout: LayoutRegions | None,
) -> LayoutRegions:
    """Return ``layout`` with screenshot-measured boundaries applied."""
    if layout is None:
        return layout  # type: ignore[return-value]
    visual = VisualContext.load(image_path)
    if visual is None:
        return layout
    rect = layout.window_rect
    if rect.width < 320 or rect.height < 320:
        return layout
    try:
        return _refine(visual, layout)
    except (ValueError, IndexError, ZeroDivisionError):
        return layout


def _refine(visual: VisualContext, layout: LayoutRegions) -> LayoutRegions:
    layout = _refine_native_title_boundary(visual, layout)
    origin_left = layout.window_rect.left
    origin_top = layout.window_rect.top
    title = _to_image(layout.title_rect, origin_left, origin_top)
    chat_list = _to_image(layout.chat_list_rect, origin_left, origin_top)
    content = _to_image(layout.content_rect, origin_left, origin_top)
    input_rect = _to_image(layout.input_rect, origin_left, origin_top)
    if content.width < 200 or content.height < 200:
        return layout

    content_right = _detect_content_right(visual, content, chat_list)
    content_left = _detect_content_left(visual, title, content, chat_list, content_right)
    input_top = _detect_input_top(visual, content, input_rect, content_left, content_right)
    separator_probe = Rect(content_left + 4, content.top, content_right - 4, content.bottom)
    boundary_verified = any(
        visual.horizontal_separator_support(separator_probe, input_top + delta, run=run) >= 0.85
        for delta in (-1, 0, 1) for run in (1, 2, 3)
    )
    panel_proof = False
    if not boundary_verified and layout.target_app.value == "wechat" and layout.client_rect is not None:
        panel_top = _closed_native_input_panel_top(
            visual, content, input_rect, content_left, content_right,
            _to_image(layout.client_rect, origin_left, origin_top),
        )
        if panel_top is not None:
            input_top, boundary_verified, panel_proof = panel_top, True, True

    left_shift = content.left - content_left
    right_shift = content.right - content_right
    height_shift = input_rect.top - input_top
    if not left_shift and not right_shift and not height_shift:
        return replace(layout, input_boundary_verified=boundary_verified,
            reason=layout.reason + ("；native client 内闭合输入面板像素边界" if panel_proof else ""))

    notes: list[str] = ["native client 内闭合输入面板像素边界"] if panel_proof else []
    updated = replace(layout, input_boundary_verified=boundary_verified)
    if left_shift:
        notes.append(f"内容区左边界由 {layout.content_rect.left} 校正为 {layout.content_rect.left - left_shift}")
        updated = replace(
            updated,
            chat_list_rect=Rect(
                updated.chat_list_rect.left,
                updated.chat_list_rect.top,
                updated.chat_list_rect.right - left_shift,
                updated.chat_list_rect.bottom,
            ),
            content_rect=_move_left_edge(updated.content_rect, left_shift),
            title_rect=_move_left_edge(updated.title_rect, left_shift),
            message_rect=_move_left_edge(updated.message_rect, left_shift),
            input_rect=_move_left_edge(updated.input_rect, left_shift),
        )
    if right_shift:
        notes.append(f"内容区右边界由 {layout.content_rect.right} 校正为 {layout.content_rect.right - right_shift}")
        updated = replace(
            updated,
            content_rect=Rect(
                updated.content_rect.left,
                updated.content_rect.top,
                updated.content_rect.right - right_shift,
                updated.content_rect.bottom,
            ),
            title_rect=Rect(
                updated.title_rect.left,
                updated.title_rect.top,
                updated.title_rect.right - right_shift,
                updated.title_rect.bottom,
            ),
            message_rect=Rect(
                updated.message_rect.left,
                updated.message_rect.top,
                updated.message_rect.right - right_shift,
                updated.message_rect.bottom,
            ),
            input_rect=Rect(
                updated.input_rect.left,
                updated.input_rect.top,
                updated.input_rect.right - right_shift,
                updated.input_rect.bottom,
            ),
        )
    if height_shift:
        notes.append(f"输入框上边界由 {layout.input_rect.top} 校正为 {layout.input_rect.top - height_shift}")
        updated = replace(
            updated,
            input_rect=Rect(
                updated.input_rect.left,
                updated.input_rect.top - height_shift,
                updated.input_rect.right,
                updated.input_rect.bottom,
            ),
            message_rect=Rect(
                updated.message_rect.left,
                updated.message_rect.top,
                updated.message_rect.right,
                updated.message_rect.bottom - height_shift,
            ),
        )
    if not notes:
        return updated
    return replace(updated, reason=f"{layout.reason}；截图几何校正：" + "；".join(notes))


def _refine_native_title_boundary(visual: VisualContext, layout: LayoutRegions) -> LayoutRegions:
    """Measure the native WeChat header separator independently of window height.

    A short window's ratio prior may end above the title text. Only a unique,
    nearly full-width thin separator in the DPI-bounded header band can move
    the title/message boundary. No text is borrowed from the chat list or
    previous customer, and this measurement grants no identity authority.
    """
    if (layout.target_app.value != "wechat" or layout.source.value != "auto"
            or layout.client_rect is None):
        return layout
    try:
        scale = float(layout.dpi_scale)
    except (TypeError, ValueError):
        return layout
    if not 0.75 <= scale <= 3.0:
        return layout
    outer, client = layout.window_rect, layout.client_rect
    if (client.left < outer.left or client.top < outer.top or client.right > outer.right
            or client.bottom > outer.bottom or client.width <= 0 or client.height <= 0
            or visual.width != outer.width or visual.height != outer.height):
        return layout
    left = max(client.left, layout.content_rect.left) - outer.left + 4
    right = min(client.right, layout.content_rect.right) - outer.left - 4
    top, bottom = client.top - outer.top, client.bottom - outer.top
    if right - left < round(240 * scale):
        return layout
    probe = Rect(left, top, right, bottom)
    start = top + round(48 * scale)
    end = min(top + round(112 * scale), layout.input_rect.top - outer.top - round(96 * scale))
    candidates = []
    pixels = visual._gray.load()
    for y in range(start, end):
        # A cropped top bubble can touch the header from below. Require a
        # nearly uniform line and contrast to the header across its FULL
        # width; also require both-side separator evidence over most columns.
        # A bubble edge or title glyph alone cannot meet the full-width test.
        if visual.horizontal_separator_support(probe, y, run=1) < 0.70:
            continue
        line_level = _median([pixels[x, y] for x in range(left, right)])
        support = sum(abs(pixels[x, y] - line_level) <= 2
            and abs(sum(pixels[x, row] for row in range(y - 4, y - 1)) / 3 - line_level) >= 5
            for x in range(left, right))
        if support / (right - left) >= 0.96:
            candidates.append(y)
    # Adjacent detections can describe the same 1–3px border; separated lines
    # are ambiguous and must not expand the title crop into chat content.
    if not candidates or candidates[-1] - candidates[0] > 2:
        return layout
    boundary = outer.top + candidates[0]
    if boundary <= layout.title_rect.top or boundary >= layout.message_rect.bottom:
        return layout
    if boundary == layout.title_rect.bottom and boundary == layout.message_rect.top:
        return layout
    return replace(layout,
        title_rect=Rect(layout.title_rect.left, layout.title_rect.top, layout.title_rect.right, boundary),
        message_rect=Rect(layout.message_rect.left, boundary, layout.message_rect.right, layout.message_rect.bottom),
        reason=layout.reason + "；native client 标题分隔线像素边界")


def _detect_content_right(visual: VisualContext, content: Rect, chat_list: Rect) -> int:
    """Trim a never-painted band on the right edge (some window frames expose one)."""
    band = Rect(content.left, content.top, content.right, content.bottom)
    means = visual.column_mean_luminance(band)
    if len(means) < 200:
        return content.right
    reference = _median(means[4:204])
    if reference < CONTENT_REFERENCE_MIN:
        return content.right
    right = content.right
    while right - 1 > content.left and means[(right - 1) - content.left] <= DEAD_COLUMN_MEAN:
        right -= 1
    trimmed = content.right - right
    if trimmed < DEAD_BAND_MIN_COLUMNS:
        return content.right
    if trimmed > content.width * DEAD_BAND_MAX_SHARE:
        return content.right
    if right - chat_list.left < MIN_CHAT_LIST_WIDTH + 120:
        return content.right
    return right


def _detect_content_left(
    visual: VisualContext,
    title: Rect,
    content: Rect,
    chat_list: Rect,
    content_right: int,
) -> int:
    """Locate the vertical chat-list / conversation divider inside the title band."""
    band = _header_band(visual, title, content, content_right)
    if band is None:
        inset = max(1, round(title.height * DIVIDER_BAND_INSET))
        band = Rect(title.left, title.top + inset, title.right, title.bottom - inset)
    if band.height < 6:
        return content.left
    x0 = chat_list.left + MIN_CHAT_LIST_WIDTH
    x1 = min(content_right, content.right)
    if x1 - x0 < DIVIDER_PROBE_RUN:
        return content.left
    fractions = visual.bright_column_fractions(Rect(x0, band.top, x1, band.bottom), BRIGHT_LUMINANCE)
    index = content.left - 1 - x0
    if index <= 0 or index >= len(fractions):
        return _detect_dark_content_left(visual, title, content, chat_list, x0, x1)
    while index > 0 and fractions[index] >= DIVIDER_BRIGHT_FRACTION:
        index -= 1
    if index <= 0:
        # The walk reached the left limit of the probe without ever meeting a
        # dim column, which means the chat list and the conversation header
        # share one continuous bright surface and the divider is simply not
        # distinguishable here.  Returning ``x0`` would invent a boundary at the
        # probe floor and drag chat-list pixels into the content crop (which also
        # makes an unrelated chat-list repaint look like a new conversation), so
        # the prior estimate is kept instead.
        return _detect_dark_content_left(visual, title, content, chat_list, x0, x1)
    candidate = x0 + index + 1
    shift = content.left - candidate
    if shift <= 0:
        return _detect_dark_content_left(visual, title, content, chat_list, x0, x1)
    if shift > content.width * DIVIDER_MAX_SHIFT_SHARE:
        return _detect_dark_content_left(visual, title, content, chat_list, x0, x1)
    if candidate < chat_list.left + MIN_CHAT_LIST_WIDTH:
        return _detect_dark_content_left(visual, title, content, chat_list, x0, x1)
    probe = fractions[index + 1 : index + 1 + DIVIDER_PROBE_RUN]
    if len(probe) < DIVIDER_PROBE_RUN or _median(probe) < DIVIDER_BRIGHT_FRACTION:
        return _detect_dark_content_left(visual, title, content, chat_list, x0, x1)
    return candidate


def _detect_dark_content_left(
    visual: VisualContext,
    title: Rect,
    content: Rect,
    chat_list: Rect,
    x0: int,
    x1: int,
) -> int:
    """Locate a dark-theme sidebar divider from two sustained surface tones.

    This is a conservative fallback: it only corrects the prior to the left,
    requires an abrupt edge plus low-variance plateaus on both sides, and keeps
    all of the same maximum-shift / minimum-sidebar guards as the light-theme
    path.  Weak evidence therefore leaves the ratio prior unchanged.
    """
    top = title.top + round(title.height * DARK_DIVIDER_TITLE_TOP_SHARE)
    bottom = title.top + round(title.height * DARK_DIVIDER_TITLE_BOTTOM_SHARE)
    if bottom - top < 6:
        return content.left
    right = min(x1, content.left)
    if right - x0 < DARK_DIVIDER_RUN_MIN * 2 + 2:
        return content.left
    means = visual.column_mean_luminance(Rect(x0, top, right, bottom))
    if len(means) < DARK_DIVIDER_RUN_MIN * 2 + 2:
        return content.left
    run = max(DARK_DIVIDER_RUN_MIN, min(DARK_DIVIDER_RUN_MAX, round(content.width * 0.025)))
    # Do not reinterpret light-theme text edges.  The columns immediately left
    # of the prior boundary are part of the conversation header on both the
    # real frame and the ratio prior, so they are a safe theme reference.
    if _median(means[-run:]) >= CONTENT_REFERENCE_MIN:
        return content.left
    best: tuple[float, int] | None = None
    for index in range(run, len(means) - run):
        candidate = x0 + index
        shift = content.left - candidate
        if shift <= 1 or shift > content.width * DIVIDER_MAX_SHIFT_SHARE:
            continue
        if candidate < chat_list.left + MIN_CHAT_LIST_WIDTH:
            continue
        left_run = means[index - run : index]
        right_run = means[index : index + run]
        left_level = _median(left_run)
        right_level = _median(right_run)
        contrast = abs(right_level - left_level)
        if contrast < DARK_DIVIDER_MIN_CONTRAST:
            continue
        left_mad = _median([abs(value - left_level) for value in left_run])
        right_mad = _median([abs(value - right_level) for value in right_run])
        if left_mad > DARK_DIVIDER_MAX_MAD or right_mad > DARK_DIVIDER_MAX_MAD:
            continue
        edge = abs(means[index] - means[index - 1])
        if edge < contrast * DARK_DIVIDER_EDGE_SHARE:
            continue
        score = contrast + edge - left_mad - right_mad
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best is not None else content.left


def _header_band(visual: VisualContext, title: Rect, content: Rect, content_right: int) -> Rect | None:
    """Rows of the title band that carry the conversation header surface.

    The top of a WeChat frame can hold a grey window-chrome strip (luminance
    ~224) before the white header starts, and the frame may expose a dead band
    on the right.  Both dilute the column profile, so the band is narrowed to
    the rows that are bright across the right half of the *live* content, where
    the contact name never reaches.
    """
    reference_left = content.left + round(content.width * HEADER_REFERENCE_SHARE)
    right = min(content_right, content.right)
    if right - reference_left < 120 or title.height < 8:
        return None
    fractions = visual.bright_row_fractions(
        Rect(reference_left, title.top, right, title.bottom),
        BRIGHT_LUMINANCE,
    )
    if not fractions:
        return None
    best_start = 0
    best_length = 0
    start: int | None = None
    for index, value in enumerate([*fractions, 0.0]):
        if value >= HEADER_ROW_BRIGHT_FRACTION:
            if start is None:
                start = index
            continue
        if start is not None:
            length = index - start
            if length > best_length:
                best_start, best_length = start, length
            start = None
    if best_length < 6:
        return None
    return Rect(reference_left, title.top + best_start, right, title.top + best_start + best_length)


def _detect_input_top(
    visual: VisualContext,
    content: Rect,
    input_rect: Rect,
    content_left: int,
    content_right: int,
) -> int:
    """Find the thin full-width separator above the input box.

    Light theme uses bright-row fractions and requires bright surfaces on both
    sides of a very dark, one-to-three-row line. Dark theme mirrors the rule
    with local mean-luminance plateaus. Weak evidence always keeps the geometry
    prior; message bubbles and input text cannot become divider evidence.
    """
    left = content_left + 4
    right = content_right - 4
    if right - left < 120:
        return input_rect.top
    probe = Rect(left, content.top, right, content.bottom)
    fractions = visual.bright_row_fractions(probe, BRIGHT_LUMINANCE)
    if len(fractions) < 200:
        return input_rect.top
    search_start = round(len(fractions) * (1.0 - INPUT_SEARCH_FROM_BOTTOM_SHARE))
    search_end = len(fractions) - INPUT_SEPARATOR_SIDE_ROWS - INPUT_SEPARATOR_MAX_ROWS
    light_candidates: list[int] = []
    for offset in range(search_start + INPUT_SEPARATOR_SIDE_ROWS, search_end):
        before = fractions[offset - INPUT_SEPARATOR_SIDE_ROWS : offset]
        for run in range(1, INPUT_SEPARATOR_MAX_ROWS + 1):
            after = fractions[offset + run : offset + run + INPUT_SEPARATOR_SIDE_ROWS]
            line = fractions[offset : offset + run]
            if (
                _median(before) >= INPUT_LIGHT_SIDE_BRIGHT
                and _median(after) >= INPUT_LIGHT_SIDE_BRIGHT
                and max(line) <= INPUT_LIGHT_LINE_BRIGHT
            ):
                candidate = content.top + offset
                if _valid_input_candidate(candidate, content, input_rect):
                    light_candidates.append(candidate)
                break
    if light_candidates:
        return min(light_candidates, key=lambda value: abs(value - input_rect.top))

    means = visual.row_mean_luminance(probe)
    dark_candidates: list[tuple[float, int]] = []
    for offset in range(search_start + INPUT_SEPARATOR_SIDE_ROWS, search_end):
        before = means[offset - INPUT_SEPARATOR_SIDE_ROWS : offset]
        after = means[offset + 1 : offset + 1 + INPUT_SEPARATOR_SIDE_ROWS]
        before_level = _median(before)
        after_level = _median(after)
        if abs(before_level - after_level) > INPUT_DARK_SIDE_MAX_DIFFERENCE:
            continue
        before_mad = _median([abs(value - before_level) for value in before])
        after_mad = _median([abs(value - after_level) for value in after])
        if before_mad > INPUT_DARK_SIDE_MAX_MAD or after_mad > INPUT_DARK_SIDE_MAX_MAD:
            continue
        contrast = min(abs(means[offset] - before_level), abs(means[offset] - after_level))
        if contrast < INPUT_DARK_LINE_MIN_CONTRAST:
            continue
        candidate = content.top + offset
        if (_valid_input_candidate(candidate, content, input_rect)
                and visual.horizontal_separator_support(probe, candidate) >= 0.85):
            dark_candidates.append((contrast, candidate))
    if dark_candidates:
        dark_candidates.sort(key=lambda item: (-item[0], abs(item[1] - input_rect.top)))
        return dark_candidates[0][1]
    return input_rect.top


def _valid_input_candidate(candidate: int, content: Rect, input_rect: Rect) -> bool:
    if candidate - content.top < content.height * INPUT_MIN_HEIGHT_SHARE:
        return False
    if abs(candidate - input_rect.top) > content.height * INPUT_MAX_SHIFT_SHARE:
        return False
    return candidate < input_rect.bottom - 40


def _closed_native_input_panel_top(visual, content, input_rect, content_left, content_right, client):
    """A separate closed-panel proof, not a weaker separator threshold.

    A photo can touch the editor top, invalidating the old two-sided plateau.
    Require the full OS-owned client, a long neutral top border, a bright
    interior below it, BOTH continuous sides and a closing bottom border near
    the client bottom. Clipped/missing/ambiguous rectangles grant nothing.
    """
    if (client.left < 0 or client.top < 0 or client.right > visual.width or client.bottom > visual.height
            or content_right != client.right or content.bottom != client.bottom
            or input_rect.bottom != client.bottom or content_left < client.left
            or content_right-content_left < 240):
        return None
    rgb = visual._rgb
    def neutral(pixel, level, tolerance=5):
        return max(pixel)-min(pixel) <= 8 and abs(sum(pixel)/3-level) <= tolerance
    def share(points, level, tolerance=5):
        values = [rgb.getpixel(point) for point in points]
        return sum(neutral(pixel,level,tolerance) for pixel in values)/max(1,len(values))
    start = max(content.top+round(content.height*.55), input_rect.top-round(content.height*.30))
    stop = min(input_rect.bottom-80, input_rect.top+round(content.height*.30))
    inner_left, inner_right = content_left+16, content_right-16
    if inner_right-inner_left < 200:
        return None
    matches=[]
    for top in range(start, stop):
        if not _valid_input_candidate(top,content,input_rect):
            continue
        row=[rgb.getpixel((x,top)) for x in range(inner_left,inner_right)]
        levels=sorted(sum(pixel)/3 for pixel in row)
        level=levels[len(levels)//2]
        if not 190 <= level <= 244 or share(((x,top) for x in range(inner_left,inner_right)),level) < .95:
            continue
        inside=[rgb.getpixel((x,y)) for y in (top+4,top+6,top+8) for x in range(inner_left,inner_right)]
        bright=sum(min(p)>=238 and max(p)-min(p)<=8 and sum(p)/3>=level+6 for p in inside)/len(inside)
        if bright < .96:
            continue
        for bottom in range(client.bottom-18,client.bottom-2):
            if not 80 <= bottom-top <= content.height*.40:
                continue
            if share(((x,bottom) for x in range(inner_left,inner_right)),level,7) < .95:
                continue
            # Closing bottom is visible, not the native frame or image edge.
            if bottom+2 >= client.bottom or share(((x,bottom+2) for x in range(inner_left,inner_right)),250,8) < .95:
                continue
            lefts=[x for x in range(content_left+3,content_left+13)
                if share(((x,y) for y in range(top+10,bottom-10)),level,7)>=.93
                and share(((x+3,y) for y in range(top+12,bottom-40)),250,12)>=.95]
            rights=[x for x in range(content_right-12,content_right-2)
                if share(((x,y) for y in range(top+10,bottom-10)),level,7)>=.93
                and share(((x-3,y) for y in range(top+12,bottom-40)),250,12)>=.95]
            if not lefts or not rights:
                continue
            for left in lefts:
                for right in rights:
                    if right-left < content_right-content_left-24:
                        continue
                    # Rounded corners must join top/bottom to each side. A
                    # free horizontal stroke above a card is not a panel.
                    corners=[]
                    for cx,cy,sx,sy in ((left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)):
                        rows=[any(neutral(rgb.getpixel((cx+sx*x,cy+sy*y)),level,10) for x in range(10)) for y in range(10)]
                        cols=[any(neutral(rgb.getpixel((cx+sx*x,cy+sy*y)),level,10) for y in range(10)) for x in range(10)]
                        corners.append(sum(rows)>=9 and sum(cols)>=9)
                    if all(corners):
                        matches.append(top)
    if not matches or max(matches)-min(matches)>3:
        return None
    return min(matches)


def _to_image(rect: Rect, origin_left: int, origin_top: int) -> Rect:
    return Rect(rect.left - origin_left, rect.top - origin_top, rect.right - origin_left, rect.bottom - origin_top)


def _move_left_edge(rect: Rect, amount: int) -> Rect:
    """Widen a rect to the left without moving its right edge."""
    return Rect(rect.left - amount, rect.top, rect.right, rect.bottom)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2
