from __future__ import annotations

import re
from dataclasses import dataclass, replace

from pathlib import Path
from PIL import Image

from whochat.core.models import Speaker
from whochat.core.runtime import LayoutRegions, PageClassification, PageType, Rect, TargetApp
from whochat.ocr.models import OcrRegion, OcrResult, OcrTextBox, ParsedOcrMessage
from whochat.vision.bubbles import (
    BubbleRegion, bubble_for_box, bubble_for_box_from_image, detect_bubbles,
    outgoing_avatar_text_regions,
    incoming_bubble_avatar_text_regions,
    incoming_transcribed_voice_metadata,
)
from whochat.vision.visual import KIND_DARK_OVERLAY, KIND_SECONDARY, KIND_UNKNOWN, VisualContext, load_rgb_image


def normalize_ocr_regions(result: OcrResult, layout: LayoutRegions) -> OcrResult:
    boxes = [replace(box, region=classify_box_region(box.rect, layout)) for box in result.boxes]
    return OcrResult(boxes=boxes, source_image=result.source_image, engine=result.engine, warning=result.warning)


def classify_box_region(rect: Rect, layout: LayoutRegions) -> OcrRegion:
    candidates = [
        (OcrRegion.TITLE, _overlap_ratio(rect, layout.title_rect)),
        (OcrRegion.CHAT_LIST, _overlap_ratio(rect, layout.chat_list_rect)),
        (OcrRegion.MESSAGE, _overlap_ratio(rect, layout.message_rect)),
        (OcrRegion.INPUT, _overlap_ratio(rect, layout.input_rect)),
    ]
    region, score = max(candidates, key=lambda item: item[1])
    return region if score >= 0.35 else OcrRegion.UNKNOWN


def classify_page_from_ocr(result: OcrResult, layout: LayoutRegions) -> PageClassification:
    normalized = normalize_ocr_regions(result, layout)
    message_boxes = [box for box in normalized.boxes if box.region == OcrRegion.MESSAGE and box.confidence >= 0.45]
    stable_message_boxes = [box for box in message_boxes if not _is_edge_partial_box(box, layout)]
    input_boxes = [box for box in normalized.boxes if box.region == OcrRegion.INPUT and box.confidence >= 0.4]
    title_boxes = [box for box in normalized.boxes if box.region == OcrRegion.TITLE and box.confidence >= 0.45]
    chat_list_boxes = [box for box in normalized.boxes if box.region == OcrRegion.CHAT_LIST and box.confidence >= 0.4]

    non_chat = _classify_non_chat_page(title_boxes, message_boxes, input_boxes)
    if non_chat is not None:
        return non_chat

    group_title = any(_looks_like_group_title(box.text) for box in title_boxes)
    # A textless picture chat is not an empty incoming message. Require
    # repeated positive avatar/card structure in addition to the ordinary
    # title/input boundaries; title + an input box alone never authorizes it.
    if (not message_boxes and not group_title and title_boxes and input_boxes
            and max(box.confidence for box in title_boxes) >= .65
            and any(_compact_text(box.text).startswith(("按住鼠标", "输入消息", "请输入消息"))
                    for box in input_boxes)
            and layout.input_boundary_verified and result.source_image):
        from whochat.ocr.media_regions import chat_image_regions
        ox, oy = layout.window_rect.left, layout.window_rect.top
        picture_layout = replace(layout, **{
            name: Rect(getattr(layout, name).left-ox, getattr(layout, name).top-oy,
                       getattr(layout, name).right-ox, getattr(layout, name).bottom-oy)
            for name in ("window_rect", "nav_rect", "chat_list_rect", "content_rect", "title_rect", "message_rect", "input_rect")
        })
        if chat_image_regions(result.source_image, picture_layout):
            return PageClassification(PageType.CHAT_DM, .70, "verified_complete_chat_photo")
        from whochat.vision.current_window_structure import textless_chat_structure, WINDOW_PAGE_REASON
        if textless_chat_structure(result.source_image, layout) is not None:
            return PageClassification(PageType.CHAT_DM, .70, WINDOW_PAGE_REASON)
    chat_evidence = _chat_page_evidence(layout, title_boxes, stable_message_boxes, input_boxes)
    if chat_evidence.strong:
        confidence = min(
            0.82,
            0.57
            + min(5, len(stable_message_boxes)) * 0.035
            + (0.06 if input_boxes else 0.0)
            + (0.05 if chat_evidence.bilateral else 0.0)
            + (0.04 if chat_evidence.bubble_count >= 2 else 0.0),
        )
        page_type = PageType.CHAT_GROUP if group_title else PageType.CHAT_DM
        return PageClassification(
            page_type,
            confidence,
            f"聊天证据优先：score={chat_evidence.score}，标题={bool(title_boxes)}，"
            f"输入区={bool(input_boxes)}，稳定消息={len(stable_message_boxes)}，"
            f"左右结构={chat_evidence.bilateral}，气泡={chat_evidence.bubble_count}"
            + ("；标题呈现群聊特征" if group_title else ""),
        )

    article = _classify_article_page(
        title_boxes,
        stable_message_boxes,
        input_boxes,
        layout,
        chat_evidence,
    )
    if article is not None:
        return article

    if len(stable_message_boxes) >= 2 and title_boxes:
        confidence = min(0.72, 0.54 + len(stable_message_boxes) * 0.05 + len(title_boxes) * 0.03)
        page_type = PageType.CHAT_GROUP if group_title else PageType.CHAT_DM
        return PageClassification(
            page_type,
            confidence,
            f"OCR 证据包含标题和 {len(stable_message_boxes)} 个稳定消息候选；输入区未识别到文字但不阻断聊天页判断"
            + ("；标题呈现群聊特征" if group_title else ""),
        )
    if len(stable_message_boxes) >= 3 and input_boxes:
        return PageClassification(
            PageType.CHAT_DM,
            0.67,
            f"OCR 证据包含输入区和 {len(stable_message_boxes)} 个稳定消息候选；标题漏检，按私聊候选处理",
        )
    if len(stable_message_boxes) >= 2 and (input_boxes or title_boxes):
        return PageClassification(
            PageType.UNKNOWN,
            0.58,
            "OCR 看到消息结构，但标题或输入区证据不足，暂不允许 AI",
        )
    if chat_list_boxes and not message_boxes:
        return PageClassification(
            PageType.SEARCH,
            0.46,
            "OCR 主要落在左侧列表，右侧聊天结构不足",
        )
    return PageClassification(PageType.UNKNOWN, 0.35, "OCR 证据不足，保持未知页面")


def account_identity_boundary_reason(
    result: OcrResult,
    layout: LayoutRegions,
    page: PageClassification,
) -> str:
    """Return only explicit, high-confidence login/account-switch evidence.

    Chat text may mention login or QR codes, so no marker can freeze an account
    while the frame still has a credible chat layout. Multiple UI-specific
    markers are required on a non-chat page.
    """

    if page.page_type in {PageType.CHAT_DM, PageType.CHAT_GROUP, PageType.FILE_HELPER}:
        return ""
    normalized = normalize_ocr_regions(result, layout)
    text = " ".join(
        box.text.strip()
        for box in normalized.boxes
        if box.confidence >= 0.60 and box.text.strip()
    )
    compact = _compact_text(text)
    if not compact:
        return ""
    qr_login = (
        ("扫码登录" in compact or "扫描二维码登录" in compact)
        and ("手机微信" in compact or "二维码" in compact)
    )
    account_switch = (
        "切换账号" in compact
        and ("登录微信" in compact or "扫码登录" in compact or "二维码" in compact)
    )
    logged_out = "登录微信" in compact and "二维码" in compact
    if account_switch:
        return "explicit_account_switch_page"
    if qr_login or logged_out:
        return "explicit_login_page"
    return ""


def _classify_non_chat_page(
    title_boxes: list[OcrTextBox],
    message_boxes: list[OcrTextBox],
    input_boxes: list[OcrTextBox],
) -> PageClassification | None:
    title_text = " ".join(box.text.strip() for box in title_boxes if box.text.strip())
    body_text = " ".join(box.text.strip() for box in message_boxes if box.text.strip())
    input_text = " ".join(box.text.strip() for box in input_boxes if box.text.strip())
    combined = " ".join([title_text, body_text, input_text])
    if not combined:
        return None
    if _contains_any(title_text, ("设置", "通用", "账号与安全", "消息通知", "隐私", "关于微信")):
        return PageClassification(PageType.SETTINGS, 0.74, "OCR 标题呈现设置页特征，阻断聊天回复")
    if _contains_any(title_text, ("订阅号", "公众号", "服务号")) or _contains_any(body_text, ("关注公众号", "进入公众号", "服务通知")):
        return PageClassification(PageType.OFFICIAL_ACCOUNT, 0.72, "OCR 呈现公众号或服务号特征，阻断聊天回复")
    if _contains_any(title_text, ("文件预览", "文档预览", "图片预览")):
        return PageClassification(PageType.FILE_PREVIEW, 0.74, "OCR 标题明确呈现文件预览页，阻断聊天回复")
    file_markers = ("用其他应用打开", "另存为", "文件大小", "下载文件")
    if not input_boxes and sum(marker in combined for marker in file_markers) >= 2:
        return PageClassification(PageType.FILE_PREVIEW, 0.70, "OCR 呈现多个文件预览操作，阻断聊天回复")
    if (
        not message_boxes
        and _contains_any(combined, ("搜索指定内容", "搜索聊天记录", "联系人", "群聊", "聊天记录"))
        and not input_boxes
    ):
        return PageClassification(PageType.SEARCH, 0.60, "OCR 呈现搜索或列表页特征，阻断聊天回复")
    return None


@dataclass(frozen=True)
class _ChatPageEvidence:
    score: int
    strong: bool
    bilateral: bool
    bubble_count: int


def _chat_page_evidence(
    layout: LayoutRegions,
    title_boxes: list[OcrTextBox],
    stable_message_boxes: list[OcrTextBox],
    input_boxes: list[OcrTextBox],
) -> _ChatPageEvidence:
    left, right, bubble_count = _message_side_evidence(stable_message_boxes, layout)
    bilateral = left and right
    score = 0
    score += 2 if title_boxes else 0
    score += 3 if input_boxes else 0
    score += 3 if len(stable_message_boxes) >= 2 else 0
    score += 1 if len(stable_message_boxes) >= 4 else 0
    score += 3 if bilateral else 0
    score += 3 if bubble_count >= 2 else 0
    score += 1 if bilateral else 0
    strong = len(stable_message_boxes) >= 2 and score >= 8 and (
        bool(input_boxes) and (bool(title_boxes) or bilateral or bubble_count >= 2)
        or bilateral and (bool(title_boxes) or bubble_count >= 2)
        or bubble_count >= 2 and bool(title_boxes)
    )
    return _ChatPageEvidence(score=score, strong=strong, bilateral=bilateral, bubble_count=bubble_count)


def _message_side_evidence(boxes: list[OcrTextBox], layout: LayoutRegions) -> tuple[bool, bool, int]:
    """Return left/right lanes and bubble-like OCR geometry without rescanning pixels."""
    left = False
    right = False
    bubble_like = 0
    for box in boxes:
        width_ratio = box.rect.width / max(1, layout.message_rect.width)
        center_ratio = _horizontal_ratio(box, layout)
        left_ratio = _edge_ratio(box.rect.left, layout)
        right_ratio = _edge_ratio(box.rect.right, layout)
        left_like = width_ratio <= 0.72 and left_ratio <= 0.25 and center_ratio <= 0.55
        right_like = width_ratio <= 0.72 and right_ratio >= 0.85 and center_ratio >= 0.55
        if left_like:
            left = True
        if right_like:
            right = True
        if left_like or right_like:
            bubble_like += 1
    return left, right, bubble_like


_ARTICLE_MARKER_GROUPS: dict[str, tuple[str, ...]] = {
    "metadata": ("作者", "发布于"),
    "navigation": ("阅读全文", "原文链接"),
    "engagement": ("阅读量", "在看", "点赞", "赞"),
    "sharing": ("分享",),
}


def _classify_article_page(
    title_boxes: list[OcrTextBox],
    message_boxes: list[OcrTextBox],
    input_boxes: list[OcrTextBox],
    layout: LayoutRegions,
    chat: _ChatPageEvidence,
) -> PageClassification | None:
    """Classify an article only from several independent layout/content signals.

    Ordinary customer conversations frequently contain words such as “分享”,
    “作者” or “发布于”. Those words are hints, never a verdict. A news page
    must also look like a document: multiple marker groups, several long wide
    blocks, a title band, and no normal chat input or bilateral message layout.
    """
    if not title_boxes or input_boxes or chat.bilateral or chat.bubble_count >= 2:
        return None
    body_text = " ".join(box.text.strip() for box in message_boxes if box.text.strip())
    if len(body_text) < 80:
        return None
    marker_groups = {
        name
        for name, markers in _ARTICLE_MARKER_GROUPS.items()
        if _contains_any(body_text, markers)
    }
    if len(marker_groups) < 2 or not marker_groups.intersection({"metadata", "navigation"}):
        return None
    long_blocks = [box for box in message_boxes if len(_compact_text(box.text)) >= 24]
    wide_blocks = [
        box
        for box in message_boxes
        if box.rect.width / max(1, layout.message_rect.width) >= 0.52
    ]
    if len(long_blocks) < 2 or len(wide_blocks) < 2:
        return None
    score = (
        len(marker_groups) * 2
        + 2  # title band
        + 2  # no chat input
        + 2  # no bilateral chat structure
        + 2  # multiple long text blocks
        + 2  # multiple wide document-like blocks
        + (1 if len(body_text) >= 120 else 0)
    )
    if score < 12:
        return None
    confidence = min(0.88, 0.52 + score * 0.02)
    return PageClassification(
        PageType.NEWS_ARTICLE,
        confidence,
        f"文章证据联合评分={score}：特征组={','.join(sorted(marker_groups))}，"
        f"长文本块={len(long_blocks)}，宽文本块={len(wide_blocks)}，无输入框且无左右聊天结构",
    )


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


@dataclass(frozen=True)
class MessagePartition:
    """Result of splitting the message strip into roles."""

    messages: list[OcrTextBox]
    anchors: set[OcrTextBox]
    times: dict[OcrTextBox, str]
    labels: set[OcrTextBox]
    senders: dict[OcrTextBox, str]
    durations: list[OcrTextBox]

    @property
    def blockers(self) -> set[OcrTextBox]:
        return self.anchors | self.labels


_MEDIA_DURATION_PATTERN = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")

MIN_MEDIA_DURATION_SHARE = 0.45


_FILE_CARD_NAME = re.compile(r"\.(docx?|xlsx?|pptx?|pdf)$", re.IGNORECASE)
_FILE_CARD_SIZE = re.compile(r"^\d+(?:\.\d+)?\s*(?:[KMGT](?:B)?|B|字节)$", re.IGNORECASE)


@dataclass(frozen=True)
class _NativeFileCard:
    filename_boxes: tuple[OcrTextBox, ...]
    metadata_boxes: frozenset[OcrTextBox]
    speaker: Speaker

    @property
    def filename(self) -> str:
        return "".join(box.text.strip() for box in self.filename_boxes)


def _native_file_cards(boxes: list[OcrTextBox], layout: LayoutRegions, source_image: str | Path | Image.Image | None) -> list[_NativeFileCard]:
    """Recognize a native file card, never a lone low-confidence letter.

    A supported filename extension, size, WeChat source label, matching
    coloured file icon, compact card geometry and neutral card canvas are
    all mandatory. Only metadata inside that proved structure is omitted;
    the filename remains a message with its original OCR confidence.
    """
    if not source_image or layout.target_app.value != "wechat":
        return []
    try:
        image = load_rgb_image(source_image)
        if image is None:
            return []
        if image.size != (layout.window_rect.width, layout.window_rect.height):
            return []
    except (OSError, ValueError):
        return []
    supported_icons = {"doc": "w", "docx": "w", "xls": "x", "xlsx": "x",
                       "ppt": "p", "pptx": "p", "pdf": "pdf"}
    cards, used = [], set()
    ordered = sorted(boxes, key=lambda box: (box.rect.top, box.rect.left))
    for tail in ordered:
        match = _FILE_CARD_NAME.search(tail.text.strip())
        if tail in used or match is None or tail.confidence < .65:
            continue
        extension = match.group(1).casefold()
        names = [tail]
        # WeChat wraps a filename into consecutive left-aligned lines. No
        # preceding chat bubble or time/source line may join the filename.
        for previous in reversed(ordered[:ordered.index(tail)]):
            gap = names[0].rect.top - previous.rect.bottom
            if gap > max(12, round(tail.rect.height * .85)):
                break
            if (0 <= gap and abs(previous.rect.left - tail.rect.left) <= 5
                    and previous.confidence >= .65 and len(previous.text.strip()) >= 4
                    and not _looks_like_time_text(previous.text.strip())
                    and not _FILE_CARD_SIZE.fullmatch(previous.text.strip())
                    and len(names) < 3):
                names.insert(0, previous)
        filename_rect = _union_box_rects(names)
        if len("".join(box.text.strip() for box in names)) < 6:
            continue
        sizes = [box for box in ordered if box not in used and box.confidence >= .65
                 and _FILE_CARD_SIZE.fullmatch(box.text.strip())
                 and abs(box.rect.left - filename_rect.left) <= 12
                 and 0 <= box.rect.top - filename_rect.bottom <= 24]
        if len(sizes) != 1:
            continue
        size = sizes[0]
        sources = [box for box in ordered if box not in used and box.confidence >= .65
                   and re.sub(r"\s+", "", box.text) == "微信电脑版"
                   and 0 <= box.rect.top - size.rect.bottom <= 36
                   and -8 <= box.rect.left - filename_rect.left <= 42]
        if len(sources) != 1:
            continue
        source = sources[0]
        icons = [box for box in ordered if box not in used
                 and box.text.strip().casefold() == supported_icons[extension]
                 and filename_rect.right + 4 <= box.rect.left <= filename_rect.right + 90
                 and filename_rect.top - 8 <= box.rect.top <= filename_rect.bottom + 12
                 and 8 <= box.rect.width <= 48 and 8 <= box.rect.height <= 42]
        if len(icons) != 1:
            continue
        icon = icons[0]
        card = Rect(filename_rect.left - 12, filename_rect.top - 12,
                    max(filename_rect.right, icon.rect.right) + 12, source.rect.bottom + 12)
        if (card.left < layout.message_rect.left or card.right > layout.message_rect.right
                or card.top < layout.message_rect.top or card.bottom > layout.message_rect.bottom
                or card.height > 170 or card.width > layout.message_rect.width * .80):
            continue
        if not _file_card_pixels_match(image, card, icon.rect, extension):
            continue
        left_gap, right_gap = card.left - layout.message_rect.left, layout.message_rect.right - card.right
        if left_gap >= layout.message_rect.width * .20 and right_gap <= max(96, layout.message_rect.width * .22):
            speaker = Speaker.ME
        elif left_gap <= max(64, layout.message_rect.width * .17) and right_gap >= layout.message_rect.width * .25:
            speaker = Speaker.OTHER
        else:
            from whochat.ocr.media_regions import native_file_card_avatar_speaker
            speaker = native_file_card_avatar_speaker(image, layout, card)
            if speaker is None:
                continue
        metadata = frozenset((size, source, icon))
        card_boxes = set(names) | set(metadata)
        if any(box not in card_boxes and box.rect.left >= card.left and box.rect.right <= card.right
               and box.rect.top >= card.top and box.rect.bottom <= card.bottom for box in boxes):
            # Unknown extra text inside a candidate is not safe to discard.
            continue
        cards.append(_NativeFileCard(tuple(names), metadata, speaker))
        used.update(card_boxes)
    return cards


def _union_box_rects(boxes) -> Rect:
    return Rect(min(box.rect.left for box in boxes), min(box.rect.top for box in boxes),
                max(box.rect.right for box in boxes), max(box.rect.bottom for box in boxes))


def _file_card_pixels_match(image, card: Rect, icon: Rect, extension: str) -> bool:
    if card.left < 0 or card.top < 0 or card.right > image.width or card.bottom > image.height:
        return False
    crop = image.crop(card.as_tuple())
    colors = crop.getcolors(crop.width * crop.height) or []
    # The native card, unlike a green outgoing text bubble, has a substantial
    # exactly-flat neutral surface. The document icon must independently have
    # the expected Office/PDF colour; text saying "w" is not such an icon.
    dominant_neutral = max((count for count, color in colors
                            if max(color) - min(color) <= 5 and 35 <= sum(color) / 3 <= 250), default=0)
    if dominant_neutral < crop.width * crop.height * .45:
        return False
    rect = (max(card.left, icon.left - 4), max(card.top, icon.top - 6),
            min(card.right, icon.right + 4), min(card.bottom, icon.bottom + 6))
    pixels = list(image.crop(rect).getdata())
    if not pixels:
        return False
    def colour(value):
        red, green, blue = value
        if extension in {"doc", "docx"}:
            return blue >= 130 and blue - red >= 55 and blue - green >= 25
        if extension in {"xls", "xlsx"}:
            return green >= 85 and green - red >= 25 and green - blue >= 15
        return red >= 140 and red - green >= 35 and red - blue >= 35
    return sum(colour(value) for value in pixels) >= len(pixels) * .35


def parse_visible_messages(result: OcrResult, layout: LayoutRegions, *, frame_image: Image.Image | None = None) -> list[ParsedOcrMessage]:
    return parse_visible_messages_with_evidence(result, layout, frame_image=frame_image)[0]


def parse_visible_messages_with_evidence(result: OcrResult, layout: LayoutRegions, *, direct_chat: bool = False,
                                         frame_image: Image.Image | None = None) -> tuple[list[ParsedOcrMessage], dict]:
    """Parse once and account for every recognized message-region box.

    This is evidence about recognized pixels, not proof of OCR accuracy or
    unseen history. Dropped low-confidence text and unexplained grey text
    invalidate completeness instead of silently exposing an older tail.

    ``frame_image`` is the caller's current window frame, never retained or
    mutated here. Otherwise decode ``source_image`` once for this call only.
    A failed decode keeps the existing geometry fallback without retrying the
    same file for each OCR box. Coordinate normalization shares the same frame.
    """
    rgb = load_rgb_image(frame_image if frame_image is not None else result.source_image)
    try:
        return _parse_visible_messages_with_evidence(result, layout, direct_chat=direct_chat, frame_image=rgb)
    finally:
        if rgb is not None and rgb is not frame_image:
            rgb.close()


def _parse_visible_messages_with_evidence(result: OcrResult, layout: LayoutRegions, *, direct_chat: bool,
                                          frame_image: Image.Image | None) -> tuple[list[ParsedOcrMessage], dict]:
    origin_x, origin_y = layout.window_rect.left, layout.window_rect.top
    if origin_x or origin_y:
        # Pipeline boxes/layout use screen coordinates; source_image is the
        # captured WINDOW image whose pixels start at (0, 0). The previous
        # pixel helpers sampled outside that image after moving WeChat to the
        # right/another monitor, losing bubble colours and speaker identity.
        def local_rect(rect: Rect) -> Rect:
            return Rect(rect.left - origin_x, rect.top - origin_y,
                        rect.right - origin_x, rect.bottom - origin_y)

        local_layout = replace(layout, **{
            name: local_rect(getattr(layout, name)) for name in
            ("window_rect", "nav_rect", "chat_list_rect", "content_rect", "title_rect", "message_rect", "input_rect")
        })
        local_result = replace(result, boxes=[replace(box, rect=local_rect(box.rect)) for box in result.boxes])
        parsed, report = _parse_visible_messages_with_evidence(local_result, local_layout,
                                                               direct_chat=direct_chat, frame_image=frame_image)
        return [replace(message, rect=Rect(message.rect.left + origin_x, message.rect.top + origin_y,
                                         message.rect.right + origin_x, message.rect.bottom + origin_y))
                for message in parsed], report
    normalized = normalize_ocr_regions(result, layout)
    source_image = frame_image
    all_message_boxes = [box for box in normalized.boxes if box.region == OcrRegion.MESSAGE and box.text.strip()]
    history_evidence = (_history_navigation_pixel_evidence(source_image, layout, all_message_boxes)
                        if direct_chat else {})
    history_present = bool(history_evidence)
    history_boxes = {box for box in all_message_boxes if history_present
        and re.sub(r"\s+", "", box.text) in {"跳转到最新消息", "跳至最新消息"}
        and _overlap_ratio(box.rect, Rect(*history_evidence["rect"])) >= .90}
    avatar_regions = outgoing_avatar_text_regions(
        source_image, [box.rect for box in all_message_boxes], layout,
    ) if source_image else set()
    avatar_boxes = {box for box in all_message_boxes if box.rect in avatar_regions}
    if direct_chat and source_image:
        incoming_avatar_regions = incoming_bubble_avatar_text_regions(
            source_image, [box.rect for box in all_message_boxes], layout,
        )
        avatar_boxes |= {box for box in all_message_boxes if box.rect in incoming_avatar_regions}
    unresolved_voice_regions: set[Rect] = set()
    complete_transcript_regions: set[Rect] = set()
    voice_regions, voice_avatar_regions = (incoming_transcribed_voice_metadata(
        source_image, [(box.rect, box.text) for box in all_message_boxes], layout,
        unresolved_voice_regions=unresolved_voice_regions,
        complete_transcript_regions=complete_transcript_regions,
    ) if direct_chat and source_image else (set(), set()))
    voice_boxes = {box for box in all_message_boxes if box.rect in voice_regions}
    voice_avatar_boxes = {box for box in all_message_boxes if box.rect in voice_avatar_regions}
    avatar_boxes |= voice_avatar_boxes
    from whochat.ocr.native_cards import mini_program_cards
    app_cards = mini_program_cards(source_image, layout, all_message_boxes) if direct_chat else ()
    app_boxes = {box for card in app_cards for box in card.boxes}
    # Pictures are occurrences, not chat sentences made out of the lettering
    # printed inside them. Recognition is strictly structural and deliberately
    # makes no claim about products, crops, pests or image meaning.
    from whochat.ocr.media_regions import chat_image_regions
    media_regions = chat_image_regions(source_image, layout, all_message_boxes) if direct_chat else ()
    if direct_chat and source_image and media_regions:
        image_avatar_regions = incoming_bubble_avatar_text_regions(
            source_image, [box.rect for box in all_message_boxes], layout,
            image_cards=tuple(region.rect for region in media_regions
                              if region.speaker == Speaker.OTHER and not region.quoted),
        )
        avatar_boxes |= {box for box in all_message_boxes if box.rect in image_avatar_regions}
    media_boxes = {
        box for box in all_message_boxes
        if any((region.rect.left-4 <= box.rect.left and box.rect.right <= region.rect.right+4
                and region.rect.top-4 <= box.rect.top and box.rect.bottom <= region.rect.bottom+4
                and _overlap_ratio(box.rect,region.rect)>=.90)
               or box.rect in region.caption_rects
               for region in media_regions)
    }
    file_cards = _native_file_cards(all_message_boxes, layout, source_image)
    file_names = {box for card in file_cards for box in card.filename_boxes}
    file_metadata = {box for card in file_cards for box in card.metadata_boxes}
    file_boxes = file_names | file_metadata
    # The operator explicitly excludes this WeChat compatibility notice.
    # Match the whole, high-confidence notice, never a substring, clipped
    # fragment, image lettering or an arbitrary low-confidence tail.
    unsupported_boxes = {
        box for box in all_message_boxes
        if direct_chat and layout.target_app == TargetApp.WECHAT
        and box not in media_boxes and box not in app_boxes and box not in file_boxes
        and _is_ignored_unsupported_notice(box, layout)
    }
    message_boxes = [
        box for box in normalized.boxes
        if box.region == OcrRegion.MESSAGE and box.text.strip() and box.confidence >= 0.35
        and box not in avatar_boxes and box not in file_boxes and box not in media_boxes and box not in voice_boxes
        and box not in history_boxes
        and box not in app_boxes
        and box not in unsupported_boxes
    ]
    if not message_boxes and not file_cards and not media_regions and not app_cards:
        return [], {"message_box_count": len(all_message_boxes), "parsed_box_count": 0,
                    "ignored_metadata_box_count": len(avatar_boxes | voice_boxes | history_boxes | unsupported_boxes),
                    "ignored_avatar_box_count": len(avatar_boxes),
                    "unaccounted_box_count": len(all_message_boxes) - len(avatar_boxes | voice_boxes | history_boxes | unsupported_boxes),
                    "ignored_unsupported_notice_count": len(unsupported_boxes),
                    "complete": False, "tail_complete": False, "parsed_coverage_complete": False,
                    "input_boundary_verified": bool(layout.input_boundary_verified),
                    "history_navigation_present": history_present,
                    "history_navigation_evidence": history_evidence,
                    "ignored_history_navigation_box_count": len(history_boxes)}
    message_boxes.sort(key=lambda box: (box.rect.top, box.rect.left))
    visual = VisualContext.load(source_image)
    bubbles = detect_bubbles(source_image, layout) if source_image is not None else []
    bubble_map = {box: _bubble_for_ocr_box(box, bubbles, source_image, layout) for box in message_boxes}
    # Nickname detection must not use the ring-based fallback: a nickname sits
    # directly above its bubble, so the ring would sample that bubble and the
    # label would be mistaken for bubble content.  Only detected bubbles count.
    component_bubble_map = {box: bubble_for_box(box.rect, bubbles) for box in message_boxes}
    kinds = _classify_box_kinds(message_boxes, layout, visual)
    recall_notices = _verified_recall_notices(message_boxes, layout, visual)
    partition = _partition_message_boxes(message_boxes, layout, bubble_map, component_bubble_map, kinds,
                                         allow_sender_labels=not direct_chat, system_notices=recall_notices)
    merged = _merge_message_fragments(
        partition.messages,
        layout,
        bubble_map,
        partition.blockers,
        visual,
    )
    parsed = [
        _message_from_box(
            box,
            layout,
            partition.times.get(box),
            partition.senders.get(box),
            _bubble_for_ocr_box(box, bubbles, source_image, layout),
            kinds.get(box, KIND_UNKNOWN),
            complete_bottom_rect=next((rect for rect in complete_transcript_regions
                if rect.left + 4 <= box.rect.left and box.rect.right <= rect.right - 4
                and rect.top + 4 <= box.rect.top and box.rect.bottom <= rect.bottom - 4), None),
        )
        for box in merged
    ]
    parsed.extend(_duration_from_box(box) for box in partition.durations)
    for card in app_cards:
        parsed.append(ParsedOcrMessage(
            speaker=card.speaker, text="[小程序卡片，内部内容未作为聊天文字读取]", rect=card.rect,
            confidence=.95, partial=False, reason="native_mini_program:frame+footer+avatar+gutter",
        ))
    for region in media_regions:
        parsed.append(ParsedOcrMessage(
            speaker=region.speaker, text="[引用图片，内容未识别]" if region.quoted else "[图片，内容未识别]", rect=region.rect,
            confidence=.95, partial=region.partial,
            reason=("chat_image:top_clipped+texture+two_outgoing_avatar_rails; partial_content_not_read"
                    if region.partial else "chat_image:complete_rect+texture+avatar+gutter; image_content_not_recognized"),
            content_type="image",
            media_refs=() if region.partial else ({"content_hash": region.content_hash, "content_type": "image"},),
        ))
    for card in file_cards:
        filename_rect = _union_box_rects(card.filename_boxes)
        parsed.append(ParsedOcrMessage(
            speaker=card.speaker, text="[文件] " + card.filename, rect=filename_rect,
            confidence=min(.95, *(box.confidence for box in card.filename_boxes)),
            partial=any(_is_edge_partial_box(box, layout) for box in card.filename_boxes),
            reason="native_file_card:filename+size+source+icon+pixels; metadata_not_chat_text",
        ))
    parsed.sort(key=lambda message: (message.rect.top, message.rect.left))
    metadata = partition.anchors | partition.labels | avatar_boxes | file_metadata | media_boxes | voice_boxes
    metadata |= {box for box in message_boxes if _looks_like_non_message_overlay(box, layout)}
    metadata |= history_boxes
    metadata |= app_boxes
    metadata |= unsupported_boxes
    accounted = set(partition.messages) | file_names | metadata | set(partition.durations)
    unaccounted = [box for box in all_message_boxes if box not in accounted]
    covered = all(
        any(_overlap_ratio(box.rect, message.rect) >= 0.90 for message in parsed)
        for box in set(partition.messages) | file_names
    )
    tail_complete = bool(parsed and not parsed[-1].partial and parsed[-1].confidence >= 0.65
                         and parsed[-1].speaker in {Speaker.ME, Speaker.OTHER}
                         and not result.warning and not unaccounted and covered and layout.input_boundary_verified
                         and not unresolved_voice_regions and not history_present)
    complete = bool(parsed and not result.warning and not unaccounted and covered
                    and layout.input_boundary_verified and not unresolved_voice_regions
                    and not history_present and all(not message.partial and message.confidence >= 0.65
                            and message.speaker in {Speaker.ME, Speaker.OTHER} for message in parsed))
    return parsed, {"message_box_count": len(all_message_boxes), "parsed_box_count": len(partition.messages) + len(file_names),
                    "ignored_metadata_box_count": len(metadata), "ignored_avatar_box_count": len(avatar_boxes),
                    "ignored_recall_notice_count": len(recall_notices),
                    "ignored_unsupported_notice_count": len(unsupported_boxes),
                    "transcribed_voice_metadata_box_count": len(voice_boxes),
                    "ignored_voice_avatar_box_count": len(voice_avatar_boxes),
                    "untranscribed_voice_count": len(unresolved_voice_regions),
                    "native_file_card_count": len(file_cards), "ignored_file_metadata_box_count": len(file_metadata),
                    "native_mini_program_count": len(app_cards),
                    "image_occurrence_count": len(media_regions), "image_ocr_box_count": len(media_boxes),
                    "quoted_image_count": sum(region.quoted for region in media_regions),
                    "unaccounted_box_count": len(unaccounted),
                    "complete": complete, "tail_complete": tail_complete, "parsed_coverage_complete": covered and not history_present,
                    "input_boundary_verified": bool(layout.input_boundary_verified),
                    "history_navigation_present": history_present,
                    "history_navigation_evidence": history_evidence,
                    "ignored_history_navigation_box_count": len(history_boxes)}


def _is_ignored_unsupported_notice(box: OcrTextBox, layout: LayoutRegions) -> bool:
    text = re.sub(r"\s+", "", box.text).replace(",", "，")
    band, rect = layout.message_rect, box.rect
    return bool(text.rstrip("。.") == "暂不支持该内容，请在手机上查看"
                and box.confidence >= .95 and layout.input_boundary_verified
                and band.left <= rect.left < rect.right <= band.right
                and band.top <= rect.top < rect.bottom <= band.bottom)


def _history_navigation_pixel_evidence(rgb, layout, boxes):
    """Negative viewport evidence only: OCR + a native-shaped floating pill.

    A keyword in a customer's bubble/picture is not navigation chrome. Require
    lower-right position, four visible neutral/shadow edges, bright interior,
    rounded corners and two green downward chevrons independently of OCR.
    Clipped/ambiguous pixels remain unknown. No chat text is stored here.
    """
    if (rgb is None or layout.target_app.value != "wechat" or rgb.mode != "RGB"
            or rgb.size != (layout.window_rect.width,layout.window_rect.height)):
        return {}
    band=layout.message_rect
    def neutral(p,level,tolerance=5):
        return max(p)-min(p)<=8 and abs(sum(p)/3-level)<=tolerance
    def line(points):
        values=[rgb.getpixel(point) for point in points]
        if not values:
            return None
        levels=sorted(sum(p)/3 for p in values)
        level=levels[len(levels)//2]
        return level if 210<=level<=247 and sum(neutral(p,level) for p in values)/len(values)>=.85 else None
    try:
        for box in boxes:
            compact=re.sub(r"\s+","",box.text)
            if (not any(marker in compact for marker in ("跳转到最新消息","跳至最新消息"))
                    or box.confidence<.65 or not 8<=box.rect.height<=26
                    or not 55<=box.rect.width<=190 or box.rect.left<band.left+band.width*.50
                    or not band.bottom-100<=box.rect.top<band.bottom-8):
                continue
            for top in range(max(band.top,box.rect.top-12),box.rect.top-2):
                top_level=line(((x,top) for x in range(max(0,box.rect.left-12),min(rgb.width,box.rect.right+4))))
                if top_level is None:
                    continue
                for bottom in range(box.rect.bottom+3,min(band.bottom,box.rect.bottom+12)):
                    if not 24<=bottom-top<=42 or band.bottom-bottom>64:
                        continue
                    bottom_level=line(((x,bottom) for x in range(max(0,box.rect.left-12),min(rgb.width,box.rect.right+4))))
                    if bottom_level is None:
                        continue
                    middle=(top+bottom)//2
                    lefts=[x for x in range(max(band.left,box.rect.left-30),box.rect.left-15)
                           if line(((x,y) for y in range(middle-1,middle+2))) is not None]
                    rights=[x for x in range(box.rect.right+5,min(band.right-1,box.rect.right+18))
                            if line(((x,y) for y in range(middle-1,middle+2))) is not None]
                    for left in lefts:
                        for right in rights:
                            if not 96<=right-left<=180 or not 2<=band.right-right<=24:
                                continue
                            body=[rgb.getpixel((x,y)) for y in range(top+3,bottom-2)
                                  for x in range(left+6,right-5)]
                            if sum(min(p)>=245 and max(p)-min(p)<=8 for p in body)/len(body)<.60:
                                continue
                            # Rounded caps: neither a flat white message card
                            # nor an unbounded image edge establishes a pill.
                            if (any(neutral(rgb.getpixel((x,y)),top_level,3)
                                    for x in (left+1,right-1) for y in (top,bottom))
                                    or not all(neutral(rgb.getpixel((x,(top+bottom)//2)),250,6)
                                               for x in (left+3,right-3))):
                                continue
                            icon=Rect(left+6,top+5,min(left+23,right-4),bottom-4)
                            chevrons=_double_down_chevrons(rgb,icon)
                            if chevrons:
                                return {"kind":"wechat_jump_latest_button","coordinate_space":"window_local",
                                    "rect":[left,top,right+1,bottom+1],"icon_rect":list(icon.as_tuple()),
                                    "ocr_marker_supported":True,"shadow_edges_verified":True,
                                    "green_double_down_chevrons":chevrons,"position":"lower_right_above_input"}
    except (OSError,ValueError,TypeError,AttributeError,IndexError):
        return {}
    return {}


def _double_down_chevrons(rgb,rect):
    rows=[]
    for y in range(rect.top,rect.bottom):
        xs=[x for x in range(rect.left,rect.right) if (lambda p:p[1]>=100 and p[1]-p[0]>=25 and p[1]-p[2]>=10)(rgb.getpixel((x,y)))]
        rows.append(xs)
    candidates=[]
    for start in range(len(rows)-2):
        first=rows[start]
        if not first or not 4<=max(first)-min(first)<=12:
            continue
        center=(min(first)+max(first))/2
        for depth in (2,3,4):
            if start+depth>=len(rows):
                continue
            last=rows[start+depth]
            if (last and max(last)-min(last)<=3 and abs((min(last)+max(last))/2-center)<=2
                    and all(rows[i] for i in range(start,start+depth+1))):
                candidates.append((start,start+depth,center))
    return any(2<=b[0]-a[1]<=5 and abs(a[2]-b[2])<=2 for a in candidates for b in candidates if b[0]>a[1])


def _classify_box_kinds(
    boxes: list[OcrTextBox],
    layout: LayoutRegions,
    visual: VisualContext | None,
) -> dict[OcrTextBox, str]:
    """Tone of every message-region box: dark ink, mid-grey metadata, or dark overlay.

    This is the pixel-level signal that separates WeChat chrome (nicknames,
    timestamps, media duration badges) from actual message text.  Without an
    image to look at the helper returns ``unknown`` for everything and the
    parser falls back to pure geometry.
    """
    if visual is None:
        return {box: KIND_UNKNOWN for box in boxes}
    reference = visual.page_background(_background_reference_rect(layout, visual))
    if reference <= 0:
        return {box: KIND_UNKNOWN for box in boxes}
    return {box: visual.box_kind(box.rect, reference) for box in boxes}


def _background_reference_rect(layout: LayoutRegions, visual: VisualContext) -> Rect:
    message = layout.message_rect
    top = message.top + max(0, round(message.height * 0.05))
    bottom = message.bottom - max(0, round(message.height * 0.05))
    if bottom - top < 40:
        top, bottom = message.top, message.bottom
    return Rect(message.left, top, message.right, bottom)


def _partition_message_boxes(
    boxes: list[OcrTextBox],
    layout: LayoutRegions,
    bubble_map: dict[OcrTextBox, BubbleRegion | None],
    component_bubble_map: dict[OcrTextBox, BubbleRegion | None],
    kinds: dict[OcrTextBox, str],
    *, allow_sender_labels: bool = True, system_notices: set[OcrTextBox] | None = None,
) -> "MessagePartition":
    timeline = _assign_time_anchors(boxes, layout, kinds)
    anchors: set[OcrTextBox] = timeline["anchors"] | (system_notices or set())  # type: ignore[operator]
    times: dict[OcrTextBox, str] = timeline["times"]  # type: ignore[assignment]
    # Confirmed direct-chat pages have no per-message group nickname labels.
    # A short customer sentence above the next bubble must not disappear as a
    # guessed sender name. Group/unknown callers retain the original rule.
    labels_result = (_assign_sender_labels(boxes, layout, anchors, component_bubble_map, kinds)
                     if allow_sender_labels else {"labels": set(), "senders": {}})
    labels: set[OcrTextBox] = labels_result["labels"]  # type: ignore[assignment]
    senders: dict[OcrTextBox, str] = labels_result["senders"]  # type: ignore[assignment]

    messages: list[OcrTextBox] = []
    durations: list[OcrTextBox] = []
    for box in boxes:
        if box in anchors or box in labels:
            continue
        if _looks_like_non_message_overlay(box, layout):
            continue
        kind = kinds.get(box, KIND_UNKNOWN)
        if kind == KIND_DARK_OVERLAY and _looks_like_media_duration(box.text):
            durations.append(box)
            continue
        in_bubble = (
            bubble_map.get(box) is not None or component_bubble_map.get(box) is not None
        )
        if (
            kind == KIND_SECONDARY
            and not in_bubble
            and not _probable_multiline_continuation(box, messages, layout, bubble_map)
        ):
            # Mid-grey ink directly on the conversation surface is WeChat chrome
            # (a nickname whose message is an image, a hint, a tag).  It is never
            # message text, so it must not enter the transcript.  Grey ink inside
            # a bubble stays: that is a quoted sender line of a real message.
            continue
        messages.append(box)
    return MessagePartition(
        messages=messages,
        anchors=anchors,
        times=times,
        labels=labels,
        senders=senders,
        durations=durations,
    )


def _looks_like_media_duration(text: str) -> bool:
    return bool(_MEDIA_DURATION_PATTERN.match(_compact_text(text)))


def _duration_from_box(box: OcrTextBox) -> ParsedOcrMessage:
    return ParsedOcrMessage(
        speaker=Speaker.SYSTEM,
        text=box.text.strip(),
        rect=box.rect,
        confidence=min(0.95, max(0.60, box.confidence)),
        partial=False,
        reason="浅色文字位于深色底上，判定为语音/视频时长徽标；仅作时长信息，不并入聊天消息",
    )


def _bubble_for_ocr_box(
    box: OcrTextBox,
    bubbles: list[BubbleRegion],
    source_image: str | Path | Image.Image | None,
    layout: LayoutRegions,
) -> BubbleRegion | None:
    bubble = bubble_for_box(box.rect, bubbles)
    if bubble is not None:
        return bubble
    if not source_image:
        return None
    return bubble_for_box_from_image(source_image, box.rect, layout)


def _merge_message_fragments(
    boxes: list[OcrTextBox],
    layout: LayoutRegions,
    bubble_map: dict[OcrTextBox, BubbleRegion | None] | None = None,
    blockers: set[OcrTextBox] | None = None,
    visual: VisualContext | None = None,
) -> list[OcrTextBox]:
    if len(boxes) < 2:
        return boxes
    ordered = sorted(boxes, key=lambda box: (box.rect.top, box.rect.left))
    merged: list[OcrTextBox] = []
    current = ordered[0]
    current_bubble = bubble_map.get(current) if bubble_map else None
    for box in ordered[1:]:
        next_bubble = bubble_map.get(box) if bubble_map else None
        if _should_merge_fragments(current, box, layout, current_bubble, next_bubble, blockers, visual):
            current = _merge_boxes(current, box)
            # When one visual bubble is split into one component per OCR line,
            # keep the most recent component as the adjacency anchor.  Keeping
            # only the first component makes line 3 compare with line 1 and
            # truncates 4--6 line messages after their second line.
            current_bubble = next_bubble or current_bubble
        else:
            merged.append(current)
            current = box
            current_bubble = next_bubble
    merged.append(current)
    return merged


def _should_merge_fragments(
    left: OcrTextBox,
    right: OcrTextBox,
    layout: LayoutRegions,
    left_bubble: BubbleRegion | None = None,
    right_bubble: BubbleRegion | None = None,
    blockers: set[OcrTextBox] | None = None,
    visual: VisualContext | None = None,
) -> bool:
    left_speaker, _left_reason = _speaker_from_geometry(left, layout, left_bubble)
    right_speaker, _right_reason = _speaker_from_geometry(right, layout, right_bubble)
    if left_speaker != right_speaker:
        return False
    if _merge_blocked(blockers, left, right):
        return False
    if _looks_like_non_message_overlay(left, layout) or _looks_like_non_message_overlay(right, layout):
        return False

    same_bubble = left_bubble is not None and right_bubble is not None and left_bubble == right_bubble
    if (
        not same_bubble
        and left_bubble is not None
        and right_bubble is not None
        and _bubble_fragments_share_message(left, right, left_bubble, right_bubble, layout)
    ):
        # Ring sampling and a split connected-component mask can produce one
        # small BubbleRegion per OCR line even though the lines live in one
        # tall WeChat bubble.  Treat overlapping, same-lane fragments as one
        # bubble while retaining the hard boundary between separated bubbles.
        same_bubble = True
    if _is_edge_partial_box(left, layout) or _is_edge_partial_box(right, layout):
        if not (
            same_bubble
            and (
                (_is_top_edge_partial_box(left, layout)
                    and not _is_top_edge_partial_box(right, layout)
                    and not _is_bottom_edge_partial_box(right, layout))
                or (_is_bottom_edge_partial_box(right, layout)
                    and not _is_top_edge_partial_box(left, layout)
                    and not _is_top_edge_partial_box(right, layout))
            )
        ):
            return False
        # Keep a proved top bubble together: its continuation must inherit
        # partial via the merged rectangle, never masquerade as a full row.
    vertical_gap = right.rect.top - left.rect.bottom
    if same_bubble and vertical_gap <= max(22, round(layout.message_rect.height * 0.05)):
        return True
    if (
        visual is not None
        and left_bubble is not None
        and right_bubble is not None
        and 0 < vertical_gap <= max(28, round(layout.message_rect.height * 0.05))
        and visual.green_fill_fraction(left.rect, right.rect) >= MIN_MEDIA_DURATION_SHARE
    ):
        # A quoted line inside one outgoing bubble is still the same message.
        return True
    if left_bubble is not None and right_bubble is not None and left_bubble != right_bubble:
        return False

    same_row = _vertical_overlap_ratio(left.rect, right.rect) >= 0.45
    close_rows = abs(right.rect.top - left.rect.bottom) <= max(10, round(layout.message_rect.height * 0.025))
    horizontal_gap = right.rect.left - left.rect.right
    adjacent_horizontal = -max(8, round(layout.message_rect.width * 0.015)) <= horizontal_gap <= max(56, round(layout.message_rect.width * 0.09))
    aligned_left = abs(left.rect.left - right.rect.left) <= max(30, round(layout.message_rect.width * 0.05))
    close_vertical = 0 <= right.rect.top - left.rect.bottom <= max(18, round(layout.message_rect.height * 0.035))
    same_bubble_width = _merged_width(left.rect, right.rect) <= layout.message_rect.width * 0.72
    if same_row and adjacent_horizontal and same_bubble_width:
        return True
    return close_rows and close_vertical and aligned_left and same_bubble_width


def _probable_multiline_continuation(
    box: OcrTextBox,
    accepted: list[OcrTextBox],
    layout: LayoutRegions,
    bubble_map: dict[OcrTextBox, BubbleRegion | None],
) -> bool:
    """Keep a grey OCR line when geometry proves it continues a prior line.

    WeChat's antialiasing can make a lower line look like secondary chrome. If
    bubble detection misses only that line, dropping it here permanently
    truncates the message before the fragment-merging stage can recover it.
    """
    if not accepted:
        return False
    previous = accepted[-1]
    previous_speaker, _ = _speaker_from_geometry(previous, layout, bubble_map.get(previous))
    current_speaker, _ = _speaker_from_geometry(box, layout, bubble_map.get(box))
    if previous_speaker != current_speaker or current_speaker in {Speaker.SYSTEM, Speaker.UNKNOWN}:
        return False
    vertical_gap = box.rect.top - previous.rect.bottom
    aligned_left = abs(box.rect.left - previous.rect.left) <= max(30, round(layout.message_rect.width * 0.05))
    combined_width = _merged_width(previous.rect, box.rect) <= layout.message_rect.width * 0.72
    return 0 <= vertical_gap <= max(18, round(layout.message_rect.height * 0.035)) and aligned_left and combined_width


def _bubble_fragments_share_message(
    left: OcrTextBox,
    right: OcrTextBox,
    left_bubble: BubbleRegion,
    right_bubble: BubbleRegion,
    layout: LayoutRegions,
) -> bool:
    if left_bubble.speaker != right_bubble.speaker or left_bubble.profile != right_bubble.profile:
        return False
    line_gap = right.rect.top - left.rect.bottom
    # OCR polygons from adjacent wrapped lines can overlap by one or two
    # antialiased pixels. A small overlap is still vertical continuation, while
    # a substantial overlap remains a separate same-row element.
    if line_gap < -4 or line_gap > max(22, round(layout.message_rect.height * 0.05)):
        return False
    overlap = max(
        0,
        min(left_bubble.rect.right, right_bubble.rect.right)
        - max(left_bubble.rect.left, right_bubble.rect.left),
    )
    overlap_ratio = overlap / max(1, min(left_bubble.rect.width, right_bubble.rect.width))
    bubble_gap = max(
        0,
        right_bubble.rect.top - left_bubble.rect.bottom,
        left_bubble.rect.top - right_bubble.rect.bottom,
    )
    return overlap_ratio >= 0.55 and bubble_gap <= max(4, round(layout.message_rect.height * 0.01))


def _merge_blocked(
    blockers: set[OcrTextBox] | None,
    left: OcrTextBox,
    right: OcrTextBox,
) -> bool:
    """Never merge across a timestamp or nickname label sitting between two boxes."""
    if not blockers:
        return False
    top = left.rect.bottom
    bottom = right.rect.top
    if bottom <= top:
        return False
    for blocker in blockers:
        if top <= blocker.rect.top and blocker.rect.bottom <= bottom + 2:
            return True
    return False


def _merge_boxes(left: OcrTextBox, right: OcrTextBox) -> OcrTextBox:
    same_row = _vertical_overlap_ratio(left.rect, right.rect) >= 0.45
    separator = "" if same_row else "\n"
    rect = Rect(
        min(left.rect.left, right.rect.left),
        min(left.rect.top, right.rect.top),
        max(left.rect.right, right.rect.right),
        max(left.rect.bottom, right.rect.bottom),
    )
    confidence = min(left.confidence, right.confidence)
    return OcrTextBox(
        text=f"{left.text.strip()}{separator}{right.text.strip()}",
        rect=rect,
        confidence=confidence,
        region=left.region,
        source=left.source,
    )


def _vertical_overlap_ratio(left: Rect, right: Rect) -> float:
    overlap = max(0, min(left.bottom, right.bottom) - max(left.top, right.top))
    return overlap / max(1, min(left.height, right.height))


def _merged_width(left: Rect, right: Rect) -> int:
    return max(left.right, right.right) - min(left.left, right.left)


def _message_from_box(
    box: OcrTextBox,
    layout: LayoutRegions,
    time_text: str | None = None,
    sender_name: str | None = None,
    bubble: BubbleRegion | None = None,
    kind: str = KIND_UNKNOWN,
    *, complete_bottom_rect: Rect | None = None,
) -> ParsedOcrMessage:
    speaker, geometry_reason = _speaker_from_geometry(box, layout, bubble)
    partial = _is_edge_partial_box(box, layout)
    # A complete, independently proved voice transcript can sit within the
    # heuristic bottom margin. Its actual card boundary and glyph padding
    # are stronger evidence than distance alone. Never clear a top clip or
    # an unverified input boundary, or infer completeness from a colour ring.
    bottom_proved = bool(
        partial and not _is_top_edge_partial_box(box, layout)
        and layout.input_boundary_verified is True and complete_bottom_rect is not None
        and layout.message_rect.left + 2 < complete_bottom_rect.left
        and complete_bottom_rect.right < layout.message_rect.right - 2
        and layout.message_rect.top + 2 < complete_bottom_rect.top
        and complete_bottom_rect.bottom < layout.message_rect.bottom - 2
        and complete_bottom_rect.left + 4 <= box.rect.left
        and box.rect.right <= complete_bottom_rect.right - 4
        and complete_bottom_rect.top + 4 <= box.rect.top
        and box.rect.bottom <= complete_bottom_rect.bottom - 4)
    if bottom_proved:
        partial = False
    confidence = min(0.95, max(0.0, box.confidence - (0.08 if partial else 0.0)))
    reason = f"根据消息区坐标推断说话人：{geometry_reason}"
    if kind != KIND_UNKNOWN:
        reason += f"；文字色调={kind}"
    if sender_name:
        reason += f"；上方坐标标签识别为发送者：{sender_name}"
    if bubble is not None:
        reason += f"；气泡={bubble.profile}/{bubble.speaker.value}"
    if partial:
        reason += "；文本接近消息区边缘，标记为 partial"
    elif bottom_proved:
        reason += "；语音转写气泡边界及文字留白完整，底部未截断"
    return ParsedOcrMessage(
        speaker=speaker,
        text=_normalize_message_text(box.text),
        rect=box.rect,
        confidence=confidence,
        partial=partial,
        reason=reason,
        time_text=time_text,
        sender_name=sender_name,
    )


def _normalize_message_text(value: str) -> str:
    """Canonicalize a narrowly proven RapidOCR punctuation substitution.

    RapidOCR renders the Chinese question-mark glyph as ASCII ``?`` in the
    real WeChat title/body font. Only replace a question mark immediately
    following CJK text; English questions and all other OCR characters remain
    untouched.
    """
    return re.sub(r"(?<=[\u3400-\u9fff])\?", "？", value.strip())


def _is_edge_partial_box(box: OcrTextBox, layout: LayoutRegions) -> bool:
    edge_margin = max(8, round(layout.message_rect.height * 0.025))
    return box.rect.top <= layout.message_rect.top + edge_margin or box.rect.bottom >= layout.message_rect.bottom - edge_margin


def _is_top_edge_partial_box(box: OcrTextBox, layout: LayoutRegions) -> bool:
    edge_margin = max(8, round(layout.message_rect.height * 0.025))
    return box.rect.top <= layout.message_rect.top + edge_margin


def _is_bottom_edge_partial_box(box: OcrTextBox, layout: LayoutRegions) -> bool:
    edge_margin = max(8, round(layout.message_rect.height * 0.025))
    return box.rect.bottom >= layout.message_rect.bottom - edge_margin


def _overlap_ratio(left: Rect, right: Rect) -> float:
    overlap_left = max(left.left, right.left)
    overlap_top = max(left.top, right.top)
    overlap_right = min(left.right, right.right)
    overlap_bottom = min(left.bottom, right.bottom)
    overlap_width = max(0, overlap_right - overlap_left)
    overlap_height = max(0, overlap_bottom - overlap_top)
    overlap_area = overlap_width * overlap_height
    area = max(1, left.width * left.height)
    return overlap_area / area


def _looks_like_group_title(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if re.search(r"[（(]\s*\d{2,}\s*[)）]", text):
        return True
    markers = ("群", "班级", "项目组", "交流群", "讨论组")
    return any(marker in text for marker in markers)


def _assign_sender_labels(
    boxes: list[OcrTextBox],
    layout: LayoutRegions,
    ignored: set[OcrTextBox],
    bubble_map: dict[OcrTextBox, BubbleRegion | None] | None = None,
    kinds: dict[OcrTextBox, str] | None = None,
) -> dict[str, object]:
    """Split nickname labels from message text.

    Nicknames (and group "@" hints) render in mid-grey ink, so the pixel tone is
    the primary signal: a grey left-lane run of text is a label even when its
    message is a pure image with no OCR text below it.  When the tone is
    unavailable the legacy geometry heuristic still applies, and dark-ink boxes
    additionally keep it so a dark-rendered nickname is not lost.
    """
    kinds = kinds or {}
    labels: set[OcrTextBox] = set()
    senders: dict[OcrTextBox, str] = {}
    usable = [box for box in boxes if box not in ignored and not _looks_like_non_message_overlay(box, layout)]
    for index, box in enumerate(usable):
        kind = kinds.get(box, KIND_UNKNOWN)
        if kind == KIND_DARK_OVERLAY:
            continue
        in_bubble = bool(bubble_map and bubble_map.get(box) is not None)
        continues_previous_line = _continues_previous_message_line(
            box,
            usable[:index],
            layout,
            bubble_map,
        )
        if kind == KIND_SECONDARY:
            if (
                in_bubble
                or continues_previous_line
                or not _looks_like_sender_label_text(box.text)
                or not _is_left_label_lane(box, layout)
            ):
                continue
            labels.add(box)
            target = _next_lower_left_message_box(box, usable[index + 1 :], layout, bubble_map, kinds)
            if target is not None:
                senders[target] = box.text.strip()
            continue
        if continues_previous_line:
            continue
        next_box = _next_lower_left_message_box(box, usable[index + 1 :], layout, bubble_map, kinds)
        if _looks_like_sender_label(box, next_box, layout, bubble_map):
            labels.add(box)
            senders[next_box] = box.text.strip()
    return {"labels": labels, "senders": senders}


def _continues_previous_message_line(
    box: OcrTextBox,
    previous_boxes: list[OcrTextBox],
    layout: LayoutRegions,
    bubble_map: dict[OcrTextBox, BubbleRegion | None] | None,
) -> bool:
    """Protect a wrapped short line from the group-nickname heuristic."""
    for previous in reversed(previous_boxes):
        vertical_gap = box.rect.top - previous.rect.bottom
        if vertical_gap < 0:
            continue
        if vertical_gap > max(18, round(layout.message_rect.height * 0.035)):
            return False
        previous_bubble = bubble_map.get(previous) if bubble_map else None
        current_bubble = bubble_map.get(box) if bubble_map else None
        previous_speaker, _ = _speaker_from_geometry(previous, layout, previous_bubble)
        current_speaker, _ = _speaker_from_geometry(box, layout, current_bubble)
        aligned_left = abs(box.rect.left - previous.rect.left) <= max(
            30,
            round(layout.message_rect.width * 0.05),
        )
        reasonable_width = _merged_width(previous.rect, box.rect) <= layout.message_rect.width * 0.72
        return (
            previous_speaker == current_speaker
            and current_speaker not in {Speaker.SYSTEM, Speaker.UNKNOWN}
            and aligned_left
            and reasonable_width
        )
    return False


def _next_lower_left_message_box(
    box: OcrTextBox,
    candidates: list[OcrTextBox],
    layout: LayoutRegions,
    bubble_map: dict[OcrTextBox, BubbleRegion | None] | None = None,
    kinds: dict[OcrTextBox, str] | None = None,
) -> OcrTextBox | None:
    kinds = kinds or {}
    max_gap = max(34, round(layout.message_rect.height * 0.08))
    for candidate in candidates:
        if candidate.rect.top < box.rect.bottom:
            continue
        if candidate.rect.top - box.rect.bottom > max_gap:
            return None
        if kinds.get(candidate, KIND_UNKNOWN) == KIND_DARK_OVERLAY:
            continue
        bubble = bubble_map.get(candidate) if bubble_map else None
        if bubble is None and _looks_like_sender_label_text(candidate.text):
            continue
        if (bubble is not None and bubble.speaker == Speaker.OTHER) or _is_left_bubble_lane(candidate, layout):
            return candidate
    return None


def _looks_like_sender_label(
    box: OcrTextBox,
    next_box: OcrTextBox | None,
    layout: LayoutRegions,
    bubble_map: dict[OcrTextBox, BubbleRegion | None] | None = None,
) -> bool:
    text = box.text.strip()
    if next_box is None or not text:
        return False
    if bubble_map and bubble_map.get(box) is not None:
        return False
    if not _looks_like_sender_label_text(text):
        return False
    if not _is_left_label_lane(box, layout):
        return False
    next_bubble = bubble_map.get(next_box) if bubble_map else None
    if next_bubble is not None and next_bubble.speaker != Speaker.OTHER:
        return False
    vertical_gap = next_box.rect.top - box.rect.bottom
    horizontal_gap = abs(next_box.rect.left - box.rect.left)
    aligned_with_next = horizontal_gap <= max(64, round(layout.message_rect.width * 0.12))
    compact_label = box.rect.height <= max(24, round(layout.message_rect.height * 0.045))
    narrower_than_message = box.rect.width <= max(190, round(next_box.rect.width * 0.88))
    return 0 <= vertical_gap <= max(34, round(layout.message_rect.height * 0.08)) and aligned_with_next and compact_label and narrower_than_message


def _looks_like_sender_label_text(text: str) -> bool:
    value = text.strip()
    if len(value) > 16:
        return False
    if len(value) < 2:
        # A single glyph is a reply ("有", "好", "嗯", "行"), never a nickname.
        # Treating it as label-like made it ineligible as the message its
        # nickname belongs to, which silently detached the sender name.
        return False
    if any(ch in value for ch in "，。！？,.?!:：；;、"):
        return False
    if _looks_like_time_text(value) or _looks_like_new_message_text(_compact_text(value)):
        return False
    common_short_messages = {
        "好",
        "好的",
        "收到",
        "可以",
        "行",
        "嗯",
        "嗯嗯",
        "OK",
        "ok",
        "谢谢",
        "辛苦了",
    }
    return value not in common_short_messages


def _verified_recall_notices(
    boxes: list[OcrTextBox], layout: LayoutRegions, visual: VisualContext | None,
) -> set[OcrTextBox]:
    """Recognize native recall chrome, not sentences merely quoting that text.

    The nickname can make a notice wider than an ordinary timestamp. Require
    the complete native phrase, centered neutral ink on the SAME flat canvas
    as both sides of the chat, with a full blank rim. Text in a real bubble,
    a cropped row or an OCR-only observation is not granted this exemption.
    Work in window-local coordinates, as the caller already does for pixels.
    """
    if visual is None or not layout.input_boundary_verified:
        return set()
    band = layout.message_rect
    notices: set[OcrTextBox] = set()
    for box in boxes:
        text = _compact_text(box.text)
        rect = box.rect
        if (box.confidence < .65
                or not re.fullmatch(r"(?:你|对方|.+?)撤回了(?:一条)?消息", text)
                or not .45 <= _horizontal_ratio(box, layout) <= .55
                or not 8 <= rect.height <= max(32, round(band.height * .06))
                or rect.left < band.left + 8 or rect.right > band.right - 8
                or rect.top < band.top + 3 or rect.bottom > band.bottom - 3):
            continue
        tone = visual.tone(rect)
        # message_rect ends at the native window edge, not necessarily at
        # the client canvas: GetWindowRect captures frame/shadow/scrollbar.
        # The left reference is an internal pane boundary. On the right use
        # the blank corridor between this centered notice and the outer edge,
        # not pixels belonging to the native frame. Both references, ink and
        # the full rim must STILL agree; bubble text gets no exemption.
        corridor_x = (rect.right + band.right) // 2
        sides = [visual.tone(Rect(band.left+3,rect.top,band.left+6,rect.bottom)),
                 visual.tone(Rect(corridor_x-1,rect.top,corridor_x+2,rect.bottom))]
        if (tone is None or not tone.is_neutral or tone.is_flat
                or any(s is None or not s.is_neutral or not s.is_flat for s in sides)):
            continue
        background = (sides[0].surface + sides[1].surface) / 2
        # Background equality rejects the gray/green/white surface of a real
        # message even if its text exactly repeats a WeChat recall sentence.
        if (abs(sides[0].surface-sides[1].surface) > 6
                or abs(tone.surface-background) > 6):
            continue
        contrast = background-tone.low if background >= 160 else tone.high-background
        if not 20 <= contrast <= 130:
            continue
        rim = (
            Rect(rect.left-3,rect.top-3,rect.right+3,rect.top),
            Rect(rect.left-3,rect.bottom,rect.right+3,rect.bottom+3),
            Rect(rect.left-3,rect.top,rect.left,rect.bottom),
            Rect(rect.right,rect.top,rect.right+3,rect.bottom),
        )
        samples = [visual.tone(r) for r in rim]
        if any(s is None or not s.is_neutral or not s.is_flat
               or abs(s.surface-background) > 6 for s in samples):
            continue
        notices.add(box)
    return notices


def _looks_like_non_message_overlay(box: OcrTextBox, layout: LayoutRegions) -> bool:
    text = _compact_text(box.text)
    if not text:
        return True
    # Recall sentences need the pixel-backed classification above. Geometry
    # alone previously dropped customer quotations and missed long nicknames.
    if _is_center_system_hint(box, layout) and text in {"以下为新消息", "对方正在输入"}:
        return True
    if _looks_like_new_message_text(text) and _is_floating_new_message_hint(box, layout):
        return True
    if _looks_like_message_count_text(text) and _is_message_count_hint(box, layout):
        return True
    if _is_center_system_hint(box, layout) and _looks_like_time_text(text):
        return True
    return False


def _looks_like_system_notice_text(text: str) -> bool:
    value = _compact_text(text)
    return bool(
        re.match(r"^(?:你|对方|.+?)撤回了(?:一条)?消息$", value)
        or value in {"以下为新消息", "对方正在输入"}
    )


def _looks_like_new_message_text(text: str) -> bool:
    return bool(
        re.match(r"^[↑⬆上个]?\d{1,4}条新消息$", text)
        or re.match(r"^[^\d]{0,2}\d{1,4}条新消息$", text)
    )


def _looks_like_message_count_text(text: str) -> bool:
    value = _compact_text(text)
    return bool(re.match(r"^(共|还有)\d{1,4}条(?:新消息)?$", value))


def _is_message_count_hint(box: OcrTextBox, layout: LayoutRegions) -> bool:
    center_ratio = _horizontal_ratio(box, layout)
    width_ratio = box.rect.width / max(1, layout.message_rect.width)
    height_ratio = box.rect.height / max(1, layout.message_rect.height)
    top_ratio = (box.rect.top - layout.message_rect.top) / max(1, layout.message_rect.height)
    return 0.25 <= center_ratio <= 0.75 and width_ratio <= 0.28 and height_ratio <= 0.11 and top_ratio <= 0.20


def _speaker_from_geometry(box: OcrTextBox, layout: LayoutRegions, bubble: BubbleRegion | None = None) -> tuple[Speaker, str]:
    if bubble is not None:
        reason = (
            f"bubble_rect={bubble.rect.as_tuple()} bubble_conf={bubble.confidence:.2f} "
            f"bubble_partial={bubble.partial} profile={bubble.profile}"
        )
        return bubble.speaker, reason
    center_ratio = _horizontal_ratio(box, layout)
    left_ratio = _edge_ratio(box.rect.left, layout)
    right_ratio = _edge_ratio(box.rect.right, layout)
    left_aligned = _is_left_bubble_lane(box, layout)
    right_aligned = _is_right_bubble_lane(box, layout)
    speaker = Speaker.ME if center_ratio >= 0.54 or (right_aligned and not left_aligned) else Speaker.OTHER
    reason = (
        f"rect={box.rect.as_tuple()} center_x={center_ratio:.2f} "
        f"left={left_ratio:.2f} right={right_ratio:.2f} "
        f"left_lane={left_aligned} right_lane={right_aligned}"
    )
    return speaker, reason


def _is_left_bubble_lane(box: OcrTextBox, layout: LayoutRegions) -> bool:
    left_ratio = _edge_ratio(box.rect.left, layout)
    center_ratio = _horizontal_ratio(box, layout)
    return left_ratio <= 0.22 and center_ratio <= 0.62


def _is_left_label_lane(box: OcrTextBox, layout: LayoutRegions) -> bool:
    left_ratio = _edge_ratio(box.rect.left, layout)
    center_ratio = _horizontal_ratio(box, layout)
    return left_ratio <= 0.26 and center_ratio <= 0.50


def _is_right_bubble_lane(box: OcrTextBox, layout: LayoutRegions) -> bool:
    right_ratio = _edge_ratio(box.rect.right, layout)
    center_ratio = _horizontal_ratio(box, layout)
    return right_ratio >= 0.90 and center_ratio >= 0.48


def _is_center_system_hint(box: OcrTextBox, layout: LayoutRegions) -> bool:
    center_ratio = _horizontal_ratio(box, layout)
    narrow = box.rect.width <= layout.message_rect.width * 0.36
    return narrow and 0.38 <= center_ratio <= 0.62


def _is_floating_new_message_hint(box: OcrTextBox, layout: LayoutRegions) -> bool:
    center_ratio = _horizontal_ratio(box, layout)
    width_ratio = box.rect.width / max(1, layout.message_rect.width)
    height_ratio = box.rect.height / max(1, layout.message_rect.height)
    if _is_left_bubble_lane(box, layout) or _is_right_bubble_lane(box, layout):
        return False
    return 0.55 <= center_ratio <= 0.90 and width_ratio <= 0.34 and height_ratio <= 0.11


def _horizontal_ratio(box: OcrTextBox, layout: LayoutRegions) -> float:
    center_x = (box.rect.left + box.rect.right) / 2
    return _edge_ratio(center_x, layout)


def _edge_ratio(value: float, layout: LayoutRegions) -> float:
    return (value - layout.message_rect.left) / max(1, layout.message_rect.width)


def _assign_time_anchors(
    boxes: list[OcrTextBox],
    layout: LayoutRegions,
    kinds: dict[OcrTextBox, str] | None = None,
) -> dict[str, object]:
    kinds = kinds or {}
    anchors: set[OcrTextBox] = set()
    times: dict[OcrTextBox, str] = {}
    current: str | None = None
    for box in boxes:
        if _looks_like_time_anchor(box, layout, kinds.get(box, KIND_UNKNOWN)):
            anchors.add(box)
            current = box.text.strip()
            continue
        if current:
            times[box] = current
    return {"anchors": anchors, "times": times}


def _looks_like_time_anchor(box: OcrTextBox, layout: LayoutRegions, kind: str = KIND_UNKNOWN) -> bool:
    text = _compact_text(box.text)
    if not text or len(text) > 24:
        return False
    if kind == KIND_DARK_OVERLAY:
        # "0:08" printed on a dark badge is a voice/video duration, not a
        # conversation timestamp; it must not re-time the following messages.
        return False
    if not _looks_like_time_text(text):
        return False
    if kind == KIND_SECONDARY:
        # Grey ink is exactly how WeChat renders the centred timestamp pill.
        return True
    center_x = (box.rect.left + box.rect.right) / 2
    middle = layout.message_rect.left + layout.message_rect.width / 2
    return abs(center_x - middle) <= layout.message_rect.width * 0.18


def _looks_like_time_text(text: str) -> bool:
    value = _compact_text(text)
    clock = r"(?:上午|下午|晚上|中午|凌晨)?\d{1,2}[:：]\d{2}"
    weekday = r"(?:(?:周|星期)[一二三四五六日天])?"
    patterns = [
        rf"^{clock}$",
        rf"^(昨天|今天|周[一二三四五六日天]|星期[一二三四五六日天]){clock}$",
        rf"^(?:\d{{2,4}}年)?\d{{1,2}}月\d{{1,2}}日{weekday}{clock}$",
        rf"^\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}{weekday}{clock}$",
    ]
    return any(re.match(pattern, value) for pattern in patterns)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())
