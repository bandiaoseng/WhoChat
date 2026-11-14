"""Native mini-program structure, not the document text printed inside it.

An OCR keyword alone grants nothing: require a complete grey rectangle, four
continuous borders, a separated footer, an aligned independent avatar and a
blank gutter. Partial/ambiguous cards stay unclassified and cannot be ignored.
"""
from dataclasses import dataclass
from collections import Counter
from whochat.core.models import Speaker
from whochat.core.runtime import Rect
from whochat.vision.bubbles import _connected_components


@dataclass(frozen=True)
class MiniProgramCard:
    rect: Rect
    speaker: Speaker
    boxes: tuple


def mini_program_cards(rgb, layout, boxes):
    if (rgb is None or rgb.mode != 'RGB' or layout.target_app.value != 'wechat'
            or not layout.input_boundary_verified):
        return ()
    band = layout.message_rect
    footers = [b for b in boxes if ''.join(b.text.split()) == '小程序' and b.confidence >= .80]
    if not footers:
        return ()
    crop = rgb.crop(band.as_tuple())
    from whochat.ocr.media_regions import _chat_background, _rgb_difference_mask, _components, _matching_avatar_sides
    background = _chat_background(crop)
    if background is None:
        return ()
    avatar_mask = _rgb_difference_mask(crop, background, 23)
    avatar_components = _components(bytearray(avatar_mask), crop.width, crop.height)
    mask = bytearray(max(p)-min(p) <= 6 and 225 <= sum(p)/3 <= 243 for p in crop.getdata())
    components = _connected_components(mask, crop.width, crop.height)
    found = []
    for x1,y1,x2,y2,area in components:
        # The native speech tail protrudes a few pixels beside the top of the
        # card. Prove a repeated straight body edge; never simply shave pixels
        # off an arbitrary candidate to make its border pass.
        spans = []
        for y in range(y1+8,y2-8):
            xs = [x for x in range(x1,x2) if mask[y*crop.width+x]]
            if xs:
                spans.append((y,min(xs),max(xs)+1))
        if not spans:
            continue
        (body_left, body_right), frequency = Counter((l,r) for _,l,r in spans).most_common(1)[0]
        if (frequency < len(spans)*.70 or not 0 <= body_left-x1 <= 10
                or not 0 <= x2-body_right <= 10
                or any((l<body_left or r>body_right) and y>y1+45 for y,l,r in spans)):
            continue
        x1,x2 = body_left,body_right
        w,h = x2-x1,y2-y1
        if (not 140 <= w <= band.width*.80 or not 80 <= h <= 500
                or x1 < 4 or y1 < 4 or x2 > band.width-4 or y2 > band.height-4
                or area/(w*h) < .25):
            continue
        rect = Rect(band.left+x1, band.top+y1, band.left+x2, band.top+y2)
        footer = [b for b in footers if rect.left+8 <= b.rect.left <= rect.left+55
                  and b.rect.right < rect.right-8 and 4 <= rect.bottom-b.rect.bottom <= 20
                  and b.rect.top >= rect.bottom-35]
        if len(footer) != 1:
            continue
        # Flat continuous perimeter plus footer surface, not many nearby words.
        edges = [(x,y) for x in range(x1+8,x2-8) for y in (y1+2,y2-3)]
        edges += [(x,y) for y in range(y1+8,y2-8) for x in (x1+2,x2-3)]
        if sum(mask[y*crop.width+x] for x,y in edges)/len(edges) < .90:
            continue
        footer_pixels = [(x,y) for x in range(x1+8,x2-8) for y in range(y2-26,y2-4)]
        if sum(mask[y*crop.width+x] for x,y in footer_pixels)/len(footer_pixels) < .80:
            continue
        # Use independent texture geometry on either side, mirroring outgoing
        # geometry. No lexical sender guess from the card's internal words.
        candidates = _matching_avatar_sides(crop, avatar_mask, avatar_components,
                                           (x1,y1,x2,y2), background)
        if len(candidates) != 1:
            continue
        owned = tuple(b for b in boxes if rect.left-1 <= b.rect.left and b.rect.right <= rect.right+1
                      and rect.top-1 <= b.rect.top and b.rect.bottom <= rect.bottom+1)
        if any(b not in owned and b.rect.left < rect.right and b.rect.right > rect.left
               and b.rect.top < rect.bottom and b.rect.bottom > rect.top for b in boxes):
            continue
        found.append(MiniProgramCard(rect, candidates[0], owned))
    return tuple(found)
