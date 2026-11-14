"""Conservative *layout* evidence for a textless picture conversation.

This does not recognize a photograph, crop, disease, sender, or message time.
It only recognizes repeated, separate avatar/card geometry in a real chat
surface. Missing/ambiguous evidence is a miss, never an empty incoming message.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from whochat.core.runtime import LayoutRegions, Rect, TargetApp


WINDOW_STRUCTURE_SCHEMA = "whochat.textless_chat_structure.v1"
WINDOW_PAGE_REASON = "verified_textless_picture_chat_structure"


def textless_chat_structure(image_path: str | Path | Image.Image, layout: LayoutRegions) -> dict | None:
    """Require two independent left-avatar/card rows, including a full tail.

    Title, input text, two-frame account proof, ownership and page exclusions
    are deliberately the caller's independent gates. A random image, blank
    chat, plain title/input box or a one-card preview cannot pass this helper.
    Coordinates in the returned proof are window-local, never screen absolute.
    """
    try:
        if (layout.target_app != TargetApp.WECHAT or not layout.input_boundary_verified
                or layout.message_rect.width < 240 or layout.message_rect.height < 240):
            return None
        origin = layout.window_rect
        band = layout.message_rect
        local = Rect(band.left-origin.left, band.top-origin.top, band.right-origin.left, band.bottom-origin.top)
        if isinstance(image_path,Image.Image):
            rgb=image_path.convert("RGB")
        else:
            with Image.open(image_path) as original:
                rgb = original.convert("RGB")
        if (rgb.size != (origin.width, origin.height) or local.left < 0 or local.top < 0
                or local.right > rgb.width or local.bottom > rgb.height):
            return None
        crop = rgb.crop(local.as_tuple())
        width, height = crop.size
        scale = min(1.0, 640 / width, 900 / height)
        small = crop.resize((max(1, round(width*scale)), max(1, round(height*scale))), Image.Resampling.NEAREST)
        sw, sh = small.size
        pixels = list(small.getdata())
        # The right-hand neutral rail is independent of the left picture lane.
        probes = [pixels[y*sw+x] for y in range(2, sh-2, 7) for x in (sw-5, sw-3)]
        background = tuple(sorted(p[c] for p in probes)[len(probes)//2] for c in range(3))
        if max(background)-min(background) > 14:
            return None
        mask = bytearray(max(abs(pixel[c]-background[c]) for c in range(3)) >= 22 for pixel in pixels)
        components = _components(mask, sw, sh)
        avatars, cards = [], []
        for x0,y0,x1,y1,area in components:
            x,y,w,h = x0/scale,y0/scale,(x1-x0)/scale,(y1-y0)/scale
            density = area / max(1,(x1-x0)*(y1-y0))
            if (18 <= w <= 52 and 18 <= h <= 52 and .70 <= w/h <= 1.43
                    and x <= width*.15 and density >= .45):
                avatars.append((x,y,w,h))
            if (w >= 70 and h >= 90 and w <= width*.68 and h <= height*.92
                    and width*.10 <= x <= width*.30 and density >= .40):
                region = crop.crop((round(x),round(y),round(x+w),round(y+h)))
                # Require genuine multi-tone content, not one colored bubble.
                tones = region.resize((16,16)).getdata()
                luminances = sorted((p[0]+p[1]+p[2])/3 for p in tones)
                if luminances[-20]-luminances[20] >= 28:
                    cards.append((x,y,w,h))
        pairs = []
        for card in cards:
            x,y,w,h = card
            matches = [avatar for avatar in avatars if abs(avatar[1]-y) <= 12
                       and 3 <= x-(avatar[0]+avatar[2]) <= 36]
            if len(matches) == 1:
                avatar = matches[0]
                pairs.append({"avatar": [round(v) for v in avatar], "card": [round(v) for v in card]})
        pairs.sort(key=lambda item:item["card"][1])
        if (len(pairs) < 2 or len(pairs) > 8
                or any(a["card"][1]+a["card"][3]+8 > b["card"][1] for a,b in zip(pairs,pairs[1:]))
                or pairs[-1]["card"][1]+pairs[-1]["card"][3] > height-5):
            return None
        payload = {"schema": WINDOW_STRUCTURE_SCHEMA, "pair_count": len(pairs), "pairs": pairs,
                   "message_size": [width,height], "tail_card_complete": True}
        payload["structure_sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        payload["body_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
        return payload
    except (OSError, ValueError, TypeError, IndexError, AttributeError, ZeroDivisionError):
        return None


def _components(mask: bytearray, width: int, height: int) -> list[tuple[int,int,int,int,int]]:
    components = []
    for start, active in enumerate(mask):
        if not active:
            continue
        mask[start] = 0
        pending = [start]
        left=right=start%width
        top=bottom=start//width
        area=0
        while pending:
            current = pending.pop()
            x,y=current%width,current//width
            left,right,top,bottom=min(left,x),max(right,x),min(top,y),max(bottom,y)
            area += 1
            for adjacent in (current-1 if x else -1,current+1 if x+1<width else -1,
                             current-width if y else -1,current+width if y+1<height else -1):
                if adjacent >= 0 and mask[adjacent]:
                    mask[adjacent] = 0
                    pending.append(adjacent)
        if area >= 40:
            components.append((left,top,right+1,bottom+1,area))
    return components
