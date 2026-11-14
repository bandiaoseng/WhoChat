"""Conservative chat-picture geometry, not image-content recognition.

Complete textured surfaces require an independently visible adjacent avatar.
A top-clipped outgoing surface requires two independent native owner anchors
and remains partial; the existing leading-prefix policy alone may omit it.
Image lettering is not an ordinary chat sentence. No image meaning is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
from pathlib import Path

from PIL import Image

from whochat.core.models import Speaker
from whochat.core.runtime import Rect
from whochat.vision.bubbles import _native_connected_components, _is_green_pixel
from whochat.vision.visual import load_rgb_image


@dataclass(frozen=True)
class ChatImageRegion:
    rect: Rect
    speaker: Speaker
    content_hash: str
    caption_rects: tuple[Rect, ...] = ()
    quoted: bool = False

    @property
    def partial(self) -> bool:
        return False


class _TopClippedImageRegion(ChatImageRegion):
    """Explicit partial occurrence; keep complete-image serialization stable."""
    @property
    def partial(self) -> bool:
        return True


def _rgb_difference_mask_python(image, background, threshold):
    """Original per-pixel arithmetic, also the compatibility fallback."""
    return bytearray(max(abs(pixel[c]-background[c]) for c in range(3)) >= threshold
                     for pixel in image.getdata())


def _rgb_difference_mask(image, background, threshold):
    """The same row-major RGB distance mask, without per-pixel Python work.

    Only ordinary RGB/byte-valued backgrounds enter the vector path. Signed
    arithmetic is essential: uint8 subtraction would wrap at 0/255 and change
    which pixels belong to a picture. Every buffer belongs to this call; no
    image, result, customer or frame state is cached or mutated.
    """
    try:
        if (image.mode == "RGB" and len(background) == 3
                and all(type(value) is int and 0 <= value <= 255 for value in background)
                and type(threshold) is int and 0 <= threshold <= 255):
            import numpy as np

            pixels = np.asarray(image, dtype=np.int16)
            if pixels.shape == (image.height, image.width, 3):
                difference = np.abs(pixels - np.asarray(background, dtype=np.int16))
                mask = np.max(difference, axis=2) >= threshold
                return bytearray(mask.astype(np.uint8).tobytes(order="C"))
    except (ImportError, AttributeError, TypeError, ValueError, OverflowError):
        pass
    return _rgb_difference_mask_python(image, background, threshold)


def chat_image_regions(image_path: str | Path | Image.Image, layout, boxes=()) -> tuple[ChatImageRegion, ...]:
    """Window-local coordinates; clipped leading media remains explicitly partial."""
    if (not image_path or layout.target_app.value != "wechat"
            or layout.input_boundary_verified is not True):
        return ()
    try:
        image = load_rgb_image(image_path)
        if image is None:
            return ()
        if image.size != (layout.window_rect.width, layout.window_rect.height):
            return ()
        band = layout.message_rect
        if (band.left < 0 or band.top < 0 or band.right > image.width
                or band.bottom > image.height or band.width < 220 or band.height < 160):
            return ()
        crop = image.crop(band.as_tuple())
        width, height = crop.size
        background = _chat_background(crop)
        if background is None:
            return ()
        mask = _rgb_difference_mask(crop, background, 23)
        components = _components(bytearray(mask), width, height)
        found = []
        for x0,y0,x1,y1,area in components:
            w,h = x1-x0,y1-y0
            top_clipped = 0 <= y0 <= 2
            # Rectangular photos/posters only, not text, file cards or avatars.
            if (w < 70 or h < 72 or w > width*.74 or h > height*.85
                    or w/h < .28 or w/h > 2.6 or x0 < 3 or (y0 < 4 and not top_clipped)
                    or x1 > width-3 or y1 > height-5 or area/(w*h) < .40):
                continue
            picture = crop.crop((x0,y0,x1,y1))
            samples = list(picture.resize((24,24)).getdata())
            tones = sorted(sum(pixel)/3 for pixel in samples)
            quantized = [(r//24,g//24,b//24) for r,g,b in samples]
            if (len(set(quantized)) < 16 or tones[-30]-tones[30] < 40
                    or max(quantized.count(value) for value in set(quantized)) > len(samples)*.64):
                continue
            # All four sides must independently bound a filled surface; a
            # collection of unrelated text lines is not a rectangular picture.
            sides = [
                sum(mask[y*width+x0] or mask[y*width+min(x0+1,x1-1)] for y in range(y0,y1))/h,
                sum(mask[y*width+x1-1] or mask[y*width+max(x1-2,x0)] for y in range(y0,y1))/h,
                sum(mask[y0*width+x] or mask[min(y0+1,y1-1)*width+x] for x in range(x0,x1))/w,
                sum(mask[(y1-1)*width+x] or mask[max(y1-2,y0)*width+x] for x in range(x0,x1))/w,
            ]
            if min(sides) < .46:
                continue
            if top_clipped:
                # A viewport may cut off the original avatar along with the
                # top of an outgoing picture. Never promote it to a complete
                # image: require three intact edges and two independent
                # later native bubbles proving the same outgoing rail.
                if (min(sides[0], sides[1], sides[3]) < .75
                        or not _top_picture_outgoing_rail(crop, mask, components,
                            (x0,y0,x1,y1), background)):
                    continue
                speaker = Speaker.ME
            else:
                avatar_sides = _matching_avatar_sides(crop, mask, components, (x0,y0,x1,y1), background)
                if len(avatar_sides) != 1:
                    continue
                speaker = avatar_sides[0]
            rect = Rect(band.left+x0,band.top+y0,band.left+x1,band.top+y1)
            digest = hashlib.sha256(picture.tobytes()).hexdigest()
            region_type = _TopClippedImageRegion if top_clipped else ChatImageRegion
            found.append(region_type(rect,speaker,digest))
        found.extend(_white_paper_regions(crop, mask, components, band, background, found))
        found.extend(_quoted_image_regions(crop,mask,components,band,background,boxes))
        found.sort(key=lambda item:(item.rect.top,item.rect.left))
        # Overlapping candidates are ambiguous, not permission to hide OCR.
        return tuple(item for item in found if all(item is other or not _intersects(item.rect,other.rect) for other in found))
    except (OSError, ValueError, TypeError, AttributeError, IndexError):
        return ()


def _top_picture_outgoing_rail(crop, mask, components, picture, background):
    """Only a leading clipped picture, with independent native owner evidence.

    Two complete green bubbles below it must each have their own textured
    right avatar, a clear gutter and the same right edge (the tail adds up to
    seven pixels). Printed bubbles INSIDE the image cannot be these anchors.
    The result permits only a partial prefix; it never authorizes its content.
    """
    x0,y0,x1,y1=picture
    width,height=crop.size
    anchors=[]
    for ax,ay,bx,by,area in components:
        w,h=bx-ax,by-ay
        if (ay < y1+8 or by > height-5 or w < 48 or not 24 <= h <= 100
                or area/(w*h) < .65 or not 0 <= bx-x1 <= 7):
            continue
        bubble=crop.crop((ax,ay,bx,by))
        try:
            if sum(_is_green_pixel(p) for p in bubble.resize((24,16)).getdata()) < 24*16*.70:
                continue
        finally:
            bubble.close()
        if _matching_avatar_sides(crop,mask,components,(ax,ay,bx,by),background) == [Speaker.ME]:
            anchors.append((ay,by))
    anchors.sort()
    if not any(a[1]+8 <= b[0] for a in anchors for b in anchors):
        return False
    gutter=[crop.getpixel((x,y)) for x in range(x0,x1)
            for y in range(y1+2,min(height,y1+6))]
    return bool(gutter and sum(max(abs(p[c]-background[c]) for c in range(3)) <= 16
                               for p in gutter) >= len(gutter)*.96)


def native_file_card_avatar_speaker(image, layout, estimated):
    """Resolve a metadata-proved card by its actual surface and adjacent avatar.

    Used only when window-fraction geometry is inconclusive. A low-contrast
    native card must form one closed component close to the independently
    estimated filename/icon bounds. No arbitrary wider region or lone avatar
    grants a sender. The caller still requires all file metadata and pixels.
    """
    band=layout.message_rect
    crop=image.crop(band.as_tuple())
    try:
        background=_chat_background(crop)
        if background is None:
            return None
        mask=_rgb_difference_mask(crop,background,12)
        parts=_components(bytearray(mask),crop.width,crop.height)
        ex=(estimated.left-band.left,estimated.top-band.top,
            estimated.right-band.left,estimated.bottom-band.top)
        matches=[]
        for x0,y0,x1,y1,area in parts:
            w,h=x1-x0,y1-y0
            if (w < 100 or h < 48 or h > 170 or w > band.width*.80
                    or x0 < 3 or y0 < 4 or x1 > band.width-3 or y1 > band.height-5
                    or area/(w*h) < .70
                    or any(abs(a-b)>16 for a,b in zip((x0,y0,x1,y1),ex))):
                continue
            sides=_matching_avatar_sides(crop,mask,parts,(x0,y0,x1,y1),background)
            if len(sides)==1:
                matches.append(sides[0])
        return matches[0] if len(matches)==1 else None
    finally:
        crop.close()


def _chat_background(crop):
    """Agree on the real canvas from both interior gutters, not window chrome.

    A scrollbar/window border can occupy an entire outer sample column. A
    channel-wise median of that column and the opposite canvas invents a
    third colour and connects a white poster to the whole chat background.
    Require the same dominant *observed* neutral pixel on both interior rails.
    Ambiguous/covered rails grant no image-text exemption.
    """
    width, height = crop.size
    rails = []
    # A native GetWindowRect capture contains the invisible resize border as
    # well as the painted client. With only two columns, a single outer-frame
    # column supplied half of the samples and vetoed otherwise complete media.
    # Sample a band of independent interior columns instead. The same real
    # neutral colour must still dominate BOTH sides; conflicting/covered bands
    # do not grant permission to classify their text as image metadata.
    offsets = (8, 16, 24, 32)
    for columns in (offsets, tuple(width - 1 - x for x in offsets)):
        probes = [crop.getpixel((x, y)) for x in columns for y in range(8, height - 8, 9)]
        neutral = Counter(p for p in probes if max(p) - min(p) <= 12)
        if len(probes) < 12 or not neutral:
            return None
        colour, count = neutral.most_common(1)[0]
        if count < len(probes) * .55:
            return None
        rails.append(colour)
    return rails[0] if rails[0] == rails[1] else None


def _white_paper_regions(crop, strong_mask, components, band, background, existing):
    """Bound light paper by its own edge, not by disconnected printed words.

    White product labels can differ from the light chat canvas by only five
    levels. This separate path requires a closed bright paper surface, varied
    coloured printing and the same independent avatar/gutter as a normal
    photograph. It does not lower the threshold for ordinary chat bubbles.
    """
    if max(background)-min(background) > 5 or not 240 <= min(background) <= 252:
        return []
    width, height = crop.size
    mask = _rgb_difference_mask(crop, background, 3)
    found = []
    for x0,y0,x1,y1,area in _components(bytearray(mask), width, height):
        w,h = x1-x0,y1-y0
        if (w < 70 or h < 72 or w > width*.74 or h > height*.85
                or not .28 <= w/h <= 2.6 or x0 < 3 or y0 < 4
                or x1 > width-3 or y1 > height-5 or area/(w*h) < .80):
            continue
        rect = Rect(band.left+x0,band.top+y0,band.left+x1,band.top+y1)
        # The ordinary textured-photo path already owns this surface. Never
        # create a second overlapping occurrence with a slightly larger edge.
        if any(_intersects(rect, item.rect) for item in existing):
            continue
        picture = crop.crop((x0,y0,x1,y1))
        values = list(picture.getdata())
        coloured = sum(max(pixel)-min(pixel) >= 30 for pixel in values)
        if not _has_bright_canvas(picture, background) or coloured/len(values) < .025:
            continue
        samples = list(picture.resize((24,24)).getdata())
        tones = sorted(sum(pixel)/3 for pixel in samples)
        if (len({(r//24,g//24,b//24) for r,g,b in samples}) < 16
                or tones[-30]-tones[30] < 40):
            continue
        sides = [
            sum(mask[y*width+x0] or mask[y*width+x0+1] for y in range(y0,y1))/h,
            sum(mask[y*width+x1-1] or mask[y*width+x1-2] for y in range(y0,y1))/h,
            sum(mask[y0*width+x] or mask[(y0+1)*width+x] for x in range(x0,x1))/w,
            sum(mask[(y1-1)*width+x] or mask[(y1-2)*width+x] for x in range(x0,x1))/w,
        ]
        if min(sides) < .75:
            continue
        sides = _matching_avatar_sides(crop, strong_mask, components, (x0,y0,x1,y1), background)
        if len(sides) == 1:
            found.append(ChatImageRegion(rect, sides[0], hashlib.sha256(picture.tobytes()).hexdigest()))
    return found


def _has_bright_canvas(picture, background):
    """Prove connected paper behind the printing, not a majority-white image.

    A poster may contain a large photo collage. Its paper still connects across
    the canvas; the fraction of printed/photo pixels says nothing about whether
    the outside rectangle is complete. Tiny white glyphs/highlights cannot
    establish this surface. Edge closure, texture, avatar and gutter checks are
    separately required by the caller.
    """
    width, height = picture.size
    mask = bytearray(min(p) >= 248 and max(p)-min(p) <= 5
                     and min(p)-max(background) >= 3 for p in picture.getdata())
    columns=set()
    for x0,y0,x1,y1,area in _components(mask, width, height):
        # Printed borders may split one poster into several panels. Each
        # contributing panel must be a substantial surface, not a glyph or a
        # thin highlight; together the paper must span the image canvas.
        if (x1-x0 >= max(8,width*.05) and y1-y0 >= height*.25
                and area >= (x1-x0)*(y1-y0)*.25):
            columns.update(range(x0,x1))
    return len(columns)>=width*.80


def _quoted_image_regions(crop,mask,components,band,background,boxes):
    """An outgoing image quote: grey author + thumbnail + proved upper bubble.

    The author label is metadata inside this independent structure, not an
    ordinary chat sentence. A colon, thumbnail or grey line alone never grants
    an exemption. Incoming/partial/ambiguous references remain unclassified.
    """
    width,height=crop.size
    found=[]
    candidates=list(components)
    # Referenced product posters keep their aspect ratio and light canvas;
    # their printed marks need not form a connected square thumbnail.
    if max(background)-min(background) <= 5 and 240 <= min(background) <= 252:
        light=_rgb_difference_mask(crop, background, 3)
        for part in _components(bytearray(light),width,height):
            x0,y0,x1,y1,area=part
            w,h=x1-x0,y1-y0
            # At thumbnail resolution printing breaks the white pixels into
            # islands. Prove the closed miniature here; the grey author label,
            # textured pixels and independently owned parent are required below.
            if not (28<=w<=52 and 28<=h<=52 and area/(w*h)>=.80):
                continue
            edges=[sum(light[y*width+x0] or light[y*width+x0+1] for y in range(y0,y1))/h,
                   sum(light[y*width+x1-1] or light[y*width+x1-2] for y in range(y0,y1))/h,
                   sum(light[y0*width+x] or light[(y0+1)*width+x] for x in range(x0,x1))/w,
                   sum(light[(y1-1)*width+x] or light[(y1-2)*width+x] for x in range(x0,x1))/w]
            if min(edges)<.75:
                continue
            candidates=[p for p in candidates if not (x0<=p[0] and y0<=p[1] and p[2]<=x1 and p[3]<=y1)]
            candidates.append(part)
    for x0,y0,x1,y1,area in candidates:
        w,h=x1-x0,y1-y0
        if not (28<=w<=52 and 28<=h<=52 and area/(w*h)>=.45
                and x0>=width*.25 and x1<=width*.82 and y0>5 and y1<height-5):
            continue
        thumb=crop.crop((x0,y0,x1,y1))
        tones=list(thumb.resize((12,12)).getdata())
        if len({(r//24,g//24,b//24) for r,g,b in tones})<12:
            continue
        labels=[box for box in boxes if box.confidence>=.65 and box.text.rstrip().endswith((":","："))
                and 70<=box.rect.width<=width*.70 and 8<=box.rect.height<=24
                and 0<=band.left+x0-box.rect.right<=26
                and -4<=box.rect.top-(band.top+y0)<=14]
        if len(labels)!=1:
            continue
        label=labels[0]
        local=Rect(label.rect.left-band.left,label.rect.top-band.top,
                   label.rect.right-band.left,label.rect.bottom-band.top)
        if local.left<0 or local.top<0:
            continue
        ink=[pixel for pixel in crop.crop(local.as_tuple()).getdata()
             if max(abs(pixel[c]-background[c]) for c in range(3))>=35]
        if not ink or sum(max(pixel)-min(pixel)<=18 and 85<=sum(pixel)/3<=220 for pixel in ink)/len(ink)<.90:
            continue
        parents=[]
        for bx,by,ex,ey,_ in components:
            if (ex-bx<100 or ey-by<24 or not 3<=y0-ey<=25 or local.left<bx-4
                    or abs(ex-x1)>28 or ex<=x0 or ey-by>height*.5):
                continue
            pixels=list(crop.crop((bx,by,ex,ey)).getdata())
            if sum(g-r>=20 and g-b>=20 for r,g,b in pixels)/len(pixels)<.60:
                continue
            if _matching_avatar_sides(crop,mask,components,(bx,by,ex,ey),background)==[Speaker.ME]:
                parents.append((bx,by,ex,ey))
        if len(parents)!=1:
            continue
        rect=Rect(band.left+x0,band.top+y0,band.left+x1,band.top+y1)
        found.append(ChatImageRegion(rect,Speaker.ME,hashlib.sha256(thumb.tobytes()).hexdigest(),
                                     (label.rect,),True))
    return found


def _matching_avatar_sides(crop, mask, components, card, background):
    width,height=crop.size
    x0,y0,x1,y1=card
    matches=set()
    # Connected pieces can belong to one avatar (a logo has white gutters).
    # Require the union to remain a square on the outer avatar rail, with a
    # clean vertical page-background gutter to the picture.
    seeds=[part for part in components if 3 <= part[2]-part[0] <= 52
           and 3 <= part[3]-part[1] <= 52 and abs(part[1]-y0) <= 8
           and (part[0] >= width*.80 or part[2] <= width*.18)]
    for seed in seeds:
        sx,sy,ex,ey,_=seed
        near=[part for part in components if part[0] >= sx-4 and part[2] <= ex+4
              and part[1] >= y0-5 and part[3] <= y0+52]
        if not near:
            continue
        ax,ay,bx,by=min(p[0] for p in near),min(p[1] for p in near),max(p[2] for p in near),max(p[3] for p in near)
        aw,ah=bx-ax,by-ay
        if not (23 <= aw <= 52 and 23 <= ah <= 52 and .72 <= aw/ah <= 1.39 and abs(ay-y0) <= 8):
            continue
        if ax >= x1+7 and ax-x1 <= 42 and ax >= width*.80:
            speaker=Speaker.ME
            left,right=x1+2,ax-2
        elif bx <= x0-7 and x0-bx <= 42 and bx <= width*.18:
            speaker=Speaker.OTHER
            left,right=bx+2,x0-2
        else:
            continue
        if right-left < 2:
            continue
        gutter=[crop.getpixel((x,y)) for x in range(left,right)
                for y in range(max(0,y0),min(height,by))]
        if not gutter or sum(max(abs(pixel[c]-background[c]) for c in range(3)) <= 16 for pixel in gutter)/len(gutter) < .96:
            continue
        avatar=crop.crop((ax,ay,bx,by))
        values=list(avatar.resize((12,12)).getdata())
        if len({(r//32,g//32,b//32) for r,g,b in values}) < 8:
            continue
        matches.add(speaker)
    return sorted(matches,key=lambda item:item.value)


def _intersects(a,b):
    return min(a.right,b.right)>max(a.left,b.left) and min(a.bottom,b.bottom)>max(a.top,b.top)


def _components(mask,width,height):
    native = _native_connected_components(mask, width, height)
    if native is not None:
        # The original media DFS consumes its mutable bitmap, including tiny
        # components that are not returned. Preserve that separate contract.
        if type(mask) is bytearray:
            mask[:] = bytes(len(mask))
        else:
            for index, value in enumerate(mask):
                if value:
                    mask[index] = 0
        return [part for part in native if part[4] >= 9]
    return _components_dfs(mask, width, height)


def _components_dfs(mask,width,height):
    result=[]
    for start in range(len(mask)):
        if not mask[start]:
            continue
        mask[start]=0
        stack=[start]
        x0=x1=start%width
        y0=y1=start//width
        area=0
        while stack:
            current=stack.pop()
            x,y=current%width,current//width
            x0,y0,x1,y1=min(x0,x),min(y0,y),max(x1,x),max(y1,y)
            area+=1
            for next_item in ((current-1 if x else -1),(current+1 if x+1<width else -1),
                              (current-width if y else -1),(current+width if y+1<height else -1)):
                if next_item >= 0 and mask[next_item]:
                    mask[next_item]=0
                    stack.append(next_item)
        if area >= 9:
            result.append((x0,y0,x1+1,y1+1,area))
    return result
