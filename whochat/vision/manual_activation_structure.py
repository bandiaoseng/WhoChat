"""Positive chat geometry for an explicit history-only activation button.

No OCR text, media meaning, current issue or delivery status is certified here.
Unknown old attachments stay unknown; the entire body remains fingerprinted.
"""
import hashlib
import json
from collections import Counter
from PIL import Image
from whochat.core.models import Speaker
from whochat.vision.bubbles import detect_bubbles, _is_green_pixel
from whochat.ocr.media_regions import _chat_background, _rgb_difference_mask, _components, _matching_avatar_sides

SCHEMA='whochat.manual_activation_structure.v1'

def _separate_outgoing_avatar(crop, mask, parts, local, background):
    """Recover a native green card when the bubble grouper included its logo.

    Work only inside an already detected outgoing bubble. The card must be a
    separate, mostly flat green component aligned with its left/top/bottom,
    plus the ordinary independent textured avatar and clean gutter. No OCR
    text is discarded or certified, and this is used only by explicit
    history-only activation.
    """
    x0,y0,x1,y1=local
    candidates=[]
    for left,top,right,bottom,_area in parts:
        if (abs(left-x0)>2 or abs(top-y0)>2 or abs(bottom-y1)>2
                or right-left<40 or bottom-top<22 or right>x1-30):
            continue
        card=(left,top,right,bottom)
        pixels=list(crop.crop(card).getdata())
        if not pixels: continue
        dominant=Counter(pixels).most_common(1)[0][0]
        if not _is_green_pixel(dominant): continue
        flat=sum(max(abs(pixel[c]-dominant[c]) for c in range(3))<=12 for pixel in pixels)
        if flat/len(pixels)<.75: continue
        owners=_matching_avatar_sides(crop,mask,parts,card,background)
        if owners==[Speaker.ME]: candidates.append(card)
    return candidates[0] if len(candidates)==1 else None

def manual_activation_structure(image,layout):
    if layout.target_app.value!='wechat' or layout.input_boundary_verified is not True:
        return None
    try:
        if isinstance(image,Image.Image): rgb=image.convert('RGB')
        else:
            with Image.open(image) as source: rgb=source.convert('RGB')
        band=layout.message_rect
        if (rgb.size!=(layout.window_rect.width,layout.window_rect.height)
                or band.left<0 or band.top<0 or band.right>rgb.width or band.bottom>rgb.height
                or band.width<240 or band.height<240):
            return None
        crop=rgb.crop(band.as_tuple()); background=_chat_background(crop)
        if background is None: return None
        mask=_rgb_difference_mask(crop,background,23)
        parts=_components(bytearray(mask),crop.width,crop.height)
        pairs=[]
        for bubble in detect_bubbles(rgb,layout):
            r=bubble.rect
            if (bubble.partial or r.width<40 or r.height<22 or r.width>band.width*.90
                    or r.left<band.left+3 or r.right>band.right-3
                    or r.top<band.top+3 or r.bottom>band.bottom-5): continue
            local=(r.left-band.left,r.top-band.top,r.right-band.left,r.bottom-band.top)
            owners=_matching_avatar_sides(crop,mask,parts,local,background)
            if len(owners)!=1 or owners[0]!=bubble.speaker:
                if bubble.speaker!=Speaker.ME: continue
                separated=_separate_outgoing_avatar(crop,mask,parts,local,background)
                if separated is None: continue
                local=separated
                owners=[Speaker.ME]
            pairs.append({'rect':list(local),'side':owners[0].value})
        pairs.sort(key=lambda p:p['rect'][1])
        if len(pairs)<2 or len(pairs)>24: return None
        if any(a['rect'][3]>b['rect'][1] for a,b in zip(pairs,pairs[1:])): return None
        payload={'schema':SCHEMA,'pairs':pairs,'message_size':[band.width,band.height],
                 'operator_declared_old_topic_closed':True,'current_message_content_used':False}
        payload['structure_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        payload['body_sha256']=hashlib.sha256(crop.tobytes()).hexdigest()
        return payload
    except (OSError,ValueError,TypeError,AttributeError,IndexError):
        return None

def validate_manual_activation_structure(structure,proof):
    expected={'schema','pairs','message_size','operator_declared_old_topic_closed','current_message_content_used',
              'structure_sha256','body_sha256'}
    if (set(structure)!=expected or proof.get('operator_selected_activation') is not True
            or structure.get('operator_declared_old_topic_closed') is not True
            or structure.get('current_message_content_used') is not False):
        raise ValueError('current_window_explicit_activation_required')
    unread=proof.get('unread_current_rows_sha256')
    if not isinstance(unread,str) or len(unread)!=64 or any(c not in '0123456789abcdef' for c in unread):
        raise ValueError('current_window_unread_rows_digest_invalid')
    size=structure.get('message_size'); pairs=structure.get('pairs')
    if (not isinstance(size,list) or len(size)!=2 or any(type(x) is not int or x<240 for x in size)
            or not isinstance(pairs,list) or not 2<=len(pairs)<=24):
        raise ValueError('current_window_activation_geometry_invalid')
    width,height=size; bottom=-1
    for pair in pairs:
        if not isinstance(pair,dict) or set(pair)!={'rect','side'} or pair['side'] not in {'me','other'}:
            raise ValueError('current_window_activation_geometry_invalid')
        rect=pair['rect']
        if not isinstance(rect,list) or len(rect)!=4 or any(type(x) is not int for x in rect):
            raise ValueError('current_window_activation_geometry_invalid')
        left,top,right,end=rect
        if not (3<=left<right<=width-3 and 40<=right-left<=width*.90
                and max(3,bottom)<=top<end<=height-5 and end-top>=22):
            raise ValueError('current_window_activation_geometry_invalid')
        bottom=end
    payload={k:v for k,v in structure.items() if k not in {'structure_sha256','body_sha256'}}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=structure['structure_sha256'] or proof.get('window_body_sha256')!=structure['body_sha256']:
        raise ValueError('current_window_structure_changed')
    for key in ('body_sha256','structure_sha256'):
        if not isinstance(structure[key],str) or len(structure[key])!=64 or any(c not in '0123456789abcdef' for c in structure[key]):
            raise ValueError('current_window_structure_digest_invalid')
