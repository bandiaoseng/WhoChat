"""Business routing AFTER current-source validation, never a capture bypass.

A shipping artefact is background, not a customer question. An explicit,
dated outgoing shipment update can close an OLD shipping exchange. The router
does not prove receipt, fulfilment, identity, crop stage, or permission to copy.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re

TZ = timezone(timedelta(hours=8))
_SHIPPED = re.compile(r'已(?:经)?(?:给您|给你|为您|帮您)?(?:安排)?发(?:出|货)|'
                      r'(?:昨天|前天|周[一二三四五六日天]|星期[一二三四五六日天]).{0,3}发的|'
                      r'(?:给您|给你|已经|已).{0,4}(?:寄出|发走)|已揽收')
_OPEN = re.compile(r'[?？]|吗|么|何时|什么时候|怎么|多久|催|投诉|没收到|未收到|还没|尚未|'
                   r'还未|没有|未发|没发|不是|不能|无法|别发|不要发|退货|退款|取消|破损|发错|缺货|'
                   r'异常|丢件|怎么办|再确认|确认一下|稍后回复|等我查|再查|帮你查|帮您查')
_ACK = re.compile(r'^(?:好(?:的|了)?|收到(?:了)?|知道了|明白(?:了)?|谢谢|行|嗯|可以)[\s。！!，,]*$')


@dataclass(frozen=True)
class CommunicationIntent:
    mode: str
    reason: str
    evidence_ordinal: int | None = None


def select_communication_intent(view, *, now=None):
    """Return a narrow local hint from a caller-validated view.

    Never consumes raw OCR, unlinked archives, observation timestamps, or title
    guesses. Unknown dates/new concerns keep the existing reply/next-step path.
    """
    if view is None:
        return CommunicationIntent('reply', 'current_view_required')
    if view.source_kind == 'current_window_reactivation':
        return CommunicationIntent('reactivation', 'verified_textless_window')
    rows = view.messages
    default = 'next_step' if rows and rows[-1].speaker.value == 'me' else 'reply'
    if not rows or any(m.partial or m.order_uncertain for m in rows):
        return CommunicationIntent(default, 'current_rows_not_complete')
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        raise ValueError('communication_intent_timezone_required')
    # At most two literal acknowledgements after a shipment declaration; not
    # a general "last speaker is me" rule or a question/complaint classifier.
    for index in range(len(rows)-1, max(-1,len(rows)-4), -1):
        message = rows[index]
        if (message.speaker.value != 'me' or message.content_type != 'text'
                or not _SHIPPED.search(message.text) or _OPEN.search(message.text)):
            continue
        suffix = rows[index+1:]
        if any(m.content_type != 'text' or not _ACK.fullmatch(m.text.strip()) for m in suffix):
            continue
        if any(_OPEN.search(m.text) for m in suffix):
            continue
        # A previous unrelated problem is not answered by "I sent the goods".
        incoming = next((m for m in reversed(rows[:index]) if m.speaker.value == 'other'), None)
        if incoming is not None and incoming.content_type == 'text' and _OPEN.search(incoming.text):
            if not re.fullmatch(r'(?:什么时候|何时|今天|明天|现在)?(?:能|可以|会)?(?:给我)?发货[吗么?？。 ]*', incoming.text.strip()):
                return CommunicationIntent(default, 'unresolved_customer_concern')
        times = []
        for m in (message, *suffix):
            if m.time_source not in {'ocr','verified','platform','source'} or not m.message_time:
                return CommunicationIntent(default, 'shipment_time_unverified')
            try:
                stamp = datetime.fromisoformat(m.message_time.replace('Z','+00:00'))
            except (ValueError, TypeError):
                return CommunicationIntent(default, 'shipment_time_unverified')
            if stamp.tzinfo is None or stamp > now:
                return CommunicationIntent(default, 'shipment_time_unverified')
            times.append(stamp.astimezone(TZ))
        if max(times).date() >= now.astimezone(TZ).date():
            return CommunicationIntent(default, 'shipment_exchange_today')
        return CommunicationIntent('reactivation', 'old_outgoing_shipment_topic_closed', message.ordinal)
    return CommunicationIntent(default, 'current_conversation')
