"""Rule-based long-term-memory extraction.

The product needs "what did this customer tell me before" to be remembered per
contact.  Messages already land in SQLite through ``ChatIngestionService``; this
module is the missing producer that turns those messages into ``memories`` rows.

Design constraints (agreed with the user, do not change silently):

* **No network, no AI.**  Extraction is pure pattern matching over text that was
  already OCR'd locally.  It therefore costs nothing against the cloud request
  quota (``AIProviderConfig.max_daily_cloud_requests``) and sends nothing out.
* **Direct messages only.**  Inside a group the speaker attribution is weak
  (``Speaker.MEMBER`` inference is not reliable yet), so a fact OCR'd from a
  group could be attributed to the wrong person.  Groups are skipped with an
  explicit reason instead of writing low-confidence memories.
* **Unconfirmed contacts are skipped.**  ``block_memory_for_unconfirmed_contact``
  means OCR-discovered contacts (status ``UNCONFIRMED``/``SUSPECTED``) only
  accumulate chat messages; memory starts once the user confirms the contact in
  the UI.
* **Everything is written as ``pending``.**  A human confirms or rejects each
  memory before it may reach the AI prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from whochat.config import AppConfig, ConfigStore
from whochat.core.models import (
    Contact,
    ContactStatus,
    ConversationType,
    MemoryKind,
    Message,
    Speaker,
)
from whochat.storage.repositories import LogRepository, MemoryRepository

# A single screen can legitimately surface several facts; cap the batch so one
# dense conversation cannot flood the review queue.
MAX_MEMORIES_PER_BATCH = 8
# Temporary arrangements ("明天发货") are short-lived by nature.
TEMPORARY_TTL_DAYS = 7

_CROPS = (
    "大豆", "黄豆", "水稻", "旱稻", "玉米", "小麦", "大麦", "高粱", "谷子", "糜子",
    "马铃薯", "土豆", "红薯", "甘薯", "木薯", "山药", "芋头",
    "花生", "油菜", "向日葵", "芝麻", "棉花", "甘蔗", "甜菜", "烟叶", "烤烟", "茶叶",
    "辣椒", "懒椒", "线椒", "螺丝椒", "甜椒", "彩椒", "番茄", "西红柿", "黄瓜", "苦瓜",
    "丝瓜", "南瓜", "冬瓜", "西瓜", "甜瓜", "哈密瓜", "茄子", "豆角", "豇豆", "豌豆",
    "菜豆", "白菜", "青菜", "菠菜", "生菜", "油麦菜", "芹菜", "韭菜", "大葱", "小葱",
    "香葱", "洋葱", "大蒜", "生姜", "大姜", "小黄姜", "萝卜", "胡萝卜", "甘蓝", "花菜",
    "西兰花", "莲藕", "荸荠", "茭白", "芦笋", "秋葵", "龙牙百合", "百合", "中药材",
    "三七", "人参", "枸杞", "石斛", "金银花", "柑橘", "柑桔", "沃柑", "砂糖橘", "沙糖桔",
    "脐橙", "冰糖橙", "柚子", "蜜柚", "红美人", "爱媛", "丑橘", "春见", "金桔",
    "柠檬", "橙子", "橘子", "苹果", "梨", "桃子", "黄桃", "油桃", "蟠桃", "李子",
    "杏", "樱桃", "大樱桃", "车厘子", "葡萄", "阳光玫瑰", "夏黑", "猕猴桃",
    "石榴", "枇杷", "杨梅", "草莓", "蓝莓", "树莓", "火龙果", "百香果", "香蕉",
    "芒果", "荔枝", "龙眼", "菠萝", "菠萝蜜", "榴莲", "山竹", "椰子", "槟榔",
    "木瓜", "番石榴", "释迦", "莲雾", "杨桃", "核桃", "板栗", "枣", "冬枣", "花椒",
    "八角", "油茶", "油桐", "毛竹", "苗木", "花卉", "月季", "牡丹", "芍药", "菊花",
)

_PROVINCES = (
    "内蒙古", "黑龙江", "广西", "西藏", "新疆", "宁夏", "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃",
    "青海", "台湾", "香港", "澳门",
)

# Words that look like a place name tail but carry no location meaning.
_REGION_TAIL_STOPWORDS = (
    "这边", "那边", "这个", "那个", "什么", "哪里", "情况", "地区", "地方", "附近",
    "一块", "一起", "市场", "价格", "多久", "多少",
)

# Verbs / particles that terminate a place name ("黑河种了" -> "黑河").
_REGION_TAIL_STOP_CHARS = "种在想有了的这那我和你他她它都和跟用打来去要就还也说明问发买卖收送附近亩"

_PRODUCTS = (
    "甲霜·噁霉灵", "甲霜噁霉灵", "噁霉灵", "恶霉灵", "精甲霜灵", "吡唑醚菌酯",
    "苯醚甲环唑", "咪鲜胺", "代森锰锌", "百菌清", "多菌灵", "甲基硫菌灵",
    "戊唑醇", "氟硅唑", "丙环唑", "烯酰吗啉", "霜脲氰", "氢氧化铜", "王铜",
    "春雷霉素", "中生菌素", "噻唑锌", "氯溴异氰尿酸", "辛菌胺",
    "阿维菌素", "甲维盐", "高效氯氟氰菊酯", "联苯菊酯", "噻虫嗪", "吡虫啉",
    "啶虫脒", "螺虫乙酯", "乙基多杀菌素", "氯虫苯甲酰胺", "虫螨腈", "哒螨灵",
    "螺螨酯", "乙螨唑", "矿物油", "石硫合剂",
    "稼镁·钛", "稼镁钛", "Q5Pro", "q5pro", "生根小蜜蜂", "小蜜蜂",
    "复硝酚钠", "芸苔素", "芸苔素内酯", "胺鲜酯", "氨基寡糖素", "海藻酸",
    "矿源黄腐酸钾", "黄腐酸钾", "腐植酸", "磷钾母液", "磷酸二氢钾",
    "海藻钙镁硼锌铁", "钙镁硼锌铁", "糖醇钙", "糖醇硼", "螯合铁", "硫酸钾",
    "水溶肥", "叶面肥", "生根粉", "生根剂", "氨基酸", "鱼蛋白", "微生物菌剂",
)

_GENERIC_PRODUCT_RE = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9·]{2,12}?"
    r"(?:水溶肥|叶面肥|菌肥|有机肥|复合肥|冲施肥|母液|悬浮剂|微乳剂|水剂|乳油|"
    r"可湿性粉剂|水分散粒剂|颗粒剂|种衣剂|杀菌剂|杀虫剂|杀螨剂|调节剂|"
    r"生根剂|生根粉|菌剂|营养液))"
)

_DISEASES = (
    "溃疡病", "霜霉病", "灰霉病", "炭疽病", "白粉病", "根腐病", "茎基腐病", "青枯病",
    "病毒病", "花叶病", "疫病", "晚疫病", "早疫病", "绵疫病", "锈病", "条锈病",
    "叶锈病", "秆锈病", "纹枯病", "稻瘟病", "立枯病", "猝倒病", "枯萎病", "黄萎病",
    "软腐病", "黑腐病", "疮痂病", "黑斑病", "褐斑病", "叶斑病", "斑点病", "圆斑病",
    "蔓枯病", "黑星病", "轮纹病", "流胶病", "黄龙病", "黄化病", "赤霉病", "全蚀病",
    "紫斑病", "菌核病", "白绢病", "根肿病", "沤根", "烂根", "黑胫病", "溃疡",
    "根结线虫", "根线虫", "线虫",
)

_PESTS = (
    "蚜虫", "棉蚜", "桃蚜", "红蜘蛛", "叶螨", "蓟马", "白粉虱", "烟粉虱", "木虱",
    "介壳虫", "蚧壳虫", "矢尖蚧", "红蜡蚧", "斜纹夜蛾", "甜菜夜蛾", "草地贪夜蛾",
    "棉铃虫", "螟虫", "二化螟", "三化螟", "稻纵卷叶螟", "稻飞虱", "稻蓟马",
    "地老虎", "蛴螬", "金针虫", "蝼蛄", "斑潜蝇", "果实蝇", "实蝇", "蝽蟓", "椿象",
    "跳甲", "猿叶甲", "菜青虫", "小菜蛾", "蝗虫", "蜗牛", "蛞蝓", "白蚁", "天牛",
    "象甲", "盲蝽", "螨",
)

_STAGES = (
    "苗期", "育苗", "扦插", "嫁接", "移栽", "缓苗", "返青", "分蘖", "拔节", "孕穗",
    "抽穗", "扬花", "灌浆", "乳熟", "现蕾", "初花", "盛花", "谢花", "坐果", "幼果",
    "膨果", "着色", "转色", "成熟", "采收", "采后", "月子肥", "旺长", "越冬", "萌芽",
)

_AREA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:多|余)?\s*(万亩|万公顷|亩|公顷|晌|垧)")

# Only the customer's own words are trusted for area figures ("我种了 5000 亩"),
# and both sides are scanned so a quote relayed by the salesperson still counts.
_PROBLEM_CUES = ("病", "虫", "螨", "打药", "用药", "防治", "治", "防")
_PREFERENCE_CUES = (
    ("价格", ("价格", "价钱", "便宜", "优惠", "划算", "性价比", "贵")),
    ("见效速度", ("见效", "效果快", "几天见效", "立竿见影", "打下去")),
    ("技术指导", ("指导", "配方", "怎么用", "用法", "用量", "搭配", "混配")),
    ("样品/试用", ("样品", "试用", "小样", "先试")),
    ("发货节奏", ("发货", "走货", "物流", "到货", "尽快", "着急")),
)
_TEMPORARY_CUES = ("发货", "寄", "安排", "打款", "付款", "发样品", "再打", "补货", "打药")
_TIME_CUES = (
    "今天", "明天", "后天", "这两天", "这几天", "本周", "这周", "下周", "周内",
    "月底", "月初", "尽快", "过两天",
)
_AVOID_RE = re.compile(r"(?:不要|别|不能|避免)(?:再)?\s*([\u4e00-\u9fa5A-Za-z0-9]{2,12})")

# Speakers whose text may be used.  Members/unknown/system lines are excluded for
# the same reason ``transcript_stitcher._usable`` excludes them.
_USABLE_SPEAKERS = {Speaker.ME, Speaker.OTHER}


@dataclass(frozen=True)
class MemoryCandidate:
    kind: MemoryKind
    content: str
    confidence: float
    category: str
    evidence: str
    source_message_id: str | None = None
    expires_at: str | None = None


@dataclass(frozen=True)
class MemoryExtractionResult:
    created: int
    candidates: tuple[MemoryCandidate, ...] = ()
    skipped_reason: str | None = None
    blocked: bool = False

    @property
    def contents(self) -> tuple[str, ...]:
        return tuple(candidate.content for candidate in self.candidates)


class MemoryExtractionService:
    """Turns newly ingested messages into pending ``memories`` rows for one contact."""

    def __init__(
        self,
        memories: MemoryRepository,
        logs: LogRepository | None = None,
        config_provider: Callable[[], AppConfig] | None = None,
    ) -> None:
        self.memories = memories
        self.logs = logs
        # Config is read lazily so a settings change (e.g. turning long-term
        # memory off) takes effect on the next ingestion instead of restarting.
        self._config_provider = config_provider or ConfigStore().load

    def extract(self, contact: Contact, messages: Iterable[Message]) -> MemoryExtractionResult:
        config = self._config_provider()
        reason = _policy_block_reason(contact, config)
        if reason is not None:
            return self._finish(MemoryExtractionResult(0, (), reason, blocked=True), contact)

        candidates = extract_memory_candidates(messages)
        if not candidates:
            return self._finish(MemoryExtractionResult(0, (), "no_candidates"), contact)

        created = 0
        stored: list[MemoryCandidate] = []
        for candidate in candidates:
            memory = self.memories.add_or_ignore_pending(
                contact.id,
                candidate.kind,
                candidate.content,
                confidence=candidate.confidence,
                source_message_id=candidate.source_message_id,
                expires_at=candidate.expires_at,
            )
            if memory is not None:
                created += 1
                stored.append(candidate)
        skipped = None if created else "already_known"
        return self._finish(MemoryExtractionResult(created, tuple(stored), skipped), contact)

    def _finish(self, result: MemoryExtractionResult, contact: Contact) -> MemoryExtractionResult:
        if self.logs:
            self.logs.append(
                "info" if not result.blocked else "warning",
                "memory",
                "memory_extraction",
                (
                    f"contact={contact.display_name} created={result.created} "
                    f"reason={result.skipped_reason or 'created'} "
                    f"contents={'|'.join(result.contents) or '-'}"
                ),
            )
        return result


def _policy_block_reason(contact: Contact, config: AppConfig) -> str | None:
    if not config.privacy.enable_long_term_memory:
        return "long_term_memory_disabled"
    if contact.status in {ContactStatus.IGNORED, ContactStatus.MERGED} or contact.merged_into:
        return "contact_not_tracked"
    if config.capture.block_memory_for_unconfirmed_contact and contact.status in {
        ContactStatus.UNCONFIRMED,
        ContactStatus.SUSPECTED,
    }:
        return "contact_unconfirmed"
    if contact.conversation_type == ConversationType.GROUP:
        return "group_scope"
    return None


def extract_memory_candidates(messages: Iterable[Message]) -> list[MemoryCandidate]:
    """Pure extraction: no database, no config, no network."""
    usable = [_text(message) for message in messages if _usable(message)]
    if not usable:
        return []

    collected: dict[tuple[MemoryKind, str], MemoryCandidate] = {}
    hits: dict[str, int] = {}

    for text, message, speaker in usable:
        for candidate in _candidates_for_message(text, message, speaker):
            key = (candidate.kind, candidate.content)
            hits[candidate.category] = hits.get(candidate.category, 0) + 1
            current = collected.get(key)
            if current is None or candidate.confidence > current.confidence:
                collected[key] = candidate

    ordered = sorted(collected.values(), key=lambda item: (_CATEGORY_ORDER.get(item.category, 99), -item.confidence))
    boosted = [
        candidate
        if hits.get(candidate.category, 0) < 3
        else _with_confidence(candidate, min(0.95, candidate.confidence + 0.05))
        for candidate in ordered
    ]
    return _select(boosted)


# Priority when a batch is dense: identity of the customer first, then the buying
# signal, then soft signals.  Stage is the weakest signal and is dropped first.
_CATEGORY_ORDER = {
    "crop": 0,
    "area": 1,
    "region": 2,
    "problem": 3,
    "product": 4,
    "preference": 5,
    "temporary": 6,
    "avoid": 7,
    "stage": 8,
}

# One category must not crowd out the others ("关注点" can trigger on every line).
MAX_PER_CATEGORY = 2


def _select(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    selected: list[MemoryCandidate] = []
    per_category: dict[str, int] = {}
    for candidate in candidates:
        if per_category.get(candidate.category, 0) >= MAX_PER_CATEGORY:
            continue
        per_category[candidate.category] = per_category.get(candidate.category, 0) + 1
        selected.append(candidate)
        if len(selected) >= MAX_MEMORIES_PER_BATCH:
            break
    return selected


def _candidates_for_message(text: str, message: Message, speaker: Speaker) -> list[MemoryCandidate]:
    source_id = getattr(message, "id", None)
    base = 0.8 if speaker == Speaker.OTHER else 0.65
    found: list[MemoryCandidate] = []

    for crop in _match_terms(text, _CROPS):
        found.append(_candidate(MemoryKind.FACT, f"作物：{crop}", base, "crop", text, source_id))

    for match in _AREA_RE.finditer(text):
        area = re.sub(r"\s+", "", match.group(0))
        found.append(_candidate(MemoryKind.FACT, f"面积：{area}", base, "area", text, source_id))

    region = _match_region(text)
    if region:
        found.append(_candidate(MemoryKind.FACT, f"地区：{region}", base, "region", text, source_id))

    for product in _match_terms(text, _PRODUCTS) + [m.group(1) for m in _GENERIC_PRODUCT_RE.finditer(text)]:
        found.append(_candidate(MemoryKind.FACT, f"产品：{product}", base, "product", text, source_id))

    problem = _match_problem(text)
    if problem:
        found.append(_candidate(MemoryKind.FACT, f"关注问题：{problem}", base, "problem", text, source_id))

    for stage in _match_terms(text, _STAGES):
        found.append(_candidate(MemoryKind.FACT, f"生育阶段：{stage}", base, "stage", text, source_id))

    for label, cues in _PREFERENCE_CUES:
        if any(cue in text for cue in cues):
            found.append(
                _candidate(MemoryKind.PREFERENCE, f"关注点：{label}", round(base - 0.05, 2), "preference", text, source_id)
            )

    if speaker == Speaker.ME:
        temporary = _match_temporary(text)
        if temporary:
            found.append(
                _candidate(
                    MemoryKind.TEMPORARY,
                    f"近期安排：{temporary}",
                    round(base - 0.05, 2),
                    "temporary",
                    text,
                    source_id,
                    expires_at=(datetime.now(timezone.utc) + timedelta(days=TEMPORARY_TTL_DAYS)).isoformat(),
                )
            )

    avoid = _AVOID_RE.search(text)
    if avoid and any(cue in avoid.group(1) for cue in _PROBLEM_CUES):
        found.append(
            _candidate(MemoryKind.AVOID, f"禁忌：{avoid.group(1)}", round(base - 0.1, 2), "avoid", text, source_id)
        )
    return found


def _match_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    """Longest-first so "龙牙百合" wins over "百合" and "大樱桃" over "樱桃"."""
    matched: list[str] = []
    consumed = text
    for term in sorted(terms, key=len, reverse=True):
        if term in consumed:
            matched.append(term)
            consumed = consumed.replace(term, "　")
    return matched


def _match_problem(text: str) -> str | None:
    diseases = _match_terms(text, _DISEASES)
    if diseases:
        return diseases[0]
    if any(cue in text for cue in _PROBLEM_CUES):
        pests = _match_terms(text, _PESTS)
        if pests:
            return pests[0]
    return None


def _match_region(text: str) -> str | None:
    for province in sorted(_PROVINCES, key=len, reverse=True):
        index = text.find(province)
        if index < 0:
            continue
        tail = text[index + len(province):]
        tail = re.sub(r"^(?:省|市|自治区|壮族自治区|维吾尔自治区|回族自治区)", "", tail)
        tail = re.match(r"[\u4e00-\u9fa5]{0,4}", tail).group(0)
        tail = _cut_at_stopword(tail)
        if len(tail) >= 2 and tail not in _REGION_TAIL_STOPWORDS:
            return f"{province}{tail}"
        return province
    return None


def _cut_at_stopword(tail: str) -> str:
    """Drop the trailing sentence fragment after a province.

    "黑龙江黑河种了 5000 亩大豆" must yield ``黑河``, not ``黑河种了``: the string
    after a province is only a place name until the first verb/particle.
    """
    for position, char in enumerate(tail):
        if char in _REGION_TAIL_STOP_CHARS:
            return tail[:position]
    return tail


def _match_temporary(text: str) -> str | None:
    if not any(cue in text for cue in _TEMPORARY_CUES):
        return None
    for cue in _TIME_CUES:
        index = text.find(cue)
        if index < 0:
            continue
        window = text[index: index + 14].strip()
        return window[:12] or cue
    return None


def _candidate(
    kind: MemoryKind,
    content: str,
    confidence: float,
    category: str,
    evidence: str,
    source_message_id: str | None,
    expires_at: str | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        kind=kind,
        content=content,
        confidence=round(min(0.95, max(0.3, confidence)), 2),
        category=category,
        evidence=evidence.strip()[:200],
        source_message_id=source_message_id,
        expires_at=expires_at,
    )


def _with_confidence(candidate: MemoryCandidate, confidence: float) -> MemoryCandidate:
    return MemoryCandidate(
        kind=candidate.kind,
        content=candidate.content,
        confidence=round(min(0.95, confidence), 2),
        category=candidate.category,
        evidence=candidate.evidence,
        source_message_id=candidate.source_message_id,
        expires_at=candidate.expires_at,
    )


def _usable(message: Message) -> bool:
    if getattr(message, "partial", False):
        return False
    if message.speaker not in _USABLE_SPEAKERS:
        return False
    if getattr(message, "content_type", "text") != "text":
        return False
    text = _text(message)[0]
    return len(text) >= 2 and not text.isdigit()


def _text(message: Message) -> tuple[str, Message, Speaker]:
    return " ".join((message.text or "").split()), message, message.speaker
