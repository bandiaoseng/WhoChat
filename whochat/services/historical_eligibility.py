from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

from whochat.services.historical_backfill import BackfillChunkPolicy


VALUE_TAGS = frozenset({"A", "B"})
SMALL_CUSTOMER_TAGS = frozenset({"C", "Z"})
ATTRIBUTE_TAGS = frozenset({"H", "X", "D"}) | SMALL_CUSTOMER_TAGS
WORTHLESS_TAGS = frozenset({"S", "0"})
ALL_SINGLE_TAGS = VALUE_TAGS | ATTRIBUTE_TAGS | WORTHLESS_TAGS
_SEPARATORS = re.compile(r"[\s\-_\/|+＋,，、;；:：()（）\[\]【】{}<>《》.!！?？·]+")
_COMPACT_PREFIX = re.compile(r"^([ABCHXSZD]{1,3})(?=[\u3400-\u9fff\d]|$)")
_SUSPICIOUS_PREFIX = re.compile(r"^[ABCHXSZD]{4,}(?=[\u3400-\u9fff\d]|$)")
_EMBEDDED_COMPACT_TAGS = re.compile(r"(?<![A-Z])([ABCHXSZD]{1,3})(?=[\u3400-\u9fff\d])")
_NUMBER_RUN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")


@dataclass(frozen=True)
class RemarkTagParseResult:
    value_tier: str | None
    has_transaction_history_tag: bool
    info_unknown_tag: bool
    is_worthless: bool
    worthless_reason: str | None
    tag_parse_status: str
    raw_remark_hash: str
    parsed_tags: tuple[str, ...]

    @property
    def is_small_customer(self) -> bool:
        return bool(SMALL_CUSTOMER_TAGS.intersection(self.parsed_tags))

    @property
    def is_dealer(self) -> bool:
        return "D" in self.parsed_tags


class CustomerRemarkTagParser:
    """Parse the user's explicit remark-prefix convention without substring matching.

    The four production accounts use both delimited tags (``A-H-...``) and a
    compact leading tag region (``AH10-...`` or ``SC...``).  A one-character
    compact tag run embedded between Chinese text and a following Chinese
    character or number is also explicit (for example ``南瓜Ah江西`` and
    ``A苹果S20亩``).  The negative Latin boundary keeps normal words such as
    ``Sam``, ``Apple`` and ``David`` from becoming tags.
    ``无价值`` is an explicit, unambiguous business phrase and may appear as an
    appended suffix in real remarks, so it is recognised wherever it occurs.

    The numeric worthless marker is recognised from complete numeric runs.
    Only a run whose complete normalized value is exactly ``0`` is a tag.
    Consequently ``柑橘树0黄叶`` is worthless, while ``20亩``, phone numbers,
    and ``0.5亩`` are not.
    """

    def parse(self, remark: str | None, *, explicit_labels: Iterable[str] = ()) -> RemarkTagParseResult:
        raw = remark or ""
        normalized = unicodedata.normalize("NFKC", raw).strip()
        upper = normalized.upper()
        tags: list[str] = []
        ambiguous = False

        def add(tag: str) -> None:
            if tag in ALL_SINGLE_TAGS and tag not in tags:
                tags.append(tag)

        # Explicit labels supplied by a future adapter are accepted only as
        # exact one-token labels.  No association is guessed from label names.
        for label in explicit_labels:
            token = unicodedata.normalize("NFKC", str(label)).strip().upper()
            if token in ALL_SINGLE_TAGS:
                add(token)
            elif token == "无价值" and "无价值" not in tags:
                tags.append("无价值")

        if "无价值" in normalized and "无价值" not in tags:
            tags.append("无价值")

        if upper:
            # Separator-bounded tokens may occur after a person's name.
            tokens = [item for item in _SEPARATORS.split(upper) if item]
            for token in tokens:
                if token in ALL_SINGLE_TAGS and token != "0":
                    add(token)

            # Parse only the leading, explicit tag area for compact prefixes.
            leading = upper.lstrip(" -_/|+＋,，、;；:：()（）[]【】{}<>《》.!！?？·")
            compact = _COMPACT_PREFIX.match(leading)
            if compact:
                for tag in compact.group(1):
                    add(tag)
            if _SUSPICIOUS_PREFIX.match(leading):
                ambiguous = True

            # A sequence of delimited leading tag tokens (A-H, B / X, etc.).
            for token in tokens[:4]:
                if token in ALL_SINGLE_TAGS and token != "0":
                    add(token)
                    continue
                if re.fullmatch(r"[ABCHXSZD]{1,3}", token):
                    for tag in token:
                        add(tag)
                    continue
                attached = _COMPACT_PREFIX.match(token)
                if attached:
                    for tag in attached.group(1):
                        add(tag)
                break

            # Explicit compact tags may occur after crop/name text in actual
            # WeChat remarks (``红薯/南瓜Ah江西...``).  Parse the whole bounded
            # run through the same canonical tag set; never substring-match a
            # Latin word.
            for match in _EMBEDDED_COMPACT_TAGS.finditer(upper):
                for tag in match.group(1):
                    add(tag)

            # Numeric parsing is deliberately independent from separators.
            # The full run must be exactly zero; digits inside 20/100/phone
            # numbers and the leading zero in 0.5 are not worthless markers.
            if any(match.group(0) == "0" for match in _NUMBER_RUN.finditer(upper)):
                add("0")

        tiers = sorted(VALUE_TAGS.intersection(tags))
        reasons = [item for item in ("S", "0", "无价值") if item in tags]
        is_worthless = bool(reasons)
        if len(tiers) > 1:
            status = "conflicting"
            value_tier = None
        elif ambiguous:
            status = "ambiguous"
            value_tier = tiers[0] if tiers else None
        elif tags:
            status = "parsed"
            value_tier = tiers[0] if tiers else None
        else:
            status = "none"
            value_tier = None
        return RemarkTagParseResult(
            value_tier=value_tier,
            has_transaction_history_tag="H" in tags,
            info_unknown_tag="X" in tags,
            is_worthless=is_worthless,
            worthless_reason=reasons[0] if reasons else None,
            tag_parse_status=status,
            raw_remark_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            parsed_tags=tuple(tags),
        )


@dataclass(frozen=True)
class HistoricalBackfillEligibility:
    eligible_for_backfill: bool
    skip_reason: str | None
    value_tier: str | None
    has_transaction_history: bool
    info_unknown: bool
    is_worthless: bool
    worthless_reason: str | None
    tag_parse_status: str
    raw_remark_hash: str
    parsed_tags: tuple[str, ...]
    history_message_count: int
    history_start_at: str | None
    history_end_at: str | None
    estimated_business_messages: int
    estimated_characters: int
    estimated_tokens: int
    estimated_chunks: int
    priority_features: tuple[str, ...]


class HistoricalBackfillEligibilityService:
    def __init__(self, parser: CustomerRemarkTagParser | None = None) -> None:
        self.parser = parser or CustomerRemarkTagParser()

    def evaluate(
        self,
        *,
        remark: str | None,
        history_message_count: int,
        history_start_at: str | None,
        history_end_at: str | None,
        estimated_business_messages: int = 0,
        estimated_characters: int = 0,
        estimated_chunks: int = 0,
        explicit_labels: Iterable[str] = (),
        has_existing_memory: bool = False,
    ) -> HistoricalBackfillEligibility:
        parsed = self.parser.parse(remark, explicit_labels=explicit_labels)
        features: list[str] = []
        if parsed.value_tier:
            features.append(f"tier_{parsed.value_tier.lower()}")
        if parsed.has_transaction_history_tag:
            features.append("transaction_history")
        if parsed.info_unknown_tag:
            features.append("information_unknown")
        if parsed.is_small_customer:
            features.append("small_customer")
        if parsed.is_dealer:
            features.append("dealer")
        if not parsed.parsed_tags:
            features.append("untagged")
        if history_message_count == 0:
            features.append("no_history")
        elif history_message_count <= 5:
            features.append("low_history")
        if has_existing_memory:
            features.append("existing_memory")
        return HistoricalBackfillEligibility(
            eligible_for_backfill=not parsed.is_worthless,
            skip_reason=(f"explicit_worthless_tag:{parsed.worthless_reason}" if parsed.is_worthless else None),
            value_tier=parsed.value_tier,
            has_transaction_history=parsed.has_transaction_history_tag,
            info_unknown=parsed.info_unknown_tag,
            is_worthless=parsed.is_worthless,
            worthless_reason=parsed.worthless_reason,
            tag_parse_status=parsed.tag_parse_status,
            raw_remark_hash=parsed.raw_remark_hash,
            parsed_tags=parsed.parsed_tags,
            history_message_count=max(0, int(history_message_count)),
            history_start_at=history_start_at,
            history_end_at=history_end_at,
            estimated_business_messages=max(0, int(estimated_business_messages)),
            estimated_characters=max(0, int(estimated_characters)),
            estimated_tokens=estimate_tokens(estimated_characters, estimated_business_messages),
            estimated_chunks=max(0, int(estimated_chunks)),
            priority_features=tuple(features),
        )


def estimate_tokens(characters: int, messages: int) -> int:
    """Conservative local estimate only; it is never used for provider billing."""
    return int(math.ceil(max(0, characters) / 2.0) + max(0, messages) * 12)


@dataclass
class ChunkEstimator:
    policy: BackfillChunkPolicy = field(
        default_factory=lambda: BackfillChunkPolicy(
            max_messages=75, max_characters=10_000, break_after_days=90
        )
    )
    effective_messages: int = 0
    characters: int = 0
    chunks: int = 0
    _bucket_messages: int = 0
    _bucket_characters: int = 0
    _previous_at: datetime | None = None

    def add(self, *, text: str, sent_at: str) -> None:
        self.add_size(character_count=len(text), sent_at=sent_at)

    def add_size(self, *, character_count: int, sent_at: str) -> None:
        current_at = _parse_time(sent_at)
        gap_break = bool(
            self._bucket_messages and self._previous_at and current_at
            and abs((current_at - self._previous_at).total_seconds())
            > self.policy.break_after_days * 86400
        )
        size_break = bool(
            self._bucket_messages and (
                self._bucket_messages >= self.policy.max_messages
                or self._bucket_characters + character_count > self.policy.max_characters
            )
        )
        if gap_break or size_break:
            self._flush()
        self._bucket_messages += 1
        self._bucket_characters += character_count
        self.effective_messages += 1
        self.characters += character_count
        self._previous_at = current_at or self._previous_at

    def finish(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._bucket_messages:
            return
        self.chunks += 1
        self._bucket_messages = 0
        self._bucket_characters = 0


def is_effective_business_message(*, sender: str, message_type: str, text: str) -> bool:
    """Exact local prefilter used by the Pilot, expressed without DB row objects."""
    if sender not in {"other", "me"} or message_type != "text":
        return False
    compact = re.sub(r"\s+", "", text)
    if not compact or compact in {"[图片]", "[表情]", "[语音]", "图片", "表情", "收到", "好的", "嗯", "哦"}:
        return False
    if re.fullmatch(r"[\W_]+", compact, flags=re.UNICODE):
        return False
    return True


@dataclass(frozen=True)
class HistoricalContactPlan:
    account_scope_id: str
    safe_contact_reference: str
    display_name: str
    eligibility: HistoricalBackfillEligibility
    estimated_calls_min: int
    estimated_calls_expected: float
    estimated_calls_max: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(payload.pop("eligibility"))
        payload["backfill_eligible"] = payload.pop("eligible_for_backfill")
        return payload


@dataclass(frozen=True)
class BackfillBatchLimits:
    max_customers: int
    max_chunks: int
    max_provider_calls: int
    max_runtime_seconds: int


@dataclass
class BackfillBatchCheckpoint:
    queued_references: list[str]
    completed_references: list[str]
    consumed_chunks: int = 0
    consumed_provider_calls: int = 0
    elapsed_seconds: float = 0.0
    paused: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BatchBackfillScheduler:
    """Offline/Fake scheduler model; it never invokes a provider."""

    def __init__(self, plans: Iterable[HistoricalContactPlan], limits: BackfillBatchLimits) -> None:
        self.limits = limits
        eligible = [
            item for item in plans
            if item.eligibility.eligible_for_backfill and not item.eligibility.is_worthless
            and item.eligibility.estimated_chunks > 0
        ]
        self._plans = {item.safe_contact_reference: item for item in eligible}
        self.checkpoint = BackfillBatchCheckpoint(
            queued_references=[item.safe_contact_reference for item in eligible[: limits.max_customers]],
            completed_references=[],
        )

    def pause(self) -> None:
        self.checkpoint.paused = True

    def resume(self) -> None:
        self.checkpoint.paused = False

    def claim_next(self, *, realtime_reply_pending: bool = False) -> HistoricalContactPlan | None:
        if self.checkpoint.paused:
            self.checkpoint.stop_reason = "paused"
            return None
        if realtime_reply_pending:
            self.checkpoint.stop_reason = "yield_to_realtime_reply"
            return None
        if self.checkpoint.elapsed_seconds >= self.limits.max_runtime_seconds:
            self.checkpoint.stop_reason = "max_runtime"
            return None
        if not self.checkpoint.queued_references:
            self.checkpoint.stop_reason = "queue_empty"
            return None
        reference = self.checkpoint.queued_references[0]
        plan = self._plans[reference]
        next_chunks = self.checkpoint.consumed_chunks + plan.eligibility.estimated_chunks
        next_calls = self.checkpoint.consumed_provider_calls + plan.estimated_calls_max
        if next_chunks > self.limits.max_chunks:
            self.checkpoint.stop_reason = "max_chunks"
            return None
        if next_calls > self.limits.max_provider_calls:
            self.checkpoint.stop_reason = "max_provider_calls"
            return None
        self.checkpoint.queued_references.pop(0)
        self.checkpoint.stop_reason = None
        return plan

    def complete(self, plan: HistoricalContactPlan, *, chunks: int, provider_calls: int, elapsed: float) -> None:
        self.checkpoint.completed_references.append(plan.safe_contact_reference)
        self.checkpoint.consumed_chunks += max(0, int(chunks))
        self.checkpoint.consumed_provider_calls += max(0, int(provider_calls))
        self.checkpoint.elapsed_seconds += max(0.0, float(elapsed))

    @classmethod
    def restore(
        cls,
        plans: Iterable[HistoricalContactPlan],
        limits: BackfillBatchLimits,
        payload: dict[str, Any],
    ) -> "BatchBackfillScheduler":
        scheduler = cls(plans, limits)
        scheduler.checkpoint = BackfillBatchCheckpoint(**payload)
        return scheduler


@dataclass(frozen=True)
class HistoricalActivityMetrics:
    last_customer_message_at: str | None
    last_any_message_at: str | None
    recent_message_count_30d: int
    recent_message_count_90d: int
    recent_message_count_180d: int
    recent_customer_message_count_30d: int = 0
    recent_customer_message_count_90d: int = 0
    recent_customer_message_count_180d: int = 0


@dataclass(frozen=True)
class HistoricalPriorityResult:
    eligible: bool
    score: int
    recency_band: str
    components: tuple[str, ...]
    excluded_reason: str | None = None


class HistoricalBackfillPriorityScorer:
    """Transparent, non-persistent scheduling score for production preflight.

    This score is deliberately not a customer fact and must never be written
    to Customer Profile.  Its only purpose is ordering a bounded batch.  The
    explicit worthless rule always runs before every other component.
    """

    def score(
        self,
        *,
        eligibility: HistoricalBackfillEligibility,
        activity: HistoricalActivityMetrics,
        historical_backfill_completed: bool,
    ) -> HistoricalPriorityResult:
        if eligibility.is_worthless or not eligibility.eligible_for_backfill:
            return HistoricalPriorityResult(
                eligible=False, score=-10_000, recency_band="excluded",
                components=("explicit_worthless",), excluded_reason="explicit_worthless",
            )
        if historical_backfill_completed:
            return HistoricalPriorityResult(
                eligible=False, score=-9_000, recency_band="completed",
                components=("historical_backfill_completed",),
                excluded_reason="historical_backfill_completed",
            )

        total = 0
        components: list[str] = []
        if activity.recent_customer_message_count_30d:
            total += 45
            recency_band = "active_30d"
            components.append("customer_active_30d:+45")
        elif activity.recent_customer_message_count_90d:
            total += 34
            recency_band = "active_90d"
            components.append("customer_active_90d:+34")
        elif activity.recent_customer_message_count_180d:
            total += 23
            recency_band = "active_180d"
            components.append("customer_active_180d:+23")
        elif activity.last_customer_message_at:
            total += 2
            recency_band = "inactive_over_180d"
            components.append("customer_inactive_over_180d:+2")
        else:
            recency_band = "no_customer_message"
            components.append("no_customer_message:+0")

        # A/B are the only explicit value tiers and form the first scheduling
        # layer.  Their base scores stay above the maximum possible non-A/B
        # score so recency/H/history can order customers inside each tier but
        # cannot silently redefine eligibility or turn attributes into value.
        tier_points = {"A": 200, "B": 160}.get(eligibility.value_tier, 0)
        total += tier_points
        if eligibility.value_tier:
            components.append(f"tier_{eligibility.value_tier}:+{tier_points}")
        if eligibility.has_transaction_history:
            total += 22
            components.append("transaction_history:+22")
        if eligibility.info_unknown:
            components.append("information_unknown:+0")
        if "small_customer" in eligibility.priority_features:
            components.append("small_customer_attribute:+0")
        if "dealer" in eligibility.priority_features:
            components.append("dealer_attribute:+0")
        if not eligibility.parsed_tags:
            components.append("untagged:+0")

        recent_volume = min(12, activity.recent_message_count_30d * 2)
        if recent_volume:
            total += recent_volume
            components.append(f"recent_volume:+{recent_volume}")
        business_volume = (
            9 if eligibility.estimated_business_messages >= 100 else
            6 if eligibility.estimated_business_messages >= 30 else
            3 if eligibility.estimated_business_messages >= 5 else 0
        )
        total += business_volume
        if business_volume:
            components.append(f"business_history:+{business_volume}")
        if "existing_memory" in eligibility.priority_features:
            total += 3
            components.append("existing_memory:+3")
        chunks = eligibility.estimated_chunks
        if 1 <= chunks <= 8:
            total += 10
            components.append("production_batch_fit:+10")
        elif chunks > 8:
            total -= min(30, chunks - 8)
            components.append(f"oversized_history:-{min(30, chunks - 8)}")
        return HistoricalPriorityResult(
            eligible=True, score=total, recency_band=recency_band,
            components=tuple(components), excluded_reason=None,
        )


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
