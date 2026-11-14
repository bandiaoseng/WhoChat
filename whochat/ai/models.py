from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Any

from whochat.core.models import Contact, Memory, Message, Strategy
from whochat.core.knowledge_models import ProductKnowledgeContext
from whochat.core.runtime import RuntimeState
from whochat.core.current_conversation import CurrentMessageView
from whochat.ai.current_work_authorization import CurrentWorkCloudAuthorization
from whochat.ai.photos import ReplyPhoto


@dataclass(frozen=True)
class ReplyMemoryContext:
    """Deterministic, compact customer-memory snapshot used by one reply request."""

    status: str
    account_scope_id: str
    contact_id: str
    conversation_id: str
    current_message_id: str
    profile: dict[str, Any] | None = None
    primary_context: dict[str, Any] | None = None
    candidate_contexts: tuple[dict[str, Any], ...] = ()
    context_resolution: str = "none"
    context_ambiguous: bool = False
    confirmed_facts: tuple[dict[str, Any], ...] = ()
    qualified_facts: tuple[dict[str, Any], ...] = ()
    relevant_events: tuple[dict[str, Any], ...] = ()
    pending_followups: tuple[dict[str, Any], ...] = ()
    error: str = ""
    current_view_revision: str = ""
    current_view_source_ref: str = ""
    current_view_source_kind: str = ""
    current_view_verified: bool = False
    # Locally verified historical customer statements, not operator-confirmed
    # profile facts. Only the DB-backed builder may attach this bounded input.
    observed_history: dict[str, Any] | None = None
    imported_history: dict[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "whochat.reply_memory.v1",
            "status": self.status,
            "identity": {
                "account_scope_id": self.account_scope_id,
                "contact_id": self.contact_id,
                "conversation_id": self.conversation_id,
                "current_message_id": self.current_message_id,
            },
            "profile": self.profile,
            "primary_context": self.primary_context,
            "candidate_contexts": list(self.candidate_contexts),
            "context_resolution": self.context_resolution,
            "context_ambiguous": self.context_ambiguous,
            "confirmed_facts": list(self.confirmed_facts),
            "qualified_facts": list(self.qualified_facts),
            "relevant_events": list(self.relevant_events),
            "pending_followups": list(self.pending_followups),
            "limits": {
                "primary_contexts": 1,
                "candidate_contexts": 3,
                "confirmed_facts": 20,
                "qualified_facts": 8,
                "relevant_events": 12,
                "pending_followups": 5,
                "recent_messages": 12,
            },
            "error": self.error,
        }
        if self.observed_history:
            payload["observed_history"] = self.observed_history
        if self.imported_history:
            payload["imported_history"] = self.imported_history
        if self.current_view_revision:
            payload["current_view"] = {
                "content_revision": self.current_view_revision,
                "source_ref": self.current_view_source_ref,
                "source_kind": self.current_view_source_kind,
            }
        return payload

    def business_payload(self) -> dict[str, Any]:
        payload = self.as_payload()
        if self.current_view_revision:
            payload["current_view"] = {
                "content_revision": self.current_view_revision,
                "source_kind": self.current_view_source_kind,
            }
        return payload

    @property
    def memory_hash(self) -> str:
        serialized = json.dumps(
            self.business_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def item_count(self) -> int:
        return (
            int(self.profile is not None)
            + int(self.primary_context is not None)
            + len(self.candidate_contexts)
            + len(self.confirmed_facts)
            + len(self.qualified_facts)
            + len(self.relevant_events)
            + len(self.pending_followups)
            + len((self.observed_history or {}).get("excerpts", ()))
            + len((self.imported_history or {}).get("excerpts", ()))
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReplyMemoryContext":
        if payload.get("schema") != "whochat.reply_memory.v1":
            raise ValueError("unsupported_reply_memory_schema")
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("reply_memory_identity_missing")
        current_view_payload = payload.get("current_view") or {}
        if not isinstance(current_view_payload, dict):
            raise ValueError("reply_memory_current_view_invalid")
        return cls(
            status=str(payload.get("status") or "empty"),
            account_scope_id=str(identity.get("account_scope_id") or ""),
            contact_id=str(identity.get("contact_id") or ""),
            conversation_id=str(identity.get("conversation_id") or ""),
            current_message_id=str(identity.get("current_message_id") or ""),
            profile=payload.get("profile") if isinstance(payload.get("profile"), dict) else None,
            primary_context=(
                payload.get("primary_context")
                if isinstance(payload.get("primary_context"), dict)
                else None
            ),
            candidate_contexts=tuple(
                item for item in payload.get("candidate_contexts", []) if isinstance(item, dict)
            ),
            context_resolution=str(payload.get("context_resolution") or "none"),
            context_ambiguous=bool(payload.get("context_ambiguous")),
            confirmed_facts=tuple(
                item for item in payload.get("confirmed_facts", []) if isinstance(item, dict)
            ),
            qualified_facts=tuple(
                item for item in payload.get("qualified_facts", []) if isinstance(item, dict)
            ),
            relevant_events=tuple(
                item for item in payload.get("relevant_events", []) if isinstance(item, dict)
            ),
            pending_followups=tuple(
                item for item in payload.get("pending_followups", []) if isinstance(item, dict)
            ),
            error=str(payload.get("error") or ""),
            current_view_revision=str(current_view_payload.get("content_revision") or ""),
            current_view_source_ref=str(current_view_payload.get("source_ref") or ""),
            current_view_source_kind=str(current_view_payload.get("source_kind") or ""),
            # A saved payload is not proof of a freshly validated observation.
            current_view_verified=False,
            # A serialized old result is not permission to revive historical
            # evidence. Fresh builder validation must reattach it.
            observed_history=None,
        )


@dataclass(frozen=True)
class ReplyContext:
    runtime: RuntimeState
    contact: Contact | None
    strategy: Strategy | None
    messages: list[Message]
    memories: list[Memory]
    memory_context: ReplyMemoryContext | None = None
    current_message_id: str = ""
    knowledge_context: ProductKnowledgeContext | None = None
    current_work_cloud_authorization: CurrentWorkCloudAuthorization | None = None
    current_message_view: CurrentMessageView | None = None
    current_work_mode: str = "reply"
    # One process-local attempt deadline, never business identity or consent.
    current_work_started_monotonic: float | None = None
    current_work_deadline_monotonic: float | None = None
    # Ephemeral pixels only; never serialize into memories, SQL or diagnostics.
    photos: tuple[ReplyPhoto, ...] = field(default=(), repr=False)
    photo_error: str = ""
    # One explicit click, not identity/consent or part of the business cache key.
    manual_retry_id: str = field(default="", repr=False)
    # Learning input is a request snapshot, NEVER a customer/source identity.
    writing_feedback: dict[str, Any] | None = field(default=None, repr=False)
    # Process-local frozen screenshot capability. A string alone grants nothing.
    frozen_screenshot_id: str = field(default="", repr=False)

    @property
    def effective_messages(self):
        return self.current_message_view.messages if self.current_message_view is not None else self.messages

    @property
    def effective_current_message(self):
        if self.current_message_view is not None:
            return self.current_message_view.current_message
        return next(
            (message for message in self.messages if message.id == self.current_message_id),
            self.messages[-1] if self.messages else None,
        )


def reply_memory_item_count(context: ReplyContext) -> int:
    structured = context.memory_context.item_count if context.memory_context is not None else 0
    return structured + len(context.memories)


@dataclass(frozen=True)
class ReplySuggestion:
    label: str
    text: str
    risk: str = "low"
    rationale: str = ""


@dataclass(frozen=True)
class IngredientAdvice:
    name: str
    role: str
    rationale: str
    applicable_conditions: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    knowledge_kind: str = ""


INGREDIENT_REPLY_POLICY_VERSION = "agro-reply-modes-v9-20261031"


@dataclass(frozen=True)
class ReplyGenerationResult:
    allowed: bool
    status: str
    suggestions: list[ReplySuggestion]
    provider: str
    customer_summary: str = ""
    intent: str = ""
    relevant_history: tuple[str, ...] = ()
    ask_first: str = ""
    sales_judgement: str = ""
    next_action: str = ""
    performance_trace: dict[str, Any] = field(default_factory=dict)
    policy_version: str = ""
    recommendation_mode: str = ""
    problem_assessment: str = ""
    ingredient_advice: tuple[IngredientAdvice, ...] = ()
    uncertainties: tuple[str, ...] = ()
    product_introduction_evidence: tuple[str, ...] = ()


def structured_reply_result_error(
    result: ReplyGenerationResult,
    *,
    require_current_policy: bool = True,
    context: ReplyContext | None = None,
) -> str | None:
    """Return a validation error for an incomplete cloud analysis result."""
    required_scalars = {
        "customer_summary": result.customer_summary,
        "intent": result.intent,
        "ask_first": result.ask_first,
        "sales_judgement": result.sales_judgement,
        "next_action": result.next_action,
    }
    missing = [name for name, value in required_scalars.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        return "missing_fields=" + ",".join(missing)
    if not result.relevant_history or any(not isinstance(item, str) or not item.strip() for item in result.relevant_history):
        return "missing_fields=relevant_history"
    if len(result.suggestions) != 2:
        return f"reply_suggestions_count={len(result.suggestions)}"
    if any(not isinstance(item.text, str) or not item.text.strip() for item in result.suggestions):
        return "empty_reply_suggestion"
    first, second = (_normalized_reply_text(item.text) for item in result.suggestions)
    if first == second:
        return "reply_suggestions_too_similar"
    if min(len(first), len(second)) >= 24 and SequenceMatcher(None, first, second).ratio() >= 0.94:
        return "reply_suggestions_too_similar"
    if context is not None and context.current_work_mode == "reactivation":
        # An identity-only current window never supplies a present problem or
        # an observed product introduction. Even legacy-result restoration
        # cannot turn this narrow opening task into treatment or purchase work.
        if result.recommendation_mode != "not_applicable":
            return "reactivation_requires_not_applicable"
        if result.ingredient_advice:
            return "reactivation_ingredient_advice_not_allowed"
        if result.product_introduction_evidence:
            return "reactivation_product_evidence_not_allowed"
    if not require_current_policy:
        return None
    if result.policy_version != INGREDIENT_REPLY_POLICY_VERSION:
        return "reply_policy_version_mismatch"
    if result.recommendation_mode not in {"ingredients", "clarify", "not_applicable"}:
        return "invalid_recommendation_mode"
    if not _nonempty_text(result.problem_assessment, 1200):
        return "missing_fields=problem_assessment"
    if not _text_tuple(result.uncertainties, minimum=0, maximum=8, text_limit=400):
        return "invalid_field=uncertainties"
    if not _text_tuple(result.product_introduction_evidence, minimum=0, maximum=2, text_limit=2000):
        return "invalid_field=product_introduction_evidence"
    if type(result.ingredient_advice) is not tuple or len(result.ingredient_advice) > 6:
        return "invalid_field=ingredient_advice"
    for item in result.ingredient_advice:
        if (
            type(item) is not IngredientAdvice
            or not _nonempty_text(item.name, 120)
            or not _nonempty_text(item.role, 600)
            or not _nonempty_text(item.rationale, 1000)
            or not _text_tuple(item.applicable_conditions, minimum=1, maximum=6, text_limit=400)
            or not _text_tuple(item.evidence_refs, minimum=1, maximum=6, text_limit=160)
            or item.knowledge_kind not in {"generic_category", "active_ingredient", "functional_ingredient"}
        ):
            return "invalid_field=ingredient_advice_item"
    if result.recommendation_mode == "ingredients":
        if not result.ingredient_advice:
            return "missing_fields=ingredient_advice"
    elif result.ingredient_advice:
        return "unexpected_ingredient_advice"
    if result.recommendation_mode == "clarify" and result.ask_first.strip() in {
        "无需追加追问", "无需追问", "无", "暂无", "不需要",
    }:
        return "clarify_requires_key_question"
    if result.recommendation_mode == "clarify" and not result.uncertainties:
        return "clarify_requires_uncertainty"
    has_package_offer = _has_positive_action(_PACKAGE_OFFER, _customer_action_text(result))
    if result.recommendation_mode != "not_applicable" and has_package_offer:
        return "package_before_product_selection"
    if _has_purchase_cta(_customer_action_text(result)) or has_package_offer:
        if result.recommendation_mode != "not_applicable":
            return "purchase_before_product_introduction"
        if not result.product_introduction_evidence:
            return "purchase_requires_product_introduction_evidence"
    if context is not None:
        error = ingredient_result_context_error(result, context)
        if error:
            return error
        from whochat.ai.agronomy_grounding import premise_error
        return premise_error(result, context)
    return None


def _nonempty_text(value: object, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _text_tuple(value: object, *, minimum: int, maximum: int, text_limit: int) -> bool:
    return (
        type(value) is tuple and minimum <= len(value) <= maximum
        and all(_nonempty_text(item, text_limit) for item in value)
    )


_PURCHASE_CTA = re.compile(
    r"(?:来|买|订|试|安排|发)(?:上|个|一|两|2|1)?(?:套|份|组)|下单|购买|订购|付款|打款"
)
_PACKAGE_OFFER = re.compile(r"(?:推荐|用|上|试|选|搭配).{0,12}(?:套餐|套装|品牌|SKU)|我们家的|我家的|我司的")
_DOSAGE_TOKEN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:毫升|毫克|千克|公斤|克|升|倍液|倍|袋|瓶|桶)(?:/亩)?|"
    r"(?:每|隔|间隔)\s*\d+\s*天|\d+\s*天\s*(?:一|1)次|\d+\s*次|"
    r"[一二三四五六七八九十半两]+(?:袋|瓶|桶|克|毫升|倍)"
)

UNREVIEWED_QUANTITY_MARKER = "【原数值保留在本地，待人工核对】"


def project_own_message_for_drafting(text: str) -> str:
    """Separate an observed outgoing quantity from permission to prescribe it.

    Projection is for the model's drafting view only. Original messages,
    fingerprints, evidence validation, persistence and customer input remain
    intact. Match exactly the quantity language the output contract forbids;
    never silently repair a returned dose or treat history as operator review.
    """
    return _DOSAGE_TOKEN.sub(UNREVIEWED_QUANTITY_MARKER, text)


def _customer_action_text(result: ReplyGenerationResult) -> str:
    return "\n".join([*(item.text for item in result.suggestions), result.next_action])


def _has_purchase_cta(text: str) -> bool:
    return _has_positive_action(_PURCHASE_CTA, text)


def _has_positive_action(pattern: re.Pattern[str], text: str) -> bool:
    for clause in re.split(r"[。！？!?；;\n]", text):
        for match in pattern.finditer(clause):
            prefix = clause[max(0, match.start() - 10):match.start()]
            # "要不要买一套" is an offer/question, not the negation "不要买".
            prefix = prefix.replace("要不要", "")
            if not re.search(r"不要|不应|不建议|无需|不需|暂不|不能|未决定|不直接|不推荐|不急于|不再|不$", prefix):
                return True
    return False


def eligible_ingredient_knowledge(context: ReplyContext) -> dict[str, dict]:
    """The same provenance boundary drives prompting and final validation."""
    knowledge = context.knowledge_context
    products = knowledge.products if knowledge is not None else ()
    return {
        item.get("product_id"): item for item in products
        if isinstance(item, dict)
        and item.get("status") == "active"
        and item.get("review_state") == "verified"
        and item.get("knowledge_kind") in {"generic_category", "active_ingredient", "functional_ingredient"}
        and isinstance(item.get("product_id"), str)
    }


def reply_mode_policy(context: ReplyContext) -> dict[str, Any]:
    """Request capability, not a diagnosis or a keyword-based intent router.

    Missing treatment inputs do not prohibit ordinary conversation. The model
    still interprets the complete dialogue; it may never invent a prescription
    to satisfy a schema. Counts/codes alone are safe for provider diagnostics.
    """
    knowledge = context.knowledge_context
    eligible = eligible_ingredient_knowledge(context)
    blockers = []
    if knowledge is None:
        blockers.append("knowledge_missing")
    else:
        if knowledge.requires_clarification:
            blockers.append("knowledge_requires_clarification")
        if not eligible:
            blockers.append("no_verified_generic_knowledge")
    if context.current_work_mode == "reactivation":
        blockers.append("reactivation_only")
        modes = ["not_applicable"]
    else:
        modes = ["clarify", "not_applicable"] if blockers else ["ingredients", "clarify", "not_applicable"]
    return {"allowed_recommendation_modes": modes,
            "ingredient_blockers": blockers, "eligible_knowledge_count": len(eligible)}


def ingredient_result_context_error(result: ReplyGenerationResult, context: ReplyContext) -> str | None:
    """Check provenance, not the truth of a diagnosis or agronomic efficacy.

    Only supplied active/verified generic knowledge can justify a named
    ingredient.  Free-form explanations still require human sales/technical
    acceptance; a matching ID is not proof that every claim is true.
    """
    if UNREVIEWED_QUANTITY_MARKER in json.dumps(asdict(result), ensure_ascii=False):
        return "unreviewed_quantity_marker_in_result"
    eligible = eligible_ingredient_knowledge(context)
    if result.recommendation_mode == "ingredients":
        if "ingredients" not in reply_mode_policy(context)["allowed_recommendation_modes"]:
            return "ingredient_knowledge_unavailable_or_requires_clarification"
        for advice in result.ingredient_advice:
            for reference in advice.evidence_refs:
                source = eligible.get(reference)
                if source is None:
                    return "ingredient_evidence_not_verified"
                if advice.knowledge_kind != source.get("knowledge_kind"):
                    return "ingredient_kind_not_in_evidence"
                names = {str(source.get("generic_name") or ""), str(source.get("product_name") or "")}
                active_ingredients = source.get("active_ingredients")
                if isinstance(active_ingredients, (list, tuple)):
                    names.update(str(item.get("name") or "") for item in active_ingredients if isinstance(item, dict))
                if advice.name.strip() not in {name.strip() for name in names if name.strip()}:
                    return "ingredient_name_not_in_evidence"
    action_text = _customer_action_text(result)
    if _has_purchase_cta(action_text) or _has_positive_action(_PACKAGE_OFFER, action_text):
        latest_own = next(
            (message for message in reversed(context.effective_messages) if message.speaker.value == "me"), None,
        )
        own_texts = {latest_own.text.strip()} if latest_own is not None else set()
        for evidence in result.product_introduction_evidence:
            # Literal own-message evidence is required; an old purchase, a
            # customer question, or a model-written quotation cannot qualify.
            if (
                evidence.strip() not in own_texts
                or not re.search(r"产品|套餐|这款|这套", evidence)
                or re.search(r"[?？]|还没|尚未|没有|没推荐|不推荐|是否|要不要|考虑|准备|打算|稍后", evidence)
            ):
                return "product_introduction_evidence_not_observed"
    recommendation_text = "\n".join([
        action_text, result.ask_first,
        *(text for item in result.ingredient_advice for text in (item.role, item.rationale, *item.applicable_conditions)),
    ])
    tokens = _DOSAGE_TOKEN.findall(recommendation_text)
    if tokens:
        # The first ingredient-first contract does not contain a reviewed
        # dosage/mixture operation.  A quantity copied from unrelated context
        # must not sneak into free-form customer text as a prescription.
        return "dosage_requires_explicit_operator_review"
    return None


def _normalized_reply_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def reply_generation_result_from_json(
    payload: str,
    *,
    require_current_policy: bool = True,
    context: ReplyContext | None = None,
) -> tuple[ReplyGenerationResult | None, str | None]:
    """Strictly restore a completed structured result from durable event JSON.

    The durable payload format is the existing ``asdict(ReplyGenerationResult)``
    representation.  Recovery is intentionally stricter than a normal display
    fallback: incomplete data or a reply count other than two must never be
    presented as a completed sales analysis.
    """
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "result_not_object"
    if data.get("allowed") is not True:
        return None, "result_not_allowed"

    status = data.get("status")
    provider = data.get("provider")
    if not isinstance(status, str) or not status.strip():
        return None, "missing_fields=status"
    if not isinstance(provider, str) or not provider.strip():
        return None, "missing_fields=provider"

    scalar_names = ("customer_summary", "intent", "ask_first", "sales_judgement", "next_action")
    scalar_values: dict[str, str] = {}
    for name in scalar_names:
        value = data.get(name)
        if not isinstance(value, str):
            return None, f"invalid_field={name}"
        scalar_values[name] = value

    history = data.get("relevant_history")
    if not isinstance(history, (list, tuple)) or any(not isinstance(item, str) for item in history):
        return None, "invalid_field=relevant_history"

    raw_suggestions = data.get("suggestions")
    if not isinstance(raw_suggestions, list):
        return None, "invalid_field=suggestions"
    suggestions: list[ReplySuggestion] = []
    for index, item in enumerate(raw_suggestions):
        if not isinstance(item, dict):
            return None, f"invalid_suggestion={index}"
        label = item.get("label")
        text = item.get("text")
        risk = item.get("risk", "low")
        rationale = item.get("rationale", "")
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(text, str)
            or not isinstance(risk, str)
            or not isinstance(rationale, str)
        ):
            return None, f"invalid_suggestion={index}"
        suggestions.append(ReplySuggestion(label=label, text=text, risk=risk, rationale=rationale))

    policy_version = data.get("policy_version", "")
    recommendation_mode = data.get("recommendation_mode", "")
    problem_assessment = data.get("problem_assessment", "")
    if any(not isinstance(item, str) for item in (policy_version, recommendation_mode, problem_assessment)):
        return None, "invalid_ingredient_scalar"
    raw_advice = data.get("ingredient_advice", [])
    if not isinstance(raw_advice, (list, tuple)) or len(raw_advice) > 6:
        return None, "invalid_field=ingredient_advice"
    advice: list[IngredientAdvice] = []
    for item in raw_advice:
        if not isinstance(item, dict):
            return None, "invalid_field=ingredient_advice_item"
        name, role, rationale = item.get("name"), item.get("role"), item.get("rationale")
        conditions, refs = item.get("applicable_conditions"), item.get("evidence_refs")
        if (
            any(not isinstance(value, str) for value in (name, role, rationale))
            or not isinstance(conditions, (list, tuple))
            or not isinstance(refs, (list, tuple))
            or any(not isinstance(value, str) for value in (*conditions, *refs))
        ):
            return None, "invalid_field=ingredient_advice_item"
        kind = item.get("knowledge_kind", "")
        if not isinstance(kind, str):
            return None, "invalid_field=ingredient_advice_kind"
        advice.append(IngredientAdvice(name, role, rationale, tuple(conditions), tuple(refs), kind))
    uncertainties = data.get("uncertainties", [])
    introduction = data.get("product_introduction_evidence", [])
    if (
        not isinstance(uncertainties, (list, tuple))
        or not isinstance(introduction, (list, tuple))
        or any(not isinstance(value, str) for value in (*uncertainties, *introduction))
    ):
        return None, "invalid_ingredient_list"

    result = ReplyGenerationResult(
        allowed=True,
        status=status,
        suggestions=suggestions,
        provider=provider,
        customer_summary=scalar_values["customer_summary"],
        intent=scalar_values["intent"],
        relevant_history=tuple(history),
        ask_first=scalar_values["ask_first"],
        sales_judgement=scalar_values["sales_judgement"],
        next_action=scalar_values["next_action"],
        performance_trace=data.get("performance_trace") if isinstance(data.get("performance_trace"), dict) else {},
        policy_version=policy_version,
        recommendation_mode=recommendation_mode,
        problem_assessment=problem_assessment,
        ingredient_advice=tuple(advice),
        uncertainties=tuple(uncertainties),
        product_introduction_evidence=tuple(introduction),
    )
    error = structured_reply_result_error(result, require_current_policy=require_current_policy, context=context)
    return (None, error) if error else (result, None)


@dataclass(frozen=True)
class AIConnectionTestResult:
    ok: bool
    status: str
    provider: str
    detail: str
    elapsed_ms: int = 0
