from __future__ import annotations

from typing import Any, Iterable

from whochat.ai.models import ReplyMemoryContext
from whochat.core.knowledge_models import KnowledgeKind, Product, ProductKnowledgeContext, Solution
from whochat.storage.product_knowledge import ProductKnowledgeRepository, normalize_knowledge_value


GENERIC_KNOWLEDGE_KINDS = (
    KnowledgeKind.GENERIC_CATEGORY,
    KnowledgeKind.ACTIVE_INGREDIENT,
    KnowledgeKind.FUNCTIONAL_INGREDIENT,
)


# Customers rarely repeat the short controlled vocabulary used by the
# knowledge records verbatim.  Keep this list deliberately narrow and map
# only expressions that preserve the same agronomic meaning.  It is used for
# retrieval, while the original message remains untouched in the prompt and
# evidence trail.
PROBLEM_MESSAGE_NORMALIZATIONS: tuple[tuple[str, str], ...] = (
    ("缓苗比较慢", "缓苗慢"),
    ("缓苗很慢", "缓苗慢"),
    ("缓苗特别慢", "缓苗慢"),
    ("缓苗不太好", "缓苗慢"),
    ("缓苗不好", "缓苗慢"),
    ("不怎么发新根", "不发新根"),
    ("没怎么发新根", "不发新根"),
    ("没有发新根", "不发新根"),
    ("没发新根", "不发新根"),
    ("不长新根", "不发新根"),
    ("返盐", "土壤盐害"),
    ("返碱", "土壤盐害"),
)


class ProductKnowledgeContextBuilder:
    """Build a bounded, verified generic-agronomy context for one reply."""

    def __init__(
        self,
        repository: ProductKnowledgeRepository,
        *,
        product_limit: int = 8,
        solution_limit: int = 5,
        allowed_kinds: Iterable[str | KnowledgeKind] | None = None,
    ) -> None:
        self.repository = repository
        self.product_limit = max(1, min(int(product_limit), 8))
        self.solution_limit = max(1, min(int(solution_limit), 5))
        self.allowed_kinds = tuple(allowed_kinds) if allowed_kinds is not None else tuple(KnowledgeKind)

    def build(
        self,
        *,
        memory_context: ReplyMemoryContext | dict[str, Any] | None = None,
        current_message: str = "",
        growing_context: dict[str, Any] | None = None,
        condition_facts: dict[str, Any] | None = None,
    ) -> ProductKnowledgeContext:
        context = growing_context or _primary_context(memory_context)
        repository = self.repository.reply_read_snapshot(knowledge_kinds=self.allowed_kinds)
        products = repository.list_active_products(
            limit=1000, knowledge_kinds=self.allowed_kinds
        )
        # The customer's current explicit statement outranks older profile
        # context. This also prevents an old apple context from suppressing a
        # current, explicit strawberry or tomato question.
        crop = _detect_crop(current_message, products) or _text(context.get("crop"))
        stage = _detect_stage(current_message, products) or _text(context.get("growth_stage"))
        region = _text(context.get("region"))
        problem = _detect_problem(
            current_message, products, repository, self.allowed_kinds
        )
        unmatched_problem_signal = bool(_has_problem_signal(current_message) and not problem)

        ranked_products: list[tuple[int, Product]] = []
        for product in products:
            score = _product_score(
                product,
                current_message,
                crop=crop,
                problem=problem,
                stage=stage,
                unmatched_problem_signal=unmatched_problem_signal,
            )
            if score is not None:
                ranked_products.append((score, product))
        ranked_products.sort(
            key=lambda item: (-item[0], normalize_knowledge_value(item[1].product_name))
        )

        selected_solutions: list[Solution] = []
        if any((crop, problem, stage)) and not unmatched_problem_signal:
            for _score, solution in repository.rank_matching_solutions(
                crop=crop,
                problem=problem,
                stage=stage,
                region=region,
                condition_facts=condition_facts,
                limit=100,
            ):
                linked = [repository.get_product(item.product_id) for item in solution.items]
                if linked and all(_eligible(product, self.allowed_kinds) for product in linked):
                    selected_solutions.append(solution)
                if len(selected_solutions) >= self.solution_limit:
                    break

        selected_products: list[Product] = []
        seen_products: set[str] = set()
        for _score, product in ranked_products:
            if product.product_id not in seen_products:
                selected_products.append(product)
                seen_products.add(product.product_id)
            if len(selected_products) >= self.product_limit:
                break
        for solution in selected_solutions:
            if len(selected_products) >= self.product_limit:
                break
            for item in solution.items:
                if len(selected_products) >= self.product_limit:
                    break
                product = repository.get_product(item.product_id)
                if _eligible(product, self.allowed_kinds) and product.product_id not in seen_products:
                    selected_products.append(product)
                    seen_products.add(product.product_id)

        precautions: list[str] = []
        seen_precautions: set[str] = set()
        for value in (
            *[
                item
                for product in selected_products
                for item in (*product.precautions, *product.incompatibilities, *product.risk_notes)
            ],
            *[
                item
                for solution in selected_solutions
                for item in (*solution.precautions, *solution.risk_notes)
            ],
        ):
            key = normalize_knowledge_value(value)
            if value and key not in seen_precautions:
                seen_precautions.add(key)
                precautions.append(value)
            if len(precautions) >= 20:
                break

        questions = _clarification_questions(
            current_message=current_message, crop=crop, stage=stage, problem=problem
        )
        # A matching direction does not erase missing agronomic facts.  For
        # example, a thrips ingredient may match while crop/registration and
        # growth stage are still unknown; the reply must ask before prescribing.
        requires_clarification = bool(questions)
        return ProductKnowledgeContext(
            products=tuple(_product_payload(product) for product in selected_products),
            solutions=tuple(
                _solution_payload(solution, repository) for solution in selected_solutions
            ),
            precautions=tuple(precautions),
            crop=crop,
            problem=problem,
            stage=stage,
            region=region,
            status="ready" if selected_products or selected_solutions else "empty",
            requires_clarification=requires_clarification,
            clarification_questions=tuple(questions),
        )


def _eligible(product: Product | None, allowed: Iterable[str | KnowledgeKind]) -> bool:
    if product is None:
        return False
    allowed_values = {str(item) for item in allowed}
    return (
        product.status.value == "active"
        and product.review_state.value == "verified"
        and product.knowledge_kind.value in allowed_values
    )


def _primary_context(memory_context: ReplyMemoryContext | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(memory_context, ReplyMemoryContext):
        return memory_context.primary_context or {}
    if isinstance(memory_context, dict):
        value = memory_context.get("primary_context")
        return value if isinstance(value, dict) else memory_context
    return {}


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _contains(value: str, options: tuple[str, ...]) -> bool:
    key = normalize_knowledge_value(value)
    return any(normalize_knowledge_value(option) == key for option in options)


def _detect_crop(message: str, products: list[Product]) -> str:
    normalized = normalize_knowledge_value(message)
    terms = sorted(
        {value for product in products for value in product.applicable_crops},
        key=len,
        reverse=True,
    )
    return next((value for value in terms if normalize_knowledge_value(value) in normalized), "")


def _detect_stage(message: str, products: list[Product]) -> str:
    normalized = normalize_knowledge_value(message)
    terms = sorted(
        {value for product in products for value in product.applicable_stages},
        key=len,
        reverse=True,
    )
    for value in terms:
        key = normalize_knowledge_value(value)
        aliases = (key, key[:-1]) if key.endswith("期") and len(key) > 2 else (key,)
        if any(alias and alias in normalized for alias in aliases):
            return value
    return ""


def _detect_problem(
    message: str,
    products: list[Product],
    repository: ProductKnowledgeRepository,
    allowed_kinds: Iterable[str | KnowledgeKind],
) -> str:
    normalized = _normalize_problem_message(message)
    if not normalized:
        return ""
    terms = {value for product in products for value in product.applicable_problems}
    for _score, solution in repository.rank_matching_solutions(limit=500):
        if solution.items and all(
            _eligible(repository.get_product(item.product_id), allowed_kinds)
            for item in solution.items
        ):
            terms.update(solution.target_problems)
    return next(
        (
            value
            for value in sorted(terms, key=len, reverse=True)
            if normalize_knowledge_value(value) in normalized
        ),
        "",
    )


def _normalize_problem_message(message: str) -> str:
    normalized = normalize_knowledge_value(message)
    for source, target in PROBLEM_MESSAGE_NORMALIZATIONS:
        normalized = normalized.replace(
            normalize_knowledge_value(source),
            normalize_knowledge_value(target),
        )
    return normalized


def _product_score(
    product: Product,
    message: str,
    *,
    crop: str,
    problem: str,
    stage: str,
    unmatched_problem_signal: bool = False,
) -> int | None:
    # A crop/stage coincidence is not evidence for treating a clearly stated
    # but unsupported problem.  In that case the safe behavior is to ask and
    # record a knowledge gap, not to surface an unrelated fertilizer direction.
    if unmatched_problem_signal:
        return None
    score = 0
    for query, values, points in (
        (crop, product.applicable_crops, 30),
        (problem, product.applicable_problems, 40),
        (stage, product.applicable_stages, 20),
    ):
        if query and values:
            if not _contains(query, values):
                return None
            score += points
    normalized_message = normalize_knowledge_value(message)
    if normalize_knowledge_value(product.product_name) in normalized_message:
        score += 100
    elif any(normalize_knowledge_value(alias) in normalized_message for alias in product.aliases):
        score += 90
    if score == 0:
        return None
    return score


def _has_problem_signal(message: str) -> bool:
    normalized = normalize_knowledge_value(message)
    return any(
        normalize_knowledge_value(term) in normalized
        for term in (
            "病害", "炭疽", "灰霉", "白粉", "软腐", "疫病", "叶斑",
            "虫害", "蓟马", "红蜘蛛", "白粉虱", "烟粉虱", "蚜虫", "线虫",
            "生根", "缓苗", "不发根", "黄叶", "发黄", "缺素", "干尖",
            "旺长", "控旺", "徒长", "保花", "保果", "落花", "落果",
            "膨果", "转色", "增甜", "采后", "树势弱", "返盐", "盐害",
            "根腐", "烂根", "高温", "干旱", "积水", "雨水过多",
        )
    )


def _clarification_questions(*, current_message: str, crop: str, stage: str, problem: str) -> list[str]:
    questions: list[str] = []
    if not crop:
        questions.append("先确认作物和品种")
    if not stage:
        questions.append("确认当前生育阶段")
    normalized = normalize_knowledge_value(current_message)
    if not problem:
        questions.append("确认症状部位、出现时间、发生范围及近期水肥用药")
    if any(term in normalized for term in ("黄叶", "发黄", "叶黄")):
        questions.extend(("确认新叶还是老叶发黄", "查看根系、积水、病斑和是否整片发生"))
    if "旺长" in normalized:
        questions.extend(("确认是否已开花坐果", "确认近期氮肥、浇水和夜温情况"))
    result: list[str] = []
    for item in questions:
        if item not in result:
            result.append(item)
    return result[:5]


def _product_payload(product: Product) -> dict[str, Any]:
    return {
        "product_id": product.product_id,
        "product_name": product.product_name,
        "generic_name": product.product_name,
        "knowledge_kind": product.knowledge_kind.value,
        "category": product.category,
        "active_ingredients": list(product.active_ingredients),
        "key_functions": list(product.key_functions),
        "applicable_crops": list(product.applicable_crops),
        "applicable_stages": list(product.applicable_stages),
        "applicable_problems": list(product.applicable_problems),
        "usage_methods": list(product.usage_methods),
        "dosage_text": product.dosage_text,
        "dosage_structured": product.dosage_structured,
        "precautions": list(product.precautions),
        "incompatibilities": list(product.incompatibilities),
        "risk_notes": list(product.risk_notes),
        "source_id": product.source_id,
        "status": product.status.value,
        "review_state": product.review_state.value,
        "version": product.version,
    }


def _solution_payload(
    solution: Solution, repository: ProductKnowledgeRepository
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in solution.items:
        product = repository.get_product(item.product_id)
        items.append(
            {
                "product_id": item.product_id,
                "generic_name": product.product_name if product else "",
                "knowledge_kind": product.knowledge_kind.value if product else "",
                "role": item.role,
                "dosage": item.dosage,
                "sequence": item.sequence,
                "timing": item.timing,
                "notes": item.notes,
            }
        )
    return {
        "solution_id": solution.solution_id,
        "solution_name": solution.solution_name,
        "crops": list(solution.crops),
        "target_problems": list(solution.target_problems),
        "growth_stages": list(solution.growth_stages),
        "region_constraints": list(solution.region_constraints),
        "conditions": solution.conditions,
        "expected_goal": solution.expected_goal,
        "precautions": list(solution.precautions),
        "risk_notes": list(solution.risk_notes),
        "source_id": solution.source_id,
        "status": solution.status.value,
        "review_state": solution.review_state.value,
        "version": solution.version,
        "items": items,
    }
