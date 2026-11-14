from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class KnowledgeStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"


class KnowledgeReviewState(StrEnum):
    VERIFIED = "verified"
    UNREVIEWED = "unreviewed"
    REVIEW_REQUIRED = "review_required"
    CONFLICTING = "conflicting"


class KnowledgeKind(StrEnum):
    COMPANY_PRODUCT = "company_product"
    GENERIC_CATEGORY = "generic_category"
    ACTIVE_INGREDIENT = "active_ingredient"
    FUNCTIONAL_INGREDIENT = "functional_ingredient"


class KnowledgeSourceType(StrEnum):
    MANUAL = "manual"
    COMPANY_DOCUMENT = "company_document"
    FEISHU = "feishu"
    LEXIANG = "lexiang"
    HISTORICAL_SOLUTION = "historical_solution"
    VERIFIED_CHAT = "verified_chat"
    OTHER = "other"


class ProductClaimType(StrEnum):
    FUNCTION = "function"
    CROP = "crop"
    PROBLEM = "problem"
    STAGE = "stage"
    DOSAGE = "dosage"
    USAGE = "usage"
    PRECAUTION = "precaution"
    COMPATIBILITY = "compatibility"
    RISK = "risk"
    INGREDIENT = "ingredient"
    SPECIFICATION = "specification"
    OTHER = "other"


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    source_type: KnowledgeSourceType
    title: str
    source_reference: str
    source_hash: str
    metadata: dict[str, Any]
    status: KnowledgeStatus
    imported_at: str
    updated_at: str


@dataclass(frozen=True)
class Product:
    product_id: str
    product_name: str
    knowledge_kind: KnowledgeKind
    aliases: tuple[str, ...]
    category: str
    formulation: str
    active_ingredients: tuple[dict[str, Any], ...]
    ingredient_content: str
    specifications: dict[str, Any]
    key_functions: tuple[str, ...]
    applicable_crops: tuple[str, ...]
    applicable_stages: tuple[str, ...]
    applicable_problems: tuple[str, ...]
    usage_methods: tuple[str, ...]
    dosage_text: str
    dosage_structured: dict[str, Any]
    precautions: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    risk_notes: tuple[str, ...]
    notes: str
    status: KnowledgeStatus
    review_state: KnowledgeReviewState
    source_id: str
    version: int
    content_hash: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ProductClaim:
    claim_id: str
    product_id: str
    claim_type: ProductClaimType
    claim_value: str
    normalized_value: str
    source_id: str
    source_text: str
    status: KnowledgeStatus
    review_state: KnowledgeReviewState
    conflict_group_id: str
    dosage_structured: dict[str, Any]
    confidence: float
    version: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SolutionItem:
    solution_item_id: str
    solution_id: str
    product_id: str
    role: str
    dosage: str
    sequence: int
    timing: str
    notes: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class UnresolvedSolutionItem:
    unresolved_item_id: str
    solution_id: str
    raw_product_name: str
    normalized_product_name: str
    role: str
    dosage: str
    sequence: int
    timing: str
    notes: str
    source_id: str
    review_state: KnowledgeReviewState
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Solution:
    solution_id: str
    solution_name: str
    crops: tuple[str, ...]
    target_problems: tuple[str, ...]
    growth_stages: tuple[str, ...]
    region_constraints: tuple[str, ...]
    conditions: dict[str, Any]
    expected_goal: str
    precautions: tuple[str, ...]
    risk_notes: tuple[str, ...]
    notes: str
    status: KnowledgeStatus
    review_state: KnowledgeReviewState
    source_id: str
    version: int
    content_hash: str
    created_at: str
    updated_at: str
    items: tuple[SolutionItem, ...] = ()
    unresolved_items: tuple[UnresolvedSolutionItem, ...] = ()


@dataclass(frozen=True)
class ProductKnowledgeContext:
    products: tuple[dict[str, Any], ...] = ()
    solutions: tuple[dict[str, Any], ...] = ()
    precautions: tuple[str, ...] = ()
    crop: str = ""
    problem: str = ""
    stage: str = ""
    region: str = ""
    status: str = "empty"
    requires_clarification: bool = False
    clarification_questions: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema": "whochat.product_knowledge.v2",
            "status": self.status,
            "query": {
                "crop": self.crop,
                "problem": self.problem,
                "stage": self.stage,
                "region": self.region,
            },
            "products": list(self.products),
            "solutions": list(self.solutions),
            "precautions": list(self.precautions),
            "decision_policy": {
                "eligible_knowledge": "active_verified_generic_only",
                "requires_clarification": self.requires_clarification,
                "clarification_questions": list(self.clarification_questions),
                "dosage_rule": "only_explicit_verified_matching_context",
                "brand_rule": "never_recommend_company_sku_or_brand",
            },
            "limits": {"products": 8, "solutions": 5},
        }
