from __future__ import annotations

import hashlib
import json
from typing import Any

from whochat.core.knowledge_models import ProductKnowledgeContext
from whochat.core.models import Message
from whochat.storage.knowledge_gaps import KnowledgeGapRepository, ReplyPipelineMetricRepository
from whochat.storage.product_knowledge import ProductKnowledgeRepository, normalize_knowledge_value


ISSUE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("disease", ("病害", "炭疽", "灰霉", "白粉", "软腐", "疫病", "叶斑")),
    ("pest", ("虫害", "蓟马", "红蜘蛛", "白粉虱", "烟粉虱", "蚜虫", "线虫")),
    ("rooting_seedling", ("生根", "缓苗", "不发根", "新根")),
    ("yellowing_nutrition", ("黄叶", "发黄", "缺素", "干尖")),
    ("vigor_control", ("旺长", "控旺", "徒长")),
    ("flower_fruit_set", ("保花", "保果", "落花", "落果", "坐果")),
    ("fruit_expansion", ("膨果", "果实膨大")),
    ("color_sweetness", ("转色", "增甜", "着色")),
    ("postharvest_recovery", ("采后", "恢复树势", "树势弱")),
    ("soil_salinity", ("返盐", "盐害", "返碱")),
    ("root_rot", ("根腐", "烂根", "死根")),
    ("heat_drought", ("高温", "干旱", "缺水")),
    ("excess_rain", ("雨水过多", "连续下雨", "积水", "涝")),
)


class KnowledgeGapAnalyzer:
    """Record what verified knowledge is missing; never create knowledge."""

    def __init__(self, repository: KnowledgeGapRepository) -> None:
        self.repository = repository

    def analyze(
        self,
        *,
        source_message: Message,
        knowledge: ProductKnowledgeContext | None,
    ) -> dict[str, Any]:
        context = knowledge or ProductKnowledgeContext(status="empty")
        text = source_message.text.strip()
        issue_type = classify_issue(text, context.problem)
        product_ids = [str(item.get("product_id") or "") for item in context.products if item.get("product_id")]
        solution_ids = [str(item.get("solution_id") or "") for item in context.solutions if item.get("solution_id")]
        missing: list[str] = []
        if not product_ids and not solution_ids:
            missing.append("no_matching_knowledge")
        if not context.crop:
            missing.append("missing_crop")
        if not context.stage:
            missing.append("missing_growth_stage")
        if not context.problem:
            missing.append("missing_problem")
        if (context.crop or context.problem or context.stage) and not solution_ids:
            missing.append("missing_solution")
        if (product_ids or solution_ids) and not context.precautions:
            missing.append("missing_precautions")
        reliable_dosage = any(
            bool(item.get("dosage_text") or item.get("dosage_structured"))
            for item in context.products
        ) or any(
            bool(line.get("dosage"))
            for solution in context.solutions
            for line in solution.get("items", [])
        )
        dose_intent = _contains_any(text, ("用量", "剂量", "多少倍", "多少克", "多少毫升", "一亩", "兑水", "几次", "间隔"))
        if (product_ids or solution_ids) and not reliable_dosage:
            missing.append("missing_reliable_dosage")

        missing = list(dict.fromkeys(missing))
        if not product_ids and not solution_ids:
            capability = "clarification_only"
        elif context.requires_clarification or (dose_intent and not reliable_dosage):
            capability = "partial_direction"
        else:
            capability = "direct_direction"
        affects_reply = capability != "direct_direction" or bool(
            {"no_matching_knowledge", "missing_solution"}.intersection(missing)
        ) or (dose_intent and "missing_reliable_dosage" in missing)
        available = {
            "products": [
                {
                    "id": item.get("product_id"),
                    "name": item.get("generic_name") or item.get("product_name"),
                    "kind": item.get("knowledge_kind"),
                    "source_id": item.get("source_id"),
                }
                for item in context.products[:8]
            ],
            "solutions": [
                {"id": item.get("solution_id"), "name": item.get("solution_name")}
                for item in context.solutions[:5]
            ],
        }
        signature_payload = {
            "crop": normalize_knowledge_value(context.crop),
            "stage": normalize_knowledge_value(context.stage),
            "problem": normalize_knowledge_value(context.problem),
            "issue_type": issue_type,
            "missing": sorted(missing),
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.repository.record_evaluation(
            source_message=source_message,
            crop=context.crop,
            growth_stage=context.stage,
            problem=context.problem,
            issue_type=issue_type,
            issue_text=_issue_excerpt(text),
            matched_product_ids=product_ids,
            matched_solution_ids=solution_ids,
            missing_types=missing,
            reply_capability=capability,
            affects_reply=affects_reply,
            signature_hash=signature,
            available_knowledge=available,
        ) | {
            "missing_types": missing,
            "reply_capability": capability,
            "affects_reply": affects_reply,
        }


class KnowledgeCoverageReportService:
    def __init__(
        self,
        gaps: KnowledgeGapRepository,
        products: ProductKnowledgeRepository,
        metrics: ReplyPipelineMetricRepository | None = None,
    ) -> None:
        self.gaps = gaps
        self.products = products
        self.metrics = metrics

    def build(self, *, recent_limit: int = 50, gap_limit: int = 30) -> dict[str, Any]:
        with self.products.db.connect() as conn:
            product_rows = conn.execute(
                """
                SELECT product_name,knowledge_kind,applicable_crops_json,
                       applicable_problems_json,dosage_text,review_state,status
                  FROM product_catalog
                 WHERE knowledge_kind<>'company_product'
                """
            ).fetchall()
            active_rows = [
                row for row in product_rows
                if row["status"] == "active" and row["review_state"] == "verified"
            ]
            solution_rows = conn.execute(
                """
                SELECT crops_json,target_problems_json,status,review_state
                  FROM solution_templates
                """
            ).fetchall()
        active_solutions = [
            row for row in solution_rows
            if row["status"] == "active" and row["review_state"] == "verified"
        ]
        crops = _json_terms(active_rows, "applicable_crops_json") | _json_terms(active_solutions, "crops_json")
        problems = _json_terms(active_rows, "applicable_problems_json") | _json_terms(active_solutions, "target_problems_json")
        recent = self.gaps.recent_evaluations(recent_limit)
        top = self.gaps.top_gaps(gap_limit)
        hits = [row for row in recent if row["reply_capability"] == "direct_direction"]
        misses = [row for row in recent if row["reply_capability"] == "clarification_only"]
        partial = [row for row in recent if row["reply_capability"] == "partial_direction"]
        return {
            "knowledge": {
                "active_generic": len(active_rows),
                "verified_solutions": len(active_solutions),
                "draft": sum(1 for row in product_rows if row["status"] == "draft"),
                "conflicting": sum(1 for row in product_rows if row["review_state"] == "conflicting"),
                "review_required": sum(1 for row in product_rows if row["review_state"] == "review_required"),
                "crops": sorted(crops),
                "problems": sorted(problems),
                "generic_categories": sorted(row["product_name"] for row in active_rows if row["knowledge_kind"] == "generic_category"),
                "active_ingredients": sorted(row["product_name"] for row in active_rows if row["knowledge_kind"] == "active_ingredient"),
                "functional_ingredients": sorted(row["product_name"] for row in active_rows if row["knowledge_kind"] == "functional_ingredient"),
                "without_reliable_dosage": sorted(row["product_name"] for row in active_rows if not row["dosage_text"]),
            },
            "coverage": {
                **self.gaps.counts(),
                "recent_total": len(recent),
                "direct_direction": len(hits),
                "partial_direction": len(partial),
                "clarification_only": len(misses),
                "recent_hits": [_evaluation_summary(row) for row in hits[:10]],
                "recent_partial": [_evaluation_summary(row) for row in partial[:10]],
                "recent_misses": [_evaluation_summary(row) for row in misses[:10]],
            },
            "top_gaps": [_gap_summary(row) for row in top],
            "timing": self.metrics.timing_summary() if self.metrics else {},
        }


def classify_issue(text: str, problem: str = "") -> str:
    value = normalize_knowledge_value(f"{problem} {text}")
    for issue, terms in ISSUE_TERMS:
        if any(normalize_knowledge_value(term) in value for term in terms):
            return issue
    return "unknown"


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    value = normalize_knowledge_value(text)
    return any(normalize_knowledge_value(term) in value for term in terms)


def _issue_excerpt(text: str) -> str:
    return " ".join(text.split())[:240]


def _json_terms(rows, column: str) -> set[str]:
    result: set[str] = set()
    for row in rows:
        try:
            values = json.loads(row[column] or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        result.update(str(item) for item in values if str(item).strip())
    return result


def _evaluation_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "crop": row["crop"],
        "stage": row["growth_stage"],
        "problem": row["problem"],
        "issue_type": row["issue_type"],
        "capability": row["reply_capability"],
        "missing": json.loads(row["missing_types_json"]),
        "evaluated_at": row["evaluated_at"],
    }


def _gap_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "crop": row["crop"],
        "stage": row["growth_stage"],
        "problem": row["problem"],
        "issue_type": row["issue_type"],
        "missing": json.loads(row["missing_types_json"]),
        "occurrences": row["occurrence_count"],
        "last_seen_at": row["last_seen_at"],
        "affects_reply": bool(row["affects_reply"]),
        "status": row["status"],
    }
