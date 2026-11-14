from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from dataclasses import asdict
from typing import Any, Iterable

from whochat.core.knowledge_models import (
    KnowledgeKind,
    KnowledgeSource,
    KnowledgeSourceType,
    KnowledgeReviewState,
    KnowledgeStatus,
    Product,
    ProductClaim,
    ProductClaimType,
    Solution,
    SolutionItem,
    UnresolvedSolutionItem,
)
from whochat.core.models import utc_now_iso
from whochat.storage.database import Database


def normalize_knowledge_value(value: Any) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def _clean(value: Any, limit: int = 1000, *, required: bool = False) -> str:
    text = " ".join(str(value or "").strip().split())[:limit]
    if required and not text:
        raise ValueError("non_empty_value_required")
    return text


def _strings(value: Any, *, limit: int = 80, item_limit: int = 300) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw = [part for part in value.replace("；", ";").split(";") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raise ValueError("string_list_required")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw[:limit]:
        text = _clean(item, item_limit)
        key = normalize_knowledge_value(text)
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("json_object_required")
    return parsed


def _ingredients(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("active_ingredients_list_required")
    result: list[dict[str, Any]] = []
    for item in parsed[:40]:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            raise ValueError("active_ingredient_object_required")
        name = _clean(item.get("name"), 200, required=True)
        normalized = {str(k): v for k, v in item.items() if isinstance(k, str)}
        normalized["name"] = name
        result.append(normalized)
    return result


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:28]
    return f"{prefix}_{digest}"


def _status(value: Any, default: KnowledgeStatus = KnowledgeStatus.DRAFT) -> KnowledgeStatus:
    return KnowledgeStatus(str(value or default.value))


def _review_state(
    value: Any,
    default: KnowledgeReviewState = KnowledgeReviewState.UNREVIEWED,
) -> KnowledgeReviewState:
    return KnowledgeReviewState(str(value or default.value))


class ProductKnowledgeRepository:
    """SQLite repository for source-traceable products and solution templates."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_source(
        self,
        *,
        source_type: str | KnowledgeSourceType,
        title: str,
        source_reference: str = "",
        source_hash: str = "",
        metadata: dict[str, Any] | None = None,
        status: str | KnowledgeStatus = KnowledgeStatus.ACTIVE,
        source_id: str | None = None,
    ) -> KnowledgeSource:
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._upsert_source_conn(
                conn,
                {
                    "source_id": source_id,
                    "source_type": str(source_type),
                    "title": title,
                    "source_reference": source_reference,
                    "source_hash": source_hash,
                    "metadata": metadata or {},
                    "status": str(status),
                },
            )

    def get_source(self, source_id: str) -> KnowledgeSource | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM knowledge_sources WHERE source_id=?", (source_id,)).fetchone()
        return _source_from_row(row) if row else None

    def upsert_product(self, payload: dict[str, Any]) -> Product:
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._upsert_product_conn(conn, payload)

    def get_product(self, product_id: str) -> Product | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM product_catalog WHERE product_id=?", (product_id,)).fetchone()
        return _product_from_row(row) if row else None

    def reply_read_snapshot(
        self,
        *,
        knowledge_kinds: Iterable[str | KnowledgeKind] | None = None,
    ) -> _ReplyKnowledgeSnapshot:
        """Materialize one reply's knowledge from a single read transaction.

        Keep linked rows raw until the old retrieval path would inspect them:
        unrelated malformed or ineligible records must not introduce failures.
        This object is local to one build, never a cross-request cache.
        """
        kinds = tuple(str(item) for item in (knowledge_kinds or ()))
        where = "status='active' AND review_state='verified'"
        if kinds:
            where += f" AND knowledge_kind IN ({','.join('?' for _ in kinds)})"
        conn = self.db.connect_readonly()
        try:
            conn.execute("BEGIN")
            product_rows = conn.execute(
                f"""SELECT * FROM product_catalog WHERE {where}
                    ORDER BY normalized_name LIMIT 1000""", kinds,
            ).fetchall()
            solution_rows = conn.execute(
                """SELECT * FROM solution_templates
                   WHERE status='active' AND review_state='verified'
                     AND NOT EXISTS (
                         SELECT 1 FROM solution_unresolved_items u
                         WHERE u.solution_id=solution_templates.solution_id
                     )
                   ORDER BY normalized_name"""
            ).fetchall()
            item_rows = conn.execute(
                "SELECT * FROM solution_items ORDER BY solution_id,sequence,solution_item_id"
            ).fetchall()
            linked_rows = conn.execute(
                """SELECT * FROM product_catalog WHERE product_id IN (
                       SELECT product_id FROM solution_items
                   )"""
            ).fetchall()
        finally:
            conn.close()
        return _ReplyKnowledgeSnapshot(product_rows, solution_rows, item_rows, linked_rows)

    def find_product_by_name_or_alias(self, value: str, *, active_only: bool = False) -> Product | None:
        key = normalize_knowledge_value(value)
        if not key:
            return None
        active_sql = " AND p.status='active' AND p.review_state='verified'" if active_only else ""
        with self.db.connect() as conn:
            row = conn.execute(
                f"""
                SELECT DISTINCT p.* FROM product_catalog p
                LEFT JOIN product_aliases a ON a.product_id=p.product_id
                WHERE (p.normalized_name=? OR a.normalized_alias=?){active_sql}
                ORDER BY CASE p.status WHEN 'active' THEN 0 WHEN 'draft' THEN 1 ELSE 2 END,
                         p.updated_at DESC
                LIMIT 1
                """,
                (key, key),
            ).fetchone()
        return _product_from_row(row) if row else None

    def list_active_products(
        self,
        *,
        limit: int = 200,
        knowledge_kinds: Iterable[str | KnowledgeKind] | None = None,
    ) -> list[Product]:
        kinds = tuple(str(item) for item in (knowledge_kinds or ()))
        where = "status='active' AND review_state='verified'"
        params: list[Any] = []
        if kinds:
            where += f" AND knowledge_kind IN ({','.join('?' for _ in kinds)})"
            params.extend(kinds)
        params.append(max(1, min(int(limit), 1000)))
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM product_catalog
                   WHERE {where}
                   ORDER BY normalized_name LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [_product_from_row(row) for row in rows]

    def get_product_claims(self, product_id: str, *, active_only: bool = True) -> list[ProductClaim]:
        sql = "SELECT * FROM product_claims WHERE product_id=?"
        if active_only:
            sql += " AND status='active' AND review_state='verified'"
        sql += " ORDER BY claim_type,normalized_value,claim_id"
        with self.db.connect() as conn:
            rows = conn.execute(sql, (product_id,)).fetchall()
        return [_claim_from_row(row) for row in rows]

    def list_products_for_crop(self, crop: str, *, limit: int = 100) -> list[Product]:
        return self._list_products_for_claim(ProductClaimType.CROP, crop, limit)

    def list_products_for_problem(self, problem: str, *, limit: int = 100) -> list[Product]:
        return self._list_products_for_claim(ProductClaimType.PROBLEM, problem, limit)

    def list_products_for_stage(self, stage: str, *, limit: int = 100) -> list[Product]:
        return self._list_products_for_claim(ProductClaimType.STAGE, stage, limit)

    def search_products(self, text: str, *, limit: int = 50) -> list[Product]:
        term = _clean(text, 100)
        if not term:
            return []
        pattern = f"%{term}%"
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.* FROM product_catalog p
                LEFT JOIN product_aliases a ON a.product_id=p.product_id
                LEFT JOIN product_claims c ON c.product_id=p.product_id
                    AND c.status='active' AND c.review_state='verified'
                WHERE p.status='active' AND p.review_state='verified' AND (
                    p.product_name LIKE ? OR a.alias LIKE ? OR c.claim_value LIKE ? OR
                    p.notes LIKE ? OR p.key_functions_json LIKE ? OR p.applicable_problems_json LIKE ?
                )
                ORDER BY p.normalized_name LIMIT ?
                """,
                (pattern, pattern, pattern, pattern, pattern, pattern, max(1, min(int(limit), 200))),
            ).fetchall()
        return [_product_from_row(row) for row in rows]

    def upsert_solution(self, payload: dict[str, Any]) -> Solution:
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._upsert_solution_conn(conn, payload)

    def get_solution(self, solution_id: str) -> Solution | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM solution_templates WHERE solution_id=?", (solution_id,)).fetchone()
            if row is None:
                return None
            items = conn.execute(
                "SELECT * FROM solution_items WHERE solution_id=? ORDER BY sequence,solution_item_id",
                (solution_id,),
            ).fetchall()
            unresolved = conn.execute(
                "SELECT * FROM solution_unresolved_items WHERE solution_id=? ORDER BY sequence,unresolved_item_id",
                (solution_id,),
            ).fetchall()
        return _solution_from_row(row, items, unresolved)

    def list_matching_solutions(
        self,
        *,
        crop: str = "",
        problem: str = "",
        stage: str = "",
        region: str = "",
        condition_facts: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Solution]:
        return [item[1] for item in self.rank_matching_solutions(
            crop=crop,
            problem=problem,
            stage=stage,
            region=region,
            condition_facts=condition_facts,
            limit=limit,
        )]

    def rank_matching_solutions(
        self,
        *,
        crop: str = "",
        problem: str = "",
        stage: str = "",
        region: str = "",
        condition_facts: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[tuple[int, Solution]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM solution_templates
                   WHERE status='active' AND review_state='verified'
                     AND NOT EXISTS (
                         SELECT 1 FROM solution_unresolved_items u
                         WHERE u.solution_id=solution_templates.solution_id
                     )
                   ORDER BY normalized_name"""
            ).fetchall()
            item_rows = conn.execute(
                "SELECT * FROM solution_items ORDER BY solution_id,sequence,solution_item_id"
            ).fetchall()
        by_solution: dict[str, list[sqlite3.Row]] = {}
        for row in item_rows:
            by_solution.setdefault(row["solution_id"], []).append(row)
        ranked: list[tuple[int, Solution]] = []
        for row in rows:
            solution = _solution_from_row(row, by_solution.get(row["solution_id"], []), [])
            score = _solution_score(
                solution,
                crop=crop,
                problem=problem,
                stage=stage,
                region=region,
                condition_facts=condition_facts or {},
            )
            if score is not None:
                ranked.append((score, solution))
        ranked.sort(key=lambda item: (-item[0], normalize_knowledge_value(item[1].solution_name)))
        return ranked[: max(1, min(int(limit), 500))]

    def apply_bundle(self, bundle: dict[str, Any]) -> dict[str, int]:
        """Atomically apply a validated import bundle."""
        sources = bundle.get("sources") or []
        products = bundle.get("products") or []
        solutions = bundle.get("solutions") or []
        if not all(isinstance(items, list) for items in (sources, products, solutions)):
            raise ValueError("bundle_lists_required")
        result = {"sources": 0, "products": 0, "solutions": 0}
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in sources:
                self._upsert_source_conn(conn, item)
                result["sources"] += 1
            for item in products:
                self._upsert_product_conn(conn, item)
                result["products"] += 1
            for item in solutions:
                self._upsert_solution_conn(conn, item)
                result["solutions"] += 1
        return result

    def preview_action(self, entity: str, payload: dict[str, Any]) -> str:
        if entity == "product":
            prepared = self._prepare_product(payload)
            table, id_name = "product_catalog", "product_id"
        elif entity == "solution":
            prepared = self._prepare_solution(payload)
            table, id_name = "solution_templates", "solution_id"
        else:
            raise ValueError("unsupported_preview_entity")
        with self.db.connect() as conn:
            row = conn.execute(
                f"SELECT content_hash,version FROM {table} WHERE {id_name}=?",
                (prepared[id_name],),
            ).fetchone()
        if row is None:
            return "create"
        if row["content_hash"] == prepared["content_hash"]:
            return "unchanged"
        return "update" if prepared["version"] > row["version"] else "invalid_version"

    def get_product_source_links(self, product_id: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT l.*,s.source_type,s.title,s.source_reference
                FROM product_source_links l
                JOIN knowledge_sources s ON s.source_id=l.source_id
                WHERE l.product_id=? ORDER BY s.source_type,s.title,l.source_id
                """,
                (product_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_solution_source_links(self, solution_id: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT l.*,s.source_type,s.title,s.source_reference
                FROM solution_source_links l
                JOIN knowledge_sources s ON s.source_id=l.source_id
                WHERE l.solution_id=? ORDER BY s.source_type,s.title,l.source_id
                """,
                (solution_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def inspect_product(self, name_or_alias: str) -> dict[str, Any] | None:
        product = self.find_product_by_name_or_alias(name_or_alias)
        if product is None:
            return None
        with self.db.connect() as conn:
            aliases = [dict(row) for row in conn.execute(
                """
                SELECT a.*,GROUP_CONCAT(x.source_id) AS evidence_source_ids
                FROM product_aliases a
                LEFT JOIN product_alias_sources x ON x.alias_id=a.alias_id
                WHERE a.product_id=? GROUP BY a.alias_id ORDER BY a.normalized_alias
                """,
                (product.product_id,),
            ).fetchall()]
            versions = [dict(row) for row in conn.execute(
                "SELECT * FROM product_versions WHERE product_id=? ORDER BY version DESC",
                (product.product_id,),
            ).fetchall()]
        return {
            "product": asdict(product),
            "aliases": aliases,
            "claims": [asdict(item) for item in self.get_product_claims(product.product_id, active_only=False)],
            "sources": self.get_product_source_links(product.product_id),
            "versions": versions,
        }

    def inspect_solution(self, solution_id: str) -> dict[str, Any] | None:
        solution = self.get_solution(solution_id)
        if solution is None:
            return None
        with self.db.connect() as conn:
            versions = [dict(row) for row in conn.execute(
                "SELECT * FROM solution_versions WHERE solution_id=? ORDER BY version DESC",
                (solution_id,),
            ).fetchall()]
        return {
            "solution": asdict(solution),
            "sources": self.get_solution_source_links(solution_id),
            "versions": versions,
        }

    def knowledge_quality_report(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            product_rows = conn.execute("SELECT * FROM product_catalog").fetchall()
            solution_rows = conn.execute("SELECT * FROM solution_templates").fetchall()
            unresolved = int(conn.execute("SELECT COUNT(*) FROM solution_unresolved_items").fetchone()[0])
            conflicts = int(conn.execute(
                """SELECT
                     (SELECT COUNT(*) FROM product_catalog WHERE review_state='conflicting') +
                     (SELECT COUNT(*) FROM product_claims WHERE review_state='conflicting') +
                     (SELECT COUNT(*) FROM solution_templates WHERE review_state='conflicting')"""
            ).fetchone()[0])
            fully_bound = int(conn.execute(
                """SELECT COUNT(*) FROM solution_templates s
                   WHERE NOT EXISTS (
                       SELECT 1 FROM solution_unresolved_items u WHERE u.solution_id=s.solution_id
                   )"""
            ).fetchone()[0])

        total = len(product_rows)
        def ratio(predicate: Any) -> float:
            if not total:
                return 0.0
            return round(sum(1 for row in product_rows if predicate(row)) / total, 4)

        return {
            "products": {
                "total": total,
                "active": sum(1 for row in product_rows if row["status"] == "active"),
                "draft": sum(1 for row in product_rows if row["status"] == "draft"),
                "inactive": sum(1 for row in product_rows if row["status"] == "inactive"),
                "review_required": sum(1 for row in product_rows if row["review_state"] in {"review_required", "conflicting"}),
                "with_ingredients_ratio": ratio(lambda row: row["active_ingredients_json"] != "[]" or bool(row["ingredient_content"])),
                "with_dosage_ratio": ratio(lambda row: bool(row["dosage_text"])),
                "with_precautions_ratio": ratio(lambda row: row["precautions_json"] != "[]"),
                "with_sources_ratio": ratio(lambda row: bool(row["source_id"])),
                "with_versions_ratio": ratio(lambda row: int(row["version"]) >= 1),
                "with_crops_ratio": ratio(lambda row: row["applicable_crops_json"] != "[]"),
            },
            "solutions": {
                "total": len(solution_rows),
                "fully_bound": fully_bound,
                "with_unresolved_items": sum(
                    1 for row in solution_rows
                    if any(item.solution_id == row["solution_id"] for item in self.get_solution(row["solution_id"]).unresolved_items)
                ),
                "unresolved_item_count": unresolved,
            },
            "conflicting_knowledge_items": conflicts,
        }

    def count_rows(self) -> dict[str, int]:
        tables = (
            "knowledge_sources", "product_catalog", "product_aliases", "product_alias_sources",
            "product_claims", "product_source_links", "product_versions", "solution_templates",
            "solution_items", "solution_unresolved_items", "solution_source_links",
            "solution_versions", "knowledge_import_runs",
        )
        with self.db.connect() as conn:
            return {name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in tables}

    def _list_products_for_claim(self, claim_type: ProductClaimType, value: str, limit: int) -> list[Product]:
        key = normalize_knowledge_value(value)
        if not key:
            return []
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.* FROM product_catalog p
                JOIN product_claims c ON c.product_id=p.product_id
                WHERE p.status='active' AND p.review_state='verified'
                  AND c.status='active' AND c.review_state='verified'
                  AND c.claim_type=? AND c.normalized_value=?
                ORDER BY p.normalized_name LIMIT ?
                """,
                (claim_type.value, key, max(1, min(int(limit), 500))),
            ).fetchall()
        return [_product_from_row(row) for row in rows]

    def _upsert_source_conn(self, conn: sqlite3.Connection, payload: dict[str, Any]) -> KnowledgeSource:
        source_type = KnowledgeSourceType(str(payload.get("source_type") or "manual"))
        title = _clean(payload.get("title"), 300, required=True)
        reference = _clean(payload.get("source_reference"), 1000)
        source_id = _clean(payload.get("source_id"), 120) or _stable_id(
            "knowledge_source", source_type.value, reference or normalize_knowledge_value(title)
        )
        source_hash = _clean(payload.get("source_hash"), 128)
        metadata = _object(payload.get("metadata") if "metadata" in payload else payload.get("metadata_json"))
        status = _status(payload.get("status"), KnowledgeStatus.ACTIVE)
        now = utc_now_iso()
        row = conn.execute("SELECT * FROM knowledge_sources WHERE source_id=?", (source_id,)).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO knowledge_sources(
                    source_id,source_type,title,source_reference,source_hash,metadata_json,
                    status,imported_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (source_id, source_type.value, title, reference, source_hash, _json(metadata), status.value, now, now),
            )
        else:
            conn.execute(
                """
                UPDATE knowledge_sources SET source_type=?,title=?,source_reference=?,source_hash=?,
                    metadata_json=?,status=?,updated_at=? WHERE source_id=?
                """,
                (source_type.value, title, reference, source_hash, _json(metadata), status.value, now, source_id),
            )
        return _source_from_row(conn.execute("SELECT * FROM knowledge_sources WHERE source_id=?", (source_id,)).fetchone())

    def _prepare_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("product_object_required")
        source_id = _clean(payload.get("source_id"), 120, required=True)
        name = _clean(payload.get("product_name"), 240, required=True)
        normalized_name = normalize_knowledge_value(name)
        product_id = _clean(payload.get("product_id"), 120) or _stable_id("product", source_id, normalized_name)
        status = _status(payload.get("status"))
        version = int(payload.get("version") or 1)
        if version < 1:
            raise ValueError("product_version_invalid")
        aliases = _strings(payload.get("aliases"))
        aliases = [item for item in aliases if normalize_knowledge_value(item) != normalized_name]
        status = _status(payload.get("status"))
        review_default = (
            KnowledgeReviewState.VERIFIED
            if status == KnowledgeStatus.ACTIVE
            else KnowledgeReviewState.UNREVIEWED
        )
        source_links = payload.get("source_links") or [{
            "source_id": source_id,
            "source_product_name": name,
            "source_version": str(payload.get("source_version") or version),
            "source_text": payload.get("source_text") or "",
            "source_hash": payload.get("source_hash") or "",
            "status": status.value,
            "review_state": str(payload.get("review_state") or review_default.value),
        }]
        if not isinstance(source_links, list) or not all(isinstance(item, dict) for item in source_links):
            raise ValueError("product_source_links_required")
        prepared = {
            "product_id": product_id,
            "product_name": name,
            "normalized_name": normalized_name,
            "knowledge_kind": KnowledgeKind(
                str(payload.get("knowledge_kind") or KnowledgeKind.COMPANY_PRODUCT.value)
            ).value,
            "aliases": aliases,
            "category": _clean(payload.get("category"), 160),
            "formulation": _clean(payload.get("formulation"), 160),
            "active_ingredients": _ingredients(payload.get("active_ingredients")),
            "ingredient_content": _clean(payload.get("ingredient_content"), 500),
            "specifications": _object(payload.get("specifications")),
            "key_functions": _strings(payload.get("key_functions")),
            "applicable_crops": _strings(payload.get("applicable_crops")),
            "applicable_stages": _strings(payload.get("applicable_stages")),
            "applicable_problems": _strings(payload.get("applicable_problems")),
            "usage_methods": _strings(payload.get("usage_methods")),
            "dosage_text": _clean(payload.get("dosage_text"), 1000),
            "dosage_structured": _object(payload.get("dosage_structured")),
            "precautions": _strings(payload.get("precautions")),
            "incompatibilities": _strings(payload.get("incompatibilities")),
            "risk_notes": _strings(payload.get("risk_notes")),
            "notes": _clean(payload.get("notes"), 3000),
            "status": status.value,
            "review_state": _review_state(payload.get("review_state"), review_default).value,
            "source_id": source_id,
            "version": version,
            "source_text": _clean(payload.get("source_text"), 4000),
            "claims": payload.get("claims") or [],
            "source_links": [
                {
                    "source_id": _clean(item.get("source_id"), 120, required=True),
                    "source_product_name": _clean(item.get("source_product_name") or name, 240),
                    "source_version": _clean(item.get("source_version"), 120),
                    "source_text": _clean(item.get("source_text"), 4000),
                    "source_hash": _clean(item.get("source_hash"), 128),
                    "status": _status(item.get("status"), status).value,
                    "review_state": _review_state(item.get("review_state"), review_default).value,
                }
                for item in source_links
            ],
            "alias_sources": payload.get("alias_sources") or {},
        }
        if not isinstance(prepared["alias_sources"], dict):
            raise ValueError("alias_sources_object_required")
        hash_payload = {key: value for key, value in prepared.items() if key not in {"claims", "source_text"}}
        hash_payload["claims"] = self._prepared_claims(prepared)
        prepared["content_hash"] = _hash(hash_payload)
        return prepared

    def _prepared_claims(self, product: dict[str, Any]) -> list[dict[str, Any]]:
        automatic: list[tuple[str, str]] = []
        fields = (
            ("function", "key_functions"), ("crop", "applicable_crops"),
            ("problem", "applicable_problems"), ("stage", "applicable_stages"),
            ("usage", "usage_methods"), ("precaution", "precautions"),
            ("compatibility", "incompatibilities"), ("risk", "risk_notes"),
        )
        for claim_type, field in fields:
            automatic.extend((claim_type, value) for value in product[field])
        if product["dosage_text"]:
            automatic.append(("dosage", product["dosage_text"]))
        for ingredient in product["active_ingredients"]:
            automatic.append(("ingredient", str(ingredient.get("name") or "")))

        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        raw_claims = list(product.get("claims") or [])
        raw_claims.extend({"claim_type": kind, "claim_value": value} for kind, value in automatic)
        for raw in raw_claims:
            if not isinstance(raw, dict):
                raise ValueError("product_claim_object_required")
            claim_type = ProductClaimType(str(raw.get("claim_type") or "other"))
            value = _clean(raw.get("claim_value"), 1000, required=True)
            source_id = _clean(raw.get("source_id") or product["source_id"], 120, required=True)
            normalized = normalize_knowledge_value(raw.get("normalized_value") or value)
            key = (claim_type.value, normalized, source_id)
            if key in seen:
                continue
            seen.add(key)
            confidence = float(raw.get("confidence", 1.0))
            if not 0 <= confidence <= 1:
                raise ValueError("product_claim_confidence_invalid")
            result.append({
                "claim_type": claim_type.value,
                "claim_value": value,
                "normalized_value": normalized,
                "source_id": source_id,
                "source_text": _clean(raw.get("source_text") or product.get("source_text") or value, 4000),
                "status": _status(raw.get("status"), KnowledgeStatus(product["status"])).value,
                "review_state": _review_state(
                    raw.get("review_state"),
                    KnowledgeReviewState(product["review_state"]),
                ).value,
                "conflict_group_id": _clean(raw.get("conflict_group_id"), 120),
                "dosage_structured": _object(raw.get("dosage_structured")),
                "confidence": confidence,
            })
        return result

    def _upsert_product_conn(self, conn: sqlite3.Connection, payload: dict[str, Any]) -> Product:
        product = self._prepare_product(payload)
        self._require_source(conn, product["source_id"])
        for link in product["source_links"]:
            self._require_source(conn, link["source_id"])
        claims = self._prepared_claims(product)
        for claim in claims:
            self._require_source(conn, claim["source_id"])
        existing = conn.execute(
            "SELECT * FROM product_catalog WHERE product_id=?", (product["product_id"],)
        ).fetchone()
        if existing is not None and existing["content_hash"] == product["content_hash"]:
            return _product_from_row(existing)
        if existing is not None and product["version"] <= existing["version"]:
            raise ValueError("product_version_must_increase")
        for alias in product["aliases"]:
            collision = conn.execute(
                "SELECT product_id FROM product_aliases WHERE normalized_alias=? AND product_id<>?",
                (normalize_knowledge_value(alias), product["product_id"]),
            ).fetchone()
            if collision is not None:
                raise ValueError("product_alias_collision")
        now = utc_now_iso()
        params = (
            product["product_name"], product["normalized_name"], product["knowledge_kind"],
            _json(product["aliases"]),
            product["category"], product["formulation"], _json(product["active_ingredients"]),
            product["ingredient_content"], _json(product["specifications"]),
            _json(product["key_functions"]), _json(product["applicable_crops"]),
            _json(product["applicable_stages"]), _json(product["applicable_problems"]),
            _json(product["usage_methods"]), product["dosage_text"], _json(product["dosage_structured"]),
            _json(product["precautions"]), _json(product["incompatibilities"]),
            _json(product["risk_notes"]), product["notes"], product["status"], product["source_id"],
            product["review_state"], product["version"], product["content_hash"], now,
        )
        if existing is None:
            conn.execute(
                """
                INSERT INTO product_catalog(
                    product_name,normalized_name,knowledge_kind,aliases_json,category,formulation,
                    active_ingredients_json,ingredient_content,specifications_json,key_functions_json,
                    applicable_crops_json,applicable_stages_json,applicable_problems_json,
                    usage_methods_json,dosage_text,dosage_structured_json,precautions_json,
                    incompatibilities_json,risk_notes_json,notes,status,source_id,review_state,version,
                    content_hash,created_at,updated_at,product_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                params + (now, product["product_id"]),
            )
        else:
            conn.execute(
                """
                UPDATE product_catalog SET
                    product_name=?,normalized_name=?,knowledge_kind=?,aliases_json=?,category=?,formulation=?,
                    active_ingredients_json=?,ingredient_content=?,specifications_json=?,key_functions_json=?,
                    applicable_crops_json=?,applicable_stages_json=?,applicable_problems_json=?,
                    usage_methods_json=?,dosage_text=?,dosage_structured_json=?,precautions_json=?,
                    incompatibilities_json=?,risk_notes_json=?,notes=?,status=?,source_id=?,review_state=?,version=?,
                    content_hash=?,updated_at=? WHERE product_id=?
                """,
                params + (product["product_id"],),
            )
            conn.execute("DELETE FROM product_aliases WHERE product_id=?", (product["product_id"],))
            conn.execute("DELETE FROM product_claims WHERE product_id=?", (product["product_id"],))
            conn.execute("DELETE FROM product_source_links WHERE product_id=?", (product["product_id"],))
        for link in product["source_links"]:
            conn.execute(
                """
                INSERT INTO product_source_links(
                    product_source_link_id,product_id,source_id,source_product_name,source_version,
                    source_text,source_hash,status,review_state,imported_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _stable_id("product_source", product["product_id"], link["source_id"]),
                    product["product_id"], link["source_id"], link["source_product_name"],
                    link["source_version"], link["source_text"], link["source_hash"],
                    link["status"], link["review_state"], now, now,
                ),
            )
        for alias in product["aliases"]:
            alias_id = _stable_id("product_alias", product["product_id"], normalize_knowledge_value(alias))
            conn.execute(
                "INSERT INTO product_aliases(alias_id,product_id,alias,normalized_alias,source_id,created_at) VALUES (?,?,?,?,?,?)",
                (alias_id, product["product_id"], alias,
                 normalize_knowledge_value(alias), product["source_id"], now),
            )
            raw_sources = product["alias_sources"].get(alias) or product["alias_sources"].get(normalize_knowledge_value(alias)) or [product["source_id"]]
            if isinstance(raw_sources, str):
                raw_sources = [raw_sources]
            for source_id in raw_sources:
                source_id = _clean(source_id, 120, required=True)
                self._require_source(conn, source_id)
                conn.execute(
                    "INSERT INTO product_alias_sources(product_alias_source_id,alias_id,source_id,created_at) VALUES (?,?,?,?)",
                    (_stable_id("product_alias_source", alias_id, source_id), alias_id, source_id, now),
                )
        for claim in claims:
            conn.execute(
                """
                INSERT INTO product_claims(
                    claim_id,product_id,claim_type,claim_value,normalized_value,source_id,
                    source_text,status,review_state,conflict_group_id,dosage_structured_json,
                    confidence,version,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (_stable_id("product_claim", product["product_id"], claim["claim_type"], claim["normalized_value"], claim["source_id"]),
                 product["product_id"], claim["claim_type"], claim["claim_value"], claim["normalized_value"],
                 claim["source_id"], claim["source_text"], claim["status"], claim["review_state"],
                 claim["conflict_group_id"] or None, _json(claim["dosage_structured"]),
                 claim["confidence"], product["version"], now, now),
            )
        snapshot = {key: value for key, value in product.items() if key != "claims"}
        snapshot["claims"] = claims
        conn.execute(
            "INSERT INTO product_versions(product_version_id,product_id,version,snapshot_json,source_id,content_hash,created_at) VALUES (?,?,?,?,?,?,?)",
            (_stable_id("product_version", product["product_id"], str(product["version"])), product["product_id"],
             product["version"], _json(snapshot), product["source_id"], product["content_hash"], now),
        )
        return _product_from_row(conn.execute(
            "SELECT * FROM product_catalog WHERE product_id=?", (product["product_id"],)
        ).fetchone())

    def _prepare_solution(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("solution_object_required")
        source_id = _clean(payload.get("source_id"), 120, required=True)
        name = _clean(payload.get("solution_name"), 240, required=True)
        normalized_name = normalize_knowledge_value(name)
        solution_id = _clean(payload.get("solution_id"), 120) or _stable_id("solution", source_id, normalized_name)
        version = int(payload.get("version") or 1)
        if version < 1:
            raise ValueError("solution_version_invalid")
        raw_items = payload.get("items") or []
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("solution_items_required")
        items: list[dict[str, Any]] = []
        unresolved_items: list[dict[str, Any]] = []
        sequences: set[int] = set()
        for index, raw in enumerate(raw_items, 1):
            if not isinstance(raw, dict):
                raise ValueError("solution_item_object_required")
            sequence = int(raw.get("sequence") or index)
            if sequence < 1 or sequence in sequences:
                raise ValueError("solution_item_sequence_invalid")
            sequences.add(sequence)
            common = {
                "role": _clean(raw.get("role"), 120),
                "dosage": _clean(raw.get("dosage"), 500),
                "sequence": sequence,
                "timing": _clean(raw.get("timing"), 300),
                "notes": _clean(raw.get("notes"), 1000),
            }
            product_id = _clean(raw.get("product_id"), 120)
            if product_id:
                items.append({"product_id": product_id, **common})
            else:
                raw_name = _clean(raw.get("raw_product_name") or raw.get("product_name"), 240, required=True)
                unresolved_items.append({
                    "raw_product_name": raw_name,
                    "normalized_product_name": normalize_knowledge_value(raw_name),
                    "source_id": _clean(raw.get("source_id") or source_id, 120, required=True),
                    "review_state": _review_state(
                        raw.get("review_state"), KnowledgeReviewState.REVIEW_REQUIRED
                    ).value,
                    **common,
                })
        items.sort(key=lambda item: item["sequence"])
        unresolved_items.sort(key=lambda item: item["sequence"])
        status = _status(payload.get("status"))
        review_default = (
            KnowledgeReviewState.VERIFIED
            if status == KnowledgeStatus.ACTIVE
            else KnowledgeReviewState.UNREVIEWED
        )
        if unresolved_items:
            status = KnowledgeStatus.DRAFT
            review_default = KnowledgeReviewState.REVIEW_REQUIRED
        source_links = payload.get("source_links") or [{
            "source_id": source_id,
            "source_solution_name": name,
            "source_version": str(payload.get("source_version") or version),
            "source_text": payload.get("source_text") or "",
            "source_hash": payload.get("source_hash") or "",
            "status": status.value,
            "review_state": str(payload.get("review_state") or review_default.value),
        }]
        if not isinstance(source_links, list) or not all(isinstance(item, dict) for item in source_links):
            raise ValueError("solution_source_links_required")
        prepared = {
            "solution_id": solution_id,
            "solution_name": name,
            "normalized_name": normalized_name,
            "crops": _strings(payload.get("crops")),
            "target_problems": _strings(payload.get("target_problems")),
            "growth_stages": _strings(payload.get("growth_stages")),
            "region_constraints": _strings(payload.get("region_constraints")),
            "conditions": _object(payload.get("conditions")),
            "expected_goal": _clean(payload.get("expected_goal"), 1500),
            "precautions": _strings(payload.get("precautions")),
            "risk_notes": _strings(payload.get("risk_notes")),
            "notes": _clean(payload.get("notes"), 3000),
            "status": status.value,
            "review_state": _review_state(payload.get("review_state"), review_default).value,
            "source_id": source_id,
            "version": version,
            "items": items,
            "unresolved_items": unresolved_items,
            "source_links": [
                {
                    "source_id": _clean(item.get("source_id"), 120, required=True),
                    "source_solution_name": _clean(item.get("source_solution_name") or name, 240),
                    "source_version": _clean(item.get("source_version"), 120),
                    "source_text": _clean(item.get("source_text"), 4000),
                    "source_hash": _clean(item.get("source_hash"), 128),
                    "status": _status(item.get("status"), status).value,
                    "review_state": _review_state(item.get("review_state"), review_default).value,
                }
                for item in source_links
            ],
        }
        prepared["content_hash"] = _hash(prepared)
        return prepared

    def _upsert_solution_conn(self, conn: sqlite3.Connection, payload: dict[str, Any]) -> Solution:
        solution = self._prepare_solution(payload)
        self._require_source(conn, solution["source_id"])
        for link in solution["source_links"]:
            self._require_source(conn, link["source_id"])
        for item in solution["unresolved_items"]:
            self._require_source(conn, item["source_id"])
        for item in solution["items"]:
            if conn.execute("SELECT 1 FROM product_catalog WHERE product_id=?", (item["product_id"],)).fetchone() is None:
                raise ValueError("solution_product_not_found")
        existing = conn.execute(
            "SELECT * FROM solution_templates WHERE solution_id=?", (solution["solution_id"],)
        ).fetchone()
        if existing is not None and existing["content_hash"] == solution["content_hash"]:
            items = conn.execute(
                "SELECT * FROM solution_items WHERE solution_id=? ORDER BY sequence,solution_item_id",
                (solution["solution_id"],),
            ).fetchall()
            unresolved = conn.execute(
                "SELECT * FROM solution_unresolved_items WHERE solution_id=? ORDER BY sequence,unresolved_item_id",
                (solution["solution_id"],),
            ).fetchall()
            return _solution_from_row(existing, items, unresolved)
        if existing is not None and solution["version"] <= existing["version"]:
            raise ValueError("solution_version_must_increase")
        now = utc_now_iso()
        values = (
            solution["solution_name"], solution["normalized_name"], _json(solution["crops"]),
            _json(solution["target_problems"]), _json(solution["growth_stages"]),
            _json(solution["region_constraints"]), _json(solution["conditions"]), solution["expected_goal"],
            _json(solution["precautions"]), _json(solution["risk_notes"]), solution["notes"],
            solution["status"], solution["source_id"], solution["review_state"],
            solution["version"], solution["content_hash"], now,
        )
        if existing is None:
            conn.execute(
                """
                INSERT INTO solution_templates(
                    solution_name,normalized_name,crops_json,target_problems_json,growth_stages_json,
                    region_constraints_json,conditions_json,expected_goal,precautions_json,risk_notes_json,
                    notes,status,source_id,review_state,version,content_hash,created_at,updated_at,solution_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                values + (now, solution["solution_id"]),
            )
        else:
            conn.execute(
                """
                UPDATE solution_templates SET
                    solution_name=?,normalized_name=?,crops_json=?,target_problems_json=?,growth_stages_json=?,
                    region_constraints_json=?,conditions_json=?,expected_goal=?,precautions_json=?,risk_notes_json=?,
                    notes=?,status=?,source_id=?,review_state=?,version=?,content_hash=?,updated_at=? WHERE solution_id=?
                """,
                values + (solution["solution_id"],),
            )
            conn.execute("DELETE FROM solution_items WHERE solution_id=?", (solution["solution_id"],))
            conn.execute("DELETE FROM solution_unresolved_items WHERE solution_id=?", (solution["solution_id"],))
            conn.execute("DELETE FROM solution_source_links WHERE solution_id=?", (solution["solution_id"],))
        for link in solution["source_links"]:
            conn.execute(
                """
                INSERT INTO solution_source_links(
                    solution_source_link_id,solution_id,source_id,source_solution_name,source_version,
                    source_text,source_hash,status,review_state,imported_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _stable_id("solution_source", solution["solution_id"], link["source_id"]),
                    solution["solution_id"], link["source_id"], link["source_solution_name"],
                    link["source_version"], link["source_text"], link["source_hash"],
                    link["status"], link["review_state"], now, now,
                ),
            )
        for item in solution["items"]:
            conn.execute(
                """
                INSERT INTO solution_items(
                    solution_item_id,solution_id,product_id,role,dosage,sequence,timing,notes,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (_stable_id("solution_item", solution["solution_id"], str(item["sequence"])), solution["solution_id"],
                 item["product_id"], item["role"], item["dosage"], item["sequence"], item["timing"], item["notes"], now, now),
            )
        for item in solution["unresolved_items"]:
            conn.execute(
                """
                INSERT INTO solution_unresolved_items(
                    unresolved_item_id,solution_id,raw_product_name,normalized_product_name,
                    role,dosage,sequence,timing,notes,source_id,review_state,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _stable_id("solution_unresolved", solution["solution_id"], str(item["sequence"])),
                    solution["solution_id"], item["raw_product_name"], item["normalized_product_name"],
                    item["role"], item["dosage"], item["sequence"], item["timing"], item["notes"],
                    item["source_id"], item["review_state"], now, now,
                ),
            )
        conn.execute(
            "INSERT INTO solution_versions(solution_version_id,solution_id,version,snapshot_json,source_id,content_hash,created_at) VALUES (?,?,?,?,?,?,?)",
            (_stable_id("solution_version", solution["solution_id"], str(solution["version"])), solution["solution_id"],
             solution["version"], _json(solution), solution["source_id"], solution["content_hash"], now),
        )
        row = conn.execute("SELECT * FROM solution_templates WHERE solution_id=?", (solution["solution_id"],)).fetchone()
        items = conn.execute("SELECT * FROM solution_items WHERE solution_id=? ORDER BY sequence", (solution["solution_id"],)).fetchall()
        unresolved = conn.execute(
            "SELECT * FROM solution_unresolved_items WHERE solution_id=? ORDER BY sequence",
            (solution["solution_id"],),
        ).fetchall()
        return _solution_from_row(row, items, unresolved)

    @staticmethod
    def _require_source(conn: sqlite3.Connection, source_id: str) -> None:
        if conn.execute("SELECT 1 FROM knowledge_sources WHERE source_id=?", (source_id,)).fetchone() is None:
            raise ValueError("knowledge_source_not_found")


class _ReplyKnowledgeSnapshot:
    """In-memory repository facade owned by exactly one reply build."""

    def __init__(
        self,
        product_rows: list[sqlite3.Row],
        solution_rows: list[sqlite3.Row],
        item_rows: list[sqlite3.Row],
        linked_rows: list[sqlite3.Row],
    ) -> None:
        self._products = [_product_from_row(row) for row in product_rows]
        self._linked_rows = {row["product_id"]: row for row in linked_rows}
        self._parsed_products = {product.product_id: product for product in self._products}
        self._solution_rows = solution_rows
        self._items: dict[str, list[sqlite3.Row]] = {}
        for row in item_rows:
            self._items.setdefault(row["solution_id"], []).append(row)
        self._solutions: list[Solution] | None = None

    def list_active_products(
        self,
        *,
        limit: int = 200,
        knowledge_kinds: Iterable[str | KnowledgeKind] | None = None,
    ) -> list[Product]:
        kinds = tuple(str(item) for item in (knowledge_kinds or ()))
        products = self._products
        if kinds:
            products = [p for p in products if p.knowledge_kind.value in kinds]
        return products[: max(1, min(int(limit), 1000))]

    def get_product(self, product_id: str) -> Product | None:
        # SQL ``WHERE product_id = NULL`` never matches, even in damaged legacy data.
        if product_id is None:
            return None
        if product_id not in self._parsed_products:
            row = self._linked_rows.get(product_id)
            if row is None:
                return None
            self._parsed_products[product_id] = _product_from_row(row)
        return self._parsed_products[product_id]

    def rank_matching_solutions(
        self,
        *,
        crop: str = "",
        problem: str = "",
        stage: str = "",
        region: str = "",
        condition_facts: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[tuple[int, Solution]]:
        if self._solutions is None:
            self._solutions = [
                _solution_from_row(row, self._items.get(row["solution_id"], []), [])
                for row in self._solution_rows
            ]
        ranked: list[tuple[int, Solution]] = []
        for solution in self._solutions:
            score = _solution_score(
                solution, crop=crop, problem=problem, stage=stage, region=region,
                condition_facts=condition_facts or {},
            )
            if score is not None:
                ranked.append((score, solution))
        ranked.sort(key=lambda item: (-item[0], normalize_knowledge_value(item[1].solution_name)))
        return ranked[: max(1, min(int(limit), 500))]


def _source_from_row(row: sqlite3.Row) -> KnowledgeSource:
    return KnowledgeSource(
        source_id=row["source_id"],
        source_type=KnowledgeSourceType(row["source_type"]),
        title=row["title"],
        source_reference=row["source_reference"],
        source_hash=row["source_hash"],
        metadata=_object(row["metadata_json"]),
        status=KnowledgeStatus(row["status"]),
        imported_at=row["imported_at"],
        updated_at=row["updated_at"],
    )


def _product_from_row(row: sqlite3.Row) -> Product:
    return Product(
        product_id=row["product_id"],
        product_name=row["product_name"],
        knowledge_kind=KnowledgeKind(row["knowledge_kind"]),
        aliases=tuple(json.loads(row["aliases_json"] or "[]")),
        category=row["category"],
        formulation=row["formulation"],
        active_ingredients=tuple(json.loads(row["active_ingredients_json"] or "[]")),
        ingredient_content=row["ingredient_content"],
        specifications=_object(row["specifications_json"]),
        key_functions=tuple(json.loads(row["key_functions_json"] or "[]")),
        applicable_crops=tuple(json.loads(row["applicable_crops_json"] or "[]")),
        applicable_stages=tuple(json.loads(row["applicable_stages_json"] or "[]")),
        applicable_problems=tuple(json.loads(row["applicable_problems_json"] or "[]")),
        usage_methods=tuple(json.loads(row["usage_methods_json"] or "[]")),
        dosage_text=row["dosage_text"],
        dosage_structured=_object(row["dosage_structured_json"]),
        precautions=tuple(json.loads(row["precautions_json"] or "[]")),
        incompatibilities=tuple(json.loads(row["incompatibilities_json"] or "[]")),
        risk_notes=tuple(json.loads(row["risk_notes_json"] or "[]")),
        notes=row["notes"],
        status=KnowledgeStatus(row["status"]),
        review_state=KnowledgeReviewState(row["review_state"]),
        source_id=row["source_id"],
        version=int(row["version"]),
        content_hash=row["content_hash"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _claim_from_row(row: sqlite3.Row) -> ProductClaim:
    return ProductClaim(
        claim_id=row["claim_id"], product_id=row["product_id"],
        claim_type=ProductClaimType(row["claim_type"]), claim_value=row["claim_value"],
        normalized_value=row["normalized_value"], source_id=row["source_id"],
        source_text=row["source_text"], status=KnowledgeStatus(row["status"]),
        review_state=KnowledgeReviewState(row["review_state"]),
        conflict_group_id=row["conflict_group_id"] or "",
        dosage_structured=_object(row["dosage_structured_json"]),
        confidence=float(row["confidence"]), version=int(row["version"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _solution_from_row(
    row: sqlite3.Row,
    item_rows: Iterable[sqlite3.Row],
    unresolved_rows: Iterable[sqlite3.Row],
) -> Solution:
    items = tuple(
        SolutionItem(
            solution_item_id=item["solution_item_id"], solution_id=item["solution_id"],
            product_id=item["product_id"], role=item["role"], dosage=item["dosage"],
            sequence=int(item["sequence"]), timing=item["timing"], notes=item["notes"],
            created_at=item["created_at"], updated_at=item["updated_at"],
        )
        for item in item_rows
    )
    unresolved_items = tuple(
        UnresolvedSolutionItem(
            unresolved_item_id=item["unresolved_item_id"], solution_id=item["solution_id"],
            raw_product_name=item["raw_product_name"],
            normalized_product_name=item["normalized_product_name"], role=item["role"],
            dosage=item["dosage"], sequence=int(item["sequence"]), timing=item["timing"],
            notes=item["notes"], source_id=item["source_id"],
            review_state=KnowledgeReviewState(item["review_state"]),
            created_at=item["created_at"], updated_at=item["updated_at"],
        )
        for item in unresolved_rows
    )
    return Solution(
        solution_id=row["solution_id"], solution_name=row["solution_name"],
        crops=tuple(json.loads(row["crops_json"] or "[]")),
        target_problems=tuple(json.loads(row["target_problems_json"] or "[]")),
        growth_stages=tuple(json.loads(row["growth_stages_json"] or "[]")),
        region_constraints=tuple(json.loads(row["region_constraints_json"] or "[]")),
        conditions=_object(row["conditions_json"]), expected_goal=row["expected_goal"],
        precautions=tuple(json.loads(row["precautions_json"] or "[]")),
        risk_notes=tuple(json.loads(row["risk_notes_json"] or "[]")), notes=row["notes"],
        status=KnowledgeStatus(row["status"]), review_state=KnowledgeReviewState(row["review_state"]),
        source_id=row["source_id"], version=int(row["version"]),
        content_hash=row["content_hash"], created_at=row["created_at"], updated_at=row["updated_at"],
        items=items, unresolved_items=unresolved_items,
    )


def _solution_score(
    solution: Solution,
    *,
    crop: str,
    problem: str,
    stage: str,
    region: str,
    condition_facts: dict[str, Any],
) -> int | None:
    score = 0
    checks = (
        (crop, solution.crops, 30),
        (problem, solution.target_problems, 40),
        (stage, solution.growth_stages, 20),
    )
    for query, allowed, points in checks:
        if not query:
            if allowed:
                continue
            continue
        if allowed:
            keys = {normalize_knowledge_value(value) for value in allowed}
            if normalize_knowledge_value(query) not in keys:
                return None
            score += points
    if solution.region_constraints:
        if not region:
            return None
        region_key = normalize_knowledge_value(region)
        if not any(
            region_key.startswith(normalize_knowledge_value(value))
            or normalize_knowledge_value(value).startswith(region_key)
            for value in solution.region_constraints
        ):
            return None
        score += 10
    required = solution.conditions.get("required", {})
    if required:
        if not isinstance(required, dict):
            return None
        for key, expected in required.items():
            actual = condition_facts.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return None
            elif actual != expected:
                return None
        score += len(required) * 5
    return score
