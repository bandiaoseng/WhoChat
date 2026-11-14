from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from whochat.core.models import utc_now_iso
from whochat.storage.product_knowledge import ProductKnowledgeRepository, normalize_knowledge_value


SOURCE_PRIORITY = {
    "company_document": 700,
    "feishu": 600,
    "lexiang": 550,
    "manual": 500,
    "historical_solution": 300,
    "verified_chat": 250,
    "other": 100,
}
REVIEW_ORDER = {"verified": 0, "unreviewed": 1, "review_required": 2, "conflicting": 3}
LIST_FIELDS = (
    "key_functions", "applicable_crops", "applicable_stages", "applicable_problems",
    "usage_methods", "precautions", "incompatibilities", "risk_notes",
)
SCALAR_FIELDS = (
    "knowledge_kind", "category", "formulation", "ingredient_content", "dosage_text", "notes"
)


@dataclass(frozen=True)
class VerifiedKnowledgeImportPreview:
    valid: bool
    source_format: str
    bundle: dict[str, list[dict[str, Any]]]
    actions: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    preview_hash: str = ""

    @property
    def counts(self) -> dict[str, int]:
        counts = {
            "sources": len(self.bundle.get("sources", [])),
            "products": len(self.bundle.get("products", [])),
            "solutions": len(self.bundle.get("solutions", [])),
            "new_products": 0,
            "updated_products": 0,
            "duplicate_products": 0,
            "conflict_products": 0,
            "draft_products": 0,
            "new_solutions": 0,
            "updated_solutions": 0,
            "unresolved_solution_items": 0,
        }
        for action in self.actions:
            entity = action.get("entity")
            change = action.get("action")
            if entity == "product":
                if change == "create":
                    counts["new_products"] += 1
                elif change == "update":
                    counts["updated_products"] += 1
                elif change == "unchanged":
                    counts["duplicate_products"] += 1
                if action.get("review_state") == "conflicting":
                    counts["conflict_products"] += 1
                if action.get("status") == "draft":
                    counts["draft_products"] += 1
            elif entity == "solution":
                if change == "create":
                    counts["new_solutions"] += 1
                elif change == "update":
                    counts["updated_solutions"] += 1
                counts["unresolved_solution_items"] += int(action.get("unresolved_items") or 0)
        return counts


class VerifiedKnowledgeImportService:
    """Conservative, source-aware v24 import planning and atomic commit."""

    def __init__(self, repository: ProductKnowledgeRepository) -> None:
        self.repository = repository

    def preview_json(self, text: str) -> VerifiedKnowledgeImportPreview:
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            return VerifiedKnowledgeImportPreview(
                False, "json", _empty_bundle(), errors=(f"invalid_json:{getattr(exc, 'msg', exc)}",)
            )
        if not isinstance(value, dict):
            return VerifiedKnowledgeImportPreview(False, "json", _empty_bundle(), errors=("json_bundle_object_required",))
        return self.preview_bundle(value, source_format="json")

    def preview_csv(self, text: str, *, entity_type: str) -> VerifiedKnowledgeImportPreview:
        if entity_type not in {"sources", "products", "solutions"}:
            return VerifiedKnowledgeImportPreview(False, "csv", _empty_bundle(), errors=("csv_entity_type_invalid",))
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except csv.Error as exc:
            return VerifiedKnowledgeImportPreview(False, "csv", _empty_bundle(), errors=(f"invalid_csv:{exc}",))
        if not rows:
            return VerifiedKnowledgeImportPreview(False, "csv", _empty_bundle(), errors=("csv_has_no_rows",))
        bundle = _empty_bundle()
        try:
            bundle[entity_type] = [_csv_row(row, entity_type) for row in rows]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return VerifiedKnowledgeImportPreview(False, "csv", _empty_bundle(), errors=(f"invalid_csv_row:{exc}",))
        return self.preview_bundle(bundle, source_format="csv")

    def preview_xlsx(self, path: Path) -> VerifiedKnowledgeImportPreview:
        try:
            sheets = _read_xlsx(path)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            return VerifiedKnowledgeImportPreview(False, "xlsx", _empty_bundle(), errors=(f"invalid_xlsx:{exc}",))
        bundle = _empty_bundle()
        for key in bundle:
            rows = sheets.get(key, [])
            try:
                bundle[key] = [_csv_row(row, key) for row in rows]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return VerifiedKnowledgeImportPreview(False, "xlsx", _empty_bundle(), errors=(f"invalid_{key}_sheet:{exc}",))
        if not any(bundle.values()):
            return VerifiedKnowledgeImportPreview(False, "xlsx", bundle, errors=("xlsx_requires_sources_products_or_solutions_sheet",))
        return self.preview_bundle(bundle, source_format="xlsx")

    def preview_bundle(self, raw_bundle: dict[str, Any], *, source_format: str = "mapping") -> VerifiedKnowledgeImportPreview:
        errors: list[str] = []
        warnings: list[str] = []
        clean = _empty_bundle()
        for key in clean:
            value = raw_bundle.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                errors.append(f"{key}_objects_required")
            else:
                clean[key] = [dict(item) for item in value]
        if errors:
            return VerifiedKnowledgeImportPreview(False, source_format, clean, errors=tuple(errors))

        source_types: dict[str, str] = {}
        for source in clean["sources"]:
            source_id = str(source.get("source_id") or "").strip()
            source_type = str(source.get("source_type") or "manual").strip()
            if not source_id:
                errors.append(f"source:{source.get('title', '?')}:source_id_required")
            elif source_id in source_types:
                errors.append(f"source:{source_id}:duplicate_source_id")
            elif source_type not in SOURCE_PRIORITY:
                errors.append(f"source:{source_id}:source_type_invalid")
            elif not str(source.get("title") or "").strip():
                errors.append(f"source:{source_id}:title_required")
            else:
                source_types[source_id] = source_type

        for collection in (clean["products"], clean["solutions"]):
            for item in collection:
                source_id = str(item.get("source_id") or "").strip()
                if not source_id:
                    errors.append(f"entity:{item.get('product_name') or item.get('solution_name') or '?'}:source_id_required")
                elif source_id not in source_types:
                    source = self.repository.get_source(source_id)
                    if source is None:
                        errors.append(f"entity:{item.get('product_name') or item.get('solution_name') or '?'}:knowledge_source_not_found:{source_id}")
                    else:
                        source_types[source_id] = source.source_type.value
        if errors:
            return VerifiedKnowledgeImportPreview(False, source_format, clean, errors=tuple(errors))

        try:
            products, product_actions, product_warnings = self._canonicalize_products(clean["products"], source_types)
            solutions, solution_actions, solution_warnings = self._canonicalize_solutions(
                clean["solutions"], source_types, products
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return VerifiedKnowledgeImportPreview(False, source_format, clean, errors=(str(exc),))
        warnings.extend(product_warnings)
        warnings.extend(solution_warnings)
        bundle = {"sources": clean["sources"], "products": products, "solutions": solutions}
        preview_hash = _hash(bundle)
        return VerifiedKnowledgeImportPreview(
            True,
            source_format,
            bundle,
            actions=tuple(product_actions + solution_actions),
            warnings=tuple(warnings),
            preview_hash=preview_hash,
        )

    def commit(self, preview: VerifiedKnowledgeImportPreview) -> dict[str, Any]:
        if not preview.valid:
            raise ValueError("cannot_commit_invalid_preview")
        if _hash(preview.bundle) != preview.preview_hash:
            raise ValueError("preview_bundle_changed_after_validation")
        result = {"sources": 0, "products": 0, "solutions": 0, "preview_hash": preview.preview_hash}
        now = utc_now_iso()
        import_run_id = _stable_id("knowledge_import", preview.preview_hash)
        with self.repository.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in preview.bundle["sources"]:
                self.repository._upsert_source_conn(conn, item)
                result["sources"] += 1
            for item in preview.bundle["products"]:
                self.repository._upsert_product_conn(conn, item)
                result["products"] += 1
            for item in preview.bundle["solutions"]:
                self.repository._upsert_solution_conn(conn, item)
                result["solutions"] += 1
            conn.execute(
                """
                INSERT INTO knowledge_import_runs(
                    import_run_id,source_format,preview_hash,preview_json,result_json,status,created_at,committed_at
                ) VALUES (?,?,?,?,?,'committed',?,?)
                ON CONFLICT(preview_hash) DO UPDATE SET
                    result_json=excluded.result_json,status='committed',committed_at=excluded.committed_at
                """,
                (
                    import_run_id, preview.source_format, preview.preview_hash,
                    json.dumps({"counts": preview.counts, "actions": preview.actions}, ensure_ascii=False, sort_keys=True),
                    json.dumps(result, ensure_ascii=False, sort_keys=True), now, now,
                ),
            )
        return result

    def _canonicalize_products(
        self,
        records: list[dict[str, Any]],
        source_types: dict[str, str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        name_to_group: dict[str, tuple[str, dict[str, Any]]] = {}
        warnings: list[str] = []
        for raw in records:
            name = _text(raw.get("product_name"), required=True)
            name_key = normalize_knowledge_value(name)
            explicit_id = _text(raw.get("product_id"))
            existing = self.repository.get_product(explicit_id) if explicit_id else None
            if existing is None and not explicit_id:
                candidate = self.repository.find_product_by_name_or_alias(name)
                existing = candidate if candidate is not None and _identity_compatible(_product_payload(candidate), raw) else None
            if existing is not None:
                product_id = existing.product_id
            elif explicit_id:
                product_id = explicit_id
            elif name_key in name_to_group and _identity_compatible(name_to_group[name_key][1], raw):
                product_id = name_to_group[name_key][0]
            else:
                identity = _identity_signature(raw)
                product_id = _stable_id("product", name_key, identity)
            if name_key in name_to_group and name_to_group[name_key][0] != product_id:
                warnings.append(f"same_name_distinct_products:{name}")
            name_to_group.setdefault(name_key, (product_id, dict(raw)))
            item = dict(raw)
            item["product_id"] = product_id
            groups.setdefault(product_id, []).append(item)

        products: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        alias_owner: dict[str, str] = {}
        for product_id, items in groups.items():
            plan, conflict_fields = self._merge_product_group(product_id, items, source_types)
            for alias in plan.get("aliases", []):
                key = normalize_knowledge_value(alias)
                owner = alias_owner.get(key)
                if owner and owner != product_id:
                    raise ValueError(f"product_alias_collision_in_bundle:{alias}")
                alias_owner[key] = product_id
            action = self._versioned_action("product", plan)
            products.append(plan)
            actions.append({
                "entity": "product", "id": product_id, "name": plan["product_name"],
                "action": action, "status": plan["status"], "review_state": plan["review_state"],
                "conflicts": conflict_fields,
            })
        return products, actions, warnings

    def _merge_product_group(
        self,
        product_id: str,
        records: list[dict[str, Any]],
        source_types: dict[str, str],
    ) -> tuple[dict[str, Any], list[str]]:
        existing = self.repository.get_product(product_id)
        ordered = sorted(
            records,
            key=lambda item: (-SOURCE_PRIORITY.get(source_types.get(str(item.get("source_id")), "other"), 0), records.index(item)),
        )
        primary = dict(ordered[0])
        base: dict[str, Any] = {}
        if existing is not None:
            base = _product_payload(existing)
            for key, value in base.items():
                primary.setdefault(key, value)
            # A later source may identify the product by an existing alias.
            # Keep the already established formal catalog name unless the
            # incoming record explicitly targets that same normalized name.
            if normalize_knowledge_value(primary.get("product_name")) != normalize_knowledge_value(existing.product_name):
                primary["product_name"] = existing.product_name
        primary["product_id"] = product_id

        for field in SCALAR_FIELDS:
            primary[field] = next(
                (_text(item.get(field)) for item in ordered if _text(item.get(field))),
                _text(base.get(field) or primary.get(field)),
            )
        primary["specifications"] = next(
            (item.get("specifications") for item in ordered if item.get("specifications")),
            base.get("specifications") or primary.get("specifications") or {},
        )
        primary["active_ingredients"] = _unique_objects(
            [value for item in ordered for value in _as_list(item.get("active_ingredients"))]
            + list(base.get("active_ingredients") or primary.get("active_ingredients") or [])
        )
        for field in LIST_FIELDS:
            primary[field] = _unique_text(
                [value for item in ordered for value in _as_list(item.get(field))]
                + list(base.get(field) or primary.get(field) or [])
            )
        primary["aliases"] = _unique_text(
            [value for item in ordered for value in _as_list(item.get("aliases"))]
            + list(base.get("aliases") or primary.get("aliases") or [])
        )

        existing_links = {item["source_id"]: item for item in self.repository.get_product_source_links(product_id)} if existing else {}
        incoming_sources = {str(item.get("source_id")) for item in ordered}
        links = {key: _link_payload(value, product=True) for key, value in existing_links.items() if key not in incoming_sources}
        alias_sources: dict[str, list[str]] = {}
        claims: list[dict[str, Any]] = []
        if existing is not None:
            audit = self.repository.inspect_product(existing.product_name) or {}
            for alias in audit.get("aliases", []):
                evidence = str(alias.get("evidence_source_ids") or alias.get("source_id") or "").split(",")
                alias_sources[str(alias["alias"])] = [item for item in evidence if item and item not in incoming_sources]
            for claim in audit.get("claims", []):
                source_id = str(claim.get("source_id") or "")
                if source_id not in incoming_sources:
                    claims.append(_claim_payload(claim))

        for item in ordered:
            source_id = str(item["source_id"])
            review_state = str(item.get("review_state") or ("verified" if item.get("status") == "active" else "unreviewed"))
            links[source_id] = {
                "source_id": source_id,
                "source_product_name": _text(item.get("product_name")),
                "source_version": _text(item.get("source_version") or item.get("version")),
                "source_text": _text(item.get("source_text"), limit=4000),
                "source_hash": _text(item.get("source_hash"), limit=128),
                "status": str(item.get("status") or "draft"),
                "review_state": review_state,
            }
            for alias in _as_list(item.get("aliases")):
                alias_sources.setdefault(str(alias), []).append(source_id)
            claims.extend(_claims_for_record(item, review_state=review_state))

        conflict_fields: list[str] = []
        dosage_values = _distinct([_text(item.get("dosage_text")) for item in ordered if _text(item.get("dosage_text"))])
        ingredient_values = _distinct([_text(item.get("ingredient_content")) for item in ordered if _text(item.get("ingredient_content"))])
        formulation_values = _distinct([_text(item.get("formulation")) for item in ordered if _text(item.get("formulation"))])
        if len(dosage_values) > 1:
            conflict_fields.append("dosage")
            conflict_id = _stable_id("knowledge_conflict", product_id, "dosage", *dosage_values)
            for claim in claims:
                if claim.get("claim_type") == "dosage":
                    claim.update(status="draft", review_state="conflicting", conflict_group_id=conflict_id)
        if len(ingredient_values) > 1:
            conflict_fields.append("ingredient_content")
        if len(formulation_values) > 1:
            conflict_fields.append("formulation")

        review_states = [str(item.get("review_state") or ("verified" if item.get("status") == "active" else "unreviewed")) for item in ordered]
        review_state = max(review_states, key=lambda value: REVIEW_ORDER.get(value, 3), default="unreviewed")
        if conflict_fields:
            review_state = "conflicting" if "dosage" in conflict_fields else "review_required"
        statuses = {str(item.get("status") or "draft") for item in ordered}
        status = "active" if statuses == {"active"} and review_state == "verified" else "draft"
        primary.update({
            "product_id": product_id,
            "source_id": ordered[0]["source_id"],
            "source_text": _text(ordered[0].get("source_text"), limit=4000),
            "status": status,
            "review_state": review_state,
            "source_links": [links[key] for key in sorted(links)],
            "alias_sources": {
                key: sorted(set(alias_sources[key]))
                for key in sorted(alias_sources) if alias_sources[key]
            },
            "claims": _dedupe_claims(claims),
        })
        if not primary.get("dosage_structured") and primary.get("dosage_text"):
            primary["dosage_structured"] = parse_dosage(primary["dosage_text"])
        # Keep Preview hashes stable before and after the first commit. SQLite
        # round-trips an absent structured dosage as {}, so normalize the
        # incoming empty value to the same representation before hashing.
        primary["dosage_structured"] = primary.get("dosage_structured") or {}
        primary["version"] = existing.version if existing else max([int(item.get("version") or 1) for item in ordered] or [1])
        self.repository._prepare_product(primary)
        return primary, conflict_fields

    def _canonicalize_solutions(
        self,
        records: list[dict[str, Any]],
        source_types: dict[str, str],
        products: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        lookup: dict[str, str] = {}
        planned: dict[str, dict[str, Any]] = {}
        for product in products:
            planned[product["product_id"]] = product
            lookup[normalize_knowledge_value(product["product_name"])] = product["product_id"]
            for alias in product.get("aliases", []):
                lookup[normalize_knowledge_value(alias)] = product["product_id"]
        solutions: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        warnings: list[str] = []
        for raw in records:
            item = dict(raw)
            name = _text(item.get("solution_name"), required=True)
            existing = self.repository.get_solution(_text(item.get("solution_id"))) if item.get("solution_id") else None
            solution_id = existing.solution_id if existing else (_text(item.get("solution_id")) or _stable_id("solution", normalize_knowledge_value(name)))
            resolved_items: list[dict[str, Any]] = []
            for index, raw_item in enumerate(_as_list(item.get("items")), 1):
                if not isinstance(raw_item, dict):
                    raise ValueError(f"solution:{name}:solution_item_object_required")
                current = dict(raw_item)
                product_id = _text(current.get("product_id"))
                raw_name = _text(current.get("product_name") or current.get("raw_product_name"))
                if not product_id and raw_name:
                    product_id = lookup.get(normalize_knowledge_value(raw_name), "")
                    if not product_id:
                        found = self.repository.find_product_by_name_or_alias(raw_name)
                        product_id = found.product_id if found else ""
                if product_id:
                    current["product_id"] = product_id
                    current.pop("raw_product_name", None)
                else:
                    current.pop("product_id", None)
                    current["raw_product_name"] = raw_name or "未识别产品"
                    current["source_id"] = current.get("source_id") or item.get("source_id")
                current["sequence"] = int(current.get("sequence") or index)
                resolved_items.append(current)
            unresolved = [entry for entry in resolved_items if not entry.get("product_id")]
            review_state = str(item.get("review_state") or ("verified" if item.get("status") == "active" else "unreviewed"))
            if unresolved:
                item["status"] = "draft"
                review_state = "review_required"
                warnings.append(f"solution_unresolved_items:{name}:{len(unresolved)}")
            else:
                for entry in resolved_items:
                    product = planned.get(entry["product_id"])
                    if product is None:
                        found = self.repository.get_product(entry["product_id"])
                        product = _product_payload(found) if found else None
                    if product is None:
                        raise ValueError(f"solution:{name}:product_not_found:{entry['product_id']}")
                    if product.get("status") != "active" or product.get("review_state") != "verified":
                        item["status"] = "draft"
                        review_state = "review_required"
            item.update({
                "solution_id": solution_id,
                "items": resolved_items,
                "review_state": review_state,
                "source_links": item.get("source_links") or [{
                    "source_id": item["source_id"],
                    "source_solution_name": name,
                    "source_version": _text(item.get("source_version") or item.get("version")),
                    "source_text": _text(item.get("source_text"), limit=4000),
                    "source_hash": _text(item.get("source_hash"), limit=128),
                    "status": str(item.get("status") or "draft"),
                    "review_state": review_state,
                }],
                "version": existing.version if existing else int(item.get("version") or 1),
            })
            self.repository._prepare_solution(item)
            action = self._versioned_action("solution", item)
            solutions.append(item)
            actions.append({
                "entity": "solution", "id": solution_id, "name": name, "action": action,
                "status": item.get("status") or "draft", "review_state": review_state,
                "unresolved_items": len(unresolved),
            })
        return solutions, actions, warnings

    def _versioned_action(self, entity: str, payload: dict[str, Any]) -> str:
        action = self.repository.preview_action(entity, payload)
        if action == "invalid_version":
            if entity == "product":
                existing = self.repository.get_product(payload["product_id"])
            else:
                existing = self.repository.get_solution(payload["solution_id"])
            if existing is None:
                raise ValueError(f"{entity}_version_invalid")
            payload["version"] = existing.version + 1
            action = self.repository.preview_action(entity, payload)
        if action == "invalid_version":
            raise ValueError(f"{entity}_version_must_increase")
        return action


def parse_dosage(text: str) -> dict[str, Any]:
    """Parse only explicit, unambiguous dose tokens; always preserve raw text outside this result."""
    value = _text(text, limit=1000)
    result: dict[str, Any] = {}
    amount = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(克|g|公斤|kg|毫升|ml|升|l)(?:\s*/\s*亩)?", value, re.I)
    if amount:
        result["amount_value"] = float(amount.group(1))
        result["amount_unit"] = _unit(amount.group(2))
    water = re.search(r"兑水\s*(\d+(?:\.\d+)?)\s*(斤|公斤|kg|升|l)", value, re.I)
    if water:
        number = float(water.group(1))
        unit = water.group(2).casefold()
        if unit == "斤":
            result["water_value"] = number / 2
            result["water_unit"] = "kg"
        else:
            result["water_value"] = number
            result["water_unit"] = _unit(water.group(2))
    area = re.search(r"(?:用|冲施|滴灌|可用)?\s*(\d+(?:\.\d+)?)\s*亩", value)
    if area:
        result["area_value"] = float(area.group(1))
        result["area_unit"] = "亩"
    dilution = re.search(r"(\d+(?:\.\d+)?)\s*倍", value)
    if dilution:
        result["dilution_ratio"] = float(dilution.group(1))
    method = next((name for name in ("滴灌", "冲施", "灌根", "叶面喷施", "喷施", "撒施") if name in value), "")
    if method:
        result["application_method"] = method
    interval = re.search(r"间隔\s*(\d+)\s*(?:-|—|至)\s*(\d+)\s*天", value)
    if interval:
        result["interval"] = {"min_days": int(interval.group(1)), "max_days": int(interval.group(2))}
    frequency = re.search(r"(?:打|使用|施用)?\s*(\d+)\s*(?:-|—|至)\s*(\d+)\s*次", value)
    if frequency:
        result["max_frequency"] = int(frequency.group(2))
    return result


def _claims_for_record(record: dict[str, Any], *, review_state: str) -> list[dict[str, Any]]:
    source_id = str(record["source_id"])
    source_text = _text(record.get("source_text"), limit=4000)
    status = str(record.get("status") or "draft")
    claims: list[dict[str, Any]] = []
    mapping = {
        "key_functions": "function", "applicable_crops": "crop", "applicable_problems": "problem",
        "applicable_stages": "stage", "usage_methods": "usage", "precautions": "precaution",
        "incompatibilities": "compatibility", "risk_notes": "risk",
    }
    for field, claim_type in mapping.items():
        for value in _as_list(record.get(field)):
            claims.append(_claim(claim_type, value, source_id, source_text, status, review_state))
    dosage_text = _text(record.get("dosage_text"))
    if dosage_text:
        item = _claim("dosage", dosage_text, source_id, source_text, status, review_state)
        item["dosage_structured"] = record.get("dosage_structured") or parse_dosage(dosage_text)
        claims.append(item)
    for ingredient in _as_list(record.get("active_ingredients")):
        if isinstance(ingredient, dict) and ingredient.get("name"):
            claims.append(_claim("ingredient", ingredient["name"], source_id, source_text, status, review_state))
    for item in _as_list(record.get("claims")):
        if not isinstance(item, dict):
            raise ValueError("product_claim_object_required")
        explicit = dict(item)
        explicit.setdefault("source_id", source_id)
        explicit.setdefault("source_text", source_text)
        explicit.setdefault("status", status)
        explicit.setdefault("review_state", review_state)
        claims.append(explicit)
    return claims


def _claim(kind: str, value: Any, source_id: str, source_text: str, status: str, review_state: str) -> dict[str, Any]:
    return {
        "claim_type": kind, "claim_value": _text(value, required=True), "source_id": source_id,
        "source_text": source_text or _text(value), "status": status,
        "review_state": review_state, "confidence": 1.0,
    }


def _claim_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_type": getattr(value.get("claim_type"), "value", value.get("claim_type")),
        "claim_value": value.get("claim_value"), "normalized_value": value.get("normalized_value"),
        "source_id": value.get("source_id"), "source_text": value.get("source_text"),
        "status": getattr(value.get("status"), "value", value.get("status")),
        "review_state": getattr(value.get("review_state"), "value", value.get("review_state")),
        "conflict_group_id": value.get("conflict_group_id") or "",
        "dosage_structured": value.get("dosage_structured") or {},
        "confidence": value.get("confidence", 1.0),
    }


def _dedupe_claims(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            str(item.get("claim_type") or "other"),
            normalize_knowledge_value(item.get("normalized_value") or item.get("claim_value")),
            str(item.get("source_id") or ""),
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    result.sort(key=lambda item: (
        str(item.get("source_id") or ""),
        str(item.get("claim_type") or ""),
        normalize_knowledge_value(item.get("normalized_value") or item.get("claim_value")),
    ))
    return result


def _product_payload(product: Any) -> dict[str, Any]:
    if product is None:
        return {}
    value = asdict(product)
    value["status"] = product.status.value
    value["review_state"] = product.review_state.value
    for field in ("content_hash", "created_at", "updated_at"):
        value.pop(field, None)
    for field in ("aliases", "active_ingredients", *LIST_FIELDS):
        value[field] = list(value.get(field) or [])
    return value


def _link_payload(row: dict[str, Any], *, product: bool) -> dict[str, Any]:
    name_key = "source_product_name" if product else "source_solution_name"
    return {
        "source_id": row["source_id"], name_key: row.get(name_key, ""),
        "source_version": row.get("source_version", ""), "source_text": row.get("source_text", ""),
        "source_hash": row.get("source_hash", ""), "status": row.get("status", "draft"),
        "review_state": row.get("review_state", "unreviewed"),
    }


def _identity_signature(record: dict[str, Any]) -> str:
    parts = [
        normalize_knowledge_value(record.get("knowledge_kind") or "company_product"),
        normalize_knowledge_value(record.get("formulation")),
        normalize_knowledge_value(record.get("ingredient_content")),
        _hash(record.get("active_ingredients") or []),
        _hash(record.get("specifications") or {}),
    ]
    return "|".join(parts)


def _identity_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_kind = normalize_knowledge_value(left.get("knowledge_kind") or "company_product")
    right_kind = normalize_knowledge_value(right.get("knowledge_kind") or "company_product")
    if left_kind != right_kind:
        return False
    for field in ("formulation", "ingredient_content"):
        a = normalize_knowledge_value(left.get(field))
        b = normalize_knowledge_value(right.get(field))
        if a and b and a != b:
            return False
    left_ingredients = {
        normalize_knowledge_value(item.get("name"))
        for item in _as_list(left.get("active_ingredients")) if isinstance(item, dict) and item.get("name")
    }
    right_ingredients = {
        normalize_knowledge_value(item.get("name"))
        for item in _as_list(right.get("active_ingredients")) if isinstance(item, dict) and item.get("name")
    }
    if left_ingredients and right_ingredients and left_ingredients != right_ingredients:
        return False
    left_specs = left.get("specifications") if isinstance(left.get("specifications"), dict) else {}
    right_specs = right.get("specifications") if isinstance(right.get("specifications"), dict) else {}
    for key in set(left_specs) & set(right_specs):
        if normalize_knowledge_value(left_specs[key]) != normalize_knowledge_value(right_specs[key]):
            return False
    return True


def _distinct(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_knowledge_value(value)
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _unique_text(values: Iterable[Any]) -> list[str]:
    return _distinct([_text(value) for value in values if _text(value)])


def _unique_objects(values: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        item = {"name": value} if isinstance(value, str) else value
        if not isinstance(item, dict) or not _text(item.get("name")):
            continue
        key = _hash(item)
        if key not in seen:
            seen.add(key)
            result.append(dict(item))
    return result


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.replace("；", ";").split(";") if part.strip()]
    raise ValueError("list_value_required")


def _text(value: Any, limit: int = 1000, *, required: bool = False) -> str:
    result = " ".join(str(value or "").strip().split())[:limit]
    if required and not result:
        raise ValueError("non_empty_value_required")
    return result


def _unit(value: str) -> str:
    return {
        "克": "g", "g": "g", "公斤": "kg", "kg": "kg", "毫升": "ml", "ml": "ml",
        "升": "L", "l": "L",
    }.get(value.casefold(), value)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:28]}"


def _empty_bundle() -> dict[str, list[dict[str, Any]]]:
    return {"sources": [], "products": [], "solutions": []}


def _csv_row(row: dict[str, Any], entity_type: str) -> dict[str, Any]:
    result = {str(key): str(value or "").strip() for key, value in row.items() if key}
    list_fields = {
        "products": ("aliases", *LIST_FIELDS),
        "solutions": ("crops", "target_problems", "growth_stages", "region_constraints", "precautions", "risk_notes"),
        "sources": (),
    }
    json_fields = {
        "products": ("active_ingredients", "specifications", "dosage_structured", "claims", "source_links", "alias_sources"),
        "solutions": ("conditions", "items", "source_links"),
        "sources": ("metadata",),
    }
    for field in list_fields[entity_type]:
        if result.get(field):
            result[field] = _as_list(result[field])
    for field in json_fields[entity_type]:
        if result.get(field):
            result[field] = json.loads(result[field])
    if result.get("version"):
        result["version"] = int(result["version"])
    return result


def _read_xlsx(path: Path) -> dict[str, list[dict[str, str]]]:
    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter(f"{ns_main}t")) for item in root]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        result: dict[str, list[dict[str, str]]] = {}
        sheets = workbook.find(f"{ns_main}sheets")
        for sheet in list(sheets or []):
            name = sheet.attrib.get("name", "").strip().casefold()
            if name not in {"sources", "products", "solutions"}:
                continue
            target = rel_map[sheet.attrib[f"{ns_rel}id"]]
            entry = "xl/" + target.lstrip("/")
            root = ElementTree.fromstring(archive.read(entry))
            rows: list[list[str]] = []
            for row in root.iter(f"{ns_main}row"):
                values: list[str] = []
                for cell in row.findall(f"{ns_main}c"):
                    kind = cell.attrib.get("t")
                    value_node = cell.find(f"{ns_main}v")
                    value = "" if value_node is None else value_node.text or ""
                    if kind == "s" and value:
                        value = shared[int(value)]
                    values.append(value)
                rows.append(values)
            if rows:
                headers = rows[0]
                result[name] = [dict(zip(headers, values)) for values in rows[1:] if any(values)]
        return result
