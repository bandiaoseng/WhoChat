from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

from whochat.storage.product_knowledge import ProductKnowledgeRepository, normalize_knowledge_value


@dataclass(frozen=True)
class KnowledgeImportPreview:
    valid: bool
    source_format: str
    bundle: dict[str, list[dict[str, Any]]]
    actions: tuple[dict[str, str], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {key: len(self.bundle.get(key, [])) for key in ("sources", "products", "solutions")}


class KnowledgeImportService:
    """Preview/validate/commit imports; commit is one SQLite transaction."""

    def __init__(self, repository: ProductKnowledgeRepository) -> None:
        self.repository = repository

    def preview_json(self, text: str) -> KnowledgeImportPreview:
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            return KnowledgeImportPreview(False, "json", _empty_bundle(), errors=(f"invalid_json:{exc.msg}",))
        if not isinstance(parsed, dict):
            return KnowledgeImportPreview(False, "json", _empty_bundle(), errors=("json_bundle_object_required",))
        bundle = {key: parsed.get(key, []) for key in ("sources", "products", "solutions")}
        return self._preview_bundle(bundle, source_format="json")

    def preview_csv(self, text: str, *, entity_type: str) -> KnowledgeImportPreview:
        if entity_type not in {"products", "solutions"}:
            return KnowledgeImportPreview(False, "csv", _empty_bundle(), errors=("csv_entity_type_invalid",))
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except csv.Error as exc:
            return KnowledgeImportPreview(False, "csv", _empty_bundle(), errors=(f"invalid_csv:{exc}",))
        if not rows:
            return KnowledgeImportPreview(False, "csv", _empty_bundle(), errors=("csv_has_no_rows",))
        converted: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, row in enumerate(rows, 2):
            try:
                converted.append(self._csv_row(row, entity_type))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"row_{index}:{exc}")
        if errors:
            return KnowledgeImportPreview(False, "csv", _empty_bundle(), errors=tuple(errors))
        bundle = _empty_bundle()
        bundle[entity_type] = converted
        return self._preview_bundle(bundle, source_format="csv")

    def commit(self, preview: KnowledgeImportPreview) -> dict[str, int]:
        if not preview.valid:
            raise ValueError("cannot_commit_invalid_preview")
        return self.repository.apply_bundle(preview.bundle)

    def _preview_bundle(self, raw_bundle: dict[str, Any], *, source_format: str) -> KnowledgeImportPreview:
        errors: list[str] = []
        warnings: list[str] = []
        bundle = _empty_bundle()
        for key in bundle:
            value = raw_bundle.get(key, [])
            if not isinstance(value, list):
                errors.append(f"{key}_list_required")
                continue
            if not all(isinstance(item, dict) for item in value):
                errors.append(f"{key}_objects_required")
                continue
            bundle[key] = [dict(item) for item in value]
        if errors:
            return KnowledgeImportPreview(False, source_format, bundle, errors=tuple(errors))

        source_ids = {
            str(item.get("source_id") or "") for item in bundle["sources"] if item.get("source_id")
        }
        for item in bundle["sources"]:
            try:
                if not item.get("title"):
                    raise ValueError("source_title_required")
                source_type = str(item.get("source_type") or "manual")
                if source_type not in {
                    "manual", "company_document", "feishu", "lexiang",
                    "historical_solution", "verified_chat", "other",
                }:
                    raise ValueError("source_type_invalid")
                if not item.get("source_id"):
                    warnings.append(f"source_without_explicit_id:{item.get('title', '')}")
            except ValueError as exc:
                errors.append(f"source:{item.get('title', '?')}:{exc}")

        actions: list[dict[str, str]] = []
        prepared_product_ids: set[str] = set()
        alias_owner: dict[str, str] = {}
        for item in bundle["products"]:
            try:
                prepared = self.repository._prepare_product(item)
                prepared_product_ids.add(prepared["product_id"])
                if prepared["source_id"] not in source_ids and self.repository.get_source(prepared["source_id"]) is None:
                    raise ValueError("knowledge_source_not_found")
                for alias in prepared["aliases"]:
                    alias_key = normalize_knowledge_value(alias)
                    owner = alias_owner.get(alias_key)
                    if owner and owner != prepared["product_id"]:
                        raise ValueError("product_alias_collision_in_bundle")
                    alias_owner[alias_key] = prepared["product_id"]
                actions.append({
                    "entity": "product",
                    "id": prepared["product_id"],
                    "name": prepared["product_name"],
                    "action": self.repository.preview_action("product", item),
                })
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"product:{item.get('product_name', '?')}:{exc}")

        for item in bundle["solutions"]:
            try:
                prepared = self.repository._prepare_solution(item)
                if prepared["source_id"] not in source_ids and self.repository.get_source(prepared["source_id"]) is None:
                    raise ValueError("knowledge_source_not_found")
                for solution_item in prepared["items"]:
                    product_id = solution_item["product_id"]
                    if product_id not in prepared_product_ids and self.repository.get_product(product_id) is None:
                        raise ValueError(f"solution_product_not_found:{product_id}")
                actions.append({
                    "entity": "solution",
                    "id": prepared["solution_id"],
                    "name": prepared["solution_name"],
                    "action": self.repository.preview_action("solution", item),
                })
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"solution:{item.get('solution_name', '?')}:{exc}")

        invalid_actions = [item for item in actions if item["action"] == "invalid_version"]
        errors.extend(f"{item['entity']}:{item['name']}:version_must_increase" for item in invalid_actions)
        return KnowledgeImportPreview(
            not errors,
            source_format,
            bundle,
            actions=tuple(actions),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _csv_row(row: dict[str, str], entity_type: str) -> dict[str, Any]:
        result: dict[str, Any] = {key: value.strip() for key, value in row.items() if key and value is not None}
        list_fields = {
            "products": (
                "aliases", "key_functions", "applicable_crops", "applicable_stages",
                "applicable_problems", "usage_methods", "precautions", "incompatibilities", "risk_notes",
            ),
            "solutions": (
                "crops", "target_problems", "growth_stages", "region_constraints", "precautions", "risk_notes",
            ),
        }
        json_fields = {
            "products": ("active_ingredients", "specifications", "dosage_structured", "claims"),
            "solutions": ("conditions", "items"),
        }
        for field in list_fields[entity_type]:
            if field in result:
                result[field] = [item.strip() for item in result[field].replace("；", ";").split(";") if item.strip()]
        for field in json_fields[entity_type]:
            if result.get(field):
                result[field] = json.loads(result[field])
        if result.get("version"):
            result["version"] = int(result["version"])
        return result


def _empty_bundle() -> dict[str, list[dict[str, Any]]]:
    return {"sources": [], "products": [], "solutions": []}
