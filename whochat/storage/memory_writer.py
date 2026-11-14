from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Callable

from whochat.core.models import utc_now_iso
from whochat.services.memory_canonicalization import (
    canonicalize_fact_identity,
    growth_stage_relation,
)
from whochat.storage.database import Database


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class PlannedContext:
    reference: str
    existing_context_id: str | None
    crop: str | None
    variety: str | None
    region: str | None
    acreage_value: float | None
    acreage_unit: str | None
    growth_stage: str | None
    cultivation_type: str | None
    season_or_year: str | None
    status: str


@dataclass(frozen=True)
class PlannedFact:
    growing_context_reference: str | None
    fact_category: str
    fact_key: str
    value_text: str
    value_json: dict | list
    normalized_value: str
    confidence: float
    provenance: str
    status: str
    resolution: str
    confirmation_only: bool = False
    existing_fact_id: str | None = None
    candidate_origins: tuple[str, ...] = ("ai",)


@dataclass(frozen=True)
class PlannedEvent:
    growing_context_reference: str | None
    event_type: str
    title: str
    summary: str
    structured_data: dict | list
    confidence: float
    provenance: str
    status: str


@dataclass(frozen=True)
class PlannedFollowup:
    growing_context_reference: str | None
    followup_type: str
    title: str
    due_at: str
    priority: int


@dataclass(frozen=True)
class MemoryWritePlan:
    account_scope_id: str
    target_instance_id: str | None
    contact_id: str
    conversation_id: str
    display_name: str
    source_message_id: str
    source_text: str
    source_observed_at: str
    existing_context_ids: tuple[str, ...]
    contexts: tuple[PlannedContext, ...]
    facts: tuple[PlannedFact, ...]
    events: tuple[PlannedEvent, ...]
    followups: tuple[PlannedFollowup, ...]
    # Historical backfill reuses the same profile/fact/event writer while its
    # evidence remains in historical_chat_messages.  The default keeps every
    # existing live-memory caller unchanged.
    source_kind: str = "live"

    @property
    def item_count(self) -> int:
        return len(self.contexts) + len(self.facts) + len(self.events) + len(self.followups)


class AtomicMemoryWriter:
    """Apply one validated extraction plan in one SQLite transaction."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def apply(
        self,
        plan: MemoryWritePlan,
        *,
        failure_injector: Callable[[str], None] | None = None,
    ) -> dict:
        if plan.item_count <= 0:
            raise ValueError("memory write plan is empty")
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source = self._validate_identity(conn, plan)
            profile_id, profile_created = self._profile(conn, plan)
            self._inject(failure_injector, "profile")

            reference_map: dict[str, str] = {
                context_id: context_id for context_id in plan.existing_context_ids
            }
            context_ids: list[str] = []
            contexts_created = 0
            for item in plan.contexts:
                context_id, created = self._context(conn, profile_id, item, plan.source_observed_at)
                reference_map[item.reference] = context_id
                reference_map[context_id] = context_id
                context_ids.append(context_id)
                contexts_created += int(created)
            self._inject(failure_injector, "contexts")

            facts_created = 0
            fact_ids: list[str] = []
            confirmed_fact_ids: list[str] = []
            added_source_count = 0
            for item in plan.facts:
                context_id = self._resolve_reference(item.growing_context_reference, reference_map)
                fact_id, created, source_added = self._fact(
                    conn, profile_id, context_id, item, plan, source
                )
                fact_ids.append(fact_id)
                facts_created += int(created)
                added_source_count += int(source_added)
                if not created:
                    confirmed_fact_ids.append(fact_id)
            self._inject(failure_injector, "facts")

            events_created = 0
            event_ids: list[str] = []
            created_event_ids: list[str] = []
            for item in plan.events:
                context_id = self._resolve_reference(item.growing_context_reference, reference_map)
                event_id, created = self._event(conn, profile_id, context_id, item, plan)
                event_ids.append(event_id)
                events_created += int(created)
                if created:
                    created_event_ids.append(event_id)
            self._inject(failure_injector, "events")

            followups_created = 0
            followup_ids: list[str] = []
            for item in plan.followups:
                context_id = self._resolve_reference(item.growing_context_reference, reference_map)
                followup_id, created = self._followup(conn, profile_id, context_id, item, plan)
                followup_ids.append(followup_id)
                followups_created += int(created)
            self._refresh_next_followup(conn, profile_id)
            self._inject(failure_injector, "followups")

            return {
                "profile_id": profile_id,
                "profile_created": profile_created,
                "contexts": len(context_ids),
                "contexts_created": contexts_created,
                "context_ids": context_ids,
                "facts": len(fact_ids),
                "facts_created": facts_created,
                "fact_ids": fact_ids,
                "ai_fact_candidates": sum(
                    1 for item in plan.facts if "ai" in item.candidate_origins
                ),
                "local_reconfirmations": sum(
                    1 for item in plan.facts if "local" in item.candidate_origins
                ),
                "confirmed_fact_ids": list(dict.fromkeys(confirmed_fact_ids)),
                "added_source_count": added_source_count,
                "events": len(event_ids),
                "events_created": events_created,
                "event_ids": event_ids,
                "created_event_ids": created_event_ids,
                "followups": len(followup_ids),
                "followups_created": followups_created,
                "followup_ids": followup_ids,
            }

    def apply_confirmations_only(self, plan: MemoryWritePlan) -> dict:
        """Attach local evidence without touching profile/context/event business rows."""
        if not plan.facts or plan.contexts or plan.events or plan.followups:
            raise ValueError("confirmation_only_plan_shape_invalid")
        if any(not item.confirmation_only for item in plan.facts):
            raise ValueError("confirmation_only_plan_contains_mutating_fact")
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source = self._validate_identity(conn, plan)
            profile = conn.execute(
                "SELECT id FROM customer_profiles WHERE account_scope_id=? AND contact_id=?",
                (plan.account_scope_id, plan.contact_id),
            ).fetchone()
            if profile is None:
                raise ValueError("customer_profile_missing")
            reference_map = {
                context_id: context_id for context_id in plan.existing_context_ids
            }
            fact_ids: list[str] = []
            added_source_count = 0
            for item in plan.facts:
                context_id = self._resolve_reference(
                    item.growing_context_reference, reference_map
                )
                fact_id, created, source_added = self._fact(
                    conn, profile["id"], context_id, item, plan, source
                )
                if created:
                    raise RuntimeError("confirmation_only_created_fact")
                fact_ids.append(fact_id)
                added_source_count += int(source_added)
            return {
                "profile_id": profile["id"],
                "profile_created": False,
                "contexts": 0,
                "contexts_created": 0,
                "context_ids": [],
                "facts": len(fact_ids),
                "facts_created": 0,
                "fact_ids": fact_ids,
                "ai_fact_candidates": 0,
                "local_reconfirmations": len(plan.facts),
                "confirmed_fact_ids": list(dict.fromkeys(fact_ids)),
                "added_source_count": added_source_count,
                "events": 0,
                "events_created": 0,
                "event_ids": [],
                "followups": 0,
                "followups_created": 0,
                "followup_ids": [],
            }

    @staticmethod
    def _inject(callback: Callable[[str], None] | None, stage: str) -> None:
        if callback is not None:
            callback(stage)

    @staticmethod
    def _resolve_reference(reference: str | None, reference_map: dict[str, str]) -> str | None:
        if reference is None:
            return None
        context_id = reference_map.get(reference)
        if context_id is None:
            raise ValueError(f"unknown_context_reference:{reference}")
        return context_id

    @staticmethod
    def _validate_identity(conn: sqlite3.Connection, plan: MemoryWritePlan) -> sqlite3.Row:
        if plan.source_kind == "historical":
            source = conn.execute(
                """
                SELECT historical_message_id AS id,text,sent_at AS observed_at,
                       account_scope_id,contact_id,conversation_id,sender,mapping_status
                FROM historical_chat_messages WHERE historical_message_id=?
                """,
                (plan.source_message_id,),
            ).fetchone()
            if source is not None and source["mapping_status"] != "resolved":
                raise ValueError("historical_source_mapping_not_resolved")
        elif plan.source_kind == "live":
            source = conn.execute("SELECT * FROM messages WHERE id = ?", (plan.source_message_id,)).fetchone()
        else:
            raise ValueError("unsupported_memory_source_kind")
        if source is None:
            raise ValueError("source_message_id_not_found")
        if (
            source["account_scope_id"] != plan.account_scope_id
            or source["contact_id"] != plan.contact_id
            or source["conversation_id"] != plan.conversation_id
        ):
            raise ValueError("source_message_identity_mismatch")
        if source["text"] != plan.source_text or source["observed_at"] != plan.source_observed_at:
            raise ValueError("source_message_evidence_mismatch")
        contact = conn.execute("SELECT * FROM contacts WHERE id = ?", (plan.contact_id,)).fetchone()
        if contact is None or contact["account_scope_id"] != plan.account_scope_id:
            raise ValueError("contact_scope_mismatch")
        conversation = conn.execute("SELECT * FROM conversations WHERE id = ?", (plan.conversation_id,)).fetchone()
        if (
            conversation is None
            or conversation["account_scope_id"] != plan.account_scope_id
            or conversation["contact_id"] != plan.contact_id
        ):
            raise ValueError("conversation_scope_mismatch")
        if plan.target_instance_id:
            target = conn.execute(
                "SELECT account_scope_id FROM target_instances WHERE id = ?", (plan.target_instance_id,)
            ).fetchone()
            if target is None or target["account_scope_id"] != plan.account_scope_id:
                raise ValueError("target_instance_scope_mismatch")
        return source

    @staticmethod
    def _profile(conn: sqlite3.Connection, plan: MemoryWritePlan) -> tuple[str, bool]:
        now = utc_now_iso()
        row = conn.execute(
            "SELECT * FROM customer_profiles WHERE account_scope_id=? AND contact_id=?",
            (plan.account_scope_id, plan.contact_id),
        ).fetchone()
        if row is not None:
            conn.execute(
                """
                UPDATE customer_profiles SET
                    target_instance_id=COALESCE(?,target_instance_id),
                    conversation_id=?,
                    display_name=?,
                    latest_contact_at=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    plan.target_instance_id,
                    plan.conversation_id,
                    plan.display_name,
                    plan.source_observed_at,
                    now,
                    row["id"],
                ),
            )
            return row["id"], False
        profile_id = _new_id("customer_profile")
        conn.execute(
            """
            INSERT INTO customer_profiles(
                id,account_scope_id,target_instance_id,contact_id,conversation_id,
                display_name,sales_stage,latest_contact_at,next_followup_at,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,'new',?,NULL,?,?)
            """,
            (
                profile_id,
                plan.account_scope_id,
                plan.target_instance_id,
                plan.contact_id,
                plan.conversation_id,
                plan.display_name,
                plan.source_observed_at,
                now,
                now,
            ),
        )
        return profile_id, True

    @staticmethod
    def _context(
        conn: sqlite3.Connection,
        profile_id: str,
        item: PlannedContext,
        observed_at: str,
    ) -> tuple[str, bool, bool]:
        row = None
        if item.existing_context_id:
            row = conn.execute(
                "SELECT * FROM customer_growing_contexts WHERE id=? AND customer_profile_id=?",
                (item.existing_context_id, profile_id),
            ).fetchone()
            if row is None:
                raise ValueError("planned_existing_context_missing")
        elif item.crop:
            candidates = conn.execute(
                """
                SELECT * FROM customer_growing_contexts
                WHERE customer_profile_id=? AND lower(trim(crop))=lower(trim(?))
                  AND status IN ('active','uncertain')
                ORDER BY last_confirmed_at DESC,created_at ASC
                """,
                (profile_id, item.crop),
            ).fetchall()
            exact_season = [
                candidate for candidate in candidates
                if item.season_or_year and candidate["season_or_year"] == item.season_or_year
            ]
            if len(exact_season) == 1:
                row = exact_season[0]
            elif not item.season_or_year and len(candidates) == 1:
                row = candidates[0]
            elif item.season_or_year and len(candidates) == 1 and not candidates[0]["season_or_year"]:
                row = candidates[0]
        now = utc_now_iso()
        if row is not None:
            status = row["status"] if row["status"] == "active" and item.status == "uncertain" else item.status
            conn.execute(
                """
                UPDATE customer_growing_contexts SET
                    crop=CASE WHEN ? IS NULL THEN crop ELSE ? END,
                    variety=CASE WHEN ? IS NULL THEN variety ELSE ? END,
                    region=CASE WHEN ? IS NULL THEN region ELSE ? END,
                    acreage_value=COALESCE(?,acreage_value),
                    acreage_unit=CASE WHEN ? IS NULL THEN acreage_unit ELSE ? END,
                    growth_stage=CASE WHEN ? IS NULL THEN growth_stage ELSE ? END,
                    cultivation_type=CASE WHEN ? IS NULL THEN cultivation_type ELSE ? END,
                    season_or_year=CASE WHEN ? IS NULL THEN season_or_year ELSE ? END,
                    status=?,last_confirmed_at=?,updated_at=?
                WHERE id=?
                """,
                (
                    item.crop, item.crop,
                    item.variety, item.variety,
                    item.region, item.region,
                    item.acreage_value,
                    item.acreage_unit, item.acreage_unit,
                    item.growth_stage, item.growth_stage,
                    item.cultivation_type, item.cultivation_type,
                    item.season_or_year, item.season_or_year,
                    status, observed_at, now, row["id"],
                ),
            )
            return row["id"], False
        if not item.crop:
            raise ValueError("new_context_requires_crop")
        context_id = _new_id("growing_context")
        conn.execute(
            """
            INSERT INTO customer_growing_contexts(
                id,customer_profile_id,crop,variety,region,acreage_value,acreage_unit,
                growth_stage,cultivation_type,season_or_year,status,first_seen_at,
                last_confirmed_at,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                context_id, profile_id, item.crop, item.variety or "", item.region or "",
                item.acreage_value, item.acreage_unit or "", item.growth_stage or "",
                item.cultivation_type or "", item.season_or_year or "", item.status,
                observed_at, observed_at, now, now,
            ),
        )
        return context_id, True

    def _fact(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        context_id: str | None,
        item: PlannedFact,
        plan: MemoryWritePlan,
        source: sqlite3.Row,
    ) -> tuple[str, bool]:
        now = utc_now_iso()
        provenance = item.provenance
        authoritative = provenance in {"customer_explicit", "system_imported"}
        confidence = min(item.confidence, 0.69) if provenance == "ai_inferred" else item.confidence
        status = item.status if authoritative or item.status != "active" else "uncertain"
        incoming = canonicalize_fact_identity(
            customer_profile_id=profile_id,
            growing_context_id=context_id,
            fact_category=item.fact_category,
            fact_key=item.fact_key,
            value_text=item.value_text,
            normalized_value=item.normalized_value,
            value_json=item.value_json,
            source_text=plan.source_text,
        )
        rows = conn.execute(
            """
            SELECT * FROM customer_facts
            WHERE customer_profile_id=? AND growing_context_id IS ?
              AND status IN ('active','uncertain','conflicting')
            ORDER BY created_at,id
            """,
            (profile_id, context_id),
        ).fetchall()
        candidates: list[tuple[sqlite3.Row, object]] = []
        for row in rows:
            identity = canonicalize_fact_identity(
                customer_profile_id=profile_id,
                growing_context_id=context_id,
                fact_category=row["fact_category"],
                fact_key=row["fact_key"],
                value_text=row["value_text"],
                normalized_value=row["normalized_value"],
                value_json=row["value_json"],
            )
            if (
                identity.canonical_category == incoming.canonical_category
                and identity.canonical_key == incoming.canonical_key
                and identity.temporal_scope == incoming.temporal_scope
            ):
                candidates.append((row, identity))
        exact_pair = next(
            (pair for pair in candidates if pair[1].canonical_value == incoming.canonical_value),
            None,
        )
        compatible_pair = None
        if exact_pair is None and incoming.canonical_key == "growth_stage":
            compatible_pair = next(
                (
                    pair for pair in candidates
                    if growth_stage_relation(
                        pair[1].canonical_value,
                        incoming.canonical_value,
                        source_text=plan.source_text,
                    ) in {"same", "compatible"}
                ),
                None,
            )
        matched_pair = exact_pair or compatible_pair
        if matched_pair is not None:
            exact, existing_identity = matched_pair
            if item.confirmation_only and exact["id"] != item.existing_fact_id:
                raise ValueError("local_reconfirmation_target_mismatch")
            compatible = exact_pair is None
            if item.confirmation_only:
                conn.execute(
                    "UPDATE customer_facts SET last_confirmed_at=?,updated_at=? WHERE id=?",
                    (plan.source_observed_at, now, exact["id"]),
                )
                source_added = self._fact_source(conn, exact["id"], plan, source)
                return exact["id"], False, source_added
            other_values = [candidate for candidate, _ in candidates if candidate["id"] != exact["id"]]
            next_status = exact["status"]
            if authoritative and next_status == "uncertain" and not other_values:
                next_status = "active"
            next_provenance = self._stronger_provenance(exact["provenance"], provenance)
            next_value_text = exact["value_text"] if compatible else item.value_text
            next_value_json = (
                exact["value_json"]
                if compatible
                else json.dumps(item.value_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            next_normalized = (
                existing_identity.canonical_value if compatible else incoming.canonical_value
            )
            conn.execute(
                """
                UPDATE customer_facts SET fact_category=?,fact_key=?,value_text=?,value_json=?,
                    normalized_value=?,confidence=MAX(confidence,?),provenance=?,status=?,
                    last_confirmed_at=?,updated_at=? WHERE id=?
                """,
                (
                    incoming.canonical_category,
                    incoming.canonical_key,
                    next_value_text,
                    next_value_json,
                    next_normalized,
                    confidence,
                    next_provenance,
                    next_status,
                    plan.source_observed_at,
                    now,
                    exact["id"],
                ),
            )
            fact_id = exact["id"]
            source_added = self._fact_source(conn, fact_id, plan, source)
            return fact_id, False, source_added

        if item.confirmation_only:
            raise ValueError("local_reconfirmation_target_missing")

        conflict_group_id = None
        authoritative_candidates = [
            candidate for candidate, _ in candidates
            if candidate["provenance"] in {"customer_explicit", "system_imported"}
        ]
        resolution = item.resolution
        supersede_candidates = [candidate for candidate, _ in candidates]
        if resolution == "supersede" and incoming.canonical_key == "growth_stage":
            supersede_candidates = [
                candidate
                for candidate, identity in candidates
                if growth_stage_relation(
                    identity.canonical_value,
                    incoming.canonical_value,
                    source_text=plan.source_text,
                ) == "advance"
            ]
            if not supersede_candidates:
                resolution = "coexist"
                status = "uncertain"
        if resolution == "conflict" and authoritative and authoritative_candidates:
            conflict_group_id = next(
                (candidate["conflict_group_id"] for candidate in authoritative_candidates if candidate["conflict_group_id"]),
                _new_id("fact_conflict"),
            )
            ids = [candidate["id"] for candidate in authoritative_candidates]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE customer_facts SET status='conflicting',conflict_group_id=?,updated_at=? "
                f"WHERE id IN ({placeholders})",
                (conflict_group_id, now, *ids),
            )
            status = "conflicting"
        if resolution == "supersede" and not authoritative:
            raise ValueError("non_authoritative_fact_cannot_supersede")

        fact_id = _new_id("customer_fact")
        conn.execute(
            """
            INSERT INTO customer_facts(
                id,customer_profile_id,growing_context_id,fact_category,fact_key,value_text,
                value_json,normalized_value,confidence,provenance,status,first_seen_at,
                last_confirmed_at,valid_from,valid_to,conflict_group_id,
                superseded_by_fact_id,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,NULL,?,?)
            """,
            (
                fact_id, profile_id, context_id, incoming.canonical_category, incoming.canonical_key,
                item.value_text,
                json.dumps(item.value_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                incoming.canonical_value, confidence, provenance, status,
                plan.source_observed_at, plan.source_observed_at, plan.source_observed_at,
                conflict_group_id, now, now,
            ),
        )
        if resolution == "supersede":
            for candidate in supersede_candidates:
                conn.execute(
                    """
                    UPDATE customer_facts SET status='superseded',valid_to=?,
                        superseded_by_fact_id=?,updated_at=? WHERE id=?
                    """,
                    (plan.source_observed_at, fact_id, now, candidate["id"]),
                )
        source_added = self._fact_source(conn, fact_id, plan, source)
        return fact_id, True, source_added

    @staticmethod
    def _stronger_provenance(existing: str, incoming: str) -> str:
        priority = {"ai_inferred": 0, "user_explicit": 1, "customer_explicit": 2, "system_imported": 3}
        return incoming if priority[incoming] > priority[existing] else existing

    @staticmethod
    def _fact_source(
        conn: sqlite3.Connection,
        fact_id: str,
        plan: MemoryWritePlan,
        source: sqlite3.Row,
    ) -> bool:
        live_id = source["id"] if plan.source_kind == "live" else None
        historical_id = source["id"] if plan.source_kind == "historical" else None
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO customer_fact_sources(
                id,fact_id,message_id,conversation_id,source_text,observed_at,created_at,
                historical_message_id
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                _new_id("fact_source"), fact_id, live_id, plan.conversation_id,
                source["text"], source["observed_at"], utc_now_iso(), historical_id,
            ),
        )
        return cursor.rowcount > 0

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        profile_id: str,
        context_id: str | None,
        item: PlannedEvent,
        plan: MemoryWritePlan,
    ) -> tuple[str, bool]:
        source_column = "s.historical_message_id" if plan.source_kind == "historical" else "s.message_id"
        if plan.source_kind == "historical":
            # The source message is the durable event identity.  Context
            # binding may safely improve after the original import, but that
            # must never create a second copy of an event already accepted
            # from the same historical evidence.
            existing = conn.execute(
                f"""
                SELECT e.id FROM customer_events e
                JOIN customer_event_sources s ON s.event_id=e.id
                WHERE e.customer_profile_id=? AND e.event_type=? AND e.title=?
                  AND {source_column}=?
                LIMIT 1
                """,
                (profile_id, item.event_type, item.title, plan.source_message_id),
            ).fetchone()
        else:
            existing = conn.execute(
                f"""
                SELECT e.id FROM customer_events e
                JOIN customer_event_sources s ON s.event_id=e.id
                WHERE e.customer_profile_id=? AND e.growing_context_id IS ?
                  AND e.event_type=? AND e.title=? AND {source_column}=?
                LIMIT 1
                """,
                (profile_id, context_id, item.event_type, item.title, plan.source_message_id),
            ).fetchone()
        if existing is not None:
            return existing["id"], False
        if plan.source_kind == "historical":
            equivalent = conn.execute(
                """
                SELECT e.id FROM customer_events e
                JOIN customer_event_sources s ON s.event_id=e.id
                WHERE e.customer_profile_id=? AND e.growing_context_id IS ?
                  AND e.event_type=? AND e.title=? AND s.source_text=?
                  AND e.status IN ('active','uncertain')
                ORDER BY e.event_at DESC LIMIT 1
                """,
                (profile_id, context_id, item.event_type, item.title, plan.source_text),
            ).fetchone()
            if equivalent is not None:
                conn.execute(
                    """INSERT OR IGNORE INTO customer_event_sources(
                        id,event_id,message_id,conversation_id,source_text,observed_at,
                        created_at,historical_message_id
                    ) VALUES (?,?,NULL,?,?,?,?,?)""",
                    (
                        _new_id("event_source"), equivalent["id"], plan.conversation_id,
                        plan.source_text, plan.source_observed_at, utc_now_iso(),
                        plan.source_message_id,
                    ),
                )
                return equivalent["id"], False
        now = utc_now_iso()
        event_id = _new_id("customer_event")
        confidence = min(item.confidence, 0.69) if item.provenance == "ai_inferred" else item.confidence
        status = "uncertain" if item.provenance == "ai_inferred" and item.status == "active" else item.status
        conn.execute(
            """
            INSERT INTO customer_events(
                id,customer_profile_id,growing_context_id,event_type,title,summary,
                structured_data_json,event_at,confidence,provenance,status,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id, profile_id, context_id, item.event_type, item.title, item.summary,
                json.dumps(item.structured_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                plan.source_observed_at, confidence, item.provenance, status, now, now,
            ),
        )
        conn.execute(
            """
            INSERT INTO customer_event_sources(
                id,event_id,message_id,conversation_id,source_text,observed_at,created_at,
                historical_message_id
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                _new_id("event_source"), event_id,
                plan.source_message_id if plan.source_kind == "live" else None,
                plan.conversation_id, plan.source_text, plan.source_observed_at, now,
                plan.source_message_id if plan.source_kind == "historical" else None,
            ),
        )
        return event_id, True

    @staticmethod
    def _followup(
        conn: sqlite3.Connection,
        profile_id: str,
        context_id: str | None,
        item: PlannedFollowup,
        plan: MemoryWritePlan,
    ) -> tuple[str, bool]:
        source_column = (
            "source_historical_message_id"
            if plan.source_kind == "historical"
            else "source_message_id"
        )
        exact_source = conn.execute(
            f"""
            SELECT id FROM customer_followups
            WHERE customer_profile_id=? AND followup_type=? AND {source_column}=?
            ORDER BY created_at,id LIMIT 1
            """,
            (profile_id, item.followup_type, plan.source_message_id),
        ).fetchone()
        if exact_source is not None:
            # Saved-response replay must not recreate an already completed or
            # cancelled explicit task.  Its original source identity is the
            # durable idempotency key, independent of current task status.
            return exact_source["id"], False
        existing = conn.execute(
            """
            SELECT * FROM customer_followups
            WHERE customer_profile_id=? AND growing_context_id IS ? AND followup_type=?
              AND status='pending'
            ORDER BY created_at,id LIMIT 1
            """,
            (profile_id, context_id, item.followup_type),
        ).fetchone()
        if existing is not None:
            # A manually created task is authoritative.  Explicit memory
            # candidates may replace a system suggestion, but never the task
            # the operator deliberately entered.
            if existing["source"] != "user_created":
                conn.execute(
                    """
                    UPDATE customer_followups
                    SET title=?,due_at=?,priority=?,source='explicit',
                        source_message_id=?,source_historical_message_id=?,reason=?,updated_at=?
                    WHERE id=?
                    """,
                    (
                        item.title,
                        item.due_at,
                        item.priority,
                        plan.source_message_id if plan.source_kind == "live" else None,
                        plan.source_message_id if plan.source_kind == "historical" else None,
                        "聊天中明确提出的回访安排",
                        utc_now_iso(),
                        existing["id"],
                    ),
                )
            return existing["id"], False
        now = utc_now_iso()
        followup_id = _new_id("customer_followup")
        conn.execute(
            """
            INSERT INTO customer_followups(
                id,customer_profile_id,growing_context_id,related_event_id,followup_type,
                title,due_at,status,priority,completed_at,cancelled_at,source,
                source_message_id,reason,created_at,updated_at,source_historical_message_id
            ) VALUES (?,?,?,NULL,?,?,?,'pending',?,NULL,NULL,'explicit',?,?,?,?,?)
            """,
            (
                followup_id, profile_id, context_id, item.followup_type,
                item.title, item.due_at, item.priority,
                plan.source_message_id if plan.source_kind == "live" else None,
                "聊天中明确提出的回访安排", now, now,
                plan.source_message_id if plan.source_kind == "historical" else None,
            ),
        )
        return followup_id, True

    @staticmethod
    def _refresh_next_followup(conn: sqlite3.Connection, profile_id: str) -> None:
        row = conn.execute(
            "SELECT MIN(due_at) FROM customer_followups WHERE customer_profile_id=? AND status='pending'",
            (profile_id,),
        ).fetchone()
        conn.execute(
            "UPDATE customer_profiles SET next_followup_at=?,updated_at=? WHERE id=?",
            (row[0], utc_now_iso(), profile_id),
        )
