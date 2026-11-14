from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from typing import Any, Callable

from whochat.core.models import (
    CustomerEvent,
    CustomerEventSource,
    CustomerEventStatus,
    CustomerEventType,
    CustomerFact,
    CustomerFactProvenance,
    CustomerFactSource,
    CustomerFactStatus,
    CustomerFollowup,
    CustomerFollowupSource,
    CustomerFollowupStatus,
    CustomerGrowingContext,
    CustomerMemoryEdit,
    CustomerProfile,
    CustomerSalesStage,
    CustomerSalesStageHistory,
    CustomerSalesStageSource,
    GrowingContextStatus,
    utc_now_iso,
)
from whochat.storage.database import Database
from whochat.services.memory_canonicalization import (
    canonicalize_acreage,
    canonicalize_fact_identity,
    extract_acreage,
    format_number,
    growth_stage_relation,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _clean(value: str, *, required: bool = False, limit: int = 500) -> str:
    cleaned = " ".join((value or "").strip().split())[:limit]
    if required and not cleaned:
        raise ValueError("a non-empty value is required")
    return cleaned


def _normalise_fact(value: str) -> str:
    return " ".join((value or "").strip().split()).casefold()


def _json_text(value: str | dict[str, Any] | list[Any] | None) -> str:
    if value is None or value == "":
        return "{}"
    parsed = json.loads(value) if isinstance(value, str) else value
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _confidence(value: float) -> float:
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return result


class CustomerProfileRepository:
    """Local customer records with no OCR, AI-provider, or UI dependency."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_or_create_customer_profile(
        self,
        *,
        account_scope_id: str,
        contact_id: str,
        target_instance_id: str | None = None,
        conversation_id: str | None = None,
        display_name: str | None = None,
        latest_contact_at: str | None = None,
    ) -> CustomerProfile:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            contact = self._require_contact(conn, account_scope_id, contact_id)
            self._validate_optional_bindings(
                conn,
                account_scope_id=account_scope_id,
                contact_id=contact_id,
                target_instance_id=target_instance_id,
                conversation_id=conversation_id,
            )
            row = conn.execute(
                "SELECT * FROM customer_profiles WHERE account_scope_id = ? AND contact_id = ?",
                (account_scope_id, contact_id),
            ).fetchone()
            name = _clean(display_name or contact["display_name"], required=True, limit=200)
            if row:
                conn.execute(
                    """
                    UPDATE customer_profiles
                    SET target_instance_id = COALESCE(?, target_instance_id),
                        conversation_id = COALESCE(?, conversation_id),
                        display_name = ?,
                        latest_contact_at = COALESCE(?, latest_contact_at),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (target_instance_id, conversation_id, name, latest_contact_at, now, row["id"]),
                )
                row = conn.execute("SELECT * FROM customer_profiles WHERE id = ?", (row["id"],)).fetchone()
                return _profile_from_row(row)
            item = CustomerProfile(
                id=_new_id("customer_profile"),
                account_scope_id=account_scope_id,
                target_instance_id=target_instance_id,
                contact_id=contact_id,
                conversation_id=conversation_id,
                display_name=name,
                sales_stage=CustomerSalesStage.NEW,
                sales_stage_source=CustomerSalesStageSource.SYSTEM_DERIVED,
                sales_stage_updated_at=None,
                latest_contact_at=latest_contact_at,
                next_followup_at=None,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO customer_profiles(
                    id, account_scope_id, target_instance_id, contact_id, conversation_id,
                    display_name, sales_stage, sales_stage_source, sales_stage_updated_at,
                    latest_contact_at, next_followup_at,
                    created_at, updated_at
                ) VALUES (
                    :id, :account_scope_id, :target_instance_id, :contact_id, :conversation_id,
                    :display_name, :sales_stage, :sales_stage_source, :sales_stage_updated_at,
                    :latest_contact_at, :next_followup_at,
                    :created_at, :updated_at
                )
                """,
                _profile_params(item),
            )
            return item

    def get_customer_profile(
        self,
        profile_id: str | None = None,
        *,
        account_scope_id: str | None = None,
        contact_id: str | None = None,
    ) -> CustomerProfile | None:
        with self.db.connect() as conn:
            if profile_id:
                row = conn.execute("SELECT * FROM customer_profiles WHERE id = ?", (profile_id,)).fetchone()
            elif account_scope_id and contact_id:
                row = conn.execute(
                    "SELECT * FROM customer_profiles WHERE account_scope_id = ? AND contact_id = ?",
                    (account_scope_id, contact_id),
                ).fetchone()
            else:
                raise ValueError("profile_id or account_scope_id + contact_id is required")
        return _profile_from_row(row) if row else None

    def list_active_growing_contexts(self, customer_profile_id: str) -> list[CustomerGrowingContext]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM customer_growing_contexts WHERE customer_profile_id = ? AND status = 'active' ORDER BY last_confirmed_at DESC, created_at ASC",
                (customer_profile_id,),
            ).fetchall()
        return [_growing_context_from_row(row) for row in rows]

    def create_growing_context(
        self,
        customer_profile_id: str,
        *,
        crop: str = "",
        variety: str = "",
        region: str = "",
        acreage_value: float | None = None,
        acreage_unit: str = "",
        growth_stage: str = "",
        cultivation_type: str = "",
        season_or_year: str = "",
        status: GrowingContextStatus | str = GrowingContextStatus.UNCERTAIN,
        first_seen_at: str | None = None,
        last_confirmed_at: str | None = None,
    ) -> CustomerGrowingContext:
        status = GrowingContextStatus(status)
        if acreage_value is not None and float(acreage_value) < 0:
            raise ValueError("acreage_value cannot be negative")
        now = utc_now_iso()
        first_seen_at = first_seen_at or now
        last_confirmed_at = last_confirmed_at or first_seen_at
        with self.db.connect() as conn:
            self._require_profile(conn, customer_profile_id)
            item = CustomerGrowingContext(
                id=_new_id("growing_context"),
                customer_profile_id=customer_profile_id,
                crop=_clean(crop, limit=100),
                variety=_clean(variety, limit=100),
                region=_clean(region, limit=200),
                acreage_value=float(acreage_value) if acreage_value is not None else None,
                acreage_unit=_clean(acreage_unit, limit=30),
                growth_stage=_clean(growth_stage, limit=100),
                cultivation_type=_clean(cultivation_type, limit=100),
                season_or_year=_clean(season_or_year, limit=100),
                status=status,
                first_seen_at=first_seen_at,
                last_confirmed_at=last_confirmed_at,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO customer_growing_contexts(
                    id, customer_profile_id, crop, variety, region, acreage_value,
                    acreage_unit, growth_stage, cultivation_type, season_or_year,
                    status, first_seen_at, last_confirmed_at, created_at, updated_at
                ) VALUES (
                    :id, :customer_profile_id, :crop, :variety, :region, :acreage_value,
                    :acreage_unit, :growth_stage, :cultivation_type, :season_or_year,
                    :status, :first_seen_at, :last_confirmed_at, :created_at, :updated_at
                )
                """,
                _growing_context_params(item),
            )
        return item

    def add_or_confirm_growing_context(
        self,
        customer_profile_id: str,
        *,
        context_id: str | None = None,
        crop: str = "",
        variety: str = "",
        region: str = "",
        acreage_value: float | None = None,
        acreage_unit: str = "",
        growth_stage: str = "",
        cultivation_type: str = "",
        season_or_year: str = "",
        status: GrowingContextStatus | str = GrowingContextStatus.UNCERTAIN,
        confirmed_at: str | None = None,
    ) -> tuple[CustomerGrowingContext, bool]:
        """Update one known crop context or create it without duplicating a clear match."""
        status = GrowingContextStatus(status)
        if acreage_value is not None and float(acreage_value) < 0:
            raise ValueError("acreage_value cannot be negative")
        crop = _clean(crop, limit=100)
        season_or_year = _clean(season_or_year, limit=100)
        now = utc_now_iso()
        confirmed_at = confirmed_at or now
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_profile(conn, customer_profile_id)
            existing = None
            if context_id:
                existing = conn.execute(
                    "SELECT * FROM customer_growing_contexts WHERE id = ? AND customer_profile_id = ?",
                    (context_id, customer_profile_id),
                ).fetchone()
                if existing is None:
                    raise ValueError("growing context does not belong to the customer profile")
            elif crop:
                rows = conn.execute(
                    """
                    SELECT * FROM customer_growing_contexts
                    WHERE customer_profile_id = ? AND lower(trim(crop)) = lower(trim(?))
                      AND status IN ('active', 'uncertain')
                    ORDER BY last_confirmed_at DESC, created_at ASC
                    """,
                    (customer_profile_id, crop),
                ).fetchall()
                exact_season = [row for row in rows if season_or_year and row["season_or_year"] == season_or_year]
                if len(exact_season) == 1:
                    existing = exact_season[0]
                elif not season_or_year and len(rows) == 1:
                    existing = rows[0]
                elif season_or_year and len(rows) == 1 and not rows[0]["season_or_year"]:
                    existing = rows[0]
            if existing is not None:
                new_status = status.value
                if existing["status"] == GrowingContextStatus.ACTIVE.value and status == GrowingContextStatus.UNCERTAIN:
                    new_status = existing["status"]
                conn.execute(
                    """
                    UPDATE customer_growing_contexts SET
                        crop = CASE WHEN ? = '' THEN crop ELSE ? END,
                        variety = CASE WHEN ? = '' THEN variety ELSE ? END,
                        region = CASE WHEN ? = '' THEN region ELSE ? END,
                        acreage_value = COALESCE(?, acreage_value),
                        acreage_unit = CASE WHEN ? = '' THEN acreage_unit ELSE ? END,
                        growth_stage = CASE WHEN ? = '' THEN growth_stage ELSE ? END,
                        cultivation_type = CASE WHEN ? = '' THEN cultivation_type ELSE ? END,
                        season_or_year = CASE WHEN ? = '' THEN season_or_year ELSE ? END,
                        status = ?, last_confirmed_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        crop, crop,
                        _clean(variety, limit=100), _clean(variety, limit=100),
                        _clean(region, limit=200), _clean(region, limit=200),
                        float(acreage_value) if acreage_value is not None else None,
                        _clean(acreage_unit, limit=30), _clean(acreage_unit, limit=30),
                        _clean(growth_stage, limit=100), _clean(growth_stage, limit=100),
                        _clean(cultivation_type, limit=100), _clean(cultivation_type, limit=100),
                        season_or_year, season_or_year,
                        new_status, confirmed_at, now, existing["id"],
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM customer_growing_contexts WHERE id = ?", (existing["id"],)
                ).fetchone()
                return _growing_context_from_row(row), False
            item = CustomerGrowingContext(
                id=_new_id("growing_context"),
                customer_profile_id=customer_profile_id,
                crop=crop,
                variety=_clean(variety, limit=100),
                region=_clean(region, limit=200),
                acreage_value=float(acreage_value) if acreage_value is not None else None,
                acreage_unit=_clean(acreage_unit, limit=30),
                growth_stage=_clean(growth_stage, limit=100),
                cultivation_type=_clean(cultivation_type, limit=100),
                season_or_year=season_or_year,
                status=status,
                first_seen_at=confirmed_at,
                last_confirmed_at=confirmed_at,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO customer_growing_contexts(
                    id, customer_profile_id, crop, variety, region, acreage_value,
                    acreage_unit, growth_stage, cultivation_type, season_or_year,
                    status, first_seen_at, last_confirmed_at, created_at, updated_at
                ) VALUES (
                    :id, :customer_profile_id, :crop, :variety, :region, :acreage_value,
                    :acreage_unit, :growth_stage, :cultivation_type, :season_or_year,
                    :status, :first_seen_at, :last_confirmed_at, :created_at, :updated_at
                )
                """,
                _growing_context_params(item),
            )
            return item, True

    def add_or_confirm_fact(
        self,
        customer_profile_id: str,
        *,
        fact_category: str,
        fact_key: str,
        value_text: str,
        growing_context_id: str | None = None,
        value_json: str | dict[str, Any] | list[Any] | None = None,
        normalized_value: str | None = None,
        confidence: float = 1.0,
        provenance: CustomerFactProvenance | str = CustomerFactProvenance.CUSTOMER_EXPLICIT,
        status: CustomerFactStatus | str = CustomerFactStatus.ACTIVE,
        first_seen_at: str | None = None,
        valid_from: str | None = None,
        resolution: str = "conflict",
        source_message_id: str | None = None,
        source_conversation_id: str | None = None,
        source_text: str = "",
        source_observed_at: str | None = None,
    ) -> tuple[CustomerFact, bool]:
        """Insert, confirm, conflict, or supersede a fact without overwriting history."""
        category = _clean(fact_category, required=True, limit=100)
        key = _clean(fact_key, required=True, limit=100)
        value = _clean(value_text, required=True, limit=2000)
        normal = _normalise_fact(normalized_value or value)
        if not normal:
            raise ValueError("normalized fact value is required")
        json_value = _json_text(value_json)
        incoming = canonicalize_fact_identity(
            customer_profile_id=customer_profile_id,
            growing_context_id=growing_context_id,
            fact_category=category,
            fact_key=key,
            value_text=value,
            normalized_value=normal,
            value_json=json_value,
            source_text=source_text,
        )
        category = incoming.canonical_category
        key = incoming.canonical_key
        normal = incoming.canonical_value
        confidence = _confidence(confidence)
        provenance = CustomerFactProvenance(provenance)
        requested_status = CustomerFactStatus(status)
        if resolution not in {"conflict", "supersede", "coexist"}:
            raise ValueError("resolution must be conflict, supersede, or coexist")
        authoritative = provenance in {
            CustomerFactProvenance.CUSTOMER_EXPLICIT,
            CustomerFactProvenance.SYSTEM_IMPORTED,
        }
        effective_status = requested_status
        if requested_status == CustomerFactStatus.ACTIVE and not authoritative:
            effective_status = CustomerFactStatus.UNCERTAIN
        if provenance == CustomerFactProvenance.AI_INFERRED:
            confidence = min(confidence, 0.69)
        if resolution == "supersede" and not authoritative:
            raise ValueError("only customer-explicit or system-imported facts may supersede an active fact")
        now = utc_now_iso()
        first_seen_was_supplied = first_seen_at is not None
        first_seen_at = first_seen_at or source_observed_at or now
        last_confirmed_at = source_observed_at or first_seen_at
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            profile = self._require_profile(conn, customer_profile_id)
            self._validate_context(conn, customer_profile_id, growing_context_id)
            if source_message_id and source_observed_at is None:
                source_message = conn.execute(
                    "SELECT observed_at FROM messages WHERE id = ?", (source_message_id,)
                ).fetchone()
                if source_message is not None:
                    source_observed_at = source_message["observed_at"]
                    if not first_seen_was_supplied:
                        first_seen_at = source_observed_at
                    last_confirmed_at = source_observed_at
            rows = conn.execute(
                """
                SELECT * FROM customer_facts
                WHERE customer_profile_id = ? AND growing_context_id IS ?
                  AND status IN ('active', 'uncertain', 'conflicting')
                ORDER BY created_at, id
                """,
                (customer_profile_id, growing_context_id),
            ).fetchall()
            candidates = []
            for row in rows:
                identity = canonicalize_fact_identity(
                    customer_profile_id=customer_profile_id,
                    growing_context_id=growing_context_id,
                    fact_category=row["fact_category"],
                    fact_key=row["fact_key"],
                    value_text=row["value_text"],
                    normalized_value=row["normalized_value"],
                    value_json=row["value_json"],
                )
                if (
                    identity.canonical_category == category
                    and identity.canonical_key == key
                    and identity.temporal_scope == incoming.temporal_scope
                ):
                    candidates.append((row, identity))
            exact_pair = next(
                (pair for pair in candidates if pair[1].canonical_value == normal),
                None,
            )
            compatible_pair = None
            if exact_pair is None and key == "growth_stage":
                compatible_pair = next(
                    (
                        pair for pair in candidates
                        if growth_stage_relation(
                            pair[1].canonical_value,
                            normal,
                            source_text=source_text,
                        ) in {"same", "compatible"}
                    ),
                    None,
                )
            matched_pair = exact_pair or compatible_pair
            if matched_pair is not None:
                exact, existing_identity = matched_pair
                compatible = exact_pair is None
                exact_status = CustomerFactStatus(exact["status"])
                new_status = exact_status
                other_values = [row for row, _ in candidates if row["id"] != exact["id"]]
                if authoritative and exact_status == CustomerFactStatus.UNCERTAIN and not other_values:
                    new_status = CustomerFactStatus.ACTIVE
                new_provenance = _stronger_provenance(CustomerFactProvenance(exact["provenance"]), provenance)
                next_value = exact["value_text"] if compatible else value
                next_json = exact["value_json"] if compatible else json_value
                next_normal = existing_identity.canonical_value if compatible else normal
                conn.execute(
                    """
                    UPDATE customer_facts
                    SET fact_category = ?, fact_key = ?, value_text = ?, value_json = ?,
                        normalized_value = ?, confidence = MAX(confidence, ?),
                        provenance = ?, status = ?, last_confirmed_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        category, key, next_value, next_json, next_normal,
                        confidence, new_provenance.value,
                        new_status.value, last_confirmed_at, now, exact["id"],
                    ),
                )
                item = _fact_from_row(conn.execute("SELECT * FROM customer_facts WHERE id = ?", (exact["id"],)).fetchone())
                self._add_fact_source_on_connection(
                    conn,
                    fact=item,
                    profile=profile,
                    message_id=source_message_id,
                    conversation_id=source_conversation_id,
                    source_text=source_text,
                    observed_at=source_observed_at,
                )
                return item, False

            conflict_group_id: str | None = None
            new_status = effective_status
            authoritative_candidates = [
                row for row, _ in candidates
                if CustomerFactProvenance(row["provenance"])
                in {
                    CustomerFactProvenance.CUSTOMER_EXPLICIT,
                    CustomerFactProvenance.USER_EXPLICIT,
                    CustomerFactProvenance.SYSTEM_IMPORTED,
                }
            ]
            effective_resolution = resolution
            supersede_candidates = [row for row, _ in candidates]
            if effective_resolution == "supersede" and key == "growth_stage":
                supersede_candidates = [
                    row
                    for row, identity in candidates
                    if growth_stage_relation(
                        identity.canonical_value,
                        normal,
                        source_text=source_text,
                    ) == "advance"
                ]
                if not supersede_candidates:
                    effective_resolution = "coexist"
                    new_status = CustomerFactStatus.UNCERTAIN
            if effective_resolution == "conflict" and authoritative and authoritative_candidates:
                conflict_group_id = next(
                    (row["conflict_group_id"] for row in authoritative_candidates if row["conflict_group_id"]),
                    _new_id("fact_conflict"),
                )
                ids = [row["id"] for row in authoritative_candidates]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE customer_facts SET status = 'conflicting', conflict_group_id = ?, updated_at = ? WHERE id IN ({placeholders})",
                    (conflict_group_id, now, *ids),
                )
                new_status = CustomerFactStatus.CONFLICTING

            item = CustomerFact(
                id=_new_id("customer_fact"),
                customer_profile_id=customer_profile_id,
                growing_context_id=growing_context_id,
                fact_category=category,
                fact_key=key,
                value_text=value,
                value_json=json_value,
                normalized_value=normal,
                confidence=confidence,
                provenance=provenance,
                status=new_status,
                first_seen_at=first_seen_at,
                last_confirmed_at=last_confirmed_at,
                valid_from=valid_from or first_seen_at,
                valid_to=None,
                conflict_group_id=conflict_group_id,
                superseded_by_fact_id=None,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO customer_facts(
                    id, customer_profile_id, growing_context_id, fact_category, fact_key,
                    value_text, value_json, normalized_value, confidence, provenance, status,
                    first_seen_at, last_confirmed_at, valid_from, valid_to,
                    conflict_group_id, superseded_by_fact_id, created_at, updated_at
                ) VALUES (
                    :id, :customer_profile_id, :growing_context_id, :fact_category, :fact_key,
                    :value_text, :value_json, :normalized_value, :confidence, :provenance, :status,
                    :first_seen_at, :last_confirmed_at, :valid_from, :valid_to,
                    :conflict_group_id, :superseded_by_fact_id, :created_at, :updated_at
                )
                """,
                _fact_params(item),
            )
            if effective_resolution == "supersede":
                for row in supersede_candidates:
                    conn.execute(
                        """
                        UPDATE customer_facts
                        SET status = 'superseded', valid_to = ?, superseded_by_fact_id = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (item.valid_from or now, item.id, now, row["id"]),
                    )
            self._add_fact_source_on_connection(
                conn,
                fact=item,
                profile=profile,
                message_id=source_message_id,
                conversation_id=source_conversation_id,
                source_text=source_text,
                observed_at=source_observed_at,
            )
            return item, True

    def list_active_facts(
        self,
        customer_profile_id: str,
        *,
        growing_context_id: str | None = None,
    ) -> list[CustomerFact]:
        sql = "SELECT * FROM customer_facts WHERE customer_profile_id = ? AND status = 'active'"
        params: list[Any] = [customer_profile_id]
        if growing_context_id is not None:
            sql += " AND growing_context_id = ?"
            params.append(growing_context_id)
        sql += " ORDER BY fact_category, fact_key, last_confirmed_at DESC"
        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_fact_from_row(row) for row in rows]

    def list_growing_contexts(
        self,
        customer_profile_id: str,
        *,
        include_historical: bool = False,
    ) -> list[CustomerGrowingContext]:
        sql = "SELECT * FROM customer_growing_contexts WHERE customer_profile_id = ?"
        params: list[Any] = [customer_profile_id]
        if not include_historical:
            sql += " AND status <> 'historical'"
        sql += " ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'uncertain' THEN 1 ELSE 2 END, last_confirmed_at DESC, created_at DESC"
        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_growing_context_from_row(row) for row in rows]

    def list_facts(
        self,
        customer_profile_id: str,
        *,
        statuses: tuple[CustomerFactStatus | str, ...] | None = None,
        growing_context_id: str | None = None,
    ) -> list[CustomerFact]:
        sql = "SELECT * FROM customer_facts WHERE customer_profile_id = ?"
        params: list[Any] = [customer_profile_id]
        if statuses:
            values = [CustomerFactStatus(item).value for item in statuses]
            sql += f" AND status IN ({','.join('?' for _ in values)})"
            params.extend(values)
        if growing_context_id is not None:
            sql += " AND growing_context_id = ?"
            params.append(growing_context_id)
        sql += " ORDER BY last_confirmed_at DESC, created_at DESC, id DESC"
        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_fact_from_row(row) for row in rows]

    def list_tracking_evidence(
        self,
        customer_profile_id: str,
        *,
        since_iso: str,
        fact_limit: int = 120,
        event_limit: int = 160,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return bounded structured evidence without scanning raw chat history.

        Fact/event source text is copied into dedicated source tables at
        extraction time, so the tracking read path never needs to join or scan
        ``historical_chat_messages``.  Current and previous season rows are
        enough for continuity reasoning while keeping snapshot construction
        deterministic and bounded.
        """
        with self.db.connect() as conn:
            fact_rows = conn.execute(
                """
                SELECT
                    f.id,
                    f.growing_context_id,
                    f.fact_category AS category,
                    f.fact_key AS evidence_key,
                    f.value_text AS text,
                    f.normalized_value,
                    f.value_json AS structured_json,
                    f.last_confirmed_at AS occurred_at,
                    f.confidence,
                    f.provenance,
                    f.status,
                    COALESCE(group_concat(fs.source_text, ' || '), '') AS source_text
                FROM customer_facts AS f
                LEFT JOIN customer_fact_sources AS fs ON fs.fact_id=f.id
                WHERE f.customer_profile_id=?
                  AND f.status IN ('active','uncertain','conflicting')
                  AND f.last_confirmed_at>=?
                GROUP BY f.id
                ORDER BY f.last_confirmed_at DESC, f.created_at DESC, f.id DESC
                LIMIT ?
                """,
                (customer_profile_id, since_iso, max(1, min(int(fact_limit), 300))),
            ).fetchall()
            event_rows = conn.execute(
                """
                SELECT
                    e.id,
                    e.growing_context_id,
                    e.event_type AS category,
                    e.event_type AS evidence_key,
                    e.title || CASE WHEN e.summary='' THEN '' ELSE '：' || e.summary END AS text,
                    '' AS normalized_value,
                    e.structured_data_json AS structured_json,
                    e.event_at AS occurred_at,
                    e.confidence,
                    e.provenance,
                    e.status,
                    COALESCE(group_concat(es.source_text, ' || '), '') AS source_text
                FROM customer_events AS e
                LEFT JOIN customer_event_sources AS es ON es.event_id=e.id
                WHERE e.customer_profile_id=?
                  AND e.status<>'invalid'
                  AND e.event_at>=?
                GROUP BY e.id
                ORDER BY e.event_at DESC, e.created_at DESC, e.id DESC
                LIMIT ?
                """,
                (customer_profile_id, since_iso, max(1, min(int(event_limit), 400))),
            ).fetchall()
        return {
            "facts": [dict(row) for row in fact_rows],
            "events": [dict(row) for row in event_rows],
        }

    def get_fact(self, fact_id: str) -> CustomerFact | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
        return _fact_from_row(row) if row else None

    def list_fact_source_details(self, fact_id: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT sources.*, messages.speaker AS message_speaker,
                       historical.sender AS historical_sender
                FROM customer_fact_sources AS sources
                LEFT JOIN messages ON messages.id = sources.message_id
                LEFT JOIN historical_chat_messages AS historical
                  ON historical.historical_message_id = sources.historical_message_id
                WHERE sources.fact_id = ?
                ORDER BY sources.observed_at DESC, sources.created_at DESC, sources.id DESC
                """,
                (fact_id,),
            ).fetchall()
        return [
            {
                **{
                    key: row[key] for key in row.keys()
                    if key not in {"message_speaker", "historical_sender"}
                },
                "sender": row["message_speaker"] or row["historical_sender"] or "manual",
            }
            for row in rows
        ]

    def list_memory_edits(self, customer_profile_id: str, *, limit: int = 100) -> list[CustomerMemoryEdit]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM customer_memory_edits WHERE customer_profile_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (customer_profile_id, max(1, int(limit))),
            ).fetchall()
        return [CustomerMemoryEdit(**dict(row)) for row in rows]

    def confirm_fact(
        self,
        fact_id: str,
        *,
        reason: str = "",
        actor: str = "local_user",
        failure_injector: Callable[[str], None] | None = None,
    ) -> CustomerFact:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                raise ValueError("fact does not exist")
            if row["status"] not in {"active", "uncertain", "conflicting"}:
                raise ValueError("only active, uncertain, or conflicting facts can be confirmed")
            profile = self._require_profile(conn, row["customer_profile_id"])
            old_snapshot = _fact_snapshot(row)
            identity_rows = self._fact_identity_rows(conn, row)
            for other in identity_rows:
                if other["id"] == fact_id:
                    continue
                conn.execute(
                    """
                    UPDATE customer_facts
                    SET status='superseded', valid_to=?, superseded_by_fact_id=?, updated_at=?
                    WHERE id=?
                    """,
                    (now, fact_id, now, other["id"]),
                )
            conn.execute(
                """
                UPDATE customer_facts
                SET provenance='user_explicit', status='active', confidence=MAX(confidence, 0.99),
                    conflict_group_id=NULL, superseded_by_fact_id=NULL, valid_to=NULL,
                    last_confirmed_at=?, updated_at=?
                WHERE id=?
                """,
                (now, now, fact_id),
            )
            current = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
            self._add_manual_fact_source(conn, current, profile, "人工确认", reason, now)
            self._sync_context_from_fact(conn, current, clear=False, now=now)
            if failure_injector:
                failure_injector("fact")
            self._insert_memory_edit(
                conn,
                profile_id=row["customer_profile_id"],
                entity_id=fact_id,
                operation="confirm",
                old_value=old_snapshot,
                new_value=_fact_snapshot(current),
                reason=reason,
                actor=actor,
                created_at=now,
            )
            conn.execute("UPDATE customer_profiles SET updated_at=? WHERE id=?", (now, row["customer_profile_id"]))
            if failure_injector:
                failure_injector("audit")
            return _fact_from_row(current)

    def correct_fact(
        self,
        fact_id: str,
        *,
        value_text: str,
        reason: str = "",
        actor: str = "local_user",
        failure_injector: Callable[[str], None] | None = None,
    ) -> tuple[CustomerFact, CustomerFact]:
        value = _clean(value_text, required=True, limit=2000)
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
            if old is None:
                raise ValueError("fact does not exist")
            if old["status"] == CustomerFactStatus.INVALID.value:
                raise ValueError("an invalid fact cannot be corrected")
            profile = self._require_profile(conn, old["customer_profile_id"])
            value_json: dict[str, Any] = {}
            if canonicalize_fact_identity(
                customer_profile_id=old["customer_profile_id"],
                growing_context_id=old["growing_context_id"],
                fact_category=old["fact_category"],
                fact_key=old["fact_key"],
                value_text=old["value_text"],
                normalized_value=old["normalized_value"],
                value_json=old["value_json"],
            ).canonical_key == "acreage":
                parsed = extract_acreage(value)
                if parsed is None:
                    raise ValueError("面积必须包含数值和单位，例如 30亩 或 3个棚")
                value_json = {"value": parsed[0], "unit": parsed[1]}
            incoming = canonicalize_fact_identity(
                customer_profile_id=old["customer_profile_id"],
                growing_context_id=old["growing_context_id"],
                fact_category=old["fact_category"],
                fact_key=old["fact_key"],
                value_text=value,
                normalized_value=value,
                value_json=value_json,
                source_text=value,
            )
            identity_rows = self._fact_identity_rows(conn, old)
            replacement = next(
                (
                    item for item in identity_rows
                    if canonicalize_fact_identity(
                        customer_profile_id=item["customer_profile_id"],
                        growing_context_id=item["growing_context_id"],
                        fact_category=item["fact_category"],
                        fact_key=item["fact_key"],
                        value_text=item["value_text"],
                        normalized_value=item["normalized_value"],
                        value_json=item["value_json"],
                    ).canonical_value == incoming.canonical_value
                ),
                None,
            )
            if replacement is None:
                replacement_id = _new_id("customer_fact")
                conn.execute(
                    """
                    INSERT INTO customer_facts(
                        id, customer_profile_id, growing_context_id, fact_category, fact_key,
                        value_text, value_json, normalized_value, confidence, provenance, status,
                        first_seen_at, last_confirmed_at, valid_from, valid_to, conflict_group_id,
                        superseded_by_fact_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, 'user_explicit', 'active', ?, ?, ?, NULL, NULL, NULL, ?, ?)
                    """,
                    (
                        replacement_id,
                        old["customer_profile_id"],
                        old["growing_context_id"],
                        incoming.canonical_category,
                        incoming.canonical_key,
                        value,
                        _json_text(value_json),
                        incoming.canonical_value,
                        now,
                        now,
                        now,
                        now,
                        now,
                    ),
                )
            else:
                replacement_id = replacement["id"]
                conn.execute(
                    """
                    UPDATE customer_facts
                    SET value_text=?, value_json=?, normalized_value=?, provenance='user_explicit',
                        status='active', confidence=1.0, conflict_group_id=NULL,
                        superseded_by_fact_id=NULL, valid_to=NULL, last_confirmed_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (value, _json_text(value_json), incoming.canonical_value, now, now, replacement_id),
                )
            for item in identity_rows:
                if item["id"] == replacement_id:
                    continue
                conn.execute(
                    """
                    UPDATE customer_facts
                    SET status='superseded', valid_to=?, superseded_by_fact_id=?, updated_at=?
                    WHERE id=?
                    """,
                    (now, replacement_id, now, item["id"]),
                )
            current = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (replacement_id,)).fetchone()
            self._add_manual_fact_source(conn, current, profile, "人工纠正", reason, now)
            self._sync_context_from_fact(conn, current, clear=False, now=now)
            if failure_injector:
                failure_injector("fact")
            self._insert_memory_edit(
                conn,
                profile_id=old["customer_profile_id"],
                entity_id=fact_id,
                operation="correct",
                old_value=_fact_snapshot(old),
                new_value=_fact_snapshot(current),
                reason=reason,
                actor=actor,
                created_at=now,
            )
            conn.execute("UPDATE customer_profiles SET updated_at=? WHERE id=?", (now, old["customer_profile_id"]))
            if failure_injector:
                failure_injector("audit")
            old_after = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
            return _fact_from_row(old_after), _fact_from_row(current)

    def invalidate_fact(
        self,
        fact_id: str,
        *,
        reason: str = "",
        actor: str = "local_user",
        failure_injector: Callable[[str], None] | None = None,
    ) -> CustomerFact:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                raise ValueError("fact does not exist")
            profile = self._require_profile(conn, row["customer_profile_id"])
            old_snapshot = _fact_snapshot(row)
            conn.execute(
                "UPDATE customer_facts SET status='invalid', valid_to=?, conflict_group_id=NULL, updated_at=? WHERE id=?",
                (now, now, fact_id),
            )
            current = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
            self._add_manual_fact_source(conn, current, profile, "人工标记无效", reason, now)
            self._sync_context_from_fact(conn, current, clear=True, now=now)
            if failure_injector:
                failure_injector("fact")
            self._insert_memory_edit(
                conn,
                profile_id=row["customer_profile_id"],
                entity_id=fact_id,
                operation="invalidate",
                old_value=old_snapshot,
                new_value=_fact_snapshot(current),
                reason=reason,
                actor=actor,
                created_at=now,
            )
            conn.execute("UPDATE customer_profiles SET updated_at=? WHERE id=?", (now, row["customer_profile_id"]))
            if failure_injector:
                failure_injector("audit")
            return _fact_from_row(current)

    def list_conflicting_facts(self, customer_profile_id: str) -> list[CustomerFact]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM customer_facts WHERE customer_profile_id = ? AND status = 'conflicting' ORDER BY conflict_group_id, created_at, id",
                (customer_profile_id,),
            ).fetchall()
        return [_fact_from_row(row) for row in rows]

    def supersede_fact(
        self,
        old_fact_id: str,
        new_fact_id: str,
        *,
        valid_to: str | None = None,
    ) -> tuple[CustomerFact, CustomerFact]:
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (old_fact_id,)).fetchone()
            new = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (new_fact_id,)).fetchone()
            if old is None or new is None:
                raise ValueError("both old and new facts must exist")
            identity = ("customer_profile_id", "growing_context_id", "fact_category", "fact_key")
            if any(old[column] != new[column] for column in identity):
                raise ValueError("facts must belong to the same profile, context, category, and key")
            provenance = CustomerFactProvenance(new["provenance"])
            if provenance not in {
                CustomerFactProvenance.CUSTOMER_EXPLICIT,
                CustomerFactProvenance.USER_EXPLICIT,
                CustomerFactProvenance.SYSTEM_IMPORTED,
            }:
                raise ValueError("the replacement fact is not authoritative")
            conn.execute(
                "UPDATE customer_facts SET status = 'active', conflict_group_id = NULL, updated_at = ? WHERE id = ?",
                (now, new_fact_id),
            )
            conn.execute(
                """
                UPDATE customer_facts
                SET status = 'superseded', valid_to = ?, superseded_by_fact_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (valid_to or new["valid_from"] or now, new_fact_id, now, old_fact_id),
            )
            old = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (old_fact_id,)).fetchone()
            new = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (new_fact_id,)).fetchone()
        return _fact_from_row(old), _fact_from_row(new)

    def add_fact_source(
        self,
        fact_id: str,
        *,
        message_id: str | None = None,
        conversation_id: str | None = None,
        source_text: str = "",
        observed_at: str | None = None,
    ) -> tuple[CustomerFactSource, bool]:
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM customer_facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                raise ValueError("fact does not exist")
            fact = _fact_from_row(row)
            profile = self._require_profile(conn, fact.customer_profile_id)
            result = self._add_fact_source_on_connection(
                conn,
                fact=fact,
                profile=profile,
                message_id=message_id,
                conversation_id=conversation_id,
                source_text=source_text,
                observed_at=observed_at,
                required=True,
            )
            assert result is not None
            return result

    def list_fact_sources(self, fact_id: str) -> list[CustomerFactSource]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM customer_fact_sources WHERE fact_id = ? ORDER BY observed_at, created_at, id",
                (fact_id,),
            ).fetchall()
        return [_fact_source_from_row(row) for row in rows]

    def create_customer_event(
        self,
        customer_profile_id: str,
        *,
        event_type: CustomerEventType | str,
        title: str,
        summary: str = "",
        structured_data_json: str | dict[str, Any] | list[Any] | None = None,
        event_at: str | None = None,
        confidence: float = 1.0,
        provenance: CustomerFactProvenance | str = CustomerFactProvenance.CUSTOMER_EXPLICIT,
        status: CustomerEventStatus | str = CustomerEventStatus.ACTIVE,
        growing_context_id: str | None = None,
    ) -> CustomerEvent:
        event_type = CustomerEventType(event_type)
        provenance = CustomerFactProvenance(provenance)
        status = CustomerEventStatus(status)
        confidence = _confidence(confidence)
        if provenance == CustomerFactProvenance.AI_INFERRED:
            confidence = min(confidence, 0.69)
            if status == CustomerEventStatus.ACTIVE:
                status = CustomerEventStatus.UNCERTAIN
        now = utc_now_iso()
        item = CustomerEvent(
            id=_new_id("customer_event"),
            customer_profile_id=customer_profile_id,
            growing_context_id=growing_context_id,
            event_type=event_type,
            title=_clean(title, required=True, limit=300),
            summary=_clean(summary, limit=4000),
            structured_data_json=_json_text(structured_data_json),
            event_at=event_at or now,
            confidence=confidence,
            provenance=provenance,
            status=status,
            created_at=now,
            updated_at=now,
        )
        with self.db.connect() as conn:
            profile = self._require_profile(conn, customer_profile_id)
            self._validate_context(conn, customer_profile_id, growing_context_id)
            conn.execute(
                """
                INSERT INTO customer_events(
                    id, customer_profile_id, growing_context_id, event_type, title,
                    summary, structured_data_json, event_at, confidence, provenance,
                    status, created_at, updated_at
                ) VALUES (
                    :id, :customer_profile_id, :growing_context_id, :event_type, :title,
                    :summary, :structured_data_json, :event_at, :confidence, :provenance,
                    :status, :created_at, :updated_at
                )
                """,
                _event_params(item),
            )
        return item

    def list_customer_events(
        self,
        customer_profile_id: str,
        *,
        event_type: CustomerEventType | str | None = None,
        include_invalid: bool = False,
        limit: int = 500,
    ) -> list[CustomerEvent]:
        sql = "SELECT * FROM customer_events WHERE customer_profile_id = ?"
        params: list[Any] = [customer_profile_id]
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(CustomerEventType(event_type).value)
        if not include_invalid:
            sql += " AND status <> 'invalid'"
        sql += " ORDER BY event_at DESC, created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_event_from_row(row) for row in rows]

    def add_event_source(
        self,
        event_id: str,
        *,
        message_id: str | None = None,
        conversation_id: str | None = None,
        source_text: str = "",
        observed_at: str | None = None,
    ) -> tuple[CustomerEventSource, bool]:
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM customer_events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                raise ValueError("event does not exist")
            event = _event_from_row(row)
            profile = self._require_profile(conn, event.customer_profile_id)
            return self._add_event_source_on_connection(
                conn,
                event=event,
                profile=profile,
                message_id=message_id,
                conversation_id=conversation_id,
                source_text=source_text,
                observed_at=observed_at,
            )

    def list_event_sources(self, event_id: str) -> list[CustomerEventSource]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM customer_event_sources WHERE event_id = ? ORDER BY observed_at, created_at, id",
                (event_id,),
            ).fetchall()
        return [_event_source_from_row(row) for row in rows]

    def list_event_source_details_bulk(self, event_ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        ids = tuple(dict.fromkeys(str(item) for item in event_ids if item))
        if not ids:
            return []
        result: list[dict[str, Any]] = []
        with self.db.connect() as conn:
            for offset in range(0, len(ids), 400):
                chunk = ids[offset : offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT sources.*, messages.speaker AS message_speaker,
                           historical.sender AS historical_sender,
                           historical.text AS historical_text,
                           historical.sent_at AS historical_sent_at
                    FROM customer_event_sources AS sources
                    LEFT JOIN messages ON messages.id=sources.message_id
                    LEFT JOIN historical_chat_messages AS historical
                      ON historical.historical_message_id=sources.historical_message_id
                    WHERE sources.event_id IN ({placeholders})
                    ORDER BY sources.observed_at, sources.created_at, sources.id
                    """,
                    chunk,
                ).fetchall()
                result.extend(dict(row) for row in rows)
        return result

    def create_followup(
        self,
        customer_profile_id: str,
        *,
        followup_type: str,
        title: str,
        due_at: str,
        growing_context_id: str | None = None,
        related_event_id: str | None = None,
        status: CustomerFollowupStatus | str = CustomerFollowupStatus.PENDING,
        priority: int = 50,
        completed_at: str | None = None,
        cancelled_at: str | None = None,
        source: CustomerFollowupSource | str = CustomerFollowupSource.EXPLICIT,
        source_message_id: str | None = None,
        reason: str = "",
    ) -> CustomerFollowup:
        status = CustomerFollowupStatus(status)
        source = CustomerFollowupSource(source)
        priority = int(priority)
        if not 0 <= priority <= 100:
            raise ValueError("priority must be between 0 and 100")
        if status == CustomerFollowupStatus.COMPLETED and completed_at is None:
            completed_at = utc_now_iso()
        if status != CustomerFollowupStatus.COMPLETED and completed_at is not None:
            raise ValueError("completed_at is only valid for a completed followup")
        if status == CustomerFollowupStatus.CANCELLED and cancelled_at is None:
            cancelled_at = utc_now_iso()
        if status != CustomerFollowupStatus.CANCELLED and cancelled_at is not None:
            raise ValueError("cancelled_at is only valid for a cancelled followup")
        now = utc_now_iso()
        item = CustomerFollowup(
            id=_new_id("customer_followup"),
            customer_profile_id=customer_profile_id,
            growing_context_id=growing_context_id,
            related_event_id=related_event_id,
            followup_type=_clean(followup_type, required=True, limit=100),
            title=_clean(title, required=True, limit=300),
            due_at=_clean(due_at, required=True, limit=80),
            status=status,
            priority=priority,
            completed_at=completed_at,
            cancelled_at=cancelled_at,
            source=source,
            source_message_id=source_message_id,
            reason=_clean(reason, limit=1000),
            created_at=now,
            updated_at=now,
        )
        with self.db.connect() as conn:
            profile = self._require_profile(conn, customer_profile_id)
            self._validate_context(conn, customer_profile_id, growing_context_id)
            if related_event_id:
                event = conn.execute(
                    "SELECT customer_profile_id FROM customer_events WHERE id = ?", (related_event_id,)
                ).fetchone()
                if event is None or event["customer_profile_id"] != customer_profile_id:
                    raise ValueError("related event does not belong to the customer profile")
            self._validate_followup_source_message(conn, profile, source_message_id)
            conn.execute(
                """
                INSERT INTO customer_followups(
                    id, customer_profile_id, growing_context_id, related_event_id,
                    followup_type, title, due_at, status, priority, completed_at,
                    cancelled_at, source, source_message_id, reason, created_at, updated_at
                ) VALUES (
                    :id, :customer_profile_id, :growing_context_id, :related_event_id,
                    :followup_type, :title, :due_at, :status, :priority, :completed_at,
                    :cancelled_at, :source, :source_message_id, :reason, :created_at, :updated_at
                )
                """,
                _followup_params(item),
            )
            self._refresh_next_followup(conn, customer_profile_id, now)
        return item

    def list_pending_followups(
        self,
        customer_profile_id: str | None = None,
        *,
        due_before: str | None = None,
        limit: int = 500,
    ) -> list[CustomerFollowup]:
        sql = "SELECT * FROM customer_followups WHERE status = 'pending'"
        params: list[Any] = []
        if customer_profile_id is not None:
            sql += " AND customer_profile_id = ?"
            params.append(customer_profile_id)
        if due_before is not None:
            sql += " AND due_at <= ?"
            params.append(due_before)
        sql += " ORDER BY due_at ASC, priority DESC, created_at ASC LIMIT ?"
        params.append(max(1, int(limit)))
        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_followup_from_row(row) for row in rows]

    def list_customer_profiles(self, *, limit: int = 1000) -> list[CustomerProfile]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM customer_profiles ORDER BY updated_at DESC, id LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [_profile_from_row(row) for row in rows]

    def get_customer_event(self, event_id: str) -> CustomerEvent | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM customer_events WHERE id = ?", (event_id,)).fetchone()
        return _event_from_row(row) if row else None

    def set_sales_stage(
        self,
        customer_profile_id: str,
        stage: CustomerSalesStage | str,
        *,
        source: CustomerSalesStageSource | str,
        reason: str,
        source_event_id: str | None = None,
        changed_at: str | None = None,
    ) -> tuple[CustomerProfile, bool]:
        stage = CustomerSalesStage(stage)
        source = CustomerSalesStageSource(source)
        changed = changed_at or utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._require_profile(conn, customer_profile_id)
            if source_event_id:
                event = conn.execute(
                    "SELECT customer_profile_id FROM customer_events WHERE id = ?",
                    (source_event_id,),
                ).fetchone()
                if event is None or event["customer_profile_id"] != customer_profile_id:
                    raise ValueError("source event does not belong to the customer profile")
            old_stage = CustomerSalesStage(row["sales_stage"])
            old_source = CustomerSalesStageSource(row["sales_stage_source"])
            if old_stage == stage and old_source == source:
                return _profile_from_row(row), False
            conn.execute(
                """
                UPDATE customer_profiles
                SET sales_stage=?, sales_stage_source=?, sales_stage_updated_at=?, updated_at=?
                WHERE id=?
                """,
                (stage.value, source.value, changed, changed, customer_profile_id),
            )
            history = CustomerSalesStageHistory(
                id=_new_id("sales_stage_history"),
                customer_profile_id=customer_profile_id,
                old_stage=old_stage,
                new_stage=stage,
                reason=_clean(reason, required=True, limit=1000),
                source_event_id=source_event_id,
                source=source,
                changed_at=changed,
            )
            conn.execute(
                """
                INSERT INTO customer_sales_stage_history(
                    id,customer_profile_id,old_stage,new_stage,reason,
                    source_event_id,source,changed_at
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    history.id,
                    history.customer_profile_id,
                    history.old_stage.value,
                    history.new_stage.value,
                    history.reason,
                    history.source_event_id,
                    history.source.value,
                    history.changed_at,
                ),
            )
            updated = conn.execute("SELECT * FROM customer_profiles WHERE id=?", (customer_profile_id,)).fetchone()
            return _profile_from_row(updated), True

    def list_sales_stage_history(
        self,
        customer_profile_id: str,
        *,
        limit: int = 100,
    ) -> list[CustomerSalesStageHistory]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM customer_sales_stage_history
                WHERE customer_profile_id=?
                ORDER BY changed_at DESC,id DESC LIMIT ?
                """,
                (customer_profile_id, max(1, int(limit))),
            ).fetchall()
        return [_sales_stage_history_from_row(row) for row in rows]

    def get_followup(self, followup_id: str) -> CustomerFollowup | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM customer_followups WHERE id=?", (followup_id,)).fetchone()
        return _followup_from_row(row) if row else None

    def list_followups(
        self,
        customer_profile_id: str | None = None,
        *,
        status: CustomerFollowupStatus | str | None = None,
        limit: int = 1000,
    ) -> list[CustomerFollowup]:
        sql = "SELECT * FROM customer_followups WHERE 1=1"
        params: list[Any] = []
        if customer_profile_id:
            sql += " AND customer_profile_id=?"
            params.append(customer_profile_id)
        if status is not None:
            sql += " AND status=?"
            params.append(CustomerFollowupStatus(status).value)
        sql += " ORDER BY due_at ASC,priority DESC,created_at ASC LIMIT ?"
        params.append(max(1, int(limit)))
        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_followup_from_row(row) for row in rows]

    def upsert_pending_followup(
        self,
        customer_profile_id: str,
        *,
        followup_type: str,
        title: str,
        due_at: str,
        growing_context_id: str | None = None,
        related_event_id: str | None = None,
        priority: int = 50,
        source: CustomerFollowupSource | str = CustomerFollowupSource.SYSTEM_RULE,
        source_message_id: str | None = None,
        reason: str = "",
    ) -> tuple[CustomerFollowup, bool]:
        source = CustomerFollowupSource(source)
        priority = int(priority)
        if not 0 <= priority <= 100:
            raise ValueError("priority must be between 0 and 100")
        clean_type = _clean(followup_type, required=True, limit=100)
        clean_title = _clean(title, required=True, limit=300)
        clean_due = _clean(due_at, required=True, limit=80)
        clean_reason = _clean(reason, limit=1000)
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            profile = self._require_profile(conn, customer_profile_id)
            self._validate_context(conn, customer_profile_id, growing_context_id)
            if related_event_id:
                event = conn.execute(
                    "SELECT customer_profile_id FROM customer_events WHERE id=?",
                    (related_event_id,),
                ).fetchone()
                if event is None or event["customer_profile_id"] != customer_profile_id:
                    raise ValueError("related event does not belong to the customer profile")
            self._validate_followup_source_message(conn, profile, source_message_id)
            row = conn.execute(
                """
                SELECT * FROM customer_followups
                WHERE customer_profile_id=? AND growing_context_id IS ?
                  AND followup_type=? AND status='pending'
                ORDER BY created_at,id LIMIT 1
                """,
                (customer_profile_id, growing_context_id, clean_type),
            ).fetchone()
            if row is not None:
                existing_source = CustomerFollowupSource(row["source"])
                if source == CustomerFollowupSource.SYSTEM_RULE and existing_source != CustomerFollowupSource.SYSTEM_RULE:
                    return _followup_from_row(row), False
                merged_due = min(str(row["due_at"]), clean_due) if source == CustomerFollowupSource.SYSTEM_RULE else clean_due
                merged_priority = max(int(row["priority"]), priority) if source == CustomerFollowupSource.SYSTEM_RULE else priority
                conn.execute(
                    """
                    UPDATE customer_followups
                    SET title=?,due_at=?,priority=?,related_event_id=COALESCE(?,related_event_id),
                        source=?,source_message_id=COALESCE(?,source_message_id),reason=?,updated_at=?
                    WHERE id=?
                    """,
                    (
                        clean_title, merged_due, merged_priority, related_event_id,
                        source.value, source_message_id, clean_reason, now, row["id"],
                    ),
                )
                self._refresh_next_followup(conn, customer_profile_id, now)
                updated = conn.execute("SELECT * FROM customer_followups WHERE id=?", (row["id"],)).fetchone()
                return _followup_from_row(updated), False
            item = CustomerFollowup(
                id=_new_id("customer_followup"),
                customer_profile_id=customer_profile_id,
                growing_context_id=growing_context_id,
                related_event_id=related_event_id,
                followup_type=clean_type,
                title=clean_title,
                due_at=clean_due,
                status=CustomerFollowupStatus.PENDING,
                priority=priority,
                completed_at=None,
                cancelled_at=None,
                source=source,
                source_message_id=source_message_id,
                reason=clean_reason,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO customer_followups(
                    id,customer_profile_id,growing_context_id,related_event_id,
                    followup_type,title,due_at,status,priority,completed_at,
                    cancelled_at,source,source_message_id,reason,created_at,updated_at
                ) VALUES (
                    :id,:customer_profile_id,:growing_context_id,:related_event_id,
                    :followup_type,:title,:due_at,:status,:priority,:completed_at,
                    :cancelled_at,:source,:source_message_id,:reason,:created_at,:updated_at
                )
                """,
                _followup_params(item),
            )
            self._refresh_next_followup(conn, customer_profile_id, now)
            return item, True

    def update_followup(
        self,
        followup_id: str,
        *,
        title: str,
        due_at: str,
        priority: int,
    ) -> CustomerFollowup:
        priority = int(priority)
        if not 0 <= priority <= 100:
            raise ValueError("priority must be between 0 and 100")
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM customer_followups WHERE id=?", (followup_id,)).fetchone()
            if row is None:
                raise ValueError("followup does not exist")
            if row["status"] != CustomerFollowupStatus.PENDING.value:
                raise ValueError("only pending followups can be edited")
            conn.execute(
                "UPDATE customer_followups SET title=?,due_at=?,priority=?,updated_at=? WHERE id=?",
                (
                    _clean(title, required=True, limit=300),
                    _clean(due_at, required=True, limit=80),
                    priority,
                    now,
                    followup_id,
                ),
            )
            self._refresh_next_followup(conn, row["customer_profile_id"], now)
            updated = conn.execute("SELECT * FROM customer_followups WHERE id=?", (followup_id,)).fetchone()
            return _followup_from_row(updated)

    def resolve_followup(
        self,
        followup_id: str,
        *,
        status: CustomerFollowupStatus | str,
    ) -> CustomerFollowup:
        status = CustomerFollowupStatus(status)
        if status not in {CustomerFollowupStatus.COMPLETED, CustomerFollowupStatus.CANCELLED}:
            raise ValueError("followup can only be completed or cancelled")
        now = utc_now_iso()
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM customer_followups WHERE id=?", (followup_id,)).fetchone()
            if row is None:
                raise ValueError("followup does not exist")
            if row["status"] != CustomerFollowupStatus.PENDING.value:
                return _followup_from_row(row)
            completed_at = now if status == CustomerFollowupStatus.COMPLETED else None
            cancelled_at = now if status == CustomerFollowupStatus.CANCELLED else None
            conn.execute(
                """
                UPDATE customer_followups
                SET status=?,completed_at=?,cancelled_at=?,updated_at=? WHERE id=?
                """,
                (status.value, completed_at, cancelled_at, now, followup_id),
            )
            self._refresh_next_followup(conn, row["customer_profile_id"], now)
            updated = conn.execute("SELECT * FROM customer_followups WHERE id=?", (followup_id,)).fetchone()
            return _followup_from_row(updated)

    def list_pending_followup_details(self, *, limit: int = 2000) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.*, p.account_scope_id, p.contact_id, p.display_name,
                       p.sales_stage, p.latest_contact_at,
                       a.local_name AS account_name,
                       c.display_name AS contact_display_name,
                       c.remark AS contact_remark,
                       COALESCE((
                           SELECT MAX(e.event_at) FROM customer_events e
                           WHERE e.customer_profile_id=p.id AND e.status='active'
                       ), p.latest_contact_at, '') AS latest_business_at,
                       COALESCE(g.crop, '') AS followup_crop,
                       COALESCE(eg.crop, '') AS event_crop,
                       COALESCE((
                           SELECT GROUP_CONCAT(DISTINCT g2.crop)
                           FROM customer_growing_contexts g2
                           WHERE g2.customer_profile_id=p.id
                             AND g2.status='active' AND TRIM(g2.crop)<>''
                       ), '') AS active_context_crops
                FROM customer_followups f
                JOIN customer_profiles p ON p.id=f.customer_profile_id
                JOIN account_scopes a ON a.id=p.account_scope_id
                JOIN contacts c ON c.id=p.contact_id
                LEFT JOIN customer_growing_contexts g ON g.id=f.growing_context_id
                LEFT JOIN customer_events re ON re.id=f.related_event_id
                LEFT JOIN customer_growing_contexts eg ON eg.id=re.growing_context_id
                WHERE f.status='pending'
                ORDER BY f.due_at ASC,f.priority DESC,f.created_at ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _fact_identity_rows(conn: sqlite3.Connection, fact: sqlite3.Row) -> list[sqlite3.Row]:
        target = canonicalize_fact_identity(
            customer_profile_id=fact["customer_profile_id"],
            growing_context_id=fact["growing_context_id"],
            fact_category=fact["fact_category"],
            fact_key=fact["fact_key"],
            value_text=fact["value_text"],
            normalized_value=fact["normalized_value"],
            value_json=fact["value_json"],
        )
        rows = conn.execute(
            """
            SELECT * FROM customer_facts
            WHERE customer_profile_id=? AND growing_context_id IS ?
              AND status IN ('active','uncertain','conflicting')
            ORDER BY last_confirmed_at DESC, created_at DESC, id DESC
            """,
            (fact["customer_profile_id"], fact["growing_context_id"]),
        ).fetchall()
        result: list[sqlite3.Row] = []
        for row in rows:
            identity = canonicalize_fact_identity(
                customer_profile_id=row["customer_profile_id"],
                growing_context_id=row["growing_context_id"],
                fact_category=row["fact_category"],
                fact_key=row["fact_key"],
                value_text=row["value_text"],
                normalized_value=row["normalized_value"],
                value_json=row["value_json"],
            )
            if (
                identity.canonical_category == target.canonical_category
                and identity.canonical_key == target.canonical_key
                and identity.temporal_scope == target.temporal_scope
            ):
                result.append(row)
        return result

    def _add_manual_fact_source(
        self,
        conn: sqlite3.Connection,
        fact: sqlite3.Row,
        profile: sqlite3.Row,
        action: str,
        reason: str,
        observed_at: str,
    ) -> None:
        note = action if not reason.strip() else f"{action}：{_clean(reason, limit=500)}"
        self._add_fact_source_on_connection(
            conn,
            fact=_fact_from_row(fact),
            profile=profile,
            message_id=None,
            conversation_id=profile["conversation_id"],
            source_text=note,
            observed_at=observed_at,
            required=True,
        )

    @staticmethod
    def _sync_context_from_fact(
        conn: sqlite3.Connection,
        fact: sqlite3.Row,
        *,
        clear: bool,
        now: str,
    ) -> None:
        context_id = fact["growing_context_id"]
        if not context_id:
            return
        identity = canonicalize_fact_identity(
            customer_profile_id=fact["customer_profile_id"],
            growing_context_id=context_id,
            fact_category=fact["fact_category"],
            fact_key=fact["fact_key"],
            value_text=fact["value_text"],
            normalized_value=fact["normalized_value"],
            value_json=fact["value_json"],
        )
        key = identity.canonical_key
        column_by_key = {
            "region": "region",
            "crop": "crop",
            "variety": "variety",
            "growth_stage": "growth_stage",
            "cultivation_type": "cultivation_type",
            "season_or_year": "season_or_year",
        }
        if key not in column_by_key and key != "acreage":
            return
        source = fact
        if clear:
            candidates = conn.execute(
                """
                SELECT * FROM customer_facts
                WHERE customer_profile_id=? AND growing_context_id=? AND status='active' AND id<>?
                ORDER BY CASE provenance WHEN 'user_explicit' THEN 0 WHEN 'customer_explicit' THEN 1 ELSE 2 END,
                         last_confirmed_at DESC, created_at DESC, id DESC
                """,
                (fact["customer_profile_id"], context_id, fact["id"]),
            ).fetchall()
            source = None
            for candidate in candidates:
                candidate_identity = canonicalize_fact_identity(
                    customer_profile_id=candidate["customer_profile_id"],
                    growing_context_id=context_id,
                    fact_category=candidate["fact_category"],
                    fact_key=candidate["fact_key"],
                    value_text=candidate["value_text"],
                    normalized_value=candidate["normalized_value"],
                    value_json=candidate["value_json"],
                )
                if candidate_identity.canonical_key == key:
                    source = candidate
                    break
        if key == "acreage":
            acreage_value: float | None = None
            acreage_unit = ""
            if source is not None:
                parsed = extract_acreage(source["value_text"])
                if parsed is None and "|" in source["normalized_value"]:
                    raw_value, raw_unit = source["normalized_value"].split("|", 1)
                    try:
                        parsed = (float(raw_value), raw_unit)
                    except ValueError:
                        parsed = None
                if parsed is not None:
                    acreage_value, acreage_unit = parsed
            conn.execute(
                """
                UPDATE customer_growing_contexts
                SET acreage_value=?, acreage_unit=?, status='active', last_confirmed_at=?, updated_at=?
                WHERE id=? AND customer_profile_id=?
                """,
                (acreage_value, acreage_unit, now, now, context_id, fact["customer_profile_id"]),
            )
            return
        column = column_by_key[key]
        value = source["value_text"] if source is not None else ""
        conn.execute(
            f"UPDATE customer_growing_contexts SET {column}=?, status='active', last_confirmed_at=?, updated_at=? WHERE id=? AND customer_profile_id=?",
            (value, now, now, context_id, fact["customer_profile_id"]),
        )

    @staticmethod
    def _insert_memory_edit(
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        entity_id: str,
        operation: str,
        old_value: dict[str, Any],
        new_value: dict[str, Any],
        reason: str,
        actor: str,
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO customer_memory_edits(
                id, customer_profile_id, entity_type, entity_id, operation,
                old_value_json, new_value_json, reason, actor, created_at
            ) VALUES (?, ?, 'fact', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id("memory_edit"),
                profile_id,
                entity_id,
                operation,
                json.dumps(old_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                json.dumps(new_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                _clean(reason, limit=1000),
                _clean(actor, required=True, limit=100),
                created_at,
            ),
        )

    @staticmethod
    def _require_contact(conn: sqlite3.Connection, account_scope_id: str, contact_id: str) -> sqlite3.Row:
        scope = conn.execute("SELECT id FROM account_scopes WHERE id = ?", (account_scope_id,)).fetchone()
        if scope is None:
            raise ValueError("account scope does not exist")
        contact = conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        if contact is None:
            raise ValueError("contact does not exist")
        if contact["account_scope_id"] != account_scope_id:
            raise ValueError("contact does not belong to the account scope")
        return contact

    @staticmethod
    def _require_profile(conn: sqlite3.Connection, profile_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM customer_profiles WHERE id = ?", (profile_id,)).fetchone()
        if row is None:
            raise ValueError("customer profile does not exist")
        return row

    @staticmethod
    def _validate_optional_bindings(
        conn: sqlite3.Connection,
        *,
        account_scope_id: str,
        contact_id: str,
        target_instance_id: str | None,
        conversation_id: str | None,
    ) -> None:
        if target_instance_id:
            target = conn.execute(
                "SELECT account_scope_id FROM target_instances WHERE id = ?", (target_instance_id,)
            ).fetchone()
            if target is None or target["account_scope_id"] != account_scope_id:
                raise ValueError("target instance does not belong to the account scope")
        if conversation_id:
            conversation = conn.execute(
                "SELECT account_scope_id, contact_id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if (
                conversation is None
                or conversation["account_scope_id"] != account_scope_id
                or conversation["contact_id"] != contact_id
            ):
                raise ValueError("conversation does not belong to the account scope/contact")

    @staticmethod
    def _validate_context(
        conn: sqlite3.Connection,
        customer_profile_id: str,
        growing_context_id: str | None,
    ) -> None:
        if not growing_context_id:
            return
        row = conn.execute(
            "SELECT customer_profile_id FROM customer_growing_contexts WHERE id = ?",
            (growing_context_id,),
        ).fetchone()
        if row is None or row["customer_profile_id"] != customer_profile_id:
            raise ValueError("growing context does not belong to the customer profile")

    @staticmethod
    def _validate_followup_source_message(
        conn: sqlite3.Connection,
        profile: sqlite3.Row,
        source_message_id: str | None,
    ) -> None:
        if not source_message_id:
            return
        message = conn.execute(
            "SELECT account_scope_id,contact_id FROM messages WHERE id=?",
            (source_message_id,),
        ).fetchone()
        if message is None:
            raise ValueError("source message does not exist")
        if (
            message["account_scope_id"] != profile["account_scope_id"]
            or message["contact_id"] != profile["contact_id"]
        ):
            raise ValueError("source message belongs to a different account/contact")

    @staticmethod
    def _validated_source(
        conn: sqlite3.Connection,
        *,
        profile: sqlite3.Row,
        message_id: str | None,
        conversation_id: str | None,
        source_text: str,
        observed_at: str | None,
    ) -> tuple[str | None, str | None, str, str]:
        text = _clean(source_text, limit=4000)
        if message_id:
            message = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
            if message is None:
                raise ValueError("source message does not exist")
            if (
                message["account_scope_id"] != profile["account_scope_id"]
                or message["contact_id"] != profile["contact_id"]
            ):
                raise ValueError("source message belongs to a different account/contact")
            message_conversation_id = message["conversation_id"]
            if conversation_id and message_conversation_id and conversation_id != message_conversation_id:
                raise ValueError("source conversation does not match the message")
            conversation_id = conversation_id or message_conversation_id
            text = text or message["text"]
            observed_at = observed_at or message["observed_at"]
        if conversation_id:
            conversation = conn.execute(
                "SELECT account_scope_id, contact_id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if (
                conversation is None
                or conversation["account_scope_id"] != profile["account_scope_id"]
                or conversation["contact_id"] != profile["contact_id"]
            ):
                raise ValueError("source conversation belongs to a different account/contact")
        if not text:
            raise ValueError("source_text or a source message is required")
        return message_id, conversation_id, text, observed_at or utc_now_iso()

    def _add_fact_source_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        fact: CustomerFact,
        profile: sqlite3.Row,
        message_id: str | None,
        conversation_id: str | None,
        source_text: str,
        observed_at: str | None,
        required: bool = False,
    ) -> tuple[CustomerFactSource, bool] | None:
        if not required and not message_id and not source_text.strip():
            return None
        message_id, conversation_id, text, observed = self._validated_source(
            conn,
            profile=profile,
            message_id=message_id,
            conversation_id=conversation_id,
            source_text=source_text,
            observed_at=observed_at,
        )
        item = CustomerFactSource(
            id=_new_id("fact_source"),
            fact_id=fact.id,
            message_id=message_id,
            conversation_id=conversation_id,
            source_text=text,
            observed_at=observed,
            created_at=utc_now_iso(),
        )
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO customer_fact_sources(
                id, fact_id, message_id, conversation_id, source_text, observed_at, created_at
            ) VALUES (:id, :fact_id, :message_id, :conversation_id, :source_text, :observed_at, :created_at)
            """,
            asdict(item),
        )
        if cursor.rowcount:
            return item, True
        if message_id:
            row = conn.execute(
                "SELECT * FROM customer_fact_sources WHERE fact_id = ? AND message_id = ?",
                (fact.id, message_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM customer_fact_sources
                WHERE fact_id = ? AND message_id IS NULL AND conversation_id IS ?
                  AND source_text = ? AND observed_at = ?
                """,
                (fact.id, conversation_id, text, observed),
            ).fetchone()
        return _fact_source_from_row(row), False

    def _add_event_source_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        event: CustomerEvent,
        profile: sqlite3.Row,
        message_id: str | None,
        conversation_id: str | None,
        source_text: str,
        observed_at: str | None,
    ) -> tuple[CustomerEventSource, bool]:
        message_id, conversation_id, text, observed = self._validated_source(
            conn,
            profile=profile,
            message_id=message_id,
            conversation_id=conversation_id,
            source_text=source_text,
            observed_at=observed_at,
        )
        item = CustomerEventSource(
            id=_new_id("event_source"),
            event_id=event.id,
            message_id=message_id,
            conversation_id=conversation_id,
            source_text=text,
            observed_at=observed,
            created_at=utc_now_iso(),
        )
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO customer_event_sources(
                id, event_id, message_id, conversation_id, source_text, observed_at, created_at
            ) VALUES (:id, :event_id, :message_id, :conversation_id, :source_text, :observed_at, :created_at)
            """,
            asdict(item),
        )
        if cursor.rowcount:
            return item, True
        if message_id:
            row = conn.execute(
                "SELECT * FROM customer_event_sources WHERE event_id = ? AND message_id = ?",
                (event.id, message_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM customer_event_sources
                WHERE event_id = ? AND message_id IS NULL AND conversation_id IS ?
                  AND source_text = ? AND observed_at = ?
                """,
                (event.id, conversation_id, text, observed),
            ).fetchone()
        return _event_source_from_row(row), False

    @staticmethod
    def _refresh_next_followup(conn: sqlite3.Connection, profile_id: str, now: str) -> None:
        row = conn.execute(
            "SELECT MIN(due_at) AS next_due FROM customer_followups WHERE customer_profile_id = ? AND status = 'pending'",
            (profile_id,),
        ).fetchone()
        conn.execute(
            "UPDATE customer_profiles SET next_followup_at = ?, updated_at = ? WHERE id = ?",
            (row["next_due"], now, profile_id),
        )


def _stronger_provenance(
    existing: CustomerFactProvenance,
    incoming: CustomerFactProvenance,
) -> CustomerFactProvenance:
    # ``user_explicit`` is authoritative only when written by the dedicated
    # manual-correction transaction.  Normal extraction may also describe the
    # salesperson's own message as user_explicit; that must never promote a
    # customer fact.  Conversely, later extraction must not silently downgrade
    # an already manually confirmed fact.
    if existing == CustomerFactProvenance.USER_EXPLICIT:
        return existing
    if incoming == CustomerFactProvenance.USER_EXPLICIT:
        return existing
    priority = {
        CustomerFactProvenance.AI_INFERRED: 0,
        CustomerFactProvenance.SYSTEM_IMPORTED: 1,
        CustomerFactProvenance.CUSTOMER_EXPLICIT: 2,
        CustomerFactProvenance.USER_EXPLICIT: 3,
    }
    return incoming if priority[incoming] > priority[existing] else existing


def _profile_from_row(row: sqlite3.Row) -> CustomerProfile:
    data = dict(row)
    data["sales_stage"] = CustomerSalesStage(data["sales_stage"])
    data["sales_stage_source"] = CustomerSalesStageSource(data["sales_stage_source"])
    return CustomerProfile(**data)


def _growing_context_from_row(row: sqlite3.Row) -> CustomerGrowingContext:
    data = dict(row)
    data["status"] = GrowingContextStatus(data["status"])
    return CustomerGrowingContext(**data)


def _fact_from_row(row: sqlite3.Row) -> CustomerFact:
    data = dict(row)
    data["provenance"] = CustomerFactProvenance(data["provenance"])
    data["status"] = CustomerFactStatus(data["status"])
    return CustomerFact(**data)


def _fact_source_from_row(row: sqlite3.Row) -> CustomerFactSource:
    return CustomerFactSource(**dict(row))


def _event_from_row(row: sqlite3.Row) -> CustomerEvent:
    data = dict(row)
    data["event_type"] = CustomerEventType(data["event_type"])
    data["provenance"] = CustomerFactProvenance(data["provenance"])
    data["status"] = CustomerEventStatus(data["status"])
    return CustomerEvent(**data)


def _event_source_from_row(row: sqlite3.Row) -> CustomerEventSource:
    return CustomerEventSource(**dict(row))


def _followup_from_row(row: sqlite3.Row) -> CustomerFollowup:
    data = dict(row)
    data["status"] = CustomerFollowupStatus(data["status"])
    data["source"] = CustomerFollowupSource(data["source"])
    return CustomerFollowup(**data)


def _sales_stage_history_from_row(row: sqlite3.Row) -> CustomerSalesStageHistory:
    data = dict(row)
    data["old_stage"] = CustomerSalesStage(data["old_stage"])
    data["new_stage"] = CustomerSalesStage(data["new_stage"])
    data["source"] = CustomerSalesStageSource(data["source"])
    return CustomerSalesStageHistory(**data)


def _profile_params(item: CustomerProfile) -> dict[str, Any]:
    data = asdict(item)
    data["sales_stage"] = item.sales_stage.value
    data["sales_stage_source"] = item.sales_stage_source.value
    return data


def _growing_context_params(item: CustomerGrowingContext) -> dict[str, Any]:
    data = asdict(item)
    data["status"] = item.status.value
    return data


def _fact_params(item: CustomerFact) -> dict[str, Any]:
    data = asdict(item)
    data["provenance"] = item.provenance.value
    data["status"] = item.status.value
    return data


def _event_params(item: CustomerEvent) -> dict[str, Any]:
    data = asdict(item)
    data["event_type"] = item.event_type.value
    data["provenance"] = item.provenance.value
    data["status"] = item.status.value
    return data


def _followup_params(item: CustomerFollowup) -> dict[str, Any]:
    data = asdict(item)
    data["status"] = item.status.value
    data["source"] = item.source.value
    return data


def _fact_snapshot(row: sqlite3.Row) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "id",
            "customer_profile_id",
            "growing_context_id",
            "fact_category",
            "fact_key",
            "value_text",
            "value_json",
            "normalized_value",
            "confidence",
            "provenance",
            "status",
            "first_seen_at",
            "last_confirmed_at",
            "valid_from",
            "valid_to",
            "conflict_group_id",
            "superseded_by_fact_id",
            "created_at",
            "updated_at",
        )
    }
