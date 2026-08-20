from __future__ import annotations

from dataclasses import asdict
from typing import Any
from uuid import uuid4

from deepkeel.capability_control import (
    CapabilityPackageConflict,
    CapabilityPackageSnapshot,
)
from deepkeel.context_window_contracts import ContextSummaryRecord
from deepkeel.scope import RuntimeScope
from deepkeel.memory_sdk import (
    MemoryClaim,
    MemoryMutation,
    MemoryMutationReceipt,
    MemoryQuery,
    MemorySearchHit,
    MemorySearchPage,
)
from deepkeel.subagents.contracts import SubAgentResult, SubAgentSpec, TaskBrief

from .database import PostgresDatabase


class PostgresCapabilityPackageStore:
    """Process-shared capability catalog with optimistic revision checks."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def load(self) -> CapabilityPackageSnapshot:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT snapshot FROM {self.database.schema}.capability_catalog "
                "WHERE catalog_id = 'default'"
            )
            row = cursor.fetchone()
        return (
            CapabilityPackageSnapshot.model_validate(row["snapshot"])
            if row
            else CapabilityPackageSnapshot()
        )

    def save(
        self,
        snapshot: CapabilityPackageSnapshot,
        *,
        expected_revision: int,
    ) -> CapabilityPackageSnapshot:
        stored = snapshot.model_copy(update={"revision": expected_revision + 1}, deep=True)
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT revision FROM {self.database.schema}.capability_catalog "
                "WHERE catalog_id = 'default' FOR UPDATE"
            )
            current = cursor.fetchone()
            found = int(current["revision"]) if current else 0
            if found != expected_revision:
                raise CapabilityPackageConflict(
                    f"capability package catalog changed ({found} != {expected_revision})"
                )
            cursor.execute(
                f"""
                INSERT INTO {self.database.schema}.capability_catalog (
                    catalog_id, revision, snapshot, updated_at
                ) VALUES ('default', %s, %s::jsonb, CURRENT_TIMESTAMP)
                ON CONFLICT (catalog_id) DO UPDATE SET
                    revision = EXCLUDED.revision,
                    snapshot = EXCLUDED.snapshot,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (stored.revision, stored.model_dump_json()),
            )
        return stored


class PostgresContextSummaryCache:
    """Durable summary cache keyed by source fingerprint."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def get(
        self,
        scope: RuntimeScope,
        cache_key: str,
        source_fingerprint: str,
    ) -> ContextSummaryRecord | None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT cache_key, source_fingerprint, summary, summary_version
                FROM {self.database.schema}.context_summaries
                WHERE scope_digest = %s AND cache_key = %s AND source_fingerprint = %s
                """,
                (scope.scope_digest, str(cache_key), str(source_fingerprint)),
            )
            row = cursor.fetchone()
        return ContextSummaryRecord(**row) if row else None

    def put(self, scope: RuntimeScope, record: ContextSummaryRecord) -> None:
        if not record.cache_key or not record.source_fingerprint:
            return
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.database.schema}.context_summaries (
                    scope_digest, cache_key, source_fingerprint,
                    summary, summary_version, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (scope_digest, cache_key) DO UPDATE SET
                    source_fingerprint = EXCLUDED.source_fingerprint,
                    summary = EXCLUDED.summary,
                    summary_version = EXCLUDED.summary_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    scope.scope_digest,
                    record.cache_key,
                    record.source_fingerprint,
                    _json(record.summary),
                    record.summary_version,
                ),
            )

    def invalidate(self, scope: RuntimeScope, cache_key: str) -> None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {self.database.schema}.context_summaries "
                "WHERE scope_digest = %s AND cache_key = %s",
                (scope.scope_digest, str(cache_key)),
            )


class PostgresMemoryStore:
    """Reference durable MemoryPort; semantic retrieval remains a Host extension."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def apply(self, mutation: MemoryMutation) -> MemoryMutationReceipt:
        if mutation.idempotency_key:
            prior = self._mutation_receipt(mutation.idempotency_key)
            if prior is not None:
                return prior
        if mutation.action == "noop":
            receipt = MemoryMutationReceipt(action="noop", reason=mutation.reason)
            self._record_mutation(mutation, receipt)
            return receipt

        claim = mutation.claim
        claim_id = str(mutation.target_claim_id or (claim.claim_id if claim else "") or uuid4().hex)
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT payload, version FROM {self.database.schema}.memory_claims "
                "WHERE claim_id = %s FOR UPDATE",
                (claim_id,),
            )
            current = cursor.fetchone()
            version = int(current["version"]) + 1 if current else 1
            if mutation.action == "archive":
                if current is None:
                    receipt = MemoryMutationReceipt(
                        action="archive", claim_id=claim_id, reason="claim not found"
                    )
                else:
                    archived = MemoryClaim.model_validate(current["payload"]).model_copy(
                        update={"status": "archived"}
                    )
                    self._upsert_claim(cursor, archived, claim_id=claim_id, version=version)
                    receipt = MemoryMutationReceipt(
                        action="archive", claim_id=claim_id, applied=True, version=version
                    )
            else:
                if claim is None:
                    raise ValueError(f"memory {mutation.action} requires a claim")
                stored_claim = claim.model_copy(
                    update={
                        "claim_id": claim_id,
                        "observation_count": (
                            max(
                                claim.observation_count,
                                MemoryClaim.model_validate(current["payload"]).observation_count
                                + 1,
                            )
                            if current and mutation.action == "reinforce"
                            else claim.observation_count
                        ),
                    }
                )
                self._upsert_claim(cursor, stored_claim, claim_id=claim_id, version=version)
                receipt = MemoryMutationReceipt(
                    action=mutation.action,
                    claim_id=claim_id,
                    applied=True,
                    version=version,
                    reason=mutation.reason,
                )
            self._record_mutation(mutation, receipt, cursor=cursor)
        return receipt

    def get(self, claim_id: str) -> MemoryClaim | None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {self.database.schema}.memory_claims WHERE claim_id = %s",
                (str(claim_id),),
            )
            row = cursor.fetchone()
        return MemoryClaim.model_validate(row["payload"]) if row else None

    def search(self, query: MemoryQuery) -> MemorySearchPage:
        clauses = ["user_id = %s", "status = 'active'"]
        params: list[Any] = [query.user_id]
        for column, value in (
            ("tenant_id", query.tenant_id),
            ("subject_type", query.subject_type),
            ("subject_id", query.subject_id),
            ("profile_id", query.profile_id),
        ):
            if value:
                clauses.append(f"{column} = %s")
                params.append(value)
        if query.domains:
            clauses.append("domain = ANY(%s)")
            params.append(query.domains)
        if query.predicates:
            clauses.append("predicate = ANY(%s)")
            params.append(query.predicates)
        if query.scopes:
            clauses.append("scope = ANY(%s)")
            params.append(query.scopes)
        if not query.include_sensitive:
            clauses.append("sensitivity = 'normal'")
        params.append(max(query.limit * 4, query.limit))
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT payload FROM {self.database.schema}.memory_claims
                WHERE {" AND ".join(clauses)}
                ORDER BY updated_at DESC LIMIT %s
                """,
                tuple(params),
            )
            claims = [MemoryClaim.model_validate(row["payload"]) for row in cursor.fetchall()]
        terms = set(str(query.text or "").lower().split())
        hits = []
        for claim in claims:
            haystack = f"{claim.domain} {claim.predicate} {claim.value}".lower()
            lexical = sum(1 for term in terms if term in haystack) / max(1, len(terms))
            structured = 1.0 if query.predicates and claim.predicate in query.predicates else 0.5
            score = lexical * 0.7 + structured * 0.3
            hits.append(
                MemorySearchHit(
                    claim=claim,
                    score=score,
                    structured_score=structured,
                    lexical_score=lexical,
                    reasons=["structured filters", "lexical reference retrieval"],
                )
            )
        hits.sort(
            key=lambda item: (
                item.score,
                str(item.claim.updated_at or item.claim.created_at or ""),
            ),
            reverse=True,
        )
        return MemorySearchPage(
            hits=hits[: query.limit],
            retrieval_mode="structured_lexical",
            trace={"candidate_count": len(claims), "semantic_adapter": False},
        )

    def _upsert_claim(
        self, cursor: Any, claim: MemoryClaim, *, claim_id: str, version: int
    ) -> None:
        cursor.execute(
            f"""
            INSERT INTO {self.database.schema}.memory_claims (
                claim_id, tenant_id, user_id, subject_type, subject_id, profile_id,
                domain, predicate, scope, status, sensitivity, version, payload, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                      CURRENT_TIMESTAMP)
            ON CONFLICT (claim_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id, user_id = EXCLUDED.user_id,
                subject_type = EXCLUDED.subject_type, subject_id = EXCLUDED.subject_id,
                profile_id = EXCLUDED.profile_id, domain = EXCLUDED.domain,
                predicate = EXCLUDED.predicate, scope = EXCLUDED.scope,
                status = EXCLUDED.status, sensitivity = EXCLUDED.sensitivity,
                version = EXCLUDED.version, payload = EXCLUDED.payload,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                claim_id,
                claim.tenant_id,
                claim.user_id,
                claim.subject_type,
                claim.subject_id,
                claim.profile_id,
                claim.domain,
                claim.predicate,
                claim.scope,
                claim.status,
                claim.sensitivity,
                version,
                claim.model_dump_json(),
            ),
        )

    def _mutation_receipt(self, key: str) -> MemoryMutationReceipt | None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT receipt FROM {self.database.schema}.memory_mutations "
                "WHERE idempotency_key = %s",
                (key,),
            )
            row = cursor.fetchone()
        return MemoryMutationReceipt.model_validate(row["receipt"]) if row else None

    def _record_mutation(
        self,
        mutation: MemoryMutation,
        receipt: MemoryMutationReceipt,
        *,
        cursor: Any | None = None,
    ) -> None:
        if not mutation.idempotency_key:
            return
        if cursor is None:
            with self.database.connect() as connection, connection.cursor() as owned:
                self._insert_mutation(owned, mutation, receipt)
            return
        self._insert_mutation(cursor, mutation, receipt)

    def _insert_mutation(
        self, cursor: Any, mutation: MemoryMutation, receipt: MemoryMutationReceipt
    ) -> None:
        cursor.execute(
            f"""
            INSERT INTO {self.database.schema}.memory_mutations (
                idempotency_key, mutation, receipt
            ) VALUES (%s, %s::jsonb, %s::jsonb)
            ON CONFLICT (idempotency_key) DO NOTHING
            """,
            (mutation.idempotency_key, mutation.model_dump_json(), receipt.model_dump_json()),
        )


class PostgresSubAgentStore:
    """Durable child lineage, checkpoint, result, and suspension store."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def parent_accepts_results(self, parent_run_id: str) -> bool:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT canceled FROM {self.database.schema}.run_controls WHERE run_id = %s",
                (parent_run_id,),
            )
            row = cursor.fetchone()
        return not bool(row and row["canceled"])

    def create_child(
        self,
        *,
        child_run_id: str,
        root_run_id: str,
        parent_run_id: str,
        delegation_id: str,
        task: TaskBrief,
        spec: SubAgentSpec,
        user_id: str,
        thread_id: str,
    ) -> None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.database.schema}.subagent_runs (
                    child_run_id, root_run_id, parent_run_id, delegation_id,
                    user_id, thread_id, task, spec, phase, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                          'created', CURRENT_TIMESTAMP)
                ON CONFLICT (child_run_id) DO NOTHING
                """,
                (
                    child_run_id,
                    root_run_id,
                    parent_run_id,
                    delegation_id,
                    user_id,
                    thread_id,
                    task.model_dump_json(),
                    spec.model_dump_json(),
                ),
            )

    def settle_child(self, result: SubAgentResult) -> None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.database.schema}.subagent_runs
                SET result = %s::jsonb, phase = %s, updated_at = CURRENT_TIMESTAMP
                WHERE child_run_id = %s
                """,
                (result.model_dump_json(), result.status, result.child_run_id),
            )

    def load_child_result(self, child_run_id: str) -> SubAgentResult | None:
        return self._load_result(child_run_id, "result")

    def load_child_checkpoint(self, child_run_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT checkpoint FROM {self.database.schema}.subagent_runs "
                "WHERE child_run_id = %s",
                (child_run_id,),
            )
            row = cursor.fetchone()
        return dict(row["checkpoint"]) if row and row["checkpoint"] is not None else None

    def checkpoint_child(self, child_run_id: str, *, phase: str, state: dict[str, Any]) -> None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.database.schema}.subagent_runs
                SET phase = %s, checkpoint = %s::jsonb, updated_at = CURRENT_TIMESTAMP
                WHERE child_run_id = %s
                """,
                (phase, _json(state), child_run_id),
            )

    def cancel_requested(self, child_run_id: str, parent_run_id: str) -> bool:
        return not self.parent_accepts_results(parent_run_id)

    def suspend_child(self, result: SubAgentResult) -> None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.database.schema}.subagent_runs
                SET suspension = %s::jsonb, phase = 'needs_input', updated_at = CURRENT_TIMESTAMP
                WHERE child_run_id = %s
                """,
                (result.model_dump_json(), result.child_run_id),
            )

    def load_child_suspension(self, child_run_id: str) -> SubAgentResult | None:
        return self._load_result(child_run_id, "suspension")

    def clear_child_suspension(self, child_run_id: str) -> None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.database.schema}.subagent_runs
                SET suspension = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE child_run_id = %s
                """,
                (child_run_id,),
            )

    def _load_result(self, child_run_id: str, column: str) -> SubAgentResult | None:
        with self.database.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {column} FROM {self.database.schema}.subagent_runs "
                "WHERE child_run_id = %s",
                (child_run_id,),
            )
            row = cursor.fetchone()
        return SubAgentResult.model_validate(row[column]) if row and row[column] else None


def _json(value: Any) -> str:
    import json

    return json.dumps(
        asdict(value) if hasattr(value, "__dataclass_fields__") else value,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )
