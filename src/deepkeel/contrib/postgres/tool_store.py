from __future__ import annotations

from typing import Any
from uuid import uuid4

from deepkeel.contracts import ToolCall, ToolResult
from deepkeel.tool_execution import ToolExecutionClaim
from deepkeel.scope import RuntimeScope

from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.support import fingerprint, json_value


class PostgresToolExecutionStore:
    """Durable claim, retry, settlement, and replay for tool side effects."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def claim(
        self,
        *,
        run_id: str,
        call: ToolCall,
        lease_seconds: float,
        max_attempts: int,
        reexecution_safe: bool = True,
    ) -> ToolExecutionClaim:
        return self.claim_scoped(
            scope=RuntimeScope(),
            run_id=run_id,
            call=call,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            reexecution_safe=reexecution_safe,
        )

    def claim_scoped(
        self,
        *,
        scope: RuntimeScope,
        run_id: str,
        call: ToolCall,
        lease_seconds: float,
        max_attempts: int,
        reexecution_safe: bool = True,
    ) -> ToolExecutionClaim:
        normalized_run_id = str(run_id or "").strip()
        key = str(call.idempotency_key or "").strip()
        if not normalized_run_id or not key:
            raise ValueError("run_id and call.idempotency_key are required")
        ttl = max(0.001, float(lease_seconds))
        attempt_limit = max(1, int(max_attempts))
        call_payload = call.model_dump(mode="json")
        call_fingerprint = fingerprint(call_payload)
        schema = self.database.schema
        tenant_id, namespace, user_id = scope.storage_key
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *, lease_expires_at > CURRENT_TIMESTAMP AS lease_live
                    FROM {schema}.tool_executions
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND run_id = %s AND idempotency_key = %s
                    FOR UPDATE
                    """,
                    (tenant_id, namespace, user_id, normalized_run_id, key),
                )
                row = cursor.fetchone()
                if row is not None:
                    if row["call_fingerprint"] != call_fingerprint:
                        return _claim_from_row(
                            row,
                            status="corrupt",
                            detail="idempotency key was reused with different tool input",
                        )
                    terminal = _terminal_claim(row, max_attempts=attempt_limit)
                    if terminal is not None:
                        return terminal
                    if bool(row.get("lease_live")) and row.get("claim_owner"):
                        return _claim_from_row(row, status="busy")
                    if int(row["attempt_count"]) >= attempt_limit:
                        cursor.execute(
                            f"""
                            UPDATE {schema}.tool_executions
                            SET status = 'exhausted', claim_owner = '',
                                lease_expires_at = NULL, updated_at = CURRENT_TIMESTAMP
                            WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                              AND run_id = %s AND idempotency_key = %s
                            RETURNING *
                            """,
                            (tenant_id, namespace, user_id, normalized_run_id, key),
                        )
                        return _claim_from_row(cursor.fetchone(), status="exhausted")
                    if not reexecution_safe:
                        cursor.execute(
                            f"""
                            UPDATE {schema}.tool_executions
                            SET status = 'uncertain', claim_owner = '',
                                lease_expires_at = NULL, updated_at = CURRENT_TIMESTAMP
                            WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                              AND run_id = %s AND idempotency_key = %s
                            RETURNING *
                            """,
                            (tenant_id, namespace, user_id, normalized_run_id, key),
                        )
                        return _claim_from_row(cursor.fetchone(), status="uncertain")
                    owner = uuid4().hex
                    cursor.execute(
                        f"""
                        UPDATE {schema}.tool_executions
                        SET status = 'running', claim_owner = %s,
                            lease_expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                            attempt_count = attempt_count + 1, result = NULL,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                          AND run_id = %s AND idempotency_key = %s
                        RETURNING *
                        """,
                        (
                            owner,
                            ttl,
                            tenant_id,
                            namespace,
                            user_id,
                            normalized_run_id,
                            key,
                        ),
                    )
                    return _claim_from_row(cursor.fetchone(), status="claimed")

                owner = uuid4().hex
                record_id = uuid4().hex
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.tool_executions (
                        tenant_id, namespace, user_id, run_id, idempotency_key,
                        record_id, call_fingerprint, tool_call, status, claim_owner,
                        lease_expires_at, attempt_count
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'running', %s,
                        CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'), 1
                    ) RETURNING *
                    """,
                    (
                        tenant_id,
                        namespace,
                        user_id,
                        normalized_run_id,
                        key,
                        record_id,
                        call_fingerprint,
                        json_value(call_payload),
                        owner,
                        ttl,
                    ),
                )
                return _claim_from_row(cursor.fetchone(), status="claimed", scope=scope)

    def replay(self, claim: ToolExecutionClaim) -> ToolResult:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT result FROM {self.database.schema}.tool_executions
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND run_id = %s AND idempotency_key = %s AND record_id = %s
                    """,
                    (
                        *claim.scope.storage_key,
                        claim.run_id,
                        claim.idempotency_key,
                        claim.record_id,
                    ),
                )
                row = cursor.fetchone()
        if row is None or row.get("result") is None:
            raise LookupError("tool execution result is not available for replay")
        return ToolResult.model_validate(row["result"])

    def settle(self, claim: ToolExecutionClaim, result: ToolResult) -> None:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {schema}.tool_executions
                    SET status = %s, claim_owner = '', lease_expires_at = NULL,
                        result = %s::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE tenant_id = %s AND namespace = %s AND user_id = %s
                      AND run_id = %s AND idempotency_key = %s
                      AND record_id = %s AND claim_owner = %s
                    """,
                    (
                        result.status,
                        json_value(result.model_dump(mode="json")),
                        *claim.scope.storage_key,
                        claim.run_id,
                        claim.idempotency_key,
                        claim.record_id,
                        claim.claim_owner,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("tool execution claim is no longer owned")


def _terminal_claim(
    row: dict[str, Any],
    *,
    max_attempts: int,
) -> ToolExecutionClaim | None:
    status = str(row["status"])
    if status in {"uncertain", "exhausted"}:
        return _claim_from_row(row, status=status)
    if row.get("result") is None:
        return None
    result = ToolResult.model_validate(row["result"])
    retry_failed = (
        result.status == "failed" and result.retryable and int(row["attempt_count"]) < max_attempts
    )
    return None if retry_failed else _claim_from_row(row, status="replay")


def _claim_from_row(
    row: dict[str, Any],
    *,
    status: str,
    detail: str = "",
    scope: RuntimeScope | None = None,
) -> ToolExecutionClaim:
    lease_expires_at = row.get("lease_expires_at")
    retry_after = 0.0
    if status == "busy" and lease_expires_at is not None:
        from datetime import UTC, datetime

        if lease_expires_at.tzinfo is None:
            lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
        retry_after = max(0.0, (lease_expires_at - datetime.now(UTC)).total_seconds())
    return ToolExecutionClaim(
        status=status,  # type: ignore[arg-type]
        run_id=str(row["run_id"]),
        idempotency_key=str(row["idempotency_key"]),
        record_id=str(row["record_id"]),
        claim_owner=str(row.get("claim_owner") or ""),
        attempt_count=int(row["attempt_count"]),
        lease_expires_at=lease_expires_at,
        retry_after_seconds=retry_after,
        terminal_status=str(row.get("status") or ""),
        detail=detail,
        scope=scope
        or RuntimeScope(
            tenant_id=str(row.get("tenant_id") or ""),
            namespace=str(row.get("namespace") or "default"),
            user_id=str(row.get("user_id") or "local-device"),
        ),
    )
