from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from deepkeel.model_invocations import (
    ModelInvocationClaim,
    ModelInvocationConflict,
    ModelInvocationEnvelope,
    ModelInvocationRecord,
    ModelTurn,
)

from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.support import json_value


class PostgresModelInvocationStore:
    """Atomic model invocation ownership and exact-result replay."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def claim(
        self,
        envelope: ModelInvocationEnvelope,
        *,
        lease_seconds: float = 300.0,
    ) -> ModelInvocationClaim:
        ttl = max(1.0, float(lease_seconds))
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {schema}.model_invocations "
                    "WHERE invocation_id = %s FOR UPDATE",
                    (envelope.invocation_id,),
                )
                row = cursor.fetchone()
                if row is not None:
                    if row["request_fingerprint"] != envelope.request_fingerprint:
                        raise ModelInvocationConflict(
                            "invocation_id cannot be reused with a different request"
                        )
                    return _claim_from_row(row)

                claim_token = uuid4().hex
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.model_invocations (
                        invocation_id, request_fingerprint, envelope, status,
                        claim_token, claim_expires_at
                    ) VALUES (
                        %s, %s, %s::jsonb, 'running', %s,
                        CURRENT_TIMESTAMP + (%s * INTERVAL '1 second')
                    )
                    """,
                    (
                        envelope.invocation_id,
                        envelope.request_fingerprint,
                        json_value(envelope.model_dump(mode="json")),
                        claim_token,
                        ttl,
                    ),
                )
        return ModelInvocationClaim(
            invocation_id=envelope.invocation_id,
            outcome="acquired",
            claim_token=claim_token,
        )

    def complete(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        result: ModelTurn,
    ) -> ModelInvocationRecord:
        return self._settle(
            invocation_id,
            claim_token=claim_token,
            status="completed",
            result=result,
        )

    def fail(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        failure_type: str,
        failure_message: str,
    ) -> ModelInvocationRecord:
        return self._settle(
            invocation_id,
            claim_token=claim_token,
            status="failed",
            failure_type=failure_type,
            failure_message=failure_message,
        )

    def get_record(self, invocation_id: str) -> ModelInvocationRecord | None:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {self.database.schema}.model_invocations "
                    "WHERE invocation_id = %s",
                    (str(invocation_id or ""),),
                )
                row = cursor.fetchone()
        return _record_from_row(row) if row is not None else None

    def _settle(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        status: Literal["completed", "failed"],
        result: ModelTurn | None = None,
        failure_type: str = "",
        failure_message: str = "",
    ) -> ModelInvocationRecord:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {schema}.model_invocations "
                    "WHERE invocation_id = %s FOR UPDATE",
                    (str(invocation_id or ""),),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ModelInvocationConflict("cannot settle an unknown invocation")
                current = _record_from_row(row)
                if current.status in {"completed", "failed"}:
                    same_result = status == current.status and (
                        status == "failed" or current.result == result
                    )
                    if same_result:
                        return current
                    raise ModelInvocationConflict("invocation is already settled")
                if not claim_token or claim_token != current.claim_token:
                    raise ModelInvocationConflict("model invocation claim token changed")
                cursor.execute(
                    f"""
                    UPDATE {schema}.model_invocations
                    SET status = %s, claim_token = '', claim_expires_at = NULL,
                        result = %s::jsonb, failure_type = %s,
                        failure_message = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE invocation_id = %s
                    RETURNING *
                    """,
                    (
                        status,
                        json_value(result.model_dump(mode="json")) if result is not None else None,
                        str(failure_type or ""),
                        str(failure_message or "")[:500],
                        invocation_id,
                    ),
                )
                settled = cursor.fetchone()
        return _record_from_row(settled)


def _claim_from_row(row: dict[str, Any]) -> ModelInvocationClaim:
    invocation_id = str(row["invocation_id"])
    status = str(row["status"])
    if status == "completed" and row.get("result") is not None:
        return ModelInvocationClaim(
            invocation_id=invocation_id,
            outcome="replay",
            result=ModelTurn.model_validate(row["result"]),
        )
    if status == "failed":
        return ModelInvocationClaim(
            invocation_id=invocation_id,
            outcome="failed",
            failure_type=str(row.get("failure_type") or ""),
            failure_message=str(row.get("failure_message") or ""),
        )
    expires_at = row.get("claim_expires_at")
    if expires_at is not None:
        from datetime import UTC, datetime

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at > datetime.now(UTC):
            return ModelInvocationClaim(invocation_id=invocation_id, outcome="in_progress")
    return ModelInvocationClaim(
        invocation_id=invocation_id,
        outcome="uncertain",
        failure_type="claim_expired",
        failure_message="the previous provider invocation expired without a durable result",
    )


def _record_from_row(row: dict[str, Any]) -> ModelInvocationRecord:
    return ModelInvocationRecord(
        envelope=ModelInvocationEnvelope.model_validate(row["envelope"]),
        status=str(row["status"]),
        claim_token=str(row.get("claim_token") or ""),
        claim_expires_at=row.get("claim_expires_at"),
        result=(
            ModelTurn.model_validate(row["result"])
            if row.get("result") is not None
            else None
        ),
        failure_type=str(row.get("failure_type") or ""),
        failure_message=str(row.get("failure_message") or ""),
        updated_at=row["updated_at"],
    )
