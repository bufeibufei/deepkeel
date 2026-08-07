from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from deepkeel.budget import BudgetDecision, BudgetRequest, BudgetSnapshot
from deepkeel.failures import RunCanceledError
from deepkeel.model_health import ModelHealthSnapshot

from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.support import json_value


class PostgresBudgetLedger:
    """Atomic and idempotent budget accounting shared by all workers."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def consume(self, request: BudgetRequest) -> BudgetDecision:
        amount = max(0.0, float(request.amount))
        limit = None if request.limit is None or float(request.limit) <= 0 else float(request.limit)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.budget_usage (run_id, metric, amount)
                    VALUES (%s, %s, 0) ON CONFLICT DO NOTHING
                    """,
                    (request.run_id, request.metric),
                )
                cursor.execute(
                    f"SELECT amount FROM {schema}.budget_usage "
                    "WHERE run_id = %s AND metric = %s FOR UPDATE",
                    (request.run_id, request.metric),
                )
                used = float(cursor.fetchone()["amount"])
                if request.operation_id:
                    cursor.execute(
                        f"""
                        SELECT decision FROM {schema}.budget_decisions
                        WHERE run_id = %s AND metric = %s AND operation_id = %s
                        """,
                        (request.run_id, request.metric, request.operation_id),
                    )
                    replay = cursor.fetchone()
                    if replay is not None:
                        return _budget_decision(replay["decision"])

                projected = max(used, amount) if request.aggregation == "max" else used + amount
                allowed = limit is None or projected <= limit
                consumed = projected if allowed else used
                decision = BudgetDecision(
                    allowed=allowed,
                    metric=request.metric,
                    requested=amount,
                    used=consumed,
                    remaining=None if limit is None else max(0.0, limit - consumed),
                    limit=limit,
                    reason="budget reserved" if allowed else f"{request.metric} budget exceeded",
                )
                if allowed:
                    cursor.execute(
                        f"""
                        UPDATE {schema}.budget_usage
                        SET amount = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE run_id = %s AND metric = %s
                        """,
                        (consumed, request.run_id, request.metric),
                    )
                if request.operation_id:
                    cursor.execute(
                        f"""
                        INSERT INTO {schema}.budget_decisions (
                            run_id, metric, operation_id, decision
                        ) VALUES (%s, %s, %s, %s::jsonb)
                        """,
                        (
                            request.run_id,
                            request.metric,
                            request.operation_id,
                            json_value(decision.as_dict()),
                        ),
                    )
        return decision

    def snapshot(self, run_id: str) -> BudgetSnapshot:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT metric, amount FROM {self.database.schema}.budget_usage "
                    "WHERE run_id = %s",
                    (str(run_id or ""),),
                )
                usage = {str(row["metric"]): float(row["amount"]) for row in cursor.fetchall()}
        return BudgetSnapshot(run_id=str(run_id or ""), usage=usage)

    def restore(
        self,
        run_id: str,
        snapshot: dict[str, Any] | BudgetSnapshot | None,
    ) -> None:
        if isinstance(snapshot, BudgetSnapshot):
            usage = snapshot.usage
        elif isinstance(snapshot, dict) and isinstance(snapshot.get("usage"), dict):
            usage = snapshot["usage"]
        else:
            usage = {}
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                for metric, raw_amount in usage.items():
                    try:
                        amount = max(0.0, float(raw_amount))
                    except (TypeError, ValueError):
                        continue
                    cursor.execute(
                        f"""
                        INSERT INTO {schema}.budget_usage (run_id, metric, amount)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (run_id, metric) DO UPDATE SET
                            amount = GREATEST({schema}.budget_usage.amount, EXCLUDED.amount),
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (run_id, str(metric), amount),
                    )

    def clear(self, run_id: str) -> None:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {schema}.budget_decisions WHERE run_id = %s",
                    (str(run_id or ""),),
                )
                cursor.execute(
                    f"DELETE FROM {schema}.budget_usage WHERE run_id = %s",
                    (str(run_id or ""),),
                )


class PostgresModelHealthStore:
    """Shared provider circuit state using database time."""

    def __init__(
        self,
        database: PostgresDatabase,
        *,
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.database = database
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))

    def snapshot(self, provider_id: str, model_id: str) -> ModelHealthSnapshot:
        provider, model = _health_key(provider_id, model_id)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {schema}.model_health
                    SET consecutive_failures = 0, opened_until = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE provider_id = %s AND model_id = %s
                      AND opened_until IS NOT NULL
                      AND opened_until <= CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (provider, model),
                )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        f"SELECT * FROM {schema}.model_health "
                        "WHERE provider_id = %s AND model_id = %s",
                        (provider, model),
                    )
                    row = cursor.fetchone()
        return _health_snapshot(row) if row is not None else ModelHealthSnapshot(provider, model)

    def record_success(self, provider_id: str, model_id: str) -> ModelHealthSnapshot:
        provider, model = _health_key(provider_id, model_id)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.model_health (provider_id, model_id)
                    VALUES (%s, %s)
                    ON CONFLICT (provider_id, model_id) DO UPDATE SET
                        consecutive_failures = 0, opened_until = NULL,
                        last_failure_category = '', last_failure_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (provider, model),
                )
                row = cursor.fetchone()
        return _health_snapshot(row)

    def record_failure(
        self,
        provider_id: str,
        model_id: str,
        *,
        category: str,
        immediate: bool = False,
        retry_after_seconds: float = 0.0,
    ) -> ModelHealthSnapshot:
        provider, model = _health_key(provider_id, model_id)
        cooldown = max(self.cooldown_seconds, float(retry_after_seconds or 0.0))
        open_at = 1 if immediate else self.failure_threshold
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.model_health (
                        provider_id, model_id, consecutive_failures,
                        opened_until, last_failure_category, last_failure_at
                    ) VALUES (
                        %s, %s, 1,
                        CASE WHEN %s <= 1 THEN
                            CURRENT_TIMESTAMP + (%s * INTERVAL '1 second')
                        ELSE NULL END,
                        %s, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (provider_id, model_id) DO UPDATE SET
                        consecutive_failures = {schema}.model_health.consecutive_failures + 1,
                        opened_until = CASE
                            WHEN {schema}.model_health.consecutive_failures + 1 >= %s THEN
                                CURRENT_TIMESTAMP + (%s * INTERVAL '1 second')
                            ELSE NULL
                        END,
                        last_failure_category = EXCLUDED.last_failure_category,
                        last_failure_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (provider, model, open_at, cooldown, str(category or "provider_error"), open_at, cooldown),
                )
                row = cursor.fetchone()
        return _health_snapshot(row)


class PostgresRunControl:
    """Shared cooperative cancellation state for active runs."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def cancel(self, run_id: str) -> None:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.run_controls (run_id, canceled)
                    VALUES (%s, TRUE)
                    ON CONFLICT (run_id) DO UPDATE SET
                        canceled = TRUE, updated_at = CURRENT_TIMESTAMP
                    """,
                    (str(run_id or ""),),
                )

    def raise_if_cancelled(self, run_id: str, *, force: bool = False) -> None:
        del force
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT canceled FROM {self.database.schema}.run_controls "
                    "WHERE run_id = %s",
                    (str(run_id or ""),),
                )
                row = cursor.fetchone()
        if row is not None and bool(row["canceled"]):
            raise RunCanceledError()

    def release(self, run_id: str) -> None:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.database.schema}.run_controls WHERE run_id = %s",
                    (str(run_id or ""),),
                )


def _budget_decision(value: dict[str, Any]) -> BudgetDecision:
    return BudgetDecision(
        allowed=bool(value["allowed"]),
        metric=str(value["metric"]),
        requested=float(value["requested"]),
        used=float(value["used"]),
        remaining=(float(value["remaining"]) if value.get("remaining") is not None else None),
        limit=float(value["limit"]) if value.get("limit") is not None else None,
        reason=str(value["reason"]),
    )


def _health_key(provider_id: str, model_id: str) -> tuple[str, str]:
    return (
        str(provider_id or "unknown-provider").strip() or "unknown-provider",
        str(model_id or "unknown-model").strip() or "unknown-model",
    )


def _health_snapshot(row: dict[str, Any]) -> ModelHealthSnapshot:
    return ModelHealthSnapshot(
        provider_id=str(row["provider_id"]),
        model_id=str(row["model_id"]),
        consecutive_failures=int(row["consecutive_failures"]),
        opened_until=_aware_or_none(row.get("opened_until")),
        last_failure_category=str(row.get("last_failure_category") or ""),
        last_failure_at=_aware_or_none(row.get("last_failure_at")),
        updated_at=_aware_or_none(row.get("updated_at")),
    )


def _aware_or_none(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
