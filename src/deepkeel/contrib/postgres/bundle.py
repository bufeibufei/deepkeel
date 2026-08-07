from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepkeel.composition import (
    RuntimeGovernancePorts,
    RuntimeObservabilityPorts,
    RuntimePersistencePorts,
    RuntimeExecutionPorts,
    RuntimePorts,
)
from deepkeel.async_ports import AsyncToolExecutionStoreAdapter

from deepkeel.contrib.postgres.checkpoint_store import PostgresDurableCheckpointStore
from deepkeel.contrib.postgres.database import PostgresDatabase
from deepkeel.contrib.postgres.event_journal import PostgresRuntimeEventJournal
from deepkeel.contrib.postgres.governance import (
    PostgresBudgetLedger,
    PostgresModelHealthStore,
    PostgresRunControl,
)
from deepkeel.contrib.postgres.lease_store import PostgresRunLeaseStore
from deepkeel.contrib.postgres.model_store import PostgresModelInvocationStore
from deepkeel.contrib.postgres.state_store import PostgresRuntimeStateStore
from deepkeel.contrib.postgres.tool_store import PostgresToolExecutionStore
from deepkeel.contrib.postgres.trace_store import PostgresTraceStore


@dataclass(frozen=True, slots=True)
class PostgresRuntimeBundle:
    """Complete PostgreSQL-backed worker ports except the LangGraph saver."""

    database: PostgresDatabase
    runtime_state_store: PostgresRuntimeStateStore
    checkpoint_store: PostgresDurableCheckpointStore
    event_journal: PostgresRuntimeEventJournal
    run_lease_store: PostgresRunLeaseStore
    model_invocation_store: PostgresModelInvocationStore
    tool_execution_store: PostgresToolExecutionStore
    budget_ledger: PostgresBudgetLedger
    model_health_store: PostgresModelHealthStore
    run_control: PostgresRunControl
    trace_store: PostgresTraceStore

    @classmethod
    def create(
        cls,
        dsn: str,
        *,
        schema: str = "deepkeel",
        initialize: bool = True,
        model_failure_threshold: int = 2,
        model_cooldown_seconds: float = 60.0,
    ) -> "PostgresRuntimeBundle":
        database = PostgresDatabase(dsn, schema=schema)
        if initialize:
            database.initialize()
        return cls(
            database=database,
            runtime_state_store=PostgresRuntimeStateStore(database),
            checkpoint_store=PostgresDurableCheckpointStore(database),
            event_journal=PostgresRuntimeEventJournal(database),
            run_lease_store=PostgresRunLeaseStore(database),
            model_invocation_store=PostgresModelInvocationStore(database),
            tool_execution_store=PostgresToolExecutionStore(database),
            budget_ledger=PostgresBudgetLedger(database),
            model_health_store=PostgresModelHealthStore(
                database,
                failure_threshold=model_failure_threshold,
                cooldown_seconds=model_cooldown_seconds,
            ),
            run_control=PostgresRunControl(database),
            trace_store=PostgresTraceStore(database),
        )

    def runtime_ports(
        self,
        *,
        checkpointer: Any,
        run_lease_owner_id: str,
        run_lease_ttl_seconds: float = 60.0,
    ) -> RuntimePorts:
        """Compose production ports while keeping saver ownership in the Host."""

        return RuntimePorts.from_bundles(
            persistence=RuntimePersistencePorts(
                checkpointer=checkpointer,
                checkpoint_store=self.checkpoint_store,
                runtime_state_store=self.runtime_state_store,
                event_journal=self.event_journal,
                run_lease_store=self.run_lease_store,
                model_invocation_store=self.model_invocation_store,
                async_tool_execution_store=AsyncToolExecutionStoreAdapter(
                    self.tool_execution_store
                ),
                run_lease_owner_id=str(run_lease_owner_id or "").strip(),
                run_lease_ttl_seconds=float(run_lease_ttl_seconds),
            ),
            governance=RuntimeGovernancePorts(
                model_health_store=self.model_health_store,
                budget_ledger=self.budget_ledger,
                run_control=self.run_control,
            ),
            observability=RuntimeObservabilityPorts(telemetry=self.trace_store),
            execution=RuntimeExecutionPorts(tool_view_mode="enforced"),
        )
