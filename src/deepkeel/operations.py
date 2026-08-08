from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Iterable, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from deepkeel.control import CancellableRunControl
from deepkeel.recovery import RecoveryOutcome, classify_recovery_outcome
from deepkeel.scope import (
    RuntimeScope,
    require_legacy_compatible_scope,
    resolve_runtime_scope,
)
from deepkeel.state_store import (
    QueryableRuntimeStateStore,
    RunStateSnapshot,
    RuntimeStateStore,
    ScopedRuntimeStateStore,
)
from deepkeel.telemetry import TracePage, TraceQuery, TraceStore


class RunOperationsUnavailable(RuntimeError):
    """Raised when a Host did not install an optional operations adapter."""

    code = "RUN_OPERATIONS_UNAVAILABLE"


class RunInspection(BaseModel):
    """Payload-safe operational view of one user-scoped run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    found: bool = False
    snapshot: RunStateSnapshot | None = None
    recovery: RecoveryOutcome = Field(default_factory=RecoveryOutcome)
    trace: TracePage = Field(default_factory=TracePage)


class RunOperationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    run_id: str
    accepted: bool
    reason: str = ""
    operation_id: str = ""


class RunRecoveryAction(StrEnum):
    RESUME = "resume"
    RETRY = "retry"
    REQUEUE = "requeue"
    TERMINALIZE = "terminalize"


class RunRecoveryCommand(BaseModel):
    """Auditable request submitted to a Host-owned recovery executor."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    action: RunRecoveryAction
    run_id: str
    scope: RuntimeScope
    reason: str = ""
    target_status: str = ""
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunRecoveryExecutor(Protocol):
    def submit(self, command: RunRecoveryCommand) -> RunOperationReceipt: ...


class InMemoryRunRecoveryExecutor:
    """Deterministic command collector for embedding and tests."""

    def __init__(self) -> None:
        self.commands: list[RunRecoveryCommand] = []
        self._receipts: dict[str, RunOperationReceipt] = {}

    def submit(self, command: RunRecoveryCommand) -> RunOperationReceipt:
        replay = self._receipts.get(command.operation_id)
        if replay is not None:
            return replay.model_copy(deep=True)
        self.commands.append(command.model_copy(deep=True))
        receipt = RunOperationReceipt(
            action=command.action.value,
            run_id=command.run_id,
            operation_id=command.operation_id,
            accepted=True,
            reason="recovery_command_submitted",
        )
        self._receipts[command.operation_id] = receipt
        return receipt.model_copy(deep=True)


class RunOperations:
    """Small Host-facing control plane built only on public Core ports."""

    def __init__(
        self,
        state_store: RuntimeStateStore,
        *,
        run_control: CancellableRunControl | None = None,
        trace_store: TraceStore | None = None,
        recovery_executor: RunRecoveryExecutor | None = None,
    ) -> None:
        self.state_store = state_store
        self.run_control = run_control
        self.trace_store = trace_store
        self.recovery_executor = recovery_executor

    def inspect(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
        scope: RuntimeScope | None = None,
        recovery_attempts: int = 0,
        error_code: str = "",
        error_message: str = "",
        stale: bool = False,
        trace_limit: int = 200,
    ) -> RunInspection:
        resolved_scope = resolve_runtime_scope(scope, user_id=user_id)
        try:
            snapshot = _load_scoped_snapshot(
                self.state_store,
                str(run_id),
                session=session,
                scope=resolved_scope,
            )
        except Exception as exc:
            if _is_scope_denial(exc):
                return RunInspection()
            raise
        found = bool(snapshot.version or snapshot.sequence)
        if not found:
            return RunInspection()
        trace = TracePage()
        if self.trace_store is not None:
            scoped_trace = bool(
                getattr(self.trace_store, "supports_runtime_scope", False)
            )
            trace = self.trace_store.query(
                TraceQuery(
                    run_id=snapshot.run_id,
                    tenant_id=resolved_scope.tenant_id if scoped_trace else "",
                    user_id=resolved_scope.user_id if scoped_trace else "",
                    namespace=resolved_scope.namespace if scoped_trace else "",
                    limit=trace_limit,
                )
            )
        recovery = classify_recovery_outcome(
            runtime_status=snapshot.status,
            attempts=recovery_attempts,
            error_code=error_code,
            error_message=error_message,
            stale=stale,
        )
        return RunInspection(
            found=True,
            snapshot=snapshot,
            recovery=recovery,
            trace=trace,
        )

    def list_runs(
        self,
        *,
        session: Any = None,
        user_id: str = "",
        scope: RuntimeScope | None = None,
        statuses: Iterable[str] = (),
        limit: int = 100,
    ) -> tuple[RunStateSnapshot, ...]:
        resolved_scope = resolve_runtime_scope(scope, user_id=user_id)
        list_scoped = getattr(self.state_store, "list_snapshots_scoped", None)
        if callable(list_scoped):
            scoped_store = cast(ScopedRuntimeStateStore, self.state_store)
            return scoped_store.list_snapshots_scoped(
                session=session,
                scope=resolved_scope,
                statuses=statuses,
                limit=limit,
            )
        list_snapshots = getattr(self.state_store, "list_snapshots", None)
        if not callable(list_snapshots):
            raise RunOperationsUnavailable(
                "runtime state adapter does not support scoped run enumeration"
            )
        legacy_user_id = require_legacy_compatible_scope(
            resolved_scope,
            adapter_name=type(self.state_store).__name__,
        )
        queryable = cast(QueryableRuntimeStateStore, self.state_store)
        return queryable.list_snapshots(
            session=session,
            user_id=legacy_user_id,
            statuses=statuses,
            limit=limit,
        )

    def request_cancel(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
        scope: RuntimeScope | None = None,
    ) -> RunOperationReceipt:
        resolved_scope = resolve_runtime_scope(scope, user_id=user_id)
        inspection = self.inspect(
            run_id,
            session=session,
            scope=resolved_scope,
        )
        if not inspection.found or inspection.snapshot is None:
            return RunOperationReceipt(
                action="cancel",
                run_id=str(run_id),
                accepted=False,
                reason="run_not_found_in_scope",
            )
        if inspection.snapshot.settled:
            return RunOperationReceipt(
                action="cancel",
                run_id=str(run_id),
                accepted=False,
                reason="run_already_settled",
            )
        if self.run_control is None:
            raise RunOperationsUnavailable(
                "runtime control adapter does not support cancellation requests"
            )
        self.run_control.cancel(resolved_scope.qualify_identity(str(run_id)))
        return RunOperationReceipt(
            action="cancel",
            run_id=str(run_id),
            accepted=True,
            reason="cooperative_cancellation_requested",
        )

    def list_recovery_candidates(
        self,
        *,
        stale_before: datetime,
        session: Any = None,
        user_id: str = "",
        scope: RuntimeScope | None = None,
        limit: int = 100,
    ) -> tuple[RunStateSnapshot, ...]:
        candidates = self.list_runs(
            session=session,
            user_id=user_id,
            scope=scope,
            statuses=("preparing", "running", "task_running", "waiting_user_input"),
            limit=limit,
        )
        return tuple(
            snapshot
            for snapshot in candidates
            if snapshot.updated_at is not None and snapshot.updated_at < stale_before
        )

    def request_recovery(
        self,
        run_id: str,
        action: RunRecoveryAction,
        *,
        operation_id: str,
        session: Any = None,
        user_id: str = "",
        scope: RuntimeScope | None = None,
        reason: str = "",
        target_status: str = "",
    ) -> RunOperationReceipt:
        resolved_scope = resolve_runtime_scope(scope, user_id=user_id)
        inspection = self.inspect(
            run_id,
            session=session,
            scope=resolved_scope,
        )
        if not inspection.found or inspection.snapshot is None:
            return RunOperationReceipt(
                action=action.value,
                run_id=str(run_id),
                operation_id=operation_id,
                accepted=False,
                reason="run_not_found_in_scope",
            )
        if inspection.snapshot.settled and action is not RunRecoveryAction.RETRY:
            return RunOperationReceipt(
                action=action.value,
                run_id=str(run_id),
                operation_id=operation_id,
                accepted=False,
                reason="run_already_settled",
            )
        if self.recovery_executor is None:
            raise RunOperationsUnavailable(
                "runtime operations adapter does not support recovery commands"
            )
        return self.recovery_executor.submit(
            RunRecoveryCommand(
                operation_id=operation_id,
                action=action,
                run_id=str(run_id),
                scope=resolved_scope,
                reason=reason,
                target_status=target_status,
            )
        )


def _load_scoped_snapshot(
    store: RuntimeStateStore,
    run_id: str,
    *,
    session: Any,
    scope: RuntimeScope,
) -> RunStateSnapshot:
    load_scoped = getattr(store, "load_snapshot_scoped", None)
    if callable(load_scoped):
        scoped_store = cast(ScopedRuntimeStateStore, store)
        return scoped_store.load_snapshot_scoped(
            run_id,
            session=session,
            scope=scope,
        )
    legacy_user_id = require_legacy_compatible_scope(
        scope,
        adapter_name=type(store).__name__,
    )
    return store.load_snapshot(
        run_id,
        session=session,
        user_id=legacy_user_id,
    )


def _is_scope_denial(exc: Exception) -> bool:
    return isinstance(exc, (LookupError, PermissionError)) or int(
        getattr(exc, "status_code", 0) or 0
    ) in {403, 404}
