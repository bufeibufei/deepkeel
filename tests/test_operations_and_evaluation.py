from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from deepkeel.runtime_sdk import (
    Artifact,
    EvalCase,
    EvalExpectation,
    EvalSuiteRunner,
    FinalAnswer,
    InMemoryRunControl,
    InMemoryRunRecoveryExecutor,
    InMemoryRuntimeStateStore,
    RunContext,
    RunOperations,
    RunRecoveryAction,
    RuntimeRequest,
    RuntimeScope,
    RuntimeScopeUnsupported,
    RuntimeResult,
    RuntimeResultStatus,
    RuntimeStateMutation,
    ToolResult,
    evaluate_runtime_result,
)
from deepkeel.adapter_sdk import (
    AsyncRuntimeStateStoreAdapter,
    InMemoryTelemetry,
    TelemetryRecord,
    TraceQuery,
)


def _result() -> RuntimeResult:
    return RuntimeResult(
        question="Find a date",
        run_id="run-eval",
        thread_id="thread-eval",
        graph_thread_id="graph-eval",
        turn_id="turn-eval",
        status=RuntimeResultStatus.COMPLETED,
        stop_reason="final_answer",
        step_count=2,
        final_answer=FinalAnswer(markdown="Use the first date."),
        run_context=RunContext(
            run_id="run-eval",
            thread_id="thread-eval",
            turn_id="turn-eval",
            user_id="user-a",
        ),
        tool_results=[
            ToolResult(
                tool_call_id="call-1",
                name="calendar.search",
                status="succeeded",
            )
        ],
        artifacts=[
            Artifact(
                id="artifact-1",
                run_id="run-eval",
                artifact_type="calendar.candidates",
            )
        ],
        ui_state={
            "schema_version": "harness-run-ui-v2",
            "lifecycle": "completed",
            "execution_status": "completed",
            "composer_mode": "ready",
            "can_send": True,
            "input_strategy": "follow_up",
            "requires_user_action": False,
            "is_resumable": False,
            "show_progress": False,
            "can_cancel": False,
            "active_task": None,
            "reason": "run_terminal",
        },
    )


def test_runtime_state_reference_adapter_isolates_users_and_operations() -> None:
    store = InMemoryRuntimeStateStore()
    control = InMemoryRunControl()
    store.commit(
        RuntimeStateMutation(
            mutation_id="mutation-a",
            run_id="shared-run",
            event_type="run.started",
            target_status="task_running",
        ),
        user_id="user-a",
    )
    store.commit(
        RuntimeStateMutation(
            mutation_id="mutation-b",
            run_id="shared-run",
            event_type="run.started",
            target_status="waiting_user_input",
        ),
        user_id="user-b",
    )

    operations = RunOperations(store, run_control=control)
    assert operations.inspect("shared-run", user_id="user-a").snapshot.status == "task_running"
    assert (
        operations.inspect("shared-run", user_id="user-b").snapshot.status
        == "waiting_user_input"
    )
    assert operations.inspect("shared-run", user_id="user-c").found is False
    assert [item.status for item in operations.list_runs(user_id="user-a")] == [
        "task_running"
    ]

    receipt = operations.request_cancel("shared-run", user_id="user-a")
    assert receipt.accepted is True
    assert operations.request_cancel("missing", user_id="user-a").accepted is False


def test_operations_only_load_trace_after_scoped_run_is_found() -> None:
    store = InMemoryRuntimeStateStore()
    telemetry = InMemoryTelemetry()
    telemetry.record(
        TelemetryRecord(event_name="tool.completed", run_id="private-run", sequence=1)
    )
    operations = RunOperations(store, trace_store=telemetry)

    assert operations.inspect("private-run", user_id="other-user").trace.records == []


def test_operations_treats_adapter_scope_denial_as_not_found() -> None:
    class ScopeDenied(RuntimeError):
        status_code = 404

    class DenyingStore:
        terminal_settlement_owner = "runtime"

        def load_snapshot(self, run_id, *, session=None, user_id=""):
            raise ScopeDenied("not found")

        def commit(self, mutation, *, session=None, user_id=""):
            raise AssertionError("commit is not expected")

    assert RunOperations(DenyingStore()).inspect("private", user_id="other").found is False


def test_deterministic_eval_checks_result_and_trace_contracts() -> None:
    case = EvalCase(
        case_id="calendar-success",
        request=RuntimeRequest(question="Find a date", user_id="user-a"),
        expectation=EvalExpectation(
            required_tools=frozenset({"calendar.search"}),
            required_artifact_types=frozenset({"calendar.candidates"}),
            required_trace_events=("run.started", "tool.completed", "runtime.settled"),
            ordered_trace_events=("run.started", "tool.completed", "runtime.settled"),
            max_steps=3,
        ),
    )
    trace = [
        TelemetryRecord(event_name="run.started", run_id="run-eval", sequence=1),
        TelemetryRecord(event_name="tool.completed", run_id="run-eval", sequence=2),
        TelemetryRecord(event_name="runtime.settled", run_id="run-eval", sequence=3),
    ]

    evaluated = evaluate_runtime_result(case, _result(), trace=trace)

    assert evaluated.passed is True
    assert evaluated.metrics["tool_count"] == 1


def test_eval_suite_turns_execution_exceptions_into_results() -> None:
    case = EvalCase(
        case_id="provider-failure",
        request=RuntimeRequest(question="hello"),
    )

    def fail(_: RuntimeRequest) -> RuntimeResult:
        raise TimeoutError("provider timed out")

    report = EvalSuiteRunner(fail).run("fault-injection", [case])

    assert report.passed is False
    assert report.cases[0].violations[0].code == "execution_exception"


def test_runtime_scope_isolates_tenants_namespaces_and_trace_queries() -> None:
    store = InMemoryRuntimeStateStore()
    tenant_a = RuntimeScope(tenant_id="tenant-a", user_id="user-1")
    tenant_b = RuntimeScope(tenant_id="tenant-b", user_id="user-1")
    store.commit_scoped(
        RuntimeStateMutation(
            mutation_id="scope-a",
            run_id="shared-run",
            event_type="run.started",
            target_status="task_running",
        ),
        scope=tenant_a,
    )
    store.commit_scoped(
        RuntimeStateMutation(
            mutation_id="scope-b",
            run_id="shared-run",
            event_type="run.started",
            target_status="waiting_user_input",
        ),
        scope=tenant_b,
    )

    operations = RunOperations(store)
    assert operations.inspect("shared-run", scope=tenant_a).snapshot.status == "task_running"
    assert (
        operations.inspect("shared-run", scope=tenant_b).snapshot.status
        == "waiting_user_input"
    )

    telemetry = InMemoryTelemetry()
    telemetry.record(
        TelemetryRecord(
            event_name="run.started",
            run_id="shared-run",
            tenant_id="tenant-a",
            user_id="user-1",
        )
    )
    telemetry.record(
        TelemetryRecord(
            event_name="run.started",
            run_id="shared-run",
            tenant_id="tenant-b",
            user_id="user-1",
        )
    )
    assert len(telemetry.query(TraceQuery(tenant_id="tenant-a")).records) == 1


def test_non_default_scope_fails_closed_for_legacy_state_adapter() -> None:
    class LegacyStore:
        terminal_settlement_owner = "runtime"

        def load_snapshot(self, run_id, *, session=None, user_id=""):
            raise AssertionError("legacy adapter must not receive an unsafe scope")

        def commit(self, mutation, *, session=None, user_id=""):
            raise AssertionError("commit is not expected")

    with pytest.raises(RuntimeScopeUnsupported):
        RunOperations(LegacyStore()).inspect(
            "private-run",
            scope=RuntimeScope(tenant_id="tenant-a", user_id="user-a"),
        )


def test_recovery_commands_are_authorized_idempotent_and_auditable() -> None:
    store = InMemoryRuntimeStateStore()
    scope = RuntimeScope(tenant_id="tenant-a", user_id="user-a")
    store.commit_scoped(
        RuntimeStateMutation(
            mutation_id="waiting",
            run_id="run-recover",
            event_type="run.waiting",
            target_status="task_running",
        ),
        scope=scope,
    )
    executor = InMemoryRunRecoveryExecutor()
    operations = RunOperations(store, recovery_executor=executor)

    first = operations.request_recovery(
        "run-recover",
        RunRecoveryAction.REQUEUE,
        operation_id="operation-1",
        scope=scope,
        reason="worker lease expired",
    )
    replay = operations.request_recovery(
        "run-recover",
        RunRecoveryAction.REQUEUE,
        operation_id="operation-1",
        scope=scope,
    )

    assert first.accepted is True
    assert replay == first
    assert len(executor.commands) == 1
    assert executor.commands[0].scope == scope
    assert (
        operations.request_recovery(
            "run-recover",
            RunRecoveryAction.REQUEUE,
            operation_id="operation-other",
            scope=RuntimeScope(tenant_id="tenant-b", user_id="user-a"),
        ).accepted
        is False
    )


def test_recovery_candidate_scan_uses_durable_update_time() -> None:
    store = InMemoryRuntimeStateStore()
    store.commit(
        RuntimeStateMutation(
            mutation_id="candidate",
            run_id="run-stale",
            event_type="run.started",
            target_status="task_running",
        ),
        user_id="user-a",
    )
    cutoff = datetime.now(UTC) + timedelta(seconds=1)

    candidates = RunOperations(store).list_recovery_candidates(
        user_id="user-a",
        stale_before=cutoff,
    )

    assert [item.run_id for item in candidates] == ["run-stale"]


def test_async_state_adapter_offloads_scoped_operations() -> None:
    store = InMemoryRuntimeStateStore()
    adapter = AsyncRuntimeStateStoreAdapter(store)
    scope = RuntimeScope(tenant_id="tenant-a", user_id="user-a")

    async def exercise() -> None:
        await adapter.commit_scoped(
            RuntimeStateMutation(
                mutation_id="async-state",
                run_id="run-async",
                event_type="run.started",
                target_status="task_running",
            ),
            scope=scope,
        )
        snapshot = await adapter.load_snapshot_scoped("run-async", scope=scope)
        assert snapshot.status == "task_running"
        assert [item.run_id for item in await adapter.list_snapshots_scoped(scope=scope)] == [
            "run-async"
        ]

    asyncio.run(exercise())
