from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from harness_core.adapter_sdk import (
    HarnessRuntimeBuilder,
    InMemoryRunLeaseStore,
    RunLeaseConflict,
    RunLeaseLost,
    RuntimePorts,
    verify_run_lease_store_contract,
)
from harness_core.runtime_sdk import RuntimeRequest
from harness_core.runtime_sdk import InMemoryRuntimeStateStore, RuntimeStateConflict
from harness_core.state_store import RuntimeStateMutation
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tools import ToolExecutionContext, ToolExecutor
from harness_core.contracts import ToolCall, ToolResult


class BlockingProvider:
    model = "lease-model"
    model_role = "reasoning"

    def __init__(self, entered: Event, release: Event) -> None:
        self.entered = entered
        self.release = release

    def complete_chat(self, _messages, **_kwargs):
        self.entered.set()
        assert self.release.wait(5)
        return {
            "message": {"role": "assistant", "content": "completed"},
            "finish_reason": "stop",
            "model": self.model,
        }


class ImmediateProvider:
    model = "lease-model"
    model_role = "fast"

    def complete_chat(self, _messages, **_kwargs):
        return {
            "message": {"role": "assistant", "content": "completed"},
            "finish_reason": "stop",
            "model": self.model,
        }


def test_reference_run_lease_store_passes_contract() -> None:
    verify_run_lease_store_contract(InMemoryRunLeaseStore(), run_id="lease-contract")


def test_expired_lease_can_be_taken_over_with_a_new_fencing_generation() -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    clock = lambda: now
    store = InMemoryRunLeaseStore(clock=clock)
    first = store.claim("run-1", owner_id="worker-a", ttl_seconds=10)
    now += timedelta(seconds=11)
    second = store.claim("run-1", owner_id="worker-b", ttl_seconds=10)

    assert second.generation == first.generation + 1
    assert second.token != first.token
    with pytest.raises(RunLeaseLost):
        store.release(first)


def test_runtime_lease_prevents_two_workers_from_executing_one_run() -> None:
    entered = Event()
    release = Event()
    store = InMemoryRunLeaseStore()
    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(
            RuntimePorts(
                run_lease_store=store,
                run_lease_owner_id="worker-a",
                run_lease_ttl_seconds=3,
            )
        )
        .build()
    )
    request = RuntimeRequest(question="hello", run_id="shared-run")
    provider = BlockingProvider(entered, release)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(runtime.run, request, provider=provider)
        assert entered.wait(2)
        with pytest.raises(RunLeaseConflict):
            runtime.run(request, provider=provider)
        release.set()
        result = first.result(timeout=5)

    assert result.status.value == "completed"
    assert store.inspect("shared-run") is None


def test_runtime_state_store_rejects_an_obsolete_fencing_generation() -> None:
    store = InMemoryRuntimeStateStore()
    current = RuntimeStateMutation(
        mutation_id="fence-current",
        run_id="fenced-run",
        event_type="runtime.checkpoint.committed",
        target_status="task_running",
        fence_token="worker-b",
        fence_generation=2,
    )
    store.commit(current)

    with pytest.raises(RuntimeStateConflict, match="stale execution fence"):
        store.commit(
            RuntimeStateMutation(
                mutation_id="fence-stale",
                run_id="fenced-run",
                event_type="runtime.checkpoint.committed",
                target_status="task_running",
                fence_token="worker-a",
                fence_generation=1,
            )
        )


def test_tool_result_is_rejected_when_lease_is_lost_during_handler() -> None:
    class Fence:
        token = "tool-owner"
        generation = 3

        def __init__(self) -> None:
            self.lost = False

        def raise_if_lost(self) -> None:
            if self.lost:
                raise RunLeaseLost("lost during tool execution")

    fence = Fence()
    registry = ToolRegistry([ToolSpec(name="fixture.write", read_only=False)])
    executor = ToolExecutor(registry)

    def handler(call, _context):
        fence.lost = True
        return ToolResult(call=call, status="succeeded", summary="written")

    executor.register("fixture.write", handler)
    context = ToolExecutionContext(
        run_id="tool-fence-run",
        user_id="user-1",
        execution_fence=fence,
    )

    with pytest.raises(RunLeaseLost, match="lost during tool execution"):
        executor.execute(
            ToolCall(id="call-1", name="fixture.write", arguments={}),
            context,
        )

    assert context.fence_token == "tool-owner"
    assert context.fence_generation == 3


def test_runtime_persists_the_active_execution_fence() -> None:
    state_store = InMemoryRuntimeStateStore()
    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(
            RuntimePorts(
                runtime_state_store=state_store,
                run_lease_store=InMemoryRunLeaseStore(),
                run_lease_owner_id="worker-persist",
            )
        )
        .build()
    )

    result = runtime.run(
        RuntimeRequest(question="hello", run_id="fence-persist-run"),
        provider=ImmediateProvider(),
    )
    snapshot = state_store.load_snapshot("fence-persist-run")

    assert result.status.value == "completed"
    assert snapshot.fence_generation == 1
    assert snapshot.fence_token
