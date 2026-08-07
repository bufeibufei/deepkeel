from __future__ import annotations

import pytest

from deepkeel.adapter_sdk import HarnessRuntimeBuilder, RuntimePorts
from deepkeel.extension_sdk import (
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)
from deepkeel.runtime_sdk import (
    InMemoryDurableCheckpointStore,
    InMemoryRuntimeStateStore,
    RuntimeRequest,
    ToolCall,
    ToolResult,
)
from deepkeel.tools import InMemoryToolExecutionStore


class FailingClaimStore:
    def claim(self, **_kwargs):
        raise ConnectionError("database unavailable")

    def replay(self, claim):
        raise AssertionError(f"unexpected replay: {claim}")

    def settle(self, claim, result):
        raise AssertionError(f"unexpected settlement: {claim} {result}")


def test_tool_claim_storage_failure_fails_closed_and_remains_retryable() -> None:
    registry = ToolRegistry([ToolSpec(name="records.read", read_only=True)])
    executor = ToolExecutor(registry, execution_store=FailingClaimStore())
    executed = False

    def handler(call, context):
        nonlocal executed
        executed = True
        return ToolResult(call=call, status="succeeded")

    executor.register("records.read", handler)
    result = executor.execute(
        ToolCall(
            id="claim-failure",
            name="records.read",
            idempotency_key="claim-failure",
        ),
        ToolExecutionContext(run_id="run-fault", user_id="user-a"),
    )

    assert executed is False
    assert result.status == "failed"
    assert result.retryable is True
    assert result.metadata["runtime_metrics"]["phase"] == "idempotent_claim"


def test_parallel_tool_failure_does_not_discard_successful_sibling() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(name="batch.ok", read_only=True),
            ToolSpec(name="batch.fail", read_only=True),
        ]
    )
    executor = ToolExecutor(registry, max_parallel_tools=2)

    def succeed(call, context):
        return ToolResult(call=call, status="succeeded", summary="kept")

    def fail(call, context):
        raise TimeoutError("downstream timeout")

    executor.register("batch.ok", succeed)
    executor.register("batch.fail", fail)
    results = executor.execute_many(
        [
            ToolCall(id="ok", name="batch.ok"),
            ToolCall(id="fail", name="batch.fail"),
        ],
        ToolExecutionContext(run_id="run-parallel-fault", user_id="user-a"),
    )

    assert [result.status for result in results] == ["succeeded", "failed"]
    assert results[0].summary == "kept"
    assert "downstream timeout" in results[1].error


class SettlementFailureStore(InMemoryToolExecutionStore):
    def settle(self, claim, result):
        raise ConnectionError("settlement database unavailable")


def test_side_effect_success_without_durable_settlement_fails_closed() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="records.write",
                read_only=False,
                parallel_safe=False,
                reexecution_safe=False,
            )
        ]
    )
    executor = ToolExecutor(registry, execution_store=SettlementFailureStore())
    executed = 0

    def handler(call, context):
        nonlocal executed
        executed += 1
        return ToolResult(call=call, status="succeeded", summary="side effect completed")

    executor.register("records.write", handler)
    result = executor.execute(
        ToolCall(
            id="settlement-failure",
            name="records.write",
            idempotency_key="settlement-failure",
        ),
        ToolExecutionContext(run_id="run-settlement-fault", user_id="user-a"),
    )

    assert executed == 1
    assert result.status == "failed"
    assert result.retryable is False
    assert result.metadata["settlement_failed"] is True
    assert result.metadata["reexecution_safe"] is False


def test_terminal_commit_failure_preserves_the_recovery_checkpoint() -> None:
    def fail_before_commit(stage: str) -> None:
        if stage == "before_commit":
            raise ConnectionError("terminal state database unavailable")

    class Provider:
        model = "settlement-fault-model"
        model_role = "fast"

        def complete_chat(self, _messages, **_kwargs):
            return {
                "message": {"role": "assistant", "content": "completed"},
                "finish_reason": "stop",
                "model": self.model,
            }

    run_id = "terminal-commit-failure"
    checkpoint_store = InMemoryDurableCheckpointStore()
    checkpoint_store.save(run_id, {"sentinel": "recoverable"})
    state_store = InMemoryRuntimeStateStore(fail_before_commit)
    runtime = HarnessRuntimeBuilder().with_ports(
        RuntimePorts(
            checkpoint_store=checkpoint_store,
            runtime_state_store=state_store,
        )
    ).build()

    with pytest.raises(RuntimeError, match="atomic runtime checkpoint commit failed"):
        runtime.run(
            RuntimeRequest(question="hello", run_id=run_id),
            provider=Provider(),
        )

    assert checkpoint_store.load(run_id) == {"sentinel": "recoverable"}
    assert state_store.load_snapshot(run_id).settled is False


def test_host_owned_terminal_settlement_defers_core_checkpoint_cleanup() -> None:
    class HostOwnedStateStore(InMemoryRuntimeStateStore):
        terminal_settlement_owner = "host"

    class Provider:
        model = "host-settlement-model"
        model_role = "fast"

        def complete_chat(self, _messages, **_kwargs):
            return {
                "message": {"role": "assistant", "content": "completed"},
                "finish_reason": "stop",
                "model": self.model,
            }

    run_id = "host-owned-settlement"
    checkpoint_store = InMemoryDurableCheckpointStore()
    checkpoint_store.save(run_id, {"sentinel": "host-must-clean"})
    runtime = HarnessRuntimeBuilder().with_ports(
        RuntimePorts(
            checkpoint_store=checkpoint_store,
            runtime_state_store=HostOwnedStateStore(),
        )
    ).build()

    result = runtime.run(
        RuntimeRequest(question="hello", run_id=run_id),
        provider=Provider(),
    )

    assert result.status.value == "completed"
    assert result.diagnostics["recovery"]["atomic_checkpoint"] == (
        "host_settlement_required"
    )
    assert result.diagnostics["recovery"]["checkpoint_cleanup"] == {
        "status": "deferred",
        "reason": "host_settlement_required",
    }
    assert checkpoint_store.load(run_id) == {"sentinel": "host-must-clean"}
