from harness_core.adapter_sdk import (
    verify_durable_checkpoint_store_contract,
    verify_runtime_state_store_contract,
    verify_tool_execution_store_contract,
)
from harness_core.runtime_sdk import (
    InMemoryDurableCheckpointStore,
    InMemoryRuntimeStateStore,
)
from harness_core.tools import InMemoryToolExecutionStore


def test_runtime_state_store_reference_adapter_passes_contract() -> None:
    verify_runtime_state_store_contract(
        InMemoryRuntimeStateStore(),
        run_id="conformance-runtime",
    )


def test_tool_execution_store_reference_adapter_passes_contract() -> None:
    verify_tool_execution_store_contract(
        InMemoryToolExecutionStore(),
        run_id="conformance-tool",
    )


def test_durable_checkpoint_reference_adapter_passes_contract() -> None:
    verify_durable_checkpoint_store_contract(
        InMemoryDurableCheckpointStore(),
        run_id="conformance-checkpoint",
        user_id="user-1",
    )
