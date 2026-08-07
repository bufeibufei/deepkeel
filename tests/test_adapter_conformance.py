from deepkeel.adapter_sdk import (
    InMemoryBudgetLedger,
    InMemoryModelInvocationStore,
    InMemoryRuntimeEventJournal,
    verify_model_invocation_store_contract,
    verify_runtime_event_projection_contract,
    verify_runtime_event_journal_contract,
    verify_durable_checkpoint_store_contract,
    verify_run_lease_store_contract,
    verify_runtime_state_store_contract,
    verify_tool_execution_store_contract,
    verify_capability_package_store_contract,
    verify_budget_ledger_contract,
)
from deepkeel.extension_sdk import InMemoryCapabilityPackageStore
from deepkeel.runtime_sdk import (
    InMemoryDurableCheckpointStore,
    InMemoryRuntimeStateStore,
)
from deepkeel.adapter_sdk import InMemoryRunLeaseStore
from deepkeel.tools import InMemoryToolExecutionStore
from deepkeel.runtime_sdk import normalize_runtime_event


def test_runtime_state_store_reference_adapter_passes_contract() -> None:
    verify_runtime_state_store_contract(
        InMemoryRuntimeStateStore(),
        run_id="conformance-runtime",
    )


def test_budget_ledger_reference_adapter_passes_contract() -> None:
    verify_budget_ledger_contract(
        InMemoryBudgetLedger(),
        run_id="conformance-budget",
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


def test_run_lease_reference_adapter_passes_contract() -> None:
    verify_run_lease_store_contract(InMemoryRunLeaseStore(), run_id="conformance-lease")


def test_runtime_event_journal_reference_adapter_passes_contract() -> None:
    verify_runtime_event_journal_contract(
        InMemoryRuntimeEventJournal(),
        run_id="conformance-events",
    )


def test_runtime_event_projection_reference_adapter_passes_contract() -> None:
    verify_runtime_event_projection_contract(normalize_runtime_event)


def test_model_invocation_store_reference_adapter_passes_contract() -> None:
    verify_model_invocation_store_contract(
        InMemoryModelInvocationStore(),
        run_id="conformance-model",
    )


def test_capability_package_store_reference_adapter_passes_contract() -> None:
    verify_capability_package_store_contract(InMemoryCapabilityPackageStore())
