from __future__ import annotations

import pytest

from deepkeel.adapter_sdk import (
    AsyncRuntimeStateStoreAdapter,
    HarnessRuntimeBuilder,
    ProductionConfigurationError,
    RuntimePorts,
)
from deepkeel.state_store import InMemoryRuntimeStateStore


class DurableHostPort:
    """Structural test double representing an external durable Host adapter."""

    terminal_settlement_owner = "runtime"


def _production_ports(**overrides) -> RuntimePorts:
    durable = DurableHostPort()
    values = {
        "checkpointer": durable,
        "runtime_state_store": durable,
        "event_journal": durable,
        "run_lease_store": durable,
        "model_invocation_store": durable,
        "tool_execution_store": durable,
        "budget_ledger": durable,
        "model_health_store": durable,
        "run_control": durable,
        "telemetry": durable,
    }
    values.update(overrides)
    return RuntimePorts(**values)


def test_default_builder_fails_closed_for_production() -> None:
    builder = HarnessRuntimeBuilder()
    report = builder.production_readiness()

    assert report.ready is False
    assert {issue.port for issue in report.errors} >= {
        "checkpointer",
        "runtime_state_store",
        "event_journal",
        "run_lease_store",
        "model_invocation_store",
        "tool_execution_store",
    }
    with pytest.raises(ProductionConfigurationError) as captured:
        builder.build_production()
    assert captured.value.code == "PRODUCTION_CONFIGURATION_INVALID"


def test_production_builder_accepts_explicit_external_ports() -> None:
    builder = HarnessRuntimeBuilder().with_ports(_production_ports())

    report = builder.production_readiness()
    runtime = builder.build_production()

    assert report.ready is True
    assert runtime.runtime_state_store is not None


def test_production_readiness_rejects_in_memory_port_hidden_by_async_bridge() -> None:
    report = HarnessRuntimeBuilder().with_ports(
        _production_ports(
            runtime_state_store=None,
            async_runtime_state_store=AsyncRuntimeStateStoreAdapter(
                InMemoryRuntimeStateStore()
            ),
        )
    ).production_readiness()

    assert any(
        issue.code == "PROCESS_LOCAL_PRODUCTION_PORT"
        and issue.port == "runtime_state_store"
        for issue in report.errors
    )
