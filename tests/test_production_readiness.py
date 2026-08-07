from __future__ import annotations

import pytest

from deepkeel.adapter_sdk import (
    AsyncRuntimeStateStoreAdapter,
    HarnessRuntimeBuilder,
    ProductionConfigurationError,
    RuntimePorts,
)
from deepkeel.extension_sdk import AsyncToolExecutionStoreAdapter
from deepkeel.state_store import InMemoryRuntimeStateStore
from deepkeel.tools import InMemoryToolExecutionStore


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
        "tool_view_mode": "enforced",
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
    builder = HarnessRuntimeBuilder(profile="production").with_ports(_production_ports())

    report = builder.production_readiness()
    runtime = builder.build_production()

    assert report.ready is True
    assert runtime.runtime_state_store is not None
    assert builder.profile.name == "production"
    assert runtime.tool_view_mode == "enforced"


def test_production_profile_cannot_bypass_readiness_with_plain_build() -> None:
    builder = HarnessRuntimeBuilder(profile="production")

    with pytest.raises(ProductionConfigurationError):
        builder.build()


def test_production_rejects_non_enforced_tool_disclosure() -> None:
    report = HarnessRuntimeBuilder().with_ports(
        _production_ports(tool_view_mode="shadow")
    ).production_readiness()

    assert any(
        issue.code == "TOOL_DISCLOSURE_NOT_ENFORCED"
        and issue.port == "tool_view_mode"
        for issue in report.errors
    )


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


def test_production_accepts_native_async_tool_execution_store() -> None:
    report = HarnessRuntimeBuilder().with_ports(
        _production_ports(
            tool_execution_store=None,
            async_tool_execution_store=DurableHostPort(),
        )
    ).production_readiness()

    assert report.ready is True
    assert not any(issue.port == "tool_execution_store" for issue in report.warnings)


def test_production_rejects_ambiguous_tool_execution_stores() -> None:
    report = HarnessRuntimeBuilder().with_ports(
        _production_ports(async_tool_execution_store=DurableHostPort())
    ).production_readiness()

    assert any(
        issue.code == "AMBIGUOUS_PRODUCTION_PORT"
        and issue.port == "tool_execution_store"
        for issue in report.errors
    )


def test_production_rejects_in_memory_tool_store_hidden_by_async_bridge() -> None:
    report = HarnessRuntimeBuilder().with_ports(
        _production_ports(
            tool_execution_store=None,
            async_tool_execution_store=AsyncToolExecutionStoreAdapter(
                InMemoryToolExecutionStore()
            ),
        )
    ).production_readiness()

    assert any(
        issue.code == "PROCESS_LOCAL_PRODUCTION_PORT"
        and issue.port == "tool_execution_store"
        for issue in report.errors
    )
