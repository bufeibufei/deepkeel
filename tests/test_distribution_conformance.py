"""Run the clean-install scenarios as package-owned coverage tests."""

import harness_core

try:
    from verification.installed_conformance import (
        verify_mcp_and_subagent,
        verify_runtime_and_streaming,
        verify_skill_artifact_and_reference_contracts,
        verify_tools_parallel_failure_and_references,
        verify_wait_resume_async_and_cancel,
    )
except ModuleNotFoundError:
    from packages.harness_core.verification.installed_conformance import (
        verify_mcp_and_subagent,
        verify_runtime_and_streaming,
        verify_skill_artifact_and_reference_contracts,
        verify_tools_parallel_failure_and_references,
        verify_wait_resume_async_and_cancel,
    )


def test_runtime_and_streaming_conformance() -> None:
    verify_runtime_and_streaming()


def test_tools_parallel_failure_and_reference_conformance() -> None:
    verify_tools_parallel_failure_and_references()


def test_wait_resume_async_and_cancel_conformance() -> None:
    verify_wait_resume_async_and_cancel()


def test_skill_artifact_and_reference_conformance() -> None:
    verify_skill_artifact_and_reference_contracts()


def test_mcp_and_subagent_conformance() -> None:
    verify_mcp_and_subagent()


def test_source_installation_identity_conformance() -> None:
    assert harness_core.HARNESS_CORE_VERSION == "3.35.1"
    assert harness_core.HARNESS_CORE_CONTRACT_VERSION == "harness-core-v3"
    assert tuple(harness_core.__all__) == (
        "HARNESS_CORE_CONTRACT_VERSION",
        "HARNESS_CORE_VERSION",
        "adapter_sdk",
        "extension_sdk",
        "mcp_sdk",
        "memory_sdk",
        "orchestration_sdk",
        "runtime_sdk",
    )
    from harness_core.version import (
        EVENT_SCHEMA_VERSION,
        PACKAGE_VERSION,
        RUNTIME_CONTRACT_VERSION,
        SDK_API_VERSION,
    )

    assert PACKAGE_VERSION == "3.35.1"
    assert RUNTIME_CONTRACT_VERSION == "harness-core-v3"
    assert EVENT_SCHEMA_VERSION == "harness-runtime-event-v1"
    assert SDK_API_VERSION == "3.14.0"
