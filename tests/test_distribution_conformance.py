"""Run the clean-install scenarios as package-owned coverage tests."""

import deepkeel

try:
    from verification.installed_conformance import (
        verify_mcp_and_subagent,
        verify_runtime_and_streaming,
        verify_skill_artifact_and_reference_contracts,
        verify_tools_parallel_failure_and_references,
        verify_wait_resume_async_and_cancel,
    )
except ModuleNotFoundError:
    from packages.deepkeel.verification.installed_conformance import (
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
    assert deepkeel.DEEPKEEL_VERSION == "4.1.0rc1"
    assert deepkeel.DEEPKEEL_CONTRACT_VERSION == "harness-core-v3"
    assert tuple(deepkeel.__all__) == (
        "DEEPKEEL_CONTRACT_VERSION",
        "DEEPKEEL_VERSION",
        "adapter_sdk",
        "extension_sdk",
        "mcp_sdk",
        "memory_sdk",
        "orchestration_sdk",
        "runtime_sdk",
    )
    from deepkeel.version import (
        EVENT_SCHEMA_VERSION,
        PACKAGE_VERSION,
        RUNTIME_CONTRACT_VERSION,
        SDK_API_VERSION,
    )

    assert PACKAGE_VERSION == "4.1.0rc1"
    assert RUNTIME_CONTRACT_VERSION == "harness-core-v3"
    assert EVENT_SCHEMA_VERSION == "harness-runtime-event-v1"
    assert SDK_API_VERSION == "4.1.0rc1"
