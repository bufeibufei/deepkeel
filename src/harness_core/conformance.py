from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from harness_core.composition import CapabilityPack, HarnessRuntimeBuilder
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutor
from harness_core.version import HARNESS_CORE_CONTRACT_VERSION, HARNESS_CORE_VERSION


class CapabilityPackConformanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "harness-capability-conformance-v1"
    core_version: str = HARNESS_CORE_VERSION
    contract_version: str = HARNESS_CORE_CONTRACT_VERSION
    package_id: str
    passed: bool
    declared_tools: list[str] = Field(default_factory=list)
    registered_handlers: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)
    missing_handlers: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def validate_capability_pack(
    pack: CapabilityPack,
    *,
    declared_tools: Iterable[str],
    registry: ToolRegistry | None = None,
    executor: ToolExecutor | None = None,
) -> CapabilityPackConformanceReport:
    """Build a pack through the public SDK and verify its registration contract."""

    tool_names = list(
        dict.fromkeys(str(name).strip() for name in declared_tools if str(name).strip())
    )
    resolved_registry = registry or (
        executor.registry if executor is not None else ToolRegistry()
    )
    issues: list[str] = []
    try:
        runtime = (
            HarnessRuntimeBuilder(resolved_registry, executor)
            .add_capability_pack(pack)
            .build()
        )
    except Exception as exc:
        return CapabilityPackConformanceReport(
            package_id=str(getattr(pack, "package_id", "") or "<missing>"),
            passed=False,
            declared_tools=tool_names,
            issues=[f"composition failed: {type(exc).__name__}: {exc}"],
        )

    available_tools = {spec.name for spec in runtime.tool_registry.list_tools()}
    registered_handlers = set(runtime.tool_executor.registered_tools)
    missing_tools = sorted(set(tool_names) - available_tools)
    missing_handlers = sorted(set(tool_names) - registered_handlers)
    if missing_tools:
        issues.append("declared tools are missing from ToolRegistry")
    if missing_handlers:
        issues.append("declared tools are missing executable handlers")
    return CapabilityPackConformanceReport(
        package_id=pack.package_id,
        passed=not issues,
        declared_tools=tool_names,
        registered_handlers=sorted(registered_handlers),
        missing_tools=missing_tools,
        missing_handlers=missing_handlers,
        issues=issues,
    )
