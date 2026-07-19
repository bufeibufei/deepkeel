from __future__ import annotations

from typing import Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field

from harness_core.capabilities import (
    CapabilityPack,
    capability_pack_spec,
)
from harness_core.composition import HarnessRuntimeBuilder
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutor
from harness_core.version import HARNESS_CORE_CONTRACT_VERSION, HARNESS_CORE_VERSION


class CapabilityPackConformanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "harness-capability-conformance-v1"
    core_version: str = HARNESS_CORE_VERSION
    contract_version: str = HARNESS_CORE_CONTRACT_VERSION
    package_id: str
    package_version: str = "0.0.0"
    passed: bool
    declared_tools: list[str] = Field(default_factory=list)
    registered_handlers: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)
    missing_handlers: list[str] = Field(default_factory=list)
    undeclared_tools: list[str] = Field(default_factory=list)
    invalid_tool_contracts: list[str] = Field(default_factory=list)
    declared_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    installed_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    missing_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    undeclared_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_capability_pack(
    pack: CapabilityPack,
    *,
    declared_tools: Iterable[str] | None = None,
    registry: ToolRegistry | None = None,
    executor: ToolExecutor | None = None,
) -> CapabilityPackConformanceReport:
    """Build a pack through the public SDK and verify its registration contract."""

    pack_spec = capability_pack_spec(pack)
    declared = pack_spec.declared_tools if declared_tools is None else declared_tools
    tool_names = list(
        dict.fromkeys(str(name).strip() for name in declared if str(name).strip())
    )
    resolved_registry = registry or (
        executor.registry if executor is not None else ToolRegistry()
    )
    issues: list[str] = []
    try:
        runtime = (
            HarnessRuntimeBuilder(resolved_registry, executor)
            .with_strict_capability_conformance(False)
            .add_capability_pack(pack)
            .build()
        )
    except Exception as exc:
        return CapabilityPackConformanceReport(
            package_id=pack_spec.package_id,
            package_version=pack_spec.package_version,
            passed=False,
            declared_tools=tool_names,
            issues=[f"composition failed: {type(exc).__name__}: {exc}"],
        )

    available_tools = {tool.name for tool in runtime.tool_registry.list_tools()}
    registered_handlers = set(runtime.tool_executor.registered_tools)
    contribution = next(
        (
            item
            for item in runtime.capability_contributions
            if item.package_id == pack_spec.package_id
        ),
        None,
    )
    contributed_tools = set(contribution.tools if contribution is not None else ())
    missing_tools = sorted(set(tool_names) - available_tools)
    missing_handlers = sorted(set(tool_names) - registered_handlers)
    undeclared_tools = sorted(contributed_tools - set(tool_names)) if tool_names else []
    invalid_contracts = _invalid_tool_contracts(
        runtime.tool_registry,
        set(tool_names) | contributed_tools,
    )
    declared_capabilities = _declared_capabilities(pack_spec)
    installed_capabilities = _installed_capabilities(contribution)
    missing_capabilities = {
        kind: sorted(set(names) - set(installed_capabilities.get(kind, [])))
        for kind, names in declared_capabilities.items()
        if set(names) - set(installed_capabilities.get(kind, []))
    }
    undeclared_capabilities = {
        kind: sorted(set(names) - set(declared_capabilities.get(kind, [])))
        for kind, names in installed_capabilities.items()
        if set(names) - set(declared_capabilities.get(kind, []))
    }
    catalog_issues = _catalog_issues(runtime)
    warnings: list[str] = []
    if missing_tools:
        issues.append("declared tools are missing from ToolRegistry")
    if missing_handlers:
        issues.append("declared tools are missing executable handlers")
    if undeclared_tools:
        issues.append("installed tools are missing from the capability declaration")
    if invalid_contracts:
        issues.append("tool contracts contain invalid JSON Schema or runtime semantics")
    if missing_capabilities:
        issues.append("declared non-tool capabilities are not installed")
    if undeclared_capabilities:
        issues.append("installed non-tool capabilities are missing from the declaration")
    issues.extend(catalog_issues)
    if not tool_names:
        warnings.append("capability pack does not declare any tools")
    return CapabilityPackConformanceReport(
        package_id=pack_spec.package_id,
        package_version=pack_spec.package_version,
        passed=not issues,
        declared_tools=tool_names,
        registered_handlers=sorted(registered_handlers),
        missing_tools=missing_tools,
        missing_handlers=missing_handlers,
        undeclared_tools=undeclared_tools,
        invalid_tool_contracts=invalid_contracts,
        declared_capabilities=declared_capabilities,
        installed_capabilities=installed_capabilities,
        missing_capabilities=missing_capabilities,
        undeclared_capabilities=undeclared_capabilities,
        issues=issues,
        warnings=warnings,
    )


def _invalid_tool_contracts(registry: ToolRegistry, tool_names: set[str]) -> list[str]:
    invalid: list[str] = []
    for name in sorted(tool_names):
        try:
            tool = registry.get(name)
        except KeyError:
            continue
        parameters = tool.formal_parameters_schema()
        if not parameters:
            invalid.append(f"{name}: parameters_schema is required")
            continue
        schemas = (
            ("parameters_schema", parameters),
            ("output_schema", tool.output_schema),
        )
        for label, schema in schemas:
            if not schema:
                continue
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                invalid.append(f"{name}: invalid {label}: {exc.message}")
        properties = parameters.get("properties")
        property_names = set(properties) if isinstance(properties, dict) else set()
        unknown_required = sorted(set(tool.required_args) - property_names)
        if unknown_required:
            invalid.append(
                f"{name}: required_args missing from parameters_schema: {', '.join(unknown_required)}"
            )
        if tool.parallel_safe and not tool.read_only:
            invalid.append(f"{name}: write tools cannot be parallel_safe")
    return invalid


def _declared_capabilities(pack_spec) -> dict[str, list[str]]:
    return {
        "skills": list(pack_spec.declared_skills),
        "artifact_types": list(pack_spec.declared_artifact_types),
        "handoffs": list(pack_spec.declared_handoffs),
        "mcp_servers": list(pack_spec.declared_mcp_servers),
        "subagents": list(pack_spec.declared_subagents),
        "context_contributors": list(pack_spec.declared_context_contributors),
        "resources": list(pack_spec.declared_resources),
    }


def _installed_capabilities(contribution) -> dict[str, list[str]]:
    if contribution is None:
        return {
            "skills": [],
            "artifact_types": [],
            "handoffs": [],
            "mcp_servers": [],
            "subagents": [],
            "context_contributors": [],
            "resources": [],
        }
    return {
        field_name: list(getattr(contribution, field_name))
        for field_name in (
            "skills",
            "artifact_types",
            "handoffs",
            "mcp_servers",
            "subagents",
            "context_contributors",
            "resources",
        )
    }


def _catalog_issues(runtime) -> list[str]:
    catalog = runtime.capability_catalog
    available_tools = {tool.name for tool in runtime.tool_registry.list_tools()}
    artifact_types = set(catalog.artifact_types)
    issues: list[str] = []
    for tool_name, handoff in catalog.handoffs.items():
        if tool_name not in available_tools:
            issues.append(f"handoff {tool_name}: referenced tool is not registered")
        if (
            handoff.completion_artifact_type
            and handoff.completion_artifact_type not in artifact_types
        ):
            issues.append(
                f"handoff {tool_name}: completion artifact type is not registered: "
                f"{handoff.completion_artifact_type}"
            )
    for subagent in catalog.subagents.values():
        missing_tools = sorted(set(subagent.tool_allowlist) - available_tools)
        if missing_tools:
            issues.append(
                f"subagent {subagent.id}: tool allowlist contains unknown tools: "
                f"{', '.join(missing_tools)}"
            )
    return issues
