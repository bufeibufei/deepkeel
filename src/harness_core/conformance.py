from __future__ import annotations

from collections.abc import Callable, Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field

from harness_core.capabilities import (
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPack,
    capability_pack_spec_from_manifest,
    capability_pack_spec,
)
from harness_core.capability_control import (
    CapabilityPackageManager,
    InMemoryCapabilityPackageStore,
)
from harness_core.capability_manifest import CapabilityManifest
from harness_core.composition import HarnessRuntimeBuilder
from harness_core.evaluation import EvalCase, EvalSuiteReport, EvalSuiteRunner
from harness_core.runtime_api import RuntimeRequest, RuntimeResult
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutor
from harness_core.version import HARNESS_CORE_CONTRACT_VERSION, HARNESS_CORE_VERSION


REQUIRED_CAPABILITY_EVAL_TAGS = frozenset(
    {
        "tool_selection",
        "argument_generation",
        "task_completion",
        "recovery",
        "answer_quality",
    }
)


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
    permission_coverage: dict[str, list[str]] = Field(default_factory=dict)
    manifest_validated: bool = False
    runtime_generation_id: str = ""
    declared_permissions: list[str] = Field(default_factory=list)
    declared_budget: dict[str, object] = Field(default_factory=dict)
    state_schema_version: str = ""
    resume_compatible_versions: list[str] = Field(default_factory=list)
    declared_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    installed_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    missing_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    undeclared_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CapabilityLifecycleConformanceReport(BaseModel):
    """Control-plane lifecycle checks run without product-owned infrastructure."""

    model_config = ConfigDict(extra="forbid")

    installed: bool = False
    discovered: bool = False
    disabled: bool = False
    enabled: bool = False
    upgraded: bool = False
    rollback_succeeded: bool = False
    resume_compatible: bool = False
    generations: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues and all(
            (
                self.installed,
                self.discovered,
                self.disabled,
                self.enabled,
                self.upgraded,
                self.rollback_succeeded,
                self.resume_compatible,
            )
        )


class CapabilityPackageCertificationReport(BaseModel):
    """Release-gate report covering structure, lifecycle, and behavior."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "harness-capability-certification-v1"
    package_id: str
    package_version: str
    conformance: CapabilityPackConformanceReport
    lifecycle: CapabilityLifecycleConformanceReport
    evaluation: EvalSuiteReport | None = None
    required_eval_tags: list[str] = Field(default_factory=list)
    covered_eval_tags: list[str] = Field(default_factory=list)
    missing_eval_tags: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.conformance.passed
            and self.lifecycle.passed
            and self.evaluation is not None
            and self.evaluation.passed
            and not self.missing_eval_tags
            and not self.issues
        )


def validate_capability_pack(
    pack: CapabilityPack,
    *,
    declared_tools: Iterable[str] | None = None,
    registry: ToolRegistry | None = None,
    executor: ToolExecutor | None = None,
    manifest: CapabilityManifest | None = None,
    dependency_manifests: Iterable[CapabilityManifest] = (),
) -> CapabilityPackConformanceReport:
    """Build a pack through the public SDK and verify its registration contract."""

    pack_spec = capability_pack_spec(pack)
    materialized_dependencies = tuple(dependency_manifests)
    declared = pack_spec.declared_tools if declared_tools is None else declared_tools
    tool_names = list(
        dict.fromkeys(str(name).strip() for name in declared if str(name).strip())
    )
    resolved_registry = registry or (
        executor.registry if executor is not None else ToolRegistry()
    )
    issues: list[str] = []
    manifest_issues: list[str] = []
    generation_id = ""
    if manifest is not None:
        expected_spec = capability_pack_spec_from_manifest(manifest)
        manifest_issues = _manifest_spec_issues(pack_spec, expected_spec)
        if not manifest_issues:
            try:
                manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
                _install_dependencies(manager, manifest, materialized_dependencies)
                generation_id = manager.install(manifest).active_generation_id
            except Exception as exc:
                manifest_issues.append(
                    "manifest lifecycle validation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
    try:
        builder = HarnessRuntimeBuilder(
            resolved_registry,
            executor,
        ).with_strict_capability_conformance(False)
        if manifest is not None:
            for dependency in _dependency_order(
                manifest,
                materialized_dependencies,
            ):
                builder.add_capability_pack(
                    _ManifestOnlyPack(dependency),
                    manifest=dependency,
                )
        runtime = builder.add_capability_pack(
            pack,
            manifest=manifest,
        ).build()
    except Exception as exc:
        return CapabilityPackConformanceReport(
            package_id=pack_spec.package_id,
            package_version=pack_spec.package_version,
            passed=False,
            declared_tools=tool_names,
            manifest_validated=manifest is not None and not manifest_issues,
            runtime_generation_id=generation_id,
            declared_permissions=list(manifest.permissions) if manifest else [],
            declared_budget=manifest.budget.limits() if manifest else {},
            state_schema_version=manifest.state_schema_version if manifest else "",
            resume_compatible_versions=(
                list(manifest.resume_compatible_versions) if manifest else []
            ),
            issues=[
                *manifest_issues,
                f"composition failed: {type(exc).__name__}: {exc}",
            ],
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
    permission_coverage, permission_issues = _permission_coverage(
        runtime.tool_registry,
        manifest,
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
    issues.extend(permission_issues)
    if missing_capabilities:
        issues.append("declared non-tool capabilities are not installed")
    if undeclared_capabilities:
        issues.append("installed non-tool capabilities are missing from the declaration")
    issues.extend(catalog_issues)
    issues.extend(manifest_issues)
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
        permission_coverage=permission_coverage,
        manifest_validated=manifest is not None and not manifest_issues,
        runtime_generation_id=generation_id,
        declared_permissions=list(manifest.permissions) if manifest else [],
        declared_budget=manifest.budget.limits() if manifest else {},
        state_schema_version=manifest.state_schema_version if manifest else "",
        resume_compatible_versions=(
            list(manifest.resume_compatible_versions) if manifest else []
        ),
        declared_capabilities=declared_capabilities,
        installed_capabilities=installed_capabilities,
        missing_capabilities=missing_capabilities,
        undeclared_capabilities=undeclared_capabilities,
        issues=issues,
        warnings=warnings,
    )


def certify_capability_package(
    pack: CapabilityPack,
    *,
    manifest: CapabilityManifest,
    cases: Iterable[EvalCase],
    execute: Callable[[RuntimeRequest], RuntimeResult],
    trace_loader: Callable[[str], Iterable[object]] | None = None,
    required_eval_tags: Iterable[str] = REQUIRED_CAPABILITY_EVAL_TAGS,
    dependency_manifests: Iterable[CapabilityManifest] = (),
) -> CapabilityPackageCertificationReport:
    """Run the standard release gate for one independently installable package."""

    materialized_cases = tuple(cases)
    required_tags = {
        str(tag).strip() for tag in required_eval_tags if str(tag).strip()
    }
    covered_tags = {
        str(tag).strip()
        for case in materialized_cases
        for tag in case.tags
        if str(tag).strip()
    }
    materialized_dependencies = tuple(dependency_manifests)
    conformance = validate_capability_pack(
        pack,
        manifest=manifest,
        dependency_manifests=materialized_dependencies,
    )
    lifecycle = _validate_package_lifecycle(
        manifest,
        dependency_manifests=materialized_dependencies,
    )
    evaluation = EvalSuiteRunner(
        execute,
        trace_loader=trace_loader,  # type: ignore[arg-type]
    ).run(
        f"capability:{manifest.id}@{manifest.version}",
        materialized_cases,
    )
    missing_tags = sorted(required_tags - covered_tags)
    issues: list[str] = []
    if missing_tags:
        issues.append("required capability evaluation scenarios are missing")
    if not materialized_cases:
        issues.append("capability certification requires executable evaluation cases")
    return CapabilityPackageCertificationReport(
        package_id=manifest.id,
        package_version=manifest.version,
        conformance=conformance,
        lifecycle=lifecycle,
        evaluation=evaluation,
        required_eval_tags=sorted(required_tags),
        covered_eval_tags=sorted(covered_tags),
        missing_eval_tags=missing_tags,
        issues=issues,
    )


def _validate_package_lifecycle(
    manifest: CapabilityManifest,
    *,
    dependency_manifests: Iterable[CapabilityManifest] = (),
) -> CapabilityLifecycleConformanceReport:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
    generations: list[str] = []
    values: dict[str, bool] = {}
    issues: list[str] = []
    try:
        _install_dependencies(manager, manifest, dependency_manifests)
        installed = manager.install(manifest)
        values["installed"] = True
        values["discovered"] = installed.get(manifest.id) is not None
        generations.append(installed.active_generation_id)

        disabled = manager.disable(manifest.id)
        values["disabled"] = disabled.get(manifest.id) is not None and not bool(
            disabled.get(manifest.id).enabled  # type: ignore[union-attr]
        )
        generations.append(disabled.active_generation_id)

        enabled = manager.enable(manifest.id)
        values["enabled"] = bool(
            enabled.get(manifest.id) and enabled.get(manifest.id).enabled  # type: ignore[union-attr]
        )
        generations.append(enabled.active_generation_id)

        future_version = _next_patch_version(manifest.version)
        upgraded_manifest = manifest.model_copy(
            update={
                "version": future_version,
                "resume_compatible_versions": tuple(
                    dict.fromkeys((*manifest.resume_compatible_versions, manifest.version))
                ),
            }
        )
        upgraded = manager.upgrade(upgraded_manifest)
        values["upgraded"] = bool(
            upgraded.get(manifest.id)
            and upgraded.get(manifest.id).manifest.version == future_version  # type: ignore[union-attr]
        )
        generations.append(upgraded.active_generation_id)
        values["resume_compatible"] = (
            manager.resume_compatibility_issues(enabled.active_generation_id) == ()
        )

        rolled_back = manager.rollback(manifest.id, version=manifest.version)
        values["rollback_succeeded"] = bool(
            rolled_back.get(manifest.id)
            and rolled_back.get(manifest.id).manifest.version == manifest.version  # type: ignore[union-attr]
        )
        generations.append(rolled_back.active_generation_id)
    except Exception as exc:
        issues.append(f"package lifecycle failed: {type(exc).__name__}: {exc}")
    return CapabilityLifecycleConformanceReport(
        **values,
        generations=list(dict.fromkeys(generations)),
        issues=issues,
    )


def _install_dependencies(
    manager: CapabilityPackageManager,
    manifest: CapabilityManifest,
    dependency_manifests: Iterable[CapabilityManifest],
) -> None:
    for item in _dependency_order(manifest, tuple(dependency_manifests)):
        manager.install(item)


def _dependency_order(
    manifest: CapabilityManifest,
    dependency_manifests: tuple[CapabilityManifest, ...],
) -> tuple[CapabilityManifest, ...]:
    pending = {
        item.id: item
        for item in dependency_manifests
        if item.id != manifest.id
    }
    required_ids = set(manifest.dependencies)
    missing = sorted(required_ids - set(pending))
    if missing:
        raise ValueError(
            "dependency manifests are missing: " + ", ".join(missing)
        )
    ordered: list[CapabilityManifest] = []
    installed: set[str] = set()
    while pending:
        ready = sorted(
            (
                item
                for item in pending.values()
                if set(item.dependencies).issubset(installed)
            ),
            key=lambda item: item.id,
        )
        if not ready:
            raise ValueError(
                "dependency manifests are cyclic or incomplete: "
                + ", ".join(sorted(pending))
            )
        for item in ready:
            ordered.append(item)
            installed.add(item.id)
            pending.pop(item.id)
    return tuple(ordered)


class _ManifestOnlyPack:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.spec = capability_pack_spec_from_manifest(manifest)

    def install(
        self,
        _context: CapabilityInstallContext,
    ) -> CapabilityContribution:
        return CapabilityContribution(package_id=self.spec.package_id)


def _next_patch_version(version: str) -> str:
    core, separator, suffix = str(version).partition("-")
    parts = core.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(
            "capability certification requires a semantic package version"
        )
    major, minor, patch = (int(part) for part in parts)
    candidate = f"{major}.{minor}.{patch + 1}"
    return f"{candidate}-{suffix}" if separator and suffix else candidate


def _manifest_spec_issues(
    actual: object,
    expected: object,
) -> list[str]:
    fields = (
        "package_id",
        "contract_version",
        "package_version",
        "declared_tools",
        "declared_skills",
        "declared_artifact_types",
        "declared_handoffs",
        "declared_tool_providers",
        "declared_subagents",
        "declared_hooks",
        "declared_context_contributors",
        "declared_resources",
        "required_scopes",
    )
    return [
        f"manifest and pack spec disagree on {field_name}"
        for field_name in fields
        if getattr(actual, field_name) != getattr(expected, field_name)
    ]


def _permission_coverage(
    registry: ToolRegistry,
    manifest: CapabilityManifest | None,
    tool_names: set[str],
) -> tuple[dict[str, list[str]], list[str]]:
    if manifest is None:
        return {}, []
    coverage: dict[str, list[str]] = {}
    effective_scopes: set[str] = set()
    for name in sorted(tool_names):
        try:
            tool = registry.get(name)
        except KeyError:
            continue
        scopes = sorted(
            {
                str(scope).strip()
                for scope in tool.runtime_policy.get("required_scopes") or ()
                if str(scope).strip()
            }
        )
        coverage[name] = scopes
        effective_scopes.update(scopes)
    declared = set(manifest.permissions)
    mapped = {
        tool_name: set(scopes)
        for tool_name, scopes in manifest.tool_permissions.items()
    }
    missing_mappings = sorted(
        tool_name
        for tool_name in tool_names
        if tool_name not in mapped and declared
    )
    mismatched_mappings = sorted(
        tool_name
        for tool_name, scopes in mapped.items()
        if tool_name in coverage and set(coverage[tool_name]) != scopes
    )
    missing = sorted(declared - effective_scopes)
    undeclared = sorted(effective_scopes - declared)
    issues: list[str] = []
    if missing:
        issues.append(
            "manifest permissions are not enforced by any declared tool: "
            + ", ".join(missing)
        )
    if missing_mappings:
        issues.append(
            "declared tools are missing explicit permission mappings: "
            + ", ".join(missing_mappings)
        )
    if mismatched_mappings:
        issues.append(
            "runtime tool permissions differ from the capability manifest: "
            + ", ".join(mismatched_mappings)
        )
    if undeclared:
        issues.append(
            "tool permissions are missing from the capability manifest: "
            + ", ".join(undeclared)
        )
    return coverage, issues


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
        "tool_providers": list(pack_spec.declared_tool_providers),
        "subagents": list(pack_spec.declared_subagents),
        "hooks": list(pack_spec.declared_hooks),
        "context_contributors": list(pack_spec.declared_context_contributors),
        "resources": list(pack_spec.declared_resources),
    }


def _installed_capabilities(contribution) -> dict[str, list[str]]:
    if contribution is None:
        return {
            "skills": [],
            "artifact_types": [],
            "handoffs": [],
            "tool_providers": [],
            "subagents": [],
            "hooks": [],
            "context_contributors": [],
            "resources": [],
        }
    return {
        field_name: list(getattr(contribution, field_name))
        for field_name in (
            "skills",
            "artifact_types",
            "handoffs",
            "tool_providers",
            "subagents",
            "hooks",
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
