from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from jsonschema import Draft202012Validator

from deepkeel.entrypoints import AgentEntrypointSpec
from deepkeel.handoffs import HandoffSpec
from deepkeel.hooks import HookSpec
from deepkeel.subagents.contracts import SubAgentSpec
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tool_providers import ToolProvider, ToolProviderSpec, verify_tool_provider
from deepkeel.tools import ToolExecutor, ToolHandler
from deepkeel.version import DEEPKEEL_CONTRACT_VERSION


ContextContributor = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ArtifactTypeSpec:
    artifact_type: str
    schema: Mapping[str, Any] = field(default_factory=dict)
    title: str = ""

    def __post_init__(self) -> None:
        artifact_type = self.artifact_type.strip()
        if not artifact_type:
            raise ValueError("artifact_type must not be blank")
        Draft202012Validator.check_schema(dict(self.schema))
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "schema", MappingProxyType(dict(self.schema)))


@dataclass(frozen=True, slots=True)
class CapabilityPackSpec:
    """Versioned declaration consumed before a capability pack is installed."""

    package_id: str
    contract_version: str = DEEPKEEL_CONTRACT_VERSION
    package_version: str = "0.0.0"
    declared_tools: tuple[str, ...] = ()
    declared_skills: tuple[str, ...] = ()
    declared_artifact_types: tuple[str, ...] = ()
    declared_handoffs: tuple[str, ...] = ()
    declared_tool_providers: tuple[str, ...] = ()
    declared_subagents: tuple[str, ...] = ()
    declared_hooks: tuple[str, ...] = ()
    declared_context_contributors: tuple[str, ...] = ()
    declared_agent_entrypoints: tuple[str, ...] = ()
    declared_resources: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        package_id = self.package_id.strip()
        if not package_id:
            raise ValueError("capability pack must declare package_id")
        object.__setattr__(self, "package_id", package_id)
        object.__setattr__(self, "contract_version", self.contract_version.strip())
        object.__setattr__(self, "package_version", self.package_version.strip() or "0.0.0")
        for field_name in _DECLARATION_FIELDS:
            values = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in getattr(self, field_name)
                    if str(value).strip()
                )
            )
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class CapabilityContribution:
    """Capabilities actually installed by one package."""

    package_id: str
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    artifact_types: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()
    tool_providers: tuple[str, ...] = ()
    subagents: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    context_contributors: tuple[str, ...] = ()
    agent_entrypoints: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_id", self.package_id.strip())
        for field_name in _CONTRIBUTION_FIELDS:
            values = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in getattr(self, field_name)
                    if str(value).strip()
                )
            )
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class CapabilityCatalog:
    """Runtime-owned registry for non-tool capability extension points."""

    def __init__(self) -> None:
        self.skills: dict[str, object] = {}
        self.artifact_types: dict[str, ArtifactTypeSpec] = {}
        self.handoffs: dict[str, HandoffSpec] = {}
        self.tool_providers: dict[str, ToolProvider] = {}
        self.subagents: dict[str, SubAgentSpec] = {}
        self.hooks: dict[str, HookSpec] = {}
        self.context_contributors: dict[str, ContextContributor] = {}
        self.agent_entrypoints: dict[str, AgentEntrypointSpec] = {}
        self.resources: dict[str, object] = {}

    def register_skill(self, skill_id: str, spec: object) -> None:
        self._register(self.skills, skill_id, spec, "skill")

    def register_artifact_type(self, spec: ArtifactTypeSpec) -> None:
        self._register(self.artifact_types, spec.artifact_type, spec, "artifact type")

    def register_handoff(self, tool_name: str, spec: HandoffSpec) -> None:
        self._register(self.handoffs, tool_name, spec, "handoff")

    def register_tool_provider(self, provider: ToolProvider) -> ToolProviderSpec:
        spec = verify_tool_provider(provider)
        self._register(self.tool_providers, spec.provider_id, provider, "tool provider")
        return spec

    def register_subagent(self, spec: SubAgentSpec) -> None:
        self._register(self.subagents, spec.id, spec, "subagent")

    def register_hook(self, spec: HookSpec) -> None:
        self._register(self.hooks, spec.id, spec, "hook")

    def register_context_contributor(
        self,
        contributor_id: str,
        contributor: ContextContributor,
    ) -> None:
        if not callable(contributor):
            raise TypeError("context contributor must be callable")
        self._register(
            self.context_contributors,
            contributor_id,
            contributor,
            "context contributor",
        )

    def register_agent_entrypoint(self, spec: AgentEntrypointSpec) -> None:
        self._register(self.agent_entrypoints, spec.id, spec, "Agent entrypoint")

    def register_resource(self, resource_id: str, resource: object) -> None:
        if not callable(getattr(resource, "close", None)):
            raise TypeError("lifecycle resource must expose close()")
        self._register(self.resources, resource_id, resource, "lifecycle resource")

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            field_name: dict(getattr(self, field_name))
            for field_name in _CATALOG_FIELDS
        }

    def rollback(self, snapshot: Mapping[str, Mapping[str, Any]]) -> None:
        prior_resources = set(snapshot.get("resources", {}))
        for resource_id in reversed(tuple(self.resources)):
            if resource_id not in prior_resources:
                _close_resource(self.resources[resource_id])
        for field_name in _CATALOG_FIELDS:
            setattr(self, field_name, dict(snapshot.get(field_name, {})))

    def close(self) -> None:
        for resource in reversed(tuple(self.resources.values())):
            _close_resource(resource)
        self.resources = {}

    @staticmethod
    def _register(registry: dict[str, Any], raw_name: str, value: Any, label: str) -> None:
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError(f"{label} id must not be blank")
        if name in registry:
            raise ValueError(f"{label} is already registered: {name}")
        registry[name] = value


@dataclass(frozen=True, slots=True)
class CapabilityInstallContext:
    """Narrow SDK surface available while installing a domain capability pack."""

    registry: ToolRegistry
    executor: ToolExecutor
    catalog: CapabilityCatalog
    services: Mapping[str, object] = field(default_factory=dict)

    def register_tool(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self.registry.register(spec)
        self.executor.register(spec.name, handler)

    def register_skill(self, skill_id: str, spec: object) -> None:
        self.catalog.register_skill(skill_id, spec)

    def register_artifact_type(self, spec: ArtifactTypeSpec) -> None:
        self.catalog.register_artifact_type(spec)

    def register_handoff(self, tool_name: str, spec: HandoffSpec) -> None:
        self.catalog.register_handoff(tool_name, spec)

    def register_tool_provider(self, provider: ToolProvider) -> ToolProviderSpec:
        spec = verify_tool_provider(provider)
        tools_before = self.registry.snapshot()
        handlers_before = self.executor.snapshot_handlers()
        try:
            provider.install(registry=self.registry, executor=self.executor)
            installed = {
                tool.name for tool in self.registry.list_tools()
            } - set(tools_before)
            missing = sorted(set(spec.tool_names) - installed)
            if missing:
                raise ValueError(
                    f"tool provider {spec.provider_id} did not install declared tools: "
                    + ", ".join(missing)
                )
            self.catalog.register_tool_provider(provider)
            self.catalog.register_resource(f"tool-provider:{spec.provider_id}", provider)
            return spec
        except Exception:
            self.executor.restore_handlers(handlers_before)
            self.registry.restore(tools_before)
            try:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            raise

    def register_subagent(self, spec: SubAgentSpec) -> None:
        self.catalog.register_subagent(spec)

    def register_hook(self, spec: HookSpec) -> None:
        self.catalog.register_hook(spec)

    def register_context_contributor(
        self,
        contributor_id: str,
        contributor: ContextContributor,
    ) -> None:
        self.catalog.register_context_contributor(contributor_id, contributor)

    def register_agent_entrypoint(self, spec: AgentEntrypointSpec) -> None:
        self.catalog.register_agent_entrypoint(spec)

    def register_resource(self, resource_id: str, resource: object) -> None:
        self.catalog.register_resource(resource_id, resource)


class CapabilityPack(Protocol):
    spec: CapabilityPackSpec

    def install(
        self,
        context: CapabilityInstallContext,
    ) -> CapabilityContribution | None: ...


def capability_pack_spec(pack: object) -> CapabilityPackSpec:
    declared = getattr(pack, "spec", None)
    if not isinstance(declared, CapabilityPackSpec):
        raise TypeError("capability pack must declare a CapabilityPackSpec as spec")
    return declared


def capability_pack_spec_from_manifest(manifest: object) -> CapabilityPackSpec:
    """Derive the executable declaration from the control-plane manifest."""

    from deepkeel.capability_manifest import CapabilityManifest

    if not isinstance(manifest, CapabilityManifest):
        raise TypeError("manifest must be a CapabilityManifest")
    return CapabilityPackSpec(
        package_id=manifest.id,
        contract_version=manifest.core_contract,
        package_version=manifest.version,
        declared_tools=manifest.tools,
        declared_skills=manifest.skills,
        declared_artifact_types=manifest.artifact_types,
        declared_handoffs=manifest.handoffs,
        declared_tool_providers=manifest.mcp_servers,
        declared_subagents=manifest.subagents,
        declared_hooks=manifest.hooks,
        declared_context_contributors=manifest.context_contributors,
        declared_agent_entrypoints=manifest.agent_entrypoints,
        declared_resources=manifest.resources,
        required_scopes=manifest.permissions,
        metadata={
            **dict(manifest.metadata),
            "capability_manifest": {
                "schema_version": manifest.schema_version,
                "entrypoint": manifest.entrypoint,
                "dependencies": dict(manifest.dependencies),
                "budget": manifest.budget.limits(),
                "tool_permissions": {
                    name: list(scopes)
                    for name, scopes in manifest.tool_permissions.items()
                },
                "memory_namespaces": list(manifest.memory_namespaces),
                "agent_entrypoints": list(manifest.agent_entrypoints),
                "ui_surfaces": list(manifest.ui_surfaces),
                "state_schema_version": manifest.state_schema_version,
                "resume_compatible_versions": list(
                    manifest.resume_compatible_versions
                ),
                "state_migrations": dict(manifest.state_migrations),
            },
        },
    )


def assert_capability_contribution(
    spec: CapabilityPackSpec,
    contribution: CapabilityContribution,
    *,
    registry: ToolRegistry,
    executor: ToolExecutor,
    catalog: CapabilityCatalog,
) -> None:
    """Fail composition when declarations and installed runtime state diverge."""

    declared = {
        "tools": set(spec.declared_tools),
        "skills": set(spec.declared_skills),
        "artifact_types": set(spec.declared_artifact_types),
        "handoffs": set(spec.declared_handoffs),
        "tool_providers": set(spec.declared_tool_providers),
        "subagents": set(spec.declared_subagents),
        "hooks": set(spec.declared_hooks),
        "context_contributors": set(spec.declared_context_contributors),
        "agent_entrypoints": set(spec.declared_agent_entrypoints),
        "resources": set(spec.declared_resources),
    }
    installed = {
        field_name: set(getattr(contribution, field_name))
        for field_name in _CONTRIBUTION_FIELDS
    }
    issues: list[str] = []
    for field_name, expected in declared.items():
        actual = installed[field_name]
        missing = sorted(expected - actual)
        undeclared = sorted(actual - expected)
        if missing:
            issues.append(f"missing {field_name}: {', '.join(missing)}")
        if undeclared:
            issues.append(f"undeclared {field_name}: {', '.join(undeclared)}")
    available_tools = {tool.name for tool in registry.list_tools()}
    missing_handlers = sorted(declared["tools"] - set(executor.registered_tools))
    missing_specs = sorted(declared["tools"] - available_tools)
    if missing_specs:
        issues.append(f"missing tool specs: {', '.join(missing_specs)}")
    if missing_handlers:
        issues.append(f"missing tool handlers: {', '.join(missing_handlers)}")
    artifact_types = set(catalog.artifact_types)
    for tool_name, handoff in catalog.handoffs.items():
        if tool_name not in available_tools:
            issues.append(f"handoff references unknown tool: {tool_name}")
        if handoff.completion_artifact_type and handoff.completion_artifact_type not in artifact_types:
            issues.append(
                f"handoff {tool_name} references unknown artifact type: "
                f"{handoff.completion_artifact_type}"
            )
    for subagent in catalog.subagents.values():
        unknown = sorted(set(subagent.tool_allowlist) - available_tools)
        if unknown:
            issues.append(
                f"subagent {subagent.id} references unknown tools: {', '.join(unknown)}"
            )
    missing_hooks = sorted(declared["hooks"] - set(catalog.hooks))
    if missing_hooks:
        issues.append(f"missing hooks: {', '.join(missing_hooks)}")
    if issues:
        raise ValueError(
            f"capability pack {spec.package_id} failed conformance: " + "; ".join(issues)
        )


_DECLARATION_FIELDS = (
    "declared_tools",
    "declared_skills",
    "declared_artifact_types",
    "declared_handoffs",
    "declared_tool_providers",
    "declared_subagents",
    "declared_hooks",
    "declared_context_contributors",
    "declared_agent_entrypoints",
    "declared_resources",
    "required_scopes",
)
_CONTRIBUTION_FIELDS = (
    "tools",
    "skills",
    "artifact_types",
    "handoffs",
    "tool_providers",
    "subagents",
    "hooks",
    "context_contributors",
    "agent_entrypoints",
    "resources",
)
_CATALOG_FIELDS = (
    "skills",
    "artifact_types",
    "handoffs",
    "tool_providers",
    "subagents",
    "hooks",
    "context_contributors",
    "agent_entrypoints",
    "resources",
)


def _close_resource(resource: object) -> None:
    try:
        close = getattr(resource, "close", None)
        if callable(close):
            close()
    except Exception:
        pass
