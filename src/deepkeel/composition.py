from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal, Mapping, Self, TypedDict, Unpack, cast

from deepkeel.async_ports import (
    AsyncDurableCheckpointStore,
    AsyncRunLeaseStore,
    AsyncRuntimeEventJournal,
    AsyncRuntimeStateStore,
    AsyncToolExecutionStore,
)
from deepkeel.budget import BudgetLedger
from deepkeel.capabilities import (
    CapabilityCatalog,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPack,
    CapabilityPackSpec,
    capability_pack_spec,
    assert_capability_contribution,
)
from deepkeel.capability_manifest import (
    CapabilityManifest,
    RuntimeGeneration,
    RuntimeGenerationManager,
)
from deepkeel.control import RunControl
from deepkeel.event_journal import RuntimeEventJournal
from deepkeel.context_window import ContextWindowManager
from deepkeel.governance import GovernanceBundle, SecretProvider
from deepkeel.graph import GraphDurability
from deepkeel.hooks import HookRunner
from deepkeel.model_routing import ModelRouter
from deepkeel.model import ModelInvocationRecorder, ModelInvocationStore
from deepkeel.model_health import ModelHealthStore
from deepkeel.leases import RunLeaseStore
from deepkeel.migrations import StateMigrationRegistry
from deepkeel.memory_recall import MemoryRecallCoordinator
from deepkeel.persistence import DurableCheckpointStore
from deepkeel.ports import ContextBuilder, GraphCheckpointer, SessionFactory
from deepkeel.policy import PolicyEngine
from deepkeel.production import ProductionReadinessReport, assess_production_readiness
from deepkeel.references import ReferenceProjector
from deepkeel.runtime import HarnessRuntime, SystemPromptFactory
from deepkeel.state_store import RuntimeStateStore
from deepkeel.telemetry import TelemetryPort
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tool_disclosure import ToolDiscoveryPort
from deepkeel.tools import ToolExecutionStore, ToolExecutor, ToolHandler, ToolPreflight
from deepkeel.skill_activation import EntryToolSkillActivator
from deepkeel.turn_context import ToolViewMode


RuntimeProfileName = Literal["development", "testing", "production"]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Opinionated runtime defaults for a deployment environment."""

    name: RuntimeProfileName
    tool_view_mode: ToolViewMode
    graph_durability: GraphDurability = "exit"
    strict_capability_conformance: bool = True
    require_production_readiness: bool = False

    @classmethod
    def resolve(cls, value: RuntimeProfile | RuntimeProfileName) -> RuntimeProfile:
        if isinstance(value, RuntimeProfile):
            return value
        profiles = {
            "development": cls(name="development", tool_view_mode="legacy"),
            "testing": cls(name="testing", tool_view_mode="enforced"),
            "production": cls(
                name="production",
                tool_view_mode="enforced",
                require_production_readiness=True,
            ),
        }
        try:
            return profiles[value]
        except KeyError as exc:
            raise ValueError(f"unsupported runtime profile: {value!r}") from exc


class RuntimePortChanges(TypedDict, total=False):
    checkpointer: GraphCheckpointer | None
    checkpoint_store: DurableCheckpointStore | None
    async_checkpoint_store: AsyncDurableCheckpointStore | None
    system_prompt_factory: SystemPromptFactory | None
    session_factory: SessionFactory | None
    model_router: ModelRouter | None
    model_invocation_recorder: ModelInvocationRecorder | None
    model_invocation_store: ModelInvocationStore | None
    model_health_store: ModelHealthStore | None
    policy_engine: PolicyEngine | None
    budget_ledger: BudgetLedger | None
    run_control: RunControl | None
    tool_execution_store: ToolExecutionStore | None
    async_tool_execution_store: AsyncToolExecutionStore | None
    tool_preflight: ToolPreflight | None
    secret_provider: SecretProvider | None
    telemetry: TelemetryPort | None
    context_builder: ContextBuilder | None
    memory_recall_coordinator: MemoryRecallCoordinator | None
    context_window_manager: ContextWindowManager | None
    runtime_state_store: RuntimeStateStore | None
    async_runtime_state_store: AsyncRuntimeStateStore | None
    event_journal: RuntimeEventJournal | None
    async_event_journal: AsyncRuntimeEventJournal | None
    reference_projector: ReferenceProjector | None
    run_lease_store: RunLeaseStore | None
    async_run_lease_store: AsyncRunLeaseStore | None
    run_lease_owner_id: str
    run_lease_ttl_seconds: float
    state_migrations: StateMigrationRegistry | None
    async_stream_buffer_size: int
    async_cancel_timeout_seconds: float
    reuse_compiled_graph: bool
    graph_durability: GraphDurability
    tool_view_mode: ToolViewMode
    hook_runner: HookRunner | None
    tool_discovery_port: ToolDiscoveryPort | None
    entry_tool_skill_activator: EntryToolSkillActivator | None
    capability_services: Mapping[str, object]


class GovernedRuntimePortChanges(TypedDict, total=False):
    checkpointer: GraphCheckpointer | None
    checkpoint_store: DurableCheckpointStore | None
    async_checkpoint_store: AsyncDurableCheckpointStore | None
    system_prompt_factory: SystemPromptFactory | None
    session_factory: SessionFactory | None
    model_router: ModelRouter | None
    model_invocation_recorder: ModelInvocationRecorder | None
    model_invocation_store: ModelInvocationStore | None
    model_health_store: ModelHealthStore | None
    run_control: RunControl | None
    tool_execution_store: ToolExecutionStore | None
    async_tool_execution_store: AsyncToolExecutionStore | None
    tool_preflight: ToolPreflight | None
    telemetry: TelemetryPort | None
    context_builder: ContextBuilder | None
    memory_recall_coordinator: MemoryRecallCoordinator | None
    context_window_manager: ContextWindowManager | None
    runtime_state_store: RuntimeStateStore | None
    async_runtime_state_store: AsyncRuntimeStateStore | None
    event_journal: RuntimeEventJournal | None
    async_event_journal: AsyncRuntimeEventJournal | None
    reference_projector: ReferenceProjector | None
    run_lease_store: RunLeaseStore | None
    async_run_lease_store: AsyncRunLeaseStore | None
    run_lease_owner_id: str
    run_lease_ttl_seconds: float
    state_migrations: StateMigrationRegistry | None
    async_stream_buffer_size: int
    async_cancel_timeout_seconds: float
    reuse_compiled_graph: bool
    graph_durability: GraphDurability
    tool_view_mode: ToolViewMode
    hook_runner: HookRunner | None
    tool_discovery_port: ToolDiscoveryPort | None
    entry_tool_skill_activator: EntryToolSkillActivator | None
    capability_services: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RuntimePersistencePorts:
    """Durability and idempotency ports for a production runtime."""

    checkpointer: GraphCheckpointer | None = None
    checkpoint_store: DurableCheckpointStore | None = None
    async_checkpoint_store: AsyncDurableCheckpointStore | None = None
    runtime_state_store: RuntimeStateStore | None = None
    async_runtime_state_store: AsyncRuntimeStateStore | None = None
    event_journal: RuntimeEventJournal | None = None
    async_event_journal: AsyncRuntimeEventJournal | None = None
    run_lease_store: RunLeaseStore | None = None
    async_run_lease_store: AsyncRunLeaseStore | None = None
    model_invocation_store: ModelInvocationStore | None = None
    tool_execution_store: ToolExecutionStore | None = None
    async_tool_execution_store: AsyncToolExecutionStore | None = None
    run_lease_owner_id: str = ""
    run_lease_ttl_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class RuntimeGovernancePorts:
    """Policy, budget, health and control ports evaluated during execution."""

    model_router: ModelRouter | None = None
    model_health_store: ModelHealthStore | None = None
    policy_engine: PolicyEngine | None = None
    budget_ledger: BudgetLedger | None = None
    run_control: RunControl | None = None
    tool_preflight: ToolPreflight | None = None
    secret_provider: SecretProvider | None = None


@dataclass(frozen=True, slots=True)
class RuntimeObservabilityPorts:
    """Diagnostics, invocation recording and result projection ports."""

    model_invocation_recorder: ModelInvocationRecorder | None = None
    telemetry: TelemetryPort | None = None
    reference_projector: ReferenceProjector | None = None


@dataclass(frozen=True, slots=True)
class RuntimeExecutionPorts:
    """Execution-time customization that does not own durable state."""

    system_prompt_factory: SystemPromptFactory | None = None
    session_factory: SessionFactory | None = None
    context_builder: ContextBuilder | None = None
    memory_recall_coordinator: MemoryRecallCoordinator | None = None
    context_window_manager: ContextWindowManager | None = None
    state_migrations: StateMigrationRegistry | None = None
    async_stream_buffer_size: int = 128
    async_cancel_timeout_seconds: float = 5.0
    reuse_compiled_graph: bool = True
    graph_durability: GraphDurability = "exit"
    tool_view_mode: ToolViewMode = "legacy"
    hook_runner: HookRunner | None = None
    tool_discovery_port: ToolDiscoveryPort | None = None
    entry_tool_skill_activator: EntryToolSkillActivator | None = None
    capability_services: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimePorts:
    """Infrastructure ports used to compose a product-neutral runtime."""

    checkpointer: GraphCheckpointer | None = None
    checkpoint_store: DurableCheckpointStore | None = None
    async_checkpoint_store: AsyncDurableCheckpointStore | None = None
    system_prompt_factory: SystemPromptFactory | None = None
    session_factory: SessionFactory | None = None
    model_router: ModelRouter | None = None
    model_invocation_recorder: ModelInvocationRecorder | None = None
    model_invocation_store: ModelInvocationStore | None = None
    model_health_store: ModelHealthStore | None = None
    policy_engine: PolicyEngine | None = None
    budget_ledger: BudgetLedger | None = None
    run_control: RunControl | None = None
    tool_execution_store: ToolExecutionStore | None = None
    async_tool_execution_store: AsyncToolExecutionStore | None = None
    tool_preflight: ToolPreflight | None = None
    secret_provider: SecretProvider | None = None
    telemetry: TelemetryPort | None = None
    context_builder: ContextBuilder | None = None
    memory_recall_coordinator: MemoryRecallCoordinator | None = None
    context_window_manager: ContextWindowManager | None = None
    runtime_state_store: RuntimeStateStore | None = None
    async_runtime_state_store: AsyncRuntimeStateStore | None = None
    event_journal: RuntimeEventJournal | None = None
    async_event_journal: AsyncRuntimeEventJournal | None = None
    reference_projector: ReferenceProjector | None = None
    run_lease_store: RunLeaseStore | None = None
    async_run_lease_store: AsyncRunLeaseStore | None = None
    run_lease_owner_id: str = ""
    run_lease_ttl_seconds: float = 60.0
    state_migrations: StateMigrationRegistry | None = None
    async_stream_buffer_size: int = 128
    async_cancel_timeout_seconds: float = 5.0
    reuse_compiled_graph: bool = True
    graph_durability: GraphDurability = "exit"
    tool_view_mode: ToolViewMode = "legacy"
    hook_runner: HookRunner | None = None
    tool_discovery_port: ToolDiscoveryPort | None = None
    entry_tool_skill_activator: EntryToolSkillActivator | None = None
    capability_services: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_bundles(
        cls,
        *,
        persistence: RuntimePersistencePorts | None = None,
        governance: RuntimeGovernancePorts | None = None,
        observability: RuntimeObservabilityPorts | None = None,
        execution: RuntimeExecutionPorts | None = None,
        **overrides: Unpack[RuntimePortChanges],
    ) -> "RuntimePorts":
        """Compose grouped ports while retaining explicit flat overrides."""

        _validate_port_changes(overrides)
        values: dict[str, object] = {}
        for bundle in (persistence, governance, observability, execution):
            if bundle is None:
                continue
            values.update(
                {
                    item.name: getattr(bundle, item.name)
                    for item in fields(bundle)
                }
            )
        values.update(overrides)
        # Dataclass field discovery validates the keys above; the cast preserves
        # precise field types at the public constructor boundary for static tools.
        return cls(**cast(Any, values))

    @classmethod
    def governed(
        cls,
        governance: GovernanceBundle,
        **ports: Unpack[GovernedRuntimePortChanges],
    ) -> "RuntimePorts":
        _validate_port_changes(ports)
        return cls(
            policy_engine=governance.policy_engine,
            budget_ledger=governance.budget_ledger,
            secret_provider=governance.secret_provider,
            **ports,
        )


class HarnessRuntimeBuilder:
    """Stable composition API for runtime ports and domain Capability Packs."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        *,
        profile: RuntimeProfile | RuntimeProfileName = "development",
    ) -> None:
        if executor is not None and registry is not None and executor.registry is not registry:
            raise ValueError("executor and registry must reference the same ToolRegistry")
        self._profile = RuntimeProfile.resolve(profile)
        self._registry = registry or (executor.registry if executor is not None else ToolRegistry())
        self._executor = executor
        self._ports = RuntimePorts(
            graph_durability=self._profile.graph_durability,
            tool_view_mode=self._profile.tool_view_mode,
        )
        self._capability_packs: list[CapabilityPack] = []
        self._capability_manifests: dict[str, CapabilityManifest] = {}
        self._runtime_generation: RuntimeGeneration | None = None
        self._capability_catalog = CapabilityCatalog()
        self._installed_contributions: tuple[CapabilityContribution, ...] = ()
        self._max_steps = 12
        self._max_parallel_tools = 4
        self._strict_capability_conformance = self._profile.strict_capability_conformance
        self._built = False

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def profile(self) -> RuntimeProfile:
        return self._profile

    def with_ports(self, ports: RuntimePorts) -> Self:
        self._ensure_mutable()
        self._ports = ports
        return self

    def configure_ports(self, **changes: Unpack[RuntimePortChanges]) -> Self:
        self._ensure_mutable()
        _validate_port_changes(changes)
        self._ports = replace(self._ports, **changes)
        return self

    def with_runtime_generation(self, generation: RuntimeGeneration) -> Self:
        """Pin the worker to a generation selected by the package control plane."""
        self._ensure_mutable()
        self._runtime_generation = generation
        return self

    def add_capability_pack(
        self,
        pack: CapabilityPack,
        *,
        manifest: CapabilityManifest | None = None,
    ) -> Self:
        self._ensure_mutable()
        from deepkeel.version import DEEPKEEL_CONTRACT_VERSION

        spec = capability_pack_spec(pack)
        if spec.contract_version != DEEPKEEL_CONTRACT_VERSION:
            raise ValueError(
                f"capability pack {spec.package_id} requires unsupported contract "
                f"{spec.contract_version or '<missing>'}; expected {DEEPKEEL_CONTRACT_VERSION}"
            )
        if any(
            capability_pack_spec(existing).package_id == spec.package_id
            for existing in self._capability_packs
        ):
            raise ValueError(f"capability pack is already registered: {spec.package_id}")
        resolved_manifest = manifest or _manifest_from_pack(pack, spec)
        _validate_pack_manifest(spec, resolved_manifest)
        self._capability_packs.append(pack)
        self._capability_manifests[spec.package_id] = resolved_manifest
        return self

    @property
    def installed_contributions(self) -> tuple[CapabilityContribution, ...]:
        return self._installed_contributions

    @property
    def capability_catalog(self) -> CapabilityCatalog:
        return self._capability_catalog

    def with_max_steps(self, max_steps: int) -> Self:
        self._ensure_mutable()
        self._max_steps = max(2, int(max_steps))
        return self

    def with_max_parallel_tools(self, max_parallel_tools: int) -> Self:
        self._ensure_mutable()
        self._max_parallel_tools = max(1, int(max_parallel_tools))
        return self

    def with_strict_capability_conformance(self, enabled: bool) -> Self:
        self._ensure_mutable()
        self._strict_capability_conformance = bool(enabled)
        return self

    def production_readiness(self) -> ProductionReadinessReport:
        """Return an executable readiness report without freezing the builder."""

        return assess_production_readiness(self._ports)

    def build_production(self) -> HarnessRuntime:
        """Build only when required multi-worker Host ports are explicit."""

        self.production_readiness().require_ready()
        return self.build()

    def build(self) -> HarnessRuntime:
        self._ensure_mutable()
        if self._profile.require_production_readiness:
            self.production_readiness().require_ready()
        executor = self._executor or ToolExecutor(
            self._registry,
            preflight=self._ports.tool_preflight,
            max_parallel_tools=self._max_parallel_tools,
            execution_store=self._ports.tool_execution_store,
            async_execution_store=self._ports.async_tool_execution_store,
            policy_engine=self._ports.policy_engine,
            budget_ledger=self._ports.budget_ledger,
        )
        if executor.registry is not self._registry:
            raise ValueError("ToolExecutor registry changed during runtime composition")
        contributions = [
            self._install_pack(pack, executor) for pack in self._capability_packs
        ]
        self._installed_contributions = tuple(contributions)
        hook_runner = self._ports.hook_runner or HookRunner()
        for hook in self._capability_catalog.hooks.values():
            if hook.id not in hook_runner.registered_hooks:
                hook_runner.register(hook)
        executor.configure_artifact_schemas(
            {
                name: spec.schema
                for name, spec in self._capability_catalog.artifact_types.items()
            }
        )
        composed_generation = RuntimeGenerationManager().activate(
            tuple(self._capability_manifests.values()),
            catalog_version=self._registry.catalog_version(),
        )
        runtime_generation = self._runtime_generation or composed_generation
        if self._runtime_generation is not None:
            _validate_runtime_generation(
                self._runtime_generation,
                tuple(self._capability_manifests.values()),
            )
        self._built = True
        return HarnessRuntime(
            self._registry,
            executor,
            checkpointer=self._ports.checkpointer,
            checkpoint_store=self._ports.checkpoint_store,
            async_checkpoint_store=self._ports.async_checkpoint_store,
            system_prompt_factory=self._ports.system_prompt_factory,
            session_factory=self._ports.session_factory,
            max_steps=self._max_steps,
            model_router=self._ports.model_router,
            model_invocation_recorder=self._ports.model_invocation_recorder,
            model_invocation_store=self._ports.model_invocation_store,
            model_health_store=self._ports.model_health_store,
            policy_engine=self._ports.policy_engine,
            budget_ledger=self._ports.budget_ledger,
            run_control=self._ports.run_control,
            capability_contributions=self._installed_contributions,
            capability_catalog=self._capability_catalog,
            telemetry=self._ports.telemetry,
            context_builder=self._ports.context_builder,
            memory_recall_coordinator=self._ports.memory_recall_coordinator,
            context_window_manager=self._ports.context_window_manager,
            runtime_state_store=self._ports.runtime_state_store,
            async_runtime_state_store=self._ports.async_runtime_state_store,
            event_journal=self._ports.event_journal,
            async_event_journal=self._ports.async_event_journal,
            reference_projector=self._ports.reference_projector,
            run_lease_store=self._ports.run_lease_store,
            async_run_lease_store=self._ports.async_run_lease_store,
            run_lease_owner_id=self._ports.run_lease_owner_id,
            run_lease_ttl_seconds=self._ports.run_lease_ttl_seconds,
            state_migrations=self._ports.state_migrations,
            async_stream_buffer_size=self._ports.async_stream_buffer_size,
            async_cancel_timeout_seconds=self._ports.async_cancel_timeout_seconds,
            reuse_compiled_graph=self._ports.reuse_compiled_graph,
            graph_durability=self._ports.graph_durability,
            tool_view_mode=self._ports.tool_view_mode,
            hook_runner=hook_runner,
            tool_discovery_port=self._ports.tool_discovery_port,
            entry_tool_skill_activator=self._ports.entry_tool_skill_activator,
            runtime_generation=runtime_generation,
        )

    def _install_pack(
        self,
        pack: CapabilityPack,
        executor: ToolExecutor,
    ) -> CapabilityContribution:
        spec = capability_pack_spec(pack)
        tools_before = self._registry.snapshot()
        handlers_before = executor.snapshot_handlers()
        catalog_before = self._capability_catalog.snapshot()
        contribution: CapabilityContribution | None = None
        try:
            install = getattr(pack, "install", None)
            if not callable(install):
                raise TypeError(
                    f"capability pack {spec.package_id} must implement install(context)"
                )
            contribution = install(
                CapabilityInstallContext(
                    registry=self._registry,
                    executor=executor,
                    catalog=self._capability_catalog,
                    services={
                        **dict(self._ports.capability_services or {}),
                        "secret_provider": self._ports.secret_provider,
                    },
                )
            )
            if contribution is not None and contribution.package_id != spec.package_id:
                raise ValueError(
                    f"capability contribution package_id {contribution.package_id!r} "
                    f"does not match {spec.package_id!r}"
                )
        except Exception:
            self._rollback_install(
                executor,
                tools_before=tools_before,
                handlers_before=handlers_before,
                catalog_before=catalog_before,
            )
            raise

        tools_after = {tool.name for tool in self._registry.list_tools()}
        handlers_after = set(executor.registered_tools)
        installed_tools = tuple(
            sorted((tools_after - set(tools_before)) | (handlers_after - set(handlers_before)))
        )
        self._apply_manifest_tool_permissions(
            spec.package_id,
            installed_tools or spec.declared_tools,
        )
        catalog_after = self._capability_catalog.snapshot()
        metadata = contribution.metadata if contribution is not None else {}
        resolved = CapabilityContribution(
            package_id=spec.package_id,
            tools=installed_tools or spec.declared_tools,
            skills=_catalog_delta(catalog_before, catalog_after, "skills"),
            artifact_types=_catalog_delta(
                catalog_before, catalog_after, "artifact_types"
            ),
            handoffs=_catalog_delta(catalog_before, catalog_after, "handoffs"),
            tool_providers=_catalog_delta(
                catalog_before, catalog_after, "tool_providers"
            ),
            subagents=_catalog_delta(catalog_before, catalog_after, "subagents"),
            hooks=_catalog_delta(catalog_before, catalog_after, "hooks"),
            context_contributors=_catalog_delta(
                catalog_before, catalog_after, "context_contributors"
            ),
            resources=_catalog_delta(catalog_before, catalog_after, "resources"),
            metadata=metadata,
        )
        try:
            if self._strict_capability_conformance:
                assert_capability_contribution(
                    spec,
                    resolved,
                    registry=self._registry,
                    executor=executor,
                    catalog=self._capability_catalog,
                )
        except Exception:
            self._rollback_install(
                executor,
                tools_before=tools_before,
                handlers_before=handlers_before,
                catalog_before=catalog_before,
            )
            raise
        return resolved

    def _apply_manifest_tool_permissions(
        self,
        package_id: str,
        tool_names: tuple[str, ...],
    ) -> None:
        manifest = self._capability_manifests[package_id]
        for tool_name in tool_names:
            scopes = manifest.tool_permissions.get(tool_name, ())
            if not scopes:
                continue
            tool = self._registry.get(tool_name)
            runtime_policy = dict(tool.runtime_policy)
            existing = tuple(
                str(scope).strip()
                for scope in runtime_policy.get("required_scopes") or ()
                if str(scope).strip()
            )
            runtime_policy["required_scopes"] = list(
                dict.fromkeys((*existing, *scopes))
            )
            self._registry.register(
                tool.model_copy(update={"runtime_policy": runtime_policy}),
                replace=True,
            )

    def _rollback_install(
        self,
        executor: ToolExecutor,
        *,
        tools_before: dict[str, ToolSpec],
        handlers_before: dict[str, ToolHandler],
        catalog_before: dict[str, dict[str, object]],
    ) -> None:
        executor.restore_handlers(handlers_before)
        self._registry.restore(tools_before)
        self._capability_catalog.rollback(catalog_before)

    def _ensure_mutable(self) -> None:
        if self._built:
            raise RuntimeError("HarnessRuntimeBuilder cannot be reused after build")


def _validate_port_changes(changes: Mapping[str, object]) -> None:
    supported = {field.name for field in fields(RuntimePorts)}
    unknown = sorted(set(changes) - supported)
    if unknown:
        raise TypeError(f"unknown runtime ports: {', '.join(unknown)}")


def _catalog_delta(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
    field_name: str,
) -> tuple[str, ...]:
    return tuple(sorted(set(after.get(field_name, ())) - set(before.get(field_name, ()))))


def _manifest_from_pack(
    pack: CapabilityPack,
    spec: CapabilityPackSpec,
) -> CapabilityManifest:
    return CapabilityManifest(
        id=spec.package_id,
        version=spec.package_version,
        core_contract=spec.contract_version,
        core_version="*",
        entrypoint=f"{type(pack).__module__}:{type(pack).__qualname__}",
        skills=spec.declared_skills,
        tools=spec.declared_tools,
        artifact_types=spec.declared_artifact_types,
        subagents=spec.declared_subagents,
        handoffs=spec.declared_handoffs,
        hooks=spec.declared_hooks,
        context_contributors=spec.declared_context_contributors,
        mcp_servers=spec.declared_tool_providers,
        resources=spec.declared_resources,
        permissions=spec.required_scopes,
        tool_permissions={
            tool_name: spec.required_scopes
            for tool_name in spec.declared_tools
            if spec.required_scopes
        },
        metadata=dict(spec.metadata),
    )


def _validate_pack_manifest(
    spec: CapabilityPackSpec,
    manifest: CapabilityManifest,
) -> None:
    if manifest.id != spec.package_id:
        raise ValueError(
            f"capability manifest id {manifest.id!r} does not match "
            f"package id {spec.package_id!r}"
        )
    comparisons = {
        "tools": (set(spec.declared_tools), set(manifest.tools)),
        "skills": (set(spec.declared_skills), set(manifest.skills)),
        "artifact_types": (
            set(spec.declared_artifact_types),
            set(manifest.artifact_types),
        ),
        "subagents": (set(spec.declared_subagents), set(manifest.subagents)),
        "handoffs": (set(spec.declared_handoffs), set(manifest.handoffs)),
        "hooks": (set(spec.declared_hooks), set(manifest.hooks)),
        "context_contributors": (
            set(spec.declared_context_contributors),
            set(manifest.context_contributors),
        ),
        "resources": (set(spec.declared_resources), set(manifest.resources)),
    }
    mismatches = [
        name
        for name, (declared, manifested) in comparisons.items()
        if declared != manifested
    ]
    if mismatches:
        raise ValueError(
            f"capability manifest {manifest.id} diverges from pack declaration: "
            + ", ".join(mismatches)
        )


def _validate_runtime_generation(
    generation: RuntimeGeneration,
    manifests: tuple[CapabilityManifest, ...],
) -> None:
    expected = {manifest.id: manifest for manifest in generation.packages}
    installed = {manifest.id: manifest for manifest in manifests}
    missing = sorted(set(expected) - set(installed))
    unexpected = sorted(set(installed) - set(expected))
    changed = sorted(
        package_id
        for package_id in set(expected) & set(installed)
        if expected[package_id] != installed[package_id]
    )
    issues: list[str] = []
    if missing:
        issues.append(f"missing packages: {', '.join(missing)}")
    if unexpected:
        issues.append(f"unexpected packages: {', '.join(unexpected)}")
    if changed:
        issues.append(f"changed manifests: {', '.join(changed)}")
    if issues:
        raise ValueError(
            f"runtime generation {generation.generation_id} does not match "
            f"installed Capability Packs ({'; '.join(issues)})"
        )
