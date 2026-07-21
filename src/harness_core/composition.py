from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Mapping, Self, TypedDict, Unpack

from harness_core.budget import BudgetLedger
from harness_core.capabilities import (
    CapabilityCatalog,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPack,
    CapabilityPackSpec,
    capability_pack_spec,
    assert_capability_contribution,
)
from harness_core.control import RunControl
from harness_core.event_journal import RuntimeEventJournal
from harness_core.context_window import ContextWindowManager
from harness_core.governance import GovernanceBundle, SecretProvider
from harness_core.model_routing import ModelRouter
from harness_core.model import ModelInvocationRecorder, ModelInvocationStore
from harness_core.leases import RunLeaseStore
from harness_core.migrations import StateMigrationRegistry
from harness_core.persistence import DurableCheckpointStore
from harness_core.ports import ContextBuilder, GraphCheckpointer, SessionFactory
from harness_core.policy import PolicyEngine
from harness_core.references import ReferenceProjector
from harness_core.runtime import HarnessRuntime, SystemPromptFactory
from harness_core.state_store import RuntimeStateStore
from harness_core.telemetry import TelemetryPort
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tools import ToolExecutionStore, ToolExecutor, ToolHandler, ToolPreflight
from harness_core.turn_context import ToolViewMode


class RuntimePortChanges(TypedDict, total=False):
    checkpointer: GraphCheckpointer | None
    checkpoint_store: DurableCheckpointStore | None
    system_prompt_factory: SystemPromptFactory | None
    session_factory: SessionFactory | None
    model_router: ModelRouter | None
    model_invocation_recorder: ModelInvocationRecorder | None
    model_invocation_store: ModelInvocationStore | None
    policy_engine: PolicyEngine | None
    budget_ledger: BudgetLedger | None
    run_control: RunControl | None
    tool_execution_store: ToolExecutionStore | None
    tool_preflight: ToolPreflight | None
    secret_provider: SecretProvider | None
    telemetry: TelemetryPort | None
    context_builder: ContextBuilder | None
    context_window_manager: ContextWindowManager | None
    runtime_state_store: RuntimeStateStore | None
    event_journal: RuntimeEventJournal | None
    reference_projector: ReferenceProjector | None
    run_lease_store: RunLeaseStore | None
    run_lease_owner_id: str
    run_lease_ttl_seconds: float
    state_migrations: StateMigrationRegistry | None
    async_stream_buffer_size: int
    async_cancel_timeout_seconds: float
    reuse_compiled_graph: bool
    tool_view_mode: ToolViewMode
    capability_services: Mapping[str, object]


class GovernedRuntimePortChanges(TypedDict, total=False):
    checkpointer: GraphCheckpointer | None
    checkpoint_store: DurableCheckpointStore | None
    system_prompt_factory: SystemPromptFactory | None
    session_factory: SessionFactory | None
    model_router: ModelRouter | None
    model_invocation_recorder: ModelInvocationRecorder | None
    model_invocation_store: ModelInvocationStore | None
    run_control: RunControl | None
    tool_execution_store: ToolExecutionStore | None
    tool_preflight: ToolPreflight | None
    telemetry: TelemetryPort | None
    context_builder: ContextBuilder | None
    context_window_manager: ContextWindowManager | None
    runtime_state_store: RuntimeStateStore | None
    event_journal: RuntimeEventJournal | None
    reference_projector: ReferenceProjector | None
    run_lease_store: RunLeaseStore | None
    run_lease_owner_id: str
    run_lease_ttl_seconds: float
    state_migrations: StateMigrationRegistry | None
    async_stream_buffer_size: int
    async_cancel_timeout_seconds: float
    reuse_compiled_graph: bool
    tool_view_mode: ToolViewMode
    capability_services: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RuntimePorts:
    """Infrastructure ports used to compose a product-neutral runtime."""

    checkpointer: GraphCheckpointer | None = None
    checkpoint_store: DurableCheckpointStore | None = None
    system_prompt_factory: SystemPromptFactory | None = None
    session_factory: SessionFactory | None = None
    model_router: ModelRouter | None = None
    model_invocation_recorder: ModelInvocationRecorder | None = None
    model_invocation_store: ModelInvocationStore | None = None
    policy_engine: PolicyEngine | None = None
    budget_ledger: BudgetLedger | None = None
    run_control: RunControl | None = None
    tool_execution_store: ToolExecutionStore | None = None
    tool_preflight: ToolPreflight | None = None
    secret_provider: SecretProvider | None = None
    telemetry: TelemetryPort | None = None
    context_builder: ContextBuilder | None = None
    context_window_manager: ContextWindowManager | None = None
    runtime_state_store: RuntimeStateStore | None = None
    event_journal: RuntimeEventJournal | None = None
    reference_projector: ReferenceProjector | None = None
    run_lease_store: RunLeaseStore | None = None
    run_lease_owner_id: str = ""
    run_lease_ttl_seconds: float = 60.0
    state_migrations: StateMigrationRegistry | None = None
    async_stream_buffer_size: int = 128
    async_cancel_timeout_seconds: float = 5.0
    reuse_compiled_graph: bool = True
    tool_view_mode: ToolViewMode = "legacy"
    capability_services: Mapping[str, object] = field(default_factory=dict)

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
    ) -> None:
        if executor is not None and registry is not None and executor.registry is not registry:
            raise ValueError("executor and registry must reference the same ToolRegistry")
        self._registry = registry or (executor.registry if executor is not None else ToolRegistry())
        self._executor = executor
        self._ports = RuntimePorts()
        self._capability_packs: list[CapabilityPack] = []
        self._capability_catalog = CapabilityCatalog()
        self._installed_contributions: tuple[CapabilityContribution, ...] = ()
        self._max_steps = 12
        self._max_parallel_tools = 4
        self._strict_capability_conformance = True
        self._built = False

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def with_ports(self, ports: RuntimePorts) -> Self:
        self._ensure_mutable()
        self._ports = ports
        return self

    def configure_ports(self, **changes: Unpack[RuntimePortChanges]) -> Self:
        self._ensure_mutable()
        _validate_port_changes(changes)
        self._ports = replace(self._ports, **changes)
        return self

    def add_capability_pack(self, pack: CapabilityPack) -> Self:
        self._ensure_mutable()
        from harness_core.version import HARNESS_CORE_CONTRACT_VERSION

        spec = capability_pack_spec(pack)
        if spec.contract_version != HARNESS_CORE_CONTRACT_VERSION:
            raise ValueError(
                f"capability pack {spec.package_id} requires unsupported contract "
                f"{spec.contract_version or '<missing>'}; expected {HARNESS_CORE_CONTRACT_VERSION}"
            )
        if any(
            capability_pack_spec(existing).package_id == spec.package_id
            for existing in self._capability_packs
        ):
            raise ValueError(f"capability pack is already registered: {spec.package_id}")
        self._capability_packs.append(pack)
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

    def build(self) -> HarnessRuntime:
        self._ensure_mutable()
        executor = self._executor or ToolExecutor(
            self._registry,
            preflight=self._ports.tool_preflight,
            max_parallel_tools=self._max_parallel_tools,
            execution_store=self._ports.tool_execution_store,
            policy_engine=self._ports.policy_engine,
            budget_ledger=self._ports.budget_ledger,
        )
        if executor.registry is not self._registry:
            raise ValueError("ToolExecutor registry changed during runtime composition")
        contributions = [
            self._install_pack(pack, executor) for pack in self._capability_packs
        ]
        self._installed_contributions = tuple(contributions)
        executor.configure_artifact_schemas(
            {
                name: spec.schema
                for name, spec in self._capability_catalog.artifact_types.items()
            }
        )
        self._built = True
        return HarnessRuntime(
            self._registry,
            executor,
            checkpointer=self._ports.checkpointer,
            checkpoint_store=self._ports.checkpoint_store,
            system_prompt_factory=self._ports.system_prompt_factory,
            session_factory=self._ports.session_factory,
            max_steps=self._max_steps,
            model_router=self._ports.model_router,
            model_invocation_recorder=self._ports.model_invocation_recorder,
            model_invocation_store=self._ports.model_invocation_store,
            policy_engine=self._ports.policy_engine,
            budget_ledger=self._ports.budget_ledger,
            run_control=self._ports.run_control,
            capability_contributions=self._installed_contributions,
            capability_catalog=self._capability_catalog,
            telemetry=self._ports.telemetry,
            context_builder=self._ports.context_builder,
            context_window_manager=self._ports.context_window_manager,
            runtime_state_store=self._ports.runtime_state_store,
            event_journal=self._ports.event_journal,
            reference_projector=self._ports.reference_projector,
            run_lease_store=self._ports.run_lease_store,
            run_lease_owner_id=self._ports.run_lease_owner_id,
            run_lease_ttl_seconds=self._ports.run_lease_ttl_seconds,
            state_migrations=self._ports.state_migrations,
            async_stream_buffer_size=self._ports.async_stream_buffer_size,
            async_cancel_timeout_seconds=self._ports.async_cancel_timeout_seconds,
            reuse_compiled_graph=self._ports.reuse_compiled_graph,
            tool_view_mode=self._ports.tool_view_mode,
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
