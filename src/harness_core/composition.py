from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, Self

from harness_core.budget import BudgetLedger
from harness_core.control import RunControl
from harness_core.governance import GovernanceBundle, SecretProvider
from harness_core.model_routing import ModelRouter
from harness_core.persistence import CheckpointStore
from harness_core.policy import PolicyEngine
from harness_core.runtime import HarnessRuntime, SystemPromptFactory
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutionStore, ToolExecutor, ToolPreflight


class CapabilityPack(Protocol):
    """A domain package that contributes handlers to one Harness runtime."""

    package_id: str
    contract_version: str

    def register(self, executor: ToolExecutor) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimePorts:
    """Infrastructure ports used to compose a product-neutral runtime."""

    checkpointer: Any = None
    checkpoint_store: CheckpointStore | None = None
    system_prompt_factory: SystemPromptFactory | None = None
    session_factory: Callable[[], Any] | None = None
    model_router: ModelRouter | None = None
    policy_engine: PolicyEngine | None = None
    budget_ledger: BudgetLedger | None = None
    run_control: RunControl | None = None
    tool_execution_store: ToolExecutionStore | None = None
    tool_preflight: ToolPreflight | None = None
    secret_provider: SecretProvider | None = None

    @classmethod
    def governed(cls, governance: GovernanceBundle, **ports: Any) -> "RuntimePorts":
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
        self._max_steps = 12
        self._max_parallel_tools = 4
        self._built = False

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def with_ports(self, ports: RuntimePorts) -> Self:
        self._ensure_mutable()
        self._ports = ports
        return self

    def configure_ports(self, **changes: Any) -> Self:
        self._ensure_mutable()
        self._ports = replace(self._ports, **changes)
        return self

    def add_capability_pack(self, pack: CapabilityPack) -> Self:
        self._ensure_mutable()
        from harness_core.version import HARNESS_CORE_CONTRACT_VERSION

        package_id = str(getattr(pack, "package_id", "") or "").strip()
        if not package_id:
            raise ValueError("capability pack must declare package_id")
        contract_version = str(getattr(pack, "contract_version", "") or "").strip()
        if contract_version != HARNESS_CORE_CONTRACT_VERSION:
            raise ValueError(
                f"capability pack {package_id} requires unsupported contract "
                f"{contract_version or '<missing>'}; expected {HARNESS_CORE_CONTRACT_VERSION}"
            )
        if any(existing.package_id == package_id for existing in self._capability_packs):
            raise ValueError(f"capability pack is already registered: {package_id}")
        self._capability_packs.append(pack)
        return self

    def with_max_steps(self, max_steps: int) -> Self:
        self._ensure_mutable()
        self._max_steps = max(2, int(max_steps))
        return self

    def with_max_parallel_tools(self, max_parallel_tools: int) -> Self:
        self._ensure_mutable()
        self._max_parallel_tools = max(1, int(max_parallel_tools))
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
        for pack in self._capability_packs:
            pack.register(executor)
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
            policy_engine=self._ports.policy_engine,
            budget_ledger=self._ports.budget_ledger,
            run_control=self._ports.run_control,
        )

    def _ensure_mutable(self) -> None:
        if self._built:
            raise RuntimeError("HarnessRuntimeBuilder cannot be reused after build")
