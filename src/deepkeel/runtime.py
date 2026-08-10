from __future__ import annotations
import asyncio
import time
from collections.abc import AsyncIterator
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from deepkeel.budget import (
    INPUT_TOKENS,
    MODEL_CALLS,
    MODEL_RETRIES,
    OUTPUT_TOKENS,
    TOOL_CALLS,
    TOOL_CONCURRENCY,
    BudgetLedger,
    InMemoryBudgetLedger,
)
from deepkeel.async_ports import (
    AsyncDurableCheckpointStore,
    AsyncRunLeaseStoreAdapter,
    AsyncRunLeaseStore,
    AsyncRuntimeEventJournal,
    AsyncRuntimeStateStore,
    run_sync_adapter,
)
from deepkeel.contracts import (
    Artifact,
    FinalAnswer,
    Observation,
    PendingAction,
    RunContext,
    RunStatus,
    ToolCall,
    ToolResult,
)
from deepkeel.context import build_context_snapshot
from deepkeel.context_window import (
    ContextWindowManager,
    DeterministicContextWindowManager,
)
from deepkeel.control import NoopRunControl, RunControl
from deepkeel.event_journal import RuntimeEventJournal
from deepkeel.events import AgentEventPersistenceError, envelope_runtime_event
from deepkeel.failures import RuntimeFailure, classify_runtime_failure
from deepkeel.graph import (
    GraphDurability,
    HarnessGraph,
    create_harness_graph,
)
from deepkeel.hooks import HookAudit, HookRunner
from deepkeel.langgraph_adapter import (
    LangGraphCheckpointerAdapter,
    checkpointer_supports_async,
    compiler_checkpointer,
)
from deepkeel.model import (
    ModelInvocationRecorder,
    ModelInvocationStore,
    ModelProviderAdapter,
)
from deepkeel.model_health import InMemoryModelHealthStore, ModelHealthStore
from deepkeel.model_routing import AdaptiveStepModelRouter, ModelRouter
from deepkeel.leases import (
    AsyncRunLeaseGuard,
    ExecutionFence,
    RunLeaseGuard,
    RunLeaseStore,
)
from deepkeel.migrations import StateMigrationRegistry, default_state_migrations
from deepkeel.memory_recall import MemoryRecallCoordinator
from deepkeel.persistence import (
    CheckpointCompatibilityError,
    DurableCheckpointStore,
)
from deepkeel.planning.tool import execution_planning_prompt, install_execution_planning
from deepkeel.capabilities import CapabilityCatalog, CapabilityContribution
from deepkeel.capability_manifest import RuntimeGeneration
from deepkeel.ports import ContextBuilder, GraphCheckpointer, SessionFactory
from deepkeel.prompts import harness_system_prompt
from deepkeel.policy import DefaultPolicyEngine, PolicyEngine
from deepkeel.references import (
    DefaultReferenceProjector,
    ReferenceProjector,
)
from deepkeel.runtime_api import (
    RuntimeRequest,
    RuntimeResult,
    RuntimeResultStatus,
    RuntimeStreamEvent,
)
from deepkeel.runtime_events import RuntimeEventEmitter
from deepkeel.runtime_graph_execution import execute_graph_turn
from deepkeel.runtime_lifecycle import run_start_lifecycle_hooks
from deepkeel.runtime_settlement import project_and_settle_runtime_result
from deepkeel.runtime_persistence import (
    acleanup_run as cleanup_runtime_async,
    aload_authoritative_checkpoint as load_authoritative_checkpoint_async,
    event_latest_sequence,
    host_owns_terminal_settlement,
    persist_runtime_snapshot,
    record_checkpoint_cleanup_event,
)
from deepkeel.runtime_async_stream import stream_runtime_async
from deepkeel.skills import SkillPolicy
from deepkeel.skill_activation import EntryToolSkillActivator
from deepkeel.scope import (
    RuntimeScope,
    require_legacy_compatible_scope,
    scoped_adapter_operation,
)
from deepkeel.state_store import (
    RuntimeStateStore,
    ScopedRuntimeStateStore,
)
from deepkeel.telemetry import NoopTelemetry, TelemetryPort, TelemetryRecord
from deepkeel.type_narrowing import as_dict
from deepkeel.tool_registry import ToolRegistry
from deepkeel.tool_disclosure import ToolDiscoveryPort, install_tool_discovery
from deepkeel.tools import ToolExecutionContext, ToolExecutor
from deepkeel.turn_context import ToolViewMode, TurnExecutionContext
from deepkeel.ui import project_run_ui_state
from deepkeel.version import DEEPKEEL_CONTRACT_VERSION, DEEPKEEL_VERSION
from deepkeel.runtime_policy import (
    _budget_limits,
    _conservative_model_context_profile,
    _max_elapsed_seconds,
    _merge_skill_activation,
    _model_providers,
    _prior_budget_state,
    _prior_diagnostics,
    _resolved_model_policy,
)
from deepkeel.runtime_results import (
    _failed_runtime_state,
    project_harness_result,
)
from deepkeel.runtime_model_pipeline import build_runtime_model_gateway
from deepkeel.runtime_execution_support import hook_audit_dict
from deepkeel.runtime_turn_execution import RuntimeTurnExecutionMixin
from deepkeel.checkpoint_authority import CheckpointAuthority


EventSink = Callable[[dict[str, Any]], None]
SystemPromptFactory = Callable[[dict[str, Any]], str]


def _default_system_prompt_factory(skill_activation: dict[str, Any]) -> str:
    return harness_system_prompt(
        skill_instructions=str(skill_activation.get("prompt_instructions") or "").strip()
    )


class HarnessRuntime(RuntimeTurnExecutionMixin):
    """Product-neutral execution loop composed with explicit runtime ports."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        *,
        checkpointer: GraphCheckpointer | None = None,
        checkpoint_store: DurableCheckpointStore | None = None,
        async_checkpoint_store: AsyncDurableCheckpointStore | None = None,
        system_prompt_factory: SystemPromptFactory | None = None,
        session_factory: SessionFactory | None = None,
        max_steps: int = 12,
        model_router: ModelRouter | None = None,
        model_invocation_recorder: ModelInvocationRecorder | None = None,
        model_invocation_store: ModelInvocationStore | None = None,
        model_health_store: ModelHealthStore | None = None,
        policy_engine: PolicyEngine | None = None,
        budget_ledger: BudgetLedger | None = None,
        run_control: RunControl | None = None,
        capability_contributions: tuple[CapabilityContribution, ...] = (),
        capability_catalog: CapabilityCatalog | None = None,
        telemetry: TelemetryPort | None = None,
        context_builder: ContextBuilder | None = None,
        memory_recall_coordinator: MemoryRecallCoordinator | None = None,
        context_window_manager: ContextWindowManager | None = None,
        runtime_state_store: RuntimeStateStore | None = None,
        async_runtime_state_store: AsyncRuntimeStateStore | None = None,
        event_journal: RuntimeEventJournal | None = None,
        async_event_journal: AsyncRuntimeEventJournal | None = None,
        reference_projector: ReferenceProjector | None = None,
        run_lease_store: RunLeaseStore | None = None,
        async_run_lease_store: AsyncRunLeaseStore | None = None,
        run_lease_owner_id: str = "",
        run_lease_ttl_seconds: float = 60.0,
        state_migrations: StateMigrationRegistry | None = None,
        async_stream_buffer_size: int = 128,
        async_cancel_timeout_seconds: float = 5.0,
        reuse_compiled_graph: bool = True,
        graph_durability: GraphDurability = "exit",
        tool_view_mode: ToolViewMode = "legacy",
        hook_runner: HookRunner | None = None,
        tool_discovery_port: ToolDiscoveryPort | None = None,
        entry_tool_skill_activator: EntryToolSkillActivator | None = None,
        runtime_generation: RuntimeGeneration | None = None,
        planning_enabled: bool = False,
    ) -> None:
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.checkpointer = checkpointer or LangGraphCheckpointerAdapter()
        self.checkpoint_store = checkpoint_store
        self.async_checkpoint_store = async_checkpoint_store
        base_system_prompt_factory = system_prompt_factory or _default_system_prompt_factory
        self.planning_enabled = bool(planning_enabled)
        if self.planning_enabled:
            self.system_prompt_factory = lambda skill: "\n\n".join(
                part
                for part in (
                    base_system_prompt_factory(skill),
                    execution_planning_prompt(skill),
                )
                if part.strip()
            )
        else:
            self.system_prompt_factory = base_system_prompt_factory
        self.session_factory = session_factory
        self.max_steps = max(2, int(max_steps))
        self.model_router = model_router or AdaptiveStepModelRouter()
        self.model_invocation_recorder = model_invocation_recorder
        self.model_invocation_store = model_invocation_store
        self.model_health_store = model_health_store or InMemoryModelHealthStore()
        self.policy_engine = (
            policy_engine or getattr(tool_executor, "policy_engine", None) or DefaultPolicyEngine()
        )
        self.budget_ledger = (
            budget_ledger or getattr(tool_executor, "budget_ledger", None) or InMemoryBudgetLedger()
        )
        self.run_control = run_control or NoopRunControl()
        self.capability_contributions = capability_contributions
        self.capability_catalog = capability_catalog or CapabilityCatalog()
        self.telemetry = telemetry or NoopTelemetry()
        self.context_builder = context_builder
        self.memory_recall_coordinator = memory_recall_coordinator
        self.context_window_manager = context_window_manager or DeterministicContextWindowManager()
        self.runtime_state_store = runtime_state_store
        self.async_runtime_state_store = async_runtime_state_store
        self.event_journal = event_journal
        self.async_event_journal = async_event_journal
        self.reference_projector = reference_projector or DefaultReferenceProjector()
        self.run_lease_store = run_lease_store
        self.async_run_lease_store = async_run_lease_store
        self.run_lease_owner_id = str(run_lease_owner_id or f"runtime-{uuid4().hex}")
        self.run_lease_ttl_seconds = float(run_lease_ttl_seconds)
        self.state_migrations = state_migrations or default_state_migrations()
        self.async_stream_buffer_size = max(1, int(async_stream_buffer_size))
        self.async_cancel_timeout_seconds = max(0.1, float(async_cancel_timeout_seconds))
        self.reuse_compiled_graph = bool(reuse_compiled_graph)
        self.graph_durability = graph_durability
        self.tool_view_mode = tool_view_mode
        self.hook_runner = hook_runner or HookRunner()
        self.entry_tool_skill_activator = entry_tool_skill_activator
        self.runtime_generation = runtime_generation
        if self.tool_view_mode != "legacy":
            install_tool_discovery(
                self.tool_registry,
                self.tool_executor,
                discovery_port=tool_discovery_port,
            )
        if self.planning_enabled:
            install_execution_planning(self.tool_registry, self.tool_executor)
        self._compiled_graph: HarnessGraph | None = None
        self._graph_compile_count = 0
        self._graph_compile_lock = Lock()
        self.tool_executor.policy_engine = self.policy_engine
        self.tool_executor.budget_ledger = self.budget_ledger
        self.tool_executor.hook_runner = self.hook_runner
        self._validate_exclusive_io_ports()

    def _validate_exclusive_io_ports(self) -> None:
        pairs = (
            ("checkpoint_store", self.checkpoint_store, self.async_checkpoint_store),
            (
                "runtime_state_store",
                self.runtime_state_store,
                self.async_runtime_state_store,
            ),
            ("event_journal", self.event_journal, self.async_event_journal),
            ("run_lease_store", self.run_lease_store, self.async_run_lease_store),
        )
        for name, synchronous, asynchronous in pairs:
            if synchronous is not None and asynchronous is not None:
                raise ValueError(f"configure either {name} or async_{name}, not both")

    @property
    def graph_compile_count(self) -> int:
        return self._graph_compile_count

    @staticmethod
    def _hook_audit_payload(audit: HookAudit) -> dict[str, Any]:
        return hook_audit_dict(audit)

    def _shared_compiled_graph(self) -> HarnessGraph:
        graph = self._compiled_graph
        if graph is not None:
            return graph
        with self._graph_compile_lock:
            graph = self._compiled_graph
            if graph is None:
                graph = create_harness_graph(
                    tool_executor=self.tool_executor,
                    tool_registry=self.tool_registry,
                    max_steps=self.max_steps,
                    checkpointer=compiler_checkpointer(self.checkpointer),
                    supports_async_checkpointer=checkpointer_supports_async(self.checkpointer),
                    budget_ledger=self.budget_ledger,
                    run_control=self.run_control,
                    durability=self.graph_durability,
                )
                self._compiled_graph = graph
                self._graph_compile_count += 1
        return graph

    def close(self) -> None:
        self.capability_catalog.close()

    def replay_events(
        self,
        run_id: str,
        *,
        scope: RuntimeScope | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeStreamEvent, ...]:
        if self.event_journal is None:
            if self.async_event_journal is not None:
                raise RuntimeError("async event journal is configured; use areplay_events()")
            return ()
        resolved_scope = scope or RuntimeScope()
        operation = (
            scoped_adapter_operation(self.event_journal, "read_after", resolved_scope)
            if scope is not None
            else self.event_journal.read_after
        )
        kwargs: dict[str, Any] = {
            "after_sequence": after_sequence,
            "limit": limit,
        }
        if scope is not None and getattr(operation, "__name__", "") == "read_after_scoped":
            kwargs["scope"] = resolved_scope
        return tuple(
            RuntimeStreamEvent.model_validate(event) for event in operation(run_id, **kwargs)
        )

    async def areplay_events(
        self,
        run_id: str,
        *,
        scope: RuntimeScope | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeStreamEvent, ...]:
        resolved_scope = scope or RuntimeScope()
        kwargs: dict[str, Any] = {
            "after_sequence": after_sequence,
            "limit": limit,
        }
        if self.async_event_journal is not None:
            operation = (
                scoped_adapter_operation(
                    self.async_event_journal,
                    "read_after",
                    resolved_scope,
                )
                if scope is not None
                else self.async_event_journal.read_after
            )
            if scope is not None and getattr(operation, "__name__", "") == "read_after_scoped":
                kwargs["scope"] = resolved_scope
            events = await operation(run_id, **kwargs)
        elif self.event_journal is not None:
            operation = (
                scoped_adapter_operation(self.event_journal, "read_after", resolved_scope)
                if scope is not None
                else self.event_journal.read_after
            )
            if scope is not None and getattr(operation, "__name__", "") == "read_after_scoped":
                kwargs["scope"] = resolved_scope
            events = await run_sync_adapter(operation, run_id, **kwargs)
        else:
            return ()
        return tuple(RuntimeStreamEvent.model_validate(event) for event in events)

    @staticmethod
    def supports_native_tools(provider: Any) -> bool:
        return (
            isinstance(provider, ModelProviderAdapter)
            or callable(getattr(provider, "stream_chat", None))
            or callable(getattr(provider, "complete_chat", None))
        )

    def run(
        self,
        request: RuntimeRequest,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
        event_sink: EventSink | None = None,
    ) -> RuntimeResult:
        prepared = self._ensure_request_identity(request)
        if self.async_run_lease_store is not None:
            return asyncio.run(
                self.arun(
                    prepared,
                    provider=provider,
                    providers=providers,
                    session=session,
                    event_sink=event_sink,
                )
            )
        if self.run_lease_store is None:
            return self._run_claimed(
                prepared,
                provider=provider,
                providers=providers,
                session=session,
                event_sink=event_sink,
            )
        with RunLeaseGuard(
            self.run_lease_store,
            run_id=prepared.run_id,
            owner_id=self.run_lease_owner_id,
            ttl_seconds=self.run_lease_ttl_seconds,
            scope=prepared.runtime_scope,
        ) as lease_guard:

            def guarded_sink(event: dict[str, Any]) -> None:
                lease_guard.raise_if_lost()
                if event_sink is not None:
                    event_sink(event)

            result = self._run_claimed(
                prepared,
                provider=provider,
                providers=providers,
                session=session,
                event_sink=guarded_sink,
                execution_fence=lease_guard,
            )
            lease_guard.raise_if_lost()
            return result

    async def arun(
        self,
        request: RuntimeRequest,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
        event_sink: EventSink | None = None,
    ) -> RuntimeResult:
        """Run the canonical async state machine in the host event loop."""
        prepared = self._ensure_request_identity(request)
        if self.async_run_lease_store is not None:
            async with AsyncRunLeaseGuard(
                self.async_run_lease_store,
                run_id=prepared.run_id,
                owner_id=self.run_lease_owner_id,
                ttl_seconds=self.run_lease_ttl_seconds,
                scope=prepared.runtime_scope,
            ) as async_lease_guard:

                def guarded_async_sink(event: dict[str, Any]) -> None:
                    async_lease_guard.raise_if_lost()
                    if event_sink is not None:
                        event_sink(event)

                result = await self._arun_claimed(
                    prepared,
                    provider=provider,
                    providers=providers,
                    session=session,
                    event_sink=guarded_async_sink,
                    execution_fence=async_lease_guard,
                )
                async_lease_guard.raise_if_lost()
                return result
        if self.run_lease_store is None:
            return await self._arun_claimed(
                prepared,
                provider=provider,
                providers=providers,
                session=session,
                event_sink=event_sink,
            )
        async with AsyncRunLeaseGuard(
            AsyncRunLeaseStoreAdapter(self.run_lease_store),
            run_id=prepared.run_id,
            owner_id=self.run_lease_owner_id,
            ttl_seconds=self.run_lease_ttl_seconds,
            scope=prepared.runtime_scope,
        ) as sync_lease_guard:

            def guarded_sink(event: dict[str, Any]) -> None:
                sync_lease_guard.raise_if_lost()
                if event_sink is not None:
                    event_sink(event)

            result = await self._arun_claimed(
                prepared,
                provider=provider,
                providers=providers,
                session=session,
                event_sink=guarded_sink,
                execution_fence=sync_lease_guard,
            )
            sync_lease_guard.raise_if_lost()
            return result

    async def astream(
        self,
        request: RuntimeRequest,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        """Stream runtime events with cooperative cancellation for async hosts."""
        stream = stream_runtime_async(
            self,
            request,
            provider=provider,
            providers=providers,
            session=session,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    def _ensure_request_identity(self, request: RuntimeRequest) -> RuntimeRequest:
        bundle = dict(request.context_bundle)
        run_id = str(
            request.run_id
            or bundle.get("agent_session_id")
            or bundle.get("agent_run_id")
            or bundle.get("run_id")
            or uuid4()
        )
        thread_id = str(
            request.thread_id or bundle.get("thread_id") or bundle.get("ask_thread_id") or run_id
        )
        turn_id = str(request.turn_id or bundle.get("turn_id") or f"turn-{uuid4()}")
        bundle.setdefault("run_id", run_id)
        bundle.setdefault("thread_id", thread_id)
        bundle.setdefault("turn_id", turn_id)
        return request.model_copy(
            update={
                "run_id": run_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "context_bundle": bundle,
            }
        )

    def _run_claimed(
        self,
        request: RuntimeRequest,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
        event_sink: EventSink | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> RuntimeResult:
        return asyncio.run(
            self._arun_claimed(
                request,
                provider=provider,
                providers=providers,
                session=session,
                event_sink=event_sink,
                execution_fence=execution_fence,
            )
        )

    def cleanup_run(
        self,
        result: RuntimeResult | None,
        *,
        run_id: str,
        session: Any = None,
        user_id: str = "",
        scope: RuntimeScope | None = None,
        graph_thread_ids: list[str] | None = None,
    ) -> dict[str, str]:
        cleanup: dict[str, str] = {}
        errors: dict[str, str] = {}
        if self.checkpoint_store is not None:
            try:
                legacy_user_id = require_legacy_compatible_scope(
                    scope or RuntimeScope(user_id=user_id),
                    adapter_name=type(self.checkpoint_store).__name__,
                )
                self.checkpoint_store.delete(
                    run_id,
                    session=session,
                    user_id=legacy_user_id,
                )
                cleanup["durable"] = "deleted"
            except Exception as exc:
                cleanup["durable"] = "failed"
                errors["durable"] = str(exc)
        else:
            cleanup["durable"] = "not_configured"
        delete_thread = getattr(self.checkpointer, "delete_thread", None)
        if callable(delete_thread):
            resolved_scope = scope or RuntimeScope(user_id=user_id)
            thread_ids = list(
                dict.fromkeys([resolved_scope.qualify_identity(run_id), *(graph_thread_ids or [])])
            )
            for thread_id in thread_ids:
                try:
                    delete_thread(thread_id)
                except Exception as exc:
                    errors[f"langgraph:{thread_id}"] = str(exc)
            cleanup["langgraph"] = (
                "failed" if any(key.startswith("langgraph:") for key in errors) else "deleted"
            )
        else:
            cleanup["langgraph"] = "unsupported"
        if result is not None:
            diagnostics = result.diagnostics
            recovery = as_dict(diagnostics.get("recovery"))
            recovery["checkpoint_cleanup"] = cleanup
            if errors:
                recovery["checkpoint_cleanup_errors"] = errors
            diagnostics["recovery"] = recovery
        operational_run_id = (scope or RuntimeScope(user_id=user_id)).qualify_identity(run_id)
        self.budget_ledger.clear(operational_run_id)
        self.run_control.release(operational_run_id)
        return cleanup

    async def _acleanup_run(
        self,
        result: RuntimeResult | None,
        *,
        run_id: str,
        session: Any = None,
        user_id: str = "",
        scope: RuntimeScope | None = None,
        graph_thread_ids: list[str] | None = None,
    ) -> dict[str, str]:
        return await cleanup_runtime_async(
            self,
            result,
            run_id=run_id,
            session=session,
            user_id=user_id,
            scope=scope,
            graph_thread_ids=graph_thread_ids,
        )

    def _load_durable_checkpoint(
        self, run_id: str, *, session: Any, user_id: str
    ) -> dict[str, Any]:
        if self.checkpoint_store is None:
            return {}
        try:
            loaded = self.checkpoint_store.load(run_id, session=session, user_id=user_id)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _load_authoritative_checkpoint(
        self,
        run_id: str,
        *,
        session: Any,
        user_id: str,
        scope: RuntimeScope | None = None,
    ) -> tuple[dict[str, Any], CheckpointAuthority, list[str]]:
        """Load portable state from the canonical store before compatibility fallbacks."""

        errors: list[str] = []
        if self.runtime_state_store is not None:
            try:
                load_scoped = getattr(
                    self.runtime_state_store,
                    "load_snapshot_scoped",
                    None,
                )
                if scope is not None and callable(load_scoped):
                    snapshot = load_scoped(
                        run_id,
                        session=session,
                        scope=scope,
                    )
                else:
                    legacy_user_id = require_legacy_compatible_scope(
                        scope or RuntimeScope(user_id=user_id),
                        adapter_name=type(self.runtime_state_store).__name__,
                    )
                    snapshot = self.runtime_state_store.load_snapshot(
                        run_id,
                        session=session,
                        user_id=legacy_user_id,
                    )
                state = snapshot.checkpoint_state
                if isinstance(state, dict) and state:
                    return dict(state), CheckpointAuthority.RUNTIME_STATE_STORE, errors
            except Exception as exc:
                errors.append(f"runtime_state_store:{type(exc).__name__}:{exc}")
        if self.checkpoint_store is not None:
            try:
                legacy_user_id = require_legacy_compatible_scope(
                    scope or RuntimeScope(user_id=user_id),
                    adapter_name=type(self.checkpoint_store).__name__,
                )
                legacy_state = self.checkpoint_store.load(
                    run_id,
                    session=session,
                    user_id=legacy_user_id,
                )
                if isinstance(legacy_state, dict) and legacy_state:
                    return dict(legacy_state), CheckpointAuthority.DURABLE_CHECKPOINT_STORE, errors
            except Exception as exc:
                errors.append(f"durable_checkpoint_store:{type(exc).__name__}:{exc}")
        return {}, CheckpointAuthority.SESSION_PROJECTION, errors

    async def _aload_authoritative_checkpoint(
        self,
        run_id: str,
        *,
        session: Any,
        user_id: str,
        scope: RuntimeScope,
    ) -> tuple[dict[str, Any], str, list[str]]:
        return await load_authoritative_checkpoint_async(
            self,
            run_id,
            session=session,
            user_id=user_id,
            scope=scope,
        )

    async def _event_latest_sequence(
        self,
        run_id: str,
        *,
        scope: RuntimeScope | None = None,
        fallback: int = 0,
    ) -> int:
        return await event_latest_sequence(
            self,
            run_id,
            scope=scope,
            fallback=fallback,
        )

    def _host_owns_terminal_settlement(self) -> bool:
        return host_owns_terminal_settlement(self)

    async def _record_checkpoint_cleanup_event(
        self,
        result: RuntimeResult,
        *,
        run_id: str,
        thread_id: str,
        turn_id: str,
        scope: RuntimeScope,
        event_sink: EventSink | None,
        fallback_sequence: int = 0,
    ) -> None:
        await record_checkpoint_cleanup_event(
            self,
            result,
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            scope=scope,
            event_sink=event_sink,
            fallback_sequence=fallback_sequence,
        )

    def _has_graph_checkpoint(self, thread_id: str) -> bool:
        exists = getattr(self.checkpointer, "exists", None)
        if callable(exists):
            try:
                return bool(exists(thread_id))
            except Exception:
                return False
        has_checkpoint = getattr(self.checkpointer, "has_checkpoint", None)
        if callable(has_checkpoint):
            try:
                return bool(has_checkpoint(thread_id))
            except Exception:
                return False
        target = compiler_checkpointer(self.checkpointer)
        get_tuple = getattr(target, "get_tuple", None)
        if not callable(get_tuple):
            return False
        try:
            return get_tuple({"configurable": {"thread_id": thread_id}}) is not None
        except Exception:
            return False

    def _capability_manifest(self) -> dict[str, Any]:
        result = {
            "packages": [
                {
                    "package_id": contribution.package_id,
                    "tools": len(contribution.tools),
                    "skills": len(contribution.skills),
                    "artifact_types": len(contribution.artifact_types),
                    "subagents": len(contribution.subagents),
                    "resources": len(contribution.resources),
                }
                for contribution in self.capability_contributions
            ],
            "catalog": {
                "skills": len(self.capability_catalog.skills),
                "artifact_types": len(self.capability_catalog.artifact_types),
                "handoffs": len(self.capability_catalog.handoffs),
                "tool_providers": len(self.capability_catalog.tool_providers),
                "subagents": len(self.capability_catalog.subagents),
                "context_contributors": len(self.capability_catalog.context_contributors),
                "resources": len(self.capability_catalog.resources),
                "hooks": len(self.capability_catalog.hooks),
            },
        }
        if self.runtime_generation is not None:
            result["generation"] = self.runtime_generation.model_dump(mode="json")
        return result

    async def _persist_runtime_snapshot(
        self,
        result: RuntimeResult,
        *,
        run_id: str,
        thread_id: str,
        session: Any,
        user_id: str,
        context_bundle: dict[str, Any],
        execution_fence: ExecutionFence | None = None,
        scope: RuntimeScope | None = None,
    ) -> None:
        await persist_runtime_snapshot(
            self,
            result,
            run_id=run_id,
            thread_id=thread_id,
            session=session,
            user_id=user_id,
            context_bundle=context_bundle,
            execution_fence=execution_fence,
            scope=scope,
        )
