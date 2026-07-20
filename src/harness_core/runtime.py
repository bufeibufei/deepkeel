from __future__ import annotations
import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import hashlib
import json
import time
from collections.abc import AsyncIterator
from threading import Event as ThreadEvent
from typing import Any, Callable
from uuid import uuid4

from harness_core.budget import (
    ELAPSED_SECONDS,
    INPUT_TOKENS,
    MODEL_CALLS,
    MODEL_RETRIES,
    OUTPUT_TOKENS,
    TOOL_CALLS,
    TOOL_CONCURRENCY,
    BudgetLedger,
    BudgetRequest,
    InMemoryBudgetLedger,
)
from harness_core.contracts import (
    Artifact,
    FinalAnswer,
    Observation,
    PendingAction,
    RunContext,
    RunStatus,
    ToolCall,
    ToolResult,
)
from harness_core.context import build_context_snapshot, build_initial_messages
from harness_core.context_window import (
    ContextWindowManager,
    DeterministicContextWindowManager,
)
from harness_core.control import NoopRunControl, RunControl
from harness_core.events import AgentEventPersistenceError, project_runtime_event
from harness_core.failures import RuntimeFailure, classify_runtime_failure
from harness_core.graph import create_harness_graph
from harness_core.langgraph_adapter import (
    LangGraphCheckpointerAdapter,
    compiler_checkpointer,
)
from harness_core.model import (
    ModelInvocationRecorder,
    ModelProviderAdapter,
    RoutedModelGateway,
)
from harness_core.model_routing import AdaptiveStepModelRouter, ModelRouter
from harness_core.leases import ExecutionFence, RunLeaseGuard, RunLeaseStore
from harness_core.migrations import StateMigrationRegistry, default_state_migrations
from harness_core.persistence import (
    DurableCheckpointStore,
    checkpoint_from_durable_state,
    checkpoint_from_runtime,
    durable_state_from_result,
    restore_run_context,
    resume_payload_from_context,
)
from harness_core.capabilities import CapabilityCatalog, CapabilityContribution
from harness_core.ports import ContextBuilder, GraphCheckpointer, SessionFactory
from harness_core.prompts import harness_system_prompt
from harness_core.policy import DefaultPolicyEngine, PolicyEngine
from harness_core.references import (
    DefaultReferenceProjector,
    ReferenceProjector,
)
from harness_core.runtime_api import (
    RuntimeRequest,
    RuntimeResult,
    RuntimeResultStatus,
    RuntimeStreamEvent,
)
from harness_core.skills import SkillPolicy
from harness_core.state_store import RuntimeStateMutation, RuntimeStateStore
from harness_core.telemetry import NoopTelemetry, TelemetryPort, TelemetryRecord
from harness_core.type_narrowing import as_dict
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutionContext, ToolExecutor
from harness_core.ui import project_run_ui_state
from harness_core.version import HARNESS_CORE_CONTRACT_VERSION, HARNESS_CORE_VERSION
from harness_core.runtime_policy import (
    _budget_limits,
    _max_elapsed_seconds,
    _merge_skill_activation,
    _model_providers,
    _prior_budget_state,
    _prior_diagnostics,
    _resolved_model_policy,
)
from harness_core.runtime_results import (
    _failed_runtime_state,
    _new_context,
    _skill_precondition_tool_calls,
    project_harness_result,
)


EventSink = Callable[[dict[str, Any]], None]
SystemPromptFactory = Callable[[dict[str, Any]], str]


def _default_system_prompt_factory(skill_activation: dict[str, Any]) -> str:
    return harness_system_prompt(
        skill_instructions=str(skill_activation.get("prompt_instructions") or "").strip()
    )


def _runtime_state_mutation_id(
    run_id: str,
    status: str,
    durable_state: dict[str, Any],
) -> str:
    encoded = json.dumps(
        durable_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"{run_id}:{status}:{digest}"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class HarnessRuntime:
    """Product-neutral execution loop composed with explicit runtime ports."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        *,
        checkpointer: GraphCheckpointer | None = None,
        checkpoint_store: DurableCheckpointStore | None = None,
        system_prompt_factory: SystemPromptFactory | None = None,
        session_factory: SessionFactory | None = None,
        max_steps: int = 12,
        model_router: ModelRouter | None = None,
        model_invocation_recorder: ModelInvocationRecorder | None = None,
        policy_engine: PolicyEngine | None = None,
        budget_ledger: BudgetLedger | None = None,
        run_control: RunControl | None = None,
        capability_contributions: tuple[CapabilityContribution, ...] = (),
        capability_catalog: CapabilityCatalog | None = None,
        telemetry: TelemetryPort | None = None,
        context_builder: ContextBuilder | None = None,
        context_window_manager: ContextWindowManager | None = None,
        runtime_state_store: RuntimeStateStore | None = None,
        reference_projector: ReferenceProjector | None = None,
        run_lease_store: RunLeaseStore | None = None,
        run_lease_owner_id: str = "",
        run_lease_ttl_seconds: float = 60.0,
        state_migrations: StateMigrationRegistry | None = None,
        async_stream_buffer_size: int = 128,
        async_cancel_timeout_seconds: float = 5.0,
    ) -> None:
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.checkpointer = checkpointer or LangGraphCheckpointerAdapter()
        self.checkpoint_store = checkpoint_store
        self.system_prompt_factory = system_prompt_factory or _default_system_prompt_factory
        self.session_factory = session_factory
        self.max_steps = max(2, int(max_steps))
        self.model_router = model_router or AdaptiveStepModelRouter()
        self.model_invocation_recorder = model_invocation_recorder
        self.policy_engine = policy_engine or getattr(tool_executor, "policy_engine", None) or DefaultPolicyEngine()
        self.budget_ledger = budget_ledger or getattr(tool_executor, "budget_ledger", None) or InMemoryBudgetLedger()
        self.run_control = run_control or NoopRunControl()
        self.capability_contributions = capability_contributions
        self.capability_catalog = capability_catalog or CapabilityCatalog()
        self.telemetry = telemetry or NoopTelemetry()
        self.context_builder = context_builder
        self.context_window_manager = (
            context_window_manager or DeterministicContextWindowManager()
        )
        self.runtime_state_store = runtime_state_store
        self.reference_projector = reference_projector or DefaultReferenceProjector()
        self.run_lease_store = run_lease_store
        self.run_lease_owner_id = str(run_lease_owner_id or f"runtime-{uuid4().hex}")
        self.run_lease_ttl_seconds = float(run_lease_ttl_seconds)
        self.state_migrations = state_migrations or default_state_migrations()
        self.async_stream_buffer_size = max(1, int(async_stream_buffer_size))
        self.async_cancel_timeout_seconds = max(0.1, float(async_cancel_timeout_seconds))
        self.tool_executor.policy_engine = self.policy_engine
        self.tool_executor.budget_ledger = self.budget_ledger

    def close(self) -> None:
        self.capability_catalog.close()

    @staticmethod
    def supports_native_tools(provider: Any) -> bool:
        return isinstance(provider, ModelProviderAdapter) or callable(
            getattr(provider, "stream_chat", None)
        ) or callable(getattr(provider, "complete_chat", None))

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
        """Run the canonical state machine without blocking an async host."""
        prepared = self._ensure_request_identity(request)
        return await asyncio.to_thread(
            self.run,
            prepared,
            provider=provider,
            providers=providers,
            session=session,
            event_sink=event_sink,
        )

    async def astream(
        self,
        request: RuntimeRequest,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        """Stream runtime events with cooperative cancellation for async hosts."""
        prepared = self._ensure_request_identity(request)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(
            maxsize=self.async_stream_buffer_size
        )
        closed = ThreadEvent()

        def sink(event: dict[str, Any]) -> None:
            if closed.is_set():
                self.run_control.raise_if_cancelled(prepared.run_id, force=True)
                return
            future = asyncio.run_coroutine_threadsafe(queue.put(("event", event)), loop)
            while True:
                try:
                    future.result(timeout=0.1)
                    return
                except FutureTimeoutError:
                    if closed.is_set():
                        future.cancel()
                        self.run_control.raise_if_cancelled(prepared.run_id, force=True)
                        return

        async def execute() -> None:
            try:
                result = await self.arun(
                    prepared,
                    provider=provider,
                    providers=providers,
                    session=session,
                    event_sink=sink,
                )
                if not closed.is_set():
                    await queue.put(("result", result))
            except BaseException as exc:
                if not closed.is_set():
                    await queue.put(("error", exc))

        task = asyncio.create_task(execute())
        completed = False
        try:
            while True:
                kind, value = await queue.get()
                if kind == "event":
                    yield RuntimeStreamEvent.model_validate(value)
                    continue
                if kind == "error":
                    raise value
                result = value
                yield RuntimeStreamEvent(
                    event_type="runtime.result",
                    title="Runtime result",
                    summary=result.final_answer.summary,
                    payload={"result": result.model_dump(mode="json")},
                )
                completed = True
                return
        finally:
            closed.set()
            if not completed and not task.done():
                cancel = getattr(self.run_control, "cancel", None)
                if callable(cancel):
                    cancel(prepared.run_id)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=self.async_cancel_timeout_seconds,
                    )
                except (TimeoutError, asyncio.CancelledError):
                    task.cancel()

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
            request.thread_id
            or bundle.get("thread_id")
            or bundle.get("ask_thread_id")
            or run_id
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
        """Execute one turn through the typed, product-neutral public API."""
        question = request.question
        user_id = request.user_id
        short_context = request.short_context
        context_bundle = request.context_bundle
        skill_activation = request.skill_activation
        model_policy = request.model_policy
        run_started_monotonic = time.monotonic()
        short = short_context if isinstance(short_context, dict) else {}
        bundle = context_bundle if isinstance(context_bundle, dict) else {}
        context_window_diagnostics: dict[str, Any] = {}
        try:
            if self.context_builder is not None:
                bundle = self.context_builder(question, short, bundle)
                if not isinstance(bundle, dict):
                    raise TypeError("context builder must return a mapping")
            for contributor_id, contributor in self.capability_catalog.context_contributors.items():
                contributed = contributor(dict(bundle))
                if not isinstance(contributed, dict):
                    raise TypeError(
                        f"context contributor {contributor_id} must return a mapping"
                    )
                bundle = contributed
            prepared_context = self.context_window_manager.prepare(
                question,
                short,
                bundle,
            )
            bundle = prepared_context.context_bundle
            context_window_diagnostics = dict(prepared_context.diagnostics)
        except Exception as exc:
            return self._context_setup_failure(
                question,
                exc,
                short_context=short,
                context_bundle=bundle,
                user_id=user_id,
                skill_activation=skill_activation,
                model_policy=model_policy,
                session=session,
                event_sink=event_sink,
                execution_fence=execution_fence,
            )
        run_id = str(
            request.run_id
            or bundle.get("agent_session_id")
            or bundle.get("agent_run_id")
            or bundle.get("run_id")
            or uuid4()
        )
        durable_state = (
            self._load_durable_checkpoint(run_id, session=session, user_id=str(user_id or "local-device"))
            if short.get("resume")
            else {}
        )
        resolved_model_policy = _resolved_model_policy(
            model_policy,
            provider=provider,
            providers=providers,
            max_steps=self.max_steps,
        )
        max_elapsed_seconds = _max_elapsed_seconds(resolved_model_policy)
        deadline_monotonic = (
            time.monotonic() + max_elapsed_seconds
            if max_elapsed_seconds > 0
            else None
        )
        self.budget_ledger.restore(run_id, _prior_budget_state(durable_state, short))
        previous_diagnostics = _prior_diagnostics(durable_state, short)
        resolved_skill = _merge_skill_activation(
            durable_state=durable_state,
            session_projection=short,
            explicit=skill_activation,
        )
        skill_policy = SkillPolicy.from_snapshot(resolved_skill)
        skill = skill_policy.runtime_snapshot()
        conversation_thread_id = str(
            request.thread_id
            or bundle.get("thread_id")
            or bundle.get("ask_thread_id")
            or short.get("ask_thread_id")
            or run_id
        )
        graph_thread_id = run_id
        turn_id = str(
            request.turn_id
            or bundle.get("turn_id")
            or short.get("turn_id")
            or f"turn-{uuid4()}"
        )
        events: list[dict[str, Any]] = []
        telemetry_error_count = 0
        telemetry_last_error = ""
        answer_delta_streamed = False

        def emit(event: dict[str, Any]) -> None:
            nonlocal answer_delta_streamed, telemetry_error_count, telemetry_last_error
            if execution_fence is not None:
                execution_fence.raise_if_lost()
            self.run_control.raise_if_cancelled(run_id)
            projected = project_runtime_event(event)
            payload = as_dict(projected.get("payload"))
            payload.setdefault("skill_id", str(skill.get("skill_id") or ""))
            projected["payload"] = payload
            if projected.get("event_type") == "answer.delta":
                answer_delta_streamed = True
            if not projected.get("ephemeral"):
                events.append(projected)
            try:
                self.telemetry.record(
                    TelemetryRecord.from_runtime_event(
                        projected,
                        run_id=run_id,
                        thread_id=conversation_thread_id,
                        turn_id=turn_id,
                    )
                )
            except Exception as exc:
                telemetry_error_count += 1
                telemetry_last_error = f"{type(exc).__name__}: {exc}"
            if event_sink is not None:
                event_sink(projected)

        model_providers = _model_providers(provider, providers, resolved_model_policy)
        model_gateway = RoutedModelGateway(
            model_providers,
            router=self.model_router,
            policy_engine=self.policy_engine,
            budget_ledger=self.budget_ledger,
            invocation_recorder=self.model_invocation_recorder,
        )
        graph = create_harness_graph(
            model=model_gateway,
            tool_executor=self.tool_executor,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt_factory(skill),
            max_steps=self.max_steps,
            checkpointer=compiler_checkpointer(self.checkpointer),
            budget_ledger=self.budget_ledger,
            deadline_monotonic=deadline_monotonic,
            run_control=self.run_control,
        )
        budget_limits = _budget_limits(resolved_model_policy)
        tool_context = ToolExecutionContext(
            run_id=run_id,
            user_id=str(user_id or "local-device"),
            thread_id=conversation_thread_id,
            turn_id=turn_id,
            session=session,
            session_factory=self.session_factory,
            context_bundle=bundle,
            metadata={
                "skill_activation": skill,
                "tenant_id": str(bundle.get("tenant_id") or ""),
                "governance_scope": {
                    "tenant_id": str(bundle.get("tenant_id") or ""),
                    "user_id": str(user_id or "local-device"),
                    "skill_id": str(skill.get("skill_id") or ""),
                    "scopes": list(bundle.get("governance_scopes") or []),
                },
                "model_providers": model_providers,
                "model_policy": resolved_model_policy,
                "event_sink": emit,
                "budget_ledger": self.budget_ledger,
            },
            budget_limits=budget_limits,
            deadline_monotonic=deadline_monotonic,
            run_control=self.run_control,
            execution_fence=execution_fence,
        )

        recovery_source = ""
        active_graph_thread_id = graph_thread_id
        try:
            if short.get("recover_interrupted"):
                if self._has_graph_checkpoint(graph_thread_id):
                    state = dict(graph.recover(
                        graph_thread_id,
                        tool_context=tool_context,
                        event_sink=emit,
                    ))
                    if not isinstance(state, dict) or not state:
                        raise RuntimeError("langgraph recovery checkpoint is unavailable")
                    recovery_source = "durable_langgraph_restart"
                else:
                    context = _new_context(
                        question,
                        run_id=run_id,
                        thread_id=graph_thread_id,
                        turn_id=turn_id,
                        user_id=str(user_id or "local-device"),
                        short_context=short,
                        context_bundle=bundle,
                        skill_activation=skill,
                        model_policy=resolved_model_policy,
                        budget_state=self.budget_ledger.snapshot(run_id).as_dict(),
                    )
                    state = dict(graph.invoke(
                        context,
                        tool_context=tool_context,
                        event_sink=emit,
                    ))
                    recovery_source = "restart_replay_without_checkpoint"
            elif short.get("resume"):
                resume_payload = resume_payload_from_context(short)
                try:
                    state = dict(graph.resume(
                        graph_thread_id,
                        resume_payload,
                        tool_context=tool_context,
                        event_sink=emit,
                    ))
                    if not isinstance(state, dict):
                        raise RuntimeError("langgraph checkpoint is unavailable")
                    recovery_source = "live_langgraph"
                except AgentEventPersistenceError:
                    raise
                except (RuntimeError, ValueError, AttributeError):
                    previous_runtime = short.get("previous_runtime")
                    recovered_checkpoint = checkpoint_from_durable_state(
                        durable_state,
                        migrations=self.state_migrations,
                    )
                    if not recovered_checkpoint:
                        recovered_checkpoint = checkpoint_from_runtime(
                            previous_runtime if isinstance(previous_runtime, dict) else {},
                            migrations=self.state_migrations,
                        )
                    if not recovered_checkpoint:
                        raise RuntimeError("durable checkpoint is unavailable")
                    recovery_source = (
                        "agent_run_checkpoint"
                        if checkpoint_from_durable_state(
                            durable_state,
                            migrations=self.state_migrations,
                        )
                        else "session_projection"
                    )
                    recovered_thread_id = f"{graph_thread_id}:recovered:{uuid4().hex[:8]}"
                    recovered = restore_run_context(
                        checkpoint=recovered_checkpoint,
                        resume_payload=resume_payload,
                        run_id=run_id,
                        thread_id=recovered_thread_id,
                        turn_id=turn_id,
                        user_id=str(user_id or "local-device"),
                        skill_activation=skill,
                        model_policy=resolved_model_policy,
                        migrations=self.state_migrations,
                    )
                    if not any(message.role == "user" for message in recovered.messages):
                        recovered.messages = [*build_initial_messages(question, short, bundle), *recovered.messages]
                    active_graph_thread_id = recovered_thread_id
                    state = dict(graph.invoke(
                        recovered,
                        tool_context=tool_context,
                        event_sink=emit,
                    ))
            else:
                precondition_calls = _skill_precondition_tool_calls(
                    skill,
                    context={**short, **bundle},
                    tool_registry=self.tool_registry,
                )
                context = _new_context(
                    question,
                    run_id=run_id,
                    thread_id=graph_thread_id,
                    turn_id=turn_id,
                    user_id=str(user_id or "local-device"),
                    short_context=short,
                    context_bundle=bundle,
                    skill_activation=skill,
                    model_policy=resolved_model_policy,
                    budget_state=self.budget_ledger.snapshot(run_id).as_dict(),
                    pending_tool_calls=precondition_calls,
                )
                state = dict(graph.invoke(
                    context,
                    tool_context=tool_context,
                    event_sink=emit,
                ))
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            failure = classify_runtime_failure(exc)
            if failure.category != "canceled":
                emit({
                    "event_type": "agent.failed",
                    "title": "Agent run failed",
                    "summary": failure.user_message,
                    "payload": {
                        "error": failure.detail,
                        "error_type": failure.exception_type,
                        "error_code": failure.code,
                        "failure": failure.as_dict(),
                    },
                })
            state = _failed_runtime_state(
                question,
                exc,
                run_id=run_id,
                thread_id=graph_thread_id,
                turn_id=turn_id,
                user_id=str(user_id or "local-device"),
                short_context=short,
                context_bundle=bundle,
                skill_activation=skill,
                model_policy=resolved_model_policy,
                budget_state=self.budget_ledger.snapshot(run_id).as_dict(),
                events=events,
                failure=failure,
            )
        result = project_harness_result(
            state,
            question=question,
            context_bundle=bundle,
            short_context=short,
            skill_activation=skill,
            streamed_events=events,
            user_id=str(user_id or "local-device"),
            answer_delta_streamed=answer_delta_streamed,
            observation_kinds={
                spec.name: str(spec.observation_contract.get("primary_kind") or "")
                for spec in self.tool_registry.list_tools()
            },
            task_kinds={spec.name: spec.task_kind for spec in self.tool_registry.list_tools()},
            max_steps=self.max_steps,
            previous_diagnostics=previous_diagnostics,
            capability_manifest=self._capability_manifest(),
            reference_projector=self.reference_projector,
        )
        if recovery_source:
            diagnostics = result.diagnostics
            recovery = as_dict(diagnostics.get("recovery"))
            recovery["checkpoint_source"] = recovery_source
            diagnostics["recovery"] = recovery
        diagnostics = result.diagnostics
        diagnostics["context_window"] = context_window_diagnostics
        elapsed_budget = self.budget_ledger.consume(
            BudgetRequest(
                run_id=run_id,
                metric=ELAPSED_SECONDS,
                amount=max(0.0, time.monotonic() - run_started_monotonic),
                limit=_budget_limits(resolved_model_policy).get(ELAPSED_SECONDS),
                operation_id=f"runtime-elapsed:{turn_id}",
                metadata={"turn_id": turn_id},
            )
        )
        diagnostics["budget"] = {
            "elapsed": elapsed_budget.as_dict(),
            "snapshot": self.budget_ledger.snapshot(run_id).as_dict(),
        }
        result.checkpoint["budget_state"] = self.budget_ledger.snapshot(run_id).as_dict()
        try:
            self.telemetry.record(
                TelemetryRecord(
                    event_name="runtime.settled",
                    run_id=run_id,
                    thread_id=conversation_thread_id,
                    turn_id=turn_id,
                    status=result.status.value,
                    attributes={
                        "status": result.status.value,
                        "stop_reason": result.stop_reason,
                        "skill_id": str(skill.get("skill_id") or ""),
                        "recovery_source": recovery_source,
                    },
                )
            )
        except Exception as exc:
            telemetry_error_count += 1
            telemetry_last_error = f"{type(exc).__name__}: {exc}"
        if telemetry_error_count:
            diagnostics["telemetry"] = {
                "status": "degraded",
                "error_count": telemetry_error_count,
                "last_error": telemetry_last_error,
            }
        self._persist_runtime_snapshot(
            result,
            run_id=run_id,
            thread_id=conversation_thread_id,
            session=session,
            user_id=str(user_id or "local-device"),
            context_bundle=bundle,
            execution_fence=execution_fence,
        )
        if result.status.value in {"completed", "failed", "canceled"}:
            self.cleanup_run(
                result,
                run_id=run_id,
                session=session,
                user_id=str(user_id or "local-device"),
                graph_thread_ids=[active_graph_thread_id],
            )
        return result

    def cleanup_run(
        self,
        result: RuntimeResult | None,
        *,
        run_id: str,
        session: Any = None,
        user_id: str = "",
        graph_thread_ids: list[str] | None = None,
    ) -> dict[str, str]:
        cleanup: dict[str, str] = {}
        errors: dict[str, str] = {}
        if self.checkpoint_store is not None:
            try:
                self.checkpoint_store.delete(
                    run_id,
                    session=session,
                    user_id=user_id,
                )
                cleanup["durable"] = "deleted"
            except Exception as exc:
                cleanup["durable"] = "failed"
                errors["durable"] = str(exc)
        else:
            cleanup["durable"] = "not_configured"
        delete_thread = getattr(self.checkpointer, "delete_thread", None)
        if callable(delete_thread):
            thread_ids = list(dict.fromkeys([run_id, *(graph_thread_ids or [])]))
            for thread_id in thread_ids:
                try:
                    delete_thread(thread_id)
                except Exception as exc:
                    errors[f"langgraph:{thread_id}"] = str(exc)
            cleanup["langgraph"] = "failed" if any(
                key.startswith("langgraph:") for key in errors
            ) else "deleted"
        else:
            cleanup["langgraph"] = "unsupported"
        if result is not None:
            diagnostics = result.diagnostics
            recovery = as_dict(diagnostics.get("recovery"))
            recovery["checkpoint_cleanup"] = cleanup
            if errors:
                recovery["checkpoint_cleanup_errors"] = errors
            diagnostics["recovery"] = recovery
        self.budget_ledger.clear(run_id)
        self.run_control.release(run_id)
        return cleanup

    def _load_durable_checkpoint(self, run_id: str, *, session: Any, user_id: str) -> dict[str, Any]:
        if self.checkpoint_store is None:
            return {}
        try:
            loaded = self.checkpoint_store.load(run_id, session=session, user_id=user_id)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

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

    def _context_setup_failure(
        self,
        question: str,
        exc: Exception,
        *,
        short_context: dict[str, Any],
        context_bundle: dict[str, Any],
        user_id: str,
        skill_activation: dict[str, Any] | None,
        model_policy: dict[str, Any] | None,
        session: Any,
        event_sink: EventSink | None,
        execution_fence: ExecutionFence | None,
    ) -> RuntimeResult:
        run_id = str(
            context_bundle.get("agent_session_id")
            or context_bundle.get("agent_run_id")
            or context_bundle.get("run_id")
            or uuid4()
        )
        thread_id = str(
            context_bundle.get("thread_id")
            or context_bundle.get("ask_thread_id")
            or short_context.get("ask_thread_id")
            or run_id
        )
        turn_id = str(
            context_bundle.get("turn_id")
            or short_context.get("turn_id")
            or f"turn-{uuid4()}"
        )
        failure = classify_runtime_failure(exc)
        resolved_skill = SkillPolicy.from_snapshot(skill_activation).runtime_snapshot()
        event = project_runtime_event(
            {
                "event_type": "agent.failed",
                "title": "Agent context setup failed",
                "summary": failure.user_message,
                "payload": {
                    "error": failure.detail,
                    "error_type": failure.exception_type,
                    "error_code": failure.code,
                    "failure": failure.as_dict(),
                    "phase": "context_setup",
                    "skill_id": str(resolved_skill.get("skill_id") or ""),
                },
            }
        )
        try:
            self.telemetry.record(
                TelemetryRecord.from_runtime_event(
                    event,
                    run_id=run_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            )
        except Exception:
            pass
        if event_sink is not None:
            event_sink(event)
        resolved_policy = dict(model_policy) if isinstance(model_policy, dict) else {}
        state = _failed_runtime_state(
            question,
            exc,
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            user_id=str(user_id or "local-device"),
            short_context=short_context,
            context_bundle=context_bundle,
            skill_activation=resolved_skill,
            model_policy=resolved_policy,
            budget_state={},
            events=[event],
            failure=failure,
            phase="context_setup",
        )
        result = project_harness_result(
            state,
            question=question,
            context_bundle=context_bundle,
            short_context=short_context,
            skill_activation=resolved_skill,
            streamed_events=[event],
            user_id=str(user_id or "local-device"),
            answer_delta_streamed=False,
            observation_kinds={},
            task_kinds={},
            max_steps=self.max_steps,
            capability_manifest=self._capability_manifest(),
        )
        try:
            self.telemetry.record(
                TelemetryRecord(
                    event_name="runtime.settled",
                    run_id=run_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    status="failed",
                    attributes={
                        "status": "failed",
                        "stop_reason": result.stop_reason,
                        "skill_id": str(resolved_skill.get("skill_id") or ""),
                        "recovery_source": "",
                        "phase": "context_setup",
                    },
                )
            )
        except Exception:
            pass
        self._persist_runtime_snapshot(
            result,
            run_id=run_id,
            thread_id=thread_id,
            session=session,
            user_id=str(user_id or "local-device"),
            context_bundle=context_bundle,
            execution_fence=execution_fence,
        )
        return result

    def _capability_manifest(self) -> dict[str, Any]:
        return {
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
                "context_contributors": len(
                    self.capability_catalog.context_contributors
                ),
                "resources": len(self.capability_catalog.resources),
            },
        }

    def _persist_runtime_snapshot(
        self,
        result: RuntimeResult,
        *,
        run_id: str,
        thread_id: str,
        session: Any,
        user_id: str,
        context_bundle: dict[str, Any],
        execution_fence: ExecutionFence | None = None,
    ) -> None:
        status = result.status.value
        if status not in {
            "waiting_user_action",
            "waiting_user_input",
            "task_running",
            "completed",
            "failed",
            "canceled",
        }:
            return
        diagnostics = result.diagnostics
        recovery = as_dict(diagnostics.get("recovery"))
        durable_state = durable_state_from_result(
            result,
            run_id=run_id,
            thread_id=thread_id,
        )
        if self.runtime_state_store is not None:
            if execution_fence is not None:
                execution_fence.raise_if_lost()
            resume_token = str(result.checkpoint.get("resume_token") or "")
            terminal = status in {"completed", "failed", "canceled"}
            if terminal and getattr(
                self.runtime_state_store,
                "terminal_settlement_owner",
                "runtime",
            ) == "host":
                recovery["atomic_checkpoint"] = "host_settlement_required"
                diagnostics["recovery"] = recovery
                return
            event_type = "run.settled" if terminal else "runtime.checkpoint.committed"
            mutation_payload = {
                "status": status,
                "stop_reason": result.stop_reason,
                "resume_token": resume_token,
                "checkpoint_schema_version": str(durable_state.get("schema_version") or ""),
            }
            mutation_id = _runtime_state_mutation_id(
                run_id,
                status,
                durable_state,
            )
            try:
                receipt = self.runtime_state_store.commit(
                    RuntimeStateMutation(
                        mutation_id=mutation_id,
                        run_id=run_id,
                        event_type=event_type,
                        target_status=status,
                        event_payload=mutation_payload,
                        event_visibility="public" if terminal else "internal",
                        checkpoint_type="terminal" if terminal else "runtime",
                        checkpoint_state=durable_state,
                        resume_token=resume_token,
                        error_code=(
                            str(result.error["code"] if result.error is not None else "RUN_FAILED")
                            if status == "failed"
                            else None
                        ),
                        error_message=(
                            str(result.error["message"] if result.error is not None else "")
                            if status == "failed"
                            else None
                        ),
                        delete_checkpoint_types=(
                            ("runtime", "suspended", "resume", "settling")
                            if terminal
                            else ()
                        ),
                        expected_version=_optional_int(
                            context_bundle.get("runtime_state_version")
                        ),
                        expected_sequence=_optional_int(
                            context_bundle.get("runtime_state_sequence")
                        ),
                        fence_token=(
                            execution_fence.token if execution_fence is not None else ""
                        ),
                        fence_generation=(
                            execution_fence.generation if execution_fence is not None else 0
                        ),
                    ),
                    session=session,
                    user_id=user_id,
                )
            except Exception as exc:
                recovery["atomic_checkpoint"] = "failed"
                recovery["checkpoint_error"] = str(exc)
                diagnostics["recovery"] = recovery
                raise RuntimeError("atomic runtime checkpoint commit failed") from exc
            recovery["atomic_checkpoint"] = "settled" if terminal else "persisted"
            recovery["state_receipt"] = receipt.as_dict()
            diagnostics["recovery"] = recovery
            return
        if self.checkpoint_store is None:
            return
        try:
            self.checkpoint_store.save(
                run_id,
                durable_state,
                session=session,
                user_id=user_id,
            )
            recovery["durable_checkpoint"] = "persisted"
        except Exception as exc:
            recovery["durable_checkpoint"] = "failed"
            recovery["checkpoint_error"] = str(exc)
        diagnostics["recovery"] = recovery
