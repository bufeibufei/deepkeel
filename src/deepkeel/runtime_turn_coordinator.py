from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from deepkeel.async_ports import run_sync_adapter
from deepkeel.entrypoints import (
    _compose_entrypoint_prompt,
    _entrypoint_identity,
    _entrypoint_resolved_event,
    _validate_entrypoint_skill,
)
from deepkeel.events import AgentEventPersistenceError
from deepkeel.failures import classify_runtime_failure
from deepkeel.execution_engine import TurnExecutionEngine, create_langgraph_execution_engine
from deepkeel.guardrails import GuardrailAudit
from deepkeel.hooks import HookAudit
from deepkeel.langgraph_adapter import checkpointer_supports_async, compiler_checkpointer
from deepkeel.leases import ExecutionFence
from deepkeel.persistence import CheckpointCompatibilityError
from deepkeel.runtime_api import RuntimeRequest, RuntimeResult
from deepkeel.runtime_events import RuntimeEventEmitter
from deepkeel.runtime_execution_support import EventSink, emit_runtime_hook_audits
from deepkeel.runtime_graph_execution import execute_graph_turn
from deepkeel.runtime_input_guardrails import guard_runtime_input
from deepkeel.runtime_lifecycle import run_start_lifecycle_hooks
from deepkeel.runtime_model_pipeline import build_runtime_model_gateway
from deepkeel.runtime_policy import (
    _budget_limits,
    _max_elapsed_seconds,
    _merge_skill_activation,
    _prior_budget_state,
    _prior_diagnostics,
)
from deepkeel.runtime_results import _failed_runtime_state
from deepkeel.runtime_settlement import project_and_settle_runtime_result
from deepkeel.runtime_turn_preparation import (
    PreparedTurnIdentity,
    PreparedTurnInputs,
    prepare_turn_identity,
    prepare_turn_inputs,
)
from deepkeel.runtime_turn_events import (
    emit_input_guardrail_audits,
    emit_memory_recall_events,
)
from deepkeel.skills import SkillPolicy
from deepkeel.tools import ToolExecutionContext
from deepkeel.turn_context import TurnExecutionContext
from deepkeel.type_narrowing import as_dict


@dataclass(slots=True)
class _TurnState:
    inputs: PreparedTurnInputs
    identity: PreparedTurnIdentity
    short: dict[str, Any]
    bundle: dict[str, Any]
    model_policy: dict[str, Any]
    providers: dict[str, Any]
    skill: dict[str, Any]
    previous_diagnostics: dict[str, Any]
    context_window_diagnostics: dict[str, Any]
    deadline_monotonic: float | None
    emitter: RuntimeEventEmitter
    package_ids: tuple[str, ...]
    lifecycle_audits: list[HookAudit] = field(default_factory=list)


@dataclass(slots=True)
class _GraphRuntime:
    engine: TurnExecutionEngine
    model_gateway: Any
    tool_context: ToolExecutionContext
    turn_context: TurnExecutionContext


@dataclass(slots=True)
class _GraphOutcome:
    state: dict[str, Any]
    recovery_source: str
    active_graph_thread_id: str


class RuntimeTurnCoordinator:
    """Owns the typed lifecycle of one claimed runtime turn."""

    def __init__(
        self,
        runtime: Any,
        request: RuntimeRequest,
        *,
        provider: Any,
        providers: dict[str, Any] | None,
        session: Any,
        event_sink: EventSink | None,
        execution_fence: ExecutionFence | None,
    ) -> None:
        self.runtime = runtime
        self.request = request
        self.provider = provider
        self.providers = providers
        self.session = session
        self.event_sink = event_sink
        self.execution_fence = execution_fence
        self.started_monotonic = time.monotonic()
        self._input_guardrail_audits: tuple[GuardrailAudit, ...] = ()

    async def run(self) -> RuntimeResult:
        try:
            guarded = await guard_runtime_input(self.runtime.guardrail_runner, self.request)
        except Exception as exc:
            return await self._context_failure(
                exc,
                short=self._request_short(),
                bundle=self._request_bundle(),
            )
        self.request = guarded.request
        self._input_guardrail_audits = guarded.audits
        if guarded.error:
            return await self._context_failure(
                RuntimeError(guarded.error),
                short=self._request_short(),
                bundle=self._request_bundle(),
            )
        prepared = await self._prepare_inputs()
        if isinstance(prepared, RuntimeResult):
            return prepared
        identity = await self._prepare_identity(prepared)
        if isinstance(identity, RuntimeResult):
            return identity
        try:
            state = await self._initialize_state(prepared, identity)
        except Exception as exc:
            return await self._context_failure(exc, short=prepared.short, bundle=prepared.bundle)
        lifecycle_failure = await self._run_lifecycle(state)
        if lifecycle_failure is not None:
            return lifecycle_failure
        graph_runtime = self._build_graph_runtime(state)
        outcome = await self._execute_graph(state, graph_runtime)
        return await self._settle(state, outcome)

    async def _prepare_inputs(self) -> PreparedTurnInputs | RuntimeResult:
        try:
            return await prepare_turn_inputs(
                self.runtime,
                self.request,
                provider=self.provider,
                providers=self.providers,
            )
        except Exception as exc:
            return await self._context_failure(
                exc,
                short=self._request_short(),
                bundle=self._request_bundle(),
            )

    async def _prepare_identity(
        self,
        prepared: PreparedTurnInputs,
    ) -> PreparedTurnIdentity | RuntimeResult:
        try:
            return await prepare_turn_identity(
                self.runtime,
                self.request,
                prepared,
                session=self.session,
            )
        except CheckpointCompatibilityError as exc:
            return await self._context_failure(exc, short=prepared.short, bundle=prepared.bundle)

    async def _initialize_state(
        self,
        prepared: PreparedTurnInputs,
        identity: PreparedTurnIdentity,
    ) -> _TurnState:
        prior_budget = _prior_budget_state(identity.durable_state, prepared.short)
        await run_sync_adapter(
            self.runtime.budget_ledger.restore,
            identity.operational_run_id,
            prior_budget,
        )
        resolved_skill = _merge_skill_activation(
            durable_state=identity.durable_state,
            session_projection=prepared.short,
            explicit=self.request.skill_activation,
        )
        skill = SkillPolicy.from_snapshot(resolved_skill).runtime_snapshot()
        _validate_entrypoint_skill(prepared.capability_view, str(skill.get("skill_id") or ""))
        emitter = RuntimeEventEmitter(
            run_id=identity.run_id,
            thread_id=identity.conversation_thread_id,
            turn_id=identity.turn_id,
            run_version=identity.run_version,
            initial_sequence=identity.event_sequence,
            skill_id=str(skill.get("skill_id") or ""),
            scope=self.request.runtime_scope,
            run_control=self.runtime.run_control,
            telemetry=self.runtime.telemetry,
            event_journal=self.runtime.event_journal,
            async_event_journal=self.runtime.async_event_journal,
            event_sink=self.event_sink,
            execution_fence=self.execution_fence,
        )
        state = _TurnState(
            inputs=prepared,
            identity=identity,
            short=prepared.short,
            bundle=prepared.bundle,
            model_policy=prepared.resolved_model_policy,
            providers=prepared.model_providers,
            skill=skill,
            previous_diagnostics=_prior_diagnostics(
                identity.durable_state,
                prepared.short,
            ),
            context_window_diagnostics=dict(prepared.context_window_diagnostics),
            deadline_monotonic=self._deadline(prepared.resolved_model_policy),
            emitter=emitter,
            package_ids=prepared.capability_view.package_ids,
        )
        state.emitter(_entrypoint_resolved_event(prepared.capability_view))
        emit_input_guardrail_audits(state.emitter, self._input_guardrail_audits)
        emit_memory_recall_events(state.emitter, state.bundle, state.short)
        return state

    async def _run_lifecycle(self, state: _TurnState) -> RuntimeResult | None:
        started = await run_start_lifecycle_hooks(
            hook_runner=self.runtime.hook_runner,
            emit=state.emitter,
            emit_audits=emit_runtime_hook_audits,
            run_id=state.identity.run_id,
            thread_id=state.identity.conversation_thread_id,
            turn_id=state.identity.turn_id,
            package_ids=state.package_ids,
            skill_id=str(state.skill.get("skill_id") or ""),
            question=self.request.question,
            context_window=state.context_window_diagnostics,
            user_id=str(self.request.runtime_scope.user_id or "local-device"),
            resumed=bool(state.short.get("resume")),
        )
        state.lifecycle_audits.extend(started.audits)
        if started.context_patch:
            state.bundle.update(started.context_patch)
        if started.stopped_reason:
            return await self._context_failure(
                RuntimeError(started.stopped_reason),
                short=state.short,
                bundle=state.bundle,
            )
        if started.context_patch:
            self._reprepare_context(state)
        return None

    def _reprepare_context(self, state: _TurnState) -> None:
        state.bundle["_model_context_profile"] = state.inputs.model_context_profile.as_dict()
        state.bundle["_configured_input_limit"] = state.inputs.configured_input_limit
        state.bundle["_runtime_scope"] = self.request.runtime_scope.model_dump(mode="json")
        prepared = self.runtime.context_window_manager.prepare(
            self.request.question,
            state.short,
            state.bundle,
        )
        state.bundle = prepared.context_bundle
        state.context_window_diagnostics = dict(prepared.diagnostics)

    def _build_graph_runtime(self, state: _TurnState) -> _GraphRuntime:
        gateway = build_runtime_model_gateway(
            state.providers,
            router=self.runtime.model_router,
            policy_engine=self.runtime.policy_engine,
            budget_ledger=self.runtime.budget_ledger,
            invocation_recorder=self.runtime.model_invocation_recorder,
            invocation_store=self.runtime.model_invocation_store,
            model_health_store=self.runtime.model_health_store,
            context_compactor=getattr(self.runtime.context_window_manager, "compactor", None),
        )
        engine = self._resolve_execution_engine(state, gateway)
        tool_context = self._tool_context(state, gateway)
        system_prompt = _compose_entrypoint_prompt(
            self.runtime.system_prompt_factory(state.skill),
            state.inputs.capability_view,
        )
        turn_context = TurnExecutionContext(
            model=gateway,
            system_prompt=system_prompt,
            tool_context=tool_context,
            event_sink=state.emitter,
            deadline_monotonic=state.deadline_monotonic,
            tool_view_mode=self.runtime.tool_view_mode,
            hook_runner=self.runtime.hook_runner,
            guardrail_runner=self.runtime.guardrail_runner,
            entry_tool_skill_activator=self.runtime.entry_tool_skill_activator,
        )
        return _GraphRuntime(engine, gateway, tool_context, turn_context)

    def _resolve_execution_engine(self, state: _TurnState, gateway: Any) -> TurnExecutionEngine:
        if self.runtime.reuse_compiled_graph:
            return self.runtime._shared_execution_engine()
        return create_langgraph_execution_engine(
            model=gateway,
            tool_executor=self.runtime.tool_executor,
            tool_registry=self.runtime.tool_registry,
            system_prompt=self.runtime.system_prompt_factory(state.skill),
            max_steps=self.runtime.max_steps,
            checkpointer=compiler_checkpointer(self.runtime.checkpointer),
            supports_async_checkpointer=checkpointer_supports_async(self.runtime.checkpointer),
            budget_ledger=self.runtime.budget_ledger,
            deadline_monotonic=state.deadline_monotonic,
            run_control=self.runtime.run_control,
            durability=self.runtime.graph_durability,
        )

    def _tool_context(self, state: _TurnState, gateway: Any) -> ToolExecutionContext:
        scope = self.request.runtime_scope
        return ToolExecutionContext(
            run_id=state.identity.run_id,
            user_id=str(scope.user_id or "local-device"),
            thread_id=state.identity.conversation_thread_id,
            turn_id=state.identity.turn_id,
            session=self.session,
            session_factory=self.runtime.session_factory,
            context_bundle=state.bundle,
            metadata={
                "skill_activation": state.skill,
                "tenant_id": str(state.bundle.get("tenant_id") or ""),
                "governance_scope": {
                    "tenant_id": scope.tenant_id,
                    "user_id": str(scope.user_id or "local-device"),
                    "namespace": scope.namespace,
                    "skill_id": str(state.skill.get("skill_id") or ""),
                    "scopes": list(state.bundle.get("governance_scopes") or []),
                },
                "operational_run_id": state.identity.operational_run_id,
                "model_providers": state.providers,
                "model_policy": state.model_policy,
                "event_sink": state.emitter,
                "budget_ledger": self.runtime.budget_ledger,
                "capability_package_ids": list(state.package_ids),
                "agent_entrypoint": _entrypoint_identity(state.inputs.capability_view),
                "capability_view": state.inputs.capability_view.as_dict(),
            },
            budget_limits=_budget_limits(state.model_policy),
            deadline_monotonic=state.deadline_monotonic,
            run_control=self.runtime.run_control,
            execution_fence=self.execution_fence,
            scope=scope,
        )

    async def _execute_graph(
        self,
        state: _TurnState,
        graph_runtime: _GraphRuntime,
    ) -> _GraphOutcome:
        budget = await run_sync_adapter(
            self.runtime.budget_ledger.snapshot,
            state.identity.operational_run_id,
        )
        try:
            outcome = await execute_graph_turn(
                engine=graph_runtime.engine,
                question=self.request.question,
                run_id=state.identity.run_id,
                graph_thread_id=state.identity.graph_thread_id,
                turn_id=state.identity.turn_id,
                user_id=str(self.request.runtime_scope.user_id or "local-device"),
                short_context=state.short,
                context_bundle=state.bundle,
                skill_activation=state.skill,
                model_policy=state.model_policy,
                budget_state=budget.as_dict(),
                input_parts=list(self.request.input_parts),
                durable_state=state.identity.durable_state,
                state_migrations=self.runtime.state_migrations,
                tool_registry=self.runtime.tool_registry,
                tool_context=graph_runtime.tool_context,
                turn_context=graph_runtime.turn_context,
                emit=state.emitter,
                has_graph_checkpoint=self.runtime._has_graph_checkpoint,
            )
            return _GraphOutcome(
                state=dict(outcome.state),
                recovery_source=outcome.recovery_source,
                active_graph_thread_id=outcome.active_graph_thread_id,
            )
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            return await self._failed_graph_outcome(state, exc)

    async def _failed_graph_outcome(
        self,
        state: _TurnState,
        exc: Exception,
    ) -> _GraphOutcome:
        failure = classify_runtime_failure(exc)
        if failure.category != "canceled":
            state.emitter(
                {
                    "event_type": "agent.failed",
                    "title": "Agent run failed",
                    "summary": failure.user_message,
                    "payload": {
                        "error": failure.detail,
                        "error_type": failure.exception_type,
                        "error_code": failure.code,
                        "failure": failure.as_dict(),
                    },
                }
            )
        budget = await run_sync_adapter(
            self.runtime.budget_ledger.snapshot,
            state.identity.operational_run_id,
        )
        failed_state = _failed_runtime_state(
            self.request.question,
            exc,
            run_id=state.identity.run_id,
            thread_id=state.identity.graph_thread_id,
            turn_id=state.identity.turn_id,
            user_id=str(self.request.runtime_scope.user_id or "local-device"),
            short_context=state.short,
            context_bundle=state.bundle,
            skill_activation=state.skill,
            model_policy=state.model_policy,
            budget_state=budget.as_dict(),
            events=state.emitter.events,
            failure=failure,
        )
        return _GraphOutcome(
            state=dict(failed_state),
            recovery_source="",
            active_graph_thread_id=state.identity.graph_thread_id,
        )

    async def _settle(self, state: _TurnState, outcome: _GraphOutcome) -> RuntimeResult:
        identity = state.identity
        return await project_and_settle_runtime_result(
            runtime=self.runtime,
            state=outcome.state,
            question=self.request.question,
            context_bundle=state.bundle,
            short_context=state.short,
            skill_activation=state.skill,
            model_policy=state.model_policy,
            previous_diagnostics=state.previous_diagnostics,
            context_window_diagnostics=state.context_window_diagnostics,
            emitter=state.emitter,
            lifecycle_audits=state.lifecycle_audits,
            package_ids=state.package_ids,
            recovery_source=outcome.recovery_source,
            checkpoint_authority=identity.checkpoint_authority,
            checkpoint_load_errors=identity.checkpoint_load_errors,
            run_started_monotonic=self.started_monotonic,
            run_id=identity.run_id,
            conversation_thread_id=identity.conversation_thread_id,
            turn_id=identity.turn_id,
            user_id=str(self.request.runtime_scope.user_id or "local-device"),
            runtime_scope=self.request.runtime_scope,
            active_graph_thread_id=outcome.active_graph_thread_id,
            session=self.session,
            execution_fence=self.execution_fence,
            emit_hook_audits=emit_runtime_hook_audits,
        )

    async def _context_failure(
        self,
        exc: Exception,
        *,
        short: dict[str, Any],
        bundle: dict[str, Any],
    ) -> RuntimeResult:
        return await self.runtime._context_setup_failure(
            self.request.question,
            exc,
            short_context=short,
            context_bundle=bundle,
            user_id=self.request.runtime_scope.user_id,
            skill_activation=self.request.skill_activation,
            model_policy=self.request.model_policy,
            session=self.session,
            event_sink=self.event_sink,
            execution_fence=self.execution_fence,
            scope=self.request.runtime_scope,
        )

    def _request_short(self) -> dict[str, Any]:
        value = self.request.short_context
        return value if isinstance(value, dict) else {}

    def _request_bundle(self) -> dict[str, Any]:
        value = self.request.context_bundle
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _deadline(model_policy: dict[str, Any]) -> float | None:
        maximum = _max_elapsed_seconds(model_policy)
        return time.monotonic() + maximum if maximum > 0 else None
