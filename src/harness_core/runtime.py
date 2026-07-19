from __future__ import annotations

import hashlib
import json
import time
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
from harness_core.contracts import RunContext, RunStatus, ToolCall
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
from harness_core.model import ModelProviderAdapter, RoutedModelGateway
from harness_core.model_routing import AdaptiveStepModelRouter, ModelRouter
from harness_core.persistence import (
    CheckpointStore,
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
from harness_core.skills import SkillPolicy
from harness_core.state_store import RuntimeStateMutation, RuntimeStateStore
from harness_core.telemetry import NoopTelemetry, TelemetryPort, TelemetryRecord
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutionContext, ToolExecutor
from harness_core.ui import project_run_ui_state
from harness_core.version import HARNESS_CORE_CONTRACT_VERSION, HARNESS_CORE_VERSION


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
        checkpoint_store: CheckpointStore | None = None,
        system_prompt_factory: SystemPromptFactory | None = None,
        session_factory: SessionFactory | None = None,
        max_steps: int = 12,
        model_router: ModelRouter | None = None,
        policy_engine: PolicyEngine | None = None,
        budget_ledger: BudgetLedger | None = None,
        run_control: RunControl | None = None,
        capability_contributions: tuple[CapabilityContribution, ...] = (),
        capability_catalog: CapabilityCatalog | None = None,
        telemetry: TelemetryPort | None = None,
        context_builder: ContextBuilder | None = None,
        context_window_manager: ContextWindowManager | None = None,
        runtime_state_store: RuntimeStateStore | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.checkpointer = checkpointer or LangGraphCheckpointerAdapter()
        self.checkpoint_store = checkpoint_store
        self.system_prompt_factory = system_prompt_factory or _default_system_prompt_factory
        self.session_factory = session_factory
        self.max_steps = max(2, int(max_steps))
        self.model_router = model_router or AdaptiveStepModelRouter()
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
        self.tool_executor.policy_engine = self.policy_engine
        self.tool_executor.budget_ledger = self.budget_ledger

    def close(self) -> None:
        self.capability_catalog.close()

    @staticmethod
    def supports_native_tools(provider: Any) -> bool:
        return isinstance(provider, ModelProviderAdapter) or callable(
            getattr(provider, "stream_chat", None)
        ) or callable(getattr(provider, "complete_chat", None))

    def run_turn(
        self,
        question: str,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
        user_id: str = "",
        short_context: dict[str, Any] | None = None,
        context_bundle: dict[str, Any] | None = None,
        skill_activation: dict[str, Any] | None = None,
        model_policy: dict[str, Any] | None = None,
        event_sink: EventSink | None = None,
    ) -> dict[str, Any]:
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
                event_sink=event_sink,
            )
        run_id = str(bundle.get("agent_session_id") or bundle.get("agent_run_id") or bundle.get("run_id") or uuid4())
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
            bundle.get("thread_id") or bundle.get("ask_thread_id") or short.get("ask_thread_id") or run_id
        )
        graph_thread_id = run_id
        turn_id = str(bundle.get("turn_id") or short.get("turn_id") or f"turn-{uuid4()}")
        events: list[dict[str, Any]] = []
        telemetry_error_count = 0
        telemetry_last_error = ""
        answer_delta_streamed = False

        def emit(event: dict[str, Any]) -> None:
            nonlocal answer_delta_streamed, telemetry_error_count, telemetry_last_error
            self.run_control.raise_if_cancelled(run_id)
            projected = project_runtime_event(event)
            payload = (
                dict(projected.get("payload"))
                if isinstance(projected.get("payload"), dict)
                else {}
            )
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
        )

        recovery_source = ""
        active_graph_thread_id = graph_thread_id
        try:
            if short.get("recover_interrupted"):
                if self._has_graph_checkpoint(graph_thread_id):
                    state = graph.recover(
                        graph_thread_id,
                        tool_context=tool_context,
                        event_sink=emit,
                    )
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
                    state = graph.invoke(
                        context,
                        tool_context=tool_context,
                        event_sink=emit,
                    )
                    recovery_source = "restart_replay_without_checkpoint"
            elif short.get("resume"):
                resume_payload = resume_payload_from_context(short)
                try:
                    state = graph.resume(graph_thread_id, resume_payload, tool_context=tool_context, event_sink=emit)
                    if not isinstance(state, dict):
                        raise RuntimeError("langgraph checkpoint is unavailable")
                    recovery_source = "live_langgraph"
                except AgentEventPersistenceError:
                    raise
                except (RuntimeError, ValueError, AttributeError):
                    previous_runtime = short.get("previous_agent_runtime")
                    recovered_checkpoint = checkpoint_from_durable_state(durable_state)
                    if not recovered_checkpoint:
                        recovered_checkpoint = checkpoint_from_runtime(
                            previous_runtime if isinstance(previous_runtime, dict) else {}
                        )
                    if not recovered_checkpoint:
                        raise RuntimeError("durable checkpoint is unavailable")
                    recovery_source = "agent_run_checkpoint" if checkpoint_from_durable_state(durable_state) else "session_projection"
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
                    )
                    if not any(message.role == "user" for message in recovered.messages):
                        recovered.messages = [*build_initial_messages(question, short, bundle), *recovered.messages]
                    active_graph_thread_id = recovered_thread_id
                    state = graph.invoke(recovered, tool_context=tool_context, event_sink=emit)
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
                state = graph.invoke(context, tool_context=tool_context, event_sink=emit)
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            failure = classify_runtime_failure(exc)
            if failure.category != "canceled":
                emit({
                    "event_type": "agent.failed",
                    "title": "Agent 运行失败",
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
            answer_delta_streamed=answer_delta_streamed,
            observation_kinds={
                spec.name: str(spec.observation_contract.get("primary_kind") or "")
                for spec in self.tool_registry.list_tools()
            },
            task_kinds={spec.name: spec.task_kind for spec in self.tool_registry.list_tools()},
            max_steps=self.max_steps,
            previous_diagnostics=previous_diagnostics,
            capability_manifest=self._capability_manifest(),
        )
        if recovery_source:
            runtime_state = result.get("agent_runtime") if isinstance(result.get("agent_runtime"), dict) else {}
            diagnostics = runtime_state.get("diagnostics") if isinstance(runtime_state.get("diagnostics"), dict) else {}
            recovery = diagnostics.get("recovery") if isinstance(diagnostics.get("recovery"), dict) else {}
            recovery["checkpoint_source"] = recovery_source
            diagnostics["recovery"] = recovery
            runtime_state["diagnostics"] = diagnostics
        runtime_state = (
            result.get("agent_runtime")
            if isinstance(result.get("agent_runtime"), dict)
            else {}
        )
        diagnostics = (
            runtime_state.get("diagnostics")
            if isinstance(runtime_state.get("diagnostics"), dict)
            else {}
        )
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
        checkpoint = (
            runtime_state.get("checkpoint")
            if isinstance(runtime_state.get("checkpoint"), dict)
            else None
        )
        if checkpoint is not None:
            checkpoint["budget_state"] = self.budget_ledger.snapshot(run_id).as_dict()
        runtime_state["diagnostics"] = diagnostics
        try:
            self.telemetry.record(
                TelemetryRecord(
                    event_name="runtime.settled",
                    run_id=run_id,
                    thread_id=conversation_thread_id,
                    turn_id=turn_id,
                    status=str(runtime_state.get("status") or ""),
                    attributes={
                        "status": str(runtime_state.get("status") or ""),
                        "stop_reason": str(runtime_state.get("stop_reason") or ""),
                        "skill_id": str(skill.get("skill_id") or ""),
                        "recovery_source": recovery_source,
                    },
                )
            )
        except Exception as exc:
            telemetry_error_count += 1
            telemetry_last_error = f"{type(exc).__name__}: {exc}"
        if telemetry_error_count:
            runtime_state = (
                result.get("agent_runtime")
                if isinstance(result.get("agent_runtime"), dict)
                else {}
            )
            diagnostics = (
                runtime_state.get("diagnostics")
                if isinstance(runtime_state.get("diagnostics"), dict)
                else {}
            )
            diagnostics["telemetry"] = {
                "status": "degraded",
                "error_count": telemetry_error_count,
                "last_error": telemetry_last_error,
            }
            runtime_state["diagnostics"] = diagnostics
        if str((result.get("agent_runtime") or {}).get("status") or "") in {
            "completed",
            "failed",
            "canceled",
        }:
            self.cleanup_run(
                result,
                run_id=run_id,
                session=session,
                user_id=str(user_id or "local-device"),
                graph_thread_ids=[active_graph_thread_id],
            )
        else:
            self._persist_resumable_checkpoint(
                result,
                run_id=run_id,
                thread_id=conversation_thread_id,
                session=session,
                user_id=str(user_id or "local-device"),
                context_bundle=bundle,
            )
        return result

    def cleanup_run(
        self,
        result: dict[str, Any] | None,
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
        if isinstance(result, dict):
            runtime = result.get("agent_runtime") if isinstance(result.get("agent_runtime"), dict) else {}
            diagnostics = runtime.get("diagnostics") if isinstance(runtime.get("diagnostics"), dict) else {}
            recovery = diagnostics.get("recovery") if isinstance(diagnostics.get("recovery"), dict) else {}
            recovery["checkpoint_cleanup"] = cleanup
            if errors:
                recovery["checkpoint_cleanup_errors"] = errors
            diagnostics["recovery"] = recovery
            runtime["diagnostics"] = diagnostics
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
        event_sink: EventSink | None,
    ) -> dict[str, Any]:
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
                "title": "Agent 上下文准备失败",
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
                        "stop_reason": str(
                            (result.get("agent_runtime") or {}).get("stop_reason") or ""
                        ),
                        "skill_id": str(resolved_skill.get("skill_id") or ""),
                        "recovery_source": "",
                        "phase": "context_setup",
                    },
                )
            )
        except Exception:
            pass
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
                "mcp_servers": len(self.capability_catalog.mcp_servers),
                "subagents": len(self.capability_catalog.subagents),
                "context_contributors": len(
                    self.capability_catalog.context_contributors
                ),
                "resources": len(self.capability_catalog.resources),
            },
        }

    def _persist_resumable_checkpoint(
        self,
        result: dict[str, Any],
        *,
        run_id: str,
        thread_id: str,
        session: Any,
        user_id: str,
        context_bundle: dict[str, Any],
    ) -> None:
        runtime = result.get("agent_runtime") if isinstance(result.get("agent_runtime"), dict) else {}
        if runtime.get("status") not in {
            "waiting_user_action",
            "waiting_user_input",
            "task_running",
        }:
            return
        diagnostics = runtime.get("diagnostics") if isinstance(runtime.get("diagnostics"), dict) else {}
        recovery = diagnostics.get("recovery") if isinstance(diagnostics.get("recovery"), dict) else {}
        durable_state = durable_state_from_result(
            result,
            run_id=run_id,
            thread_id=thread_id,
        )
        if self.runtime_state_store is not None:
            status = str(runtime.get("status") or "")
            resume_token = str(runtime.get("resume_token") or "")
            mutation_payload = {
                "status": status,
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
                        event_type="runtime.checkpoint.committed",
                        target_status=status,
                        event_payload=mutation_payload,
                        checkpoint_state=durable_state,
                        resume_token=resume_token,
                        expected_version=_optional_int(
                            context_bundle.get("runtime_state_version")
                        ),
                        expected_sequence=_optional_int(
                            context_bundle.get("runtime_state_sequence")
                        ),
                    ),
                    session=session,
                    user_id=user_id,
                )
            except Exception as exc:
                recovery["atomic_checkpoint"] = "failed"
                recovery["checkpoint_error"] = str(exc)
                diagnostics["recovery"] = recovery
                runtime["diagnostics"] = diagnostics
                raise RuntimeError("atomic runtime checkpoint commit failed") from exc
            recovery["atomic_checkpoint"] = "persisted"
            recovery["state_receipt"] = receipt.as_dict()
            diagnostics["recovery"] = recovery
            runtime["diagnostics"] = diagnostics
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
        runtime["diagnostics"] = diagnostics


def _resolved_model_policy(
    value: dict[str, Any] | None,
    *,
    provider: Any,
    providers: dict[str, Any] | None,
    max_steps: int,
) -> dict[str, Any]:
    policy = dict(value) if isinstance(value, dict) else {}
    catalog = _model_providers(provider, providers, policy)
    if policy.get("mode") not in {"single", "adaptive"}:
        policy["mode"] = "adaptive" if len(catalog) > 1 else "single"
    primary = str(policy.get("primary_role") or "reasoning")
    if primary not in catalog:
        primary = "reasoning" if "reasoning" in catalog else next(iter(catalog), "reasoning")
    policy["primary_role"] = primary
    policy["available_roles"] = list(catalog)
    budget = policy.get("budget") if isinstance(policy.get("budget"), dict) else {}
    policy["budget"] = {
        "max_model_calls": _positive_limit(budget.get("max_model_calls"), max_steps),
        "max_tool_calls": _positive_limit(budget.get("max_tool_calls"), 0),
        "max_elapsed_seconds": _positive_number(
            budget.get("max_elapsed_seconds"),
            900.0,
        ),
        "max_total_elapsed_seconds": _positive_number(
            budget.get("max_total_elapsed_seconds"),
            0,
        ),
        "max_request_seconds": _positive_number(
            budget.get("max_request_seconds"),
            0,
        ),
        "max_input_tokens_total": _positive_limit(
            budget.get("max_input_tokens_total"),
            0,
        ),
        "max_input_tokens_per_call": _positive_limit(
            budget.get("max_input_tokens_per_call"),
            0,
        ),
        "max_output_tokens_total": _positive_limit(
            budget.get("max_output_tokens_total"),
            0,
        ),
        "max_output_tokens_per_call": _positive_limit(
            budget.get("max_output_tokens_per_call"),
            0,
        ),
        "max_model_retries": _positive_limit(
            budget.get("max_model_retries"),
            0,
        ),
        "max_parallel_tools": _positive_limit(
            budget.get("max_parallel_tools"),
            4,
        ),
        "roles": {
            str(role): dict(limits)
            for role, limits in (
                budget.get("roles")
                if isinstance(budget.get("roles"), dict)
                else {}
            ).items()
            if isinstance(limits, dict)
        },
    }
    return policy


def _model_providers(
    provider: Any,
    providers: dict[str, Any] | None,
    model_policy: dict[str, Any],
) -> dict[str, Any]:
    catalog = {
        str(role): candidate
        for role, candidate in (providers or {}).items()
        if candidate is not None
    }
    if provider is not None:
        adapter_role = (
            provider.info.model_role if isinstance(provider, ModelProviderAdapter) else ""
        )
        role = str(
            adapter_role
            or getattr(provider, "model_role", "")
            or model_policy.get("primary_role")
            or "reasoning"
        )
        catalog.setdefault(role, provider)
        catalog.setdefault("reasoning", provider)
    return catalog


def _budget_limits(model_policy: dict[str, Any]) -> dict[str, float]:
    budget = model_policy.get("budget") if isinstance(model_policy.get("budget"), dict) else {}
    return {
        MODEL_CALLS: float(budget.get("max_model_calls") or 0),
        TOOL_CALLS: float(budget.get("max_tool_calls") or 0),
        INPUT_TOKENS: float(budget.get("max_input_tokens_total") or 0),
        OUTPUT_TOKENS: float(budget.get("max_output_tokens_total") or 0),
        MODEL_RETRIES: float(budget.get("max_model_retries") or 0),
        TOOL_CONCURRENCY: float(budget.get("max_parallel_tools") or 4),
        ELAPSED_SECONDS: float(budget.get("max_total_elapsed_seconds") or 0),
    }


def _positive_limit(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _positive_number(value: Any, default: float) -> float:
    if value is None:
        return max(0.0, float(default))
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, float(default))


def _max_elapsed_seconds(model_policy: dict[str, Any]) -> float:
    budget = model_policy.get("budget") if isinstance(model_policy.get("budget"), dict) else {}
    return _positive_number(budget.get("max_elapsed_seconds"), 900.0)


def _prior_budget_state(
    durable_state: dict[str, Any],
    short_context: dict[str, Any],
) -> dict[str, Any]:
    durable_runtime = (
        durable_state.get("agent_runtime")
        if isinstance(durable_state.get("agent_runtime"), dict)
        else {}
    )
    previous_runtime = (
        short_context.get("previous_agent_runtime")
        if isinstance(short_context.get("previous_agent_runtime"), dict)
        else {}
    )
    sources = (
        durable_state,
        durable_runtime.get("checkpoint"),
        previous_runtime.get("checkpoint"),
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        snapshot = source.get("budget_state")
        if isinstance(snapshot, dict):
            return snapshot
    return {}


def _merge_skill_activation(
    *,
    durable_state: dict[str, Any],
    session_projection: dict[str, Any],
    explicit: dict[str, Any] | None,
) -> dict[str, Any]:
    durable_context = (
        durable_state.get("context_snapshot")
        if isinstance(durable_state.get("context_snapshot"), dict)
        else {}
    )
    projected_context = (
        session_projection.get("context_snapshot")
        if isinstance(session_projection.get("context_snapshot"), dict)
        else {}
    )
    sources = (
        durable_context.get("skill_activation"),
        durable_state.get("skill_activation"),
        projected_context.get("skill_activation"),
        session_projection.get("skill_activation"),
        explicit,
    )
    merged: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _prior_diagnostics(
    durable_state: dict[str, Any],
    short_context: dict[str, Any],
) -> dict[str, Any]:
    durable_runtime = (
        durable_state.get("agent_runtime")
        if isinstance(durable_state.get("agent_runtime"), dict)
        else {}
    )
    previous_runtime = (
        short_context.get("previous_agent_runtime")
        if isinstance(short_context.get("previous_agent_runtime"), dict)
        else {}
    )
    for runtime in (durable_runtime, previous_runtime):
        diagnostics = runtime.get("diagnostics") if isinstance(runtime.get("diagnostics"), dict) else {}
        if diagnostics:
            return diagnostics
    return {}


def project_harness_result(
    state: dict[str, Any],
    *,
    question: str,
    context_bundle: dict[str, Any],
    short_context: dict[str, Any],
    skill_activation: dict[str, Any],
    streamed_events: list[dict[str, Any]],
    answer_delta_streamed: bool = False,
    observation_kinds: dict[str, str] | None = None,
    task_kinds: dict[str, str] | None = None,
    max_steps: int = 12,
    previous_diagnostics: dict[str, Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_status = str(state.get("status") or "failed")
    state_error = (
        (state.get("metadata") or {}).get("runtime_error")
        if isinstance(state.get("metadata"), dict)
        else None
    )
    skill = dict(skill_activation) if isinstance(skill_activation, dict) else {}
    state_skill = state.get("skill_activation") if isinstance(state.get("skill_activation"), dict) else {}
    skill.update(state_skill)
    checkpoint_pending_action = (
        state.get("pending_action") if isinstance(state.get("pending_action"), dict) else None
    )
    waiting_for_input = (
        graph_status == "waiting_user"
        and str((checkpoint_pending_action or {}).get("action_type") or "") == "clarification"
    )
    runtime_status = {
        "completed": "completed",
        "waiting_user": "waiting_user_input" if waiting_for_input else "waiting_user_action",
        "waiting_async": "task_running",
        "failed": "failed",
        "canceled": "canceled",
    }.get(graph_status, "running")
    stop_reason = {
        "completed": "final_answer",
        "waiting_user": "requires_user_input" if waiting_for_input else "requires_user_action",
        "waiting_async": "task_running",
        "failed": "runtime_failed",
        "canceled": "canceled",
    }.get(graph_status, "running")
    if graph_status == "failed" and isinstance(state_error, dict) and state_error.get("code"):
        stop_reason = str(state_error["code"]).lower()
    typed_tool_results = [item for item in state.get("tool_results", []) if isinstance(item, dict)]
    checkpoint_pending_async = (
        state.get("pending_async") if isinstance(state.get("pending_async"), dict) else None
    )
    pending_action = _project_pending_action(checkpoint_pending_action, typed_tool_results)
    legacy_results = [_legacy_tool_result(item) for item in typed_tool_results]
    legacy_results = [item for item in legacy_results if item]
    final_answer = _project_final_answer(state, runtime_status, pending_action, legacy_results)
    references = _project_answer_references(typed_tool_results, final_answer)
    final_answer["references"] = references
    literature_evidence = [item for item in references if item.get("kind") == "literature"]
    if literature_evidence and not final_answer.get("evidence"):
        final_answer["evidence"] = literature_evidence
    checkpoint_observations = [
        item for item in state.get("observations", []) if isinstance(item, dict)
    ]
    projected_observations = [
        _project_observation(item, observation_kinds or {}) for item in checkpoint_observations
    ]
    checkpoint = {
        "schema_version": "harness-checkpoint-v1",
        "run_id": str(state.get("run_id") or ""),
        "graph_thread_id": str(state.get("thread_id") or ""),
        "turn_id": str(state.get("turn_id") or ""),
        "messages": [item for item in state.get("messages", []) if isinstance(item, dict)],
        "observations": checkpoint_observations,
        "artifacts": [item for item in state.get("artifacts", []) if isinstance(item, dict)],
        "pending_action": checkpoint_pending_action,
        "pending_async": checkpoint_pending_async,
        "pending_tool_calls": [
            item for item in state.get("pending_tool_calls", []) if isinstance(item, dict)
        ],
        "budget_state": state.get("budget_state")
        if isinstance(state.get("budget_state"), dict)
        else {},
        "step_count": int(state.get("step_count") or 0),
        "status": graph_status,
        "metadata": state.get("metadata") if isinstance(state.get("metadata"), dict) else {},
    }
    trace = _trace_from_events(streamed_events, observation_kinds=observation_kinds or {})
    context_snapshot = build_context_snapshot(question, context_bundle, short_context, skill)
    runtime = {
        "schema_version": "harness-runtime-v1",
        "core_contract_version": HARNESS_CORE_CONTRACT_VERSION,
        "core_version": HARNESS_CORE_VERSION,
        "loop_engine": "langgraph_native_tools",
        "mode": "react_loop",
        "status": runtime_status,
        "stop_reason": stop_reason,
        "step_count": int(state.get("step_count") or 0),
        "observations": projected_observations,
        "artifacts": checkpoint["artifacts"],
        "pending_action": pending_action,
        "checkpoint": checkpoint,
        "trace": trace,
        "answer_delta_streamed": answer_delta_streamed,
    }
    runtime["diagnostics"] = _runtime_diagnostics(
        state,
        runtime,
        streamed_events,
        context_snapshot=context_snapshot,
        skill_activation=skill,
        max_steps=max_steps,
        previous_diagnostics=previous_diagnostics,
        capability_manifest=capability_manifest,
    )
    active_task = _active_task(legacy_results, typed_tool_results, task_kinds or {})
    if active_task:
        runtime["active_task"] = active_task
    runtime["ui_state"] = project_run_ui_state(
        runtime_status,
        pending_action=pending_action,
        active_task=active_task,
    )
    result = {
        "question": question,
        "final_answer": final_answer,
        "agent_runtime": runtime,
        "agent_tool_results": legacy_results,
        "events": streamed_events,
        "context_snapshot": context_snapshot,
        "skill_activation": skill,
        "references": references,
        "evidence": literature_evidence,
        "needs_user_input": runtime_status in {"waiting_user_action", "waiting_user_input"},
        "answer_delta_streamed": answer_delta_streamed,
    }
    if answer_delta_streamed:
        final_answer["answer_delta_streamed"] = True
    if isinstance(state_error, dict):
        result["error"] = {
            "type": str(state_error.get("type") or "RuntimeFailure"),
            "code": str(state_error.get("code") or "RUNTIME_INTERNAL_ERROR"),
            "category": str(state_error.get("category") or "internal"),
            "retryable": bool(state_error.get("retryable")),
            "message": str(
                state_error.get("user_message")
                or "The run ended safely and can be retried."
            ),
        }
    if pending_action is not None:
        result["pending_action"] = pending_action
    return result


def _project_answer_references(
    typed_tool_results: list[dict[str, Any]],
    final_answer: dict[str, Any],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in final_answer.get("references") or []:
        if isinstance(item, dict):
            candidates.append(_normalize_reference(item, source_tool="agent.final"))
    for tool_result in typed_tool_results:
        tool_name = str(tool_result.get("name") or "")
        data = tool_result.get("data") if isinstance(tool_result.get("data"), dict) else {}
        query = str(data.get("query") or "")
        if tool_name == "web.search":
            for reference in data.get("results") or []:
                if isinstance(reference, dict):
                    candidates.append(
                        _normalize_reference(
                            reference,
                            kind="web",
                            source_tool=tool_name,
                            query=query,
                        )
                    )
        for reference in _nested_reference_candidates(data):
            candidates.append(
                _normalize_reference(
                    reference,
                    source_tool=tool_name,
                    query=query,
                )
            )
        for artifact in tool_result.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            artifact_data = artifact.get("data") if isinstance(artifact.get("data"), dict) else {}
            for reference in _nested_reference_candidates(artifact_data):
                candidates.append(
                    _normalize_reference(
                        reference,
                        source_tool=tool_name,
                        query=query,
                    )
                )

    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        identity = str(
            candidate.get("unit_id")
            or candidate.get("url")
            or candidate.get("reference_id")
            or f"{candidate.get('kind')}:{candidate.get('title')}:{candidate.get('snippet')}"
        )
        if identity in seen:
            continue
        seen.add(identity)
        references.append(candidate)
        if len(references) >= limit:
            break
    return references


def _nested_reference_candidates(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 5:
        return []
    if isinstance(value, list):
        candidates: list[dict[str, Any]] = []
        for item in value:
            candidates.extend(_nested_reference_candidates(item, depth=depth + 1))
        return candidates
    if not isinstance(value, dict):
        return []
    candidates = []
    for key, item in value.items():
        if key in {"citations", "evidence", "evidence_refs", "evidence_ledger", "references"} and isinstance(item, list):
            candidates.extend(reference for reference in item if isinstance(reference, dict))
            continue
        if isinstance(item, (dict, list)):
            candidates.extend(_nested_reference_candidates(item, depth=depth + 1))
    return candidates


def _normalize_reference(
    value: dict[str, Any],
    *,
    kind: str = "",
    source_tool: str = "",
    query: str = "",
) -> dict[str, Any]:
    unit_id = str(value.get("unit_id") or value.get("id") or "").strip()
    url = str(value.get("url") or "").strip()
    resolved_kind = kind or str(value.get("kind") or ("web" if url else "literature" if unit_id else ""))
    title = str(value.get("title") or value.get("title_cn") or value.get("site_name") or "").strip()
    snippet = str(
        value.get("snippet")
        or value.get("summary")
        or value.get("text_preview")
        or ""
    ).strip()[:360]
    if not resolved_kind or (not unit_id and not url and not title):
        return {}
    reference_id = unit_id or url or f"{resolved_kind}:{title}:{snippet[:80]}"
    normalized = {
        "reference_id": reference_id,
        "kind": resolved_kind,
        "title": title or ("网页来源" if resolved_kind == "web" else "典籍依据"),
        "snippet": snippet,
        "source_tool": source_tool or str(value.get("source_tool") or ""),
        "query": query or str(value.get("query") or ""),
    }
    optional = {
        "unit_id": unit_id,
        "url": url,
        "site_name": str(value.get("site_name") or ""),
        "publish_time": str(value.get("publish_time") or ""),
        "flow": str(value.get("flow") or ""),
        "quality_grade": str(value.get("quality_grade") or value.get("grade") or ""),
        "page": str(value.get("page") or ""),
        "chapter": str(value.get("chapter") or ""),
        "source_file": str(value.get("source_file") or ""),
    }
    normalized.update({key: item for key, item in optional.items() if item})
    return normalized


def _failed_runtime_state(
    question: str,
    exc: Exception,
    *,
    run_id: str,
    thread_id: str,
    turn_id: str,
    user_id: str,
    short_context: dict[str, Any],
    context_bundle: dict[str, Any],
    skill_activation: dict[str, Any],
    model_policy: dict[str, Any],
    budget_state: dict[str, Any],
    events: list[dict[str, Any]],
    failure: RuntimeFailure | None = None,
    phase: str = "runtime",
) -> dict[str, Any]:
    failure = failure or classify_runtime_failure(exc)
    context = _new_context(
        question,
        run_id=run_id,
        thread_id=thread_id,
        turn_id=turn_id,
        user_id=user_id,
        short_context=short_context,
        context_bundle=context_bundle,
        skill_activation=skill_activation,
        model_policy=model_policy,
        budget_state=budget_state,
    )
    state = context.model_dump(mode="json")
    state.update(
        {
            "status": (
                RunStatus.CANCELED.value
                if failure.category == "canceled"
                else RunStatus.FAILED.value
            ),
            "step_count": sum(
                1 for event in events if event.get("event_type") in {"model.completed", "model.failed"}
            ),
            "tool_results": [],
            "pending_tool_calls": [],
            "pending_action": None,
            "final_answer": {
                "answer_mode": "bubble",
                "summary": failure.user_message,
                "markdown": failure.user_message,
                "status": "canceled" if failure.category == "canceled" else "failed",
                "stop_reason": (
                    "user_cancelled"
                    if failure.category == "canceled"
                    else failure.code.lower()
                ),
                "metadata": {
                    "error_code": failure.code,
                    "failure_category": failure.category,
                    "retryable": failure.retryable,
                },
            },
            "metadata": {
                **(state.get("metadata") if isinstance(state.get("metadata"), dict) else {}),
                "runtime_error": {
                    "type": failure.exception_type,
                    "code": failure.code,
                    "category": failure.category,
                    "retryable": failure.retryable,
                    "phase": phase,
                    "message": failure.detail,
                    "user_message": failure.user_message,
                },
            },
        }
    )
    return state


def _new_context(
    question: str,
    *,
    run_id: str,
    thread_id: str,
    turn_id: str,
    user_id: str,
    short_context: dict[str, Any],
    context_bundle: dict[str, Any],
    skill_activation: dict[str, Any],
    model_policy: dict[str, Any],
    budget_state: dict[str, Any],
    pending_tool_calls: list[ToolCall] | None = None,
) -> RunContext:
    initial_calls = list(pending_tool_calls or [])
    return RunContext(
        run_id=run_id,
        thread_id=thread_id,
        turn_id=turn_id,
        user_id=user_id,
        status=RunStatus.EXECUTING_TOOLS if initial_calls else RunStatus.PREPARING,
        messages=build_initial_messages(question, short_context, context_bundle),
        pending_tool_calls=initial_calls,
        skill_activation=skill_activation,
        model_policy=model_policy,
        budget_state=budget_state,
        metadata={
            "context_source": str(context_bundle.get("source") or "runtime"),
            "governance_scope": {
                "tenant_id": str(context_bundle.get("tenant_id") or ""),
                "user_id": user_id,
                "skill_id": str(skill_activation.get("skill_id") or ""),
                "scopes": list(context_bundle.get("governance_scopes") or []),
            },
        },
    )


def _skill_precondition_tool_calls(
    skill: dict[str, Any],
    *,
    context: dict[str, Any],
    tool_registry: ToolRegistry,
) -> list[ToolCall]:
    """Translate declarative missing Skill preconditions into one safe handoff call."""
    preconditions = skill.get("preconditions") if isinstance(skill.get("preconditions"), list) else []
    for item in preconditions:
        if not isinstance(item, dict):
            continue
        paths = item.get("context_any_of") if isinstance(item.get("context_any_of"), list) else []
        if not paths or any(_context_path_value(context, str(path)) for path in paths):
            continue
        tool_name = str(item.get("tool_name") or "").strip()
        if not tool_name:
            continue
        try:
            spec = tool_registry.get(tool_name)
        except KeyError:
            continue
        precondition_id = str(item.get("id") or tool_name).strip()
        arguments = item.get("tool_arguments") if isinstance(item.get("tool_arguments"), dict) else {}
        return [
            ToolCall(
                id=f"precondition-{precondition_id}-{uuid4().hex[:10]}",
                name=tool_name,
                arguments=dict(arguments),
                read_only=spec.read_only,
                parallel_safe=spec.parallel_safe,
                metadata={
                    "origin": "skill_precondition",
                    "precondition_id": precondition_id,
                    "on_missing": str(item.get("on_missing") or ""),
                },
            )
        ]
    return []


def _context_path_value(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for segment in (part for part in path.split(".") if part):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _legacy_tool_result(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    legacy = metadata.get("legacy") if isinstance(metadata.get("legacy"), dict) else None
    if legacy is not None:
        return legacy
    status = str(item.get("status") or "failed")
    pending_action = item.get("pending_action") if isinstance(item.get("pending_action"), dict) else {}
    waiting_for_input = status == "requires_user_action" and pending_action.get("action_type") == "clarification"
    return {
        "tool_name": str(item.get("name") or ""),
        "tool_args": {},
        "status": {
            "succeeded": "ok",
            "requires_user_action": "waiting_user_input" if waiting_for_input else "requires_user_action",
            "waiting_async": "task_running",
        }.get(status, "error"),
        "summary": str(item.get("summary") or ""),
        "result": item.get("data") if isinstance(item.get("data"), dict) else {},
        "requires_user_action": status == "requires_user_action" and not waiting_for_input,
        "error": str(item.get("error") or ""),
        "artifact": {},
    }


def _project_pending_action(
    value: dict[str, Any] | None,
    tool_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if value is None:
        return None
    projected = dict(value)
    tool_call_id = str(projected.get("tool_call_id") or "")
    matching_result = next(
        (
            item
            for item in reversed(tool_results)
            if str(item.get("status") or "") == "requires_user_action"
            and (not tool_call_id or str(item.get("tool_call_id") or "") == tool_call_id)
        ),
        None,
    )
    if matching_result is not None:
        projected.setdefault("tool_name", str(matching_result.get("name") or ""))
        projected.setdefault("tool_status", "requires_user_action")
    projected.setdefault("needs_user_input", True)
    return projected


def _project_observation(
    value: dict[str, Any],
    observation_kinds: dict[str, str],
) -> dict[str, Any]:
    projected = dict(value)
    source = str(projected.get("source") or projected.get("tool_name") or "")
    data = projected.get("data") if isinstance(projected.get("data"), dict) else {}
    kind = str(
        observation_kinds.get(source)
        or data.get("kind")
        or data.get("observation_kind")
        or (source.split(".", 1)[0] if "." in source else source)
        or "tool"
    )
    projected.setdefault("tool_name", source)
    projected.setdefault("kind", kind)
    return projected


def _project_final_answer(
    state: dict[str, Any],
    runtime_status: str,
    pending_action: dict[str, Any] | None,
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(state.get("final_answer"), dict):
        answer = dict(state["final_answer"])
        answer.setdefault("answer_mode", "bubble")
        return answer
    latest_summary = next(
        (str(item.get("summary") or "") for item in reversed(tool_results) if str(item.get("summary") or "").strip()),
        "",
    )
    if runtime_status in {"waiting_user_action", "waiting_user_input"}:
        summary = str((pending_action or {}).get("prompt") or latest_summary or "需要你先完成一项操作，我会继续处理当前问题。")
        return {
            "answer_mode": "tool_handoff",
            "summary": summary,
            "markdown": summary,
            "pending_action": pending_action or {},
        }
    if runtime_status == "task_running":
        summary = latest_summary or "任务已开始生成，完成后我会继续承接当前问题。"
        return {"answer_mode": "task_running", "summary": summary, "markdown": summary}
    summary = "本轮执行未能形成有效答复。"
    return {"answer_mode": "bubble", "summary": summary, "markdown": summary, "status": "failed"}


def _active_task(
    tool_results: list[dict[str, Any]],
    typed_tool_results: list[dict[str, Any]],
    task_kinds: dict[str, str],
) -> dict[str, Any]:
    for item in reversed(tool_results):
        status = str(item.get("status") or "")
        if status not in {"task_running", "waiting_async"}:
            continue
        tool_name = str(item.get("tool_name") or "")
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        artifact = item.get("artifact") if isinstance(item.get("artifact"), dict) else {}
        kind = task_kinds.get(tool_name) or "task"
        return {
            "kind": kind,
            "tool_name": tool_name,
            "source_id": str(
                result.get("case_id")
                or result.get("session_id")
                or artifact.get("run_id")
                or artifact.get("source_id")
                or ""
            ),
            "summary": str(item.get("summary") or ""),
        }
    for item in reversed(typed_tool_results):
        if str(item.get("status") or "") != "waiting_async":
            continue
        tool_name = str(item.get("name") or "")
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        artifacts = item.get("artifacts") if isinstance(item.get("artifacts"), list) else []
        artifact = next((value for value in artifacts if isinstance(value, dict)), {})
        artifact_data = artifact.get("data") if isinstance(artifact.get("data"), dict) else {}
        artifact_type = str(artifact.get("type") or artifact_data.get("artifact_type") or "")
        return {
            "kind": task_kinds.get(tool_name) or "task",
            "tool_name": tool_name,
            "artifact_type": artifact_type,
            "source_id": str(
                data.get("case_id")
                or data.get("session_id")
                or artifact_data.get("run_id")
                or artifact_data.get("source_id")
                or ""
            ),
            "summary": str(item.get("summary") or ""),
        }
    return {}


def _trace_from_events(
    events: list[dict[str, Any]],
    *,
    observation_kinds: dict[str, str],
) -> list[dict[str, Any]]:
    trace = []
    for index, event in enumerate(events):
        if event.get("event_type") == "answer.delta":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        action = str(event.get("event_type") or "")
        item = {
            "index": int(event.get("sequence") or index + 1),
            "action": action,
            "summary": str(event.get("summary") or ""),
            "created_at": str(event.get("created_at") or ""),
        }
        if action in {"model.completed", "model.failed"}:
            item.update(
                {
                    "model_role": str(payload.get("model_role") or ""),
                    "model_id": str(payload.get("model_id") or ""),
                    "status": "failed" if action == "model.failed" else "completed",
                    "latency_ms": int(payload.get("latency_ms") or 0),
                    "answer_stream": {
                        "first_token_latency_ms": payload.get("first_token_latency_ms"),
                        "latency_ms": int(payload.get("latency_ms") or 0),
                        "delta_count": int(payload.get("delta_count") or 0),
                        "delta_chars": int(payload.get("delta_chars") or 0),
                    },
                    "route_reason": str(payload.get("route_reason") or ""),
                    "router_id": str(payload.get("router_id") or ""),
                    "policy": payload.get("policy") if isinstance(payload.get("policy"), dict) else {},
                    "budget": payload.get("budget") if isinstance(payload.get("budget"), dict) else {},
                }
            )
        if action == "model.route.selected":
            item.update(
                {
                    "model_role": str(payload.get("role") or ""),
                    "model_id": str(payload.get("model_id") or ""),
                    "route_reason": str(payload.get("reason") or ""),
                    "router_id": str(payload.get("router_id") or ""),
                    "policy": payload.get("policy") if isinstance(payload.get("policy"), dict) else {},
                    "budget": payload.get("budget") if isinstance(payload.get("budget"), dict) else {},
                }
            )
        if (action.startswith("tool.call.") and action != "tool.call.started") or action == "run.waiting_async":
            tool_result = payload.get("tool_result") if isinstance(payload.get("tool_result"), dict) else {}
            metrics = tool_result.get("runtime_metrics") if isinstance(tool_result.get("runtime_metrics"), dict) else {}
            metadata = tool_result.get("metadata") if isinstance(tool_result.get("metadata"), dict) else {}
            governance = metadata.get("governance") if isinstance(metadata.get("governance"), dict) else {}
            item.update(
                {
                    "latency_ms": int(metrics.get("latency_ms") or payload.get("latency_ms") or 0),
                    "status": str(payload.get("typed_tool_status") or payload.get("tool_status") or tool_result.get("status") or ""),
                    "outcome": str(tool_result.get("outcome") or ""),
                    "diagnostics": (
                        tool_result.get("diagnostics")
                        if isinstance(tool_result.get("diagnostics"), dict)
                        else {}
                    ),
                    "tool_calls": [{"tool_name": str(payload.get("tool_name") or tool_result.get("name") or "")}],
                    "observations": [
                        _project_observation(tool_result["observation"], observation_kinds)
                    ]
                    if isinstance(tool_result.get("observation"), dict)
                    else [],
                    "policy": governance.get("policy") if isinstance(governance.get("policy"), dict) else {},
                    "budget": governance.get("budget") if isinstance(governance.get("budget"), dict) else {},
                    "artifact_types": sorted(
                        {
                            str(artifact.get("artifact_type") or "")
                            for artifact in tool_result.get("artifacts") or []
                            if isinstance(artifact, dict)
                            and str(artifact.get("artifact_type") or "")
                        }
                    ),
                    "artifact_contract_failed": bool(
                        metadata.get("artifact_contract_failed")
                    ),
                }
            )
        trace.append(item)
    return trace


def _runtime_diagnostics(
    state: dict[str, Any],
    runtime: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    context_snapshot: dict[str, Any],
    skill_activation: dict[str, Any],
    max_steps: int,
    previous_diagnostics: dict[str, Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = previous_diagnostics if isinstance(previous_diagnostics, dict) else {}
    previous_counts = previous.get("counts") if isinstance(previous.get("counts"), dict) else {}
    previous_timings = previous.get("timings") if isinstance(previous.get("timings"), dict) else {}
    previous_skill = previous.get("skill") if isinstance(previous.get("skill"), dict) else {}
    previous_recovery = previous.get("recovery") if isinstance(previous.get("recovery"), dict) else {}
    previous_governance = previous.get("governance") if isinstance(previous.get("governance"), dict) else {}
    trace = runtime.get("trace") if isinstance(runtime.get("trace"), list) else []
    model_rows = [item for item in trace if item.get("action") in {"model.completed", "model.failed"}]
    route_rows = [item for item in trace if item.get("action") == "model.route.selected"]
    tool_rows = [
        item
        for item in trace
        if (
            str(item.get("action") or "").startswith("tool.call.")
            and item.get("action") != "tool.call.started"
        )
        or item.get("action") == "run.waiting_async"
    ]
    confirmation_rows = [
        item for item in trace if str(item.get("action") or "").startswith("policy.")
    ]
    completed_tools = {
        str((item.get("tool_calls") or [{}])[0].get("tool_name") or "")
        for item in tool_rows
        if str(item.get("status") or "") in {"ok", "succeeded", "completed"}
    }
    completed_tools.update(
        str(name)
        for name in previous_skill.get("completed_tools", [])
        if str(name)
    )
    completed_tools.update(
        str(name)
        for name in state.get("completed_tools", [])
        if str(name)
    )
    state_skill = state.get("skill_activation") if isinstance(state.get("skill_activation"), dict) else {}
    completed_tools.update(
        str(name)
        for name in state_skill.get("completed_tools", [])
        if str(name)
    )
    context_selection = context_snapshot.get("context_selection") if isinstance(context_snapshot.get("context_selection"), dict) else {}
    state_model_policy = state.get("model_policy") if isinstance(state.get("model_policy"), dict) else {}
    state_budget_policy = (
        state_model_policy.get("budget")
        if isinstance(state_model_policy.get("budget"), dict)
        else {}
    )
    skill = dict(skill_activation) if isinstance(skill_activation, dict) else {}
    if skill:
        skill["completed_tools"] = sorted(name for name in completed_tools if name)
        skill["completion_outcome"] = str(runtime.get("status") or "")
        skill["policy_violation_count"] = len(skill.get("policy_violations") or [])
        skill["policy_phase"] = str(state.get("policy_phase") or skill.get("policy_phase") or "")
        missing = (
            state.get("missing_requirements")
            if isinstance(state.get("missing_requirements"), dict)
            else {}
        )
        skill["missing_requirements"] = {
            "tools": list(missing.get("tools") or []),
            "artifacts": list(missing.get("artifacts") or []),
        }
        skill["repair_count"] = int(
            state.get("repair_count") or skill.get("repair_count") or 0
        )
        output_contract = skill.get("output_contract") if isinstance(skill.get("output_contract"), dict) else {}
        skill["required_artifact"] = str(output_contract.get("requires_artifact") or skill.get("required_artifact") or "")
    return {
        "schema_version": "harness-runtime-diagnostics-v1",
        "context": {
            "context_version": str(context_snapshot.get("schema_version") or ""),
            "selection_strategy": str(context_selection.get("strategy") or "harness_context_v1"),
        },
        "capabilities": dict(capability_manifest or {}),
        "loop": {
            "status": str(runtime.get("status") or ""),
            "stop_reason": str(runtime.get("stop_reason") or ""),
            "step_count": int(runtime.get("step_count") or 0),
            "max_steps": int(max_steps),
            "max_tool_calls_per_step": 0,
            "max_elapsed_seconds": float(state_budget_policy.get("max_elapsed_seconds") or 0),
        },
        "counts": {
            "trace_steps": int(previous_counts.get("trace_steps") or 0) + len(trace),
            "model_calls": int(previous_counts.get("model_calls") or 0) + len(model_rows),
            "tool_calls": int(previous_counts.get("tool_calls") or 0) + len(tool_rows),
            "degraded_tool_results": int(previous_counts.get("degraded_tool_results") or 0)
            + sum(str(item.get("outcome") or "") == "degraded" for item in tool_rows),
            "skipped_tool_results": int(previous_counts.get("skipped_tool_results") or 0)
            + sum(str(item.get("outcome") or "") == "skipped" for item in tool_rows),
            "tool_results": max(
                int(previous_counts.get("tool_results") or 0),
                len(state.get("tool_results") or []),
            ),
            "observations": max(
                int(previous_counts.get("observations") or 0),
                len(state.get("observations") or []),
            ),
            "events": int(previous_counts.get("events") or 0) + len(events),
        },
        "recovery": {
            **previous_recovery,
            "resume_mode": str((runtime.get("pending_action") or {}).get("resume_mode") or ""),
            "pending_action": runtime.get("pending_action") or {},
        },
        "governance": {
            "model_routes": (
                previous_governance.get("model_routes")
                if isinstance(previous_governance.get("model_routes"), list)
                else []
            )
            + [
                {
                    "step": item.get("index"),
                    "role": str(item.get("model_role") or ""),
                    "model_id": str(item.get("model_id") or ""),
                    "reason": str(item.get("route_reason") or ""),
                    "router_id": str(item.get("router_id") or ""),
                    "policy": item.get("policy") if isinstance(item.get("policy"), dict) else {},
                    "budget": item.get("budget") if isinstance(item.get("budget"), dict) else {},
                    "budget_metrics": (
                        item.get("budget_metrics")
                        if isinstance(item.get("budget_metrics"), dict)
                        else {}
                    ),
                    "usage": item.get("usage") if isinstance(item.get("usage"), dict) else {},
                    "max_output_tokens": item.get("max_output_tokens"),
                }
                for item in route_rows
            ],
            "budget": state.get("budget_state")
            if isinstance(state.get("budget_state"), dict)
            else {},
            "tool_policies": (
                previous_governance.get("tool_policies")
                if isinstance(previous_governance.get("tool_policies"), list)
                else []
            )
            + [
                {
                    "step": item.get("index"),
                    "tool_name": str((item.get("tool_calls") or [{}])[0].get("tool_name") or ""),
                    "status": str(item.get("status") or ""),
                    "outcome": str(item.get("outcome") or ""),
                    "policy": item.get("policy") if isinstance(item.get("policy"), dict) else {},
                    "budget": item.get("budget") if isinstance(item.get("budget"), dict) else {},
                }
                for item in tool_rows
                if item.get("policy") or item.get("budget")
            ],
            "confirmations": (
                previous_governance.get("confirmations")
                if isinstance(previous_governance.get("confirmations"), list)
                else []
            )
            + [
                {
                    "step": item.get("index"),
                    "action": str(item.get("action") or ""),
                    "summary": str(item.get("summary") or ""),
                }
                for item in confirmation_rows
            ],
        },
        "skill": skill,
        "replay": {"prompt_hashes": [], "model_output_hashes": []},
        "timings": {
            "model_calls": (
                previous_timings.get("model_calls")
                if isinstance(previous_timings.get("model_calls"), list)
                else []
            )
            + [
                {
                    "step": item.get("index"),
                    "role": str(item.get("model_role") or ""),
                    "model_id": str(item.get("model_id") or ""),
                    "status": "failed" if item.get("action") == "model.failed" else "completed",
                    "latency_ms": int(item.get("latency_ms") or 0),
                    "answer_stream": item.get("answer_stream") if isinstance(item.get("answer_stream"), dict) else {},
                    "error": str(item.get("summary") or "") if item.get("action") == "model.failed" else "",
                }
                for item in model_rows
            ],
            "tool_calls": (
                previous_timings.get("tool_calls")
                if isinstance(previous_timings.get("tool_calls"), list)
                else []
            )
            + [
                {
                    "tool_name": str((item.get("tool_calls") or [{}])[0].get("tool_name") or ""),
                    "status": str(item.get("status") or ""),
                    "latency_ms": int(item.get("latency_ms") or 0),
                    "error": str(item.get("summary") or "") if item.get("action") == "tool.call.failed" else "",
                }
                for item in tool_rows
            ],
        },
    }
