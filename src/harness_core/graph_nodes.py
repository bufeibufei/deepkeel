from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from harness_core.budget import BudgetLedger
from harness_core.contracts import AgentMessage, FinalAnswer, PendingAction, ToolCall
from harness_core.control import RunControl
from harness_core.deadlines import ensure_time_remaining
from harness_core.events import AgentEventPersistenceError
from harness_core.hooks import HookAction, HookAudit, HookInvocation, HookPoint
from harness_core.model import ModelGateway, ModelTurn, model_tools_from_registry
from harness_core.model_failures import ModelToolContractError
from harness_core.model_routing import ModelStepContext
from harness_core.skills import SkillPolicy
from harness_core.skill_activation import EntryToolActivationRequest
from harness_core.tool_registry import ToolRegistry
from harness_core.tool_disclosure import resolve_tool_view
from harness_core.tools import ToolExecutionContext, ToolExecutor
from harness_core.turn_context import TurnContextRegistry, TurnExecutionContext
from harness_core.type_narrowing import as_dict
from harness_core.workflow_policy import evaluate_workflow_completion
from harness_core.graph_state import (
    _allowed_tool_names, _apply_policy_confirmation_resume, _apply_resume_payload,
    _apply_tool_result, _copy_state, _forced_workflow_tool_name, _hydrate_call,
    _is_policy_confirmation, _is_suspending_call, _messages, _model_available_roles,
    _parallel_suspension_rejected, _skill_tool_parameter_overrides, _stable_tool_calls,
    HarnessGraphState, migrate_legacy_graph_state,
)
from harness_core.graph_workflow import (
    TRUNCATED_FINISH_REASONS,
    _answer_summary, _complete_continued_answer, _config_value,
    _continue_or_fail_truncated_model_response, _emit, _finish_failed, _latency_ms,
    _record_completed_tool, _repair_or_fail_workflow,
    _retry_or_fail_empty_model_response, _set_policy_state,
    _wait_for_workflow_input, _workflow_can_wait_for_user_input,
)


def _latest_user_question(messages: list[Any]) -> str:
    for item in reversed(messages):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("role") or "") == "user":
            return str(item.get("content") or "").strip()
    return ""


def _model_hook_invocation(
    point: HookPoint,
    state: Mapping[str, Any],
    turn_context: TurnExecutionContext,
    *,
    payload: Mapping[str, Any],
) -> HookInvocation:
    skill = as_dict(state.get("skill_activation"))
    metadata = as_dict(state.get("metadata"))
    package_ids = tuple(
        str(value)
        for value in turn_context.tool_context.metadata.get(
            "capability_package_ids", ()
        )
        if str(value).strip()
    )
    return HookInvocation(
        point=point,
        operation_id=(
            f"{state.get('run_id')}:{state.get('turn_id')}:"
            f"model:{int(state.get('step_count') or 0)}:{point.value}"
        ),
        run_id=str(state.get("run_id") or ""),
        thread_id=str(state.get("thread_id") or ""),
        turn_id=str(state.get("turn_id") or ""),
        package_ids=package_ids,
        skill_id=str(skill.get("skill_id") or ""),
        payload=dict(payload),
        metadata={
            "governance_scope": dict(metadata.get("governance_scope") or {}),
        },
    )


def _forced_tool_clarification_fallback(
    registry: ToolRegistry,
    forced_tool_name: str,
    error: ModelToolContractError,
) -> ModelTurn | None:
    try:
        spec = registry.get(forced_tool_name)
    except KeyError:
        return None
    clarification = as_dict(as_dict(spec.argument_contract).get("clarification"))
    if not clarification or not (spec.required_args or spec.required_arg_groups):
        return None
    return ModelTurn(
        content="",
        tool_calls=[
            ToolCall(
                id=f"contract-clarification-{uuid4()}",
                name=forced_tool_name,
                arguments={},
            )
        ],
        finish_reason="tool_calls",
        raw={
            "synthetic": True,
            "recovery": "forced_tool_clarification",
            "error": str(error),
        },
    )


def _emit_hook_audits(
    state: dict[str, Any],
    config: RunnableConfig,
    audits: tuple[HookAudit, ...],
) -> None:
    for audit in audits:
        _emit(
            state,
            config,
            "hook.executed",
            "Lifecycle hook",
            f"{audit.point.value}: {audit.status}",
            {
                "hook_id": audit.hook_id,
                "hook_point": audit.point.value,
                "operation_id": audit.operation_id,
                "status": audit.status,
                "duration_ms": audit.duration_ms,
                "replayed": audit.replayed,
                "required": audit.required,
                "error": audit.error,
                "diagnostics": dict(audit.diagnostics),
                "visible": False,
            },
        )


class GraphNodes:
    def __init__(
        self, *, model: ModelGateway | None, tool_executor: ToolExecutor,
        tool_registry: ToolRegistry, prompt: str, max_steps: int,
        ledger: BudgetLedger, deadline_monotonic: float | None, control: RunControl,
        turn_contexts: TurnContextRegistry,
    ) -> None:
        self.model = model
        self.tool_executor = tool_executor
        self.tool_registry = tool_registry
        self.prompt = prompt
        self.max_steps = max_steps
        self.ledger = ledger
        self.deadline_monotonic = deadline_monotonic
        self.control = control
        self.turn_contexts = turn_contexts

    def turn_context(
        self,
        config: RunnableConfig | None,
        state: Mapping[str, Any] | None = None,
    ) -> TurnExecutionContext | None:
        context = _config_value(config or {}, "turn_context")
        if isinstance(context, TurnExecutionContext):
            return context
        current = state or {}
        return self.turn_contexts.resolve(
            str(current.get("run_id") or ""),
            str(current.get("thread_id") or ""),
        )

    def ensure_active(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
        *,
        force: bool = False,
    ) -> None:
        self.control.raise_if_cancelled(str(state.get("run_id") or ""), force=force)
        turn_context = self.turn_context(config, state)
        deadline = (
            turn_context.deadline_monotonic
            if turn_context is not None
            else self.deadline_monotonic
        )
        ensure_time_remaining(deadline)

    @staticmethod
    def normalize_state(state: dict[str, Any], config: RunnableConfig) -> HarnessGraphState:
        context = _config_value(config, "tool_context")
        thread_id = str(_config_value(config, "thread_id") or state.get("thread_id") or "")
        normalized = dict(state)
        if isinstance(context, ToolExecutionContext):
            normalized.setdefault("run_id", context.run_id)
            normalized.setdefault("user_id", context.user_id)
        return migrate_legacy_graph_state(normalized, thread_id=thread_id or str(normalized.get("run_id") or "checkpoint"))

    def model_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return asyncio.run(self.amodel_node(state, config))

    async def amodel_node(
        self,
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        normalized_state = self.normalize_state(state, config)
        if not normalized_state["messages"]:
            raise ValueError("graph checkpoint cannot resume model execution without messages")
        turn_context = self.turn_context(config, normalized_state)
        model = turn_context.model if turn_context is not None else self.model
        if model is None:
            raise RuntimeError("a model gateway is required in TurnExecutionContext")
        tool_registry = self.tool_registry
        prompt = turn_context.system_prompt if turn_context is not None else self.prompt
        max_steps = self.max_steps
        ledger = self.ledger
        deadline_monotonic = (
            turn_context.deadline_monotonic
            if turn_context is not None
            else self.deadline_monotonic
        )
        self.ensure_active(normalized_state, config, force=True)
        current = _copy_state(normalized_state)
        if int(current.get("step_count") or 0) >= max_steps:
            skill_policy = SkillPolicy.from_snapshot(current.get("skill_activation"))
            completion = evaluate_workflow_completion(skill_policy, current)
            if (
                skill_policy.active
                and skill_policy.durable
                and not completion.allowed
                and str(current.get("policy_phase") or "") == "repair"
            ):
                return _repair_or_fail_workflow(current, skill_policy, completion, config)
            return _finish_failed(current, "The agent reached the maximum step count without a reliable answer.", config)
        current["status"] = "reasoning"
        _emit(current, config, "agent.reasoning", "Reasoning", "The agent is evaluating context and selecting the next step.")
        allowed_tools = _allowed_tool_names(current, tool_registry)
        skill_policy = SkillPolicy.from_snapshot(current.get("skill_activation"))
        workflow_completion = evaluate_workflow_completion(skill_policy, current)
        workflow_is_finalizing = (
            skill_policy.active
            and skill_policy.durable
            and workflow_completion.allowed
        )
        workflow_finalization_tools: list[str] = []
        if workflow_is_finalizing:
            workflow_finalization_tools = sorted(allowed_tools or ())
            allowed_tools = set()
        tool_view = resolve_tool_view(
            registry=tool_registry,
            allowed_names=allowed_tools,
            skill=skill_policy,
            mode=turn_context.tool_view_mode if turn_context is not None else "legacy",
            discovered_names=set(
                str(name)
                for name in as_dict(current.get("metadata")).get(
                    "discovered_tool_names", []
                )
                if str(name).strip()
            ),
        )
        metadata = current.setdefault("metadata", {})
        if workflow_is_finalizing:
            finalization = as_dict(metadata.get("workflow_finalization"))
            if not finalization.get("instruction_injected"):
                current.setdefault("messages", []).append(
                    AgentMessage(
                        id=f"workflow-finalization-{uuid4()}",
                        role="system",
                        content=(
                            "The active workflow contract is fully satisfied. "
                            "Do not call or discover any more tools. Return the final "
                            "user-facing answer using the completed observations and artifacts."
                        ),
                        metadata={"kind": "workflow_finalization_guard"},
                    ).model_dump(mode="json")
                )
            metadata["workflow_finalization"] = {
                **finalization,
                "contract_satisfied": True,
                "suppressed_tool_names": workflow_finalization_tools,
                "instruction_injected": True,
            }
            if not finalization.get("instruction_injected"):
                _emit(
                    current,
                    config,
                    "workflow.finalizing",
                    "Workflow result ready",
                    "The workflow contract is satisfied; the model must now finalize the answer.",
                    {
                        "suppressed_tool_names": workflow_finalization_tools,
                        "visible": False,
                    },
                )
        previous_tool_view = metadata.get("tool_view")
        metadata["tool_view"] = tool_view.as_dict()
        if previous_tool_view != metadata["tool_view"]:
            _emit(
                current,
                config,
                "tools.disclosure.resolved",
                "Tool view resolved",
                f"{len(tool_view.exposed_names)} tools exposed",
                {**tool_view.as_dict(), "visible": False},
            )
        tools = model_tools_from_registry(
            tool_registry,
            allowed_names=set(tool_view.exposed_names),
            parameter_overrides=_skill_tool_parameter_overrides(current, tool_registry),
        )
        forced_tool_name = _forced_workflow_tool_name(current, tools)
        hook_runner = turn_context.hook_runner if turn_context is not None else None
        if hook_runner is not None and turn_context is not None:
            before_model = await hook_runner.arun(
                _model_hook_invocation(
                    HookPoint.MODEL_BEFORE,
                    current,
                    turn_context,
                    payload={
                        "message_count": len(current.get("messages") or []),
                        "tool_names": [
                            str(item.get("function", {}).get("name") or "")
                            for item in tools
                            if isinstance(item, dict)
                        ],
                        "system_prompt": prompt,
                    },
                )
            )
            _emit_hook_audits(current, config, before_model.audits)
            if before_model.decision.action == HookAction.DENY:
                return _finish_failed(
                    current,
                    before_model.decision.reason
                    or "model invocation denied by lifecycle hook",
                    config,
                )
            if before_model.decision.action == HookAction.WAIT_FOR_CONFIRMATION:
                pending = PendingAction(
                    id=f"hook-model:{current.get('run_id')}:{current.get('step_count')}",
                    run_id=str(current.get("run_id") or ""),
                    action_type="confirm_model_invocation",
                    title=before_model.decision.confirmation_title or "Confirm continuation",
                    prompt=(
                        before_model.decision.confirmation_message
                        or before_model.decision.reason
                        or "Please confirm before the agent continues."
                    ),
                    payload={"source": "lifecycle_hook"},
                )
                current["pending_action"] = pending.model_dump(mode="json")
                current["status"] = "waiting_user"
                _emit(
                    current,
                    config,
                    "agent.waiting_user",
                    pending.title,
                    pending.prompt,
                    {"pending_action": current["pending_action"]},
                )
                return current
            model_patch = dict(before_model.decision.model_input_patch)
            prompt = str(model_patch.get("system_prompt") or prompt)
            appended_messages = model_patch.get("append_messages")
            if isinstance(appended_messages, list):
                for item in appended_messages:
                    message = AgentMessage.model_validate(item)
                    current.setdefault("messages", []).append(
                        message.model_dump(mode="json")
                    )
        delta_index = 0
        delta_chars = 0
        model_started_at = time.perf_counter()
        first_delta_at: float | None = None
        route_payload: dict[str, Any] = {}
        routed_attempts: set[tuple[int, str]] = set()

        def on_route(payload: dict[str, Any]) -> None:
            route_payload.update(payload)
            route_key = (
                int(payload.get("attempt_index") or 1),
                str(payload.get("retry_kind") or "primary"),
            )
            if route_key in routed_attempts:
                _emit(
                    current,
                    config,
                    "budget.usage.recorded",
                    "Model usage recorded",
                    str((payload.get("usage") or {}).get("source") or ""),
                    {
                        "role": str(payload.get("role") or ""),
                        "model_id": str(payload.get("model_id") or ""),
                        "attempt_index": route_key[0],
                        "usage": dict(payload.get("usage") or {}),
                        "budget_metrics": dict(payload.get("budget_metrics") or {}),
                        "visible": False,
                    },
                )
                return
            routed_attempts.add(route_key)
            _emit(
                current,
                config,
                "model.route.selected",
                "Model route selected",
                str(payload.get("reason") or ""),
                {**payload, "visible": False},
            )

        def on_delta(delta: str) -> None:
            nonlocal delta_chars, delta_index, first_delta_at
            self.ensure_active(current, config)
            if first_delta_at is None:
                first_delta_at = time.perf_counter()
            delta_chars += len(delta)
            _emit(
                current,
                config,
                "model.delta",
                "",
                "",
                {"delta": delta, "index": delta_index, "stream_mode": "provider_stream"},
                ephemeral=True,
            )
            delta_index += 1

        try:
            model_step_context = ModelStepContext(
                    run_id=str(current.get("run_id") or ""),
                    user_id=str(current.get("user_id") or ""),
                    thread_id=str(current.get("thread_id") or ""),
                    turn_id=str(current.get("turn_id") or ""),
                    step_index=int(current.get("step_count") or 0),
                    message_count=len(current.get("messages") or []),
                    observation_count=len(current.get("observations") or []),
                    tool_result_count=len(current.get("tool_results") or []),
                    available_roles=_model_available_roles(model),
                    observation_sources=tuple(
                        str(item.get("source") or "")
                        for item in (current.get("observations") or [])
                        if isinstance(item, dict)
                    ),
                    tool_result_names=tuple(
                        str(item.get("name") or "")
                        for item in (current.get("tool_results") or [])
                        if isinstance(item, dict)
                    ),
                    model_policy=dict(current.get("model_policy") or {}),
                    skill_activation=dict(current.get("skill_activation") or {}),
                    policy_phase=str(current.get("policy_phase") or ""),
                    forced_tool_name=forced_tool_name,
                    governance_scope=dict(
                        (current.get("metadata") or {}).get("governance_scope") or {}
                    ),
                    deadline_monotonic=deadline_monotonic,
                )
            async_run_turn = getattr(model, "arun_turn", None)
            try:
                if callable(async_run_turn):
                    turn = await async_run_turn(
                        _messages(current),
                        tools=tools,
                        system_prompt=prompt,
                        on_text_delta=None if forced_tool_name else on_delta,
                        step_context=model_step_context,
                        on_route=on_route,
                    )
                else:
                    turn = await asyncio.to_thread(
                        model.run_turn,
                        _messages(current),
                        tools=tools,
                        system_prompt=prompt,
                        on_text_delta=None if forced_tool_name else on_delta,
                        step_context=model_step_context,
                        on_route=on_route,
                    )
            except ModelToolContractError as exc:
                turn = _forced_tool_clarification_fallback(
                    tool_registry,
                    forced_tool_name,
                    exc,
                )
                if turn is None:
                    raise
                _emit(
                    current,
                    config,
                    "model.tool_contract.recovered",
                    "Required tool clarification recovered",
                    forced_tool_name,
                    {
                        "tool_name": forced_tool_name,
                        "recovery": "forced_tool_clarification",
                        "visible": False,
                    },
                )
            self.ensure_active(current, config, force=True)
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            current["budget_state"] = ledger.snapshot(current["run_id"]).as_dict()
            latency_ms = int((time.perf_counter() - model_started_at) * 1000)
            if route_payload.get("budget_metrics"):
                _emit(
                    current,
                    config,
                    "budget.usage.recorded",
                    "Model usage recorded",
                    str((route_payload.get("usage") or {}).get("source") or "estimated_failure"),
                    {
                        "role": str(route_payload.get("role") or ""),
                        "model_id": str(route_payload.get("model_id") or ""),
                        "usage": dict(route_payload.get("usage") or {}),
                        "budget_metrics": dict(route_payload.get("budget_metrics") or {}),
                        "visible": False,
                    },
                )
            _emit(
                current,
                config,
                "model.failed",
                "Model call failed",
                str(exc),
                {
                    "model_id": str(route_payload.get("model_id") or ""),
                    "model_role": str(route_payload.get("role") or ""),
                    "latency_ms": latency_ms,
                    "first_token_latency_ms": _latency_ms(model_started_at, first_delta_at),
                    "delta_count": delta_index,
                    "delta_chars": delta_chars,
                    "error_type": type(exc).__name__,
                    "error_code": str(getattr(exc, "code", "") or ""),
                    "route": dict(route_payload),
                },
            )
            raise
        if hook_runner is not None and turn_context is not None:
            after_model = await hook_runner.arun(
                _model_hook_invocation(
                    HookPoint.MODEL_AFTER,
                    current,
                    turn_context,
                    payload={
                        "model_id": turn.model_id,
                        "model_role": turn.model_role,
                        "finish_reason": turn.finish_reason,
                        "content_chars": len(turn.content),
                        "tool_call_count": len(turn.tool_calls),
                    },
                )
            )
            _emit_hook_audits(current, config, after_model.audits)
        current["budget_state"] = ledger.snapshot(current["run_id"]).as_dict()
        model_latency_ms = int((time.perf_counter() - model_started_at) * 1000)
        model_metrics: dict[str, Any] = {
            "model_id": turn.model_id,
            "model_role": turn.model_role,
            "finish_reason": turn.finish_reason,
            "latency_ms": model_latency_ms,
            "first_token_latency_ms": _latency_ms(model_started_at, first_delta_at),
            "delta_count": delta_index,
            "delta_chars": delta_chars,
            "content_chars": len(turn.content),
            "tool_call_count": len(turn.tool_calls),
            "route_reason": str(route_payload.get("reason") or ""),
            "router_id": str(route_payload.get("router_id") or ""),
            "policy": route_payload.get("policy") if isinstance(route_payload.get("policy"), dict) else {},
            "budget": route_payload.get("budget") if isinstance(route_payload.get("budget"), dict) else {},
            "budget_metrics": (
                route_payload.get("budget_metrics")
                if isinstance(route_payload.get("budget_metrics"), dict)
                else {}
            ),
            "usage": route_payload.get("usage") if isinstance(route_payload.get("usage"), dict) else {},
            "max_output_tokens": route_payload.get("max_output_tokens"),
            "forced_tool_name": forced_tool_name,
        }
        _emit(
            current,
            config,
            "budget.usage.recorded",
            "Model usage recorded",
            str(as_dict(model_metrics["usage"]).get("source") or ""),
            {
                "role": turn.model_role,
                "model_id": turn.model_id,
                "usage": model_metrics["usage"],
                "budget_metrics": model_metrics["budget_metrics"],
                "max_output_tokens": model_metrics["max_output_tokens"],
                "visible": False,
            },
        )
        current["step_count"] = int(current.get("step_count") or 0) + 1
        reported_tool_calls = _stable_tool_calls(
            turn.tool_calls,
            run_id=str(current.get("run_id") or ""),
            step_index=int(current["step_count"]),
        )
        exposed_tool_names = set(tool_view.exposed_names)
        registered_tool_names = {
            spec.name for spec in tool_registry.list_tools()
        }
        tool_calls = [
            call
            for call in reported_tool_calls
            if not workflow_is_finalizing
            or call.name in exposed_tool_names
            or call.name not in registered_tool_names
        ]
        rejected_tool_calls = [
            call
            for call in reported_tool_calls
            if workflow_is_finalizing
            and call.name in registered_tool_names
            and call.name not in exposed_tool_names
        ]
        if rejected_tool_calls:
            rejected_names = [call.name for call in rejected_tool_calls]
            metadata = current.setdefault("metadata", {})
            metadata.setdefault("rejected_undisclosed_tool_calls", []).append(
                {
                    "step_index": int(current["step_count"]),
                    "tool_names": rejected_names,
                }
            )
            _emit(
                current,
                config,
                "tools.disclosure.rejected",
                "Undisclosed tool call rejected",
                ", ".join(rejected_names),
                {
                    "tool_names": rejected_names,
                    "exposed_tool_names": sorted(exposed_tool_names),
                    "visible": False,
                },
            )
        if tool_calls and not SkillPolicy.from_snapshot(
            current.get("skill_activation")
        ).active:
            active_turn_context = turn_context
            activator = (
                active_turn_context.entry_tool_skill_activator
                if active_turn_context is not None
                else None
            )
            if activator is not None and active_turn_context is not None:
                activation_request = EntryToolActivationRequest(
                    tool_calls=tuple(tool_calls),
                    current_activation=dict(current.get("skill_activation") or {}),
                    run_id=str(current.get("run_id") or ""),
                    user_id=str(current.get("user_id") or ""),
                    thread_id=str(current.get("thread_id") or ""),
                    turn_id=str(current.get("turn_id") or ""),
                    question=_latest_user_question(current.get("messages") or []),
                    messages=tuple(
                        item
                        for item in current.get("messages") or []
                        if isinstance(item, dict)
                    ),
                    context_bundle=dict(
                        active_turn_context.tool_context.context_bundle
                        if isinstance(
                            active_turn_context.tool_context.context_bundle,
                            dict,
                        )
                        else {}
                    ),
                )
                decision = activator.activate(activation_request)
                if decision is not None:
                    activation = dict(decision.skill_activation)
                    if not str(activation.get("skill_id") or "").strip():
                        return _finish_failed(
                            current,
                            "Entry-tool Skill activation returned an invalid snapshot.",
                            config,
                        )
                    replacement_calls = list(decision.tool_calls or tuple(tool_calls))
                    if {
                        (call.id, call.name) for call in replacement_calls
                    } != {
                        (call.id, call.name) for call in tool_calls
                    }:
                        return _finish_failed(
                            current,
                            "Entry-tool Skill activation changed the selected tool calls.",
                            config,
                        )
                    tool_calls = replacement_calls
                    current["skill_activation"] = activation
                    activation_metadata = current.setdefault("metadata", {})
                    activation_metadata["entry_tool_skill_activation"] = {
                        "skill_id": str(activation.get("skill_id") or ""),
                        "source": str(activation.get("source") or "model"),
                        "reason": str(decision.reason or "entry_tool_selected"),
                        "tool_names": [call.name for call in tool_calls],
                    }
                    active_turn_context.tool_context.metadata["skill_activation"] = activation
                    governance_scope = as_dict(
                        active_turn_context.tool_context.metadata.get("governance_scope")
                    )
                    active_turn_context.tool_context.metadata["governance_scope"] = {
                        **governance_scope,
                        "skill_id": str(activation.get("skill_id") or ""),
                    }
                    _emit(
                        current,
                        config,
                        "skill.activated",
                        "Skill activated",
                        str(activation.get("label") or activation.get("skill_id") or ""),
                        {
                            "skill_id": str(activation.get("skill_id") or ""),
                            "source": str(activation.get("source") or "model"),
                            "reason": str(decision.reason or "entry_tool_selected"),
                            "entry_tool_names": [call.name for call in tool_calls],
                            "visible": False,
                        },
                    )
        assistant = AgentMessage(
            id=f"assistant-{uuid4()}",
            role="assistant",
            content=turn.content,
            tool_calls=tool_calls,
            metadata={
                "model_id": turn.model_id,
                "model_role": turn.model_role,
                "finish_reason": turn.finish_reason,
                "runtime_metrics": model_metrics,
            },
        )
        current.setdefault("messages", []).append(assistant.model_dump(mode="json"))
        _emit(
            current,
            config,
            "model.completed",
            "Model response completed",
            turn.content[:160],
            model_metrics,
        )
        if (
            not tool_calls
            and turn.finish_reason.strip().lower() in TRUNCATED_FINISH_REASONS
        ):
            return _continue_or_fail_truncated_model_response(
                current,
                config,
                content=turn.content,
                finish_reason=turn.finish_reason,
                can_continue=int(current.get("step_count") or 0) < max_steps,
            )
        if not tool_calls and not turn.content.strip():
            # Do not poison the next provider request with an empty assistant
            # message when retrying a transient successful-but-empty response.
            current["messages"].pop()
            return _retry_or_fail_empty_model_response(
                current,
                config,
                can_retry=int(current.get("step_count") or 0) < max_steps,
                answer_only=workflow_is_finalizing,
            )
        metadata = current.setdefault("metadata", {})
        metadata["consecutive_empty_model_responses"] = 0
        metadata.pop("empty_model_retry_pending", None)
        if tool_calls:
            current["pending_tool_calls"] = [call.model_dump(mode="json") for call in tool_calls]
            current["status"] = "executing_tools"
            return current
        if not turn.content.strip():
            return _finish_failed(current, "The model returned no usable content for this step.", config)
        skill_policy = SkillPolicy.from_snapshot(current.get("skill_activation"))
        completion = evaluate_workflow_completion(skill_policy, current)
        if not completion.allowed:
            if _workflow_can_wait_for_user_input(skill_policy, completion, current):
                return _wait_for_workflow_input(
                    current,
                    skill_policy,
                    completion,
                    turn.content,
                    config,
                )
            return _repair_or_fail_workflow(current, skill_policy, completion, config)
        if skill_policy.active and skill_policy.durable:
            _set_policy_state(current, phase="completed", decision=completion)
        final_content = _complete_continued_answer(
            current.setdefault("metadata", {}),
            turn.content,
        )
        if hook_runner is not None and turn_context is not None:
            before_answer = await hook_runner.arun(
                _model_hook_invocation(
                    HookPoint.ANSWER_BEFORE_FINALIZE,
                    current,
                    turn_context,
                    payload={
                        "markdown": final_content,
                        "model_id": turn.model_id,
                        "artifact_ids": [
                            str(item.get("id") or "")
                            for item in current.get("artifacts", [])
                            if item.get("id")
                        ],
                    },
                )
            )
            _emit_hook_audits(current, config, before_answer.audits)
            if before_answer.decision.action == HookAction.DENY:
                return _finish_failed(
                    current,
                    before_answer.decision.reason
                    or "answer finalization denied by lifecycle hook",
                    config,
                )
            final_content = str(
                before_answer.decision.model_input_patch.get("answer_markdown")
                or final_content
            )
        answer = FinalAnswer(
            markdown=final_content,
            summary=_answer_summary(final_content),
            model_id=turn.model_id,
            model_role=turn.model_role,
            stop_reason=turn.finish_reason or "completed",
            artifact_ids=[str(item.get("id") or "") for item in current.get("artifacts", []) if item.get("id")],
        )
        current["final_answer"] = answer.model_dump(mode="json")
        current["status"] = "completed"
        current["pending_tool_calls"] = []
        _emit(
            current,
            config,
            "answer.completed",
            "Answer completed",
            answer.summary,
            {"final_answer": current["final_answer"]},
        )
        return current

    def tool_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return asyncio.run(self.atool_node(state, config))

    async def atool_node(
        self,
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        normalized_state = self.normalize_state(state, config)
        tool_registry = self.tool_registry
        tool_executor = self.tool_executor
        ledger = self.ledger
        self.ensure_active(normalized_state, config, force=True)
        current = _copy_state(normalized_state)
        calls = [_hydrate_call(item, tool_registry, current["run_id"]) for item in current.get("pending_tool_calls", [])]
        context = _config_value(config, "tool_context")
        if not isinstance(context, ToolExecutionContext):
            context = ToolExecutionContext(run_id=current["run_id"], user_id=current["user_id"])
        else:
            context = context.fork(session=context.session)
        grants = current.get("metadata", {}).get("confirmation_grants")
        if isinstance(grants, dict):
            context.metadata["confirmation_grants"] = dict(grants)
        suspending_calls = [call for call in calls if _is_suspending_call(call, tool_registry)]
        if len(suspending_calls) > 1:
            results = [_parallel_suspension_rejected(call, current["run_id"]) for call in calls]
        else:
            for call in calls:
                try:
                    tool_spec = tool_registry.get(call.name)
                    visible_label = tool_spec.visible_label or call.name
                    start_event_visible = (
                        tool_spec.runtime_policy.get("start_event_visible") is not False
                    )
                except KeyError:
                    visible_label = call.name
                    start_event_visible = True
                _emit(
                    current,
                    config,
                    "tool.started",
                    visible_label,
                    "",
                    {
                        "tool_call": call.model_dump(mode="json"),
                        "visible": start_event_visible,
                    },
                )
            results = await tool_executor.aexecute_many(calls, context)
        current["budget_state"] = ledger.snapshot(current["run_id"]).as_dict()
        current["pending_tool_calls"] = []
        current["pending_action"] = None
        current["pending_async"] = None
        for result in results:
            if _is_unexecuted_suspension_rejection(result):
                _emit(
                    current,
                    config,
                    "tool.skipped",
                    result.name,
                    "Skipped until the pending action is resolved.",
                    {
                        "tool_call": result.call.model_dump(mode="json") if result.call else {},
                        "tool_result": result.model_dump(mode="json", exclude={"call"}),
                        "visible": False,
                    },
                )
                continue
            current.setdefault("tool_results", []).append(result.model_dump(mode="json"))
            _record_completed_tool(current, result)
            _apply_tool_result(current, result, config)
        if isinstance(grants, dict):
            remaining_grants = dict(grants)
            for call in calls:
                remaining_grants.pop(call.id, None)
            current.setdefault("metadata", {})["confirmation_grants"] = remaining_grants
        if current.get("pending_action"):
            current["status"] = "waiting_user"
        elif current.get("pending_async"):
            current["status"] = "waiting_async"
        else:
            current["status"] = "reasoning"
        return current

    def await_user_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        normalized_state = self.normalize_state(state, config)
        self.ensure_active(normalized_state, config, force=True)
        current = _copy_state(normalized_state)
        resume_payload = interrupt(current.get("pending_action") or {})
        if _is_policy_confirmation(current.get("pending_action")):
            return _apply_policy_confirmation_resume(current, resume_payload, config)
        return _apply_resume_payload(current, resume_payload, config, source="user_action")

    def await_async_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        normalized_state = self.normalize_state(state, config)
        self.ensure_active(normalized_state, config, force=True)
        current = _copy_state(normalized_state)
        resume_payload = interrupt(current.get("pending_async") or {})
        return _apply_resume_payload(current, resume_payload, config, source="async_observation")


def _is_unexecuted_suspension_rejection(result: Any) -> bool:
    metadata = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
    return bool(metadata.get("suspension_rejected")) and metadata.get("executed") is False
