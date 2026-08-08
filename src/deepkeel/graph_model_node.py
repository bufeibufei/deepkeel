from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from deepkeel.budget import BudgetLedger
from deepkeel.contracts import AgentMessage, FinalAnswer, PendingAction, ToolCall
from deepkeel.control import RunControl
from deepkeel.deadlines import ensure_time_remaining
from deepkeel.events import AgentEventPersistenceError
from deepkeel.hooks import HookAction, HookAudit, HookInvocation, HookPoint
from deepkeel.model import ModelGateway, ModelTurn, model_tools_from_registry
from deepkeel.model_failures import ModelToolContractError
from deepkeel.skills import SkillPolicy
from deepkeel.skill_activation import EntryToolActivationRequest
from deepkeel.tool_registry import ToolRegistry
from deepkeel.tool_disclosure import resolve_tool_view
from deepkeel.tools import ToolExecutionContext, ToolExecutor
from deepkeel.turn_context import TurnContextRegistry, TurnExecutionContext
from deepkeel.type_narrowing import as_dict
from deepkeel.workflow_policy import evaluate_workflow_completion
from deepkeel.graph_state import (
    _allowed_tool_names,
    _apply_policy_confirmation_resume,
    _apply_resume_payload,
    _apply_tool_result,
    _copy_state,
    _forced_workflow_tool_name,
    _hydrate_call,
    _is_policy_confirmation,
    _is_suspending_call,
    _messages,
    _model_available_roles,
    _parallel_suspension_rejected,
    _skill_tool_parameter_overrides,
    _stable_tool_calls,
    HarnessGraphState,
    migrate_legacy_graph_state,
)
from deepkeel.graph_model_support import (
    _emit_hook_audits,
    _forced_tool_clarification_fallback,
    _latest_user_question,
    _model_hook_invocation,
)
from deepkeel.graph_model_step import (
    build_model_metrics,
    build_model_step_context,
    partition_model_tool_calls,
)
from deepkeel.graph_workflow import (
    TRUNCATED_FINISH_REASONS,
    _answer_summary,
    _complete_continued_answer,
    _config_value,
    _continue_or_fail_truncated_model_response,
    _emit,
    _finish_failed,
    _latency_ms,
    _record_completed_tool,
    _repair_or_fail_workflow,
    _retry_or_fail_empty_model_response,
    _set_policy_state,
    _wait_for_workflow_input,
    _workflow_can_wait_for_user_input,
)


def _operational_run_id(state: Mapping[str, Any]) -> str:
    metadata = as_dict(state.get("metadata"))
    return str(metadata.get("operational_run_id") or state.get("run_id") or "")


class GraphModelNodeMixin:
    """Model-step orchestration, disclosure, hooks and completion handling."""

    async def amodel_node(
        self: Any,
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
            turn_context.deadline_monotonic if turn_context is not None else self.deadline_monotonic
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
            return _finish_failed(
                current,
                "The agent reached the maximum step count without a reliable answer.",
                config,
            )
        current["status"] = "reasoning"
        _emit(
            current,
            config,
            "agent.reasoning",
            "Reasoning",
            "The agent is evaluating context and selecting the next step.",
        )
        allowed_tools = _allowed_tool_names(current, tool_registry)
        skill_policy = SkillPolicy.from_snapshot(current.get("skill_activation"))
        workflow_completion = evaluate_workflow_completion(skill_policy, current)
        workflow_is_finalizing = (
            skill_policy.active and skill_policy.durable and workflow_completion.allowed
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
                for name in as_dict(current.get("metadata")).get("discovered_tool_names", [])
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
                    before_model.decision.reason or "model invocation denied by lifecycle hook",
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
                    current.setdefault("messages", []).append(message.model_dump(mode="json"))
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
            model_step_context = build_model_step_context(
                current,
                available_roles=_model_available_roles(model),
                forced_tool_name=forced_tool_name,
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
                    state=current,
                    turn_context=turn_context,
                )
                if turn is None:
                    raise
                recovery = str(as_dict(turn.raw).get("recovery") or "forced_tool_clarification")
                _emit(
                    current,
                    config,
                    "model.tool_contract.recovered",
                    "Required tool contract recovered",
                    forced_tool_name,
                    {
                        "tool_name": forced_tool_name,
                        "recovery": recovery,
                        "visible": False,
                    },
                )
            self.ensure_active(current, config, force=True)
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            current["budget_state"] = ledger.snapshot(
                _operational_run_id(current)
            ).as_dict()
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
        current["budget_state"] = ledger.snapshot(_operational_run_id(current)).as_dict()
        model_latency_ms = int((time.perf_counter() - model_started_at) * 1000)
        model_metrics = build_model_metrics(
            turn,
            route_payload,
            latency_ms=model_latency_ms,
            first_token_latency_ms=_latency_ms(model_started_at, first_delta_at),
            delta_count=delta_index,
            delta_chars=delta_chars,
            forced_tool_name=forced_tool_name,
        )
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
        registered_tool_names = {spec.name for spec in tool_registry.list_tools()}
        tool_calls, rejected_tool_calls = partition_model_tool_calls(
            reported_tool_calls,
            workflow_is_finalizing=workflow_is_finalizing,
            exposed_tool_names=exposed_tool_names,
            registered_tool_names=registered_tool_names,
        )
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
        if tool_calls and not SkillPolicy.from_snapshot(current.get("skill_activation")).active:
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
                        item for item in current.get("messages") or [] if isinstance(item, dict)
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
                    if {(call.id, call.name) for call in replacement_calls} != {
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
        if not tool_calls and turn.finish_reason.strip().lower() in TRUNCATED_FINISH_REASONS:
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
            return _finish_failed(
                current, "The model returned no usable content for this step.", config
            )
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
                    before_answer.decision.reason or "answer finalization denied by lifecycle hook",
                    config,
                )
            final_content = str(
                before_answer.decision.model_input_patch.get("answer_markdown") or final_content
            )
        answer = FinalAnswer(
            markdown=final_content,
            summary=_answer_summary(final_content),
            model_id=turn.model_id,
            model_role=turn.model_role,
            stop_reason=turn.finish_reason or "completed",
            artifact_ids=[
                str(item.get("id") or "") for item in current.get("artifacts", []) if item.get("id")
            ],
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
