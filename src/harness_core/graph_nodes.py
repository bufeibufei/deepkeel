from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from harness_core.budget import BudgetLedger
from harness_core.contracts import AgentMessage, FinalAnswer
from harness_core.control import RunControl
from harness_core.deadlines import ensure_time_remaining
from harness_core.events import AgentEventPersistenceError
from harness_core.model import ModelGateway, model_tools_from_registry
from harness_core.model_routing import ModelStepContext
from harness_core.skills import SkillPolicy
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutionContext, ToolExecutor
from harness_core.workflow_policy import evaluate_workflow_completion
from harness_core.graph_state import (
    _allowed_tool_names, _apply_policy_confirmation_resume, _apply_resume_payload,
    _apply_tool_result, _copy_state, _forced_workflow_tool_name, _hydrate_call,
    _is_policy_confirmation, _is_suspending_call, _messages, _model_available_roles,
    _parallel_suspension_rejected, _skill_tool_parameter_overrides, _stable_tool_calls,
    HarnessGraphState, migrate_legacy_graph_state,
)
from harness_core.graph_workflow import (
    _answer_summary, _config_value, _emit, _finish_failed, _latency_ms,
    _record_completed_tool, _repair_or_fail_workflow,
    _retry_or_fail_empty_model_response, _set_policy_state,
    _wait_for_workflow_input, _workflow_can_wait_for_user_input,
)

class GraphNodes:
    def __init__(
        self, *, model: ModelGateway, tool_executor: ToolExecutor,
        tool_registry: ToolRegistry, prompt: str, max_steps: int,
        ledger: BudgetLedger, deadline_monotonic: float | None, control: RunControl,
    ) -> None:
        self.model = model
        self.tool_executor = tool_executor
        self.tool_registry = tool_registry
        self.prompt = prompt
        self.max_steps = max_steps
        self.ledger = ledger
        self.deadline_monotonic = deadline_monotonic
        self.control = control

    def ensure_active(self, state: dict[str, Any], *, force: bool = False) -> None:
        self.control.raise_if_cancelled(str(state.get("run_id") or ""), force=force)
        ensure_time_remaining(self.deadline_monotonic)

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
        state = self.normalize_state(state, config)
        if not state["messages"]:
            raise ValueError("graph checkpoint cannot resume model execution without messages")
        model = self.model
        tool_registry = self.tool_registry
        prompt = self.prompt
        max_steps = self.max_steps
        ledger = self.ledger
        deadline_monotonic = self.deadline_monotonic
        self.ensure_active(state, force=True)
        current = _copy_state(state)
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
        tools = model_tools_from_registry(
            tool_registry,
            allowed_names=allowed_tools,
            parameter_overrides=_skill_tool_parameter_overrides(current, tool_registry),
        )
        forced_tool_name = _forced_workflow_tool_name(current, tools)
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
            self.ensure_active(current)
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
            turn = model.run_turn(
                _messages(current),
                tools=tools,
                system_prompt=prompt,
                # Forced workflow transitions only extract tool arguments. Their
                # provisional prose must not be shown as a user-facing answer.
                on_text_delta=None if forced_tool_name else on_delta,
                step_context=ModelStepContext(
                    run_id=str(current.get("run_id") or ""),
                    user_id=str(current.get("user_id") or ""),
                    thread_id=str(current.get("thread_id") or ""),
                    turn_id=str(current.get("turn_id") or ""),
                    step_index=int(current.get("step_count") or 0),
                    message_count=len(current.get("messages") or []),
                    observation_count=len(current.get("observations") or []),
                    tool_result_count=len(current.get("tool_results") or []),
                    available_roles=_model_available_roles(model),
                    model_policy=dict(current.get("model_policy") or {}),
                    skill_activation=dict(current.get("skill_activation") or {}),
                    policy_phase=str(current.get("policy_phase") or ""),
                    forced_tool_name=forced_tool_name,
                    governance_scope=dict(
                        (current.get("metadata") or {}).get("governance_scope") or {}
                    ),
                    deadline_monotonic=deadline_monotonic,
                ),
                on_route=on_route,
            )
            self.ensure_active(current, force=True)
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
        current["budget_state"] = ledger.snapshot(current["run_id"]).as_dict()
        model_latency_ms = int((time.perf_counter() - model_started_at) * 1000)
        model_metrics = {
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
            str(model_metrics["usage"].get("source") or ""),
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
        tool_calls = _stable_tool_calls(
            turn.tool_calls,
            run_id=str(current.get("run_id") or ""),
            step_index=int(current["step_count"]),
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
        if not turn.tool_calls and not turn.content.strip():
            # Do not poison the next provider request with an empty assistant
            # message when retrying a transient successful-but-empty response.
            current["messages"].pop()
            return _retry_or_fail_empty_model_response(
                current,
                config,
                can_retry=int(current.get("step_count") or 0) < max_steps,
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
        answer = FinalAnswer(
            markdown=turn.content,
            summary=_answer_summary(turn.content),
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
        state = self.normalize_state(state, config)
        tool_registry = self.tool_registry
        tool_executor = self.tool_executor
        ledger = self.ledger
        self.ensure_active(state, force=True)
        current = _copy_state(state)
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
            results = tool_executor.execute_many(calls, context)
        current["budget_state"] = ledger.snapshot(current["run_id"]).as_dict()
        current["pending_tool_calls"] = []
        current["pending_action"] = None
        current["pending_async"] = None
        for result in results:
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
        state = self.normalize_state(state, config)
        self.ensure_active(state, force=True)
        current = _copy_state(state)
        resume_payload = interrupt(current.get("pending_action") or {})
        if _is_policy_confirmation(current.get("pending_action")):
            return _apply_policy_confirmation_resume(current, resume_payload, config)
        return _apply_resume_payload(current, resume_payload, config, source="user_action")

    def await_async_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        state = self.normalize_state(state, config)
        self.ensure_active(state, force=True)
        current = _copy_state(state)
        resume_payload = interrupt(current.get("pending_async") or {})
        return _apply_resume_payload(current, resume_payload, config, source="async_observation")
