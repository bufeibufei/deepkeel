from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from harness_core.budget import BudgetLedger
from harness_core.contracts import (
    AgentMessage,
    FinalAnswer,
    Observation,
    PendingAction,
    RunContext,
    ToolCall,
    ToolResult,
    utc_now,
)
from harness_core.control import NoopRunControl, RunControl
from harness_core.deadlines import ensure_time_remaining
from harness_core.events import AgentEventPersistenceError
from harness_core.model import ModelGateway, model_tools_from_registry
from harness_core.model_routing import ModelStepContext
from harness_core.prompts import harness_system_prompt
from harness_core.skills import DelegationPolicy, SkillPolicy
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutionContext, ToolExecutor
from harness_core.workflow_policy import (
    SKILL_CONTRACT_VIOLATION,
    WorkflowCompletionDecision,
    evaluate_workflow_completion,
    workflow_repair_prompt,
    workflow_violation_message,
)


EventSink = Callable[[dict[str, Any]], None]
EMPTY_MODEL_RESPONSE = "EMPTY_MODEL_RESPONSE"
MAX_CONSECUTIVE_EMPTY_MODEL_RETRIES = 1


@dataclass(slots=True)
class HarnessGraph:
    compiled_graph: Any

    def invoke(
        self,
        context: RunContext,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> dict[str, Any]:
        return self.compiled_graph.invoke(
            _state_from_context(context),
            config=_graph_config(context.thread_id, tool_context, event_sink),
        )

    def resume(
        self,
        thread_id: str,
        resume_payload: dict[str, Any],
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> dict[str, Any]:
        return self.compiled_graph.invoke(
            Command(resume=resume_payload),
            config=_graph_config(thread_id, tool_context, event_sink),
        )

    def recover(
        self,
        thread_id: str,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> dict[str, Any]:
        """Continue an interrupted super-step from its durable checkpoint."""
        return self.compiled_graph.invoke(
            None,
            config=_graph_config(thread_id, tool_context, event_sink),
        )


def create_harness_graph(
    *,
    model: ModelGateway,
    tool_executor: ToolExecutor,
    tool_registry: ToolRegistry,
    system_prompt: str = "",
    max_steps: int = 12,
    checkpointer=None,
    budget_ledger: BudgetLedger | None = None,
    deadline_monotonic: float | None = None,
    run_control: RunControl | None = None,
) -> HarnessGraph:
    prompt = system_prompt or harness_system_prompt()
    ledger = budget_ledger or getattr(model, "budget_ledger", None) or tool_executor.budget_ledger
    control = run_control or NoopRunControl()

    def ensure_active(state: dict[str, Any], *, force: bool = False) -> None:
        control.raise_if_cancelled(str(state.get("run_id") or ""), force=force)
        ensure_time_remaining(deadline_monotonic)

    def model_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        ensure_active(state, force=True)
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
            return _finish_failed(current, "Agent 达到最大推理步数，未能形成可靠答复。", config)
        current["status"] = "reasoning"
        _emit(current, config, "agent.reasoning", "正在分析", "Agent 正在综合上下文并决定下一步。")
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

        def on_route(payload: dict[str, Any]) -> None:
            route_payload.update(payload)
            _emit(
                current,
                config,
                "model.route.selected",
                "模型路由已选择",
                str(payload.get("reason") or ""),
                {**payload, "visible": False},
            )

        def on_delta(delta: str) -> None:
            nonlocal delta_chars, delta_index, first_delta_at
            ensure_active(current)
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
            ensure_active(current, force=True)
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            current["budget_state"] = ledger.snapshot(current["run_id"]).as_dict()
            latency_ms = int((time.perf_counter() - model_started_at) * 1000)
            _emit(
                current,
                config,
                "model.failed",
                "模型调用失败",
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
            "forced_tool_name": forced_tool_name,
        }
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
            "模型响应完成",
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
            return _finish_failed(current, "模型本轮没有返回有效内容。", config)
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
            "回答完成",
            answer.summary,
            {"final_answer": current["final_answer"]},
        )
        return current

    def tool_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        ensure_active(state, force=True)
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

    def await_user_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        ensure_active(state, force=True)
        current = _copy_state(state)
        resume_payload = interrupt(current.get("pending_action") or {})
        if _is_policy_confirmation(current.get("pending_action")):
            return _apply_policy_confirmation_resume(current, resume_payload, config)
        return _apply_resume_payload(current, resume_payload, config, source="user_action")

    def await_async_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        ensure_active(state, force=True)
        current = _copy_state(state)
        resume_payload = interrupt(current.get("pending_async") or {})
        return _apply_resume_payload(current, resume_payload, config, source="async_observation")

    graph = StateGraph(dict)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_node("await_user", await_user_node)
    graph.add_node("await_async", await_async_node)
    graph.add_conditional_edges(
        START,
        _route_from_start,
        {"tools": "tools", "model": "model"},
    )
    graph.add_conditional_edges(
        "model",
        _route_after_model,
        {"tools": "tools", "model": "model", "await_user": "await_user", "end": END},
    )
    graph.add_conditional_edges(
        "tools",
        _route_after_tools,
        {
            "model": "model",
            "await_user": "await_user",
            "await_async": "await_async",
        },
    )
    graph.add_conditional_edges(
        "await_user",
        _route_after_user_resume,
        {"tools": "tools", "model": "model"},
    )
    graph.add_edge("await_async", "model")
    return HarnessGraph(compiled_graph=graph.compile(checkpointer=checkpointer))


def _state_from_context(context: RunContext) -> dict[str, Any]:
    skill = dict(context.skill_activation)
    missing = (
        skill.get("missing_requirements")
        if isinstance(skill.get("missing_requirements"), dict)
        else {}
    )
    return {
        "run_id": context.run_id,
        "thread_id": context.thread_id,
        "turn_id": context.turn_id,
        "user_id": context.user_id,
        "status": context.status.value,
        "messages": [message.model_dump(mode="json") for message in context.messages],
        "observations": [item.model_dump(mode="json") for item in context.observations],
        "pending_tool_calls": [item.model_dump(mode="json") for item in context.pending_tool_calls],
        "pending_action": context.pending_action.model_dump(mode="json") if context.pending_action else None,
        "pending_async": dict(context.pending_async) if context.pending_async else None,
        "artifacts": [item.model_dump(mode="json") for item in context.artifacts],
        "skill_activation": skill,
        "policy_phase": str(
            skill.get("policy_phase")
            or ("pending" if skill.get("kind") == "workflow" else "")
        ),
        "missing_requirements": {
            "tools": list(missing.get("tools") or []),
            "artifacts": list(missing.get("artifacts") or []),
        },
        "repair_count": int(skill.get("repair_count") or 0),
        "model_policy": dict(context.model_policy),
        "budget_state": dict(context.budget_state),
        "metadata": dict(context.metadata),
        "step_count": context.step_count,
        "events": [],
        "tool_results": [],
        "final_answer": None,
    }


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    current = dict(state)
    for name in ("messages", "observations", "artifacts", "events", "pending_tool_calls", "tool_results"):
        current[name] = list(state.get(name) or [])
    current["skill_activation"] = dict(state.get("skill_activation") or {})
    current["metadata"] = dict(state.get("metadata") or {})
    current["budget_state"] = dict(state.get("budget_state") or {})
    missing = (
        state.get("missing_requirements")
        if isinstance(state.get("missing_requirements"), dict)
        else {}
    )
    current["missing_requirements"] = {
        "tools": list(missing.get("tools") or []),
        "artifacts": list(missing.get("artifacts") or []),
    }
    return current


def _messages(state: dict[str, Any]) -> list[AgentMessage]:
    return [AgentMessage.model_validate(item) for item in state.get("messages", [])]


def _model_available_roles(model: ModelGateway) -> tuple[str, ...]:
    providers = getattr(model, "providers", None)
    if isinstance(providers, dict) and providers:
        return tuple(str(role) for role in providers if role)
    provider = getattr(model, "provider", None)
    role = str(getattr(provider, "model_role", "") or "reasoning")
    return (role,)


def _allowed_tool_names(
    state: dict[str, Any],
    registry: ToolRegistry,
) -> set[str] | None:
    skill = state.get("skill_activation") if isinstance(state.get("skill_activation"), dict) else {}
    allowed = skill.get("allowed_tools") if isinstance(skill.get("allowed_tools"), list) else []
    if allowed:
        names = {str(name) for name in allowed if name}
        if "agent.delegate" in names and not DelegationPolicy.from_snapshot(skill).enabled:
            names.remove("agent.delegate")
        return names
    if skill.get("skill_id"):
        return set()
    default_tools = {
        spec.name
        for spec in registry.list_tools()
        if str(spec.runtime_policy.get("model_exposure") or "always") != "skill_only"
    }
    return default_tools


def _forced_workflow_tool_name(
    state: dict[str, Any],
    tools: list[dict[str, Any]],
) -> str:
    phase = str(state.get("policy_phase") or "")
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    resumed_after_clarification = (
        phase == "waiting_user_input"
        and int(metadata.get("workflow_clarification_resume_count") or 0) > 0
    )
    if phase != "repair" and not resumed_after_clarification:
        return ""
    skill = SkillPolicy.from_snapshot(state.get("skill_activation"))
    decision = evaluate_workflow_completion(skill, state)
    missing_tools = decision.missing_tools
    available = {
        str((item.get("function") or {}).get("name") or "")
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    return next(
        (
            str(name)
            for name in missing_tools or []
            if str(name) and str(name) in available
        ),
        "",
    )


def _skill_tool_parameter_overrides(
    state: dict[str, Any],
    registry: ToolRegistry,
) -> dict[str, dict[str, Any]]:
    skill = state.get("skill_activation") if isinstance(state.get("skill_activation"), dict) else {}
    policy = DelegationPolicy.from_snapshot(skill)
    if not policy.enabled:
        return {}
    try:
        spec = registry.get("agent.delegate")
    except KeyError:
        return {}
    formal_schema = getattr(spec, "formal_parameters_schema", None)
    schema = formal_schema() if callable(formal_schema) else {}
    if (
        not isinstance(schema, dict)
        or schema.get("type") != "object"
        or not isinstance(schema.get("properties"), dict)
    ):
        return {}
    schema = deepcopy(schema)
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    concurrency = properties.get("max_concurrency")
    if isinstance(concurrency, dict):
        concurrency["maximum"] = policy.max_concurrency
        concurrency["default"] = min(
            int(concurrency.get("default") or policy.max_concurrency),
            policy.max_concurrency,
        )
    tasks = properties.get("tasks")
    if isinstance(tasks, dict):
        tasks["maxItems"] = policy.max_tasks
        items = tasks.get("items") if isinstance(tasks.get("items"), dict) else {}
        task_properties = (
            items.get("properties") if isinstance(items.get("properties"), dict) else {}
        )
        agent_id = task_properties.get("agent_id")
        if isinstance(agent_id, dict) and policy.allowed_agents:
            agent_id["enum"] = sorted(policy.allowed_agents)
    return {"agent.delegate": schema}


def _hydrate_call(raw: dict[str, Any], registry: ToolRegistry, run_id: str) -> ToolCall:
    call = ToolCall.model_validate(raw)
    try:
        spec = registry.get(call.name)
    except KeyError:
        return call.model_copy(update={"idempotency_key": call.idempotency_key or f"{run_id}:{call.id}"})
    return call.model_copy(
        update={
            "idempotency_key": call.idempotency_key or f"{run_id}:{call.id}",
            "read_only": spec.read_only,
            "parallel_safe": spec.parallel_safe,
            "resource_key": call.resource_key or str(spec.runtime_policy.get("side_effect") or spec.name),
        }
    )


def _stable_tool_calls(
    calls: list[ToolCall],
    *,
    run_id: str,
    step_index: int,
) -> list[ToolCall]:
    stable: list[ToolCall] = []
    for ordinal, call in enumerate(calls):
        if call.idempotency_key:
            stable.append(call)
            continue
        identity = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        stable.append(
            call.model_copy(
                update={
                    "idempotency_key": (
                        f"{run_id}:step:{max(1, int(step_index))}:"
                        f"tool:{ordinal}:{digest}"
                    )
                }
            )
        )
    return stable


def _is_suspending_call(call: ToolCall, registry: ToolRegistry) -> bool:
    try:
        spec = registry.get(call.name)
    except KeyError:
        return False
    return bool(spec.requires_user_action or spec.async_tool)


def _parallel_suspension_rejected(call: ToolCall, run_id: str) -> ToolResult:
    summary = (
        "A model turn may request at most one tool that suspends execution. "
        "Retry these tools sequentially in separate turns."
    )
    return ToolResult(
        call=call,
        status="failed",
        summary=summary,
        error=summary,
        retryable=True,
        observation=Observation(
            id=f"{call.id}:parallel-suspension-rejected",
            run_id=run_id,
            tool_call_id=call.id,
            source=call.name,
            status="failed",
            summary=summary,
            error=summary,
            metadata={"error_code": "MULTIPLE_SUSPENDING_TOOLS"},
        ),
        metadata={"error_code": "MULTIPLE_SUSPENDING_TOOLS"},
    )


def _apply_tool_result(current: dict[str, Any], result: ToolResult, config: dict[str, Any]) -> None:
    if result.observation is not None:
        current.setdefault("observations", []).append(result.observation.model_dump(mode="json"))
    for artifact in result.artifacts:
        current.setdefault("artifacts", []).append(artifact.model_dump(mode="json"))
    if result.status == "requires_user_action" and result.pending_action is not None:
        current["pending_action"] = result.pending_action.model_dump(mode="json")
        waiting_for_input = result.pending_action.action_type == "clarification"
        _emit(
            current,
            config,
            "tool.requires_user_action",
            "等待用户操作",
            result.summary,
            {
                "tool_result": _tool_result_payload(result),
                "pending_action": current["pending_action"],
                "interaction_mode": "text_input" if waiting_for_input else "action",
                "visible": not waiting_for_input,
            },
        )
        return
    if result.status == "waiting_async":
        current["pending_async"] = {
            "tool_call_id": result.tool_call_id,
            "tool_name": result.name,
            "summary": result.summary,
            "data": result.data,
        }
        _emit(
            current,
            config,
            "tool.waiting_async",
            "等待后台任务",
            result.summary,
            {"tool_result": _tool_result_payload(result)},
        )
        return
    current.setdefault("messages", []).append(_tool_message(result).model_dump(mode="json"))
    event_type = "tool.completed" if result.status == "succeeded" else "tool.failed"
    _emit(
        current,
        config,
        event_type,
        "工具执行完成" if result.status == "succeeded" else "工具执行失败",
        result.summary or result.error,
        {
            "tool_result": _tool_result_payload(result),
            "visible": result.metadata.get("visible") is not False,
        },
    )


def _apply_resume_payload(
    current: dict[str, Any],
    resume_payload: Any,
    config: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    payload = resume_payload if isinstance(resume_payload, dict) else {"value": resume_payload}
    pending = current.get("pending_action") or current.get("pending_async") or {}
    is_clarification = str(pending.get("action_type") or "") == "clarification"
    tool_call_id = str(pending.get("tool_call_id") or "")
    tool_name = str(
        pending.get("tool_name")
        or payload.get("tool_name")
        or pending.get("action_type")
        or source
    )
    status = str(payload.get("status") or "succeeded")
    summary = str(payload.get("summary") or "用户操作已完成。")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    observation = Observation(
        id=f"resume-{uuid4()}",
        run_id=current["run_id"],
        tool_call_id=tool_call_id,
        source=tool_name,
        status="failed" if status == "failed" else "succeeded",
        summary=summary,
        data=data,
        error=str(payload.get("error") or "") if status == "failed" else "",
        metadata={"resume_source": source},
    )
    current.setdefault("observations", []).append(observation.model_dump(mode="json"))
    if is_clarification:
        clarification = str(
            data.get("clarification_answer")
            if isinstance(data, dict)
            else ""
        ).strip() or summary
        metadata = current.setdefault("metadata", {})
        metadata["workflow_clarification_resume_count"] = (
            int(metadata.get("workflow_clarification_resume_count") or 0) + 1
        )
        missing = current.get("missing_requirements")
        missing_tools = missing.get("tools") if isinstance(missing, dict) else []
        if missing_tools:
            current.setdefault("messages", []).append(
                AgentMessage(
                    id=f"workflow-resume-{uuid4()}",
                    role="system",
                    content=(
                        "The user has supplied the requested clarification. Continue the active "
                        "workflow now. Do not present a business result directly before satisfying "
                        f"the required tool transition(s): {', '.join(str(name) for name in missing_tools)}."
                    ),
                    metadata={
                        "kind": "workflow_clarification_resume_guard",
                        "missing_tools": [str(name) for name in missing_tools],
                    },
                ).model_dump(mode="json")
            )
        current.setdefault("messages", []).append(
            AgentMessage(
                id=f"user-resume-{uuid4()}",
                role="user",
                content=clarification,
                metadata={"resume_source": source, "clarification": True},
            ).model_dump(mode="json")
        )
    else:
        current.setdefault("messages", []).append(
            AgentMessage(
                id=f"tool-resume-{uuid4()}",
                role="tool",
                tool_call_id=tool_call_id,
                content=json.dumps(
                    {"status": observation.status, "summary": summary, "data": data},
                    ensure_ascii=False,
                ),
            ).model_dump(mode="json")
        )
    if observation.status == "succeeded":
        if not is_clarification:
            _record_completed_tool_name(current, tool_name)
        _record_resume_artifact(current, payload)
    current["pending_action"] = None
    current["pending_async"] = None
    current["status"] = "reasoning"
    _emit(current, config, "run.resumed", "继续执行", summary, {"observation": observation.model_dump(mode="json")})
    return current


def _is_policy_confirmation(pending: Any) -> bool:
    if not isinstance(pending, dict):
        return False
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    return pending.get("action_type") == "policy_confirmation" or payload.get("policy_confirmation") is True


def _apply_policy_confirmation_resume(
    current: dict[str, Any],
    resume_payload: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    resolution = resume_payload if isinstance(resume_payload, dict) else {"value": resume_payload}
    data = resolution.get("data") if isinstance(resolution.get("data"), dict) else {}
    status = str(resolution.get("status") or "").lower()
    confirmed = data.get("confirmed") is True or resolution.get("confirmed") is True or status in {
        "confirmed",
        "approved",
    }
    pending = current.get("pending_action") if isinstance(current.get("pending_action"), dict) else {}
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    deferred = payload.get("deferred_tool_call") if isinstance(payload.get("deferred_tool_call"), dict) else {}
    tool_call_id = str(pending.get("tool_call_id") or deferred.get("id") or "")
    tool_name = str(deferred.get("name") or payload.get("tool_name") or "policy_confirmation")
    current["pending_action"] = None
    current["pending_async"] = None
    if confirmed and deferred:
        current["pending_tool_calls"] = [deferred]
        current["status"] = "executing_tools"
        metadata = current.setdefault("metadata", {})
        grants = metadata.get("confirmation_grants") if isinstance(metadata.get("confirmation_grants"), dict) else {}
        metadata["confirmation_grants"] = {
            **grants,
            tool_call_id: {
                "confirmed": True,
                "run_id": current["run_id"],
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "policy_id": str((payload.get("policy_decision") or {}).get("policy_id") or ""),
            },
        }
        _emit(
            current,
            config,
            "policy.confirmed",
            "Tool execution confirmed",
            str(resolution.get("summary") or tool_name),
            {"tool_call": deferred, "policy_decision": payload.get("policy_decision") or {}},
        )
        return current

    summary = str(resolution.get("summary") or "Tool execution was canceled by the user.")
    observation = Observation(
        id=f"policy-canceled-{uuid4()}",
        run_id=current["run_id"],
        tool_call_id=tool_call_id,
        source=tool_name,
        status="failed",
        summary=summary,
        error=summary,
        metadata={"policy_confirmation_canceled": True},
    )
    current.setdefault("observations", []).append(observation.model_dump(mode="json"))
    current.setdefault("messages", []).append(
        AgentMessage(
            id=f"tool-policy-canceled-{uuid4()}",
            role="tool",
            tool_call_id=tool_call_id,
            name=tool_name,
            content=json.dumps(
                {"status": "failed", "summary": summary, "canceled": True},
                ensure_ascii=False,
            ),
        ).model_dump(mode="json")
    )
    current["pending_tool_calls"] = []
    current["status"] = "reasoning"
    _emit(
        current,
        config,
        "policy.confirmation_canceled",
        "Tool execution canceled",
        summary,
        {"observation": observation.model_dump(mode="json")},
    )
    return current


def _tool_message(result: ToolResult) -> AgentMessage:
    return AgentMessage(
        id=f"tool-{uuid4()}",
        role="tool",
        tool_call_id=result.tool_call_id,
        name=result.name,
        content=json.dumps(
            {
                "status": result.status,
                "outcome": result.outcome,
                "summary": result.summary,
                "data": result.data,
                "error": result.error,
            },
            ensure_ascii=False,
        ),
    )


def _tool_result_payload(result: ToolResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json", exclude={"call", "metadata"})
    if result.call is not None:
        payload["call"] = result.call.model_dump(mode="json")
    runtime_metrics = result.metadata.get("runtime_metrics") if isinstance(result.metadata, dict) else None
    if isinstance(runtime_metrics, dict):
        payload["runtime_metrics"] = runtime_metrics
    mcp = result.metadata.get("mcp") if isinstance(result.metadata, dict) else None
    if isinstance(mcp, dict):
        payload["mcp"] = mcp
    governance = result.metadata.get("governance") if isinstance(result.metadata, dict) else None
    if isinstance(governance, dict):
        payload["metadata"] = {"governance": governance}
    diagnostics = {
        key: result.metadata[key]
        for key in (
            "error_code",
            "schema_validation_error",
            "executed",
        )
        if key in result.metadata
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def _route_after_tools(state: dict[str, Any]) -> str:
    if state.get("status") == "waiting_user":
        return "await_user"
    if state.get("status") == "waiting_async":
        return "await_async"
    return "model"


def _route_from_start(state: dict[str, Any]) -> str:
    return "tools" if state.get("pending_tool_calls") else "model"


def _route_after_user_resume(state: dict[str, Any]) -> str:
    return "tools" if state.get("pending_tool_calls") else "model"


def _route_after_model(state: dict[str, Any]) -> str:
    if state.get("pending_tool_calls"):
        return "tools"
    if state.get("status") == "waiting_user":
        return "await_user"
    if state.get("status") == "reasoning" and state.get("policy_phase") == "repair":
        return "model"
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    if state.get("status") == "reasoning" and metadata.get("empty_model_retry_pending"):
        return "model"
    return "end"


def _retry_or_fail_empty_model_response(
    current: dict[str, Any],
    config: dict[str, Any],
    *,
    can_retry: bool,
) -> dict[str, Any]:
    metadata = current.setdefault("metadata", {})
    empty_count = int(metadata.get("consecutive_empty_model_responses") or 0) + 1
    metadata["consecutive_empty_model_responses"] = empty_count
    if can_retry and empty_count <= MAX_CONSECUTIVE_EMPTY_MODEL_RETRIES:
        metadata["empty_model_retry_pending"] = True
        current.setdefault("messages", []).append(
            AgentMessage(
                id=f"model-empty-repair-{uuid4()}",
                role="system",
                content=(
                    "上一轮模型返回了空内容且没有工具调用。请重新完成当前步骤："
                    "需要调用工具时输出合法工具调用，否则直接给出有效答复。"
                ),
                metadata={
                    "kind": "empty_model_response_repair",
                    "retry_count": empty_count,
                },
            ).model_dump(mode="json")
        )
        current["status"] = "reasoning"
        current["pending_tool_calls"] = []
        _emit(
            current,
            config,
            "model.empty_response.retrying",
            "模型空响应重试",
            "模型未返回有效内容，正在自动重试。",
            {
                "error_code": EMPTY_MODEL_RESPONSE,
                "retry_count": empty_count,
                "retry_limit": MAX_CONSECUTIVE_EMPTY_MODEL_RETRIES,
                "visible": False,
            },
        )
        return current

    metadata.pop("empty_model_retry_pending", None)
    message = "模型服务连续未返回有效内容，本轮已安全结束，请重新尝试。"
    metadata["runtime_error"] = {
        "type": "EmptyModelResponse",
        "code": EMPTY_MODEL_RESPONSE,
        "category": "upstream",
        "retryable": True,
        "message": message,
        "user_message": message,
    }
    _emit(
        current,
        config,
        "model.empty_response.exhausted",
        "模型空响应",
        message,
        {
            "error_code": EMPTY_MODEL_RESPONSE,
            "retry_count": empty_count,
            "retry_limit": MAX_CONSECUTIVE_EMPTY_MODEL_RETRIES,
            "visible": False,
        },
    )
    return _finish_failed(
        current,
        message,
        config,
        error_code=EMPTY_MODEL_RESPONSE,
    )


def _workflow_can_wait_for_user_input(
    skill: SkillPolicy,
    decision: WorkflowCompletionDecision,
    state: dict[str, Any],
) -> bool:
    if not skill.active or not skill.durable:
        return False
    if str(skill.completion_policy.get("clarification_strategy") or "model") == "tool_contract":
        # Contract-driven workflows must attempt the required tool. Missing fields
        # then come from ToolSpec.required_args instead of arbitrary model prose.
        return False
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    if int(metadata.get("workflow_clarification_resume_count") or 0) > 0:
        # Once the user has answered the workflow clarification, a model-only final
        # response must go through policy repair instead of becoming another prompt.
        # Tool-level validation can still suspend again with its own clarification.
        return False
    if (
        not decision.missing_tools
        and not decision.missing_tool_groups
    ) or str(state.get("policy_phase") or "") == "repair":
        return False
    if skill.completion_policy.get("allow_model_clarification") is not True:
        return False
    waiting_statuses = skill.completion_policy.get("waiting_statuses")
    if not isinstance(waiting_statuses, (list, tuple, set, frozenset)):
        return False
    return "waiting_user_input" in {str(status).strip() for status in waiting_statuses}


def _wait_for_workflow_input(
    current: dict[str, Any],
    skill: SkillPolicy,
    decision: WorkflowCompletionDecision,
    prompt: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    question = str(prompt or "").strip()
    pending = PendingAction(
        id=f"clarification-{uuid4()}",
        run_id=str(current.get("run_id") or ""),
        action_type="clarification",
        title="需要补充信息",
        prompt=question,
        payload={
            "state": "waiting_user_input",
            "skill_id": skill.skill_id,
            "question": question,
            "missing_requirements": decision.diagnostics(),
        },
    )
    current["pending_action"] = pending.model_dump(mode="json")
    current["pending_tool_calls"] = []
    current["status"] = "waiting_user"
    _set_policy_state(current, phase="waiting_user_input", decision=decision)
    _emit(
        current,
        config,
        "skill.waiting_user_input",
        "等待补充信息",
        question,
        {
            "skill_id": skill.skill_id,
            "pending_action": current["pending_action"],
            "missing_requirements": decision.diagnostics(),
            "visible": False,
        },
    )
    return current


def _repair_or_fail_workflow(
    current: dict[str, Any],
    skill: SkillPolicy,
    decision: WorkflowCompletionDecision,
    config: dict[str, Any],
) -> dict[str, Any]:
    repair_count = int(current.get("repair_count") or 0)
    if repair_count < skill.policy_repair_limit:
        repair_count += 1
        _set_policy_state(current, phase="repair", decision=decision, repair_count=repair_count)
        repair_prompt = workflow_repair_prompt(decision)
        current.setdefault("messages", []).append(
            AgentMessage(
                id=f"workflow-policy-{uuid4()}",
                role="system",
                content=repair_prompt,
                metadata={
                    "kind": "workflow_policy_repair",
                    "repair_count": repair_count,
                    "missing_requirements": decision.diagnostics(),
                },
            ).model_dump(mode="json")
        )
        current["status"] = "reasoning"
        current["pending_tool_calls"] = []
        _emit(
            current,
            config,
            "skill.policy_repair",
            "Workflow Skill policy repair",
            repair_prompt,
            {
                "policy_phase": "repair",
                "missing_requirements": decision.diagnostics(),
                "repair_count": repair_count,
            },
        )
        return current
    return _finish_skill_contract_violation(current, decision, config)


def _finish_skill_contract_violation(
    current: dict[str, Any],
    decision: WorkflowCompletionDecision,
    config: dict[str, Any],
) -> dict[str, Any]:
    message = workflow_violation_message(decision)
    _set_policy_state(current, phase="failed", decision=decision)
    answer = FinalAnswer(
        markdown=message,
        summary=message,
        status="failed",
        stop_reason="skill_contract_violation",
        metadata={
            "error_code": SKILL_CONTRACT_VIOLATION,
            "missing_requirements": decision.diagnostics(),
        },
    )
    current["final_answer"] = answer.model_dump(mode="json")
    current["status"] = "failed"
    current["pending_tool_calls"] = []
    current["metadata"]["runtime_error"] = {
        "type": SKILL_CONTRACT_VIOLATION,
        "code": SKILL_CONTRACT_VIOLATION,
        "message": message,
        "missing_requirements": decision.diagnostics(),
    }
    _emit(
        current,
        config,
        "agent.failed",
        "Workflow Skill contract violation",
        message,
        {
            "error_code": SKILL_CONTRACT_VIOLATION,
            "final_answer": current["final_answer"],
            "missing_requirements": decision.diagnostics(),
        },
    )
    return current


def _set_policy_state(
    current: dict[str, Any],
    *,
    phase: str,
    decision: WorkflowCompletionDecision,
    repair_count: int | None = None,
) -> None:
    current["policy_phase"] = phase
    current["missing_requirements"] = decision.diagnostics()
    if repair_count is not None:
        current["repair_count"] = repair_count
    skill = dict(current.get("skill_activation") or {})
    skill.update(
        {
            "policy_phase": phase,
            "missing_requirements": decision.diagnostics(),
            "repair_count": int(current.get("repair_count") or 0),
        }
    )
    current["skill_activation"] = skill


def _record_completed_tool(current: dict[str, Any], result: ToolResult) -> None:
    if result.status != "succeeded":
        return
    _record_completed_tool_name(current, result.name)


def _record_completed_tool_name(current: dict[str, Any], tool_name: str) -> None:
    skill = dict(current.get("skill_activation") or {})
    completed = {str(name) for name in skill.get("completed_tools", []) if name}
    completed.add(tool_name)
    skill["completed_tools"] = sorted(completed)
    current["skill_activation"] = skill


def _record_resume_artifact(current: dict[str, Any], payload: dict[str, Any]) -> None:
    artifact_type = str(payload.get("artifact_type") or "").strip()
    if not artifact_type:
        return
    artifact_id = str(
        payload.get("artifact_id")
        or payload.get("session_id")
        or payload.get("case_id")
        or payload.get("run_id")
        or f"resume-artifact-{uuid4()}"
    )
    artifacts = current.setdefault("artifacts", [])
    existing = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict) and str(item.get("id") or "") == artifact_id
        ),
        None,
    )
    if existing is not None:
        existing["summary"] = str(payload.get("summary") or existing.get("summary") or "")
        existing["data"] = {**dict(existing.get("data") or {}), **payload}
        existing["metadata"] = {
            **dict(existing.get("metadata") or {}),
            "resume_observation": True,
        }
        return
    artifacts.append(
        {
            "id": artifact_id,
            "run_id": current["run_id"],
            "artifact_type": artifact_type,
            "title": str(payload.get("title") or ""),
            "summary": str(payload.get("summary") or ""),
            "source_id": str(payload.get("session_id") or payload.get("case_id") or ""),
            "data": dict(payload),
            "created_at": utc_now().isoformat(),
            "metadata": {"resume_observation": True},
        }
    )


def _finish_failed(
    current: dict[str, Any],
    message: str,
    config: dict[str, Any],
    *,
    error_code: str = "",
) -> dict[str, Any]:
    answer = FinalAnswer(
        markdown=message,
        summary=message,
        status="failed",
        stop_reason=error_code.lower() if error_code else "runtime_failed",
        metadata={"error_code": error_code} if error_code else {},
    )
    current["final_answer"] = answer.model_dump(mode="json")
    current["status"] = "failed"
    current["pending_tool_calls"] = []
    _emit(current, config, "agent.failed", "执行失败", message, {"final_answer": current["final_answer"]})
    return current


def _emit(
    state: dict[str, Any],
    config: dict[str, Any],
    event_type: str,
    title: str,
    summary: str,
    payload: dict[str, Any] | None = None,
    *,
    ephemeral: bool = False,
) -> None:
    event = {
        "event_type": event_type,
        "title": title,
        "summary": summary,
        "payload": payload or {},
        "ephemeral": ephemeral,
        "created_at": utc_now().isoformat(),
    }
    if not ephemeral:
        event["sequence"] = len(state.get("events") or []) + 1
        state.setdefault("events", []).append(event)
    sink = _config_value(config, "event_sink")
    if callable(sink):
        sink(event)


def _latency_ms(started_at: float, completed_at: float | None) -> int | None:
    if completed_at is None:
        return None
    return max(0, int((completed_at - started_at) * 1000))


def _config_value(config: dict[str, Any], key: str) -> Any:
    configurable = config.get("configurable") if isinstance(config.get("configurable"), dict) else {}
    return configurable.get(key)


def _graph_config(
    thread_id: str,
    tool_context: ToolExecutionContext,
    event_sink: EventSink | None,
) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "tool_context": tool_context,
            "event_sink": event_sink,
        }
    }


def _answer_summary(markdown: str, limit: int = 240) -> str:
    compact = " ".join(str(markdown or "").split())
    return compact if len(compact) <= limit else f"{compact[:limit].rstrip()}…"
