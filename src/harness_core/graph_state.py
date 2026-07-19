from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from harness_core.contracts import AgentMessage, Observation, RunContext, ToolCall, ToolResult
from harness_core.model import ModelGateway
from harness_core.skills import DelegationPolicy, SkillPolicy
from harness_core.tool_registry import ToolRegistry
from harness_core.workflow_policy import evaluate_workflow_completion
from harness_core.graph_workflow import (
    _emit,
    _record_completed_tool_name,
    _record_resume_artifact,
)

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
            "Waiting for user action",
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
            "Waiting for background task",
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
        "Tool execution completed" if result.status == "succeeded" else "Tool execution failed",
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
    summary = str(payload.get("summary") or "The user action is complete.")
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
    _emit(current, config, "run.resumed", "Run resumed", summary, {"observation": observation.model_dump(mode="json")})
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
