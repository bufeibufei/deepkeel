from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections.abc import Mapping
from typing import Any, TypedDict, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from deepkeel.contracts import (
    AgentMessage,
    MessageContentPart,
    Observation,
    RunContext,
    ToolCall,
    ToolResult,
)
from deepkeel.model import ModelGateway
from deepkeel.skills import DelegationPolicy, SkillPolicy
from deepkeel.tool_registry import ToolRegistry
from deepkeel.type_narrowing import as_dict, as_list
from deepkeel.workflow_policy import evaluate_workflow_completion
from deepkeel.graph_workflow import (
    _emit,
    _record_completed_tool_name,
    _record_resume_artifact,
)


class MissingRequirementsState(TypedDict):
    tools: list[str]
    artifacts: list[str]


class HarnessGraphState(TypedDict):
    run_id: str
    thread_id: str
    turn_id: str
    user_id: str
    status: str
    messages: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    pending_action: dict[str, Any] | None
    pending_async: dict[str, Any] | None
    artifacts: list[dict[str, Any]]
    skill_activation: dict[str, Any]
    policy_phase: str
    missing_requirements: MissingRequirementsState
    repair_count: int
    model_policy: dict[str, Any]
    budget_state: dict[str, Any]
    metadata: dict[str, Any]
    step_count: int
    events: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    final_answer: dict[str, Any] | None


def validate_graph_state(state: dict[str, Any]) -> HarnessGraphState:
    """Fail fast when a graph node produces a structurally impossible state."""

    for field_name in ("run_id", "thread_id", "turn_id", "user_id", "status"):
        if not isinstance(state.get(field_name), str) or not state[field_name]:
            raise ValueError(f"graph state requires non-empty {field_name}")
    for field_name in (
        "messages", "observations", "pending_tool_calls", "artifacts", "events", "tool_results"
    ):
        if not isinstance(state.get(field_name), list):
            raise TypeError(f"graph state {field_name} must be a list")
    for field_name in ("skill_activation", "model_policy", "budget_state", "metadata"):
        if not isinstance(state.get(field_name), dict):
            raise TypeError(f"graph state {field_name} must be an object")
    if state.get("pending_action") is not None and state.get("pending_async") is not None:
        raise ValueError("graph state cannot wait for user action and async work simultaneously")
    if int(state.get("step_count") or 0) < 0 or int(state.get("repair_count") or 0) < 0:
        raise ValueError("graph counters must be non-negative")
    missing = state.get("missing_requirements")
    if not isinstance(missing, dict) or not isinstance(missing.get("tools"), list) or not isinstance(missing.get("artifacts"), list):
        raise TypeError("graph state missing_requirements must contain tools and artifacts lists")
    return cast(HarnessGraphState, state)


def migrate_legacy_graph_state(
    state: dict[str, Any],
    *,
    thread_id: str,
) -> HarnessGraphState:
    """Normalize durable v2 checkpoints before applying current invariants."""

    migrated = dict(state)
    metadata = as_dict(migrated.get("metadata"))
    run_id = str(migrated.get("run_id") or metadata.get("run_id") or thread_id)
    migrated.update(
        {
            "run_id": run_id,
            "thread_id": str(migrated.get("thread_id") or thread_id),
            "turn_id": str(migrated.get("turn_id") or metadata.get("turn_id") or run_id),
            "user_id": str(migrated.get("user_id") or metadata.get("user_id") or "checkpoint-user"),
            "status": str(migrated.get("status") or "reasoning"),
            "metadata": metadata,
        }
    )
    for field_name in (
        "messages", "observations", "pending_tool_calls", "artifacts", "events", "tool_results"
    ):
        migrated[field_name] = list(migrated.get(field_name) or [])
    for field_name in ("skill_activation", "model_policy", "budget_state"):
        migrated[field_name] = dict(migrated.get(field_name) or {})
    missing = as_dict(migrated.get("missing_requirements"))
    migrated["missing_requirements"] = {
        "tools": list(missing.get("tools") or []),
        "artifacts": list(missing.get("artifacts") or []),
    }
    migrated["policy_phase"] = str(migrated.get("policy_phase") or "")
    migrated["repair_count"] = int(migrated.get("repair_count") or 0)
    migrated["step_count"] = int(migrated.get("step_count") or 0)
    migrated.setdefault("pending_action", None)
    migrated.setdefault("pending_async", None)
    migrated.setdefault("final_answer", None)
    return validate_graph_state(migrated)

def _state_from_context(context: RunContext) -> HarnessGraphState:
    skill = dict(context.skill_activation)
    missing = as_dict(skill.get("missing_requirements"))
    return validate_graph_state({
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
    })


def _copy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    current = dict(state)
    for name in ("messages", "observations", "artifacts", "events", "pending_tool_calls", "tool_results"):
        current[name] = list(state.get(name) or [])
    current["skill_activation"] = dict(state.get("skill_activation") or {})
    current["metadata"] = dict(state.get("metadata") or {})
    current["budget_state"] = dict(state.get("budget_state") or {})
    missing = as_dict(state.get("missing_requirements"))
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
    skill = as_dict(state.get("skill_activation"))
    skill_policy = SkillPolicy.from_snapshot(skill)
    allowed = as_list(skill.get("allowed_tools"))
    if skill_policy.active and skill_policy.tool_scope_mode == "allowlist":
        names = {str(name) for name in allowed if name}
        if "agent.delegate" in names and not DelegationPolicy.from_snapshot(skill).enabled:
            names.remove("agent.delegate")
        if names and _tool_discovery_available(state):
            names.add("runtime.discover_tools")
        return names
    default_tools = {
        spec.name
        for spec in registry.list_tools()
        if str(spec.runtime_policy.get("model_exposure") or "always") != "skill_only"
    }
    if skill_policy.active:
        default_tools.update(skill_policy.required_tools)
        for group in skill_policy.required_tool_groups:
            default_tools.update(group)
    if not _tool_discovery_available(state):
        default_tools.discard("runtime.discover_tools")
    return default_tools


def _tool_discovery_available(state: Mapping[str, Any]) -> bool:
    metadata = as_dict(state.get("metadata"))
    try:
        limit = int(metadata.get("tool_discovery_attempt_limit") or 2)
    except (TypeError, ValueError):
        limit = 2
    limit = min(2, max(0, limit))
    attempts = sum(
        1
        for result in as_list(state.get("tool_results"))
        if isinstance(result, Mapping)
        and str(result.get("name") or result.get("tool_name") or "")
        == "runtime.discover_tools"
    )
    return attempts < limit


def _forced_workflow_tool_name(
    state: dict[str, Any],
    tools: list[dict[str, Any]],
) -> str:
    phase = str(state.get("policy_phase") or "")
    metadata = as_dict(state.get("metadata"))
    resumed_after_clarification = (
        phase == "waiting_user_input"
        and int(metadata.get("workflow_clarification_resume_count") or 0) > 0
    )
    skill = SkillPolicy.from_snapshot(state.get("skill_activation"))
    decision = evaluate_workflow_completion(skill, state)
    completion_policy = skill.completion_policy
    contract_driven = (
        str(completion_policy.get("clarification_strategy") or "") == "tool_contract"
        or completion_policy.get("allow_model_clarification") is False
    )
    initial_contract_transition = (
        skill.active
        and skill.durable
        and phase in {"", "pending"}
        and contract_driven
        and bool(decision.missing_tools)
    )
    if (
        phase != "repair"
        and not resumed_after_clarification
        and not initial_contract_transition
    ):
        return ""
    missing_tools = decision.missing_tools
    available = {
        str(as_dict(item.get("function")).get("name") or "")
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
    skill = as_dict(state.get("skill_activation"))
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
    properties = as_dict(schema.get("properties"))
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
        items = as_dict(tasks.get("items"))
        task_properties = as_dict(items.get("properties"))
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
    implicit_identities: set[str] = set()
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
        if identity in implicit_identities:
            continue
        implicit_identities.add(identity)
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


def _upsert_artifact(current: dict[str, Any], artifact: dict[str, Any]) -> None:
    """Merge repeated observations of one durable artifact by identity."""

    artifact_id = str(artifact.get("id") or "")
    artifacts = current.setdefault("artifacts", [])
    existing = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict) and str(item.get("id") or "") == artifact_id
        ),
        None,
    )
    if existing is None:
        artifacts.append(artifact)
        return
    if (
        str(existing.get("run_id") or "") != str(artifact.get("run_id") or "")
        or str(existing.get("artifact_type") or "")
        != str(artifact.get("artifact_type") or "")
    ):
        raise ValueError(f"artifact identity conflict for {artifact_id}")
    existing.update(
        {
            **artifact,
            "created_at": existing.get("created_at") or artifact.get("created_at"),
            "data": {
                **as_dict(existing.get("data")),
                **as_dict(artifact.get("data")),
            },
            "metadata": {
                **as_dict(existing.get("metadata")),
                **as_dict(artifact.get("metadata")),
            },
        }
    )


def _apply_tool_result(
    current: dict[str, Any],
    result: ToolResult,
    config: Mapping[str, Any],
) -> None:
    if result.name == "runtime.discover_tools" and result.status == "succeeded":
        metadata = current.setdefault("metadata", {})
        metadata["tool_discovery_attempts"] = (
            int(metadata.get("tool_discovery_attempts") or 0) + 1
        )
        discovered = result.data.get("discovered_names")
        if isinstance(discovered, list):
            existing = {
                str(name)
                for name in metadata.get("discovered_tool_names", [])
                if str(name).strip()
            }
            existing.update(str(name) for name in discovered if str(name).strip())
            metadata["discovered_tool_names"] = sorted(existing)
    if result.observation is not None:
        current.setdefault("observations", []).append(result.observation.model_dump(mode="json"))
    for artifact in result.artifacts:
        _upsert_artifact(current, artifact.model_dump(mode="json"))
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
    config: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    payload = resume_payload if isinstance(resume_payload, dict) else {"value": resume_payload}
    pending = current.get("pending_action") or current.get("pending_async") or {}
    resumed_async = bool(current.get("pending_async"))
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
    content_parts = [
        MessageContentPart.model_validate(item)
        for item in as_list(data.get("content_parts") if isinstance(data, dict) else [])
        if isinstance(item, dict)
    ]
    observation = Observation(
        id=f"resume-{uuid4()}",
        run_id=current["run_id"],
        tool_call_id=tool_call_id,
        source=tool_name,
        status="failed" if status == "failed" else "succeeded",
        summary=summary,
        data=as_dict(data),
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
                content_parts=content_parts,
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
    elif resumed_async:
        # A definitive external task failure cannot be repaired by rewording a
        # final answer. The model may explicitly issue a retry tool call, but a
        # prose-only response settles as a workflow contract violation.
        skill = SkillPolicy.from_snapshot(current.get("skill_activation"))
        current["repair_count"] = skill.policy_repair_limit
    current["pending_action"] = None
    current["pending_async"] = None
    current["status"] = "reasoning"
    _emit(current, config, "run.resumed", "Run resumed", summary, {"observation": observation.model_dump(mode="json")})
    return current


def _is_policy_confirmation(pending: Any) -> bool:
    if not isinstance(pending, dict):
        return False
    payload = as_dict(pending.get("payload"))
    return pending.get("action_type") == "policy_confirmation" or payload.get("policy_confirmation") is True


def _apply_policy_confirmation_resume(
    current: dict[str, Any],
    resume_payload: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resolution = resume_payload if isinstance(resume_payload, dict) else {"value": resume_payload}
    data = as_dict(resolution.get("data"))
    status = str(resolution.get("status") or "").lower()
    confirmed = data.get("confirmed") is True or resolution.get("confirmed") is True or status in {
        "confirmed",
        "approved",
    }
    pending = as_dict(current.get("pending_action"))
    payload = as_dict(pending.get("payload"))
    deferred = as_dict(payload.get("deferred_tool_call"))
    tool_call_id = str(pending.get("tool_call_id") or deferred.get("id") or "")
    tool_name = str(deferred.get("name") or payload.get("tool_name") or "policy_confirmation")
    current["pending_action"] = None
    current["pending_async"] = None
    if confirmed and deferred:
        current["pending_tool_calls"] = [deferred]
        current["status"] = "executing_tools"
        metadata = current.setdefault("metadata", {})
        grants = as_dict(metadata.get("confirmation_grants"))
        metadata["confirmation_grants"] = {
            **grants,
            tool_call_id: {
                "confirmed": True,
                "run_id": current["run_id"],
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "policy_id": str(as_dict(payload.get("policy_decision")).get("policy_id") or ""),
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
