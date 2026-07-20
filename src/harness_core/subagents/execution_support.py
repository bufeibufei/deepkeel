from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Callable

from harness_core.budget import MODEL_CALLS, BudgetRequest
from harness_core.contracts import AgentMessage, ToolCall
from harness_core.subagents.contracts import DelegationTask, SubAgentSpec
from harness_core.subagents.execution_types import (
    EventSink, SubAgentEmptyResponseError, _DelegationQuota,
)
from harness_core.tools import ToolExecutionContext
from harness_core.type_narrowing import as_dict

def _resolve_role(task_role: str, spec_role: str, providers: dict[str, Any]) -> str:
    for role in (task_role, spec_role, "reasoning", "fast"):
        if role != "auto" and role in providers:
            return role
    raise RuntimeError("subagent has no available model provider")


def _child_run_id(parent_run_id: str, delegation_id: str, task: DelegationTask) -> str:
    identity = f"{parent_run_id}|{delegation_id}|{task.id}|{task.agent_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    parent_prefix = str(parent_run_id or "root")[:72]
    return f"{parent_prefix}:sub:{digest}"


def _valid_resume_state(
    value: dict[str, Any] | None,
    *,
    task: DelegationTask,
    spec: SubAgentSpec,
) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    if state.get("schema_version") != "subagent-execution-v1":
        return {}
    if str(state.get("task_id") or "") != task.id:
        return {}
    if str(state.get("spec_version") or "") != spec.version:
        return {}
    return dict(state)


def _restored_messages(state: dict[str, Any]) -> list[AgentMessage]:
    messages: list[AgentMessage] = []
    for item in state.get("messages", []):
        if not isinstance(item, dict):
            continue
        try:
            messages.append(AgentMessage.model_validate(item))
        except (TypeError, ValueError):
            return []
    return messages


def _execution_checkpoint(
    *,
    task: DelegationTask,
    spec: SubAgentSpec,
    phase: str,
    round_index: int,
    messages: list[AgentMessage],
    pending_calls: list[ToolCall],
    tool_trace: list[dict[str, Any]],
    model_calls: int,
    tool_calls: int,
    empty_response_retries: int = 0,
    raw_text: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "subagent-execution-v1",
        "task_id": task.id,
        "spec_version": spec.version,
        "phase": phase,
        "round_index": round_index,
        "messages": [message.model_dump(mode="json") for message in messages],
        "pending_tool_calls": [call.model_dump(mode="json") for call in pending_calls],
        "tool_trace": list(tool_trace),
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "empty_response_retries": empty_response_retries,
        "raw_text": raw_text,
    }


def _minimum_optional(first: Any, second: Any) -> int | None:
    values: list[int] = []
    for value in (first, second):
        if value in (None, ""):
            continue
        try:
            values.append(max(0, int(value)))
        except (TypeError, ValueError):
            continue
    return min(values) if values else None


def _child_tool_context(
    context: ToolExecutionContext,
    child_run_id: str,
    spec: SubAgentSpec,
) -> tuple[ToolExecutionContext, Any | None]:
    owned_session = context.session_factory() if context.session_factory is not None else None
    child = ToolExecutionContext(
        run_id=child_run_id,
        user_id=context.user_id,
        thread_id=context.thread_id,
        turn_id=context.turn_id,
        session=owned_session or context.session,
        session_factory=context.session_factory,
        context_bundle=dict(context.context_bundle),
        metadata={
            **context.metadata,
            "subagent": {
                "agent_id": spec.id,
                "read_only": True,
                "tool_allowlist": list(spec.tool_allowlist),
            },
        },
        budget_limits=dict(context.budget_limits),
        deadline_monotonic=context.deadline_monotonic,
        run_control=context.run_control,
    )
    return child, owned_session


def _consume_model_budget(
    budget_ledger: Any,
    *,
    root_run_id: str,
    child_run_id: str,
    task: DelegationTask,
    model_call_limit: float | None,
    step_index: int,
    quota: _DelegationQuota | None = None,
) -> None:
    if quota is not None:
        quota.reserve_model_call()
    if budget_ledger is None:
        return
    budget = budget_ledger.consume(
        BudgetRequest(
            run_id=root_run_id,
            metric=MODEL_CALLS,
            amount=1,
            limit=model_call_limit,
            operation_id=f"subagent-model:{child_run_id}:{step_index}",
            metadata={
                "agent_id": task.agent_id,
                "task_id": task.id,
                "subagent_step": step_index,
            },
        )
    )
    if not budget.allowed:
        raise RuntimeError(budget.reason)


def _emit_subagent_tools(
    sink: EventSink | None,
    child_run_id: str,
    task: DelegationTask,
    calls: list[ToolCall],
    *,
    status: str,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    if sink is None:
        return
    names = [call.name for call in calls]
    sink({
        "event_type": f"subagent.tools.{status}",
        "title": "Subagent is gathering evidence" if status == "started" else "Subagent evidence verified",
        "summary": "; ".join(names),
        "payload": {
            "visible": False,
            "child_run_id": child_run_id,
            "task_id": task.id,
            "agent_id": task.agent_id,
            "status": status,
            "tool_names": names,
            "tool_trace": list(trace or []),
        },
    })


def _emit_subagent_model_retry(
    sink: EventSink | None,
    child_run_id: str,
    task: DelegationTask,
    *,
    model_calls: int,
) -> None:
    if sink is None:
        return
    sink({
        "event_type": "subagent.model.retrying",
        "title": "Retrying specialist analysis",
        "summary": "The model returned no usable content; retrying automatically.",
        "payload": {
            "visible": False,
            "child_run_id": child_run_id,
            "task_id": task.id,
            "agent_id": task.agent_id,
            "status": "retrying",
            "reason_code": "empty_model_response",
            "model_calls": model_calls,
        },
    })


def _is_empty_model_response_error(error: Exception) -> bool:
    if isinstance(error, SubAgentEmptyResponseError):
        return True
    message = str(error or "").strip().lower()
    return any(
        marker in message
        for marker in (
            "llm returned an empty response",
            "model returned an empty response",
            "empty model response",
        )
    )


def _repair_prompt(
    original_prompt: str,
    raw: str,
    schema: dict[str, Any],
    error: str,
    tool_trace: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "task": "Repair the prior output to satisfy the JSON Schema. Return only a JSON object.",
            "validation_error": error,
            "output_schema": schema,
            "invalid_output": str(raw or "")[-12000:],
            "tool_trace": tool_trace,
            "original_task": original_prompt,
        },
        ensure_ascii=False,
        default=str,
    )


def _default_system_prompt(spec: SubAgentSpec) -> str:
    return (
        f"You are {spec.label}. Complete only the delegated task; do not converse with the user or expand scope."
        "Do not delegate recursively or perform side effects. Separate input facts, inference, and uncertainty."
        "Return exactly the given JSON Schema without Markdown."
    )


def _task_prompt(task: DelegationTask, spec: SubAgentSpec) -> str:
    return json.dumps(
        {
            "objective": task.objective,
            "input": task.input_data,
            "constraints": task.constraints,
            "capabilities": spec.capabilities,
            "tool_allowlist": spec.tool_allowlist,
            "input_contract": spec.input_contract,
            "output_contract": spec.output_contract,
            "evidence_policy": spec.evidence_policy,
            "execution_policy": {
                "read_only": spec.read_only,
                "allow_delegation": False,
            },
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _invoke_provider(
    provider: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_seconds: int,
    max_tokens: int,
    output_schema: dict[str, Any],
) -> str:
    completion_budget = _subagent_completion_budget(max_tokens)
    complete = getattr(provider, "complete", None)
    if callable(complete):
        kwargs = {
            "request_timeout": timeout_seconds,
            "max_tokens": completion_budget,
            "response_format": _strict_response_format(output_schema),
        }
        supported = _supported_kwargs(complete, kwargs)
        try:
            return str(complete(system_prompt, user_prompt, **supported) or "").strip()
        except Exception as error:
            if not _response_format_not_supported(error):
                raise
            fallback = _supported_kwargs(
                complete,
                {**kwargs, "response_format": {"type": "json_object"}},
            )
            return str(complete(system_prompt, user_prompt, **fallback) or "").strip()
    complete_chat = getattr(provider, "complete_chat", None)
    if callable(complete_chat):
        common_kwargs = {
            "tools": [],
            "tool_choice": "none",
            "request_timeout": timeout_seconds,
            "max_tokens": completion_budget,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = complete_chat(
                messages,
                **_supported_kwargs(
                    complete_chat,
                    {
                        **common_kwargs,
                        "response_format": _strict_response_format(output_schema),
                    },
                ),
            )
        except Exception as error:
            if not _response_format_not_supported(error):
                raise
            response = complete_chat(
                messages,
                **_supported_kwargs(
                    complete_chat,
                    {
                        **common_kwargs,
                        "response_format": {"type": "json_object"},
                    },
                ),
            )
        message = as_dict(response.get("message")) if isinstance(response, dict) else {}
        parsed = message.get("parsed") if isinstance(message, dict) else None
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False)
        return str(message.get("content") or "").strip()
    raise RuntimeError("subagent provider does not expose complete or complete_chat")


def _strict_response_format(output_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subagent_result",
            "strict": True,
            "schema": output_schema,
        },
    }


def _response_format_not_supported(error: Exception) -> bool:
    message = str(error or "").strip().lower()
    if "response_format" not in message and "response format" not in message:
        return False
    return "json_schema" in message and any(
        marker in message
        for marker in ("not supported", "unsupported", "not valid", "invalid")
    )


def _subagent_completion_budget(requested_tokens: int) -> int:
    """Reserve room for providers that count hidden reasoning as completion tokens."""
    return min(8000, max(int(requested_tokens or 0), 4000))


def _supported_kwargs(function: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return candidates
    return {key: value for key, value in candidates.items() if key in signature.parameters}
