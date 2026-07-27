from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Callable
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from harness_core.contracts import AgentMessage, FinalAnswer, PendingAction, ToolResult, utc_now
from harness_core.skills import SkillPolicy
from harness_core.tool_lifecycle import completes_workflow_transition
from harness_core.tools import ToolExecutionContext
from harness_core.turn_context import TurnExecutionContext
from harness_core.type_narrowing import as_dict
from harness_core.workflow_policy import (
    SKILL_CONTRACT_VIOLATION,
    WorkflowCompletionDecision,
    workflow_repair_prompt,
    workflow_violation_message,
)

EventSink = Callable[[dict[str, Any]], None]
EMPTY_MODEL_RESPONSE = "EMPTY_MODEL_RESPONSE"
MAX_CONSECUTIVE_EMPTY_MODEL_RETRIES = 1
TRUNCATED_MODEL_RESPONSE = "TRUNCATED_MODEL_RESPONSE"
MAX_MODEL_OUTPUT_CONTINUATIONS = 2
TRUNCATED_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "token_limit"}
)

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
    metadata = as_dict(state.get("metadata"))
    if state.get("status") == "reasoning" and metadata.get("empty_model_retry_pending"):
        return "model"
    if state.get("status") == "reasoning" and metadata.get("output_continuation_pending"):
        return "model"
    return "end"


def _continue_or_fail_truncated_model_response(
    current: dict[str, Any],
    config: Mapping[str, Any],
    *,
    content: str,
    finish_reason: str,
    can_continue: bool,
) -> dict[str, Any]:
    metadata = current.setdefault("metadata", {})
    parts = [
        str(part)
        for part in metadata.get("partial_answer_parts", [])
        if str(part)
    ]
    if content:
        parts.append(content)
    metadata["partial_answer_parts"] = parts
    continuation_count = int(metadata.get("output_continuation_count") or 0) + 1
    metadata["output_continuation_count"] = continuation_count

    if can_continue and continuation_count <= MAX_MODEL_OUTPUT_CONTINUATIONS:
        metadata["output_continuation_pending"] = True
        current.setdefault("messages", []).append(
            AgentMessage(
                id=f"model-output-continuation-{uuid4()}",
                role="user",
                content=(
                    "The previous assistant response stopped only because the model output "
                    "token limit was reached. Continue exactly where it stopped. Do not "
                    "repeat earlier content, do not restart the answer, and do not call "
                    "tools. Finish the remaining answer directly."
                ),
                metadata={
                    "kind": "model_output_continuation",
                    "continuation_count": continuation_count,
                    "internal": True,
                },
            ).model_dump(mode="json")
        )
        current["status"] = "reasoning"
        current["pending_tool_calls"] = []
        _emit(
            current,
            config,
            "model.output_truncated.retrying",
            "Continuing truncated model output",
            "The model reached its output limit; continuation started automatically.",
            {
                "error_code": TRUNCATED_MODEL_RESPONSE,
                "finish_reason": finish_reason,
                "continuation_count": continuation_count,
                "continuation_limit": MAX_MODEL_OUTPUT_CONTINUATIONS,
                "partial_chars": len(_merge_answer_parts(parts)),
                "visible": False,
            },
        )
        return current

    metadata.pop("output_continuation_pending", None)
    message = (
        "The model repeatedly reached its output limit before completing the answer. "
        "The run ended safely; retry with a narrower question or a larger output budget."
    )
    metadata["runtime_error"] = {
        "type": "TruncatedModelResponse",
        "code": TRUNCATED_MODEL_RESPONSE,
        "category": "upstream",
        "retryable": True,
        "message": message,
        "user_message": message,
        "partial_answer": _merge_answer_parts(parts),
    }
    _emit(
        current,
        config,
        "model.output_truncated.exhausted",
        "Model output remained incomplete",
        message,
        {
            "error_code": TRUNCATED_MODEL_RESPONSE,
            "finish_reason": finish_reason,
            "continuation_count": continuation_count,
            "continuation_limit": MAX_MODEL_OUTPUT_CONTINUATIONS,
            "partial_chars": len(_merge_answer_parts(parts)),
            "visible": False,
        },
    )
    return _finish_failed(current, message, config)


def _complete_continued_answer(metadata: dict[str, Any], content: str) -> str:
    parts = [
        str(part)
        for part in metadata.pop("partial_answer_parts", [])
        if str(part)
    ]
    metadata.pop("output_continuation_pending", None)
    metadata.pop("output_continuation_count", None)
    if not parts:
        return content
    if content:
        parts.append(content)
    return _merge_answer_parts(parts)


def _merge_answer_parts(parts: list[str]) -> str:
    merged = ""
    for part in parts:
        if not merged:
            merged = part
            continue
        overlap = _suffix_prefix_overlap(merged, part)
        merged += part[overlap:]
    return merged


def _suffix_prefix_overlap(left: str, right: str, *, limit: int = 1200) -> int:
    maximum = min(len(left), len(right), limit)
    for size in range(maximum, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _retry_or_fail_empty_model_response(
    current: dict[str, Any],
    config: Mapping[str, Any],
    *,
    can_retry: bool,
    answer_only: bool = False,
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
                    (
                        "The prior model step attempted an unavailable tool after the "
                        "workflow was complete. Do not call tools. Return the final "
                        "user-facing answer now."
                    )
                    if answer_only
                    else (
                        "The prior model step returned no content or tool calls. "
                        "Complete the step again: Emit a valid tool call when needed; "
                        "otherwise return a usable answer."
                    )
                ),
                metadata={
                    "kind": "empty_model_response_repair",
                    "retry_count": empty_count,
                    "answer_only": answer_only,
                },
            ).model_dump(mode="json")
        )
        current["status"] = "reasoning"
        current["pending_tool_calls"] = []
        _emit(
            current,
            config,
            "model.empty_response.retrying",
            "Retrying empty model response",
            "The model returned no usable content; retrying automatically.",
            {
                "error_code": EMPTY_MODEL_RESPONSE,
                "retry_count": empty_count,
                "retry_limit": MAX_CONSECUTIVE_EMPTY_MODEL_RETRIES,
                "visible": False,
            },
        )
        return current

    metadata.pop("empty_model_retry_pending", None)
    message = "The model repeatedly returned no usable content. The run ended safely; try again."
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
        "Empty model response",
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
    metadata = as_dict(state.get("metadata"))
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
    config: Mapping[str, Any],
) -> dict[str, Any]:
    question = str(prompt or "").strip()
    pending = PendingAction(
        id=f"clarification-{uuid4()}",
        run_id=str(current.get("run_id") or ""),
        action_type="clarification",
        title="Additional information required",
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
        "Waiting for additional information",
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
    config: Mapping[str, Any],
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
    config: Mapping[str, Any],
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
    if not completes_workflow_transition(result):
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
    config: Mapping[str, Any],
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
    _emit(current, config, "agent.failed", "Execution failed", message, {"final_answer": current["final_answer"]})
    return current


def _emit(
    state: dict[str, Any],
    config: Mapping[str, Any],
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


def _config_value(config: Mapping[str, Any], key: str) -> Any:
    configurable = as_dict(config.get("configurable"))
    return configurable.get(key)


def _graph_config(
    thread_id: str,
    tool_context: ToolExecutionContext,
    event_sink: EventSink | None,
    turn_context: TurnExecutionContext | None = None,
) -> RunnableConfig:
    resolved_tool_context = turn_context.tool_context if turn_context is not None else tool_context
    resolved_event_sink = turn_context.event_sink if turn_context is not None else event_sink
    return {
        "configurable": {
            "thread_id": thread_id,
            # Retain legacy keys while nodes migrate to the typed context.
            "tool_context": resolved_tool_context,
            "event_sink": resolved_event_sink,
            "turn_context": turn_context,
        }
    }


def _answer_summary(markdown: str, limit: int = 240) -> str:
    compact = " ".join(str(markdown or "").split())
    return compact if len(compact) <= limit else f"{compact[:limit].rstrip()}…"
