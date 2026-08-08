from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from deepkeel.budget import (
    INPUT_TOKENS,
    MODEL_CALLS,
    MODEL_RETRIES,
    OUTPUT_TOKENS,
    TOOL_CALLS,
    TOOL_CONCURRENCY,
)
from deepkeel.artifact_views import project_artifact_views
from deepkeel.contracts import (
    AgentMessage,
    Artifact,
    FinalAnswer,
    MessageContentPart,
    Observation,
    PendingAction,
    RunContext,
    RunStatus,
    ToolCall,
    ToolResult,
)
from deepkeel.context import build_context_snapshot, build_initial_messages
from deepkeel.failures import RuntimeFailure, classify_runtime_failure
from deepkeel.references import DefaultReferenceProjector, ReferenceProjector
from deepkeel.runtime_api import (
    RuntimeActiveTask,
    RuntimeErrorPayload,
    RuntimeReference,
    RuntimeResult,
    RuntimeResultStatus,
    RuntimeStreamEvent,
    RuntimeUIState,
)
from deepkeel.tool_registry import ToolRegistry
from deepkeel.type_narrowing import as_dict, as_dict_list, as_list, as_optional_dict
from deepkeel.ui import project_run_ui_state
from deepkeel.version import DEEPKEEL_CONTRACT_VERSION, DEEPKEEL_VERSION

def project_harness_result(
    state: dict[str, Any],
    *,
    question: str,
    context_bundle: dict[str, Any],
    short_context: dict[str, Any],
    skill_activation: dict[str, Any],
    streamed_events: list[dict[str, Any]],
    user_id: str = "local-device",
    answer_delta_streamed: bool = False,
    observation_kinds: dict[str, str] | None = None,
    task_kinds: dict[str, str] | None = None,
    max_steps: int = 12,
    previous_diagnostics: dict[str, Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
    reference_projector: ReferenceProjector | None = None,
) -> RuntimeResult:
    graph_status = str(state.get("status") or "failed")
    state_error = (
        (state.get("metadata") or {}).get("runtime_error")
        if isinstance(state.get("metadata"), dict)
        else None
    )
    skill = dict(skill_activation) if isinstance(skill_activation, dict) else {}
    state_skill = as_dict(state.get("skill_activation"))
    skill.update(state_skill)
    checkpoint_pending_action = as_optional_dict(state.get("pending_action"))
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
    checkpoint_pending_async = as_optional_dict(state.get("pending_async"))
    pending_action = _project_pending_action(checkpoint_pending_action, typed_tool_results)
    final_answer = _project_final_answer(
        state,
        runtime_status,
        pending_action,
        typed_tool_results,
    )
    reference_projection = (reference_projector or DefaultReferenceProjector())(
        typed_tool_results,
        final_answer,
    )
    references = reference_projection.references
    final_answer["references"] = references
    evidence = reference_projection.evidence
    evidence_bundle = reference_projection.bundle()
    if evidence and not final_answer.get("evidence"):
        final_answer["evidence"] = evidence
    checkpoint_observations = [
        item for item in state.get("observations", []) if isinstance(item, dict)
    ]
    projected_observations = [
        _project_observation(item, observation_kinds or {}) for item in checkpoint_observations
    ]
    run_id = str(
        state.get("run_id")
        or context_bundle.get("agent_session_id")
        or context_bundle.get("run_id")
        or "unbound-run"
    )
    graph_thread_id = str(
        state.get("thread_id")
        or context_bundle.get("graph_thread_id")
        or context_bundle.get("thread_id")
        or "unbound-thread"
    )
    turn_id = str(
        state.get("turn_id")
        or context_bundle.get("turn_id")
        or "unbound-turn"
    )
    checkpoint: dict[str, Any] = {
        "schema_version": "harness-checkpoint-v2",
        "run_id": run_id,
        "graph_thread_id": graph_thread_id,
        "turn_id": turn_id,
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
        "schema_version": "harness-runtime-v2",
        "core_contract_version": DEEPKEEL_CONTRACT_VERSION,
        "core_version": DEEPKEEL_VERSION,
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
        "identity": {
            "run_id": checkpoint["run_id"],
            "thread_id": str(
                context_bundle.get("thread_id")
                or context_bundle.get("ask_thread_id")
                or short_context.get("ask_thread_id")
                or checkpoint["graph_thread_id"]
            ),
            "graph_thread_id": checkpoint["graph_thread_id"],
            "turn_id": checkpoint["turn_id"],
        },
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
    active_task = cast(
        RuntimeActiveTask | None,
        _active_task(typed_tool_results, task_kinds or {}),
    )
    if active_task:
        runtime["active_task"] = active_task
    runtime["ui_state"] = project_run_ui_state(
        runtime_status,
        pending_action=pending_action,
        active_task=dict(active_task) if active_task else None,
    )
    if answer_delta_streamed:
        final_answer["answer_delta_streamed"] = True
    error: RuntimeErrorPayload | None = None
    if isinstance(state_error, dict):
        error = {
            "type": str(state_error.get("type") or "RuntimeFailure"),
            "code": str(state_error.get("code") or "RUNTIME_INTERNAL_ERROR"),
            "category": str(state_error.get("category") or "internal"),
            "retryable": bool(state_error.get("retryable")),
            "message": str(
                state_error.get("user_message")
                or "The run ended safely and can be retried."
            ),
        }
    answer_fields = set(FinalAnswer.model_fields)
    answer_metadata = (
        as_dict(final_answer.get("metadata"))
    )
    answer_metadata.update(
        {
            key: value
            for key, value in final_answer.items()
            if key not in answer_fields
        }
    )
    answer_values = {
        key: value
        for key, value in final_answer.items()
        if key in answer_fields and key != "metadata"
    }
    answer_values["status"] = (
        "completed"
        if runtime_status == "completed"
        else "failed"
        if runtime_status == "failed"
        else "interrupted"
    )
    answer_values.setdefault("stop_reason", stop_reason)
    typed_pending_action = (
        PendingAction.model_validate(checkpoint_pending_action)
        if isinstance(checkpoint_pending_action, dict)
        else None
    )
    typed_run_context = RunContext(
        run_id=checkpoint["run_id"],
        thread_id=runtime["identity"]["thread_id"],
        turn_id=checkpoint["turn_id"],
        user_id=str(user_id or "local-device"),
        status={
            "completed": RunStatus.COMPLETED,
            "waiting_user": RunStatus.WAITING_USER,
            "waiting_async": RunStatus.WAITING_ASYNC,
            "failed": RunStatus.FAILED,
            "canceled": RunStatus.CANCELED,
        }.get(graph_status, RunStatus.REASONING),
        messages=[
            AgentMessage.model_validate(item) for item in checkpoint["messages"]
        ],
        observations=[
            Observation.model_validate(item) for item in checkpoint_observations
        ],
        pending_tool_calls=[
            ToolCall.model_validate(item) for item in checkpoint["pending_tool_calls"]
        ],
        pending_action=typed_pending_action,
        pending_async=checkpoint_pending_async,
        artifacts=[
            Artifact.model_validate(item) for item in checkpoint["artifacts"]
        ],
        skill_activation=skill,
        model_policy=as_dict(state.get("model_policy")),
        budget_state=dict(checkpoint["budget_state"]),
        metadata=dict(checkpoint["metadata"]),
        step_count=int(checkpoint["step_count"]),
    )
    typed_artifacts = [
        Artifact.model_validate(item) for item in checkpoint["artifacts"]
    ]
    output_contract = as_dict(skill.get("output_contract"))
    artifact_presentation = output_contract.get("artifact_presentation")
    artifact_views = project_artifact_views(
        typed_artifacts,
        artifact_presentation if isinstance(artifact_presentation, dict) else None,
    )
    return RuntimeResult(
        question=question,
        run_id=checkpoint["run_id"],
        thread_id=runtime["identity"]["thread_id"],
        graph_thread_id=checkpoint["graph_thread_id"],
        turn_id=checkpoint["turn_id"],
        status=RuntimeResultStatus(runtime_status),
        stop_reason=stop_reason,
        schema_version=str(runtime["schema_version"]),
        core_contract_version=str(runtime["core_contract_version"]),
        core_version=str(runtime["core_version"]),
        loop_engine=str(runtime["loop_engine"]),
        mode=str(runtime["mode"]),
        step_count=int(runtime["step_count"]),
        final_answer=FinalAnswer(**answer_values, metadata=answer_metadata),
        run_context=typed_run_context,
        observations=[
            Observation.model_validate(item) for item in checkpoint_observations
        ],
        tool_results=[ToolResult.model_validate(item) for item in typed_tool_results],
        pending_action=typed_pending_action,
        artifacts=typed_artifacts,
        artifact_views=artifact_views,
        events=[RuntimeStreamEvent.model_validate(item) for item in streamed_events],
        checkpoint=checkpoint,
        trace=trace,
        diagnostics=dict(runtime.get("diagnostics") or {}),
        context_snapshot=context_snapshot,
        skill_activation=skill,
        active_task=active_task or None,
        ui_state=cast(RuntimeUIState, as_dict(runtime.get("ui_state"))),
        references=cast(list[RuntimeReference], references),
        evidence=cast(list[RuntimeReference], evidence),
        evidence_bundle=evidence_bundle,
        needs_user_input=runtime_status in {"waiting_user_action", "waiting_user_input"},
        answer_delta_streamed=answer_delta_streamed,
        error=error,
    )


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
                **as_dict(state.get("metadata")),
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
    input_parts: list[MessageContentPart] | None = None,
    pending_tool_calls: list[ToolCall] | None = None,
) -> RunContext:
    initial_calls = list(pending_tool_calls or [])
    return RunContext(
        run_id=run_id,
        thread_id=thread_id,
        turn_id=turn_id,
        user_id=user_id,
        status=RunStatus.EXECUTING_TOOLS if initial_calls else RunStatus.PREPARING,
        messages=build_initial_messages(
            question,
            short_context,
            context_bundle,
            input_parts=input_parts,
        ),
        pending_tool_calls=initial_calls,
        skill_activation=skill_activation,
        model_policy=model_policy,
        budget_state=budget_state,
        metadata={
            "context_source": str(context_bundle.get("source") or "runtime"),
            "disabled_tool_names": [
                str(name)
                for name in as_list(context_bundle.get("disabled_tool_names"))
                if str(name).strip()
            ],
            "memory_recall": as_dict(context_bundle.get("memory_recall")),
            "operational_run_id": str(context_bundle.get("operational_run_id") or ""),
            "governance_scope": {
                "tenant_id": str(context_bundle.get("tenant_id") or ""),
                "user_id": user_id,
                "namespace": str(context_bundle.get("namespace") or "default"),
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
    preconditions = as_dict_list(skill.get("preconditions"))
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
        arguments = as_dict(item.get("tool_arguments"))
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
    data = as_dict(projected.get("data"))
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
        summary = str((pending_action or {}).get("prompt") or latest_summary or "Complete the required action and the agent will continue.")
        return {
            "answer_mode": "tool_handoff",
            "summary": summary,
            "markdown": summary,
            "pending_action": pending_action or {},
        }
    if runtime_status == "task_running":
        summary = latest_summary or "The asynchronous task has started; the agent will continue when it finishes."
        return {"answer_mode": "task_running", "summary": summary, "markdown": summary}
    summary = "The run did not produce a usable answer."
    return {"answer_mode": "bubble", "summary": summary, "markdown": summary, "status": "failed"}


def _active_task(
    typed_tool_results: list[dict[str, Any]],
    task_kinds: dict[str, str],
) -> dict[str, Any]:
    for item in reversed(typed_tool_results):
        if str(item.get("status") or "") != "waiting_async":
            continue
        tool_name = str(item.get("name") or "")
        data = as_dict(item.get("data"))
        artifacts = as_dict_list(item.get("artifacts"))
        artifact = next((value for value in artifacts if isinstance(value, dict)), {})
        artifact_data = as_dict(artifact.get("data"))
        artifact_type = str(
            artifact.get("artifact_type")
            or artifact_data.get("artifact_type")
            or ""
        )
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
        payload = as_dict(event.get("payload"))
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
                    "invocation": (
                        payload.get("invocation")
                        if isinstance(payload.get("invocation"), dict)
                        else {}
                    ),
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
                    "invocation": (
                        payload.get("invocation")
                        if isinstance(payload.get("invocation"), dict)
                        else {}
                    ),
                }
            )
        if (action.startswith("tool.call.") and action != "tool.call.started") or action == "run.waiting_async":
            tool_result = as_dict(payload.get("tool_result"))
            metadata = as_dict(tool_result.get("metadata"))
            metrics = as_dict(
                tool_result.get("runtime_metrics") or metadata.get("runtime_metrics")
            )
            governance = as_dict(metadata.get("governance"))
            tool_call = as_dict(payload.get("tool_call"))
            item.update(
                {
                    "latency_ms": int(metrics.get("latency_ms") or payload.get("latency_ms") or 0),
                    "status": str(tool_result.get("status") or ""),
                    "outcome": str(tool_result.get("outcome") or ""),
                    "diagnostics": (
                        tool_result.get("diagnostics")
                        if isinstance(tool_result.get("diagnostics"), dict)
                        else {}
                    ),
                    "tool_calls": [{"tool_name": str(tool_result.get("name") or tool_call.get("name") or "")}],
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
    previous = as_dict(previous_diagnostics)
    previous_counts = as_dict(previous.get("counts"))
    previous_timings = as_dict(previous.get("timings"))
    previous_skill = as_dict(previous.get("skill"))
    previous_recovery = as_dict(previous.get("recovery"))
    previous_governance = as_dict(previous.get("governance"))
    trace = as_dict_list(runtime.get("trace"))
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
    state_skill = as_dict(state.get("skill_activation"))
    completed_tools.update(
        str(name)
        for name in state_skill.get("completed_tools", [])
        if str(name)
    )
    context_selection = as_dict(context_snapshot.get("context_selection"))
    state_model_policy = as_dict(state.get("model_policy"))
    state_budget_policy = as_dict(state_model_policy.get("budget"))
    skill = dict(skill_activation)
    if skill:
        skill["completed_tools"] = sorted(name for name in completed_tools if name)
        skill["completion_outcome"] = str(runtime.get("status") or "")
        skill["policy_violation_count"] = len(skill.get("policy_violations") or [])
        skill["policy_phase"] = str(state.get("policy_phase") or skill.get("policy_phase") or "")
        missing = as_dict(state.get("missing_requirements"))
        skill["missing_requirements"] = {
            "tools": list(missing.get("tools") or []),
            "artifacts": list(missing.get("artifacts") or []),
        }
        skill["repair_count"] = int(
            state.get("repair_count") or skill.get("repair_count") or 0
        )
        output_contract = as_dict(skill.get("output_contract"))
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
            "resume_mode": str(as_dict(runtime.get("pending_action")).get("resume_mode") or ""),
            "pending_action": runtime.get("pending_action") or {},
        },
        "governance": {
            "model_routes": as_list(previous_governance.get("model_routes"))
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
            "tool_policies": as_list(previous_governance.get("tool_policies"))
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
            "confirmations": as_list(previous_governance.get("confirmations"))
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
            "model_calls": as_list(previous_timings.get("model_calls"))
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
            "tool_calls": as_list(previous_timings.get("tool_calls"))
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
