from __future__ import annotations

from dataclasses import dataclass
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
from deepkeel.references import EvidenceBundle, DefaultReferenceProjector, ReferenceProjector
from deepkeel.runtime_diagnostics import (
    project_observation as _project_observation,
    runtime_diagnostics as _runtime_diagnostics,
    trace_from_events as _trace_from_events,
)
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


@dataclass(frozen=True, slots=True)
class _ProjectionInputs:
    graph_status: str
    runtime_status: str
    stop_reason: str
    state_error: dict[str, Any] | None
    skill: dict[str, Any]
    typed_tool_results: list[dict[str, Any]]
    checkpoint_pending_action: dict[str, Any] | None
    checkpoint_pending_async: dict[str, Any] | None
    pending_action: dict[str, Any] | None
    final_answer: dict[str, Any]
    references: list[RuntimeReference]
    evidence: list[RuntimeReference]
    evidence_bundle: EvidenceBundle
    checkpoint_observations: list[dict[str, Any]]
    projected_observations: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _ProjectionEnvelope:
    checkpoint: dict[str, Any]
    runtime: dict[str, Any]
    context_snapshot: dict[str, Any]
    active_task: RuntimeActiveTask | None


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
    inputs = _prepare_projection_inputs(
        state,
        skill_activation=skill_activation,
        observation_kinds=observation_kinds or {},
        reference_projector=reference_projector,
    )
    envelope = _build_projection_envelope(
        state,
        inputs,
        question=question,
        context_bundle=context_bundle,
        short_context=short_context,
        streamed_events=streamed_events,
        answer_delta_streamed=answer_delta_streamed,
        observation_kinds=observation_kinds or {},
        task_kinds=task_kinds or {},
        max_steps=max_steps,
        previous_diagnostics=previous_diagnostics,
        capability_manifest=capability_manifest,
    )
    return _assemble_runtime_result(
        state,
        inputs,
        envelope,
        question=question,
        user_id=user_id,
        streamed_events=streamed_events,
        answer_delta_streamed=answer_delta_streamed,
    )


def _prepare_projection_inputs(
    state: dict[str, Any],
    *,
    skill_activation: dict[str, Any],
    observation_kinds: dict[str, str],
    reference_projector: ReferenceProjector | None,
) -> _ProjectionInputs:
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
        _project_observation(item, observation_kinds) for item in checkpoint_observations
    ]
    return _ProjectionInputs(
        graph_status=graph_status,
        runtime_status=runtime_status,
        stop_reason=stop_reason,
        state_error=state_error if isinstance(state_error, dict) else None,
        skill=skill,
        typed_tool_results=typed_tool_results,
        checkpoint_pending_action=checkpoint_pending_action,
        checkpoint_pending_async=checkpoint_pending_async,
        pending_action=pending_action,
        final_answer=final_answer,
        references=cast(list[RuntimeReference], references),
        evidence=cast(list[RuntimeReference], evidence),
        evidence_bundle=evidence_bundle,
        checkpoint_observations=checkpoint_observations,
        projected_observations=projected_observations,
    )


def _build_projection_envelope(
    state: dict[str, Any],
    inputs: _ProjectionInputs,
    *,
    question: str,
    context_bundle: dict[str, Any],
    short_context: dict[str, Any],
    streamed_events: list[dict[str, Any]],
    answer_delta_streamed: bool,
    observation_kinds: dict[str, str],
    task_kinds: dict[str, str],
    max_steps: int,
    previous_diagnostics: dict[str, Any] | None,
    capability_manifest: dict[str, Any] | None,
) -> _ProjectionEnvelope:
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
    turn_id = str(state.get("turn_id") or context_bundle.get("turn_id") or "unbound-turn")
    checkpoint: dict[str, Any] = {
        "schema_version": "harness-checkpoint-v2",
        "run_id": run_id,
        "graph_thread_id": graph_thread_id,
        "turn_id": turn_id,
        "messages": [item for item in state.get("messages", []) if isinstance(item, dict)],
        "observations": inputs.checkpoint_observations,
        "artifacts": [item for item in state.get("artifacts", []) if isinstance(item, dict)],
        "pending_action": inputs.checkpoint_pending_action,
        "pending_async": inputs.checkpoint_pending_async,
        "execution_plan": as_optional_dict(state.get("execution_plan")),
        "pending_tool_calls": [
            item for item in state.get("pending_tool_calls", []) if isinstance(item, dict)
        ],
        "budget_state": state.get("budget_state")
        if isinstance(state.get("budget_state"), dict)
        else {},
        "step_count": int(state.get("step_count") or 0),
        "status": inputs.graph_status,
        "metadata": state.get("metadata") if isinstance(state.get("metadata"), dict) else {},
    }
    trace = _trace_from_events(streamed_events, observation_kinds=observation_kinds)
    context_snapshot = build_context_snapshot(question, context_bundle, short_context, inputs.skill)
    runtime = {
        "schema_version": "harness-runtime-v2",
        "core_contract_version": DEEPKEEL_CONTRACT_VERSION,
        "core_version": DEEPKEEL_VERSION,
        "loop_engine": "langgraph_native_tools",
        "mode": "plan_execute" if checkpoint["execution_plan"] else "react_loop",
        "status": inputs.runtime_status,
        "stop_reason": inputs.stop_reason,
        "step_count": int(state.get("step_count") or 0),
        "observations": inputs.projected_observations,
        "artifacts": checkpoint["artifacts"],
        "pending_action": inputs.pending_action,
        "execution_plan": checkpoint["execution_plan"],
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
        skill_activation=inputs.skill,
        max_steps=max_steps,
        previous_diagnostics=previous_diagnostics,
        capability_manifest=capability_manifest,
    )
    active_task = cast(
        RuntimeActiveTask | None,
        _active_task(inputs.typed_tool_results, task_kinds),
    )
    if active_task:
        runtime["active_task"] = active_task
    runtime["ui_state"] = project_run_ui_state(
        inputs.runtime_status,
        pending_action=inputs.pending_action,
        active_task=dict(active_task) if active_task else None,
    )
    return _ProjectionEnvelope(
        checkpoint=checkpoint,
        runtime=runtime,
        context_snapshot=context_snapshot,
        active_task=active_task,
    )


def _assemble_runtime_result(
    state: dict[str, Any],
    inputs: _ProjectionInputs,
    envelope: _ProjectionEnvelope,
    *,
    question: str,
    user_id: str,
    streamed_events: list[dict[str, Any]],
    answer_delta_streamed: bool,
) -> RuntimeResult:
    checkpoint = envelope.checkpoint
    runtime = envelope.runtime
    final_answer = dict(inputs.final_answer)
    if answer_delta_streamed:
        final_answer["answer_delta_streamed"] = True
    error: RuntimeErrorPayload | None = None
    if inputs.state_error is not None:
        state_error = inputs.state_error
        error = {
            "type": str(state_error.get("type") or "RuntimeFailure"),
            "code": str(state_error.get("code") or "RUNTIME_INTERNAL_ERROR"),
            "category": str(state_error.get("category") or "internal"),
            "retryable": bool(state_error.get("retryable")),
            "message": str(
                state_error.get("user_message") or "The run ended safely and can be retried."
            ),
        }
    answer_fields = set(FinalAnswer.model_fields)
    answer_metadata = as_dict(final_answer.get("metadata"))
    answer_metadata.update(
        {key: value for key, value in final_answer.items() if key not in answer_fields}
    )
    answer_values = {
        key: value
        for key, value in final_answer.items()
        if key in answer_fields and key != "metadata"
    }
    answer_values["status"] = (
        "completed"
        if inputs.runtime_status == "completed"
        else "failed"
        if inputs.runtime_status == "failed"
        else "interrupted"
    )
    answer_values.setdefault("stop_reason", inputs.stop_reason)
    typed_pending_action = (
        PendingAction.model_validate(inputs.checkpoint_pending_action)
        if isinstance(inputs.checkpoint_pending_action, dict)
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
        }.get(inputs.graph_status, RunStatus.REASONING),
        messages=[AgentMessage.model_validate(item) for item in checkpoint["messages"]],
        observations=[Observation.model_validate(item) for item in inputs.checkpoint_observations],
        pending_tool_calls=[
            ToolCall.model_validate(item) for item in checkpoint["pending_tool_calls"]
        ],
        pending_action=typed_pending_action,
        pending_async=inputs.checkpoint_pending_async,
        execution_plan=checkpoint["execution_plan"],
        artifacts=[Artifact.model_validate(item) for item in checkpoint["artifacts"]],
        skill_activation=inputs.skill,
        model_policy=as_dict(state.get("model_policy")),
        budget_state=dict(checkpoint["budget_state"]),
        metadata=dict(checkpoint["metadata"]),
        step_count=int(checkpoint["step_count"]),
    )
    typed_artifacts = [Artifact.model_validate(item) for item in checkpoint["artifacts"]]
    output_contract = as_dict(inputs.skill.get("output_contract"))
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
        status=RuntimeResultStatus(inputs.runtime_status),
        stop_reason=inputs.stop_reason,
        schema_version=str(runtime["schema_version"]),
        core_contract_version=str(runtime["core_contract_version"]),
        core_version=str(runtime["core_version"]),
        loop_engine=str(runtime["loop_engine"]),
        mode=str(runtime["mode"]),
        step_count=int(runtime["step_count"]),
        final_answer=FinalAnswer(**answer_values, metadata=answer_metadata),
        run_context=typed_run_context,
        observations=[Observation.model_validate(item) for item in inputs.checkpoint_observations],
        tool_results=[ToolResult.model_validate(item) for item in inputs.typed_tool_results],
        pending_action=typed_pending_action,
        execution_plan=checkpoint["execution_plan"],
        artifacts=typed_artifacts,
        artifact_views=artifact_views,
        events=[RuntimeStreamEvent.model_validate(item) for item in streamed_events],
        checkpoint=checkpoint,
        trace=runtime["trace"],
        diagnostics=dict(runtime.get("diagnostics") or {}),
        context_snapshot=envelope.context_snapshot,
        skill_activation=inputs.skill,
        active_task=envelope.active_task or None,
        ui_state=cast(RuntimeUIState, as_dict(runtime.get("ui_state"))),
        references=inputs.references,
        evidence=inputs.evidence,
        evidence_bundle=inputs.evidence_bundle,
        needs_user_input=inputs.runtime_status in {"waiting_user_action", "waiting_user_input"},
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
                1
                for event in events
                if event.get("event_type") in {"model.completed", "model.failed"}
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
                    "user_cancelled" if failure.category == "canceled" else failure.code.lower()
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
            "agent_entrypoint": as_dict(context_bundle.get("agent_entrypoint")),
            "capability_view": as_dict(context_bundle.get("_capability_view")),
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
        (
            str(item.get("summary") or "")
            for item in reversed(tool_results)
            if str(item.get("summary") or "").strip()
        ),
        "",
    )
    if runtime_status in {"waiting_user_action", "waiting_user_input"}:
        summary = str(
            (pending_action or {}).get("prompt")
            or latest_summary
            or "Complete the required action and the agent will continue."
        )
        return {
            "answer_mode": "tool_handoff",
            "summary": summary,
            "markdown": summary,
            "pending_action": pending_action or {},
        }
    if runtime_status == "task_running":
        summary = (
            latest_summary
            or "The asynchronous task has started; the agent will continue when it finishes."
        )
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
            artifact.get("artifact_type") or artifact_data.get("artifact_type") or ""
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
