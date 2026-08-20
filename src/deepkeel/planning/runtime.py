from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from deepkeel.contracts import AgentMessage, ToolCall, ToolResult, utc_now
from deepkeel.planning.constants import PLAN_TOOL_NAME
from deepkeel.planning.contracts import ExecutionPlan, PlanStep, PlanningPolicy
from deepkeel.planning.scheduler import plan_step_tool_call, select_ready_plan_steps
from deepkeel.planning.validator import (
    ExecutionPlanValidator,
    PlanValidationError,
    merge_plan_revision,
)
from deepkeel.skills import SkillPolicy
from deepkeel.tool_registry import ToolRegistry


PlanEventEmitter = Callable[[str, str, str, dict[str, Any]], None]


def planning_policy_for_state(state: Mapping[str, Any]) -> PlanningPolicy:
    skill = state.get("skill_activation")
    return PlanningPolicy.from_snapshot(skill if isinstance(skill, dict) else {})


def advance_plan_after_tool_results(
    state: dict[str, Any],
    *,
    calls: list[ToolCall],
    results: list[ToolResult],
    registry: ToolRegistry,
    emit: PlanEventEmitter,
) -> None:
    for result in results:
        if result.name == PLAN_TOOL_NAME and result.status == "succeeded":
            _adopt_proposed_plan(state, result, registry=registry, emit=emit)

    plan = _current_plan(state)
    if plan is None:
        return
    calls_by_id = {call.id: call for call in calls}
    changed = False
    for result in results:
        call = result.call or calls_by_id.get(result.tool_call_id)
        if call is None or call.name == PLAN_TOOL_NAME:
            continue
        metadata = call.metadata if isinstance(call.metadata, dict) else {}
        if str(metadata.get("plan_id") or "") != plan.plan_id:
            continue
        if int(metadata.get("plan_revision") or 0) != plan.revision:
            continue
        step_id = str(metadata.get("plan_step_id") or "")
        plan, step_changed = _apply_step_result(plan, step_id, result, emit=emit)
        changed = changed or step_changed
    if changed:
        state["execution_plan"] = plan.model_dump(mode="json")
    _schedule_next_steps(state, registry=registry, emit=emit)


def advance_plan_after_resume(
    state: dict[str, Any],
    *,
    pending: Mapping[str, Any],
    resume_payload: Any,
    registry: ToolRegistry,
    emit: PlanEventEmitter,
) -> None:
    plan = _current_plan(state)
    if plan is None:
        return
    call_id = str(pending.get("tool_call_id") or "")
    step = next((item for item in plan.steps if item.tool_call_id == call_id), None)
    if step is None or step.status != "waiting":
        return
    payload = resume_payload if isinstance(resume_payload, dict) else {}
    failed = str(payload.get("status") or "succeeded").lower() == "failed"
    if failed and step.attempt_count < step.max_attempts and bool(payload.get("retryable")):
        replacement = step.model_copy(
            update={
                "status": "pending",
                "tool_call_id": "",
                "error": str(payload.get("error") or payload.get("summary") or "resume failed"),
            }
        )
        event_type = "plan.step.retrying"
    elif failed:
        replacement = step.model_copy(
            update={
                "status": "failed",
                "error": str(payload.get("error") or payload.get("summary") or "resume failed"),
            }
        )
        event_type = "plan.step.failed"
    else:
        replacement = step.model_copy(
            update={
                "status": "completed",
                "result_summary": str(payload.get("summary") or "External action completed."),
                "error": "",
            }
        )
        event_type = "plan.step.completed"
    plan = _replace_step(plan, replacement)
    state["execution_plan"] = plan.model_dump(mode="json")
    emit(
        event_type,
        replacement.title,
        replacement.result_summary or replacement.error,
        _step_event_payload(plan, replacement),
    )
    _schedule_next_steps(state, registry=registry, emit=emit)


def complete_plan_for_answer(
    state: dict[str, Any],
    *,
    emit: PlanEventEmitter,
) -> None:
    plan = _current_plan(state)
    if plan is None or plan.status in {"completed", "partially_completed", "canceled"}:
        return
    steps = [
        step.model_copy(update={"status": "completed"})
        if step.executor_kind == "synthesis" and step.status == "running"
        else step
        for step in plan.steps
    ]
    partial = any(
        step.status not in {"completed", "skipped"}
        for step in steps
    )
    status = "partially_completed" if partial else "completed"
    plan = plan.model_copy(update={"status": status, "steps": steps, "updated_at": utc_now()})
    state["execution_plan"] = plan.model_dump(mode="json")
    emit(
        "plan.partially_completed" if partial else "plan.completed",
        "Execution plan completed" if not partial else "Execution plan completed with limits",
        f"{plan.completed_step_count}/{len(plan.steps)} steps completed.",
        _plan_event_payload(plan),
    )


def _adopt_proposed_plan(
    state: dict[str, Any],
    result: ToolResult,
    *,
    registry: ToolRegistry,
    emit: PlanEventEmitter,
) -> None:
    raw = result.data.get("execution_plan")
    if not isinstance(raw, dict):
        return
    policy = planning_policy_for_state(state)
    skill = SkillPolicy.from_snapshot(
        state.get("skill_activation") if isinstance(state.get("skill_activation"), dict) else {}
    )
    allowed_names = (
        set(skill.allowed_tools) if skill.active and skill.tool_scope_mode == "allowlist" else None
    )
    proposed = ExecutionPlan.model_validate(raw).model_copy(
        update={"run_id": str(state.get("run_id") or raw.get("run_id") or "")}
    )
    previous = _current_plan(state)
    if previous is not None:
        proposed = merge_plan_revision(
            previous,
            proposed,
            reason=proposed.revision_reason,
        )
    validator = ExecutionPlanValidator(registry)
    try:
        plan = validator.validate(
            proposed,
            policy=policy,
            allowed_names=allowed_names,
            previous=previous,
        )
    except PlanValidationError as exc:
        state.setdefault("messages", []).append(
            AgentMessage(
                id=f"plan-validation-{uuid4()}",
                role="system",
                content=f"The proposed execution plan could not be accepted: {exc}",
                metadata={"kind": "execution_plan_validation_failure"},
            ).model_dump(mode="json")
        )
        emit(
            "plan.validation_failed",
            "Execution plan rejected",
            str(exc),
            {"visible": False, "error": str(exc)},
        )
        return
    plan = plan.model_copy(update={"status": "running", "updated_at": utc_now()})
    state["execution_plan"] = plan.model_dump(mode="json")
    emit(
        "plan.validated",
        "Execution plan validated",
        f"{len(plan.steps)} bounded steps.",
        _plan_event_payload(plan),
    )
    emit(
        "plan.revised" if previous is not None else "plan.started",
        "Execution plan revised" if previous is not None else "Execution plan started",
        plan.revision_reason or plan.objective,
        _plan_event_payload(plan),
    )


def _apply_step_result(
    plan: ExecutionPlan,
    step_id: str,
    result: ToolResult,
    *,
    emit: PlanEventEmitter,
) -> tuple[ExecutionPlan, bool]:
    step = next((item for item in plan.steps if item.id == step_id), None)
    if step is None or step.status not in {"running", "waiting"}:
        return plan, False
    if result.status == "succeeded":
        status = "completed"
        event_type = "plan.step.completed"
    elif result.status in {"requires_user_action", "waiting_async"}:
        status = "waiting"
        event_type = "plan.step.waiting"
    elif result.retryable and step.attempt_count < step.max_attempts:
        status = "pending"
        event_type = "plan.step.retrying"
    else:
        status = "failed"
        event_type = "plan.step.failed"
    replacement = step.model_copy(
        update={
            "status": status,
            "result_summary": result.summary,
            "error": result.error,
            "artifact_ids": [artifact.id for artifact in result.artifacts],
            "tool_call_id": "" if status == "pending" else result.tool_call_id,
        }
    )
    plan_status = "waiting" if status == "waiting" else plan.status
    if status == "failed":
        plan_status = "replanning"
    updated = _replace_step(plan, replacement).model_copy(
        update={"status": plan_status, "updated_at": utc_now()}
    )
    emit(
        event_type,
        replacement.title,
        replacement.result_summary or replacement.error,
        _step_event_payload(updated, replacement),
    )
    return updated, True


def _schedule_next_steps(
    state: dict[str, Any],
    *,
    registry: ToolRegistry,
    emit: PlanEventEmitter,
) -> None:
    if state.get("pending_action") or state.get("pending_async"):
        return
    if state.get("pending_tool_calls"):
        return
    plan = _current_plan(state)
    if plan is None or plan.status in {
        "completed",
        "partially_completed",
        "failed",
        "canceled",
    }:
        return
    if any(step.status == "failed" for step in plan.steps):
        if plan.status != "replanning":
            plan = plan.model_copy(update={"status": "replanning", "updated_at": utc_now()})
            state["execution_plan"] = plan.model_dump(mode="json")
        state.setdefault("messages", []).append(
            AgentMessage(
                id=f"plan-replan-{uuid4()}",
                role="system",
                content=(
                    "An execution-plan step failed. Revise the remaining plan with "
                    "runtime.create_plan when recovery is useful, otherwise provide an honest "
                    "bounded answer from completed observations."
                ),
                metadata={"kind": "execution_plan_replanning"},
            ).model_dump(mode="json")
        )
        emit(
            "plan.revision_requested",
            "Execution plan needs revision",
            "A terminal step failure requires bounded replanning or a limited answer.",
            _plan_event_payload(plan),
        )
        state["status"] = "reasoning"
        return
    policy = planning_policy_for_state(state)
    ready = select_ready_plan_steps(
        plan,
        max_parallel_steps=policy.max_parallel_steps,
    )
    if ready and ready[0].executor_kind == "synthesis":
        synthesis = ready[0].model_copy(
            update={
                "status": "running",
                "attempt_count": ready[0].attempt_count + 1,
            }
        )
        plan = _replace_step(plan, synthesis).model_copy(
            update={"status": "synthesizing", "updated_at": utc_now()}
        )
        state["execution_plan"] = plan.model_dump(mode="json")
        _append_synthesis_instruction(state, plan)
        emit(
            "plan.synthesis.started",
            synthesis.title,
            synthesis.objective,
            _step_event_payload(plan, synthesis),
        )
        state["status"] = "reasoning"
        return
    if ready:
        calls: list[ToolCall] = []
        replacements: dict[str, PlanStep] = {}
        for step in ready:
            call = plan_step_tool_call(plan, step)
            replacements[step.id] = step.model_copy(
                update={
                    "status": "running",
                    "attempt_count": step.attempt_count + 1,
                    "tool_call_id": call.id,
                }
            )
            calls.append(call)
        plan = plan.model_copy(
            update={
                "status": "running",
                "steps": [replacements.get(step.id, step) for step in plan.steps],
                "updated_at": utc_now(),
            }
        )
        state["execution_plan"] = plan.model_dump(mode="json")
        state["pending_tool_calls"] = [call.model_dump(mode="json") for call in calls]
        state["status"] = "executing_tools"
        for step in ready:
            emit(
                "plan.step.started",
                step.title,
                step.objective,
                _step_event_payload(plan, replacements[step.id]),
            )
        return
    if all(step.status in {"completed", "skipped"} for step in plan.steps):
        plan = plan.model_copy(update={"status": "synthesizing", "updated_at": utc_now()})
        state["execution_plan"] = plan.model_dump(mode="json")
        _append_synthesis_instruction(state, plan)
        emit(
            "plan.synthesis.started",
            "Synthesizing plan results",
            f"{plan.completed_step_count} execution steps completed.",
            _plan_event_payload(plan),
        )
        state["status"] = "reasoning"
        return
    blocked = [step.id for step in plan.steps if step.status == "pending"]
    if blocked:
        plan = plan.model_copy(update={"status": "replanning", "updated_at": utc_now()})
        state["execution_plan"] = plan.model_dump(mode="json")
        emit(
            "plan.blocked",
            "Execution plan is blocked",
            f"No runnable step among: {', '.join(blocked)}.",
            _plan_event_payload(plan),
        )


def _append_synthesis_instruction(state: dict[str, Any], plan: ExecutionPlan) -> None:
    state.setdefault("messages", []).append(
        AgentMessage(
            id=f"plan-synthesis-{uuid4()}",
            role="system",
            content=(
                "The execution plan has collected its available observations. Synthesize the "
                "final user-facing answer now, reconciling all evidence and clearly stating any "
                "remaining uncertainty. Do not expose the raw plan or hidden reasoning."
            ),
            metadata={"kind": "execution_plan_synthesis", "plan_id": plan.plan_id},
        ).model_dump(mode="json")
    )


def _current_plan(state: Mapping[str, Any]) -> ExecutionPlan | None:
    raw = state.get("execution_plan")
    if not isinstance(raw, dict):
        return None
    try:
        return ExecutionPlan.model_validate(raw)
    except ValueError:
        return None


def _replace_step(plan: ExecutionPlan, replacement: PlanStep) -> ExecutionPlan:
    return plan.model_copy(
        update={
            "steps": [replacement if step.id == replacement.id else step for step in plan.steps]
        }
    )


def _plan_event_payload(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "revision": plan.revision,
        "status": plan.status,
        "objective": plan.objective,
        "progress": plan.progress,
        "visible": True,
    }


def _step_event_payload(plan: ExecutionPlan, step: PlanStep) -> dict[str, Any]:
    return {
        **_plan_event_payload(plan),
        "step_id": step.id,
        "step_title": step.title,
        "step_status": step.status,
        "executor_kind": step.executor_kind,
        "capability_ref": step.capability_ref,
        "attempt_count": step.attempt_count,
        "max_attempts": step.max_attempts,
    }


__all__ = [
    "advance_plan_after_resume",
    "advance_plan_after_tool_results",
    "complete_plan_for_answer",
    "planning_policy_for_state",
]
