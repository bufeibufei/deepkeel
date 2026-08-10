from __future__ import annotations

from deepkeel.contracts import ToolCall
from deepkeel.planning.contracts import ExecutionPlan, PlanStep


def select_ready_plan_steps(
    plan: ExecutionPlan,
    *,
    max_parallel_steps: int = 4,
) -> list[PlanStep]:
    """Select one safe write or a bounded batch of independent reads."""

    completed = {step.id for step in plan.steps if step.status in {"completed", "skipped"}}
    ready = [
        step
        for step in plan.steps
        if step.status == "pending" and set(step.depends_on).issubset(completed)
    ]
    executable = [step for step in ready if step.executor_kind != "synthesis"]
    if not executable:
        return ready[:1]
    if all(step.read_only is True and step.parallel_safe is True for step in executable):
        return executable[: max(1, int(max_parallel_steps))]
    return executable[:1]


def plan_step_tool_call(plan: ExecutionPlan, step: PlanStep) -> ToolCall:
    attempt = step.attempt_count + 1
    identity = f"{plan.run_id}:plan:{plan.plan_id}:r{plan.revision}:{step.id}:a{attempt}"
    return ToolCall(
        id=identity,
        name=step.capability_ref,
        arguments=dict(step.arguments),
        idempotency_key=identity,
        resource_key=step.resource_key,
        read_only=bool(step.read_only),
        parallel_safe=bool(step.parallel_safe),
        metadata={
            "plan_id": plan.plan_id,
            "plan_revision": plan.revision,
            "plan_step_id": step.id,
            "plan_step_title": step.title,
            "plan_executor_kind": step.executor_kind,
        },
    )


__all__ = ["plan_step_tool_call", "select_ready_plan_steps"]
