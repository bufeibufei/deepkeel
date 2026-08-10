from __future__ import annotations

from collections import deque

from deepkeel.contracts import utc_now
from deepkeel.planning.contracts import ExecutionPlan, PlanStep, PlanningPolicy
from deepkeel.tool_registry import ToolRegistry


class PlanValidationError(ValueError):
    """A model-proposed plan is unsafe or cannot execute in this runtime."""


class ExecutionPlanValidator:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def validate(
        self,
        plan: ExecutionPlan,
        *,
        policy: PlanningPolicy,
        allowed_names: set[str] | None = None,
        previous: ExecutionPlan | None = None,
    ) -> ExecutionPlan:
        if not policy.enabled:
            raise PlanValidationError("execution planning is disabled by policy")
        if len(plan.steps) > policy.max_steps:
            raise PlanValidationError(
                f"execution plan exceeds the {policy.max_steps}-step policy limit"
            )
        if previous is not None:
            self._validate_revision(previous, plan, policy)
        self._validate_dependencies(plan)
        return plan.model_copy(
            update={
                "steps": [
                    self._validated_step(
                        step,
                        policy=policy,
                        allowed_names=allowed_names,
                    )
                    for step in plan.steps
                ]
            }
        )

    def _validated_step(
        self,
        step: PlanStep,
        *,
        policy: PlanningPolicy,
        allowed_names: set[str] | None,
    ) -> PlanStep:
        if step.executor_kind == "synthesis":
            return step.model_copy(
                update={
                    "read_only": True,
                    "parallel_safe": False,
                    "resource_key": "runtime.synthesis",
                    "max_attempts": min(step.max_attempts, policy.max_attempts_per_step),
                }
            )
        name = step.capability_ref
        if name.startswith("runtime."):
            raise PlanValidationError(
                f"plan step {step.id!r} cannot execute runtime control tool {name!r}"
            )
        try:
            spec = self.registry.get(name)
        except KeyError as exc:
            raise PlanValidationError(
                f"plan step {step.id!r} references unknown capability {name!r}"
            ) from exc
        if allowed_names is not None and name not in allowed_names:
            raise PlanValidationError(
                f"plan step {step.id!r} capability {name!r} is not allowed by the active Skill"
            )
        return step.model_copy(
            update={
                "read_only": spec.read_only,
                "parallel_safe": bool(
                    spec.parallel_safe
                    and not spec.requires_user_action
                    and not spec.async_tool
                ),
                "resource_key": step.resource_key
                or str(spec.runtime_policy.get("side_effect") or spec.name),
                "max_attempts": min(step.max_attempts, policy.max_attempts_per_step),
            }
        )

    @staticmethod
    def _validate_dependencies(plan: ExecutionPlan) -> None:
        step_ids = {step.id for step in plan.steps}
        unknown = sorted(
            {
                dependency
                for step in plan.steps
                for dependency in step.depends_on
                if dependency not in step_ids
            }
        )
        if unknown:
            raise PlanValidationError(
                f"execution plan references unknown dependencies: {', '.join(unknown)}"
            )
        indegree = {step.id: len(step.depends_on) for step in plan.steps}
        children: dict[str, list[str]] = {step.id: [] for step in plan.steps}
        for step in plan.steps:
            for dependency in step.depends_on:
                children[dependency].append(step.id)
        ready = deque(step_id for step_id, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            step_id = ready.popleft()
            visited += 1
            for child in children[step_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(plan.steps):
            raise PlanValidationError("execution plan dependency cycle detected")

    @staticmethod
    def _validate_revision(
        previous: ExecutionPlan,
        proposed: ExecutionPlan,
        policy: PlanningPolicy,
    ) -> None:
        if proposed.plan_id != previous.plan_id:
            raise PlanValidationError("plan revision must retain plan_id")
        if proposed.revision != previous.revision + 1:
            raise PlanValidationError("plan revision is not the next expected revision")
        if proposed.revision - 1 > policy.max_revisions:
            raise PlanValidationError("plan revision limit exceeded")
        proposed_steps = {step.id: step for step in proposed.steps}
        for completed in (
            step for step in previous.steps if step.status in {"completed", "skipped"}
        ):
            replacement = proposed_steps.get(completed.id)
            if replacement is None:
                raise PlanValidationError(f"completed plan step {completed.id!r} cannot be removed")
            if replacement.immutable_signature() != completed.immutable_signature():
                raise PlanValidationError(f"completed plan step {completed.id!r} is immutable")


def merge_plan_revision(
    previous: ExecutionPlan,
    proposed: ExecutionPlan,
    *,
    reason: str = "",
) -> ExecutionPlan:
    proposed_by_id = {step.id: step for step in proposed.steps}
    steps: list[PlanStep] = []
    for prior in previous.steps:
        if prior.status not in {"completed", "skipped"}:
            continue
        replacement = proposed_by_id.pop(prior.id, prior)
        steps.append(
            replacement.model_copy(
                update={
                    "status": prior.status,
                    "attempt_count": prior.attempt_count,
                    "tool_call_id": prior.tool_call_id,
                    "result_summary": prior.result_summary,
                    "error": prior.error,
                    "artifact_ids": list(prior.artifact_ids),
                }
            )
        )
    steps.extend(
        step.model_copy(
            update={
                "status": "pending",
                "attempt_count": 0,
                "tool_call_id": "",
                "result_summary": "",
                "error": "",
                "artifact_ids": [],
            }
        )
        for step in proposed.steps
        if step.id in proposed_by_id
    )
    return proposed.model_copy(
        update={
            "plan_id": previous.plan_id,
            "run_id": previous.run_id,
            "revision": previous.revision + 1,
            "status": "proposed",
            "steps": steps,
            "revision_reason": reason or proposed.revision_reason,
            "created_at": previous.created_at,
            "updated_at": utc_now(),
        }
    )


__all__ = [
    "ExecutionPlanValidator",
    "PlanValidationError",
    "merge_plan_revision",
]
