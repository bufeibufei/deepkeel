from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from deepkeel.hooks import (
    HookAction,
    HookAudit,
    HookInvocation,
    HookPoint,
    HookRunner,
)


EventSink = Callable[[dict[str, Any]], None]
AuditSink = Callable[[EventSink, tuple[HookAudit, ...]], None]


@dataclass(frozen=True, slots=True)
class LifecycleStartOutcome:
    audits: tuple[HookAudit, ...]
    context_patch: dict[str, Any]
    stopped_reason: str = ""


async def run_start_lifecycle_hooks(
    *,
    hook_runner: HookRunner,
    emit: EventSink,
    emit_audits: AuditSink,
    run_id: str,
    thread_id: str,
    turn_id: str,
    package_ids: tuple[str, ...],
    skill_id: str,
    question: str,
    context_window: dict[str, Any],
    user_id: str,
    resumed: bool,
) -> LifecycleStartOutcome:
    audits: list[HookAudit] = []
    context_patch: dict[str, Any] = {}
    points = (
        HookPoint.RUN_RESUMED if resumed else HookPoint.RUN_STARTED,
        HookPoint.TURN_STARTED,
        HookPoint.CONTEXT_PREPARED,
    )
    for point in points:
        hook_result = await hook_runner.arun(
            HookInvocation(
                point=point,
                operation_id=(
                    f"{run_id}:{point.value}"
                    if point is HookPoint.RUN_STARTED
                    else f"{run_id}:{turn_id}:{point.value}"
                ),
                run_id=run_id,
                thread_id=thread_id,
                turn_id=turn_id,
                package_ids=package_ids,
                skill_id=skill_id,
                payload={
                    "question": question,
                    "context_window": dict(context_window),
                },
                metadata={"user_id": user_id},
            )
        )
        audits.extend(hook_result.audits)
        emit_audits(emit, hook_result.audits)
        context_patch.update(dict(hook_result.decision.context_patch))
        if hook_result.decision.action != HookAction.CONTINUE:
            return LifecycleStartOutcome(
                audits=tuple(audits),
                context_patch=context_patch,
                stopped_reason=(
                    hook_result.decision.reason
                    or f"runtime lifecycle hook stopped at {point.value}"
                ),
            )
    return LifecycleStartOutcome(
        audits=tuple(audits),
        context_patch=context_patch,
    )


async def run_settlement_lifecycle_hooks(
    *,
    hook_runner: HookRunner,
    emit: EventSink,
    emit_audits: AuditSink,
    run_id: str,
    thread_id: str,
    turn_id: str,
    package_ids: tuple[str, ...],
    skill_id: str,
    user_id: str,
    status: str,
    stop_reason: str,
    artifact_ids: list[str],
) -> tuple[HookAudit, ...]:
    point = (
        HookPoint.RUN_SUSPENDING
        if status in {"waiting_user_action", "waiting_user_input", "task_running"}
        else HookPoint.RUN_SETTLED
    )
    hook_result = await hook_runner.arun(
        HookInvocation(
            point=point,
            operation_id=f"{run_id}:{turn_id}:{point.value}",
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            package_ids=package_ids,
            skill_id=skill_id,
            payload={
                "status": status,
                "stop_reason": stop_reason,
                "artifact_ids": artifact_ids,
            },
            metadata={"user_id": user_id},
        )
    )
    emit_audits(emit, hook_result.audits)
    return tuple(hook_result.audits)
