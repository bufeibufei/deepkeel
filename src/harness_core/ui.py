from __future__ import annotations

from typing import Any

from harness_core.contracts import TaskLifecycle

TERMINAL_RUNTIME_STATUSES = frozenset(
    {
        TaskLifecycle.COMPLETED.value,
        TaskLifecycle.FAILED.value,
        TaskLifecycle.CANCELED.value,
    }
)
ACTIVE_RUNTIME_STATUSES = frozenset(
    lifecycle.value
    for lifecycle in TaskLifecycle
    if lifecycle.value not in TERMINAL_RUNTIME_STATUSES
)


def project_run_ui_state(
    status: str,
    *,
    pending_action: dict[str, Any] | None = None,
    active_task: dict[str, Any] | None = None,
    schema_version: str = "harness-run-ui-v2",
) -> dict[str, Any]:
    execution_status = _canonical_execution_status(status)
    lifecycle = task_lifecycle(execution_status)
    lifecycle_value = lifecycle.value
    terminal = lifecycle_value in TERMINAL_RUNTIME_STATUSES
    collecting_input = lifecycle is TaskLifecycle.COLLECTING_INPUT
    waiting_action = lifecycle is TaskLifecycle.WAITING_USER_ACTION
    active = lifecycle_value in ACTIVE_RUNTIME_STATUSES
    if terminal or not active:
        composer_mode, can_send, reason = "ready", True, "run_terminal"
        if not terminal:
            reason = "run_inactive"
    elif collecting_input:
        composer_mode, can_send, reason = "ready", True, "collecting_input"
    elif waiting_action:
        composer_mode, can_send, reason = "blocked", False, "waiting_user_action"
    else:
        composer_mode, can_send, reason = "busy", False, f"run_{lifecycle_value}"
    return {
        "schema_version": schema_version,
        "lifecycle": lifecycle_value,
        "execution_status": execution_status,
        "composer_mode": composer_mode,
        "can_send": can_send,
        "input_strategy": "follow_up" if can_send else "hard_interrupt",
        "requires_user_action": waiting_action and pending_action is not None,
        "is_resumable": active and waiting_action,
        "show_progress": lifecycle in {
            TaskLifecycle.QUEUED,
            TaskLifecycle.RUNNING,
            TaskLifecycle.SYNTHESIZING,
        },
        "can_cancel": lifecycle in {
            TaskLifecycle.QUEUED,
            TaskLifecycle.RUNNING,
            TaskLifecycle.SYNTHESIZING,
        },
        "active_task": None if terminal else active_task,
        "reason": reason,
    }


def task_lifecycle(status: str) -> TaskLifecycle:
    value = _canonical_execution_status(status)
    if value in {"waiting_user_input", "waiting_input", "collecting_input"}:
        return TaskLifecycle.COLLECTING_INPUT
    if value in {
        "waiting_user",
        "waiting_user_action",
        "waiting_action",
        "requires_user_action",
    }:
        return TaskLifecycle.WAITING_USER_ACTION
    if value in {"queued", "preparing", "starting", "configuring"}:
        return TaskLifecycle.QUEUED
    if value in {"streaming_answer", "synthesizing", "settling"}:
        return TaskLifecycle.SYNTHESIZING
    if value in {"completed", "success", "succeeded"}:
        return TaskLifecycle.COMPLETED
    if value in {"failed", "error"}:
        return TaskLifecycle.FAILED
    if value in {"canceled", "cancelled"}:
        return TaskLifecycle.CANCELED
    return TaskLifecycle.RUNNING


def _canonical_execution_status(status: str) -> str:
    value = str(status or "running").strip().lower()
    return "canceled" if value == "cancelled" else value
