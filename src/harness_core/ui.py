from __future__ import annotations

from typing import Any


TERMINAL_RUNTIME_STATUSES = frozenset({"completed", "failed", "canceled"})
ACTIVE_RUNTIME_STATUSES = frozenset(
    {
        "queued",
        "preparing",
        "reasoning",
        "executing_tools",
        "streaming_answer",
        "settling",
        "waiting_user_input",
        "waiting_user_action",
        "task_running",
        "running",
    }
)


def project_run_ui_state(
    status: str,
    *,
    pending_action: dict[str, Any] | None = None,
    active_task: dict[str, Any] | None = None,
    schema_version: str = "harness-run-ui-v1",
) -> dict[str, Any]:
    lifecycle = _canonical_status(status)
    terminal = lifecycle in TERMINAL_RUNTIME_STATUSES
    waiting_input = lifecycle == "waiting_user_input"
    waiting_action = lifecycle == "waiting_user_action"
    waiting_async = lifecycle == "task_running"
    active = lifecycle in ACTIVE_RUNTIME_STATUSES
    if terminal or not active:
        composer_mode, can_send, reason = "ready", True, "run_terminal"
        if not terminal:
            reason = "run_inactive"
    elif waiting_input:
        composer_mode, can_send, reason = "ready", True, "waiting_user_input"
    elif waiting_action:
        composer_mode, can_send, reason = "blocked", False, "waiting_user_action"
    elif waiting_async:
        composer_mode, can_send, reason = "blocked", False, "waiting_async"
    else:
        composer_mode, can_send, reason = "busy", False, f"run_{lifecycle}"
    return {
        "schema_version": schema_version,
        "lifecycle": lifecycle,
        "composer_mode": composer_mode,
        "can_send": can_send,
        "requires_user_action": waiting_action and pending_action is not None,
        "is_resumable": active and (waiting_input or waiting_action or waiting_async),
        "show_progress": active and not (waiting_input or waiting_action),
        "can_cancel": active and not waiting_input and lifecycle != "settling",
        "active_task": None if terminal else active_task,
        "reason": reason,
    }


def _canonical_status(status: str) -> str:
    value = str(status or "running").strip().lower()
    return {
        "waiting_user": "waiting_user_action",
        "waiting_async": "task_running",
        "cancelled": "canceled",
    }.get(value, value)
