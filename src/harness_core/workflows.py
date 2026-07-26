from __future__ import annotations

from typing import Any

from harness_core.ui import task_lifecycle


TERMINAL_WORKFLOW_STATES = frozenset({"completed", "partial", "failed", "canceled"})

_WORKFLOW_STATE_ALIASES = {
    "cancelled": "canceled",
    "error": "failed",
    "success": "completed",
    "succeeded": "completed",
    "task_running": "running",
    "waiting_async": "running",
    "waiting_action": "waiting_user_action",
    "waiting_input": "waiting_user_input",
}

_BLOCKING_WORKFLOW_STATES = frozenset(
    {
        "loading_history",
        "queued",
        "running",
        "streaming",
        "recovering",
        "stopping",
        "waiting_user_action",
    }
)


_PHASE_PROGRESS = {
    "configuring": 0.0,
    "starting": 0.05,
    "opening": 0.25,
    "moderating": 0.48,
    "rebuttal": 0.62,
    "stopping": 0.76,
    "synthesizing": 0.86,
    "completed": 1.0,
    "failed": 1.0,
    "canceled": 1.0,
}


def workflow_projection(
    *,
    instance_id: str,
    kind: str,
    status: str,
    phase: str,
    stop_requested: bool = False,
    artifact_id: str = "",
    parent_run_id: str = "",
    metadata: dict[str, Any] | None = None,
    progress: float | None = None,
    revision: int = 0,
    event_sequence: int = 0,
    updated_at: str = "",
) -> dict[str, Any]:
    """Build the stable UI/runtime contract shared by long-running capabilities."""
    durable_status = _normalize_workflow_state(status or "running")
    durable_phase = _normalize_workflow_state(phase or durable_status)
    terminal = durable_status in TERMINAL_WORKFLOW_STATES
    if terminal:
        state = durable_status
    elif stop_requested or durable_phase == "stopping":
        state = "stopping"
    elif durable_phase in {"starting", "configuring"}:
        state = "queued"
    elif durable_phase in {"streaming", "recovering", "waiting_user_action", "waiting_user_input"}:
        state = durable_phase
    elif durable_status in {
        "idle",
        "loading_history",
        "ready",
        "queued",
        "running",
        "streaming",
        "recovering",
        "waiting_user_action",
        "waiting_user_input",
    }:
        state = durable_status
    else:
        state = "running"
    projected_progress = _progress_value(progress, durable_phase, terminal)
    lifecycle_source = durable_status
    if not terminal:
        if state in {"queued", "waiting_user_action", "waiting_user_input"}:
            lifecycle_source = state
        elif durable_phase in {"synthesizing", "settling"}:
            lifecycle_source = durable_phase
    return {
        "schema_version": "workflow-instance-v2",
        "instance_id": str(instance_id),
        "kind": str(kind),
        "revision": _non_negative_int(revision),
        "event_sequence": _non_negative_int(event_sequence),
        "updated_at": str(updated_at or ""),
        "state": state,
        "status": durable_status,
        "lifecycle": task_lifecycle(lifecycle_source).value,
        "execution_status": durable_status,
        "phase": durable_phase,
        "terminal": terminal,
        "recoverable": bool(instance_id),
        "input_blocked": state in _BLOCKING_WORKFLOW_STATES,
        "can_stop": state in {"queued", "running", "streaming", "recovering"},
        "progress": projected_progress,
        "artifact_id": str(artifact_id or ""),
        "parent_run_id": str(parent_run_id or ""),
        "metadata": dict(metadata or {}),
    }


def _non_negative_int(value: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _normalize_workflow_state(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return _WORKFLOW_STATE_ALIASES.get(normalized, normalized)


def _progress_value(value: float | None, phase: str, terminal: bool) -> float:
    if value is None:
        return _PHASE_PROGRESS.get(phase, 1.0 if terminal else 0.1)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return _PHASE_PROGRESS.get(phase, 1.0 if terminal else 0.1)
