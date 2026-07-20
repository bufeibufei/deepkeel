from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from typing import Any

from harness_core.type_narrowing import as_dict


class AgentEventPersistenceError(RuntimeError):
    """Raised when a runtime event cannot be durably recorded before publication."""


EVENT_PROJECTION = {
    "model.delta": "answer.delta",
    "tool.started": "tool.call.started",
    "tool.completed": "tool.call.completed",
    "tool.failed": "tool.call.failed",
    "tool.requires_user_action": "tool.call.requires_user_action",
    "tool.waiting_async": "run.waiting_async",
    "answer.completed": "final_answer",
}


def project_runtime_event(event: dict[str, Any]) -> dict[str, Any]:
    source_type = str(event.get("event_type") or "")
    payload = as_dict(event.get("payload"))
    return {
        **event,
        "event_type": EVENT_PROJECTION.get(source_type, source_type),
        "source_event_type": source_type,
        "payload": {**payload, "source_event_type": source_type},
    }


def envelope_runtime_event(
    event: dict[str, Any],
    *,
    run_id: str,
    thread_id: str,
    turn_id: str,
    sequence: int,
    run_version: int = 0,
) -> dict[str, Any]:
    projected = project_runtime_event(event)
    source_type = str(
        projected.get("source_event_type") or projected.get("event_type") or ""
    )
    event_key = f"{run_id}:{turn_id}:{sequence}:{source_type}"
    visibility = str(projected.get("visibility") or "internal")
    return {
        **projected,
        "schema_version": "harness-runtime-event-v1",
        "event_id": hashlib.sha256(event_key.encode("utf-8")).hexdigest(),
        "sequence": max(1, int(sequence)),
        "run_version": max(0, int(run_version)),
        "run_id": run_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "visibility": visibility if visibility in {"public", "internal"} else "internal",
        "created_at": projected.get("created_at") or datetime.now(UTC).isoformat(),
    }


def is_answer_delta(event: dict[str, Any]) -> bool:
    return str(event.get("event_type") or "") in {"model.delta", "answer.delta"}


def event_runtime_status(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("event_type") or "")
    payload = as_dict(event.get("payload"))
    if event_type in {"run.created", "user.message"}:
        return "preparing"
    if event_type in {"agent.reasoning"}:
        return "reasoning"
    if event_type in {"tool.started", "tool.call.started"}:
        return "executing_tools"
    if event_type in {"tool.requires_user_action", "tool.call.requires_user_action"}:
        return "waiting_user"
    if event_type in {"tool.waiting_async", "run.waiting_async"}:
        return "waiting_async"
    if event_type in {"model.delta", "answer.delta"}:
        return "streaming_answer"
    if event_type in {"answer.completed", "run.completed"}:
        return "completed"
    if event_type == "run.failed":
        return "failed"
    if event_type == "run.canceled":
        return "canceled"
    if event_type == "run.settled":
        status = str(payload.get("status") or "").strip().lower()
        return status if status in {"completed", "failed", "canceled"} else None
    return None
