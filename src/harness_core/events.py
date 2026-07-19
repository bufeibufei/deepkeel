from __future__ import annotations

from typing import Any


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
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    return {
        **event,
        "event_type": EVENT_PROJECTION.get(source_type, source_type),
        "source_event_type": source_type,
        "payload": {**payload, "source_event_type": source_type},
    }


def is_answer_delta(event: dict[str, Any]) -> bool:
    return str(event.get("event_type") or "") in {"model.delta", "answer.delta"}


def event_runtime_status(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("event_type") or "")
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
    return None
