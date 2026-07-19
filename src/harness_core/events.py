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
    projected_payload = _project_tool_payload(source_type, payload)
    return {
        **event,
        "event_type": EVENT_PROJECTION.get(source_type, source_type),
        "source_event_type": source_type,
        "payload": {**projected_payload, "source_event_type": source_type},
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


def _project_tool_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not event_type.startswith("tool."):
        return payload
    raw_call = payload.get("tool_call") if isinstance(payload.get("tool_call"), dict) else {}
    raw_result = payload.get("tool_result") if isinstance(payload.get("tool_result"), dict) else {}
    result_call = raw_result.get("call") if isinstance(raw_result.get("call"), dict) else {}
    call = raw_call or result_call
    pending = payload.get("pending_action") if isinstance(payload.get("pending_action"), dict) else {}
    tool_name = str(call.get("name") or raw_result.get("name") or "")
    tool_args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    typed_status = str(raw_result.get("status") or "")
    status = {
        "succeeded": "ok",
        "failed": "error",
        "requires_user_action": "requires_user_action",
        "waiting_async": "task_running",
    }.get(typed_status, "started" if event_type == "tool.started" else typed_status)
    result_data = raw_result.get("data") if isinstance(raw_result.get("data"), dict) else {}
    pending_payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    if not result_data and pending_payload:
        result_data = pending_payload
    artifact = _first_artifact(raw_result)
    handoff_view = str(
        pending.get("handoff_view")
        or result_data.get("handoff_view")
        or pending_payload.get("handoff_view")
        or ""
    )
    compatible_call = {
        **call,
        "name": tool_name,
        "args": tool_args,
        "arguments": tool_args,
        "requires_user_action": status == "requires_user_action",
        "handoff_view": handoff_view,
    }
    compatible_result = {
        **raw_result,
        "status": status,
        "result": result_data,
        "artifact": artifact,
        "requires_user_action": status == "requires_user_action",
    }
    return {
        **payload,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_status": status,
        "typed_tool_status": typed_status,
        "requires_user_action": status == "requires_user_action",
        "handoff_view": handoff_view,
        "result": result_data,
        "artifact": artifact,
        "tool_call": compatible_call,
        "tool_result": compatible_result,
    }


def _first_artifact(tool_result: dict[str, Any]) -> dict[str, Any]:
    artifacts = tool_result.get("artifacts") if isinstance(tool_result.get("artifacts"), list) else []
    typed = next((item for item in artifacts if isinstance(item, dict)), {})
    data = typed.get("data") if isinstance(typed.get("data"), dict) else {}
    return {
        **data,
        **{key: value for key, value in typed.items() if key != "data" and value not in (None, "", [], {})},
    }
