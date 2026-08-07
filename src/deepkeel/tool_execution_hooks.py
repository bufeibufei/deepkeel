from __future__ import annotations

from typing import Any, Mapping

from deepkeel.contracts import PendingAction, ToolCall, ToolResult
from deepkeel.hooks import HookAudit, HookInvocation, HookPoint
from deepkeel.tool_execution import ToolExecutionContext
from deepkeel.type_narrowing import as_dict


def _tool_hook_invocation(
    point: HookPoint,
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    payload: Mapping[str, Any],
) -> HookInvocation:
    skill = as_dict(context.metadata.get("skill_activation"))
    package_ids = tuple(
        str(value)
        for value in context.metadata.get("capability_package_ids", ())
        if str(value).strip()
    )
    return HookInvocation(
        point=point,
        operation_id=(f"{context.run_id}:{context.turn_id}:tool:{call.id}:{point.value}"),
        run_id=context.run_id,
        thread_id=context.thread_id,
        turn_id=context.turn_id,
        package_ids=package_ids,
        skill_id=str(skill.get("skill_id") or ""),
        payload={
            "tool_call_id": call.id,
            "tool_name": call.name,
            **dict(payload),
        },
        metadata={
            "tenant_id": str(context.metadata.get("tenant_id") or ""),
            "user_id": context.user_id,
        },
    )


def _emit_hook_audits(
    context: ToolExecutionContext,
    audits: tuple[HookAudit, ...],
) -> None:
    sink = context.metadata.get("event_sink")
    if not callable(sink):
        return
    for audit in audits:
        sink(
            {
                "event_type": "hook.executed",
                "title": "Lifecycle hook",
                "summary": f"{audit.point.value}: {audit.status}",
                "payload": {
                    "hook_id": audit.hook_id,
                    "hook_point": audit.point.value,
                    "operation_id": audit.operation_id,
                    "status": audit.status,
                    "duration_ms": audit.duration_ms,
                    "replayed": audit.replayed,
                    "required": audit.required,
                    "error": audit.error,
                    "diagnostics": dict(audit.diagnostics),
                },
                "visibility": "debug",
            }
        )


def _hook_confirmation_result(
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    title: str,
    message: str,
) -> ToolResult:
    pending = PendingAction(
        id=f"hook-confirmation:{call.id}",
        run_id=context.run_id,
        tool_call_id=call.id,
        action_type="confirm_tool_invocation",
        title=title or "Confirm action",
        prompt=message or "Please confirm this action before it continues.",
        payload={
            "tool_name": call.name,
            "arguments": dict(call.arguments),
            "source": "lifecycle_hook",
        },
    )
    return ToolResult(
        call=call,
        status="requires_user_action",
        outcome="partial",
        summary=pending.prompt,
        pending_action=pending,
        metadata={"hook_confirmation": True, "executed": False},
    )
