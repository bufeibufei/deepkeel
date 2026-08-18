from __future__ import annotations

from typing import Any, Literal

from deepkeel.mcp.contracts import McpCallResult, McpInputRequest, McpRemoteTool, McpTask
from deepkeel.mcp.protocol import structured_content_from_text
from deepkeel.type_narrowing import as_dict, as_list


def remote_tools(result: dict[str, Any]) -> list[McpRemoteTool]:
    tools: list[McpRemoteTool] = []
    for item in as_list(result.get("tools")):
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        tools.append(
            McpRemoteTool(
                name=str(item["name"]),
                description=str(item.get("description") or ""),
                input_schema=as_dict(item.get("inputSchema")),
                output_schema=as_dict(item.get("outputSchema")),
                annotations=as_dict(item.get("annotations")),
                task_support=_task_support(item.get("taskSupport")),
            )
        )
    return tools


def call_result(
    result: dict[str, Any],
    *,
    redact: Any,
    metadata: dict[str, Any],
) -> McpCallResult:
    content = redact(
        [item for item in as_list(result.get("content")) if isinstance(item, dict)]
    )
    structured_value = result.get("structuredContent")
    structured = (
        redact(structured_value)
        if structured_value is not None
        else structured_content_from_text(content)
    )
    raw_result_type = str(result.get("resultType") or "complete")
    if raw_result_type == "input_required":
        result_type: Literal["complete", "input_required", "task"] = "input_required"
    elif raw_result_type == "task":
        result_type = "task"
    else:
        result_type = "complete"
    task_value = result.get("task")
    task = McpTask.model_validate(task_value) if isinstance(task_value, dict) else None
    if task is not None and task.status not in {"completed", "failed", "cancelled"}:
        result_type = "task"
    return McpCallResult(
        content=content,
        structured_content=structured,
        is_error=bool(result.get("isError")),
        result_type=result_type,
        input_requests=input_requests(result.get("inputRequests")),
        request_state=redact(result.get("requestState")),
        task=task,
        metadata=metadata,
    )


def input_requests(value: Any) -> list[McpInputRequest]:
    requests: list[McpInputRequest] = []
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = [(str(key), item) for key, item in value.items()]
    elif isinstance(value, list):
        items = [
            (str(item.get("id") or index), item)
            for index, item in enumerate(value)
            if isinstance(item, dict)
        ]
    else:
        return requests
    for request_id, item in items:
        if not isinstance(item, dict):
            continue
        method = str(item.get("method") or item.get("type") or "elicitation/create")
        params = item.get("params")
        if not isinstance(params, dict):
            params = {key: value for key, value in item.items() if key not in {"id", "method"}}
        requests.append(
            McpInputRequest(id=str(request_id), method=method, params=params)
        )
    return requests


def task_result(result: dict[str, Any]) -> McpTask:
    value = result.get("task") if isinstance(result.get("task"), dict) else result
    return McpTask.model_validate(value)


def _task_support(value: Any) -> Literal["forbidden", "optional", "required"]:
    normalized = str(value or "forbidden").strip().lower()
    if normalized == "optional":
        return "optional"
    if normalized == "required":
        return "required"
    return "forbidden"
