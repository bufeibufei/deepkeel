from __future__ import annotations

from deepkeel.contracts import ToolCall, ToolResult
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor


def test_tool_executor_applies_authoritative_context_argument_bindings() -> None:
    spec = ToolSpec(
        name="profile.read",
        parameters_schema={
            "type": "object",
            "required": ["profile_id"],
            "properties": {"profile_id": {"type": "string"}},
            "additionalProperties": False,
        },
        runtime_policy={
            "context_argument_bindings": {
                "profile_id": ["profile_id", "active_profile.id"],
            }
        },
    )
    registry = ToolRegistry([spec])
    executor = ToolExecutor(registry)
    observed: list[dict] = []

    def handler(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        observed.append(dict(call.arguments))
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            status="succeeded",
            summary="read",
            call=call,
        )

    executor.register(spec.name, handler)
    result = executor.execute(
        ToolCall(
            id="call-1",
            name=spec.name,
            arguments={"profile_id": "model-invented-profile"},
        ),
        ToolExecutionContext(
            run_id="run-1",
            user_id="user-1",
            context_bundle={"active_profile": {"id": "host-profile"}},
        ),
    )

    assert result.status == "succeeded"
    assert observed == [{"profile_id": "host-profile"}]


def test_tool_executor_leaves_unbound_arguments_unchanged() -> None:
    spec = ToolSpec(name="search", parameters_schema={"type": "object"})
    registry = ToolRegistry([spec])
    executor = ToolExecutor(registry)
    observed: list[dict] = []

    def handler(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        observed.append(dict(call.arguments))
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            status="succeeded",
            summary="searched",
            call=call,
        )

    executor.register(spec.name, handler)
    executor.execute(
        ToolCall(id="call-2", name=spec.name, arguments={"query": "original"}),
        ToolExecutionContext(run_id="run-1", user_id="user-1"),
    )

    assert observed == [{"query": "original"}]
