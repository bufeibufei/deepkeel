from __future__ import annotations

import json

from deepkeel.extension_sdk import ToolExecutionContext, ToolExecutor, ToolRegistry, ToolSpec
from deepkeel.runtime_sdk import HarnessRuntime, PendingAction, RuntimeRequest, ToolCall, ToolResult


class ScriptedProvider:
    model = "approval-example-model"
    model_role = "reasoning"

    def __init__(self, turns: list[dict | str]) -> None:
        self.turns = list(turns)

    def complete_chat(self, _messages, **_kwargs):
        turn = self.turns.pop(0)
        message = {"role": "assistant", "content": turn} if isinstance(turn, str) else turn
        return {
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            "model": self.model,
        }


def tool_turn() -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "reserve-1",
                "type": "function",
                "function": {
                    "name": "inventory.reserve",
                    "arguments": json.dumps({"item": "bearing"}),
                },
            }
        ],
    }


def run_approval() -> tuple[str, str]:
    spec = ToolSpec(
        name="inventory.reserve",
        parameters_schema={
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
        required_args=["item"],
        read_only=False,
        requires_user_action=True,
    )
    registry = ToolRegistry([spec])
    executor = ToolExecutor(registry)

    def reserve(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            call=call,
            status="requires_user_action",
            summary="Reservation confirmation required.",
            pending_action=PendingAction(
                id="approval-1",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="confirmation",
                title="Confirm reservation",
                prompt="Reserve one bearing?",
                payload={"item": call.arguments["item"]},
            ),
        )

    executor.register(spec.name, reserve)
    runtime = HarnessRuntime(registry, executor)
    base = {
        "question": "Reserve one bearing",
        "user_id": "approval-user",
        "context_bundle": {"agent_session_id": "approval-run"},
    }
    waiting = runtime.run(
        RuntimeRequest(**base),
        provider=ScriptedProvider([tool_turn()]),
    )
    resumed = runtime.run(
        RuntimeRequest(
            **base,
            short_context={
                "resume": True,
                "resume_observation": {
                    "status": "succeeded",
                    "summary": "Reservation approved.",
                    "data": {"confirmed": True},
                },
                "previous_runtime": waiting.model_dump(mode="json"),
            },
        ),
        provider=ScriptedProvider(["Reservation completed."]),
    )
    return waiting.status.value, resumed.status.value


if __name__ == "__main__":
    before, after = run_approval()
    print(f"{before} -> {after}")
