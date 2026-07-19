"""Event projection tests owned by the standalone runtime package."""

from harness_core.events import event_runtime_status, is_answer_delta, project_runtime_event


def test_runtime_event_projection_keeps_source_identity_and_sse_delta_contract():
    event = project_runtime_event({"event_type": "model.delta", "payload": {"delta": "今年"}, "ephemeral": True})

    assert event["event_type"] == "answer.delta"
    assert event["source_event_type"] == "model.delta"
    assert event["payload"]["delta"] == "今年"
    assert is_answer_delta(event) is True
    assert event_runtime_status(event) == "streaming_answer"


def test_runtime_event_projection_maps_user_and_async_wait_states():
    user_event = project_runtime_event({"event_type": "tool.requires_user_action", "payload": {}})
    async_event = project_runtime_event({"event_type": "tool.waiting_async", "payload": {}})

    assert event_runtime_status(user_event) == "waiting_user"
    assert event_runtime_status(async_event) == "waiting_async"


def test_tool_event_projection_preserves_typed_tool_contract():
    call = {"id": "call-1", "name": "workflow.start", "arguments": {"question": "Should this workflow proceed?"}}
    started = project_runtime_event({"event_type": "tool.started", "payload": {"tool_call": call}})
    waiting = project_runtime_event(
        {
            "event_type": "tool.requires_user_action",
            "payload": {
                "tool_result": {
                    "name": "workflow.start",
                    "status": "requires_user_action",
                    "data": {"question": "Should this workflow proceed?", "handoff_view": "workflow_form"},
                    "call": call,
                },
                "pending_action": {"handoff_view": "workflow_form", "payload": {}},
            },
        }
    )

    assert started["payload"]["tool_call"] == call
    assert waiting["payload"]["tool_result"]["status"] == "requires_user_action"
    assert waiting["payload"]["pending_action"]["handoff_view"] == "workflow_form"
    assert waiting["payload"]["tool_result"]["data"]["question"] == "Should this workflow proceed?"
    assert "tool_args" not in started["payload"]
    assert "tool_status" not in waiting["payload"]
