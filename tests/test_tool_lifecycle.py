from harness_core.contracts import ToolResult
from harness_core.tool_lifecycle import completes_workflow_transition


def _result(status: str) -> ToolResult:
    return ToolResult(
        tool_call_id="call-1",
        name="demo.tool",
        status=status,
    )


def test_terminal_and_deferred_tool_states_complete_the_current_transition() -> None:
    assert completes_workflow_transition(_result("succeeded")) is True
    assert completes_workflow_transition(_result("requires_user_action")) is True
    assert completes_workflow_transition(_result("waiting_async")) is True


def test_failed_tool_does_not_complete_the_current_transition() -> None:
    assert completes_workflow_transition(_result("failed")) is False
