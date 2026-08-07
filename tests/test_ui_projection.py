import pytest

from deepkeel.runtime_sdk import (
    TaskLifecycle,
    project_run_ui_state,
    task_lifecycle,
)


@pytest.mark.parametrize(
    ("execution_status", "lifecycle"),
    [
        ("waiting_user_input", TaskLifecycle.COLLECTING_INPUT),
        ("waiting_user", TaskLifecycle.WAITING_USER_ACTION),
        ("preparing", TaskLifecycle.QUEUED),
        ("reasoning", TaskLifecycle.RUNNING),
        ("executing_tools", TaskLifecycle.RUNNING),
        ("waiting_async", TaskLifecycle.RUNNING),
        ("streaming_answer", TaskLifecycle.SYNTHESIZING),
        ("completed", TaskLifecycle.COMPLETED),
        ("error", TaskLifecycle.FAILED),
        ("cancelled", TaskLifecycle.CANCELED),
    ],
)
def test_task_lifecycle_normalizes_engine_and_legacy_statuses(
    execution_status: str,
    lifecycle: TaskLifecycle,
) -> None:
    assert task_lifecycle(execution_status) is lifecycle


def test_collecting_input_keeps_composer_available() -> None:
    state = project_run_ui_state("waiting_user_input")

    assert state["lifecycle"] == "collecting_input"
    assert state["composer_mode"] == "ready"
    assert state["can_send"] is True
    assert state["requires_user_action"] is False
    assert state["show_progress"] is False


def test_waiting_action_blocks_composer_and_requires_pending_action() -> None:
    without_action = project_run_ui_state("waiting_user")
    with_action = project_run_ui_state(
        "waiting_user_action",
        pending_action={"id": "action-1"},
    )

    assert with_action["lifecycle"] == "waiting_user_action"
    assert with_action["composer_mode"] == "blocked"
    assert with_action["can_send"] is False
    assert with_action["requires_user_action"] is True
    assert without_action["requires_user_action"] is False


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_statuses_unlock_composer_and_clear_active_task(status: str) -> None:
    state = project_run_ui_state(
        status,
        active_task={"kind": "report"},
    )

    assert state["composer_mode"] == "ready"
    assert state["can_send"] is True
    assert state["active_task"] is None
    assert state["show_progress"] is False
    assert state["can_cancel"] is False


def test_synthesizing_remains_busy_until_terminal_settlement() -> None:
    state = project_run_ui_state("streaming_answer")

    assert state["lifecycle"] == "synthesizing"
    assert state["execution_status"] == "streaming_answer"
    assert state["composer_mode"] == "busy"
    assert state["show_progress"] is True
    assert state["can_cancel"] is True
