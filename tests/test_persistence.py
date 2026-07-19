"""Portable checkpoint tests owned by the standalone runtime package."""

import pytest
from pydantic import ValidationError

from harness_core.persistence import (
    CheckpointCompatibilityError,
    checkpoint_from_durable_state,
    checkpoint_from_runtime,
    durable_state_from_result,
    restore_run_context,
)
from harness_core.contracts import FinalAnswer
from harness_core.runtime_api import RuntimeResult, RuntimeResultStatus


def test_restore_run_context_appends_one_correlated_tool_observation():
    context = restore_run_context(
        checkpoint={
            "schema_version": "harness-checkpoint-v2",
            "run_id": "run-1",
            "messages": [
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-1", "name": "demo.select", "arguments": {}}],
                }
            ],
            "observations": [],
            "artifacts": [],
            "pending_action": {"tool_call_id": "call-1", "action_type": "demo.select"},
            "step_count": 1,
        },
        resume_payload={"status": "succeeded", "summary": "已选择。", "data": {"id": "p1"}},
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        user_id="user-1",
    )

    assert context.messages[-1].role == "tool"
    assert context.messages[-1].tool_call_id == "call-1"
    assert context.observations[-1].tool_call_id == "call-1"
    assert context.step_count == 1


def test_restore_run_context_restores_async_skill_transition_and_artifact():
    context = restore_run_context(
        checkpoint={
            "schema_version": "harness-checkpoint-v2",
            "run_id": "run-debate",
            "messages": [],
            "observations": [],
            "artifacts": [],
            "pending_action": {
                "tool_call_id": "call-debate",
                "tool_name": "deliberation.start",
                "action_type": "school_deliberation",
            },
        },
        resume_payload={
            "status": "succeeded",
            "tool_name": "deliberation.collect_views",
            "summary": "多学派讨论已完成。",
            "artifact_type": "school_deliberation",
            "artifact_id": "debate-1",
            "source_id": "debate-1",
        },
        run_id="run-debate",
        thread_id="thread-debate",
        turn_id="turn-debate",
        user_id="user-1",
        skill_activation={
            "skill_id": "multi_view_deliberation",
            "completed_tools": [],
        },
    )

    assert context.skill_activation["completed_tools"] == ["deliberation.start"]
    assert len(context.artifacts) == 1
    assert context.artifacts[0].id == "debate-1"
    assert context.artifacts[0].artifact_type == "school_deliberation"


def test_restore_run_context_rejects_cross_run_checkpoint_data():
    with pytest.raises(ValidationError, match="observation.run_id must match run_id"):
        restore_run_context(
            checkpoint={
                "schema_version": "harness-checkpoint-v2",
                "run_id": "run-1",
                "messages": [],
                "observations": [
                    {
                        "id": "obs-foreign",
                        "run_id": "another-run",
                        "source": "demo.read",
                        "status": "succeeded",
                    }
                ],
                "artifacts": [],
                "pending_action": {
                    "tool_call_id": "call-1",
                    "action_type": "demo.confirm",
                },
            },
            resume_payload={"status": "succeeded", "summary": "confirmed"},
            run_id="run-1",
            thread_id="thread-1",
            turn_id="turn-1",
            user_id="user-1",
        )


@pytest.mark.parametrize(
    "loader,payload",
    [
        (checkpoint_from_runtime, {"checkpoint": {"messages": []}}),
        (checkpoint_from_durable_state, {"checkpoint": {"messages": []}}),
    ],
)
def test_checkpoint_readers_reject_unversioned_v1_payloads(loader, payload):
    with pytest.raises(CheckpointCompatibilityError, match="missing"):
        loader(payload)


def test_durable_state_uses_the_product_neutral_runtime_key():
    result = RuntimeResult(
        question="resume later",
        run_id="run-1",
        thread_id="thread-1",
        graph_thread_id="graph-1",
        turn_id="turn-1",
        status=RuntimeResultStatus.WAITING_USER_ACTION,
        stop_reason="requires_user_action",
        final_answer=FinalAnswer(
            markdown="Complete the action.",
            status="interrupted",
            stop_reason="requires_user_action",
        ),
        checkpoint={"schema_version": "harness-checkpoint-v2"},
    )

    durable = durable_state_from_result(result, run_id="run-1", thread_id="thread-1")

    assert durable["runtime"]["checkpoint"]["schema_version"] == (
        "harness-checkpoint-v2"
    )
    assert "agent_runtime" not in durable


@pytest.mark.parametrize(
    "loader,payload",
    [
        (
            checkpoint_from_runtime,
            {"schema_version": "harness-runtime-v3", "checkpoint": {}},
        ),
        (
            checkpoint_from_runtime,
            {
                "schema_version": "harness-runtime-v1",
                "checkpoint": {"schema_version": "harness-checkpoint-v2"},
            },
        ),
        (
            checkpoint_from_durable_state,
            {"schema_version": "harness-durable-checkpoint-v3"},
        ),
    ],
)
def test_checkpoint_readers_reject_unknown_explicit_versions(loader, payload):
    with pytest.raises(CheckpointCompatibilityError, match="unsupported"):
        loader(payload)


def test_restore_run_context_rejects_checkpoint_from_another_run():
    with pytest.raises(CheckpointCompatibilityError, match="run_id mismatch"):
        restore_run_context(
            checkpoint={
                "schema_version": "harness-checkpoint-v2",
                "run_id": "run-2",
                "messages": [],
                "observations": [],
                "artifacts": [],
            },
            resume_payload={"status": "succeeded"},
            run_id="run-1",
            thread_id="thread-1",
            turn_id="turn-1",
            user_id="user-1",
        )
