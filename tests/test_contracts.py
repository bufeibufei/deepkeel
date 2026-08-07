"""Contract tests owned by the standalone runtime package."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from deepkeel.contracts import (
    AgentMessage,
    Artifact,
    FinalAnswer,
    Observation,
    PendingAction,
    RunContext,
    RunStatus,
    RuntimeEvent,
    ToolCall,
    ToolResult,
)


def test_run_context_round_trip_preserves_provider_neutral_contracts():
    call = ToolCall(
        id="call-1",
        name="records.read_current",
        arguments={"record_id": "record-1"},
        idempotency_key="run-1:call-1",
        read_only=True,
        parallel_safe=True,
    )
    observation = Observation(
        id="obs-1",
        run_id="run-1",
        tool_call_id=call.id,
        source="records.read_current",
        status="succeeded",
        summary="Current record loaded.",
        data={"record_id": "record-1"},
    )
    context = RunContext(
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        user_id="user-1",
        status=RunStatus.REASONING,
        messages=[AgentMessage(id="msg-1", role="user", content="Show my current record")],
        observations=[observation],
        pending_tool_calls=[call],
    )

    restored = RunContext.model_validate_json(context.model_dump_json())

    assert restored == context
    assert restored.messages[0].role == "user"
    assert restored.observations[0].tool_call_id == "call-1"
    assert restored.created_at.tzinfo is not None


def test_tool_result_requires_matching_call_identity():
    call = ToolCall(id="call-1", name="system.get_current_time", arguments={})

    with pytest.raises(ValidationError):
        ToolResult(
            call=call,
            tool_call_id="call-2",
            status="succeeded",
            summary="ok",
        )


def test_pending_action_and_artifact_are_serializable_projection_inputs():
    action = PendingAction(
        id="action-1",
        run_id="run-1",
        tool_call_id="call-1",
        action_type="record.select",
        title="Select a record",
        prompt="Select a record to continue.",
        handoff_view="record_picker",
        payload={"allow_create": True},
    )
    artifact = Artifact(
        id="artifact-1",
        run_id="run-1",
        artifact_type="analysis.report",
        title="Analysis report",
        summary="Analysis completed.",
        source_id="report-1",
    )
    answer = FinalAnswer(
        markdown="报告已经生成。",
        summary="报告已经生成。",
        artifact_ids=[artifact.id],
    )

    payload = {
        "action": action.model_dump(mode="json"),
        "artifact": artifact.model_dump(mode="json"),
        "answer": answer.model_dump(mode="json"),
    }

    assert payload["action"]["status"] == "pending"
    assert payload["artifact"]["artifact_type"] == "analysis.report"
    assert payload["answer"]["artifact_ids"] == ["artifact-1"]


def test_runtime_event_supports_ephemeral_model_delta_without_sequence():
    event = RuntimeEvent(
        run_id="run-1",
        event_type="model.delta",
        payload={"delta": "今"},
        ephemeral=True,
        created_at=datetime.now(UTC),
    )

    assert event.sequence is None
    assert event.ephemeral is True


def test_non_ephemeral_runtime_event_requires_positive_sequence():
    with pytest.raises(ValidationError):
        RuntimeEvent(
            run_id="run-1",
            event_type="tool.completed",
            sequence=0,
            ephemeral=False,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        (
            "observations",
            [
                Observation(
                    id="obs-foreign",
                    run_id="run-2",
                    source="demo.read",
                    status="succeeded",
                )
            ],
            "observation.run_id must match run_id",
        ),
        (
            "artifacts",
            [
                Artifact(
                    id="artifact-foreign",
                    run_id="run-2",
                    artifact_type="demo.result",
                )
            ],
            "artifact.run_id must match run_id",
        ),
        (
            "pending_action",
            PendingAction(
                id="action-foreign",
                run_id="run-2",
                action_type="demo.confirm",
            ),
            "pending_action.run_id must match run_id",
        ),
    ],
)
def test_run_context_rejects_cross_run_projection(field_name, value, message):
    with pytest.raises(ValidationError, match=message):
        RunContext(
            run_id="run-1",
            thread_id="thread-1",
            turn_id="turn-1",
            user_id="user-1",
            **{field_name: value},
        )


@pytest.mark.parametrize(
    ("field_name", "items"),
    [
        (
            "messages",
            [
                AgentMessage(id="duplicate", role="user", content="one"),
                AgentMessage(id="duplicate", role="assistant", content="two"),
            ],
        ),
        (
            "pending_tool_calls",
            [
                ToolCall(id="duplicate", name="demo.one"),
                ToolCall(id="duplicate", name="demo.two"),
            ],
        ),
    ],
)
def test_run_context_rejects_duplicate_child_ids(field_name, items):
    with pytest.raises(ValidationError, match=f"{field_name} must have unique ids"):
        RunContext(
            run_id="run-1",
            thread_id="thread-1",
            turn_id="turn-1",
            user_id="user-1",
            **{field_name: items},
        )


def test_tool_result_rejects_mismatched_projection_correlation():
    call = ToolCall(id="call-1", name="demo.read")

    with pytest.raises(ValidationError, match="Observation.tool_call_id must match"):
        ToolResult(
            call=call,
            status="succeeded",
            observation=Observation(
                id="obs-1",
                run_id="run-1",
                tool_call_id="call-2",
                source="demo.read",
                status="succeeded",
            ),
        )


def test_tool_result_rejects_projections_from_different_runs():
    call = ToolCall(id="call-1", name="demo.read")

    with pytest.raises(ValidationError, match="must belong to one run"):
        ToolResult(
            call=call,
            status="succeeded",
            observation=Observation(
                id="obs-1",
                run_id="run-1",
                tool_call_id="call-1",
                source="demo.read",
                status="succeeded",
            ),
            artifacts=[
                Artifact(
                    id="artifact-1",
                    run_id="run-2",
                    artifact_type="demo.result",
                )
            ],
        )
