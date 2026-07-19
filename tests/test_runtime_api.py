import inspect

import pytest
from pydantic import ValidationError

from harness_core import HarnessRuntime, RuntimeRequest, RuntimeResultStatus
from harness_core.contracts import Artifact, FinalAnswer, Observation
from harness_core.runtime_api import RuntimeResult, RuntimeStreamEvent


def _runtime_result(
    *,
    status: RuntimeResultStatus = RuntimeResultStatus.COMPLETED,
) -> RuntimeResult:
    return RuntimeResult(
        question="Check inventory",
        run_id="run-1",
        thread_id="thread-1",
        graph_thread_id="graph-1",
        turn_id="turn-1",
        status=status,
        stop_reason=(
            "final_answer"
            if status is RuntimeResultStatus.COMPLETED
            else "requires_user_action"
        ),
        final_answer=FinalAnswer(
            markdown="Inventory is available.",
            summary="Available",
            status=(
                "completed"
                if status is RuntimeResultStatus.COMPLETED
                else "interrupted"
            ),
            stop_reason=(
                "final_answer"
                if status is RuntimeResultStatus.COMPLETED
                else "requires_user_action"
            ),
            references=[{"kind": "record", "id": "ref-1"}],
            metadata={"answer_mode": "bubble"},
        ),
        observations=[
            Observation(
                id="obs-1",
                run_id="run-1",
                source="inventory.lookup",
                status="succeeded",
                summary="Found one record",
            )
        ],
        artifacts=[
            Artifact(
                id="artifact-1",
                run_id="run-1",
                artifact_type="inventory.record",
            )
        ],
        events=[
            RuntimeStreamEvent(
                event_type="answer.delta",
                source_event_type="model.delta",
                payload={"delta": "Available"},
                ephemeral=True,
            )
        ],
        context_snapshot={"schema_version": "runtime-context-v2"},
        skill_activation={"skill_id": "inventory-assistant"},
        answer_delta_streamed=True,
    )


def test_runtime_result_is_a_direct_typed_contract():
    result = _runtime_result()

    assert result.status is RuntimeResultStatus.COMPLETED
    assert (result.run_id, result.thread_id, result.graph_thread_id) == (
        "run-1",
        "thread-1",
        "graph-1",
    )
    assert result.final_answer.metadata["answer_mode"] == "bubble"
    assert result.observations[0].source == "inventory.lookup"
    assert result.artifacts[0].artifact_type == "inventory.record"
    assert result.events[0].event_type == "answer.delta"
    assert "compatibility_payload" not in RuntimeResult.model_fields


def test_runtime_request_rejects_unknown_host_fields():
    with pytest.raises(ValidationError):
        RuntimeRequest(question="hello", product_database_id="db-1")


def test_waiting_result_marks_the_answer_as_interrupted():
    result = _runtime_result(status=RuntimeResultStatus.WAITING_USER_ACTION)

    assert result.final_answer.status == "interrupted"
    assert result.final_answer.stop_reason == "requires_user_action"


def test_harness_runtime_exposes_only_the_typed_execution_entrypoint():
    assert not hasattr(HarnessRuntime, "run_turn")
    signature = inspect.signature(HarnessRuntime.run)
    assert signature.parameters["request"].annotation == "RuntimeRequest"
    assert signature.return_annotation == "RuntimeResult"
