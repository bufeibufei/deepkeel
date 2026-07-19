from types import MethodType

import pytest
from pydantic import ValidationError

from harness_core import HarnessRuntime, RuntimeRequest, RuntimeResultStatus
from harness_core.runtime_api import RuntimeResult


def _compatibility_payload() -> dict:
    return {
        "question": "Check inventory",
        "final_answer": {
            "markdown": "Inventory is available.",
            "summary": "Available",
            "answer_mode": "bubble",
            "references": [{"kind": "record", "id": "ref-1"}],
        },
        "agent_runtime": {
            "status": "completed",
            "stop_reason": "final_answer",
            "identity": {
                "run_id": "run-1",
                "thread_id": "thread-1",
                "graph_thread_id": "graph-1",
                "turn_id": "turn-1",
            },
            "checkpoint": {
                "run_id": "run-1",
                "graph_thread_id": "graph-1",
                "turn_id": "turn-1",
                "observations": [
                    {
                        "id": "obs-1",
                        "run_id": "run-1",
                        "source": "inventory.lookup",
                        "status": "succeeded",
                        "summary": "Found one record",
                    }
                ],
                "artifacts": [
                    {
                        "id": "artifact-1",
                        "run_id": "run-1",
                        "artifact_type": "inventory.record",
                    }
                ],
                "pending_action": None,
            },
            "diagnostics": {"steps": 1},
        },
        "events": [
            {
                "event_type": "answer.delta",
                "source_event_type": "model.delta",
                "payload": {"delta": "Available"},
                "ephemeral": True,
            }
        ],
        "context_snapshot": {"schema_version": "runtime-context-v1"},
        "skill_activation": {"skill_id": "inventory-assistant"},
        "answer_delta_streamed": True,
    }


def test_runtime_result_projects_the_v1_mapping_into_typed_contracts():
    result = RuntimeResult.from_compatibility_payload(_compatibility_payload())

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
    assert "compatibility_payload" not in result.model_dump()
    assert result.to_compatibility_payload()["agent_runtime"]["status"] == "completed"


def test_runtime_request_rejects_unknown_host_fields():
    with pytest.raises(ValidationError):
        RuntimeRequest(question="hello", product_database_id="db-1")


def test_waiting_result_marks_the_answer_as_interrupted():
    payload = _compatibility_payload()
    payload["agent_runtime"]["status"] = "waiting_user_action"
    payload["agent_runtime"]["stop_reason"] = "requires_user_action"
    payload["final_answer"].pop("status", None)

    result = RuntimeResult.from_compatibility_payload(payload)

    assert result.status is RuntimeResultStatus.WAITING_USER_ACTION
    assert result.final_answer.status == "interrupted"
    assert result.final_answer.stop_reason == "requires_user_action"


def test_harness_runtime_typed_entrypoint_delegates_to_compatible_loop():
    runtime = object.__new__(HarnessRuntime)
    captured = {}

    def fake_run_turn(_self, question, **kwargs):
        captured.update(question=question, **kwargs)
        return _compatibility_payload()

    runtime.run_turn = MethodType(fake_run_turn, runtime)
    result = runtime.run(
        RuntimeRequest(
            question="Check inventory",
            user_id="user-1",
            context_bundle={"thread_id": "thread-1"},
        )
    )

    assert result.status is RuntimeResultStatus.COMPLETED
    assert captured["question"] == "Check inventory"
    assert captured["user_id"] == "user-1"
    assert captured["context_bundle"] == {"thread_id": "thread-1"}
