import inspect

import pytest
from pydantic import ValidationError

from deepkeel.runtime_sdk import HarnessRuntime, RuntimeRequest, RuntimeResultStatus
from deepkeel.contracts import Artifact, FinalAnswer, Observation, RunContext
from deepkeel.runtime_api import RuntimeResult, RuntimeStreamEvent
from deepkeel.scope import RuntimeScope, resolve_runtime_scope


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
        run_context=RunContext(
            run_id="run-1",
            thread_id="thread-1",
            turn_id="turn-1",
            user_id="user-1",
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
        ui_state={
            "schema_version": "harness-run-ui-v2",
            "lifecycle": "completed",
            "execution_status": "completed",
            "composer_mode": "ready",
            "can_send": True,
            "input_strategy": "follow_up",
            "requires_user_action": False,
            "is_resumable": False,
            "show_progress": False,
            "can_cancel": False,
            "active_task": None,
            "reason": "run_terminal",
        },
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
    assert result.artifact_views == []
    assert result.evidence_bundle.schema_version == "evidence-bundle-v1"
    assert result.events[0].event_type == "answer.delta"
    assert "compatibility_payload" not in RuntimeResult.model_fields


def test_runtime_request_rejects_unknown_host_fields():
    with pytest.raises(ValidationError):
        RuntimeRequest(question="hello", product_database_id="db-1")


def test_runtime_request_rejects_conflicting_scope_and_scalar_identity():
    with pytest.raises(ValidationError, match="scope conflicts"):
        RuntimeRequest(
            question="hello",
            user_id="scalar-user",
            scope=RuntimeScope(user_id="scoped-user"),
        )


def test_runtime_request_accepts_matching_scope_and_scalar_identity():
    scope = RuntimeScope(
        tenant_id="tenant-a",
        user_id="user-a",
        namespace="support",
    )
    request = RuntimeRequest(
        question="hello",
        tenant_id="tenant-a",
        user_id="user-a",
        namespace="support",
        scope=scope,
    )

    assert request.runtime_scope == scope


def test_scope_resolution_rejects_duplicate_conflicting_identity():
    with pytest.raises(ValueError, match="scope conflicts"):
        resolve_runtime_scope(
            RuntimeScope(tenant_id="tenant-a", user_id="user-a"),
            tenant_id="tenant-b",
        )


def test_waiting_result_marks_the_answer_as_interrupted():
    result = _runtime_result(status=RuntimeResultStatus.WAITING_USER_ACTION)

    assert result.final_answer.status == "interrupted"
    assert result.final_answer.stop_reason == "requires_user_action"


def test_harness_runtime_exposes_only_the_typed_execution_entrypoint():
    assert not hasattr(HarnessRuntime, "run_turn")
    signature = inspect.signature(HarnessRuntime.run)
    assert signature.parameters["request"].annotation == "RuntimeRequest"
    assert signature.return_annotation == "RuntimeResult"
