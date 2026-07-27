from harness_core.runtime_sdk import FailureDiagnosis, diagnose_failure


def test_failure_diagnosis_classifies_event_sequence_conflict() -> None:
    diagnosis = diagnose_failure(
        code="RUNTIME_INTERNAL_ERROR",
        detail="runtime event sequence is already owned by another event",
        stage="event_persist",
    )

    assert isinstance(diagnosis, FailureDiagnosis)
    assert diagnosis.error_code == "EVENT_SEQUENCE_CONFLICT"
    assert diagnosis.failure_class == "event_projection"
    assert diagnosis.source == "event_journal"
    assert diagnosis.retryable is True
    assert diagnosis.evidence == {}


def test_failure_diagnosis_uses_nested_model_evidence() -> None:
    diagnosis = diagnose_failure(
        code="RUN_FAILED",
        stage="model_stream",
        evidence={
            "model": {
                "failure_type": "IncompleteRead",
                "failure_message": "peer closed connection",
            }
        },
    )

    assert diagnosis.error_code == "MODEL_STREAM_INTERRUPTED"
    assert diagnosis.failure_class == "model_transport"
    assert diagnosis.source == "model_provider"
    assert "provider health" in diagnosis.recommended_action


def test_failure_diagnosis_preserves_specific_tool_code() -> None:
    diagnosis = diagnose_failure(
        code="ASYNC_TOOL_FAILED",
        detail="run_failed",
        source="date_selection_worker",
        evidence={"tool": {"tool_name": "date_selection.generate"}},
    )

    assert diagnosis.error_code == "ASYNC_TOOL_FAILED"
    assert diagnosis.failure_class == "tool_execution"
    assert diagnosis.source == "date_selection_worker"
    assert diagnosis.as_dict()["evidence"]["tool"]["tool_name"] == "date_selection.generate"


def test_failure_diagnosis_does_not_infer_from_empty_evidence_keys() -> None:
    diagnosis = diagnose_failure(
        code="RUN_FAILED",
        stage="failed",
        evidence={"model": None, "tool": None, "event": None},
    )

    assert diagnosis.error_code == "RUNTIME_INTERNAL_ERROR"
    assert diagnosis.failure_class == "internal"
    assert diagnosis.source == "runtime"
