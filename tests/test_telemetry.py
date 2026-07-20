from harness_core.telemetry import TelemetryRecord


def test_runtime_event_telemetry_preserves_correlations_and_excludes_content():
    event = {
        "schema_version": "harness-runtime-event-v1",
        "event_id": "event-1",
        "sequence": 7,
        "run_version": 3,
        "event_type": "model.completed",
        "source_event_type": "model.completed",
        "created_at": "2026-07-20T12:00:00+00:00",
        "payload": {
            "invocation_id": "invoke-1",
            "model_id": "model-1",
            "model_role": "reasoning",
            "status": "completed",
            "prompt": "private prompt",
            "arguments": {"secret": "value"},
            "result": "private result",
        },
    }

    record = TelemetryRecord.from_runtime_event(
        event,
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
    )

    assert record.schema_version == "harness-telemetry-v2"
    assert record.telemetry_id
    assert record.event_id == "event-1"
    assert record.sequence == 7
    assert record.run_version == 3
    assert record.component == "model"
    assert record.operation_id == "invoke-1"
    assert record.privacy_class == "operational_metadata"
    assert record.attributes == {
        "source_event_type": "model.completed",
        "invocation_id": "invoke-1",
        "model_id": "model-1",
        "model_role": "reasoning",
        "status": "completed",
    }


def test_direct_telemetry_records_receive_stable_identity():
    first = TelemetryRecord(
        event_name="runtime.settled",
        run_id="run-1",
        turn_id="turn-1",
        status="completed",
    )
    second = TelemetryRecord(
        event_name="runtime.settled",
        run_id="run-1",
        turn_id="turn-1",
        status="completed",
    )

    assert first.telemetry_id == second.telemetry_id
