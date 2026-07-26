from harness_core.runtime_sdk import workflow_projection


def test_workflow_projection_normalizes_lifecycle_and_progress() -> None:
    queued = workflow_projection(
        instance_id="workflow-1",
        kind="report",
        status="task_running",
        phase="starting",
        revision=-2,
        event_sequence=-1,
    )
    waiting = workflow_projection(
        instance_id="workflow-1",
        kind="report",
        status="waiting_input",
        phase="waiting_input",
        progress=2,
    )
    completed = workflow_projection(
        instance_id="workflow-1",
        kind="report",
        status="success",
        phase="completed",
        progress="invalid",
    )

    assert queued["state"] == "queued"
    assert queued["lifecycle"] == "queued"
    assert queued["execution_status"] == "running"
    assert queued["status"] == "running"
    assert queued["revision"] == queued["event_sequence"] == 0
    assert queued["can_stop"] is True
    assert waiting["state"] == "waiting_user_input"
    assert waiting["lifecycle"] == "collecting_input"
    assert waiting["input_blocked"] is False
    assert waiting["progress"] == 1.0
    assert completed["state"] == "completed"
    assert completed["lifecycle"] == "completed"
    assert completed["terminal"] is True
    assert completed["progress"] == 1.0


def test_workflow_projection_handles_stop_and_unknown_states() -> None:
    stopping = workflow_projection(
        instance_id="workflow-2",
        kind="job",
        status="mystery",
        phase="mystery",
        stop_requested=True,
    )
    running = workflow_projection(
        instance_id="",
        kind="job",
        status="mystery",
        phase="mystery",
        progress=-1,
    )

    assert stopping["state"] == "stopping"
    assert stopping["input_blocked"] is True
    assert running["state"] == "running"
    assert running["recoverable"] is False
    assert running["progress"] == 0.0
