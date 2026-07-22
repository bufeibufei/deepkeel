from harness_core.runtime_sdk import RecoveryOutcome, classify_recovery_outcome


def test_recovery_outcome_distinguishes_success_and_typed_exhaustion() -> None:
    succeeded = classify_recovery_outcome(runtime_status="completed", attempts=1)
    exhausted = classify_recovery_outcome(
        runtime_status="failed",
        attempts=3,
        error_code="RUN_RECOVERY_FAILED",
        error_message="checkpoint could not be resumed",
    )

    assert isinstance(succeeded, RecoveryOutcome)
    assert succeeded.state == "resumed_success"
    assert succeeded.terminalized is True
    assert exhausted.state == "exhausted_typed_failure"
    assert exhausted.diagnosed is True
    assert exhausted.failure_fingerprint


def test_recovery_outcome_rejects_generic_internal_error_as_diagnosis() -> None:
    outcome = classify_recovery_outcome(
        runtime_status="failed",
        attempts=3,
        error_code="RUNTIME_INTERNAL_ERROR",
        error_message="safe failure",
    )

    assert outcome.state == "exhausted_untyped_failure"
    assert outcome.diagnosed is False
