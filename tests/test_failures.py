"""Failure-classification tests owned by the standalone runtime package."""

from harness_core.budget import BudgetDecision, BudgetExceededError
from harness_core.failures import RunDeadlineExceededError, classify_runtime_failure
from harness_core.persistence import CheckpointCompatibilityError
from harness_core.policy import PolicyDecision, PolicyDeniedError


def test_runtime_failure_contract_separates_user_copy_from_internal_detail():
    failure = classify_runtime_failure(ConnectionError("connection refused by 10.0.0.8"))

    assert failure.code == "UPSTREAM_UNAVAILABLE"
    assert failure.category == "upstream"
    assert failure.retryable is True
    assert "10.0.0.8" not in failure.user_message
    assert "10.0.0.8" in failure.detail


def test_runtime_failure_contract_classifies_policy_and_budget_as_non_retryable():
    policy = classify_runtime_failure(
        PolicyDeniedError(PolicyDecision(allowed=False, reason="denied"))
    )
    budget_error = BudgetExceededError(BudgetDecision(
        allowed=False,
        metric="model_calls",
        requested=1,
        used=2,
        remaining=0,
        limit=2,
        reason="model_calls budget exceeded",
    ))
    budget = classify_runtime_failure(budget_error)

    assert (policy.code, policy.category, policy.retryable) == (
        "POLICY_DENIED",
        "policy",
        False,
    )
    assert (budget.code, budget.category, budget.retryable) == (
        "BUDGET_EXCEEDED",
        "budget",
        False,
    )


def test_runtime_failure_contract_preserves_run_deadline_code():
    failure = classify_runtime_failure(RunDeadlineExceededError())

    assert (failure.code, failure.category, failure.retryable) == (
        "RUN_DEADLINE_EXCEEDED",
        "timeout",
        True,
    )


def test_runtime_failure_contract_classifies_incompatible_checkpoint_as_contract_error():
    failure = classify_runtime_failure(
        CheckpointCompatibilityError("unsupported checkpoint schema version")
    )

    assert (failure.code, failure.category, failure.retryable) == (
        "CHECKPOINT_INCOMPATIBLE",
        "contract",
        False,
    )
