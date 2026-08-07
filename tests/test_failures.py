"""Failure-classification tests owned by the standalone runtime package."""

from types import SimpleNamespace
from urllib.error import URLError

from deepkeel.budget import BudgetDecision, BudgetExceededError
from deepkeel.failures import (
    RunCanceledError,
    RunDeadlineExceededError,
    classify_runtime_failure,
    failure_from_code,
)
from deepkeel.model_failures import (
    ModelToolArgumentsError,
    ModelToolContractError,
    classify_model_failure,
    provider_fingerprint,
)
from deepkeel.persistence import CheckpointCompatibilityError
from deepkeel.policy import PolicyDecision, PolicyDeniedError


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


def test_runtime_failure_contract_classifies_model_contract_cancel_and_internal_errors():
    model_contract = classify_runtime_failure(
        ModelToolContractError("report.build", [])
    )
    malformed_arguments = classify_runtime_failure(
        ModelToolArgumentsError("invalid_json", character_count=128)
    )
    canceled = classify_runtime_failure(RunCanceledError())
    internal = classify_runtime_failure(RuntimeError("unexpected"))

    assert model_contract.category == "model_contract"
    assert model_contract.retryable is True
    assert classify_model_failure(
        ModelToolContractError("report.build", [])
    ).degrades_provider_health is False
    assert malformed_arguments.code == "MODEL_TOOL_ARGUMENTS_INVALID"
    assert malformed_arguments.category == "model_contract"
    assert malformed_arguments.retryable is True
    assert classify_model_failure(
        ModelToolArgumentsError("invalid_json", character_count=128)
    ).degrades_provider_health is False
    assert canceled.category == "canceled"
    assert canceled.retryable is False
    assert internal.code == "RUNTIME_INTERNAL_ERROR"
    assert internal.category == "internal"


def test_persisted_failure_codes_reconstruct_safe_categories() -> None:
    expected = {
        "POLICY_DENIED": ("policy", False),
        "BUDGET_EXCEEDED": ("budget", False),
        "MODEL_TIMEOUT": ("timeout", True),
        "UPSTREAM_UNAVAILABLE": ("upstream", True),
        "OUTPUT_SCHEMA_INVALID": ("contract", False),
        "RUN_CANCELLED": ("canceled", False),
        "PROFILE_MISSING": ("internal", False),
        "UNKNOWN_FAILURE": ("internal", True),
    }

    for code, contract in expected.items():
        failure = failure_from_code(code, "technical detail")
        assert (failure.category, failure.retryable) == contract
        assert failure.detail == "technical detail"
        assert failure.as_dict()["code"] == code


def test_model_failure_classifier_covers_provider_failure_categories() -> None:
    rate_limited = RuntimeError("too many requests")
    rate_limited.status_code = 429
    rate_limited.headers = {"Retry-After": "1.5"}
    invalid = RuntimeError("bad request")
    invalid.response = SimpleNamespace(status_code=422, headers={})

    assert classify_model_failure(rate_limited).retry_after_seconds == 1.5
    assert classify_model_failure(TimeoutError()).category == "timeout"
    assert classify_model_failure(invalid).category == "invalid_request"
    assert classify_model_failure(URLError("offline")).category == "provider_unavailable"
    assert classify_model_failure(RuntimeError("unknown")).category == "provider_error"


def test_model_failure_classifier_retries_malformed_tool_arguments() -> None:
    failure = classify_model_failure(
        ModelToolArgumentsError("invalid_json", character_count=512)
    )

    assert failure.category == "tool_arguments_invalid"
    assert failure.retryable is True
    assert failure.status_code is None


def test_model_failure_classifier_retries_provider_parameter_drift_only() -> None:
    drift = RuntimeError(
        "HTTP Error 400: Bad Request: A parameter specified in the request is not valid"
    )
    drift.status_code = 400
    ordinary_bad_request = RuntimeError("HTTP Error 400: malformed messages")
    ordinary_bad_request.status_code = 400

    failure = classify_model_failure(drift)

    assert failure.category == "provider_parameter_drift"
    assert failure.retryable is True
    assert classify_model_failure(ordinary_bad_request).retryable is False


def test_model_failure_helpers_tolerate_malformed_metadata() -> None:
    malformed = RuntimeError("rate limit")
    malformed.code = object()
    malformed.headers = {"retry-after": "not-a-number"}
    failure = classify_model_failure(malformed)

    class Provider:
        base_url = "https://models.example.test"
        model = "reasoning-model"

    assert failure.category == "rate_limited"
    assert failure.status_code == 429
    assert failure.retry_after_seconds == 0.0
    assert provider_fingerprint(None) == ("", "", "")
    assert provider_fingerprint(Provider()) == (
        "test_model_failure_helpers_tolerate_malformed_metadata.<locals>.Provider",
        "https://models.example.test",
        "reasoning-model",
    )
