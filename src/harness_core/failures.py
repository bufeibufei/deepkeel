from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from pydantic import ValidationError

from harness_core.budget import BudgetExceededError
from harness_core.policy import PolicyDeniedError


FailureCategory = Literal[
    "policy",
    "budget",
    "timeout",
    "upstream",
    "contract",
    "model_contract",
    "canceled",
    "internal",
]

FailureClass = Literal[
    "policy",
    "budget",
    "canceled",
    "input_required",
    "model_transport",
    "model_contract",
    "tool_execution",
    "event_projection",
    "artifact_contract",
    "checkpoint_recovery",
    "lease_coordination",
    "upstream",
    "internal",
]


class RunDeadlineExceededError(TimeoutError):
    code = "RUN_DEADLINE_EXCEEDED"

    def __init__(self) -> None:
        super().__init__("agent run deadline exceeded")


class RunCanceledError(RuntimeError):
    code = "RUN_CANCELED"

    def __init__(self) -> None:
        super().__init__("agent run canceled")


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    code: str
    category: FailureCategory
    retryable: bool
    user_message: str
    detail: str
    exception_type: str

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "category": self.category,
            "retryable": self.retryable,
            "user_message": self.user_message,
            "detail": self.detail,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True, slots=True)
class FailureDiagnosis:
    """Portable failure evidence for operators and product projections."""

    error_code: str
    failure_class: FailureClass
    stage: str
    source: str
    retryable: bool
    user_message: str
    technical_detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    recommended_action: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "error_code": self.error_code,
            "failure_class": self.failure_class,
            "stage": self.stage,
            "source": self.source,
            "retryable": self.retryable,
            "user_message": self.user_message,
            "technical_detail": self.technical_detail,
            "evidence": dict(self.evidence),
            "recommended_action": self.recommended_action,
        }


def diagnose_failure(
    *,
    code: str = "",
    detail: str = "",
    stage: str = "",
    source: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> FailureDiagnosis:
    """Normalize persisted or live failure evidence into one stable contract."""

    normalized_code = str(code or "").strip().upper()
    technical_detail = str(detail or "").strip()
    normalized_stage = str(stage or "unknown").strip() or "unknown"
    normalized_source = str(source or "").strip()
    evidence_payload = dict(evidence or {})
    searchable = " ".join(
        (
            normalized_code,
            technical_detail,
            normalized_stage,
            normalized_source,
            _flatten_failure_evidence(evidence_payload),
        )
    ).casefold()

    failure_class, inferred_code = _diagnostic_class(searchable, normalized_code)
    final_code = normalized_code
    if not final_code or final_code in {"RUN_FAILED", "RUNTIME_INTERNAL_ERROR"}:
        final_code = inferred_code
    if not normalized_source:
        normalized_source = _diagnostic_source(failure_class)

    runtime_failure = failure_from_code(final_code, technical_detail)
    retryable = runtime_failure.retryable
    if failure_class in {"policy", "budget", "canceled", "input_required", "artifact_contract"}:
        retryable = False
    elif failure_class in {
        "model_transport",
        "model_contract",
        "tool_execution",
        "event_projection",
        "checkpoint_recovery",
        "lease_coordination",
        "upstream",
    }:
        retryable = True

    return FailureDiagnosis(
        error_code=final_code,
        failure_class=failure_class,
        stage=normalized_stage,
        source=normalized_source,
        retryable=retryable,
        user_message=_diagnostic_user_message(failure_class),
        technical_detail=technical_detail or final_code,
        evidence=evidence_payload,
        recommended_action=_diagnostic_recommended_action(failure_class),
    )


def classify_runtime_failure(exc: Exception) -> RuntimeFailure:
    explicit_code = str(getattr(exc, "code", "") or "").strip().upper()
    detail = str(exc or "").strip() or type(exc).__name__
    exception_type = type(exc).__name__

    if isinstance(exc, RunCanceledError) or explicit_code in {
        "RUN_CANCELED",
        "RUN_CANCELLED",
        "SUBAGENT_CANCELED",
    }:
        return RuntimeFailure(
            code=explicit_code or "RUN_CANCELED",
            category="canceled",
            retryable=False,
            user_message="The run was canceled.",
            detail=detail,
            exception_type=exception_type,
        )

    if isinstance(exc, PolicyDeniedError) or explicit_code == "POLICY_DENIED":
        return RuntimeFailure(
            code=explicit_code or "POLICY_DENIED",
            category="policy",
            retryable=False,
            user_message="The operation was denied by policy. Revise the request and try again.",
            detail=detail,
            exception_type=exception_type,
        )
    if isinstance(exc, BudgetExceededError) or explicit_code == "BUDGET_EXCEEDED":
        return RuntimeFailure(
            code=explicit_code or "BUDGET_EXCEEDED",
            category="budget",
            retryable=False,
            user_message="The run exhausted its budget. Narrow the request and try again.",
            detail=detail,
            exception_type=exception_type,
        )
    if explicit_code == "MODEL_TOOL_CONTRACT_VIOLATION":
        return RuntimeFailure(
            code=explicit_code,
            category="model_contract",
            retryable=True,
            user_message=(
                "The model did not honor the required tool call. "
                "The run ended safely and may be retried."
            ),
            detail=detail,
            exception_type=exception_type,
        )
    if isinstance(exc, (TimeoutError,)) or "timeout" in exception_type.lower():
        return RuntimeFailure(
            code=explicit_code or "UPSTREAM_TIMEOUT",
            category="timeout",
            retryable=True,
            user_message="The upstream service timed out. The run ended safely and may be retried.",
            detail=detail,
            exception_type=exception_type,
        )
    upstream_hint = detail.lower()
    if isinstance(exc, (ConnectionError, OSError)) or any(
        marker in upstream_hint
        for marker in (
            "provider unavailable",
            "provider does not support",
            "service unavailable",
            "connection refused",
            "upstream unavailable",
        )
    ):
        return RuntimeFailure(
            code=explicit_code or "UPSTREAM_UNAVAILABLE",
            category="upstream",
            retryable=True,
            user_message="A dependency is unavailable. The run ended safely; try again later.",
            detail=detail,
            exception_type=exception_type,
        )
    if isinstance(exc, (ValidationError, ValueError)):
        return RuntimeFailure(
            code=explicit_code or "RUNTIME_CONTRACT_INVALID",
            category="contract",
            retryable=False,
            user_message="The result failed contract validation and the run ended safely.",
            detail=detail,
            exception_type=exception_type,
        )
    if explicit_code in {"RUN_CANCELED", "RUN_CANCELLED", "SUBAGENT_CANCELED"}:
        return RuntimeFailure(
            code=explicit_code,
            category="canceled",
            retryable=False,
            user_message="The run was canceled.",
            detail=detail,
            exception_type=exception_type,
        )
    return RuntimeFailure(
        code=explicit_code or "RUNTIME_INTERNAL_ERROR",
        category="internal",
        retryable=True,
        user_message="The run failed safely and may be retried.",
        detail=detail,
        exception_type=exception_type,
    )


def failure_from_code(code: str, detail: str = "") -> RuntimeFailure:
    normalized = str(code or "RUN_FAILED").strip().upper()
    message = str(detail or normalized).strip()
    if normalized == "POLICY_DENIED":
        category: FailureCategory = "policy"
        retryable = False
        user_message = "The operation was denied by policy. Revise the request and try again."
    elif normalized == "BUDGET_EXCEEDED":
        category = "budget"
        retryable = False
        user_message = "The run exhausted its budget. Narrow the request and try again."
    elif "TIMEOUT" in normalized:
        category = "timeout"
        retryable = True
        user_message = "The upstream service timed out. The run ended safely and may be retried."
    elif any(marker in normalized for marker in ("PROVIDER", "UPSTREAM", "ASYNC_TOOL")):
        category = "upstream"
        retryable = True
        user_message = "A dependency is unavailable. The run ended safely; try again later."
    elif any(marker in normalized for marker in ("CONTRACT", "SCHEMA", "VALIDATION")):
        category = "contract"
        retryable = False
        user_message = "The result failed contract validation and the run ended safely."
    elif any(marker in normalized for marker in ("CANCEL", "CANCELED", "CANCELLED")):
        category = "canceled"
        retryable = False
        user_message = "The run was canceled."
    else:
        category = "internal"
        retryable = not any(marker in normalized for marker in ("MISSING", "INVALID"))
        user_message = "The run failed safely and may be retried."
    return RuntimeFailure(
        code=normalized,
        category=category,
        retryable=retryable,
        user_message=user_message,
        detail=message,
        exception_type="PersistedRunFailure",
    )


def _diagnostic_class(searchable: str, code: str) -> tuple[FailureClass, str]:
    rules: tuple[tuple[FailureClass, str, tuple[str, ...]], ...] = (
        ("canceled", "RUN_CANCELED", ("cancelled", "canceled", "run_cancel")),
        ("policy", "POLICY_DENIED", ("policy_denied", "permission denied", "forbidden")),
        ("budget", "BUDGET_EXCEEDED", ("budget_exceeded", "budget exceeded")),
        (
            "input_required",
            "INPUT_REQUIRED",
            ("input_required", "missing requirement", "requires_user_input"),
        ),
        (
            "event_projection",
            "EVENT_SEQUENCE_CONFLICT",
            (
                "event sequence",
                "sequence is already owned",
                "projection conflict",
                "eventjournalconflict",
            ),
        ),
        (
            "model_transport",
            "MODEL_STREAM_INTERRUPTED",
            (
                "incompleteread",
                "peer closed connection",
                "stream interrupted",
                "model transport",
            ),
        ),
        (
            "model_contract",
            "MODEL_RESPONSE_INVALID",
            (
                "response_format",
                "json_schema",
                "structured output",
                "model_tool_contract",
            ),
        ),
        (
            "tool_execution",
            "TOOL_EXECUTION_FAILED",
            ("tool_execution", "tool execution", "mcp", "tool timed out"),
        ),
        (
            "artifact_contract",
            "ARTIFACT_CONTRACT_INVALID",
            ("artifact", "result card", "missing required artifact"),
        ),
        (
            "checkpoint_recovery",
            "CHECKPOINT_RECOVERY_FAILED",
            ("checkpoint", "resume failed", "restore state", "recovery failed"),
        ),
        (
            "lease_coordination",
            "LEASE_COORDINATION_FAILED",
            ("lease", "fencing token", "claim conflict", "worker heartbeat"),
        ),
        (
            "upstream",
            "UPSTREAM_UNAVAILABLE",
            ("connection refused", "service unavailable", "upstream", "http 5"),
        ),
    )
    for failure_class, inferred_code, markers in rules:
        if any(marker in searchable for marker in markers):
            return failure_class, inferred_code
    if "model" in searchable or code.startswith(("MODEL_", "LLM_")):
        return "model_transport", code or "MODEL_INVOCATION_FAILED"
    if "tool" in searchable or code.startswith(("TOOL_", "MCP_")):
        return "tool_execution", code or "TOOL_EXECUTION_FAILED"
    return "internal", code or "RUNTIME_INTERNAL_ERROR"


def _diagnostic_source(failure_class: FailureClass) -> str:
    return {
        "policy": "policy_engine",
        "budget": "budget_engine",
        "canceled": "run_control",
        "input_required": "workflow_contract",
        "model_transport": "model_provider",
        "model_contract": "model_provider",
        "tool_execution": "tool_runtime",
        "event_projection": "event_journal",
        "artifact_contract": "artifact_projection",
        "checkpoint_recovery": "checkpoint_store",
        "lease_coordination": "run_coordinator",
        "upstream": "upstream_service",
        "internal": "runtime",
    }[failure_class]


def _diagnostic_user_message(failure_class: FailureClass) -> str:
    return {
        "policy": "The operation was denied by policy.",
        "budget": "The run exhausted its configured budget.",
        "canceled": "The run was canceled.",
        "input_required": "More information is required before the run can continue.",
        "model_transport": "The model response was interrupted before completion.",
        "model_contract": "The model response did not satisfy the required contract.",
        "tool_execution": "A required tool could not complete successfully.",
        "event_projection": "The run event stream could not be projected consistently.",
        "artifact_contract": "The result artifact did not satisfy its contract.",
        "checkpoint_recovery": "The run could not be restored from its checkpoint.",
        "lease_coordination": "The run could not acquire or retain execution ownership.",
        "upstream": "A required dependency is unavailable.",
        "internal": "The run failed because of an internal runtime error.",
    }[failure_class]


def _diagnostic_recommended_action(failure_class: FailureClass) -> str:
    return {
        "policy": "Review the matched policy rule and required authorization.",
        "budget": "Narrow the task or increase the relevant run budget.",
        "canceled": "Start a new run only if the cancellation was unintended.",
        "input_required": "Collect the missing fields and resume the existing run.",
        "model_transport": "Retry with the same idempotency key, then inspect provider health.",
        "model_contract": "Inspect schema capability, raw output, and repair attempts.",
        "tool_execution": "Inspect tool inputs, upstream health, timeout, and replay record.",
        "event_projection": "Repair the event sequence safely before replaying the projection.",
        "artifact_contract": "Inspect the producer output and required artifact schema.",
        "checkpoint_recovery": "Inspect the latest checkpoint, state version, and resume token.",
        "lease_coordination": "Inspect worker heartbeat, lease ownership, and fencing token.",
        "upstream": "Inspect dependency health and retry according to backoff policy.",
        "internal": "Export the run trace and inspect the first failing operation.",
    }[failure_class]


def _flatten_failure_evidence(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {_flatten_failure_evidence(item)}"
            for key, item in value.items()
            if item not in (None, "", [], {}, ())
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_failure_evidence(item) for item in value)
    return str(value or "")
