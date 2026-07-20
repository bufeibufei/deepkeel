from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
