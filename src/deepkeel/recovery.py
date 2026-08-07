from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RecoveryState = Literal[
    "not_attempted",
    "recovering",
    "resumed_success",
    "exhausted_typed_failure",
    "exhausted_untyped_failure",
    "aborted",
    "stuck",
]

_GENERIC_FAILURE_CODES = frozenset(
    {"", "INTERNAL_ERROR", "RUNTIME_INTERNAL_ERROR", "RUN_FAILED", "UNKNOWN_ERROR"}
)


class RecoveryOutcome(BaseModel):
    """Portable, payload-free summary of a durable recovery attempt."""

    model_config = ConfigDict(extra="forbid")

    state: RecoveryState = "not_attempted"
    attempts: int = Field(default=0, ge=0)
    diagnosed: bool = True
    error_code: str = ""
    failure_fingerprint: str = ""

    @property
    def terminalized(self) -> bool:
        return self.state in {
            "not_attempted",
            "resumed_success",
            "exhausted_typed_failure",
            "exhausted_untyped_failure",
            "aborted",
        }


def classify_recovery_outcome(
    *,
    runtime_status: str,
    attempts: int,
    error_code: str = "",
    error_message: str = "",
    stale: bool = False,
) -> RecoveryOutcome:
    """Classify recovery without requiring product-specific run models."""

    normalized_attempts = max(0, int(attempts or 0))
    status = str(runtime_status or "").strip().lower()
    status = "canceled" if status == "cancelled" else status
    code = str(error_code or "").strip().upper()
    message = str(error_message or "").strip()
    diagnosed = bool(code and code not in _GENERIC_FAILURE_CODES)
    fingerprint = _failure_fingerprint(code, message) if code or message else ""

    if normalized_attempts == 0:
        return RecoveryOutcome(state="not_attempted", attempts=0)
    if status == "completed":
        state: RecoveryState = "resumed_success"
        diagnosed = True
    elif status == "failed":
        state = "exhausted_typed_failure" if diagnosed else "exhausted_untyped_failure"
    elif status == "canceled":
        state = "aborted"
        diagnosed = True
    elif stale:
        state = "stuck"
        diagnosed = False
    else:
        state = "recovering"
        diagnosed = True
    return RecoveryOutcome(
        state=state,
        attempts=normalized_attempts,
        diagnosed=diagnosed,
        error_code=code,
        failure_fingerprint=fingerprint,
    )


def _failure_fingerprint(error_code: str, error_message: str) -> str:
    normalized_message = " ".join(error_message.lower().split())[:500]
    source = f"{error_code}|{normalized_message}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
