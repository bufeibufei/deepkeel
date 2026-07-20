from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable



EventSink = Callable[[dict[str, Any]], None]


class SubAgentOutputError(RuntimeError):
    def __init__(self, message: str, *, raw_text: str = "", diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.diagnostics = diagnostics or {}


class SubAgentEmptyResponseError(RuntimeError):
    """Normalized transient failure for model calls that produced no usable turn."""


class SubAgentCanceledError(RuntimeError):
    pass


class DelegationPreflightError(ValueError):
    """A delegation batch is invalid and no child run has been started."""

    code = "DELEGATION_INPUT_CONTRACT_VIOLATION"

    def __init__(self, issues: list[dict[str, str]]) -> None:
        self.issues = tuple(dict(item) for item in issues)
        details = "; ".join(
            f"{item.get('task_id') or 'unknown'}: {item.get('detail') or 'invalid task'}"
            for item in issues
        )
        super().__init__(f"delegation preflight failed: {details}")


@dataclass(slots=True)
class _DelegationQuota:
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    model_calls: int = 0
    tool_calls: int = 0
    lock: Lock = field(default_factory=Lock)

    def reserve_model_call(self) -> None:
        with self.lock:
            if self.max_model_calls is not None and self.model_calls >= self.max_model_calls:
                raise RuntimeError("delegation model call budget exceeded")
            self.model_calls += 1

    def reserve_tool_calls(self, count: int) -> None:
        with self.lock:
            if (
                self.max_tool_calls is not None
                and self.tool_calls + count > self.max_tool_calls
            ):
                raise RuntimeError("delegation tool call budget exceeded")
            self.tool_calls += count
