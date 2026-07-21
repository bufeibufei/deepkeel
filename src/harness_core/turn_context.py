from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Literal

from harness_core.model import ModelGateway
from harness_core.tools import ToolExecutionContext


EventSink = Callable[[dict[str, Any]], None]
ToolViewMode = Literal["legacy", "shadow", "enforced"]


@dataclass(frozen=True, slots=True)
class TurnExecutionContext:
    """Dependencies that vary per turn while the compiled graph stays reusable."""

    model: ModelGateway
    system_prompt: str
    tool_context: ToolExecutionContext
    event_sink: EventSink | None = None
    deadline_monotonic: float | None = None
    tool_view_mode: ToolViewMode = "legacy"


class TurnContextRegistry:
    """Process-local binding for dependencies that must never enter checkpoints."""

    def __init__(self) -> None:
        self._contexts: dict[str, TurnExecutionContext] = {}
        self._lock = Lock()

    def bind(self, context: TurnExecutionContext, *keys: str) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(str(key) for key in keys if str(key or "")))
        with self._lock:
            for key in normalized:
                self._contexts[key] = context
        return normalized

    def resolve(self, *keys: str) -> TurnExecutionContext | None:
        with self._lock:
            for key in keys:
                context = self._contexts.get(str(key or ""))
                if context is not None:
                    return context
        return None

    def release(self, context: TurnExecutionContext, *keys: str) -> None:
        with self._lock:
            for key in keys:
                normalized = str(key or "")
                if self._contexts.get(normalized) is context:
                    self._contexts.pop(normalized, None)
