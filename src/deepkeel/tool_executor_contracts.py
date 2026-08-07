from __future__ import annotations

from typing import Awaitable, Callable

from deepkeel.contracts import ToolCall, ToolResult
from deepkeel.tool_execution import ToolExecutionContext
from deepkeel.tool_registry import ToolSpec


ToolHandler = Callable[
    [ToolCall, ToolExecutionContext],
    ToolResult | Awaitable[ToolResult],
]
ToolPreflight = Callable[
    [ToolCall, ToolExecutionContext, ToolSpec],
    str | None,
]
