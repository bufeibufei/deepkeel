from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from deepkeel.contracts import RunContext
from deepkeel.graph import HarnessGraph, create_harness_graph
from deepkeel.graph_state import HarnessGraphState
from deepkeel.tools import ToolExecutionContext
from deepkeel.turn_context import TurnExecutionContext


class TurnExecutionEngine(Protocol):
    """Internal execution seam consumed by the runtime turn coordinator."""

    engine_id: str
    contract_version: str

    async def ainvoke(
        self,
        context: RunContext,
        *,
        tool_context: ToolExecutionContext,
        event_sink: Any = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState: ...

    async def aresume(
        self,
        thread_id: str,
        resume_payload: dict[str, Any],
        *,
        tool_context: ToolExecutionContext,
        event_sink: Any = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState: ...

    async def arecover(
        self,
        thread_id: str,
        *,
        tool_context: ToolExecutionContext,
        event_sink: Any = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState: ...


@dataclass(slots=True)
class LangGraphExecutionEngine:
    """Adapter that keeps LangGraph types behind DeepKeel's execution contract."""

    graph: HarnessGraph
    engine_id: str = "langgraph"
    contract_version: str = "deepkeel-execution-engine-v1"

    async def ainvoke(
        self,
        context: RunContext,
        *,
        tool_context: ToolExecutionContext,
        event_sink: Any = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        return await self.graph.ainvoke(
            context,
            tool_context=tool_context,
            event_sink=event_sink,
            turn_context=turn_context,
        )

    async def aresume(
        self,
        thread_id: str,
        resume_payload: dict[str, Any],
        *,
        tool_context: ToolExecutionContext,
        event_sink: Any = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        return await self.graph.aresume(
            thread_id,
            resume_payload,
            tool_context=tool_context,
            event_sink=event_sink,
            turn_context=turn_context,
        )

    async def arecover(
        self,
        thread_id: str,
        *,
        tool_context: ToolExecutionContext,
        event_sink: Any = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        return await self.graph.arecover(
            thread_id,
            tool_context=tool_context,
            event_sink=event_sink,
            turn_context=turn_context,
        )


def create_langgraph_execution_engine(**graph_options: Any) -> LangGraphExecutionEngine:
    """Compile the built-in graph behind the internal execution-engine seam."""

    return LangGraphExecutionEngine(create_harness_graph(**graph_options))


__all__ = [
    "LangGraphExecutionEngine",
    "TurnExecutionEngine",
    "create_langgraph_execution_engine",
]
