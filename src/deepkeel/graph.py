from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deepkeel.budget import BudgetLedger
from deepkeel.contracts import RunContext
from deepkeel.control import NoopRunControl, RunControl
from deepkeel.graph_nodes import GraphNodes
from deepkeel.graph_state import (
    HarnessGraphState,
    _state_from_context,
    migrate_legacy_graph_state,
    validate_graph_state,
)
from deepkeel.graph_workflow import (
    _graph_config,
    _route_after_model,
    _route_after_tools,
    _route_after_user_resume,
    _route_from_start,
)
from deepkeel.model import ModelGateway
from deepkeel.prompts import harness_system_prompt
from deepkeel.tool_registry import ToolRegistry
from deepkeel.tools import ToolExecutionContext, ToolExecutor
from deepkeel.turn_context import TurnContextRegistry, TurnExecutionContext


EventSink = Callable[[dict[str, Any]], None]
HARNESS_GRAPH_CONTRACT_VERSION = "harness-graph-v1"
GraphDurability = Literal["sync", "async", "exit"]


@dataclass(slots=True)
class HarnessGraph:
    compiled_graph: Any
    supports_async_checkpointer: bool = True
    durability: GraphDurability = "exit"
    contract_version: str = HARNESS_GRAPH_CONTRACT_VERSION
    turn_contexts: TurnContextRegistry | None = None

    def _bind_turn_context(
        self,
        turn_context: TurnExecutionContext | None,
        *keys: str,
    ) -> tuple[str, ...]:
        if turn_context is None or self.turn_contexts is None:
            return ()
        return self.turn_contexts.bind(
            turn_context,
            turn_context.tool_context.run_id,
            turn_context.tool_context.thread_id,
            *keys,
        )

    def _release_turn_context(
        self,
        turn_context: TurnExecutionContext | None,
        keys: tuple[str, ...],
    ) -> None:
        if turn_context is not None and self.turn_contexts is not None:
            self.turn_contexts.release(turn_context, *keys)

    def invoke(
        self,
        context: RunContext,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        keys = self._bind_turn_context(turn_context, context.run_id, context.thread_id)
        try:
            return validate_graph_state(
                self.compiled_graph.invoke(
                    _state_from_context(context),
                    config=_graph_config(context.thread_id, tool_context, event_sink, turn_context),
                    durability=self.durability,
                )
            )
        finally:
            self._release_turn_context(turn_context, keys)

    def resume(
        self,
        thread_id: str,
        resume_payload: dict[str, Any],
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        keys = self._bind_turn_context(turn_context, thread_id)
        try:
            return migrate_legacy_graph_state(
                self.compiled_graph.invoke(
                    Command(resume=resume_payload),
                    config=_graph_config(thread_id, tool_context, event_sink, turn_context),
                    durability=self.durability,
                ),
                thread_id=thread_id,
            )
        finally:
            self._release_turn_context(turn_context, keys)

    def recover(
        self,
        thread_id: str,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        """Continue an interrupted super-step from its durable checkpoint."""
        keys = self._bind_turn_context(turn_context, thread_id)
        try:
            return migrate_legacy_graph_state(
                self.compiled_graph.invoke(
                    None,
                    config=_graph_config(thread_id, tool_context, event_sink, turn_context),
                    durability=self.durability,
                ),
                thread_id=thread_id,
            )
        finally:
            self._release_turn_context(turn_context, keys)

    async def ainvoke(
        self,
        context: RunContext,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        if not self.supports_async_checkpointer:
            return await asyncio.to_thread(
                self.invoke,
                context,
                tool_context=tool_context,
                event_sink=event_sink,
                turn_context=turn_context,
            )
        keys = self._bind_turn_context(turn_context, context.run_id, context.thread_id)
        try:
            state = await self.compiled_graph.ainvoke(
                _state_from_context(context),
                config=_graph_config(context.thread_id, tool_context, event_sink, turn_context),
                durability=self.durability,
            )
            return validate_graph_state(state)
        finally:
            self._release_turn_context(turn_context, keys)

    async def aresume(
        self,
        thread_id: str,
        resume_payload: dict[str, Any],
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        if not self.supports_async_checkpointer:
            return await asyncio.to_thread(
                self.resume,
                thread_id,
                resume_payload,
                tool_context=tool_context,
                event_sink=event_sink,
                turn_context=turn_context,
            )
        keys = self._bind_turn_context(turn_context, thread_id)
        try:
            state = await self.compiled_graph.ainvoke(
                Command(resume=resume_payload),
                config=_graph_config(thread_id, tool_context, event_sink, turn_context),
                durability=self.durability,
            )
            return migrate_legacy_graph_state(state, thread_id=thread_id)
        finally:
            self._release_turn_context(turn_context, keys)

    async def arecover(
        self,
        thread_id: str,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
        turn_context: TurnExecutionContext | None = None,
    ) -> HarnessGraphState:
        if not self.supports_async_checkpointer:
            return await asyncio.to_thread(
                self.recover,
                thread_id,
                tool_context=tool_context,
                event_sink=event_sink,
                turn_context=turn_context,
            )
        keys = self._bind_turn_context(turn_context, thread_id)
        try:
            state = await self.compiled_graph.ainvoke(
                None,
                config=_graph_config(thread_id, tool_context, event_sink, turn_context),
                durability=self.durability,
            )
            return migrate_legacy_graph_state(state, thread_id=thread_id)
        finally:
            self._release_turn_context(turn_context, keys)


def create_harness_graph(
    *,
    model: ModelGateway | None = None,
    tool_executor: ToolExecutor,
    tool_registry: ToolRegistry,
    system_prompt: str = "",
    max_steps: int = 12,
    checkpointer=None,
    supports_async_checkpointer: bool = True,
    budget_ledger: BudgetLedger | None = None,
    deadline_monotonic: float | None = None,
    run_control: RunControl | None = None,
    durability: GraphDurability = "exit",
) -> HarnessGraph:
    prompt = system_prompt or harness_system_prompt()
    ledger = budget_ledger or getattr(model, "budget_ledger", None) or tool_executor.budget_ledger
    control = run_control or NoopRunControl()
    turn_contexts = TurnContextRegistry()

    nodes = GraphNodes(
        model=model,
        tool_executor=tool_executor,
        tool_registry=tool_registry,
        prompt=prompt,
        max_steps=max_steps,
        ledger=ledger,
        deadline_monotonic=deadline_monotonic,
        control=control,
        turn_contexts=turn_contexts,
    )
    graph = StateGraph(HarnessGraphState)

    def model_node(state: HarnessGraphState, config: RunnableConfig) -> HarnessGraphState:
        return cast(HarnessGraphState, nodes.model_node(dict(state), config))

    def tool_node(state: HarnessGraphState, config: RunnableConfig) -> HarnessGraphState:
        return cast(HarnessGraphState, nodes.tool_node(dict(state), config))

    async def async_model_node(
        state: HarnessGraphState,
        config: RunnableConfig,
    ) -> HarnessGraphState:
        return cast(HarnessGraphState, await nodes.amodel_node(dict(state), config))

    async def async_tool_node(
        state: HarnessGraphState,
        config: RunnableConfig,
    ) -> HarnessGraphState:
        return cast(HarnessGraphState, await nodes.atool_node(dict(state), config))

    def await_user_node(state: HarnessGraphState, config: RunnableConfig) -> HarnessGraphState:
        return cast(HarnessGraphState, nodes.await_user_node(dict(state), config))

    def await_async_node(state: HarnessGraphState, config: RunnableConfig) -> HarnessGraphState:
        return cast(HarnessGraphState, nodes.await_async_node(dict(state), config))

    graph.add_node("model", RunnableLambda(model_node, async_model_node))
    graph.add_node("tools", RunnableLambda(tool_node, async_tool_node))
    graph.add_node("await_user", await_user_node)
    graph.add_node("await_async", await_async_node)
    graph.add_conditional_edges(
        START,
        _route_from_start,
        {"tools": "tools", "model": "model"},
    )
    graph.add_conditional_edges(
        "model",
        _route_after_model,
        {"tools": "tools", "model": "model", "await_user": "await_user", "end": END},
    )
    graph.add_conditional_edges(
        "tools",
        _route_after_tools,
        {
            "tools": "tools",
            "model": "model",
            "await_user": "await_user",
            "await_async": "await_async",
        },
    )
    graph.add_conditional_edges(
        "await_user",
        _route_after_user_resume,
        {"tools": "tools", "model": "model"},
    )
    graph.add_conditional_edges(
        "await_async",
        _route_after_user_resume,
        {"tools": "tools", "model": "model"},
    )
    return HarnessGraph(
        compiled_graph=graph.compile(checkpointer=checkpointer),
        supports_async_checkpointer=supports_async_checkpointer,
        durability=durability,
        turn_contexts=turn_contexts,
    )
