from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from harness_core.budget import BudgetLedger
from harness_core.contracts import RunContext
from harness_core.control import NoopRunControl, RunControl
from harness_core.graph_nodes import GraphNodes
from harness_core.graph_state import (
    HarnessGraphState,
    _state_from_context,
    migrate_legacy_graph_state,
    validate_graph_state,
)
from harness_core.graph_workflow import (
    _graph_config,
    _route_after_model,
    _route_after_tools,
    _route_after_user_resume,
    _route_from_start,
)
from harness_core.model import ModelGateway
from harness_core.prompts import harness_system_prompt
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutionContext, ToolExecutor


EventSink = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class HarnessGraph:
    compiled_graph: Any

    def invoke(
        self,
        context: RunContext,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> HarnessGraphState:
        return validate_graph_state(self.compiled_graph.invoke(
            _state_from_context(context),
            config=_graph_config(context.thread_id, tool_context, event_sink),
        ))

    def resume(
        self,
        thread_id: str,
        resume_payload: dict[str, Any],
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> HarnessGraphState:
        return migrate_legacy_graph_state(self.compiled_graph.invoke(
            Command(resume=resume_payload),
            config=_graph_config(thread_id, tool_context, event_sink),
        ), thread_id=thread_id)

    def recover(
        self,
        thread_id: str,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> HarnessGraphState:
        """Continue an interrupted super-step from its durable checkpoint."""
        return migrate_legacy_graph_state(self.compiled_graph.invoke(
            None,
            config=_graph_config(thread_id, tool_context, event_sink),
        ), thread_id=thread_id)

    async def ainvoke(
        self,
        context: RunContext,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> HarnessGraphState:
        state = await self.compiled_graph.ainvoke(
            _state_from_context(context),
            config=_graph_config(context.thread_id, tool_context, event_sink),
        )
        return validate_graph_state(state)

    async def aresume(
        self,
        thread_id: str,
        resume_payload: dict[str, Any],
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> HarnessGraphState:
        state = await self.compiled_graph.ainvoke(
            Command(resume=resume_payload),
            config=_graph_config(thread_id, tool_context, event_sink),
        )
        return migrate_legacy_graph_state(state, thread_id=thread_id)

    async def arecover(
        self,
        thread_id: str,
        *,
        tool_context: ToolExecutionContext,
        event_sink: EventSink | None = None,
    ) -> HarnessGraphState:
        state = await self.compiled_graph.ainvoke(
            None,
            config=_graph_config(thread_id, tool_context, event_sink),
        )
        return migrate_legacy_graph_state(state, thread_id=thread_id)


def create_harness_graph(
    *,
    model: ModelGateway,
    tool_executor: ToolExecutor,
    tool_registry: ToolRegistry,
    system_prompt: str = "",
    max_steps: int = 12,
    checkpointer=None,
    budget_ledger: BudgetLedger | None = None,
    deadline_monotonic: float | None = None,
    run_control: RunControl | None = None,
) -> HarnessGraph:
    prompt = system_prompt or harness_system_prompt()
    ledger = budget_ledger or getattr(model, "budget_ledger", None) or tool_executor.budget_ledger
    control = run_control or NoopRunControl()

    nodes = GraphNodes(
        model=model,
        tool_executor=tool_executor,
        tool_registry=tool_registry,
        prompt=prompt,
        max_steps=max_steps,
        ledger=ledger,
        deadline_monotonic=deadline_monotonic,
        control=control,
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
    graph.add_edge("await_async", "model")
    return HarnessGraph(compiled_graph=graph.compile(checkpointer=checkpointer))
