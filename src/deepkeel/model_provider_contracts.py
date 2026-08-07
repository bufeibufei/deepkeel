from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from deepkeel.contracts import AgentMessage
from deepkeel.model_invocations import ModelInvocation, ModelProviderInfo, ModelTurn
from deepkeel.model_routing import ModelStepContext


ModelRouteSink = Callable[[dict[str, Any]], None]


@runtime_checkable
class ModelProviderAdapter(Protocol):
    """Explicit provider boundary used by routed model execution."""

    @property
    def info(self) -> ModelProviderInfo: ...

    def invoke(
        self,
        request: ModelInvocation,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ModelTurn: ...


@runtime_checkable
class AsyncModelProviderAdapter(Protocol):
    """Native async provider boundary used without a worker-thread bridge."""

    @property
    def info(self) -> ModelProviderInfo: ...

    async def ainvoke(
        self,
        request: ModelInvocation,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ModelTurn: ...


class ModelGateway(Protocol):
    """Provider-neutral model execution port used by the graph."""

    def run_turn(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        on_text_delta: Callable[[str], None] | None = None,
        step_context: ModelStepContext | None = None,
        on_route: ModelRouteSink | None = None,
    ) -> ModelTurn: ...

    async def arun_turn(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        on_text_delta: Callable[[str], None] | None = None,
        step_context: ModelStepContext | None = None,
        on_route: ModelRouteSink | None = None,
    ) -> ModelTurn: ...
