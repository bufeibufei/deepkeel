from __future__ import annotations

import asyncio
from typing import Any

from deepkeel.subagents.batch_coordinator import SubAgentBatchCoordinator
from deepkeel.subagents.contracts import (
    DelegationBatchResult,
    DelegationRequest,
)
from deepkeel.subagents.execution_types import EventSink
from deepkeel.tools import ToolExecutionContext


class SubAgentBatchExecutionMixin:
    """Parallel child-run scheduling around the bounded single-task executor."""

    async def aexecute_many(
        self: Any,
        request: DelegationRequest,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: EventSink | None = None,
    ) -> DelegationBatchResult:
        if context.session is not None and context.session_factory is None:
            raise RuntimeError(
                "async subagent execution requires session_factory when a session is bound"
            )
        thread_context = context.fork(session=None) if context.session is not None else context
        return await asyncio.to_thread(
            self.execute_many,
            request,
            context=thread_context,
            providers=providers,
            event_sink=event_sink,
        )

    def execute_many(
        self: Any,
        request: DelegationRequest,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: EventSink | None = None,
    ) -> DelegationBatchResult:
        return SubAgentBatchCoordinator(
            self,
            request,
            context=context,
            providers=providers,
            event_sink=event_sink,
        ).run()
