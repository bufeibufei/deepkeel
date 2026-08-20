from __future__ import annotations

from typing import Any

from deepkeel.leases import ExecutionFence
from deepkeel.runtime_api import RuntimeRequest, RuntimeResult
from deepkeel.runtime_execution_support import EventSink
from deepkeel.runtime_failure_handling import RuntimeFailureHandlingMixin
from deepkeel.runtime_turn_coordinator import RuntimeTurnCoordinator


class RuntimeTurnExecutionMixin(RuntimeFailureHandlingMixin):
    """Compatibility facade for one staged, claimed runtime turn."""

    async def _arun_claimed(
        self: Any,
        request: RuntimeRequest,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
        event_sink: EventSink | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> RuntimeResult:
        return await RuntimeTurnCoordinator(
            self,
            request,
            provider=provider,
            providers=providers,
            session=session,
            event_sink=event_sink,
            execution_fence=execution_fence,
        ).run()
