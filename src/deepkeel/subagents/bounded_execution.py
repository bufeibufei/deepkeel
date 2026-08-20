from __future__ import annotations

from typing import Any

from deepkeel.subagents.bounded_coordinator import BoundedAgentCoordinator
from deepkeel.subagents.contracts import DelegationTask, SubAgentSpec
from deepkeel.subagents.execution_types import EventSink, _DelegationQuota
from deepkeel.tools import ToolExecutionContext


class SubAgentBoundedExecutionMixin:
    """Compatibility facade for the staged bounded specialist execution."""

    def _run_bounded_agent(
        self: Any,
        task: DelegationTask,
        *,
        spec: SubAgentSpec,
        provider: Any,
        child_run_id: str,
        context: ToolExecutionContext,
        event_sink: EventSink | None,
        system_prompt: str,
        prompt: str,
        output_schema: dict[str, Any],
        root_run_id: str,
        budget_ledger: Any,
        model_call_limit: float | None,
        parent_run_id: str,
        resume_state: dict[str, Any] | None,
        quota: _DelegationQuota | None,
        task_quota: _DelegationQuota | None,
        deadline_monotonic: float | None,
    ) -> tuple[str, list[dict[str, Any]], int, dict[str, Any]]:
        return BoundedAgentCoordinator(
            self,
            task,
            spec=spec,
            provider=provider,
            child_run_id=child_run_id,
            context=context,
            event_sink=event_sink,
            system_prompt=system_prompt,
            prompt=prompt,
            output_schema=output_schema,
            root_run_id=root_run_id,
            budget_ledger=budget_ledger,
            model_call_limit=model_call_limit,
            parent_run_id=parent_run_id,
            resume_state=resume_state,
            quota=quota,
            task_quota=task_quota,
            deadline_monotonic=deadline_monotonic,
        ).run()
