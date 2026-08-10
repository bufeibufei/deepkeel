from __future__ import annotations

import time
from typing import Any

from deepkeel.failures import RunCanceledError
from deepkeel.model_capabilities import InMemoryModelCapabilityRegistry
from deepkeel.skills import DelegationPolicy
from deepkeel.subagents.batch_execution import SubAgentBatchExecutionMixin
from deepkeel.subagents.bounded_execution import SubAgentBoundedExecutionMixin
from deepkeel.subagents.contracts import (
    SUBAGENT_EVENT_SCHEMA_VERSION,
    DelegationRequest,
    DelegationTask,
    SubAgentResult,
    SubAgentSpec,
)
from deepkeel.subagents.execution_types import (
    DelegationPreflightError,
    EventSink,
    SubAgentCanceledError,
    _DelegationQuota,
)
from deepkeel.subagents.output_validation import _validate_input
from deepkeel.subagents.registry import SubAgentRegistry
from deepkeel.subagents.store import SubAgentRunStore
from deepkeel.subagents.task_execution import SubAgentTaskExecution
from deepkeel.tools import ToolExecutionContext, ToolExecutor


class SubAgentExecutor(SubAgentBatchExecutionMixin, SubAgentBoundedExecutionMixin):
    """Runs bounded specialists while the parent Agent retains final-answer ownership."""

    def __init__(
        self,
        registry: SubAgentRegistry,
        *,
        run_store: SubAgentRunStore | None = None,
        tool_executor: ToolExecutor | None = None,
        max_parallel: int = 3,
        max_depth: int = 1,
        model_capabilities: InMemoryModelCapabilityRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.run_store = run_store
        self.tool_executor = tool_executor
        self.max_parallel = max(1, min(int(max_parallel), 3))
        self.max_depth = max(1, min(int(max_depth), 1))
        self.model_capabilities = model_capabilities or InMemoryModelCapabilityRegistry()

    def _preflight_request(self, request: DelegationRequest) -> None:
        issues: list[dict[str, str]] = []
        seen_task_ids: set[str] = set()
        for task in request.tasks:
            if task.id in seen_task_ids:
                issues.append(
                    {
                        "task_id": task.id,
                        "agent_id": task.agent_id,
                        "detail": "duplicate task id",
                    }
                )
                continue
            seen_task_ids.add(task.id)
            try:
                spec = self.registry.get(task.agent_id)
            except KeyError:
                issues.append(
                    {
                        "task_id": task.id,
                        "agent_id": task.agent_id,
                        "detail": "subagent is not registered",
                    }
                )
                continue
            try:
                _validate_input(task.input_data, spec.input_contract)
            except RuntimeError as exc:
                issues.append(
                    {
                        "task_id": task.id,
                        "agent_id": task.agent_id,
                        "detail": str(exc),
                    }
                )
        if issues:
            raise DelegationPreflightError(issues)

    def _execute_one(
        self,
        task: DelegationTask,
        *,
        spec: SubAgentSpec,
        child_run_id: str,
        providers: dict[str, Any],
        root_run_id: str,
        parent_run_id: str,
        context: ToolExecutionContext,
        event_sink: EventSink | None = None,
        budget_ledger: Any = None,
        model_call_limit: float | None = None,
        quota: _DelegationQuota | None = None,
    ) -> SubAgentResult:
        return SubAgentTaskExecution(
            self,
            task,
            spec=spec,
            child_run_id=child_run_id,
            providers=providers,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            context=context,
            event_sink=event_sink,
            budget_ledger=budget_ledger,
            model_call_limit=model_call_limit,
            quota=quota,
        ).run()

    def _create_child_run(
        self,
        child_run_id: str,
        *,
        root_run_id: str,
        parent_run_id: str,
        delegation_id: str,
        task: DelegationTask,
        spec: SubAgentSpec,
        context: ToolExecutionContext,
    ) -> None:
        if self.run_store is None:
            return
        self.run_store.create_child(
            child_run_id=child_run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            delegation_id=delegation_id,
            task=task,
            spec=spec,
            user_id=context.user_id,
            thread_id=context.thread_id,
        )

    def _settle_child_run(self, result: SubAgentResult) -> None:
        if self.run_store is None:
            return
        if result.status == "needs_input":
            suspend = getattr(self.run_store, "suspend_child", None)
            if callable(suspend):
                suspend(result)
                return
            self.run_store.checkpoint_child(
                result.child_run_id,
                phase="needs_input",
                state={
                    "schema_version": "subagent-execution-v1",
                    "task_id": result.task_id,
                    "idempotency_key": result.idempotency_key,
                    "spec_version": str(result.metadata.get("spec_version") or ""),
                    "lineage": result.lineage.model_dump(mode="json"),
                    "phase": "needs_input",
                    "result": result.model_dump(mode="json"),
                },
            )
            return
        self.run_store.settle_child(result)

    def _parent_accepts_results(self, parent_run_id: str) -> bool:
        if self.run_store is None or not parent_run_id:
            return True
        return self.run_store.parent_accepts_results(parent_run_id)

    def _load_child_result(self, child_run_id: str) -> SubAgentResult | None:
        if self.run_store is None:
            return None
        return self.run_store.load_child_result(child_run_id)

    def _load_child_checkpoint(self, child_run_id: str) -> dict[str, Any] | None:
        if self.run_store is None:
            return None
        return self.run_store.load_child_checkpoint(child_run_id)

    def _checkpoint_child(
        self,
        child_run_id: str,
        *,
        phase: str,
        state: dict[str, Any],
    ) -> None:
        if self.run_store is None:
            return
        self.run_store.checkpoint_child(child_run_id, phase=phase, state=state)

    def _cancel_requested(self, child_run_id: str, parent_run_id: str) -> bool:
        if self.run_store is None:
            return not self._parent_accepts_results(parent_run_id)
        return self.run_store.cancel_requested(child_run_id, parent_run_id)

    def _raise_if_canceled(
        self,
        child_run_id: str,
        parent_run_id: str,
        *,
        context: ToolExecutionContext,
        task: DelegationTask | None = None,
    ) -> None:
        try:
            if task is None or task.cancellation.propagate_parent:
                context.run_control.raise_if_cancelled(parent_run_id or context.run_id)
            if child_run_id and child_run_id != parent_run_id:
                context.run_control.raise_if_cancelled(child_run_id)
        except RunCanceledError as exc:
            raise SubAgentCanceledError("subagent execution was canceled by run control") from exc
        if (task is None or task.cancellation.propagate_parent) and self._cancel_requested(
            child_run_id, parent_run_id
        ):
            raise SubAgentCanceledError("subagent execution was canceled by its parent run")

    @staticmethod
    def _task_timeout_seconds(task: DelegationTask, spec: SubAgentSpec) -> int:
        values = [float(spec.timeout_seconds)]
        if task.timeout_seconds is not None:
            values.append(float(task.timeout_seconds))
        if task.budget.max_elapsed_seconds is not None:
            values.append(float(task.budget.max_elapsed_seconds))
        return max(1, int(min(values)))

    @staticmethod
    def _task_max_tokens(task: DelegationTask, spec: SubAgentSpec) -> int:
        values = [int(spec.max_tokens)]
        if task.budget.max_output_tokens is not None:
            values.append(int(task.budget.max_output_tokens))
        return max(128, min(values))

    @classmethod
    def _task_deadline_monotonic(
        cls,
        context: ToolExecutionContext,
        task: DelegationTask,
        spec: SubAgentSpec,
    ) -> float:
        local_deadline = time.monotonic() + cls._task_timeout_seconds(task, spec)
        if context.deadline_monotonic is None:
            return local_deadline
        return min(float(context.deadline_monotonic), local_deadline)

    @staticmethod
    def _validate_delegation_policy(
        request: DelegationRequest,
        policy: DelegationPolicy,
    ) -> None:
        if not policy.enabled:
            raise ValueError("delegation is disabled by the active Skill policy")
        if len(request.tasks) > policy.max_tasks:
            raise ValueError(f"delegation task count exceeds Skill limit: {policy.max_tasks}")
        if request.max_concurrency > policy.max_concurrency:
            raise ValueError(
                f"delegation concurrency exceeds Skill limit: {policy.max_concurrency}"
            )
        denied = sorted(
            {task.agent_id for task in request.tasks if not policy.allows_agent(task.agent_id)}
        )
        if denied:
            raise ValueError(
                "delegated agents are not allowed by the active Skill: " + ", ".join(denied)
            )

    @staticmethod
    def _emit(
        sink: EventSink | None,
        event_type: str,
        title: str,
        summary: str,
        request: DelegationRequest,
        *,
        visible: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if sink is None:
            return
        sink(
            {
                "event_type": event_type,
                "title": title,
                "summary": summary,
                "payload": {
                    "schema_version": SUBAGENT_EVENT_SCHEMA_VERSION,
                    "visible": visible,
                    "delegation_id": request.delegation_id,
                    "root_run_id": request.root_run_id,
                    "parent_run_id": request.parent_run_id,
                    "task_count": len(request.tasks),
                    **(extra or {}),
                },
            }
        )

    @staticmethod
    def _emit_task(
        sink: EventSink | None,
        event_type: str,
        request: DelegationRequest,
        task: DelegationTask,
        spec: SubAgentSpec,
        child_run_id: str,
        *,
        status: str,
        result: SubAgentResult | None = None,
    ) -> None:
        if sink is None:
            return
        sink(
            {
                "event_type": event_type,
                "title": spec.label,
                "summary": result.conclusion if result and result.conclusion else task.objective,
                "payload": {
                    "schema_version": SUBAGENT_EVENT_SCHEMA_VERSION,
                    "visible": True,
                    "delegation_id": request.delegation_id,
                    "task_id": task.id,
                    "agent_id": spec.id,
                    "agent_label": spec.label,
                    "child_run_id": child_run_id,
                    "parent_run_id": request.parent_run_id,
                    "root_run_id": request.root_run_id,
                    "parent_task_id": task.lineage.parent_task_id,
                    "idempotency_key": task.effective_idempotency_key,
                    "cancellation_key": (
                        task.cancellation.cancellation_key
                        or spec.cancellation_policy.cancellation_key
                    ),
                    "spec_version": spec.version,
                    "status": status,
                    "model_role": result.model_role if result else task.model_role,
                    "model_id": result.model_id if result else "",
                    "duration_ms": result.duration_ms if result else 0,
                    "timeout_seconds": SubAgentExecutor._task_timeout_seconds(task, spec),
                    "budget": task.budget.model_dump(mode="json"),
                    "artifact_refs": (
                        [item.model_dump(mode="json") for item in result.artifact_refs]
                        if result
                        else [item.model_dump(mode="json") for item in task.artifact_refs]
                    ),
                    "needs_input": bool(result and result.status == "needs_input"),
                    "input_request": (
                        result.input_request.model_dump(mode="json")
                        if result and result.input_request
                        else None
                    ),
                    "error": result.error if result else "",
                },
            }
        )
