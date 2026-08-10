from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

from deepkeel.budget import MODEL_CALLS
from deepkeel.skills import DelegationPolicy
from deepkeel.subagents.contracts import (
    DelegationBatchResult,
    DelegationRequest,
    DelegationTask,
    SubAgentResult,
)
from deepkeel.subagents.execution_support import _child_run_id, _minimum_optional
from deepkeel.subagents.execution_types import (
    DelegationPreflightError,
    EventSink,
    SubAgentCanceledError,
    _DelegationQuota,
)
from deepkeel.tools import ToolExecutionContext


@dataclass(slots=True)
class _BatchState:
    request: DelegationRequest
    context: ToolExecutionContext
    providers: dict[str, Any]
    event_sink: EventSink | None
    policy: DelegationPolicy
    root_run_id: str
    parent_run_id: str
    child_ids: dict[str, str]
    started_at: float = field(default_factory=time.perf_counter)
    results: list[SubAgentResult] = field(default_factory=list)
    pending: list[DelegationTask] = field(default_factory=list)


class SubAgentBatchCoordinator:
    """Typed lifecycle coordinator for one bounded delegation batch."""

    def __init__(
        self,
        owner: Any,
        request: DelegationRequest,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: EventSink | None,
    ) -> None:
        self.owner = owner
        self.state = self._prepare_state(
            request,
            context=context,
            providers=providers,
            event_sink=event_sink,
        )

    def run(self) -> DelegationBatchResult:
        self._emit_batch_started()
        self._restore_or_schedule_children()
        if self.state.pending:
            self._execute_pending()
        batch = self._build_result()
        self._emit_batch_completed(batch)
        return batch

    def _prepare_state(
        self,
        request: DelegationRequest,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: EventSink | None,
    ) -> _BatchState:
        snapshot = context.metadata.get("skill_activation")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        policy = DelegationPolicy.from_snapshot(snapshot)
        if isinstance(snapshot.get("delegation_policy"), dict):
            self.owner._validate_delegation_policy(request, policy)
        if request.depth > self.owner.max_depth:
            raise ValueError(f"subagent depth exceeds limit: {self.owner.max_depth}")
        if len(request.tasks) > self.owner.max_parallel:
            raise ValueError(f"subagent task count exceeds limit: {self.owner.max_parallel}")
        self._preflight(request, event_sink)
        root_run_id = request.root_run_id or context.run_id
        parent_run_id = request.parent_run_id or context.run_id
        if not self.owner._parent_accepts_results(parent_run_id):
            raise RuntimeError("parent agent run is already terminal")
        child_ids = {
            task.id: _child_run_id(parent_run_id, request.delegation_id, task)
            for task in request.tasks
        }
        bound_request = request.model_copy(
            update={
                "root_run_id": root_run_id,
                "parent_run_id": parent_run_id,
                "tasks": [
                    task.bind_lineage(
                        root_run_id=root_run_id,
                        parent_run_id=parent_run_id,
                        delegation_id=request.delegation_id,
                        depth=request.depth,
                        child_run_id=child_ids[task.id],
                    )
                    for task in request.tasks
                ],
            }
        )
        return _BatchState(
            request=bound_request,
            context=context,
            providers=providers,
            event_sink=event_sink,
            policy=policy,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            child_ids=child_ids,
        )

    def _preflight(
        self,
        request: DelegationRequest,
        event_sink: EventSink | None,
    ) -> None:
        try:
            self.owner._preflight_request(request)
        except DelegationPreflightError as exc:
            self.owner._emit(
                event_sink,
                "subagent.batch.rejected",
                "Delegation rejected",
                "The specialist task contract is invalid; no child run was started.",
                request,
                visible=False,
                extra={"error_code": exc.code, "issues": [dict(item) for item in exc.issues]},
            )
            raise

    def _emit_batch_started(self) -> None:
        request = self.state.request
        self.owner._emit(
            self.state.event_sink,
            "subagent.batch.started",
            "Parallel specialist analysis",
            f"Dispatched {len(request.tasks)} bounded specialist tasks.",
            request,
            visible=True,
        )

    def _restore_or_schedule_children(self) -> None:
        state = self.state
        for task in state.request.tasks:
            spec = self.owner.registry.get(task.agent_id)
            child_id = state.child_ids[task.id]
            self.owner._create_child_run(
                child_id,
                root_run_id=state.root_run_id,
                parent_run_id=state.parent_run_id,
                delegation_id=state.request.delegation_id,
                task=task,
                spec=spec,
                context=state.context,
            )
            restored = self.owner._load_child_result(child_id)
            if restored is None:
                state.pending.append(task)
                self._emit_task("subagent.started", task, status="running")
                continue
            state.results.append(restored)
            self._emit_task(
                "subagent.replayed",
                task,
                status=restored.status,
                result=restored,
            )

    def _execute_pending(self) -> None:
        state = self.state
        quota = self._restored_quota()
        model_call_limit = _minimum_optional(
            state.context.budget_limits.get(MODEL_CALLS),
            state.policy.max_model_calls if state.policy.enabled else None,
        )
        workers = min(
            state.request.max_concurrency,
            self.owner.max_parallel,
            len(state.pending),
        )
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="harness-subagent",
        ) as pool:
            futures = {
                pool.submit(
                    self.owner._execute_one,
                    task,
                    spec=self.owner.registry.get(task.agent_id),
                    child_run_id=state.child_ids[task.id],
                    providers=state.providers,
                    root_run_id=state.root_run_id,
                    parent_run_id=state.parent_run_id,
                    context=state.context,
                    event_sink=state.event_sink,
                    budget_ledger=state.context.metadata.get("budget_ledger"),
                    model_call_limit=model_call_limit,
                    quota=quota,
                ): task
                for task in state.pending
            }
            self._collect_futures(futures)

    def _restored_quota(self) -> _DelegationQuota:
        state = self.state
        checkpoints = [
            self.owner._load_child_checkpoint(state.child_ids[task.id]) or {}
            for task in state.pending
        ]
        model_calls = sum(
            int((result.metadata or {}).get("model_calls") or 0)
            for result in state.results
        ) + sum(int(checkpoint.get("model_calls") or 0) for checkpoint in checkpoints)
        tool_calls = sum(
            len((result.metadata or {}).get("tool_trace") or [])
            for result in state.results
        ) + sum(int(checkpoint.get("tool_calls") or 0) for checkpoint in checkpoints)
        return _DelegationQuota(
            max_model_calls=(state.policy.max_model_calls if state.policy.enabled else None),
            max_tool_calls=(state.policy.max_tool_calls if state.policy.enabled else None),
            model_calls=model_calls,
            tool_calls=tool_calls,
        )

    def _collect_futures(
        self,
        futures: dict[Future[SubAgentResult], DelegationTask],
    ) -> None:
        for future in as_completed(futures):
            task = futures[future]
            result = self._future_result(future, task)
            result = self._discard_late_result(task, result)
            self.state.results.append(result)
            self.owner._settle_child_run(result)
            self._emit_task(
                {
                    "completed": "subagent.completed",
                    "canceled": "subagent.canceled",
                    "needs_input": "subagent.needs_input",
                }.get(result.status, "subagent.failed"),
                task,
                status=result.status,
                result=result,
            )

    def _future_result(
        self,
        future: Future[SubAgentResult],
        task: DelegationTask,
    ) -> SubAgentResult:
        try:
            return future.result()
        except SubAgentCanceledError as exc:
            return self._failed_result(task, status="canceled", exc=exc)
        except Exception as exc:
            return self._failed_result(task, status="failed", exc=exc)

    def _failed_result(
        self,
        task: DelegationTask,
        *,
        status: Literal["failed", "canceled"],
        exc: Exception,
    ) -> SubAgentResult:
        return SubAgentResult(
            task_id=task.id,
            agent_id=task.agent_id,
            child_run_id=self.state.child_ids[task.id],
            status=status,
            idempotency_key=task.effective_idempotency_key,
            lineage=task.lineage,
            error=str(exc),
            raw_text=str(getattr(exc, "raw_text", "") or "") if status == "failed" else "",
            metadata={
                "diagnostics": dict(getattr(exc, "diagnostics", {}) or {})
            }
            if status == "failed"
            else {},
        )

    def _discard_late_result(
        self,
        task: DelegationTask,
        result: SubAgentResult,
    ) -> SubAgentResult:
        if (
            result.status != "completed"
            or not task.cancellation.discard_late_result
            or self.owner._parent_accepts_results(self.state.parent_run_id)
        ):
            return result
        return result.model_copy(
            update={
                "status": "canceled",
                "conclusion": "",
                "evidence": [],
                "evidence_refs": [],
                "risks": [],
                "recommendations": [],
                "claims": [],
                "warnings": ["The parent run ended; the late result was not merged."],
                "output": {},
                "error": "parent agent run became terminal",
                "metadata": {**result.metadata, "late_result_discarded": True},
            }
        )

    def _build_result(self) -> DelegationBatchResult:
        state = self.state
        ordered = sorted(
            state.results,
            key=lambda result: next(
                index
                for index, task in enumerate(state.request.tasks)
                if task.id == result.task_id
            ),
        )
        counts = _result_counts(ordered)
        fail_batch = any(
            result.status in {"failed", "canceled"}
            and self.owner.registry.get(result.agent_id).failure_policy == "fail_batch"
            for result in ordered
        )
        status = _batch_status(ordered, counts, fail_batch=fail_batch)
        return DelegationBatchResult(
            delegation_id=state.request.delegation_id,
            root_run_id=state.root_run_id,
            parent_run_id=state.parent_run_id,
            status=status,
            results=ordered,
            duration_ms=round((time.perf_counter() - state.started_at) * 1000),
        )

    def _emit_batch_completed(self, batch: DelegationBatchResult) -> None:
        counts = _result_counts(batch.results)
        degraded = sum(result.outcome == "degraded" for result in batch.results)
        self.owner._emit(
            self.state.event_sink,
            "subagent.batch.completed",
            "Specialist results synthesized",
            (
                f"{counts['completed']}/{len(batch.results)} specialist results returned "
                "to the lead agent."
                + (
                    f"Of these, {degraded} used structured degraded results."
                    if degraded
                    else ""
                )
            ),
            self.state.request,
            visible=True,
            extra={
                "status": batch.status,
                "duration_ms": batch.duration_ms,
                "completed_count": counts["completed"],
                "degraded_count": degraded,
                "failed_count": counts["failed"],
                "canceled_count": counts["canceled"],
                "needs_input_count": counts["needs_input"],
            },
        )

    def _emit_task(
        self,
        event_type: str,
        task: DelegationTask,
        *,
        status: str,
        result: SubAgentResult | None = None,
    ) -> None:
        self.owner._emit_task(
            self.state.event_sink,
            event_type,
            self.state.request,
            task,
            self.owner.registry.get(task.agent_id),
            self.state.child_ids[task.id],
            status=status,
            result=result,
        )


def _result_counts(results: list[SubAgentResult]) -> dict[str, int]:
    return {
        status: sum(result.status == status for result in results)
        for status in ("completed", "failed", "canceled", "needs_input")
    }


def _batch_status(
    results: list[SubAgentResult],
    counts: dict[str, int],
    *,
    fail_batch: bool,
) -> Literal["completed", "partial", "failed", "canceled", "needs_input"]:
    if fail_batch:
        return "failed"
    if counts["needs_input"]:
        return "needs_input"
    if counts["canceled"] == len(results):
        return "canceled"
    if counts["completed"] == len(results):
        return "completed"
    if counts["completed"]:
        return "partial"
    return "failed"
