from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        skill_snapshot = context.metadata.get("skill_activation")
        skill_snapshot = skill_snapshot if isinstance(skill_snapshot, dict) else {}
        delegation_policy = DelegationPolicy.from_snapshot(skill_snapshot)
        if isinstance(skill_snapshot.get("delegation_policy"), dict):
            self._validate_delegation_policy(request, delegation_policy)
        if request.depth > self.max_depth:
            raise ValueError(f"subagent depth exceeds limit: {self.max_depth}")
        if len(request.tasks) > self.max_parallel:
            raise ValueError(f"subagent task count exceeds limit: {self.max_parallel}")
        try:
            self._preflight_request(request)
        except DelegationPreflightError as exc:
            self._emit(
                event_sink,
                "subagent.batch.rejected",
                "Delegation rejected",
                "The specialist task contract is invalid; no child run was started.",
                request,
                visible=False,
                extra={
                    "error_code": exc.code,
                    "issues": [dict(item) for item in exc.issues],
                },
            )
            raise
        started_at = time.perf_counter()
        root_run_id = request.root_run_id or context.run_id
        parent_run_id = request.parent_run_id or context.run_id
        if not self._parent_accepts_results(parent_run_id):
            raise RuntimeError("parent agent run is already terminal")
        child_ids = {
            task.id: _child_run_id(parent_run_id, request.delegation_id, task)
            for task in request.tasks
        }
        request = request.model_copy(
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
        self._emit(
            event_sink,
            "subagent.batch.started",
            "Parallel specialist analysis",
            f"Dispatched {len(request.tasks)} bounded specialist tasks.",
            request,
            visible=True,
        )
        results: list[SubAgentResult] = []
        pending_tasks: list[DelegationTask] = []
        for task in request.tasks:
            spec = self.registry.get(task.agent_id)
            self._create_child_run(
                child_ids[task.id],
                root_run_id=root_run_id,
                parent_run_id=parent_run_id,
                delegation_id=request.delegation_id,
                task=task,
                spec=spec,
                context=context,
            )
            restored_result = self._load_child_result(child_ids[task.id])
            if restored_result is not None:
                results.append(restored_result)
                self._emit_task(
                    event_sink,
                    "subagent.replayed",
                    request,
                    task,
                    spec,
                    child_ids[task.id],
                    status=restored_result.status,
                    result=restored_result,
                )
                continue
            pending_tasks.append(task)
            self._emit_task(
                event_sink,
                "subagent.started",
                request,
                task,
                spec,
                child_ids[task.id],
                status="running",
            )

        worker_count = min(request.max_concurrency, self.max_parallel, len(pending_tasks))
        budget_ledger = context.metadata.get("budget_ledger")
        model_call_limit = _minimum_optional(
            context.budget_limits.get(MODEL_CALLS),
            delegation_policy.max_model_calls if delegation_policy.enabled else None,
        )
        restored_checkpoints = [
            self._load_child_checkpoint(child_ids[task.id]) or {} for task in pending_tasks
        ]
        restored_model_calls = sum(
            int((result.metadata or {}).get("model_calls") or 0) for result in results
        ) + sum(int(state.get("model_calls") or 0) for state in restored_checkpoints)
        restored_tool_calls = sum(
            len((result.metadata or {}).get("tool_trace") or []) for result in results
        ) + sum(int(state.get("tool_calls") or 0) for state in restored_checkpoints)
        quota = _DelegationQuota(
            max_model_calls=(
                delegation_policy.max_model_calls if delegation_policy.enabled else None
            ),
            max_tool_calls=(
                delegation_policy.max_tool_calls if delegation_policy.enabled else None
            ),
            model_calls=restored_model_calls,
            tool_calls=restored_tool_calls,
        )
        if pending_tasks:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="harness-subagent",
            ) as pool:
                futures = {
                    pool.submit(
                        self._execute_one,
                        task,
                        spec=self.registry.get(task.agent_id),
                        child_run_id=child_ids[task.id],
                        providers=providers,
                        root_run_id=root_run_id,
                        parent_run_id=parent_run_id,
                        context=context,
                        event_sink=event_sink,
                        budget_ledger=budget_ledger,
                        model_call_limit=model_call_limit,
                        quota=quota,
                    ): task
                    for task in pending_tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    spec = self.registry.get(task.agent_id)
                    try:
                        result = future.result()
                    except SubAgentCanceledError as exc:
                        result = SubAgentResult(
                            task_id=task.id,
                            agent_id=task.agent_id,
                            child_run_id=child_ids[task.id],
                            status="canceled",
                            idempotency_key=task.effective_idempotency_key,
                            lineage=task.lineage,
                            error=str(exc),
                        )
                    except Exception as exc:
                        result = SubAgentResult(
                            task_id=task.id,
                            agent_id=task.agent_id,
                            child_run_id=child_ids[task.id],
                            status="failed",
                            idempotency_key=task.effective_idempotency_key,
                            lineage=task.lineage,
                            error=str(exc),
                            raw_text=str(getattr(exc, "raw_text", "") or ""),
                            metadata={
                                "diagnostics": dict(
                                    getattr(exc, "diagnostics", {}) or {}
                                ),
                            },
                        )
                    if (
                        result.status == "completed"
                        and task.cancellation.discard_late_result
                        and not self._parent_accepts_results(parent_run_id)
                    ):
                        result = result.model_copy(
                            update={
                                "status": "canceled",
                                "conclusion": "",
                                "evidence": [],
                                "evidence_refs": [],
                                "risks": [],
                                "recommendations": [],
                                "claims": [],
                                "warnings": [
                                    "The parent run ended; the late result was not merged."
                                ],
                                "output": {},
                                "error": "parent agent run became terminal",
                                "metadata": {
                                    **result.metadata,
                                    "late_result_discarded": True,
                                },
                            }
                        )
                    results.append(result)
                    self._settle_child_run(result)
                    self._emit_task(
                        event_sink,
                        {
                            "completed": "subagent.completed",
                            "canceled": "subagent.canceled",
                            "needs_input": "subagent.needs_input",
                        }.get(result.status, "subagent.failed"),
                        request,
                        task,
                        spec,
                        result.child_run_id,
                        status=result.status,
                        result=result,
                    )

        ordered = sorted(
            results,
            key=lambda item: next(
                index
                for index, task in enumerate(request.tasks)
                if task.id == item.task_id
            ),
        )
        completed = sum(result.status == "completed" for result in ordered)
        failed = sum(result.status == "failed" for result in ordered)
        canceled = sum(result.status == "canceled" for result in ordered)
        needs_input = sum(result.status == "needs_input" for result in ordered)
        degraded = sum(result.outcome == "degraded" for result in ordered)
        fail_batch = any(
            result.status in {"failed", "canceled"}
            and self.registry.get(result.agent_id).failure_policy == "fail_batch"
            for result in ordered
        )
        status: Literal["completed", "partial", "failed", "canceled", "needs_input"] = (
            "failed"
            if fail_batch
            else "needs_input"
            if needs_input
            else "canceled"
            if canceled == len(ordered)
            else "completed"
            if completed == len(ordered)
            else "partial"
            if completed
            else "failed"
        )
        batch = DelegationBatchResult(
            delegation_id=request.delegation_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            status=status,
            results=ordered,
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        self._emit(
            event_sink,
            "subagent.batch.completed",
            "Specialist results synthesized",
            (
                f"{completed}/{len(ordered)} specialist results returned to the lead agent."
                + (
                    f"Of these, {degraded} used structured degraded results."
                    if degraded
                    else ""
                )
            ),
            request,
            visible=True,
            extra={
                "status": status,
                "duration_ms": batch.duration_ms,
                "completed_count": completed,
                "degraded_count": degraded,
                "failed_count": failed,
                "canceled_count": canceled,
                "needs_input_count": needs_input,
            },
        )
        return batch
