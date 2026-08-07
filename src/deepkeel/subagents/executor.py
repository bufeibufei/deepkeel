from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from deepkeel.budget import MODEL_CALLS
from deepkeel.contracts import AgentMessage, ToolCall
from deepkeel.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from deepkeel.failures import RunCanceledError
from deepkeel.model import NativeChatProviderAdapter, model_tools_from_registry
from deepkeel.model_capabilities import InMemoryModelCapabilityRegistry
from deepkeel.model_routing import ModelStepContext
from deepkeel.skills import DelegationPolicy
from deepkeel.subagents.bounded_execution import SubAgentBoundedExecutionMixin
from deepkeel.subagents.contracts import (
    SUBAGENT_EVENT_SCHEMA_VERSION,
    DelegationBatchResult,
    DelegationRequest,
    DelegationTask,
    SubAgentArtifactRef,
    SubAgentContextRef,
    SubAgentInputRequest,
    SubAgentResult,
    SubAgentSpec,
)
from deepkeel.subagents.execution_types import (
    DelegationPreflightError,
    EventSink,
    SubAgentCanceledError,
    SubAgentEmptyResponseError,
    SubAgentOutputError,
    _DelegationQuota,
)
from deepkeel.subagents.execution_support import (
    _child_run_id,
    _child_tool_context,
    _consume_model_budget,
    _default_system_prompt,
    _emit_subagent_model_retry,
    _emit_subagent_tools,
    _execution_checkpoint,
    _invoke_provider,
    _is_empty_model_response_error,
    _minimum_optional,
    _repair_prompt,
    _resolve_role,
    _restored_messages,
    _task_prompt,
    _valid_resume_state,
)
from deepkeel.subagents.output_validation import (
    _confidence,
    _dict_list,
    _fallback_subagent_output,
    _output_schema,
    _string_list,
    _validate_input,
    _validated_json,
)
from deepkeel.subagents.registry import SubAgentRegistry
from deepkeel.subagents.store import SubAgentRunStore
from deepkeel.tools import ToolExecutionContext, ToolExecutor
from deepkeel.type_narrowing import as_dict


class SubAgentExecutor(SubAgentBoundedExecutionMixin):
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

    def execute_many(
        self,
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
            max_model_calls=delegation_policy.max_model_calls
            if delegation_policy.enabled
            else None,
            max_tool_calls=delegation_policy.max_tool_calls if delegation_policy.enabled else None,
            model_calls=restored_model_calls,
            tool_calls=restored_tool_calls,
        )
        if pending_tasks:
            with ThreadPoolExecutor(
                max_workers=worker_count, thread_name_prefix="harness-subagent"
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
                                "diagnostics": dict(getattr(exc, "diagnostics", {}) or {}),
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
                index for index, task in enumerate(request.tasks) if task.id == item.task_id
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
                + (f"Of these, {degraded} used structured degraded results." if degraded else "")
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
        started_at = time.perf_counter()
        try:
            self._raise_if_canceled(
                child_run_id,
                parent_run_id,
                context=context,
                task=task,
            )
        except SubAgentCanceledError:
            return SubAgentResult(
                task_id=task.id,
                agent_id=task.agent_id,
                child_run_id=child_run_id,
                status="canceled",
                idempotency_key=task.effective_idempotency_key,
                lineage=task.lineage,
                error="parent agent run is terminal",
                metadata={"late_result_discarded": True},
            )
        _validate_input(task.input_data, spec.input_contract)
        role = _resolve_role(task.model_role, spec.model_role, providers)
        provider = providers.get(role)
        if provider is None:
            raise RuntimeError(f"subagent model provider is unavailable for role {role}")
        prompt = _task_prompt(task, spec)
        system_prompt = spec.system_prompt or _default_system_prompt(spec)
        schema = _output_schema(spec)
        resume_state = self._load_child_checkpoint(child_run_id)
        child_deadline_monotonic = self._task_deadline_monotonic(
            context,
            task,
            spec,
        )
        task_quota = _DelegationQuota(
            max_model_calls=_minimum_optional(
                spec.max_model_calls,
                task.budget.max_model_calls,
            ),
            max_tool_calls=_minimum_optional(
                spec.max_tool_calls,
                task.budget.max_tool_calls,
            ),
            model_calls=int((resume_state or {}).get("model_calls") or 0),
            tool_calls=int((resume_state or {}).get("tool_calls") or 0),
        )
        effective_model_call_limit = _minimum_optional(
            model_call_limit,
            spec.max_model_calls,
            task.budget.max_model_calls,
        )
        raw, tool_trace, model_calls, structured_output = self._run_bounded_agent(
            task,
            spec=spec,
            provider=provider,
            child_run_id=child_run_id,
            context=context,
            event_sink=event_sink,
            system_prompt=system_prompt,
            prompt=prompt,
            output_schema=schema,
            root_run_id=root_run_id,
            budget_ledger=budget_ledger,
            model_call_limit=effective_model_call_limit,
            parent_run_id=parent_run_id,
            resume_state=resume_state,
            quota=quota,
            task_quota=task_quota,
            deadline_monotonic=child_deadline_monotonic,
        )
        self._raise_if_canceled(
            child_run_id,
            parent_run_id,
            context=context,
            task=task,
        )
        output_outcome: Literal["completed", "degraded"] = "completed"
        output_diagnostics: dict[str, Any] = {}
        try:
            parsed = _validated_json(raw, schema)
        except RuntimeError as first_error:
            if str((resume_state or {}).get("phase") or "") == "repair_completed":
                fallback = _fallback_subagent_output(raw)
                if fallback is None:
                    raise SubAgentOutputError(
                        "subagent returned invalid JSON after schema repair",
                        raw_text=raw,
                        diagnostics={
                            "repair_error": str(first_error),
                            "tool_trace": tool_trace,
                            "model_calls": model_calls,
                            "recovered_from_checkpoint": True,
                        },
                    ) from first_error
                parsed = fallback
                output_outcome = "degraded"
                output_diagnostics = {
                    "reason_code": "structured_output_recovered_from_text",
                    "repair_error": str(first_error),
                    "recovered_from_checkpoint": True,
                }
            else:
                self._raise_if_canceled(
                    child_run_id,
                    parent_run_id,
                    context=context,
                    task=task,
                )
                _consume_model_budget(
                    budget_ledger,
                    root_run_id=root_run_id,
                    child_run_id=child_run_id,
                    task=task,
                    model_call_limit=model_call_limit,
                    step_index=model_calls,
                    quota=quota,
                    task_quota=task_quota,
                )
                repair_prompt = _repair_prompt(prompt, raw, schema, str(first_error), tool_trace)
                repair_invocation = _invoke_provider(
                    provider,
                    system_prompt,
                    repair_prompt,
                    timeout_seconds=remaining_timeout_ceiling(
                        child_deadline_monotonic,
                        maximum=self._task_timeout_seconds(task, spec),
                    ),
                    max_tokens=self._task_max_tokens(task, spec),
                    output_schema=schema,
                    capability_registry=self.model_capabilities,
                )
                repaired = repair_invocation.text
                structured_output["repair"] = repair_invocation.diagnostics()
                self._raise_if_canceled(
                    child_run_id,
                    parent_run_id,
                    context=context,
                    task=task,
                )
                ensure_time_remaining(child_deadline_monotonic)
                model_calls += 1
                self._checkpoint_child(
                    child_run_id,
                    phase="repair_completed",
                    state={
                        "schema_version": "subagent-execution-v1",
                        "task_id": task.id,
                        "idempotency_key": task.effective_idempotency_key,
                        "lineage": task.lineage.model_dump(mode="json"),
                        "spec_version": spec.version,
                        "phase": "repair_completed",
                        "raw_text": repaired,
                        "tool_trace": tool_trace,
                        "model_calls": model_calls,
                        "tool_calls": len(tool_trace),
                        "structured_output": structured_output,
                    },
                )
                try:
                    parsed = _validated_json(repaired, schema)
                    raw = repaired
                except RuntimeError as repair_error:
                    fallback = _fallback_subagent_output(repaired or raw)
                    if fallback is None:
                        raise SubAgentOutputError(
                            "subagent returned invalid JSON after schema repair",
                            raw_text=repaired or raw,
                            diagnostics={
                                "initial_error": str(first_error),
                                "repair_error": str(repair_error),
                                "tool_trace": tool_trace,
                                "model_calls": model_calls,
                            },
                        ) from repair_error
                    parsed = fallback
                    raw = repaired or raw
                    output_outcome = "degraded"
                    output_diagnostics = {
                        "reason_code": "structured_output_recovered_from_text",
                        "initial_error": str(first_error),
                        "repair_error": str(repair_error),
                    }
        conclusion = str(parsed.get("conclusion") or parsed.get("summary") or raw).strip()
        if str(parsed.get("status") or "") == "needs_input":
            request_payload = parsed.get("input_request")
            request_payload = request_payload if isinstance(request_payload, dict) else {}
            prompt = str(request_payload.get("prompt") or conclusion).strip()
            input_request = SubAgentInputRequest.model_validate(
                {
                    **request_payload,
                    "prompt": prompt,
                }
            )
            return SubAgentResult(
                task_id=task.id,
                agent_id=task.agent_id,
                child_run_id=child_run_id,
                status="needs_input",
                outcome="needs_input",
                conclusion=conclusion,
                input_request=input_request,
                context_refs=list(task.context_refs),
                artifact_refs=list(task.artifact_refs),
                idempotency_key=task.effective_idempotency_key,
                lineage=task.lineage,
                output=parsed,
                model_role=role,
                model_id=str(getattr(provider, "model", "") or ""),
                duration_ms=round((time.perf_counter() - started_at) * 1000),
                raw_text=raw,
                metadata={
                    "spec_version": spec.version,
                    "model_calls": model_calls,
                    "tool_trace": tool_trace,
                    "structured_output": structured_output,
                },
            )
        if not conclusion:
            raise RuntimeError("subagent returned an empty conclusion")
        artifact_refs = list(task.artifact_refs)
        artifact_refs.extend(
            SubAgentArtifactRef.model_validate(item)
            for trace in tool_trace
            for item in _dict_list(trace.get("artifact_refs"))
        )
        artifact_refs.extend(
            SubAgentArtifactRef.model_validate(item)
            for item in _dict_list(parsed.get("artifact_refs"))
        )
        context_refs = list(task.context_refs)
        context_refs.extend(
            SubAgentContextRef.model_validate(item)
            for item in _dict_list(parsed.get("context_refs"))
        )
        return SubAgentResult(
            task_id=task.id,
            agent_id=task.agent_id,
            child_run_id=child_run_id,
            status="completed",
            outcome=output_outcome,
            conclusion=conclusion,
            evidence=_string_list(parsed.get("evidence")),
            evidence_refs=_dict_list(parsed.get("evidence_refs")),
            context_refs=context_refs,
            artifact_refs=artifact_refs,
            risks=_string_list(parsed.get("risks")),
            recommendations=_string_list(parsed.get("recommendations")),
            claims=_dict_list(parsed.get("claims")),
            warnings=_string_list(parsed.get("warnings")),
            confidence=_confidence(parsed.get("confidence")),
            abstained=bool(parsed.get("abstained", False)),
            idempotency_key=task.effective_idempotency_key,
            lineage=task.lineage,
            output=parsed,
            model_role=role,
            model_id=str(getattr(provider, "model", "") or ""),
            duration_ms=round((time.perf_counter() - started_at) * 1000),
            raw_text=raw,
            metadata={
                "spec_version": spec.version,
                "output_contract": dict(spec.output_contract),
                "read_only": spec.read_only,
                "tool_trace": tool_trace,
                "model_calls": model_calls,
                "structured_output": structured_output,
                "output_outcome": output_outcome,
                "output_diagnostics": output_diagnostics,
            },
        )

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
