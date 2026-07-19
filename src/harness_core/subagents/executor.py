from __future__ import annotations

import ast
import hashlib
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from harness_core.budget import MODEL_CALLS, BudgetRequest
from harness_core.contracts import AgentMessage, ToolCall
from harness_core.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from harness_core.failures import RunCanceledError
from harness_core.model import HarnessModelAdapter, model_tools_from_registry
from harness_core.model_routing import ModelStepContext
from harness_core.skills import DelegationPolicy
from harness_core.subagents.contracts import (
    DelegationBatchResult,
    DelegationRequest,
    DelegationTask,
    SubAgentResult,
    SubAgentSpec,
)
from harness_core.subagents.registry import SubAgentRegistry
from harness_core.subagents.store import SubAgentRunStore
from harness_core.tools import ToolExecutionContext, ToolExecutor


EventSink = Callable[[dict[str, Any]], None]


class SubAgentOutputError(RuntimeError):
    def __init__(self, message: str, *, raw_text: str = "", diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.diagnostics = diagnostics or {}


class SubAgentEmptyResponseError(RuntimeError):
    """Normalized transient failure for model calls that produced no usable turn."""


class SubAgentCanceledError(RuntimeError):
    pass


@dataclass(slots=True)
class _DelegationQuota:
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    model_calls: int = 0
    tool_calls: int = 0
    lock: Lock = field(default_factory=Lock)

    def reserve_model_call(self) -> None:
        with self.lock:
            if self.max_model_calls is not None and self.model_calls >= self.max_model_calls:
                raise RuntimeError("delegation model call budget exceeded")
            self.model_calls += 1

    def reserve_tool_calls(self, count: int) -> None:
        with self.lock:
            if (
                self.max_tool_calls is not None
                and self.tool_calls + count > self.max_tool_calls
            ):
                raise RuntimeError("delegation tool call budget exceeded")
            self.tool_calls += count


class SubAgentExecutor:
    """Runs bounded specialists while the parent Agent retains final-answer ownership."""

    def __init__(
        self,
        registry: SubAgentRegistry,
        *,
        run_store: SubAgentRunStore | None = None,
        tool_executor: ToolExecutor | None = None,
        max_parallel: int = 3,
        max_depth: int = 1,
    ) -> None:
        self.registry = registry
        self.run_store = run_store
        self.tool_executor = tool_executor
        self.max_parallel = max(1, min(int(max_parallel), 3))
        self.max_depth = max(1, min(int(max_depth), 1))

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
        started_at = time.perf_counter()
        root_run_id = request.root_run_id or context.run_id
        parent_run_id = request.parent_run_id or context.run_id
        if not self._parent_accepts_results(parent_run_id):
            raise RuntimeError("parent agent run is already terminal")
        child_ids = {
            task.id: _child_run_id(parent_run_id, request.delegation_id, task)
            for task in request.tasks
        }
        self._emit(
            event_sink,
            "subagent.batch.started",
            "并行专业判断",
            f"已分派 {len(request.tasks)} 个受约束的专业判断。",
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
            self._load_child_checkpoint(child_ids[task.id]) or {}
            for task in pending_tasks
        ]
        restored_model_calls = sum(
            int((result.metadata or {}).get("model_calls") or 0)
            for result in results
        ) + sum(int(state.get("model_calls") or 0) for state in restored_checkpoints)
        restored_tool_calls = sum(
            len((result.metadata or {}).get("tool_trace") or [])
            for result in results
        ) + sum(int(state.get("tool_calls") or 0) for state in restored_checkpoints)
        quota = _DelegationQuota(
            max_model_calls=delegation_policy.max_model_calls if delegation_policy.enabled else None,
            max_tool_calls=delegation_policy.max_tool_calls if delegation_policy.enabled else None,
            model_calls=restored_model_calls,
            tool_calls=restored_tool_calls,
        )
        if pending_tasks:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="harness-subagent") as pool:
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
                            error=str(exc),
                        )
                    except Exception as exc:
                        result = SubAgentResult(
                            task_id=task.id,
                            agent_id=task.agent_id,
                            child_run_id=child_ids[task.id],
                            status="failed",
                            error=str(exc),
                            raw_text=str(getattr(exc, "raw_text", "") or ""),
                            metadata={
                                "diagnostics": dict(getattr(exc, "diagnostics", {}) or {}),
                            },
                        )
                    if result.status == "completed" and not self._parent_accepts_results(parent_run_id):
                        result = result.model_copy(
                            update={
                                "status": "canceled",
                                "conclusion": "",
                                "evidence": [],
                                "evidence_refs": [],
                                "risks": [],
                                "recommendations": [],
                                "claims": [],
                                "warnings": ["父运行已结束，迟到结果未进入主运行上下文。"],
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
                        }.get(result.status, "subagent.failed"),
                        request,
                        task,
                        spec,
                        result.child_run_id,
                        status=result.status,
                        result=result,
                    )

        ordered = sorted(results, key=lambda item: next(
            index for index, task in enumerate(request.tasks) if task.id == item.task_id
        ))
        completed = sum(result.status == "completed" for result in ordered)
        canceled = sum(result.status == "canceled" for result in ordered)
        degraded = sum(result.outcome == "degraded" for result in ordered)
        fail_batch = any(
            result.status != "completed"
            and self.registry.get(result.agent_id).failure_policy == "fail_batch"
            for result in ordered
        )
        status = (
            "canceled"
            if canceled == len(ordered)
            else "failed"
            if fail_batch
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
            "专业判断已汇总",
            (
                f"{completed}/{len(ordered)} 个专业判断已返回主 Agent。"
                + (f"其中 {degraded} 个使用了结构化降级结果。" if degraded else "")
            ),
            request,
            visible=True,
            extra={
                "status": status,
                "duration_ms": batch.duration_ms,
                "completed_count": completed,
                "degraded_count": degraded,
                "failed_count": len(ordered) - completed,
            },
        )
        return batch

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
            self._raise_if_canceled(child_run_id, parent_run_id, context=context)
        except SubAgentCanceledError:
            return SubAgentResult(
                task_id=task.id,
                agent_id=task.agent_id,
                child_run_id=child_run_id,
                status="canceled",
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
        raw, tool_trace, model_calls = self._run_bounded_agent(
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
            model_call_limit=model_call_limit,
            parent_run_id=parent_run_id,
            resume_state=resume_state,
            quota=quota,
        )
        self._raise_if_canceled(child_run_id, parent_run_id, context=context)
        output_outcome = "completed"
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
                self._raise_if_canceled(child_run_id, parent_run_id, context=context)
                _consume_model_budget(
                    budget_ledger,
                    root_run_id=root_run_id,
                    child_run_id=child_run_id,
                    task=task,
                    model_call_limit=model_call_limit,
                    step_index=model_calls,
                    quota=quota,
                )
                repair_prompt = _repair_prompt(prompt, raw, schema, str(first_error), tool_trace)
                repaired = _invoke_provider(
                    provider,
                    system_prompt,
                    repair_prompt,
                    timeout_seconds=remaining_timeout_ceiling(
                        context.deadline_monotonic,
                        maximum=spec.timeout_seconds,
                    ),
                    max_tokens=spec.max_tokens,
                    output_schema=schema,
                )
                self._raise_if_canceled(child_run_id, parent_run_id, context=context)
                ensure_time_remaining(context.deadline_monotonic)
                model_calls += 1
                self._checkpoint_child(
                    child_run_id,
                    phase="repair_completed",
                    state={
                        "schema_version": "subagent-execution-v1",
                        "task_id": task.id,
                        "spec_version": spec.version,
                        "phase": "repair_completed",
                        "raw_text": repaired,
                        "tool_trace": tool_trace,
                        "model_calls": model_calls,
                        "tool_calls": len(tool_trace),
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
        if not conclusion:
            raise RuntimeError("subagent returned an empty conclusion")
        return SubAgentResult(
            task_id=task.id,
            agent_id=task.agent_id,
            child_run_id=child_run_id,
            status="completed",
            outcome=output_outcome,
            conclusion=conclusion,
            evidence=_string_list(parsed.get("evidence")),
            evidence_refs=_dict_list(parsed.get("evidence_refs")),
            risks=_string_list(parsed.get("risks")),
            recommendations=_string_list(parsed.get("recommendations")),
            claims=_dict_list(parsed.get("claims")),
            warnings=_string_list(parsed.get("warnings")),
            confidence=_confidence(parsed.get("confidence")),
            abstained=bool(parsed.get("abstained", False)),
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
                "output_outcome": output_outcome,
                "output_diagnostics": output_diagnostics,
            },
        )

    def _run_bounded_agent(
        self,
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
    ) -> tuple[str, list[dict[str, Any]], int]:
        allowed = set(spec.tool_allowlist)
        native_tools = bool(
            allowed
            and self.tool_executor is not None
            and (
                callable(getattr(provider, "complete_chat", None))
                or callable(getattr(provider, "stream_chat", None))
            )
        )
        restored = _valid_resume_state(resume_state, task=task, spec=spec)
        restored_phase = str(restored.get("phase") or "")
        if restored_phase in {"output_ready", "repair_completed"} and str(
            restored.get("raw_text") or ""
        ).strip():
            return (
                str(restored.get("raw_text") or ""),
                _dict_list(restored.get("tool_trace")),
                int(restored.get("model_calls") or 0),
            )
        if not native_tools:
            model_calls = int(restored.get("model_calls") or 0)
            empty_response_retries = int(restored.get("empty_response_retries") or 0)
            while True:
                self._raise_if_canceled(child_run_id, parent_run_id, context=context)
                _consume_model_budget(
                    budget_ledger,
                    root_run_id=root_run_id,
                    child_run_id=child_run_id,
                    task=task,
                    model_call_limit=model_call_limit,
                    step_index=model_calls,
                    quota=quota,
                )
                try:
                    raw = _invoke_provider(
                        provider,
                        system_prompt,
                        prompt,
                        timeout_seconds=remaining_timeout_ceiling(
                            context.deadline_monotonic,
                            maximum=spec.timeout_seconds,
                        ),
                        max_tokens=spec.max_tokens,
                        output_schema=output_schema,
                    )
                    ensure_time_remaining(context.deadline_monotonic)
                    model_calls += 1
                    if not raw:
                        raise SubAgentEmptyResponseError("subagent model returned an empty response")
                    break
                except Exception as exc:
                    if not isinstance(exc, SubAgentEmptyResponseError):
                        model_calls += 1
                    if not _is_empty_model_response_error(exc):
                        raise
                    if empty_response_retries >= 1:
                        raise SubAgentOutputError(
                            "subagent model returned an empty response after retry",
                            diagnostics={
                                "reason_code": "empty_model_response",
                                "model_calls": model_calls,
                                "empty_response_retries": empty_response_retries,
                            },
                        ) from exc
                    empty_response_retries += 1
                    self._checkpoint_child(
                        child_run_id,
                        phase="model_retrying",
                        state={
                            "schema_version": "subagent-execution-v1",
                            "task_id": task.id,
                            "spec_version": spec.version,
                            "phase": "model_retrying",
                            "raw_text": "",
                            "tool_trace": [],
                            "model_calls": model_calls,
                            "tool_calls": 0,
                            "empty_response_retries": empty_response_retries,
                        },
                    )
                    _emit_subagent_model_retry(
                        event_sink,
                        child_run_id,
                        task,
                        model_calls=model_calls,
                    )
            self._checkpoint_child(
                child_run_id,
                phase="output_ready",
                state={
                    "schema_version": "subagent-execution-v1",
                    "task_id": task.id,
                    "spec_version": spec.version,
                    "phase": "output_ready",
                    "raw_text": raw,
                    "tool_trace": [],
                    "model_calls": model_calls,
                    "tool_calls": 0,
                    "empty_response_retries": empty_response_retries,
                },
            )
            return raw, [], model_calls

        tools = model_tools_from_registry(self.tool_executor.registry, allowed)
        messages = _restored_messages(restored) or [
            AgentMessage(id=f"{child_run_id}:input", role="user", content=prompt)
        ]
        tool_trace = _dict_list(restored.get("tool_trace"))
        model_calls = int(restored.get("model_calls") or 0)
        tool_calls = int(restored.get("tool_calls") or 0)
        empty_response_retries = int(restored.get("empty_response_retries") or 0)
        round_index = int(restored.get("round_index") or 0)
        pending_calls = [
            ToolCall.model_validate(item)
            for item in restored.get("pending_tool_calls", [])
            if isinstance(item, dict)
        ]
        child_context, owned_session = _child_tool_context(context, child_run_id, spec)
        try:
            while round_index <= spec.max_tool_rounds:
                self._raise_if_canceled(child_run_id, parent_run_id, context=context)
                if not pending_calls:
                    _consume_model_budget(
                        budget_ledger,
                        root_run_id=root_run_id,
                        child_run_id=child_run_id,
                        task=task,
                        model_call_limit=model_call_limit,
                        step_index=model_calls,
                        quota=quota,
                    )
                    try:
                        turn = HarnessModelAdapter(
                            provider,
                            request_timeout=remaining_timeout_ceiling(
                                context.deadline_monotonic,
                                maximum=spec.timeout_seconds,
                            ),
                        ).run_turn(
                            messages,
                            tools=tools if round_index < spec.max_tool_rounds else [],
                            system_prompt=system_prompt,
                            step_context=ModelStepContext(
                                run_id=child_run_id,
                                user_id=context.user_id,
                                thread_id=context.thread_id,
                                turn_id=context.turn_id,
                                step_index=model_calls,
                                message_count=len(messages),
                                observation_count=len(tool_trace),
                                tool_result_count=len(tool_trace),
                                available_roles=(
                                    str(
                                        getattr(provider, "model_role", "")
                                        or spec.model_role
                                        or "reasoning"
                                    ),
                                ),
                                deadline_monotonic=context.deadline_monotonic,
                            ),
                        )
                        model_calls += 1
                        if not turn.content.strip() and not turn.tool_calls:
                            raise SubAgentEmptyResponseError(
                                "subagent model returned an empty response"
                            )
                    except Exception as exc:
                        if not isinstance(exc, SubAgentEmptyResponseError):
                            model_calls += 1
                        if not _is_empty_model_response_error(exc):
                            raise
                        if empty_response_retries >= 1:
                            raise SubAgentOutputError(
                                "subagent model returned an empty response after retry",
                                diagnostics={
                                    "reason_code": "empty_model_response",
                                    "model_calls": model_calls,
                                    "empty_response_retries": empty_response_retries,
                                    "tool_trace": tool_trace,
                                },
                            ) from exc
                        empty_response_retries += 1
                        self._checkpoint_child(
                            child_run_id,
                            phase="model_retrying",
                            state=_execution_checkpoint(
                                task=task,
                                spec=spec,
                                phase="model_retrying",
                                round_index=round_index,
                                messages=messages,
                                pending_calls=[],
                                tool_trace=tool_trace,
                                model_calls=model_calls,
                                tool_calls=tool_calls,
                                empty_response_retries=empty_response_retries,
                            ),
                        )
                        _emit_subagent_model_retry(
                            event_sink,
                            child_run_id,
                            task,
                            model_calls=model_calls,
                        )
                        continue
                    messages.append(
                        AgentMessage(
                            id=f"{child_run_id}:assistant:{round_index}",
                            role="assistant",
                            content=turn.content,
                            tool_calls=turn.tool_calls,
                        )
                    )
                    pending_calls = list(turn.tool_calls)
                    phase = "model_completed" if pending_calls else "output_ready"
                    self._checkpoint_child(
                        child_run_id,
                        phase=phase,
                        state=_execution_checkpoint(
                            task=task,
                            spec=spec,
                            phase=phase,
                            round_index=round_index,
                            messages=messages,
                            pending_calls=pending_calls,
                            tool_trace=tool_trace,
                            model_calls=model_calls,
                            tool_calls=tool_calls,
                            empty_response_retries=empty_response_retries,
                            raw_text=turn.content if not pending_calls else "",
                        ),
                    )
                    if not pending_calls:
                        return turn.content, tool_trace, model_calls
                if round_index >= spec.max_tool_rounds:
                    break
                accepted: list[ToolCall] = []
                for call in pending_calls:
                    if call.name not in allowed:
                        raise RuntimeError(f"subagent requested tool outside allowlist: {call.name}")
                    tool_spec = self.tool_executor.registry.get(call.name)
                    if not tool_spec.read_only:
                        raise RuntimeError(f"subagent requested non-read-only tool: {call.name}")
                    accepted.append(call)
                if tool_calls + len(accepted) > spec.max_tool_calls:
                    raise RuntimeError("subagent tool call limit exceeded")
                if quota is not None:
                    quota.reserve_tool_calls(len(accepted))
                tool_calls += len(accepted)
                self._raise_if_canceled(child_run_id, parent_run_id, context=context)
                _emit_subagent_tools(event_sink, child_run_id, task, accepted, status="started")
                results = self.tool_executor.execute_many(accepted, child_context)
                for result in results:
                    trace = {
                        "tool_call_id": result.tool_call_id,
                        "tool_name": result.name,
                        "status": result.status,
                        "summary": result.summary,
                        "error": result.error,
                    }
                    tool_trace.append(trace)
                    messages.append(
                        AgentMessage(
                            id=f"{child_run_id}:tool:{result.tool_call_id}",
                            role="tool",
                            name=result.name,
                            tool_call_id=result.tool_call_id,
                            content=json.dumps(
                                {
                                    "status": result.status,
                                    "summary": result.summary,
                                    "data": result.data,
                                    "error": result.error,
                                },
                                ensure_ascii=False,
                                default=str,
                            ),
                        )
                    )
                _emit_subagent_tools(event_sink, child_run_id, task, accepted, status="completed", trace=tool_trace)
                pending_calls = []
                round_index += 1
                self._checkpoint_child(
                    child_run_id,
                    phase="tools_completed",
                    state=_execution_checkpoint(
                        task=task,
                        spec=spec,
                        phase="tools_completed",
                        round_index=round_index,
                        messages=messages,
                        pending_calls=[],
                        tool_trace=tool_trace,
                        model_calls=model_calls,
                        tool_calls=tool_calls,
                        empty_response_retries=empty_response_retries,
                    ),
                )
            return "", tool_trace, model_calls
        finally:
            if owned_session is not None:
                owned_session.close()

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
    ) -> None:
        try:
            context.run_control.raise_if_cancelled(parent_run_id or context.run_id)
            if child_run_id and child_run_id != parent_run_id:
                context.run_control.raise_if_cancelled(child_run_id)
        except RunCanceledError as exc:
            raise SubAgentCanceledError(
                "subagent execution was canceled by run control"
            ) from exc
        if self._cancel_requested(child_run_id, parent_run_id):
            raise SubAgentCanceledError("subagent execution was canceled by its parent run")

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
        sink({
            "event_type": event_type,
            "title": title,
            "summary": summary,
            "payload": {
                "visible": visible,
                "delegation_id": request.delegation_id,
                "root_run_id": request.root_run_id,
                "parent_run_id": request.parent_run_id,
                "task_count": len(request.tasks),
                **(extra or {}),
            },
        })

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
        sink({
            "event_type": event_type,
            "title": spec.label,
            "summary": result.conclusion if result and result.conclusion else task.objective,
            "payload": {
                "visible": True,
                "delegation_id": request.delegation_id,
                "task_id": task.id,
                "agent_id": spec.id,
                "agent_label": spec.label,
                "child_run_id": child_run_id,
                "parent_run_id": request.parent_run_id,
                "root_run_id": request.root_run_id,
                "status": status,
                "model_role": result.model_role if result else task.model_role,
                "model_id": result.model_id if result else "",
                "duration_ms": result.duration_ms if result else 0,
                "error": result.error if result else "",
            },
        })


def _resolve_role(task_role: str, spec_role: str, providers: dict[str, Any]) -> str:
    for role in (task_role, spec_role, "reasoning", "fast"):
        if role != "auto" and role in providers:
            return role
    raise RuntimeError("subagent has no available model provider")


def _child_run_id(parent_run_id: str, delegation_id: str, task: DelegationTask) -> str:
    identity = f"{parent_run_id}|{delegation_id}|{task.id}|{task.agent_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    parent_prefix = str(parent_run_id or "root")[:72]
    return f"{parent_prefix}:sub:{digest}"


def _valid_resume_state(
    value: dict[str, Any] | None,
    *,
    task: DelegationTask,
    spec: SubAgentSpec,
) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    if state.get("schema_version") != "subagent-execution-v1":
        return {}
    if str(state.get("task_id") or "") != task.id:
        return {}
    if str(state.get("spec_version") or "") != spec.version:
        return {}
    return dict(state)


def _restored_messages(state: dict[str, Any]) -> list[AgentMessage]:
    messages: list[AgentMessage] = []
    for item in state.get("messages", []):
        if not isinstance(item, dict):
            continue
        try:
            messages.append(AgentMessage.model_validate(item))
        except (TypeError, ValueError):
            return []
    return messages


def _execution_checkpoint(
    *,
    task: DelegationTask,
    spec: SubAgentSpec,
    phase: str,
    round_index: int,
    messages: list[AgentMessage],
    pending_calls: list[ToolCall],
    tool_trace: list[dict[str, Any]],
    model_calls: int,
    tool_calls: int,
    empty_response_retries: int = 0,
    raw_text: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "subagent-execution-v1",
        "task_id": task.id,
        "spec_version": spec.version,
        "phase": phase,
        "round_index": round_index,
        "messages": [message.model_dump(mode="json") for message in messages],
        "pending_tool_calls": [call.model_dump(mode="json") for call in pending_calls],
        "tool_trace": list(tool_trace),
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "empty_response_retries": empty_response_retries,
        "raw_text": raw_text,
    }


def _minimum_optional(first: Any, second: Any) -> int | None:
    values: list[int] = []
    for value in (first, second):
        if value in (None, ""):
            continue
        try:
            values.append(max(0, int(value)))
        except (TypeError, ValueError):
            continue
    return min(values) if values else None


def _child_tool_context(
    context: ToolExecutionContext,
    child_run_id: str,
    spec: SubAgentSpec,
) -> tuple[ToolExecutionContext, Any | None]:
    owned_session = context.session_factory() if context.session_factory is not None else None
    child = ToolExecutionContext(
        run_id=child_run_id,
        user_id=context.user_id,
        thread_id=context.thread_id,
        turn_id=context.turn_id,
        session=owned_session or context.session,
        session_factory=context.session_factory,
        context_bundle=dict(context.context_bundle),
        metadata={
            **context.metadata,
            "subagent": {
                "agent_id": spec.id,
                "read_only": True,
                "tool_allowlist": list(spec.tool_allowlist),
            },
        },
        budget_limits=dict(context.budget_limits),
        deadline_monotonic=context.deadline_monotonic,
        run_control=context.run_control,
    )
    return child, owned_session


def _consume_model_budget(
    budget_ledger: Any,
    *,
    root_run_id: str,
    child_run_id: str,
    task: DelegationTask,
    model_call_limit: float | None,
    step_index: int,
    quota: _DelegationQuota | None = None,
) -> None:
    if quota is not None:
        quota.reserve_model_call()
    if budget_ledger is None:
        return
    budget = budget_ledger.consume(
        BudgetRequest(
            run_id=root_run_id,
            metric=MODEL_CALLS,
            amount=1,
            limit=model_call_limit,
            operation_id=f"subagent-model:{child_run_id}:{step_index}",
            metadata={
                "agent_id": task.agent_id,
                "task_id": task.id,
                "subagent_step": step_index,
            },
        )
    )
    if not budget.allowed:
        raise RuntimeError(budget.reason)


def _emit_subagent_tools(
    sink: EventSink | None,
    child_run_id: str,
    task: DelegationTask,
    calls: list[ToolCall],
    *,
    status: str,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    if sink is None:
        return
    names = [call.name for call in calls]
    sink({
        "event_type": f"subagent.tools.{status}",
        "title": "子 Agent 正在补充依据" if status == "started" else "子 Agent 已完成依据核验",
        "summary": "、".join(names),
        "payload": {
            "visible": False,
            "child_run_id": child_run_id,
            "task_id": task.id,
            "agent_id": task.agent_id,
            "status": status,
            "tool_names": names,
            "tool_trace": list(trace or []),
        },
    })


def _emit_subagent_model_retry(
    sink: EventSink | None,
    child_run_id: str,
    task: DelegationTask,
    *,
    model_calls: int,
) -> None:
    if sink is None:
        return
    sink({
        "event_type": "subagent.model.retrying",
        "title": "专业判断正在重试",
        "summary": "模型本次未返回有效内容，正在自动重试。",
        "payload": {
            "visible": False,
            "child_run_id": child_run_id,
            "task_id": task.id,
            "agent_id": task.agent_id,
            "status": "retrying",
            "reason_code": "empty_model_response",
            "model_calls": model_calls,
        },
    })


def _is_empty_model_response_error(error: Exception) -> bool:
    if isinstance(error, SubAgentEmptyResponseError):
        return True
    message = str(error or "").strip().lower()
    return any(
        marker in message
        for marker in (
            "llm returned an empty response",
            "model returned an empty response",
            "empty model response",
        )
    )


def _repair_prompt(
    original_prompt: str,
    raw: str,
    schema: dict[str, Any],
    error: str,
    tool_trace: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "task": "修复上一份输出，使其严格满足 JSON Schema。只输出 JSON 对象，不添加解释。",
            "validation_error": error,
            "output_schema": schema,
            "invalid_output": str(raw or "")[-12000:],
            "tool_trace": tool_trace,
            "original_task": original_prompt,
        },
        ensure_ascii=False,
        default=str,
    )


def _default_system_prompt(spec: SubAgentSpec) -> str:
    return (
        f"你是{spec.label}。只完成被委派的单一判断，不与用户对话，不扩展任务边界。"
        "不得递归委派，也不得执行有副作用的操作。必须区分输入事实、规则推论与不确定项。"
        "严格按照给定 JSON Schema 输出，不要附加 Markdown。"
    )


def _task_prompt(task: DelegationTask, spec: SubAgentSpec) -> str:
    return json.dumps(
        {
            "objective": task.objective,
            "input": task.input_data,
            "constraints": task.constraints,
            "capabilities": spec.capabilities,
            "tool_allowlist": spec.tool_allowlist,
            "input_contract": spec.input_contract,
            "output_contract": spec.output_contract,
            "evidence_policy": spec.evidence_policy,
            "execution_policy": {
                "read_only": spec.read_only,
                "allow_delegation": False,
            },
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _invoke_provider(
    provider: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_seconds: int,
    max_tokens: int,
    output_schema: dict[str, Any],
) -> str:
    completion_budget = _subagent_completion_budget(max_tokens)
    complete = getattr(provider, "complete", None)
    if callable(complete):
        kwargs = {
            "request_timeout": timeout_seconds,
            "max_tokens": completion_budget,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "subagent_result",
                    "strict": True,
                    "schema": output_schema,
                },
            },
        }
        supported = _supported_kwargs(complete, kwargs)
        return str(complete(system_prompt, user_prompt, **supported) or "").strip()
    complete_chat = getattr(provider, "complete_chat", None)
    if callable(complete_chat):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "subagent_result",
                "strict": True,
                "schema": output_schema,
            },
        }
        kwargs = _supported_kwargs(
            complete_chat,
            {
                "tools": [],
                "tool_choice": "none",
                "request_timeout": timeout_seconds,
                "max_tokens": completion_budget,
                "response_format": response_format,
            },
        )
        response = complete_chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            **kwargs,
        )
        message = response.get("message") if isinstance(response, dict) else {}
        parsed = message.get("parsed") if isinstance(message, dict) else None
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False)
        return str(message.get("content") or "").strip()
    raise RuntimeError("subagent provider does not expose complete or complete_chat")


def _subagent_completion_budget(requested_tokens: int) -> int:
    """Reserve room for providers that count hidden reasoning as completion tokens."""
    return min(8000, max(int(requested_tokens or 0), 4000))


def _supported_kwargs(function: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return {}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return candidates
    return {key: value for key, value in candidates.items() if key in signature.parameters}


def _json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _fallback_subagent_output(value: str) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    if not text or text in {"{}", "[]", "null"}:
        return None
    if text.startswith("{"):
        for key in ("conclusion", "summary"):
            marker = f'"{key}"'
            marker_index = text.find(marker)
            if marker_index < 0:
                continue
            value_start = text.find(":", marker_index + len(marker))
            quote_start = text.find('"', value_start + 1)
            quote_end = text.find('"', quote_start + 1)
            if quote_start >= 0 and quote_end > quote_start:
                candidate = text[quote_start + 1 : quote_end].strip()
                if candidate:
                    text = candidate
                    break
    return {
        "conclusion": text[:4000],
        "evidence": [],
        "evidence_refs": [],
        "risks": [],
        "recommendations": [],
        "claims": [],
        "warnings": ["模型返回了可用文本，但结构化格式未通过校验；主 Agent 应降低置信度并复核。"],
        "confidence": 0.35,
        "abstained": False,
    }


def _validated_json(value: str, schema: dict[str, Any]) -> dict[str, Any]:
    parsed = _json_object(value)
    _validate_output(parsed, schema)
    return parsed


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = _readable_list_item(item)
        if text:
            result.append(text)
    return result


def _readable_list_item(value: Any) -> str:
    if isinstance(value, dict):
        return _dict_summary(value)
    text = str(value or "").strip()
    if not text or not (text.startswith("{") and text.endswith("}")):
        return text
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
    return _dict_summary(parsed) if isinstance(parsed, dict) else text


def _dict_summary(value: dict[str, Any]) -> str:
    for key in ("summary", "conclusion", "claim", "text", "description", "title", "fact"):
        text = str(value.get(key) or "").strip()
        if text:
            return text
    return "；".join(
        f"{key}：{item}"
        for key, item in list(value.items())[:4]
        if item not in (None, "", [], {})
    )


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return None


def _output_schema(spec: SubAgentSpec) -> dict[str, Any]:
    default_properties: dict[str, Any] = {
        "conclusion": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "object"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {"type": "object"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": ["number", "null"]},
        "abstained": {"type": "boolean"},
    }
    contract = dict(spec.output_contract or {})
    nested = contract.get("schema")
    if isinstance(nested, dict):
        return dict(nested)
    if contract.get("type") == "object" and isinstance(contract.get("properties"), dict):
        schema = dict(contract)
        schema.setdefault("additionalProperties", False)
        return schema
    properties = dict(default_properties)
    properties.update(contract.get("properties") or {})
    required = contract.get("required")
    if not isinstance(required, list) or not required:
        required = ["conclusion", "evidence", "risks", "recommendations"]
    for key in required:
        properties.setdefault(str(key), {})
    return {
        "type": "object",
        "properties": properties,
        "required": [str(key) for key in required],
        "additionalProperties": bool(contract.get("additionalProperties", False)),
    }


def _validate_output(value: dict[str, Any], schema: dict[str, Any]) -> None:
    if not value:
        raise RuntimeError("subagent returned invalid JSON")
    missing = [str(key) for key in schema.get("required") or [] if key not in value]
    if missing:
        raise RuntimeError(f"subagent output is missing required fields: {', '.join(missing)}")
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for key, rule in properties.items():
        if key not in value or not isinstance(rule, dict) or not rule.get("type"):
            continue
        expected = rule["type"]
        valid = (
            any(_matches_json_type(value[key], str(item)) for item in expected)
            if isinstance(expected, list)
            else _matches_json_type(value[key], str(expected))
        )
        if not valid:
            raise RuntimeError(f"subagent output field has invalid type: {key}")


def _validate_input(value: dict[str, Any], contract: dict[str, Any]) -> None:
    required = contract.get("required") if isinstance(contract, dict) else []
    if not isinstance(required, list):
        return
    missing = [
        str(key)
        for key in required
        if str(key) not in value or value.get(str(key)) in (None, "", [], {})
    ]
    if missing:
        raise RuntimeError(f"subagent input is missing required fields: {', '.join(missing)}")


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True
