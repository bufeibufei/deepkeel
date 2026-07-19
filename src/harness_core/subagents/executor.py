from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from harness_core.budget import MODEL_CALLS
from harness_core.contracts import AgentMessage, ToolCall
from harness_core.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from harness_core.failures import RunCanceledError
from harness_core.model import NativeChatProviderAdapter, model_tools_from_registry
from harness_core.model_routing import ModelStepContext
from harness_core.skills import DelegationPolicy
from harness_core.subagents.contracts import (
    DelegationBatchResult, DelegationRequest, DelegationTask,
    SubAgentResult, SubAgentSpec,
)
from harness_core.subagents.execution_types import (
    EventSink, SubAgentCanceledError, SubAgentEmptyResponseError,
    SubAgentOutputError, _DelegationQuota,
)
from harness_core.subagents.execution_support import (
    _child_run_id, _child_tool_context, _consume_model_budget,
    _default_system_prompt, _emit_subagent_model_retry, _emit_subagent_tools,
    _execution_checkpoint, _invoke_provider, _is_empty_model_response_error,
    _minimum_optional, _repair_prompt, _resolve_role, _restored_messages,
    _task_prompt, _valid_resume_state,
)
from harness_core.subagents.output_validation import (
    _confidence, _dict_list, _fallback_subagent_output, _output_schema,
    _string_list, _validate_input, _validated_json,
)
from harness_core.subagents.registry import SubAgentRegistry
from harness_core.subagents.store import SubAgentRunStore
from harness_core.tools import ToolExecutionContext, ToolExecutor

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
                                "warnings": ["The parent run ended; the late result was not merged."],
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
                        turn = NativeChatProviderAdapter(
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
