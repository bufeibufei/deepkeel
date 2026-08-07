from __future__ import annotations

import json
from typing import Any

from deepkeel.contracts import AgentMessage, ToolCall
from deepkeel.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from deepkeel.model import NativeChatProviderAdapter, model_tools_from_registry
from deepkeel.model_routing import ModelStepContext
from deepkeel.subagents.contracts import DelegationTask, SubAgentSpec
from deepkeel.subagents.execution_support import (
    _child_tool_context,
    _consume_model_budget,
    _emit_subagent_model_retry,
    _emit_subagent_tools,
    _execution_checkpoint,
    _invoke_provider,
    _is_empty_model_response_error,
    _minimum_optional,
    _restored_messages,
    _valid_resume_state,
)
from deepkeel.subagents.execution_types import (
    EventSink,
    SubAgentEmptyResponseError,
    SubAgentOutputError,
    _DelegationQuota,
)
from deepkeel.subagents.output_validation import _dict_list
from deepkeel.tools import ToolExecutionContext
from deepkeel.type_narrowing import as_dict


class SubAgentBoundedExecutionMixin:
    """Bounded specialist ReAct loop with resumable checkpoints and quotas."""

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
        allowed = set(spec.tool_allowlist)
        tool_executor = self.tool_executor
        native_tools = bool(
            allowed
            and tool_executor is not None
            and (
                callable(getattr(provider, "complete_chat", None))
                or callable(getattr(provider, "stream_chat", None))
            )
        )
        restored = _valid_resume_state(resume_state, task=task, spec=spec)
        restored_phase = str(restored.get("phase") or "")
        if (
            restored_phase in {"output_ready", "repair_completed"}
            and str(restored.get("raw_text") or "").strip()
        ):
            return (
                str(restored.get("raw_text") or ""),
                _dict_list(restored.get("tool_trace")),
                int(restored.get("model_calls") or 0),
                as_dict(restored.get("structured_output")),
            )
        if not native_tools:
            model_calls = int(restored.get("model_calls") or 0)
            empty_response_retries = int(restored.get("empty_response_retries") or 0)
            while True:
                self._raise_if_canceled(child_run_id, parent_run_id, context=context, task=task)
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
                try:
                    invocation = _invoke_provider(
                        provider,
                        system_prompt,
                        prompt,
                        timeout_seconds=remaining_timeout_ceiling(
                            deadline_monotonic,
                            maximum=self._task_timeout_seconds(task, spec),
                        ),
                        max_tokens=self._task_max_tokens(task, spec),
                        output_schema=output_schema,
                        capability_registry=self.model_capabilities,
                    )
                    raw = invocation.text
                    ensure_time_remaining(deadline_monotonic)
                    model_calls += 1
                    if not raw:
                        raise SubAgentEmptyResponseError(
                            "subagent model returned an empty response"
                        )
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
                            "idempotency_key": task.effective_idempotency_key,
                            "lineage": task.lineage.model_dump(mode="json"),
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
                    "idempotency_key": task.effective_idempotency_key,
                    "lineage": task.lineage.model_dump(mode="json"),
                    "spec_version": spec.version,
                    "phase": "output_ready",
                    "raw_text": raw,
                    "tool_trace": [],
                    "model_calls": model_calls,
                    "tool_calls": 0,
                    "empty_response_retries": empty_response_retries,
                    "structured_output": invocation.diagnostics(),
                },
            )
            return raw, [], model_calls, invocation.diagnostics()

        if tool_executor is None:
            raise RuntimeError("subagent tool executor is unavailable")

        tools = model_tools_from_registry(tool_executor.registry, allowed)
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
        child_context, owned_session = _child_tool_context(
            context,
            child_run_id,
            spec,
            task,
            deadline_monotonic=deadline_monotonic,
        )
        try:
            while round_index <= spec.max_tool_rounds:
                self._raise_if_canceled(child_run_id, parent_run_id, context=context, task=task)
                if not pending_calls:
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
                    try:
                        turn = NativeChatProviderAdapter(
                            provider,
                            request_timeout=remaining_timeout_ceiling(
                                deadline_monotonic,
                                maximum=self._task_timeout_seconds(task, spec),
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
                                deadline_monotonic=deadline_monotonic,
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
                        return (
                            turn.content,
                            tool_trace,
                            model_calls,
                            {
                                "requested_format": "json_schema",
                                "effective_format": "prompt_only",
                                "capability_source": "native_tool_loop",
                                "degraded": True,
                                "degradation_reason": "native_tool_loop_uses_local_validation",
                            },
                        )
                if round_index >= spec.max_tool_rounds:
                    break
                accepted: list[ToolCall] = []
                for call in pending_calls:
                    if call.name not in allowed:
                        raise RuntimeError(
                            f"subagent requested tool outside allowlist: {call.name}"
                        )
                    tool_spec = tool_executor.registry.get(call.name)
                    if not tool_spec.read_only:
                        raise RuntimeError(f"subagent requested non-read-only tool: {call.name}")
                    accepted.append(call)
                task_tool_limit = _minimum_optional(
                    spec.max_tool_calls,
                    task.budget.max_tool_calls,
                )
                if task_tool_limit is not None and tool_calls + len(accepted) > task_tool_limit:
                    raise RuntimeError("subagent tool call limit exceeded")
                if quota is not None:
                    quota.reserve_tool_calls(len(accepted))
                if task_quota is not None:
                    task_quota.reserve_tool_calls(len(accepted))
                tool_calls += len(accepted)
                self._raise_if_canceled(child_run_id, parent_run_id, context=context, task=task)
                _emit_subagent_tools(event_sink, child_run_id, task, accepted, status="started")
                results = tool_executor.execute_many(accepted, child_context)
                for result in results:
                    trace = {
                        "tool_call_id": result.tool_call_id,
                        "tool_name": result.name,
                        "status": result.status,
                        "summary": result.summary,
                        "error": result.error,
                        "artifact_refs": [
                            {
                                "id": artifact.id,
                                "artifact_type": artifact.artifact_type,
                                "metadata": {
                                    "source_tool": result.name,
                                    "source_run_id": artifact.run_id,
                                },
                            }
                            for artifact in result.artifacts
                        ],
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
                _emit_subagent_tools(
                    event_sink, child_run_id, task, accepted, status="completed", trace=tool_trace
                )
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
            return (
                "",
                tool_trace,
                model_calls,
                {
                    "requested_format": "json_schema",
                    "effective_format": "prompt_only",
                    "capability_source": "native_tool_loop",
                    "degraded": True,
                    "degradation_reason": "native_tool_loop_exhausted",
                },
            )
        finally:
            if owned_session is not None:
                owned_session.close()
