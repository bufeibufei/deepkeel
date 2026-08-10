from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from deepkeel.contracts import AgentMessage, ToolCall
from deepkeel.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from deepkeel.model import NativeChatProviderAdapter, model_tools_from_registry
from deepkeel.model_invocations import ModelTurn
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
from deepkeel.tools import ToolExecutionContext, ToolExecutor
from deepkeel.type_narrowing import as_dict

BoundedResult = tuple[str, list[dict[str, Any]], int, dict[str, Any]]


@dataclass(slots=True)
class _BoundedInputs:
    task: DelegationTask
    spec: SubAgentSpec
    provider: Any
    child_run_id: str
    context: ToolExecutionContext
    event_sink: EventSink | None
    system_prompt: str
    prompt: str
    output_schema: dict[str, Any]
    root_run_id: str
    budget_ledger: Any
    model_call_limit: float | None
    parent_run_id: str
    resume_state: dict[str, Any] | None
    quota: _DelegationQuota | None
    task_quota: _DelegationQuota | None
    deadline_monotonic: float | None


@dataclass(slots=True)
class _NativeLoopState:
    messages: list[AgentMessage]
    tool_trace: list[dict[str, Any]]
    pending_calls: list[ToolCall]
    model_calls: int
    tool_calls: int
    empty_response_retries: int
    round_index: int
    tools: list[dict[str, Any]] = field(default_factory=list)


class BoundedAgentCoordinator:
    """Runs one resumable bounded specialist through explicit execution phases."""

    def __init__(
        self,
        owner: Any,
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
    ) -> None:
        self.owner = owner
        self.inputs = _BoundedInputs(
            task=task,
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
        )
        self.restored = _valid_resume_state(resume_state, task=task, spec=spec)

    def run(self) -> BoundedResult:
        restored = self._restored_result()
        if restored is not None:
            return restored
        tool_executor = self.owner.tool_executor
        if not self._supports_native_tools(tool_executor):
            return self._run_structured_only()
        if tool_executor is None:
            raise RuntimeError("subagent tool executor is unavailable")
        return self._run_native_loop(tool_executor)

    def _restored_result(self) -> BoundedResult | None:
        phase = str(self.restored.get("phase") or "")
        raw_text = str(self.restored.get("raw_text") or "")
        if phase not in {"output_ready", "repair_completed"} or not raw_text.strip():
            return None
        return (
            raw_text,
            _dict_list(self.restored.get("tool_trace")),
            int(self.restored.get("model_calls") or 0),
            as_dict(self.restored.get("structured_output")),
        )

    def _supports_native_tools(self, tool_executor: ToolExecutor | None) -> bool:
        return bool(
            self.inputs.spec.tool_allowlist
            and tool_executor is not None
            and (
                callable(getattr(self.inputs.provider, "complete_chat", None))
                or callable(getattr(self.inputs.provider, "stream_chat", None))
            )
        )

    def _run_structured_only(self) -> BoundedResult:
        model_calls = int(self.restored.get("model_calls") or 0)
        retries = int(self.restored.get("empty_response_retries") or 0)
        while True:
            self._raise_if_canceled()
            self._consume_model_budget(model_calls)
            try:
                invocation = _invoke_provider(
                    self.inputs.provider,
                    self.inputs.system_prompt,
                    self.inputs.prompt,
                    timeout_seconds=self._remaining_timeout(),
                    max_tokens=self._max_tokens(),
                    output_schema=self.inputs.output_schema,
                    capability_registry=self.owner.model_capabilities,
                )
                raw = invocation.text
                ensure_time_remaining(self.inputs.deadline_monotonic)
                model_calls += 1
                if not raw:
                    raise SubAgentEmptyResponseError("subagent model returned an empty response")
                break
            except Exception as exc:
                if not isinstance(exc, SubAgentEmptyResponseError):
                    model_calls += 1
                retries = self._handle_structured_retry(exc, model_calls, retries)
        diagnostics = invocation.diagnostics()
        self.owner._checkpoint_child(
            self.inputs.child_run_id,
            phase="output_ready",
            state=self._simple_checkpoint(
                phase="output_ready",
                raw_text=raw,
                model_calls=model_calls,
                retries=retries,
                structured_output=diagnostics,
            ),
        )
        return raw, [], model_calls, diagnostics

    def _handle_structured_retry(self, exc: Exception, model_calls: int, retries: int) -> int:
        if not _is_empty_model_response_error(exc):
            raise exc
        if retries >= 1:
            raise SubAgentOutputError(
                "subagent model returned an empty response after retry",
                diagnostics={
                    "reason_code": "empty_model_response",
                    "model_calls": model_calls,
                    "empty_response_retries": retries,
                },
            ) from exc
        retries += 1
        self.owner._checkpoint_child(
            self.inputs.child_run_id,
            phase="model_retrying",
            state=self._simple_checkpoint(
                phase="model_retrying",
                raw_text="",
                model_calls=model_calls,
                retries=retries,
            ),
        )
        _emit_subagent_model_retry(
            self.inputs.event_sink,
            self.inputs.child_run_id,
            self.inputs.task,
            model_calls=model_calls,
        )
        return retries

    def _simple_checkpoint(
        self,
        *,
        phase: str,
        raw_text: str,
        model_calls: int,
        retries: int,
        structured_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "subagent-execution-v1",
            "task_id": self.inputs.task.id,
            "idempotency_key": self.inputs.task.effective_idempotency_key,
            "lineage": self.inputs.task.lineage.model_dump(mode="json"),
            "spec_version": self.inputs.spec.version,
            "phase": phase,
            "raw_text": raw_text,
            "tool_trace": [],
            "model_calls": model_calls,
            "tool_calls": 0,
            "empty_response_retries": retries,
            **({"structured_output": structured_output} if structured_output is not None else {}),
        }

    def _run_native_loop(self, tool_executor: ToolExecutor) -> BoundedResult:
        state = self._native_state(tool_executor)
        child_context, owned_session = _child_tool_context(
            self.inputs.context,
            self.inputs.child_run_id,
            self.inputs.spec,
            self.inputs.task,
            deadline_monotonic=self.inputs.deadline_monotonic,
        )
        try:
            while state.round_index <= self.inputs.spec.max_tool_rounds:
                self._raise_if_canceled()
                if not state.pending_calls:
                    turn = self._invoke_native_turn(state)
                    if turn is None:
                        continue
                    completed = self._record_native_turn(state, turn)
                    if completed is not None:
                        return completed
                if state.round_index >= self.inputs.spec.max_tool_rounds:
                    break
                accepted = self._validate_tool_calls(state, tool_executor)
                self._execute_tool_round(state, accepted, tool_executor, child_context)
            return self._exhausted_result(state)
        finally:
            if owned_session is not None:
                owned_session.close()

    def _native_state(self, tool_executor: ToolExecutor) -> _NativeLoopState:
        pending_calls = [
            ToolCall.model_validate(item)
            for item in self.restored.get("pending_tool_calls", [])
            if isinstance(item, dict)
        ]
        return _NativeLoopState(
            messages=_restored_messages(self.restored)
            or [
                AgentMessage(
                    id=f"{self.inputs.child_run_id}:input",
                    role="user",
                    content=self.inputs.prompt,
                )
            ],
            tool_trace=_dict_list(self.restored.get("tool_trace")),
            pending_calls=pending_calls,
            model_calls=int(self.restored.get("model_calls") or 0),
            tool_calls=int(self.restored.get("tool_calls") or 0),
            empty_response_retries=int(self.restored.get("empty_response_retries") or 0),
            round_index=int(self.restored.get("round_index") or 0),
            tools=model_tools_from_registry(
                tool_executor.registry,
                set(self.inputs.spec.tool_allowlist),
            ),
        )

    def _invoke_native_turn(self, state: _NativeLoopState) -> ModelTurn | None:
        self._consume_model_budget(state.model_calls)
        try:
            turn = NativeChatProviderAdapter(
                self.inputs.provider,
                request_timeout=self._remaining_timeout(),
            ).run_turn(
                state.messages,
                tools=(
                    state.tools
                    if state.round_index < self.inputs.spec.max_tool_rounds
                    else []
                ),
                system_prompt=self.inputs.system_prompt,
                step_context=self._step_context(state),
            )
            state.model_calls += 1
            if not turn.content.strip() and not turn.tool_calls:
                raise SubAgentEmptyResponseError("subagent model returned an empty response")
            return turn
        except Exception as exc:
            if not isinstance(exc, SubAgentEmptyResponseError):
                state.model_calls += 1
            self._handle_native_retry(state, exc)
            return None

    def _handle_native_retry(self, state: _NativeLoopState, exc: Exception) -> None:
        if not _is_empty_model_response_error(exc):
            raise exc
        if state.empty_response_retries >= 1:
            raise SubAgentOutputError(
                "subagent model returned an empty response after retry",
                diagnostics={
                    "reason_code": "empty_model_response",
                    "model_calls": state.model_calls,
                    "empty_response_retries": state.empty_response_retries,
                    "tool_trace": state.tool_trace,
                },
            ) from exc
        state.empty_response_retries += 1
        self._checkpoint_native(state, phase="model_retrying")
        _emit_subagent_model_retry(
            self.inputs.event_sink,
            self.inputs.child_run_id,
            self.inputs.task,
            model_calls=state.model_calls,
        )

    def _record_native_turn(
        self,
        state: _NativeLoopState,
        turn: ModelTurn,
    ) -> BoundedResult | None:
        state.messages.append(
            AgentMessage(
                id=f"{self.inputs.child_run_id}:assistant:{state.round_index}",
                role="assistant",
                content=turn.content,
                tool_calls=turn.tool_calls,
            )
        )
        state.pending_calls = list(turn.tool_calls)
        phase = "model_completed" if state.pending_calls else "output_ready"
        self._checkpoint_native(
            state,
            phase=phase,
            raw_text=turn.content if not state.pending_calls else "",
        )
        if state.pending_calls:
            return None
        return (
            turn.content,
            state.tool_trace,
            state.model_calls,
            self._native_diagnostics("native_tool_loop_uses_local_validation"),
        )

    def _validate_tool_calls(
        self,
        state: _NativeLoopState,
        tool_executor: ToolExecutor,
    ) -> list[ToolCall]:
        allowed = set(self.inputs.spec.tool_allowlist)
        accepted: list[ToolCall] = []
        for call in state.pending_calls:
            if call.name not in allowed:
                raise RuntimeError(f"subagent requested tool outside allowlist: {call.name}")
            if not tool_executor.registry.get(call.name).read_only:
                raise RuntimeError(f"subagent requested non-read-only tool: {call.name}")
            accepted.append(call)
        task_tool_limit = _minimum_optional(
            self.inputs.spec.max_tool_calls,
            self.inputs.task.budget.max_tool_calls,
        )
        if task_tool_limit is not None and state.tool_calls + len(accepted) > task_tool_limit:
            raise RuntimeError("subagent tool call limit exceeded")
        if self.inputs.quota is not None:
            self.inputs.quota.reserve_tool_calls(len(accepted))
        if self.inputs.task_quota is not None:
            self.inputs.task_quota.reserve_tool_calls(len(accepted))
        state.tool_calls += len(accepted)
        return accepted

    def _execute_tool_round(
        self,
        state: _NativeLoopState,
        accepted: list[ToolCall],
        tool_executor: ToolExecutor,
        child_context: ToolExecutionContext,
    ) -> None:
        self._raise_if_canceled()
        _emit_subagent_tools(
            self.inputs.event_sink,
            self.inputs.child_run_id,
            self.inputs.task,
            accepted,
            status="started",
        )
        for result in tool_executor.execute_many(accepted, child_context):
            state.tool_trace.append(self._tool_trace(result))
            state.messages.append(
                AgentMessage(
                    id=f"{self.inputs.child_run_id}:tool:{result.tool_call_id}",
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
            self.inputs.event_sink,
            self.inputs.child_run_id,
            self.inputs.task,
            accepted,
            status="completed",
            trace=state.tool_trace,
        )
        state.pending_calls = []
        state.round_index += 1
        self._checkpoint_native(state, phase="tools_completed")

    @staticmethod
    def _tool_trace(result: Any) -> dict[str, Any]:
        return {
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

    def _checkpoint_native(
        self,
        state: _NativeLoopState,
        *,
        phase: str,
        raw_text: str = "",
    ) -> None:
        self.owner._checkpoint_child(
            self.inputs.child_run_id,
            phase=phase,
            state=_execution_checkpoint(
                task=self.inputs.task,
                spec=self.inputs.spec,
                phase=phase,
                round_index=state.round_index,
                messages=state.messages,
                pending_calls=state.pending_calls,
                tool_trace=state.tool_trace,
                model_calls=state.model_calls,
                tool_calls=state.tool_calls,
                empty_response_retries=state.empty_response_retries,
                raw_text=raw_text,
            ),
        )

    def _step_context(self, state: _NativeLoopState) -> ModelStepContext:
        return ModelStepContext(
            run_id=self.inputs.child_run_id,
            user_id=self.inputs.context.user_id,
            thread_id=self.inputs.context.thread_id,
            turn_id=self.inputs.context.turn_id,
            step_index=state.model_calls,
            message_count=len(state.messages),
            observation_count=len(state.tool_trace),
            tool_result_count=len(state.tool_trace),
            available_roles=(
                str(
                    getattr(self.inputs.provider, "model_role", "")
                    or self.inputs.spec.model_role
                    or "reasoning"
                ),
            ),
            deadline_monotonic=self.inputs.deadline_monotonic,
        )

    def _consume_model_budget(self, step_index: int) -> None:
        _consume_model_budget(
            self.inputs.budget_ledger,
            root_run_id=self.inputs.root_run_id,
            child_run_id=self.inputs.child_run_id,
            task=self.inputs.task,
            model_call_limit=self.inputs.model_call_limit,
            step_index=step_index,
            quota=self.inputs.quota,
            task_quota=self.inputs.task_quota,
        )

    def _raise_if_canceled(self) -> None:
        self.owner._raise_if_canceled(
            self.inputs.child_run_id,
            self.inputs.parent_run_id,
            context=self.inputs.context,
            task=self.inputs.task,
        )

    def _remaining_timeout(self) -> int:
        return remaining_timeout_ceiling(
            self.inputs.deadline_monotonic,
            maximum=self.owner._task_timeout_seconds(self.inputs.task, self.inputs.spec),
        )

    def _max_tokens(self) -> int:
        return self.owner._task_max_tokens(self.inputs.task, self.inputs.spec)

    @staticmethod
    def _native_diagnostics(reason: str) -> dict[str, Any]:
        return {
            "requested_format": "json_schema",
            "effective_format": "prompt_only",
            "capability_source": "native_tool_loop",
            "degraded": True,
            "degradation_reason": reason,
        }

    def _exhausted_result(self, state: _NativeLoopState) -> BoundedResult:
        return (
            "",
            state.tool_trace,
            state.model_calls,
            self._native_diagnostics("native_tool_loop_exhausted"),
        )
