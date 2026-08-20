from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from deepkeel.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from deepkeel.subagents.contracts import (
    DelegationTask,
    SubAgentArtifactRef,
    SubAgentContextRef,
    SubAgentInputRequest,
    SubAgentResult,
    SubAgentSpec,
)
from deepkeel.subagents.execution_support import (
    _consume_model_budget,
    _default_system_prompt,
    _invoke_provider,
    _minimum_optional,
    _repair_prompt,
    _resolve_role,
    _task_prompt,
)
from deepkeel.subagents.execution_types import (
    EventSink,
    SubAgentCanceledError,
    SubAgentOutputError,
    _DelegationQuota,
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
from deepkeel.tools import ToolExecutionContext


@dataclass(slots=True)
class _PreparedTask:
    role: str
    provider: Any
    prompt: str
    system_prompt: str
    schema: dict[str, Any]
    resume_state: dict[str, Any] | None
    deadline_monotonic: float
    task_quota: _DelegationQuota
    model_call_limit: int | None


@dataclass(slots=True)
class _TaskOutput:
    raw: str
    tool_trace: list[dict[str, Any]]
    model_calls: int
    structured_output: dict[str, Any]


@dataclass(slots=True)
class _ValidatedOutput:
    parsed: dict[str, Any]
    raw: str
    model_calls: int
    outcome: Literal["completed", "degraded"] = "completed"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SubAgentTaskExecution:
    """Coordinates preparation, bounded execution, repair and result projection."""

    def __init__(
        self,
        owner: Any,
        task: DelegationTask,
        *,
        spec: SubAgentSpec,
        child_run_id: str,
        providers: dict[str, Any],
        root_run_id: str,
        parent_run_id: str,
        context: ToolExecutionContext,
        event_sink: EventSink | None,
        budget_ledger: Any,
        model_call_limit: float | None,
        quota: _DelegationQuota | None,
    ) -> None:
        self.owner = owner
        self.task = task
        self.spec = spec
        self.child_run_id = child_run_id
        self.providers = providers
        self.root_run_id = root_run_id
        self.parent_run_id = parent_run_id
        self.context = context
        self.event_sink = event_sink
        self.budget_ledger = budget_ledger
        self.model_call_limit = model_call_limit
        self.quota = quota
        self.started_at = time.perf_counter()

    def run(self) -> SubAgentResult:
        canceled = self._initial_canceled_result()
        if canceled is not None:
            return canceled
        prepared = self._prepare()
        output = self._execute(prepared)
        self._raise_if_canceled()
        validated = self._validate_or_repair(prepared, output)
        return self._project_result(prepared, output, validated)

    def _initial_canceled_result(self) -> SubAgentResult | None:
        try:
            self._raise_if_canceled()
        except SubAgentCanceledError:
            return SubAgentResult(
                task_id=self.task.id,
                agent_id=self.task.agent_id,
                child_run_id=self.child_run_id,
                status="canceled",
                idempotency_key=self.task.effective_idempotency_key,
                lineage=self.task.lineage,
                error="parent agent run is terminal",
                metadata={"late_result_discarded": True},
            )
        return None

    def _prepare(self) -> _PreparedTask:
        _validate_input(self.task.input_data, self.spec.input_contract)
        role = _resolve_role(self.task.model_role, self.spec.model_role, self.providers)
        provider = self.providers.get(role)
        if provider is None:
            raise RuntimeError(f"subagent model provider is unavailable for role {role}")
        resume_state = self.owner._load_child_checkpoint(self.child_run_id)
        return _PreparedTask(
            role=role,
            provider=provider,
            prompt=_task_prompt(self.task, self.spec),
            system_prompt=self.spec.system_prompt or _default_system_prompt(self.spec),
            schema=_output_schema(self.spec),
            resume_state=resume_state,
            deadline_monotonic=self.owner._task_deadline_monotonic(
                self.context,
                self.task,
                self.spec,
            ),
            task_quota=_DelegationQuota(
                max_model_calls=_minimum_optional(
                    self.spec.max_model_calls,
                    self.task.budget.max_model_calls,
                ),
                max_tool_calls=_minimum_optional(
                    self.spec.max_tool_calls,
                    self.task.budget.max_tool_calls,
                ),
                model_calls=int((resume_state or {}).get("model_calls") or 0),
                tool_calls=int((resume_state or {}).get("tool_calls") or 0),
            ),
            model_call_limit=_minimum_optional(
                self.model_call_limit,
                self.spec.max_model_calls,
                self.task.budget.max_model_calls,
            ),
        )

    def _execute(self, prepared: _PreparedTask) -> _TaskOutput:
        raw, tool_trace, model_calls, structured_output = self.owner._run_bounded_agent(
            self.task,
            spec=self.spec,
            provider=prepared.provider,
            child_run_id=self.child_run_id,
            context=self.context,
            event_sink=self.event_sink,
            system_prompt=prepared.system_prompt,
            prompt=prepared.prompt,
            output_schema=prepared.schema,
            root_run_id=self.root_run_id,
            budget_ledger=self.budget_ledger,
            model_call_limit=prepared.model_call_limit,
            parent_run_id=self.parent_run_id,
            resume_state=prepared.resume_state,
            quota=self.quota,
            task_quota=prepared.task_quota,
            deadline_monotonic=prepared.deadline_monotonic,
        )
        return _TaskOutput(raw, tool_trace, model_calls, structured_output)

    def _validate_or_repair(
        self,
        prepared: _PreparedTask,
        output: _TaskOutput,
    ) -> _ValidatedOutput:
        try:
            parsed = _validated_json(output.raw, prepared.schema)
            return _ValidatedOutput(parsed=parsed, raw=output.raw, model_calls=output.model_calls)
        except RuntimeError as first_error:
            if str((prepared.resume_state or {}).get("phase") or "") == "repair_completed":
                return self._recover_from_text(output, first_error, recovered=True)
            return self._repair(prepared, output, first_error)

    def _repair(
        self,
        prepared: _PreparedTask,
        output: _TaskOutput,
        first_error: RuntimeError,
    ) -> _ValidatedOutput:
        self._raise_if_canceled()
        _consume_model_budget(
            self.budget_ledger,
            root_run_id=self.root_run_id,
            child_run_id=self.child_run_id,
            task=self.task,
            model_call_limit=prepared.model_call_limit,
            step_index=output.model_calls,
            quota=self.quota,
            task_quota=prepared.task_quota,
        )
        invocation = _invoke_provider(
            prepared.provider,
            prepared.system_prompt,
            _repair_prompt(
                prepared.prompt,
                output.raw,
                prepared.schema,
                str(first_error),
                output.tool_trace,
            ),
            timeout_seconds=remaining_timeout_ceiling(
                prepared.deadline_monotonic,
                maximum=self.owner._task_timeout_seconds(self.task, self.spec),
            ),
            max_tokens=self.owner._task_max_tokens(self.task, self.spec),
            output_schema=prepared.schema,
            capability_registry=self.owner.model_capabilities,
        )
        repaired = invocation.text
        output.structured_output["repair"] = invocation.diagnostics()
        self._raise_if_canceled()
        ensure_time_remaining(prepared.deadline_monotonic)
        model_calls = output.model_calls + 1
        self._checkpoint_repair(repaired, output, model_calls)
        try:
            parsed = _validated_json(repaired, prepared.schema)
            return _ValidatedOutput(parsed=parsed, raw=repaired, model_calls=model_calls)
        except RuntimeError as repair_error:
            return self._recover_from_text(
                _TaskOutput(
                    raw=repaired or output.raw,
                    tool_trace=output.tool_trace,
                    model_calls=model_calls,
                    structured_output=output.structured_output,
                ),
                repair_error,
                first_error=first_error,
            )

    def _recover_from_text(
        self,
        output: _TaskOutput,
        error: RuntimeError,
        *,
        first_error: RuntimeError | None = None,
        recovered: bool = False,
    ) -> _ValidatedOutput:
        fallback = _fallback_subagent_output(output.raw)
        if fallback is None:
            diagnostics = {
                "repair_error": str(error),
                "tool_trace": output.tool_trace,
                "model_calls": output.model_calls,
                **({"initial_error": str(first_error)} if first_error is not None else {}),
                **({"recovered_from_checkpoint": True} if recovered else {}),
            }
            raise SubAgentOutputError(
                "subagent returned invalid JSON after schema repair",
                raw_text=output.raw,
                diagnostics=diagnostics,
            ) from error
        diagnostics = {
            "reason_code": "structured_output_recovered_from_text",
            "repair_error": str(error),
            **({"initial_error": str(first_error)} if first_error is not None else {}),
            **({"recovered_from_checkpoint": True} if recovered else {}),
        }
        return _ValidatedOutput(
            parsed=fallback,
            raw=output.raw,
            model_calls=output.model_calls,
            outcome="degraded",
            diagnostics=diagnostics,
        )

    def _checkpoint_repair(
        self,
        repaired: str,
        output: _TaskOutput,
        model_calls: int,
    ) -> None:
        self.owner._checkpoint_child(
            self.child_run_id,
            phase="repair_completed",
            state={
                "schema_version": "subagent-execution-v1",
                "task_id": self.task.id,
                "idempotency_key": self.task.effective_idempotency_key,
                "lineage": self.task.lineage.model_dump(mode="json"),
                "spec_version": self.spec.version,
                "phase": "repair_completed",
                "raw_text": repaired,
                "tool_trace": output.tool_trace,
                "model_calls": model_calls,
                "tool_calls": len(output.tool_trace),
                "structured_output": output.structured_output,
            },
        )

    def _project_result(
        self,
        prepared: _PreparedTask,
        output: _TaskOutput,
        validated: _ValidatedOutput,
    ) -> SubAgentResult:
        parsed = validated.parsed
        conclusion = str(parsed.get("conclusion") or parsed.get("summary") or validated.raw).strip()
        if str(parsed.get("status") or "") == "needs_input":
            return self._needs_input_result(prepared, output, validated, conclusion)
        if not conclusion:
            raise RuntimeError("subagent returned an empty conclusion")
        return self._completed_result(prepared, output, validated, conclusion)

    def _needs_input_result(
        self,
        prepared: _PreparedTask,
        output: _TaskOutput,
        validated: _ValidatedOutput,
        conclusion: str,
    ) -> SubAgentResult:
        request_payload = validated.parsed.get("input_request")
        request_payload = request_payload if isinstance(request_payload, dict) else {}
        input_request = SubAgentInputRequest.model_validate(
            {**request_payload, "prompt": str(request_payload.get("prompt") or conclusion).strip()}
        )
        return SubAgentResult(
            task_id=self.task.id,
            agent_id=self.task.agent_id,
            child_run_id=self.child_run_id,
            status="needs_input",
            outcome="needs_input",
            conclusion=conclusion,
            input_request=input_request,
            context_refs=list(self.task.context_refs),
            artifact_refs=list(self.task.artifact_refs),
            idempotency_key=self.task.effective_idempotency_key,
            lineage=self.task.lineage,
            output=validated.parsed,
            model_role=prepared.role,
            model_id=str(getattr(prepared.provider, "model", "") or ""),
            duration_ms=self._duration_ms(),
            raw_text=validated.raw,
            metadata=self._base_metadata(output, validated.model_calls),
        )

    def _completed_result(
        self,
        prepared: _PreparedTask,
        output: _TaskOutput,
        validated: _ValidatedOutput,
        conclusion: str,
    ) -> SubAgentResult:
        parsed = validated.parsed
        return SubAgentResult(
            task_id=self.task.id,
            agent_id=self.task.agent_id,
            child_run_id=self.child_run_id,
            status="completed",
            outcome=validated.outcome,
            conclusion=conclusion,
            evidence=_string_list(parsed.get("evidence")),
            evidence_refs=_dict_list(parsed.get("evidence_refs")),
            context_refs=self._context_refs(parsed),
            artifact_refs=self._artifact_refs(parsed, output.tool_trace),
            risks=_string_list(parsed.get("risks")),
            recommendations=_string_list(parsed.get("recommendations")),
            claims=_dict_list(parsed.get("claims")),
            warnings=_string_list(parsed.get("warnings")),
            confidence=_confidence(parsed.get("confidence")),
            abstained=bool(parsed.get("abstained", False)),
            idempotency_key=self.task.effective_idempotency_key,
            lineage=self.task.lineage,
            output=parsed,
            model_role=prepared.role,
            model_id=str(getattr(prepared.provider, "model", "") or ""),
            duration_ms=self._duration_ms(),
            raw_text=validated.raw,
            metadata={
                **self._base_metadata(output, validated.model_calls),
                "output_contract": dict(self.spec.output_contract),
                "read_only": self.spec.read_only,
                "output_outcome": validated.outcome,
                "output_diagnostics": validated.diagnostics,
            },
        )

    def _base_metadata(self, output: _TaskOutput, model_calls: int) -> dict[str, Any]:
        return {
            "spec_version": self.spec.version,
            "model_calls": model_calls,
            "tool_trace": output.tool_trace,
            "structured_output": output.structured_output,
        }

    def _artifact_refs(
        self,
        parsed: dict[str, Any],
        tool_trace: list[dict[str, Any]],
    ) -> list[SubAgentArtifactRef]:
        refs = list(self.task.artifact_refs)
        refs.extend(
            SubAgentArtifactRef.model_validate(item)
            for trace in tool_trace
            for item in _dict_list(trace.get("artifact_refs"))
        )
        refs.extend(
            SubAgentArtifactRef.model_validate(item)
            for item in _dict_list(parsed.get("artifact_refs"))
        )
        return refs

    def _context_refs(self, parsed: dict[str, Any]) -> list[SubAgentContextRef]:
        refs = list(self.task.context_refs)
        refs.extend(
            SubAgentContextRef.model_validate(item)
            for item in _dict_list(parsed.get("context_refs"))
        )
        return refs

    def _raise_if_canceled(self) -> None:
        self.owner._raise_if_canceled(
            self.child_run_id,
            self.parent_run_id,
            context=self.context,
            task=self.task,
        )

    def _duration_ms(self) -> int:
        return round((time.perf_counter() - self.started_at) * 1000)
