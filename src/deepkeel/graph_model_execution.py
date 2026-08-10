from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from deepkeel.contracts import AgentMessage, FinalAnswer, PendingAction, ToolCall
from deepkeel.events import AgentEventPersistenceError
from deepkeel.graph_model_step import (
    build_model_metrics,
    build_model_step_context,
    partition_model_tool_calls,
)
from deepkeel.graph_model_support import (
    _emit_hook_audits,
    _forced_tool_clarification_fallback,
    _latest_user_question,
    _model_hook_invocation,
)
from deepkeel.graph_state import (
    _allowed_tool_names,
    _copy_state,
    _forced_workflow_tool_name,
    _messages,
    _model_available_roles,
    _skill_tool_parameter_overrides,
    _stable_tool_calls,
)
from deepkeel.graph_workflow import (
    TRUNCATED_FINISH_REASONS,
    _answer_summary,
    _complete_continued_answer,
    _continue_or_fail_truncated_model_response,
    _emit,
    _finish_failed,
    _latency_ms,
    _repair_or_fail_workflow,
    _retry_or_fail_empty_model_response,
    _set_policy_state,
    _wait_for_workflow_input,
    _workflow_can_wait_for_user_input,
)
from deepkeel.hooks import HookAction, HookPoint
from deepkeel.model import ModelGateway, ModelTurn, model_tools_from_registry
from deepkeel.model_failures import ModelToolContractError
from deepkeel.planning.runtime import complete_plan_for_answer
from deepkeel.skill_activation import EntryToolActivationRequest
from deepkeel.skills import SkillPolicy
from deepkeel.tool_disclosure import ToolView, resolve_tool_view
from deepkeel.tool_registry import ToolRegistry
from deepkeel.turn_context import TurnExecutionContext
from deepkeel.type_narrowing import as_dict
from deepkeel.workflow_policy import evaluate_workflow_completion


@dataclass(slots=True)
class _InvocationTelemetry:
    started_at: float = field(default_factory=time.perf_counter)
    first_delta_at: float | None = None
    delta_index: int = 0
    delta_chars: int = 0
    route_payload: dict[str, Any] = field(default_factory=dict)
    routed_attempts: set[tuple[int, str]] = field(default_factory=set)


class ModelNodeExecution:
    """One typed model phase within the reusable ReAct graph."""

    def __init__(self, owner: Any, state: dict[str, Any], config: RunnableConfig) -> None:
        self.owner = owner
        self.config = config
        normalized = owner.normalize_state(state, config)
        if not normalized["messages"]:
            raise ValueError("graph checkpoint cannot resume model execution without messages")
        self.turn_context: TurnExecutionContext | None = owner.turn_context(config, normalized)
        model: ModelGateway | None = (
            self.turn_context.model if self.turn_context is not None else owner.model
        )
        if model is None:
            raise RuntimeError("a model gateway is required in TurnExecutionContext")
        self.model = model
        self.tool_registry: ToolRegistry = owner.tool_registry
        self.prompt = self.turn_context.system_prompt if self.turn_context else owner.prompt
        self.max_steps: int = owner.max_steps
        self.ledger = owner.ledger
        self.deadline_monotonic = (
            self.turn_context.deadline_monotonic
            if self.turn_context is not None
            else owner.deadline_monotonic
        )
        owner.ensure_active(normalized, config, force=True)
        self.current = _copy_state(normalized)
        self.skill_policy = SkillPolicy.from_snapshot(self.current.get("skill_activation"))
        self.workflow_is_finalizing = False
        self.tool_view: ToolView | None = None
        self.tools: list[dict[str, Any]] = []
        self.forced_tool_name = ""

    async def run(self) -> dict[str, Any]:
        terminal = self._prepare_step()
        if terminal is not None:
            return terminal
        terminal = await self._before_model_hook()
        if terminal is not None:
            return terminal
        turn, telemetry = await self._invoke_model()
        await self._after_model_hook(turn)
        model_metrics, tool_calls = self._record_model_turn(turn, telemetry)
        tool_calls, terminal = self._activate_entry_tool_skill(tool_calls)
        if terminal is not None:
            return terminal
        self._append_assistant(turn, model_metrics, tool_calls)
        return await self._settle_turn(turn, tool_calls)

    def _prepare_step(self) -> dict[str, Any] | None:
        if int(self.current.get("step_count") or 0) >= self.max_steps:
            completion = evaluate_workflow_completion(self.skill_policy, self.current)
            if (
                self.skill_policy.active
                and self.skill_policy.durable
                and not completion.allowed
                and str(self.current.get("policy_phase") or "") == "repair"
            ):
                return _repair_or_fail_workflow(
                    self.current, self.skill_policy, completion, self.config
                )
            return _finish_failed(
                self.current,
                "The agent reached the maximum step count without a reliable answer.",
                self.config,
            )
        self.current["status"] = "reasoning"
        _emit(
            self.current,
            self.config,
            "agent.reasoning",
            "Reasoning",
            "The agent is evaluating context and selecting the next step.",
        )
        self._resolve_tools()
        return None

    def _resolve_tools(self) -> None:
        allowed_tools = _allowed_tool_names(self.current, self.tool_registry)
        completion = evaluate_workflow_completion(self.skill_policy, self.current)
        self.workflow_is_finalizing = bool(
            self.skill_policy.active and self.skill_policy.durable and completion.allowed
        )
        suppressed = sorted(allowed_tools or ()) if self.workflow_is_finalizing else []
        if self.workflow_is_finalizing:
            allowed_tools = set()
        self.tool_view = resolve_tool_view(
            registry=self.tool_registry,
            allowed_names=allowed_tools,
            skill=self.skill_policy,
            mode=self.turn_context.tool_view_mode if self.turn_context else "legacy",
            discovered_names={
                str(name)
                for name in as_dict(self.current.get("metadata")).get("discovered_tool_names", [])
                if str(name).strip()
            },
        )
        self._record_tool_view(suppressed)
        self.tools = model_tools_from_registry(
            self.tool_registry,
            allowed_names=set(self.tool_view.exposed_names),
            parameter_overrides=_skill_tool_parameter_overrides(self.current, self.tool_registry),
        )
        self.forced_tool_name = _forced_workflow_tool_name(self.current, self.tools)

    def _record_tool_view(self, suppressed: list[str]) -> None:
        if self.tool_view is None:
            return
        metadata = self.current.setdefault("metadata", {})
        if self.workflow_is_finalizing:
            finalization = as_dict(metadata.get("workflow_finalization"))
            first_injection = not finalization.get("instruction_injected")
            if first_injection:
                self.current.setdefault("messages", []).append(
                    AgentMessage(
                        id=f"workflow-finalization-{uuid4()}",
                        role="system",
                        content=(
                            "The active workflow contract is fully satisfied. "
                            "Do not call or discover any more tools. Return the final "
                            "user-facing answer using the completed observations and artifacts."
                        ),
                        metadata={"kind": "workflow_finalization_guard"},
                    ).model_dump(mode="json")
                )
                _emit(
                    self.current,
                    self.config,
                    "workflow.finalizing",
                    "Workflow result ready",
                    "The workflow contract is satisfied; the model must now finalize the answer.",
                    {"suppressed_tool_names": suppressed, "visible": False},
                )
            metadata["workflow_finalization"] = {
                **finalization,
                "contract_satisfied": True,
                "suppressed_tool_names": suppressed,
                "instruction_injected": True,
            }
        previous = metadata.get("tool_view")
        metadata["tool_view"] = self.tool_view.as_dict()
        if previous != metadata["tool_view"]:
            _emit(
                self.current,
                self.config,
                "tools.disclosure.resolved",
                "Tool view resolved",
                f"{len(self.tool_view.exposed_names)} tools exposed",
                {**self.tool_view.as_dict(), "visible": False},
            )

    async def _before_model_hook(self) -> dict[str, Any] | None:
        if self.turn_context is None or self.turn_context.hook_runner is None:
            return None
        result = await self.turn_context.hook_runner.arun(
            _model_hook_invocation(
                HookPoint.MODEL_BEFORE,
                self.current,
                self.turn_context,
                payload={
                    "message_count": len(self.current.get("messages") or []),
                    "tool_names": [
                        str(item.get("function", {}).get("name") or "")
                        for item in self.tools
                        if isinstance(item, dict)
                    ],
                    "system_prompt": self.prompt,
                },
            )
        )
        _emit_hook_audits(self.current, self.config, result.audits)
        if result.decision.action == HookAction.DENY:
            return _finish_failed(
                self.current,
                result.decision.reason or "model invocation denied by lifecycle hook",
                self.config,
            )
        if result.decision.action == HookAction.WAIT_FOR_CONFIRMATION:
            return self._wait_for_model_confirmation(result.decision)
        patch = dict(result.decision.model_input_patch)
        self.prompt = str(patch.get("system_prompt") or self.prompt)
        for item in patch.get("append_messages") or []:
            message = AgentMessage.model_validate(item)
            self.current.setdefault("messages", []).append(message.model_dump(mode="json"))
        return None

    def _wait_for_model_confirmation(self, decision: Any) -> dict[str, Any]:
        pending = PendingAction(
            id=f"hook-model:{self.current.get('run_id')}:{self.current.get('step_count')}",
            run_id=str(self.current.get("run_id") or ""),
            action_type="confirm_model_invocation",
            title=decision.confirmation_title or "Confirm continuation",
            prompt=(
                decision.confirmation_message
                or decision.reason
                or "Please confirm before the agent continues."
            ),
            payload={"source": "lifecycle_hook"},
        )
        self.current["pending_action"] = pending.model_dump(mode="json")
        self.current["status"] = "waiting_user"
        _emit(
            self.current,
            self.config,
            "agent.waiting_user",
            pending.title,
            pending.prompt,
            {"pending_action": self.current["pending_action"]},
        )
        return self.current

    async def _invoke_model(self) -> tuple[ModelTurn, _InvocationTelemetry]:
        telemetry = _InvocationTelemetry()

        def on_route(payload: dict[str, Any]) -> None:
            self._on_route(telemetry, payload)

        def on_delta(delta: str) -> None:
            self._on_delta(telemetry, delta)

        try:
            step_context = build_model_step_context(
                self.current,
                available_roles=_model_available_roles(self.model),
                forced_tool_name=self.forced_tool_name,
                deadline_monotonic=self.deadline_monotonic,
            )
            turn = await self._call_gateway(step_context, on_route, on_delta)
            self.owner.ensure_active(self.current, self.config, force=True)
            return turn, telemetry
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            self._record_model_failure(telemetry, exc)
            raise

    async def _call_gateway(
        self,
        step_context: Any,
        on_route: Any,
        on_delta: Any,
    ) -> ModelTurn:
        async_run_turn = getattr(self.model, "arun_turn", None)
        try:
            if callable(async_run_turn):
                return await async_run_turn(
                    _messages(self.current),
                    tools=self.tools,
                    system_prompt=self.prompt,
                    on_text_delta=None if self.forced_tool_name else on_delta,
                    step_context=step_context,
                    on_route=on_route,
                )
            return await asyncio.to_thread(
                self.model.run_turn,
                _messages(self.current),
                tools=self.tools,
                system_prompt=self.prompt,
                on_text_delta=None if self.forced_tool_name else on_delta,
                step_context=step_context,
                on_route=on_route,
            )
        except ModelToolContractError as exc:
            recovered = _forced_tool_clarification_fallback(
                self.tool_registry,
                self.forced_tool_name,
                exc,
                state=self.current,
                turn_context=self.turn_context,
            )
            if recovered is None:
                raise
            _emit(
                self.current,
                self.config,
                "model.tool_contract.recovered",
                "Required tool contract recovered",
                self.forced_tool_name,
                {
                    "tool_name": self.forced_tool_name,
                    "recovery": str(
                        as_dict(recovered.raw).get("recovery") or "forced_tool_clarification"
                    ),
                    "visible": False,
                },
            )
            return recovered

    def _on_route(self, telemetry: _InvocationTelemetry, payload: dict[str, Any]) -> None:
        telemetry.route_payload.update(payload)
        route_key = (
            int(payload.get("attempt_index") or 1),
            str(payload.get("retry_kind") or "primary"),
        )
        if route_key in telemetry.routed_attempts:
            _emit(
                self.current,
                self.config,
                "budget.usage.recorded",
                "Model usage recorded",
                str(as_dict(payload.get("usage")).get("source") or ""),
                {
                    "role": str(payload.get("role") or ""),
                    "model_id": str(payload.get("model_id") or ""),
                    "attempt_index": route_key[0],
                    "usage": as_dict(payload.get("usage")),
                    "budget_metrics": as_dict(payload.get("budget_metrics")),
                    "visible": False,
                },
            )
            return
        telemetry.routed_attempts.add(route_key)
        _emit(
            self.current,
            self.config,
            "model.route.selected",
            "Model route selected",
            str(payload.get("reason") or ""),
            {**payload, "visible": False},
        )

    def _on_delta(self, telemetry: _InvocationTelemetry, delta: str) -> None:
        self.owner.ensure_active(self.current, self.config)
        if telemetry.first_delta_at is None:
            telemetry.first_delta_at = time.perf_counter()
        telemetry.delta_chars += len(delta)
        _emit(
            self.current,
            self.config,
            "model.delta",
            "",
            "",
            {
                "delta": delta,
                "index": telemetry.delta_index,
                "stream_mode": "provider_stream",
            },
            ephemeral=True,
        )
        telemetry.delta_index += 1

    def _record_model_failure(
        self,
        telemetry: _InvocationTelemetry,
        exc: Exception,
    ) -> None:
        self.current["budget_state"] = self.ledger.snapshot(self._operational_run_id()).as_dict()
        payload = telemetry.route_payload
        if payload.get("budget_metrics"):
            _emit(
                self.current,
                self.config,
                "budget.usage.recorded",
                "Model usage recorded",
                str(as_dict(payload.get("usage")).get("source") or "estimated_failure"),
                {
                    "role": str(payload.get("role") or ""),
                    "model_id": str(payload.get("model_id") or ""),
                    "usage": as_dict(payload.get("usage")),
                    "budget_metrics": as_dict(payload.get("budget_metrics")),
                    "visible": False,
                },
            )
        _emit(
            self.current,
            self.config,
            "model.failed",
            "Model call failed",
            str(exc),
            {
                "model_id": str(payload.get("model_id") or ""),
                "model_role": str(payload.get("role") or ""),
                "latency_ms": int((time.perf_counter() - telemetry.started_at) * 1000),
                "first_token_latency_ms": _latency_ms(
                    telemetry.started_at, telemetry.first_delta_at
                ),
                "delta_count": telemetry.delta_index,
                "delta_chars": telemetry.delta_chars,
                "error_type": type(exc).__name__,
                "error_code": str(getattr(exc, "code", "") or ""),
                "route": dict(payload),
            },
        )

    async def _after_model_hook(self, turn: ModelTurn) -> None:
        if self.turn_context is None or self.turn_context.hook_runner is None:
            return
        result = await self.turn_context.hook_runner.arun(
            _model_hook_invocation(
                HookPoint.MODEL_AFTER,
                self.current,
                self.turn_context,
                payload={
                    "model_id": turn.model_id,
                    "model_role": turn.model_role,
                    "finish_reason": turn.finish_reason,
                    "content_chars": len(turn.content),
                    "tool_call_count": len(turn.tool_calls),
                },
            )
        )
        _emit_hook_audits(self.current, self.config, result.audits)

    def _record_model_turn(
        self,
        turn: ModelTurn,
        telemetry: _InvocationTelemetry,
    ) -> tuple[dict[str, Any], list[ToolCall]]:
        self.current["budget_state"] = self.ledger.snapshot(self._operational_run_id()).as_dict()
        metrics = build_model_metrics(
            turn,
            telemetry.route_payload,
            latency_ms=int((time.perf_counter() - telemetry.started_at) * 1000),
            first_token_latency_ms=_latency_ms(telemetry.started_at, telemetry.first_delta_at),
            delta_count=telemetry.delta_index,
            delta_chars=telemetry.delta_chars,
            forced_tool_name=self.forced_tool_name,
        )
        _emit(
            self.current,
            self.config,
            "budget.usage.recorded",
            "Model usage recorded",
            str(as_dict(metrics["usage"]).get("source") or ""),
            {
                "role": turn.model_role,
                "model_id": turn.model_id,
                "usage": metrics["usage"],
                "budget_metrics": metrics["budget_metrics"],
                "max_output_tokens": metrics["max_output_tokens"],
                "visible": False,
            },
        )
        self.current["step_count"] = int(self.current.get("step_count") or 0) + 1
        reported = _stable_tool_calls(
            turn.tool_calls,
            run_id=str(self.current.get("run_id") or ""),
            step_index=int(self.current["step_count"]),
        )
        exposed = set(self.tool_view.exposed_names if self.tool_view else ())
        calls, rejected = partition_model_tool_calls(
            reported,
            workflow_is_finalizing=self.workflow_is_finalizing,
            exposed_tool_names=exposed,
            registered_tool_names={spec.name for spec in self.tool_registry.list_tools()},
        )
        self._record_rejected_calls(rejected, exposed)
        return metrics, calls

    def _record_rejected_calls(
        self,
        rejected: list[ToolCall],
        exposed: set[str],
    ) -> None:
        if not rejected:
            return
        names = [call.name for call in rejected]
        self.current.setdefault("metadata", {}).setdefault(
            "rejected_undisclosed_tool_calls", []
        ).append({"step_index": int(self.current["step_count"]), "tool_names": names})
        _emit(
            self.current,
            self.config,
            "tools.disclosure.rejected",
            "Undisclosed tool call rejected",
            ", ".join(names),
            {"tool_names": names, "exposed_tool_names": sorted(exposed), "visible": False},
        )

    def _activate_entry_tool_skill(
        self,
        calls: list[ToolCall],
    ) -> tuple[list[ToolCall], dict[str, Any] | None]:
        if not calls or SkillPolicy.from_snapshot(self.current.get("skill_activation")).active:
            return calls, None
        context = self.turn_context
        activator = context.entry_tool_skill_activator if context is not None else None
        if activator is None or context is None:
            return calls, None
        request = EntryToolActivationRequest(
            tool_calls=tuple(calls),
            current_activation=dict(self.current.get("skill_activation") or {}),
            run_id=str(self.current.get("run_id") or ""),
            user_id=str(self.current.get("user_id") or ""),
            thread_id=str(self.current.get("thread_id") or ""),
            turn_id=str(self.current.get("turn_id") or ""),
            question=_latest_user_question(self.current.get("messages") or []),
            messages=tuple(
                item for item in self.current.get("messages") or [] if isinstance(item, dict)
            ),
            context_bundle=dict(
                context.tool_context.context_bundle
                if isinstance(context.tool_context.context_bundle, dict)
                else {}
            ),
        )
        decision = activator.activate(request)
        if decision is None:
            return calls, None
        activation = dict(decision.skill_activation)
        if not str(activation.get("skill_id") or "").strip():
            return calls, _finish_failed(
                self.current,
                "Entry-tool Skill activation returned an invalid snapshot.",
                self.config,
            )
        replacements = list(decision.tool_calls or tuple(calls))
        if {(call.id, call.name) for call in replacements} != {
            (call.id, call.name) for call in calls
        }:
            return calls, _finish_failed(
                self.current,
                "Entry-tool Skill activation changed the selected tool calls.",
                self.config,
            )
        self._record_entry_skill_activation(activation, decision.reason, replacements)
        return replacements, None

    def _record_entry_skill_activation(
        self,
        activation: dict[str, Any],
        reason: str,
        calls: list[ToolCall],
    ) -> None:
        if self.turn_context is None:
            return
        self.current["skill_activation"] = activation
        resolved_reason = str(reason or "entry_tool_selected")
        self.current.setdefault("metadata", {})["entry_tool_skill_activation"] = {
            "skill_id": str(activation.get("skill_id") or ""),
            "source": str(activation.get("source") or "model"),
            "reason": resolved_reason,
            "tool_names": [call.name for call in calls],
        }
        tool_metadata = self.turn_context.tool_context.metadata
        tool_metadata["skill_activation"] = activation
        tool_metadata["governance_scope"] = {
            **as_dict(tool_metadata.get("governance_scope")),
            "skill_id": str(activation.get("skill_id") or ""),
        }
        _emit(
            self.current,
            self.config,
            "skill.activated",
            "Skill activated",
            str(activation.get("label") or activation.get("skill_id") or ""),
            {
                "skill_id": str(activation.get("skill_id") or ""),
                "source": str(activation.get("source") or "model"),
                "reason": resolved_reason,
                "entry_tool_names": [call.name for call in calls],
                "visible": False,
            },
        )

    def _append_assistant(
        self,
        turn: ModelTurn,
        metrics: dict[str, Any],
        calls: list[ToolCall],
    ) -> None:
        assistant = AgentMessage(
            id=f"assistant-{uuid4()}",
            role="assistant",
            content=turn.content,
            tool_calls=calls,
            metadata={
                "model_id": turn.model_id,
                "model_role": turn.model_role,
                "finish_reason": turn.finish_reason,
                "runtime_metrics": metrics,
            },
        )
        self.current.setdefault("messages", []).append(assistant.model_dump(mode="json"))
        _emit(
            self.current,
            self.config,
            "model.completed",
            "Model response completed",
            turn.content[:160],
            metrics,
        )

    async def _settle_turn(
        self,
        turn: ModelTurn,
        calls: list[ToolCall],
    ) -> dict[str, Any]:
        if not calls and turn.finish_reason.strip().lower() in TRUNCATED_FINISH_REASONS:
            return _continue_or_fail_truncated_model_response(
                self.current,
                self.config,
                content=turn.content,
                finish_reason=turn.finish_reason,
                can_continue=int(self.current.get("step_count") or 0) < self.max_steps,
            )
        if not calls and not turn.content.strip():
            self.current["messages"].pop()
            return _retry_or_fail_empty_model_response(
                self.current,
                self.config,
                can_retry=int(self.current.get("step_count") or 0) < self.max_steps,
                answer_only=self.workflow_is_finalizing,
            )
        metadata = self.current.setdefault("metadata", {})
        metadata["consecutive_empty_model_responses"] = 0
        metadata.pop("empty_model_retry_pending", None)
        if calls:
            self.current["pending_tool_calls"] = [call.model_dump(mode="json") for call in calls]
            self.current["status"] = "executing_tools"
            return self.current
        return await self._finalize_answer(turn)

    async def _finalize_answer(self, turn: ModelTurn) -> dict[str, Any]:
        if not turn.content.strip():
            return _finish_failed(
                self.current,
                "The model returned no usable content for this step.",
                self.config,
            )
        skill = SkillPolicy.from_snapshot(self.current.get("skill_activation"))
        completion = evaluate_workflow_completion(skill, self.current)
        if not completion.allowed:
            if _workflow_can_wait_for_user_input(skill, completion, self.current):
                return _wait_for_workflow_input(
                    self.current, skill, completion, turn.content, self.config
                )
            return _repair_or_fail_workflow(self.current, skill, completion, self.config)
        if skill.active and skill.durable:
            _set_policy_state(self.current, phase="completed", decision=completion)
        content = _complete_continued_answer(self.current.setdefault("metadata", {}), turn.content)
        content, terminal = await self._before_answer_hook(turn, content)
        if terminal is not None:
            return terminal
        complete_plan_for_answer(
            self.current,
            emit=lambda event_type, title, summary, payload: _emit(
                self.current,
                self.config,
                event_type,
                title,
                summary,
                payload,
            ),
        )
        answer = FinalAnswer(
            markdown=content,
            summary=_answer_summary(content),
            model_id=turn.model_id,
            model_role=turn.model_role,
            stop_reason=turn.finish_reason or "completed",
            artifact_ids=[
                str(item.get("id") or "")
                for item in self.current.get("artifacts", [])
                if item.get("id")
            ],
        )
        self.current["final_answer"] = answer.model_dump(mode="json")
        self.current["status"] = "completed"
        self.current["pending_tool_calls"] = []
        _emit(
            self.current,
            self.config,
            "answer.completed",
            "Answer completed",
            answer.summary,
            {"final_answer": self.current["final_answer"]},
        )
        return self.current

    async def _before_answer_hook(
        self,
        turn: ModelTurn,
        content: str,
    ) -> tuple[str, dict[str, Any] | None]:
        if self.turn_context is None or self.turn_context.hook_runner is None:
            return content, None
        result = await self.turn_context.hook_runner.arun(
            _model_hook_invocation(
                HookPoint.ANSWER_BEFORE_FINALIZE,
                self.current,
                self.turn_context,
                payload={
                    "markdown": content,
                    "model_id": turn.model_id,
                    "artifact_ids": [
                        str(item.get("id") or "")
                        for item in self.current.get("artifacts", [])
                        if item.get("id")
                    ],
                },
            )
        )
        _emit_hook_audits(self.current, self.config, result.audits)
        if result.decision.action == HookAction.DENY:
            return content, _finish_failed(
                self.current,
                result.decision.reason or "answer finalization denied by lifecycle hook",
                self.config,
            )
        return str(result.decision.model_input_patch.get("answer_markdown") or content), None

    def _operational_run_id(self) -> str:
        metadata = as_dict(self.current.get("metadata"))
        return str(metadata.get("operational_run_id") or self.current.get("run_id") or "")
