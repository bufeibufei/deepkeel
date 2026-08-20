from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from deepkeel.contracts import AgentMessage, PendingAction, ToolCall
from deepkeel.graph_workflow import _emit, _finish_failed
from deepkeel.guardrails import GuardrailAction, GuardrailDecision, GuardrailStage
from deepkeel.model import ModelTurn
from deepkeel.model_guardrails import emit_model_guardrail_audits, model_guardrail_request
from deepkeel.turn_context import TurnExecutionContext


@dataclass(slots=True)
class ModelInvocationTelemetry:
    started_at: float = field(default_factory=time.perf_counter)
    first_delta_at: float | None = None
    delta_index: int = 0
    delta_chars: int = 0
    route_payload: dict[str, Any] = field(default_factory=dict)
    routed_attempts: set[tuple[int, str]] = field(default_factory=set)
    guardrail_buffered: bool = False

    def record_delta(
        self,
        *,
        owner: Any,
        state: dict[str, Any],
        config: RunnableConfig,
        delta: str,
        emit: bool,
    ) -> None:
        owner.ensure_active(state, config)
        if self.first_delta_at is None:
            self.first_delta_at = time.perf_counter()
        self.delta_chars += len(delta)
        if emit:
            _emit(
                state,
                config,
                "model.delta",
                "",
                "",
                {
                    "delta": delta,
                    "index": self.delta_index,
                    "stream_mode": "provider_stream",
                },
                ephemeral=True,
            )
        self.delta_index += 1


@dataclass(frozen=True, slots=True)
class GuardedModelInput:
    prompt: str
    tools: list[dict[str, Any]]
    terminal: dict[str, Any] | None = None


class ModelGuardrailCoordinator:
    """Keeps trust decisions outside the core model-step orchestration."""

    def __init__(
        self,
        state: dict[str, Any],
        config: RunnableConfig,
        turn_context: TurnExecutionContext | None,
    ) -> None:
        self.state = state
        self.config = config
        self.turn_context = turn_context

    @property
    def buffers_model_output(self) -> bool:
        runner = self._runner()
        return runner is not None and runner.has_stage(GuardrailStage.MODEL_OUTPUT)

    async def guard_input(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
    ) -> GuardedModelInput:
        turn_context = self.turn_context
        runner = self._runner()
        if turn_context is None or runner is None or not runner.has_stage(GuardrailStage.MODEL_INPUT):
            return GuardedModelInput(prompt=prompt, tools=tools)
        result = await runner.arun(
            model_guardrail_request(
                GuardrailStage.MODEL_INPUT,
                self.state,
                turn_context,
                payload={
                    "system_prompt": prompt,
                    "messages": list(self.state.get("messages") or []),
                    "tools": list(tools),
                },
            )
        )
        emit_model_guardrail_audits(self.state, self.config, result.audits)
        decision = result.decision
        terminal = self._terminal(decision, GuardrailStage.MODEL_INPUT)
        if terminal is not None:
            return GuardedModelInput(prompt=prompt, tools=tools, terminal=terminal)
        patch = dict(decision.payload_patch)
        guarded_prompt = str(patch.get("system_prompt") or prompt)
        if isinstance(patch.get("messages"), list):
            self.state["messages"] = [
                AgentMessage.model_validate(item).model_dump(mode="json")
                for item in patch["messages"]
            ]
        for item in patch.get("append_messages") or []:
            message = AgentMessage.model_validate(item)
            self.state.setdefault("messages", []).append(message.model_dump(mode="json"))
        guarded_tools = tools
        if isinstance(patch.get("tools"), list):
            guarded_tools = [dict(item) for item in patch["tools"] if isinstance(item, dict)]
        return GuardedModelInput(prompt=guarded_prompt, tools=guarded_tools)

    async def guard_output(
        self,
        turn: ModelTurn,
    ) -> tuple[ModelTurn, dict[str, Any] | None]:
        turn_context = self.turn_context
        runner = self._runner()
        if turn_context is None or runner is None or not runner.has_stage(GuardrailStage.MODEL_OUTPUT):
            return turn, None
        result = await runner.arun(
            model_guardrail_request(
                GuardrailStage.MODEL_OUTPUT,
                self.state,
                turn_context,
                payload={
                    "content": turn.content,
                    "tool_calls": [call.model_dump(mode="json") for call in turn.tool_calls],
                    "finish_reason": turn.finish_reason,
                    "model_id": turn.model_id,
                    "model_role": turn.model_role,
                },
            )
        )
        emit_model_guardrail_audits(self.state, self.config, result.audits)
        terminal = self._terminal(result.decision, GuardrailStage.MODEL_OUTPUT)
        if terminal is not None:
            return turn, terminal
        patch = dict(result.decision.payload_patch)
        tool_calls = turn.tool_calls
        if isinstance(patch.get("tool_calls"), list):
            tool_calls = [ToolCall.model_validate(item) for item in patch["tool_calls"]]
        return turn.model_copy(
            update={
                "content": str(patch.get("content", turn.content)),
                "tool_calls": tool_calls,
                "finish_reason": str(patch.get("finish_reason", turn.finish_reason)),
            }
        ), None

    async def guard_final_answer(
        self,
        turn: ModelTurn,
        content: str,
    ) -> tuple[str, dict[str, Any] | None]:
        turn_context = self.turn_context
        runner = self._runner()
        if turn_context is None or runner is None or not runner.has_stage(GuardrailStage.FINAL_OUTPUT):
            return content, None
        result = await runner.arun(
            model_guardrail_request(
                GuardrailStage.FINAL_OUTPUT,
                self.state,
                turn_context,
                payload={
                    "markdown": content,
                    "model_id": turn.model_id,
                    "model_role": turn.model_role,
                    "artifact_ids": [
                        str(item.get("id") or "")
                        for item in self.state.get("artifacts", [])
                        if item.get("id")
                    ],
                },
                operation_suffix=":final",
            )
        )
        emit_model_guardrail_audits(self.state, self.config, result.audits)
        terminal = self._terminal(result.decision, GuardrailStage.FINAL_OUTPUT)
        if terminal is not None:
            return content, terminal
        return str(result.decision.payload_patch.get("markdown") or content), None

    def release_buffered_stream(self, turn: ModelTurn, *, buffered: bool) -> None:
        if not buffered or not turn.content:
            return
        for index in range(0, len(turn.content), 96):
            _emit(
                self.state,
                self.config,
                "model.delta",
                "",
                "",
                {
                    "delta": turn.content[index : index + 96],
                    "index": index // 96,
                    "stream_mode": "guardrail_buffered",
                },
                ephemeral=True,
            )

    def _runner(self):
        return self.turn_context.guardrail_runner if self.turn_context is not None else None

    def _terminal(
        self,
        decision: GuardrailDecision,
        stage: GuardrailStage,
    ) -> dict[str, Any] | None:
        if decision.action == GuardrailAction.BLOCK:
            return _finish_failed(
                self.state,
                decision.reason or f"{stage.value} blocked by runtime guardrail",
                self.config,
            )
        if decision.action != GuardrailAction.REQUIRE_APPROVAL:
            return None
        pending = PendingAction(
            id=(
                f"guardrail-model:{stage.value}:{self.state.get('run_id')}:"
                f"{self.state.get('step_count')}"
            ),
            run_id=str(self.state.get("run_id") or ""),
            action_type="confirm_guarded_operation",
            title=decision.approval_title or "Confirm protected operation",
            prompt=(
                decision.approval_prompt
                or decision.reason
                or "Please confirm before the protected operation continues."
            ),
            payload={
                "source": "guardrail",
                "stage": stage.value,
                "guardrail_code": decision.code,
            },
        )
        self.state["pending_action"] = pending.model_dump(mode="json")
        self.state["status"] = "waiting_user"
        _emit(
            self.state,
            self.config,
            "agent.waiting_user",
            pending.title,
            pending.prompt,
            {"pending_action": self.state["pending_action"]},
        )
        return self.state


__all__ = [
    "GuardedModelInput",
    "ModelGuardrailCoordinator",
    "ModelInvocationTelemetry",
]
