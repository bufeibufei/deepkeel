from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from deepkeel.budget import BudgetLedger
from deepkeel.contracts import AgentMessage, FinalAnswer, PendingAction, ToolCall
from deepkeel.control import RunControl
from deepkeel.deadlines import ensure_time_remaining
from deepkeel.events import AgentEventPersistenceError
from deepkeel.hooks import HookAction, HookAudit, HookInvocation, HookPoint
from deepkeel.model import ModelGateway, ModelTurn, model_tools_from_registry
from deepkeel.model_failures import ModelToolContractError
from deepkeel.skills import SkillPolicy
from deepkeel.skill_activation import EntryToolActivationRequest
from deepkeel.tool_registry import ToolRegistry
from deepkeel.tool_disclosure import resolve_tool_view
from deepkeel.tools import ToolExecutionContext, ToolExecutor
from deepkeel.turn_context import TurnContextRegistry, TurnExecutionContext
from deepkeel.type_narrowing import as_dict
from deepkeel.workflow_policy import evaluate_workflow_completion
from deepkeel.graph_model_node import GraphModelNodeMixin
from deepkeel.graph_state import (
    _allowed_tool_names,
    _apply_policy_confirmation_resume,
    _apply_resume_payload,
    _apply_tool_result,
    _copy_state,
    _forced_workflow_tool_name,
    _hydrate_call,
    _is_policy_confirmation,
    _is_suspending_call,
    _messages,
    _model_available_roles,
    _parallel_suspension_rejected,
    _skill_tool_parameter_overrides,
    _stable_tool_calls,
    HarnessGraphState,
    migrate_legacy_graph_state,
)
from deepkeel.graph_model_step import (
    build_model_metrics,
    build_model_step_context,
    partition_model_tool_calls,
)
from deepkeel.graph_workflow import (
    TRUNCATED_FINISH_REASONS,
    _answer_summary,
    _complete_continued_answer,
    _config_value,
    _continue_or_fail_truncated_model_response,
    _emit,
    _finish_failed,
    _latency_ms,
    _record_completed_tool,
    _repair_or_fail_workflow,
    _retry_or_fail_empty_model_response,
    _set_policy_state,
    _wait_for_workflow_input,
    _workflow_can_wait_for_user_input,
)


class GraphNodes(GraphModelNodeMixin):
    def __init__(
        self,
        *,
        model: ModelGateway | None,
        tool_executor: ToolExecutor,
        tool_registry: ToolRegistry,
        prompt: str,
        max_steps: int,
        ledger: BudgetLedger,
        deadline_monotonic: float | None,
        control: RunControl,
        turn_contexts: TurnContextRegistry,
    ) -> None:
        self.model = model
        self.tool_executor = tool_executor
        self.tool_registry = tool_registry
        self.prompt = prompt
        self.max_steps = max_steps
        self.ledger = ledger
        self.deadline_monotonic = deadline_monotonic
        self.control = control
        self.turn_contexts = turn_contexts

    def turn_context(
        self,
        config: RunnableConfig | None,
        state: Mapping[str, Any] | None = None,
    ) -> TurnExecutionContext | None:
        context = _config_value(config or {}, "turn_context")
        if isinstance(context, TurnExecutionContext):
            return context
        current = state or {}
        return self.turn_contexts.resolve(
            str(current.get("run_id") or ""),
            str(current.get("thread_id") or ""),
        )

    def ensure_active(
        self,
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
        *,
        force: bool = False,
    ) -> None:
        metadata = as_dict(state.get("metadata"))
        operational_run_id = str(
            metadata.get("operational_run_id") or state.get("run_id") or ""
        )
        self.control.raise_if_cancelled(operational_run_id, force=force)
        turn_context = self.turn_context(config, state)
        deadline = (
            turn_context.deadline_monotonic if turn_context is not None else self.deadline_monotonic
        )
        ensure_time_remaining(deadline)

    @staticmethod
    def normalize_state(state: dict[str, Any], config: RunnableConfig) -> HarnessGraphState:
        context = _config_value(config, "tool_context")
        thread_id = str(_config_value(config, "thread_id") or state.get("thread_id") or "")
        normalized = dict(state)
        if isinstance(context, ToolExecutionContext):
            normalized.setdefault("run_id", context.run_id)
            normalized.setdefault("user_id", context.user_id)
        return migrate_legacy_graph_state(
            normalized, thread_id=thread_id or str(normalized.get("run_id") or "checkpoint")
        )

    def model_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return asyncio.run(self.amodel_node(state, config))

    def tool_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return asyncio.run(self.atool_node(state, config))

    async def atool_node(
        self,
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        normalized_state = self.normalize_state(state, config)
        tool_registry = self.tool_registry
        tool_executor = self.tool_executor
        ledger = self.ledger
        self.ensure_active(normalized_state, config, force=True)
        current = _copy_state(normalized_state)
        calls = [
            _hydrate_call(item, tool_registry, current["run_id"])
            for item in current.get("pending_tool_calls", [])
        ]
        context = _config_value(config, "tool_context")
        if not isinstance(context, ToolExecutionContext):
            context = ToolExecutionContext(run_id=current["run_id"], user_id=current["user_id"])
        else:
            context = context.fork(session=context.session)
        grants = current.get("metadata", {}).get("confirmation_grants")
        if isinstance(grants, dict):
            context.metadata["confirmation_grants"] = dict(grants)
        suspending_calls = [call for call in calls if _is_suspending_call(call, tool_registry)]
        if len(suspending_calls) > 1:
            results = [_parallel_suspension_rejected(call, current["run_id"]) for call in calls]
        else:
            for call in calls:
                try:
                    tool_spec = tool_registry.get(call.name)
                    visible_label = tool_spec.visible_label or call.name
                    start_event_visible = (
                        tool_spec.runtime_policy.get("start_event_visible") is not False
                    )
                except KeyError:
                    visible_label = call.name
                    start_event_visible = True
                _emit(
                    current,
                    config,
                    "tool.started",
                    visible_label,
                    "",
                    {
                        "tool_call": call.model_dump(mode="json"),
                        "visible": start_event_visible,
                    },
                )
            results = await tool_executor.aexecute_many(calls, context)
        current["budget_state"] = ledger.snapshot(context.operational_run_id).as_dict()
        current["pending_tool_calls"] = []
        current["pending_action"] = None
        current["pending_async"] = None
        for result in results:
            if _is_unexecuted_suspension_rejection(result):
                _emit(
                    current,
                    config,
                    "tool.skipped",
                    result.name,
                    "Skipped until the pending action is resolved.",
                    {
                        "tool_call": result.call.model_dump(mode="json") if result.call else {},
                        "tool_result": result.model_dump(mode="json", exclude={"call"}),
                        "visible": False,
                    },
                )
                continue
            current.setdefault("tool_results", []).append(result.model_dump(mode="json"))
            _record_completed_tool(current, result)
            _apply_tool_result(current, result, config)
        if isinstance(grants, dict):
            remaining_grants = dict(grants)
            for call in calls:
                remaining_grants.pop(call.id, None)
            current.setdefault("metadata", {})["confirmation_grants"] = remaining_grants
        if current.get("pending_action"):
            current["status"] = "waiting_user"
        elif current.get("pending_async"):
            current["status"] = "waiting_async"
        else:
            current["status"] = "reasoning"
        return current

    def await_user_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        normalized_state = self.normalize_state(state, config)
        self.ensure_active(normalized_state, config, force=True)
        current = _copy_state(normalized_state)
        resume_payload = interrupt(current.get("pending_action") or {})
        if _is_policy_confirmation(current.get("pending_action")):
            return _apply_policy_confirmation_resume(current, resume_payload, config)
        return _apply_resume_payload(current, resume_payload, config, source="user_action")

    def await_async_node(self, state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        normalized_state = self.normalize_state(state, config)
        self.ensure_active(normalized_state, config, force=True)
        current = _copy_state(normalized_state)
        resume_payload = interrupt(current.get("pending_async") or {})
        return _apply_resume_payload(current, resume_payload, config, source="async_observation")


def _is_unexecuted_suspension_rejection(result: Any) -> bool:
    metadata = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
    return bool(metadata.get("suspension_rejected")) and metadata.get("executed") is False
