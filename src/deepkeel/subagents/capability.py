from __future__ import annotations

import hashlib
from typing import Any, Protocol

from pydantic import ValidationError

from deepkeel.contracts import (
    Observation,
    ObservationStatus,
    PendingAction,
    ResultOutcome,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from deepkeel.skills import DelegationPolicy
from deepkeel.subagents.contracts import DelegationRequest
from deepkeel.subagents.executor import SubAgentExecutor
from deepkeel.tools import ToolExecutionContext


class DelegationDispatcher(Protocol):
    def dispatch(
        self,
        request: DelegationRequest,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: Any = None,
    ) -> dict[str, Any]: ...


class DelegationToolHandler:
    """Domain-neutral adapter exposing bounded delegation as one harness tool."""

    def __init__(
        self,
        executor: SubAgentExecutor,
        *,
        dispatcher: DelegationDispatcher | None = None,
    ) -> None:
        self.executor = executor
        self.dispatcher = dispatcher

    def __call__(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        try:
            request = DelegationRequest.model_validate(
                {
                    "delegation_id": call.arguments.get("delegation_id")
                    or _delegation_id(context.run_id, call.id),
                    "root_run_id": context.run_id,
                    "parent_run_id": context.run_id,
                    "depth": 1,
                    "max_concurrency": call.arguments.get("max_concurrency") or 3,
                    "execution_mode": call.arguments.get("execution_mode") or "auto",
                    "tasks": call.arguments.get("tasks") or [],
                }
            )
            unknown_agent_ids = sorted(
                {
                    task.agent_id
                    for task in request.tasks
                    if not self._is_registered(task.agent_id)
                }
            )
            if unknown_agent_ids:
                raise ValueError(
                    "unknown delegated agent ids: " + ", ".join(unknown_agent_ids)
                )
            self._validate_skill_policy(request, context)
            providers = context.metadata.get("model_providers")
            if not isinstance(providers, dict) or not providers:
                raise RuntimeError("model providers are unavailable for delegation")
            event_sink = context.metadata.get("event_sink")
            can_dispatch_background = (
                self.dispatcher is not None and context.session_factory is not None
            )
            should_dispatch_background = (
                request.execution_mode == "background"
                or (
                    request.execution_mode == "auto"
                    and can_dispatch_background
                )
            )
            if should_dispatch_background and not can_dispatch_background:
                raise RuntimeError(
                    "background subagent execution is unavailable in this host"
                )
            if should_dispatch_background and self.dispatcher is not None:
                dispatched = self.dispatcher.dispatch(
                    request,
                    context=context,
                    providers=providers,
                    event_sink=event_sink if callable(event_sink) else None,
                )
                return _result(
                    call,
                    context,
                    status="waiting_async",
                    summary=(
                        f"Started {len(request.tasks)} specialist tasks; the parent run "
                        "will resume when they complete."
                    ),
                    data=dispatched,
                    metadata={"visible_label": "Specialist collaboration in progress"},
                )
            batch = self.executor.execute_many(
                request,
                context=context,
                providers=providers,
                event_sink=event_sink if callable(event_sink) else None,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            return self._recoverable_invalid_request(call, exc, context)
        except (KeyError, RuntimeError) as exc:
            return _result(
                call,
                context,
                status="failed",
                summary="Specialist delegation failed.",
                error=str(exc),
                retryable=True,
                metadata={"visible_label": "Specialist collaboration incomplete"},
            )
        if batch.status == "needs_input":
            pending_result = batch.pending_input_results[0]
            input_request = pending_result.input_request
            if input_request is None:
                raise RuntimeError("needs_input delegation result is missing input_request")
            pending = PendingAction(
                id=f"{call.id}:subagent-input",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="subagent_input",
                title="More information is needed",
                prompt=input_request.prompt,
                payload={
                    "delegation": batch.parent_payload(),
                    "resume_token": input_request.resume_token,
                    "requirements": list(input_request.requirements),
                    "input_schema": dict(input_request.input_schema),
                },
            )
            return _result(
                call,
                context,
                status="requires_user_action",
                outcome="partial",
                summary=input_request.prompt,
                data=batch.parent_payload(),
                pending_action=pending,
                metadata={
                    "visible_label": "Waiting for additional information",
                    "parent_projection": True,
                    "subagent_needs_input": True,
                },
            )
        outcome = batch.status if batch.status != "failed" else "degraded"
        return _result(
            call,
            context,
            status="succeeded",
            outcome=outcome,
            summary=(
                f"Completed {len(batch.successful_results)}/{len(batch.results)} "
                "specialist tasks."
            ),
            data=batch.parent_payload(),
            metadata={
                "visible_label": "Parallel specialist analysis",
                "completed_inline": True,
                "parent_projection": True,
            },
        )

    def _is_registered(self, agent_id: str) -> bool:
        try:
            self.executor.registry.get(agent_id)
        except KeyError:
            return False
        return True

    @staticmethod
    def _validate_skill_policy(
        request: DelegationRequest,
        context: ToolExecutionContext,
    ) -> None:
        skill = context.metadata.get("skill_activation")
        skill = skill if isinstance(skill, dict) else {}
        if not isinstance(skill.get("delegation_policy"), dict):
            return
        policy = DelegationPolicy.from_snapshot(skill)
        if not policy.enabled:
            raise ValueError("delegation is disabled by the active Skill policy")
        if len(request.tasks) > policy.max_tasks:
            raise ValueError("delegation task count exceeds the active Skill policy")
        if request.max_concurrency > policy.max_concurrency:
            raise ValueError("delegation concurrency exceeds the active Skill policy")
        denied = sorted(
            {task.agent_id for task in request.tasks if not policy.allows_agent(task.agent_id)}
        )
        if denied:
            raise ValueError("delegated agents are outside the active Skill policy: " + ", ".join(denied))

    @staticmethod
    def _recoverable_invalid_request(
        call: ToolCall,
        exc: Exception,
        context: ToolExecutionContext,
    ) -> ToolResult:
        return _result(
            call,
            context,
            status="succeeded",
            outcome="skipped",
            summary=(
                "Delegation arguments violated the contract. Continue in the lead "
                "agent without delegating again."
            ),
            data={
                "status": "skipped",
                "reason_code": "invalid_delegation_request",
                "fallback": "continue_with_parent_agent",
            },
            metadata={
                "visible_label": "Returned to the lead agent",
                "visible": False,
                "internal_error": str(exc),
                "completed_inline": True,
            },
        )


def _delegation_id(run_id: str, tool_call_id: str) -> str:
    return hashlib.sha256(
        f"{run_id}\x1f{tool_call_id}".encode("utf-8")
    ).hexdigest()[:32]


def _result(
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    status: ToolResultStatus,
    summary: str,
    outcome: ResultOutcome | None = None,
    data: dict[str, Any] | None = None,
    error: str = "",
    retryable: bool = False,
    pending_action: PendingAction | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    if status == "succeeded":
        observation_status: ObservationStatus = "succeeded"
    elif status == "failed":
        observation_status = "failed"
    elif status == "waiting_async":
        observation_status = "pending"
    else:
        observation_status = "requires_user_action"
    observation = Observation(
        id=f"{call.id}:observation",
        run_id=context.run_id,
        tool_call_id=call.id,
        source=call.name,
        status=observation_status,
        outcome=outcome,
        summary=summary,
        data=dict(data or {}),
        error=error,
    )
    return ToolResult(
        call=call,
        status=status,
        outcome=outcome,
        summary=summary,
        data=dict(data or {}),
        error=error,
        retryable=retryable,
        pending_action=pending_action,
        observation=observation,
        metadata=dict(metadata or {}),
    )
