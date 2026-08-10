from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from deepkeel.context import build_initial_messages
from deepkeel.events import AgentEventPersistenceError
from deepkeel.persistence import (
    checkpoint_from_durable_state,
    checkpoint_from_runtime,
    restore_run_context,
    resume_payload_from_context,
)
from deepkeel.planning.runtime import advance_plan_after_resume
from deepkeel.runtime_results import (
    _new_context,
    _skill_precondition_tool_calls,
)


@dataclass(frozen=True, slots=True)
class GraphExecutionOutcome:
    state: dict[str, Any]
    recovery_source: str
    active_graph_thread_id: str


async def execute_graph_turn(
    *,
    graph: Any,
    question: str,
    run_id: str,
    graph_thread_id: str,
    turn_id: str,
    user_id: str,
    short_context: dict[str, Any],
    context_bundle: dict[str, Any],
    skill_activation: dict[str, Any],
    model_policy: dict[str, Any],
    budget_state: dict[str, Any],
    input_parts: list[Any],
    durable_state: dict[str, Any],
    state_migrations: Any,
    tool_registry: Any,
    tool_context: Any,
    turn_context: Any,
    emit: Callable[[dict[str, Any]], None],
    has_graph_checkpoint: Callable[[str], bool],
) -> GraphExecutionOutcome:
    recovery_source = ""
    active_graph_thread_id = graph_thread_id
    if short_context.get("recover_interrupted"):
        if has_graph_checkpoint(graph_thread_id):
            state = dict(
                await graph.arecover(
                    graph_thread_id,
                    tool_context=tool_context,
                    event_sink=emit,
                    turn_context=turn_context,
                )
            )
            if not state:
                raise RuntimeError("langgraph recovery checkpoint is unavailable")
            recovery_source = "durable_langgraph_restart"
        else:
            context = _new_context(
                question,
                run_id=run_id,
                thread_id=graph_thread_id,
                turn_id=turn_id,
                user_id=user_id,
                short_context=short_context,
                context_bundle=context_bundle,
                skill_activation=skill_activation,
                model_policy=model_policy,
                budget_state=budget_state,
                input_parts=input_parts,
            )
            state = dict(
                await graph.ainvoke(
                    context,
                    tool_context=tool_context,
                    event_sink=emit,
                    turn_context=turn_context,
                )
            )
            recovery_source = "restart_replay_without_checkpoint"
    elif short_context.get("resume"):
        resume_payload = resume_payload_from_context(short_context)
        try:
            state = dict(
                await graph.aresume(
                    graph_thread_id,
                    resume_payload,
                    tool_context=tool_context,
                    event_sink=emit,
                    turn_context=turn_context,
                )
            )
            recovery_source = "live_langgraph"
        except AgentEventPersistenceError:
            raise
        except (RuntimeError, ValueError, AttributeError) as live_resume_error:
            emit(
                {
                    "event_type": "checkpoint.live_resume_failed",
                    "title": "Live checkpoint resume failed",
                    "summary": str(live_resume_error),
                    "payload": {
                        "error_type": type(live_resume_error).__name__,
                        "error": str(live_resume_error),
                        "visible": False,
                    },
                }
            )
            previous_runtime = short_context.get("previous_runtime")
            recovered_checkpoint = checkpoint_from_durable_state(
                durable_state,
                migrations=state_migrations,
            )
            durable_checkpoint_available = bool(recovered_checkpoint)
            if not recovered_checkpoint:
                recovered_checkpoint = checkpoint_from_runtime(
                    previous_runtime if isinstance(previous_runtime, dict) else {},
                    migrations=state_migrations,
                )
            if not recovered_checkpoint:
                raise RuntimeError(
                    "durable checkpoint is unavailable after live LangGraph resume "
                    f"failed: {type(live_resume_error).__name__}: {live_resume_error}"
                ) from live_resume_error
            recovery_source = (
                "agent_run_checkpoint"
                if durable_checkpoint_available
                else "session_projection"
            )
            recovered_thread_id = f"{graph_thread_id}:recovered:{uuid4().hex[:8]}"
            recovered = restore_run_context(
                checkpoint=recovered_checkpoint,
                resume_payload=resume_payload,
                run_id=run_id,
                thread_id=recovered_thread_id,
                turn_id=turn_id,
                user_id=user_id,
                skill_activation=skill_activation,
                model_policy=model_policy,
                migrations=state_migrations,
            )
            recovered_state = recovered.model_dump(mode="json")
            pending = (
                recovered_checkpoint.get("pending_action")
                or recovered_checkpoint.get("pending_async")
                or {}
            )
            advance_plan_after_resume(
                recovered_state,
                pending=pending if isinstance(pending, dict) else {},
                resume_payload=resume_payload,
                registry=tool_registry,
                emit=lambda event_type, title, summary, payload: emit(
                    {
                        "event_type": event_type,
                        "title": title,
                        "summary": summary,
                        "payload": payload,
                    }
                ),
            )
            recovered = type(recovered).model_validate(recovered_state)
            if not any(message.role == "user" for message in recovered.messages):
                recovered.messages = [
                    *build_initial_messages(
                        question,
                        short_context,
                        context_bundle,
                        input_parts=input_parts,
                    ),
                    *recovered.messages,
                ]
            active_graph_thread_id = recovered_thread_id
            state = dict(
                await graph.ainvoke(
                    recovered,
                    tool_context=tool_context,
                    event_sink=emit,
                    turn_context=turn_context,
                )
            )
    else:
        precondition_calls = _skill_precondition_tool_calls(
            skill_activation,
            context={**short_context, **context_bundle},
            tool_registry=tool_registry,
        )
        context = _new_context(
            question,
            run_id=run_id,
            thread_id=graph_thread_id,
            turn_id=turn_id,
            user_id=user_id,
            short_context=short_context,
            context_bundle=context_bundle,
            skill_activation=skill_activation,
            model_policy=model_policy,
            budget_state=budget_state,
            input_parts=input_parts,
            pending_tool_calls=precondition_calls,
        )
        state = dict(
            await graph.ainvoke(
                context,
                tool_context=tool_context,
                event_sink=emit,
                turn_context=turn_context,
            )
        )
    return GraphExecutionOutcome(
        state=state,
        recovery_source=recovery_source,
        active_graph_thread_id=active_graph_thread_id,
    )
