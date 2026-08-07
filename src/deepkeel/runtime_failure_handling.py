from __future__ import annotations

from typing import Any
from uuid import uuid4

from deepkeel.events import AgentEventPersistenceError, envelope_runtime_event
from deepkeel.failures import classify_runtime_failure
from deepkeel.leases import ExecutionFence
from deepkeel.runtime_api import RuntimeResult, RuntimeStreamEvent
from deepkeel.runtime_execution_support import EventSink, optional_int
from deepkeel.runtime_results import _failed_runtime_state, project_harness_result
from deepkeel.scope import RuntimeScope
from deepkeel.skills import SkillPolicy
from deepkeel.telemetry import TelemetryRecord
from deepkeel.type_narrowing import as_dict


class RuntimeFailureHandlingMixin:
    """Terminalizes setup failures through the same durable result contract."""

    async def _context_setup_failure(
        self: Any,
        question: str,
        exc: Exception,
        *,
        short_context: dict[str, Any],
        context_bundle: dict[str, Any],
        user_id: str,
        skill_activation: dict[str, Any] | None,
        model_policy: dict[str, Any] | None,
        session: Any,
        event_sink: EventSink | None,
        execution_fence: ExecutionFence | None,
        scope: RuntimeScope,
    ) -> RuntimeResult:
        run_id = str(
            context_bundle.get("agent_session_id")
            or context_bundle.get("agent_run_id")
            or context_bundle.get("run_id")
            or uuid4()
        )
        thread_id = str(
            context_bundle.get("thread_id")
            or context_bundle.get("ask_thread_id")
            or short_context.get("ask_thread_id")
            or run_id
        )
        turn_id = str(
            context_bundle.get("turn_id") or short_context.get("turn_id") or f"turn-{uuid4()}"
        )
        failure = classify_runtime_failure(exc)
        resolved_skill = SkillPolicy.from_snapshot(skill_activation).runtime_snapshot()
        sequence = (
            await self._event_latest_sequence(
                run_id,
                fallback=max(
                    0,
                    optional_int(context_bundle.get("event_sequence")) or 0,
                    optional_int(short_context.get("event_sequence")) or 0,
                ),
            )
            + 1
        )
        event = envelope_runtime_event(
            {
                "event_type": "agent.failed",
                "title": "Agent context setup failed",
                "summary": failure.user_message,
                "payload": {
                    "error": failure.detail,
                    "error_type": failure.exception_type,
                    "error_code": failure.code,
                    "failure": failure.as_dict(),
                    "phase": "context_setup",
                    "skill_id": str(resolved_skill.get("skill_id") or ""),
                },
            },
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            sequence=sequence,
            run_version=max(
                0,
                optional_int(context_bundle.get("run_version")) or 0,
                optional_int(short_context.get("run_version")) or 0,
            ),
            scope=scope,
        )
        envelope = RuntimeStreamEvent.model_validate(event)
        if self.async_event_journal is not None:
            try:
                await self.async_event_journal.append(envelope)
            except Exception as journal_exc:
                raise AgentEventPersistenceError(
                    f"runtime event journal append failed: {journal_exc}"
                ) from journal_exc
        elif self.event_journal is not None:
            try:
                self.event_journal.append(envelope)
            except Exception as journal_exc:
                raise AgentEventPersistenceError(
                    f"runtime event journal append failed: {journal_exc}"
                ) from journal_exc
        event = envelope.model_dump(mode="json")
        try:
            self.telemetry.record(
                TelemetryRecord.from_runtime_event(
                    event,
                    run_id=run_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            )
        except Exception:
            pass
        if event_sink is not None:
            event_sink(event)
        resolved_policy = dict(model_policy) if isinstance(model_policy, dict) else {}
        state = _failed_runtime_state(
            question,
            exc,
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            user_id=str(user_id or "local-device"),
            short_context=short_context,
            context_bundle=context_bundle,
            skill_activation=resolved_skill,
            model_policy=resolved_policy,
            budget_state={},
            events=[event],
            failure=failure,
            phase="context_setup",
        )
        result = project_harness_result(
            state,
            question=question,
            context_bundle=context_bundle,
            short_context=short_context,
            skill_activation=resolved_skill,
            streamed_events=[event],
            user_id=str(user_id or "local-device"),
            answer_delta_streamed=False,
            observation_kinds={},
            task_kinds={},
            max_steps=self.max_steps,
            capability_manifest=self._capability_manifest(),
        )
        try:
            self.telemetry.record(
                TelemetryRecord(
                    event_name="runtime.settled",
                    run_id=run_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    status="failed",
                    attributes={
                        "status": "failed",
                        "stop_reason": result.stop_reason,
                        "skill_id": str(resolved_skill.get("skill_id") or ""),
                        "recovery_source": "",
                        "phase": "context_setup",
                    },
                )
            )
        except Exception:
            pass
        await self._persist_runtime_snapshot(
            result,
            run_id=run_id,
            thread_id=thread_id,
            session=session,
            user_id=str(user_id or "local-device"),
            context_bundle=context_bundle,
            execution_fence=execution_fence,
            scope=scope,
        )
        if result.status.value in {"completed", "failed", "canceled"}:
            if self._host_owns_terminal_settlement():
                recovery = as_dict(result.diagnostics.get("recovery"))
                recovery["checkpoint_cleanup"] = {
                    "status": "deferred",
                    "reason": "host_settlement_required",
                }
                result.diagnostics["recovery"] = recovery
            else:
                await self._acleanup_run(
                    result,
                    run_id=run_id,
                    session=session,
                    user_id=str(user_id or "local-device"),
                    scope=scope,
                    graph_thread_ids=[run_id],
                )
            await self._record_checkpoint_cleanup_event(
                result,
                run_id=run_id,
                thread_id=thread_id,
                turn_id=turn_id,
                scope=scope,
                event_sink=event_sink,
                fallback_sequence=sequence,
            )
        return result
