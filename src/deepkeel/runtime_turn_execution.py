from __future__ import annotations

import time
from typing import Any, Callable
from uuid import uuid4

from deepkeel.async_ports import run_sync_adapter
from deepkeel.capability_manifest import RuntimeGeneration
from deepkeel.events import AgentEventPersistenceError, envelope_runtime_event
from deepkeel.failures import classify_runtime_failure
from deepkeel.graph import create_harness_graph
from deepkeel.hooks import HookAudit
from deepkeel.langgraph_adapter import checkpointer_supports_async, compiler_checkpointer
from deepkeel.leases import ExecutionFence
from deepkeel.persistence import CheckpointCompatibilityError
from deepkeel.runtime_api import RuntimeRequest, RuntimeResult, RuntimeStreamEvent
from deepkeel.runtime_events import RuntimeEventEmitter
from deepkeel.runtime_execution_support import (
    EventSink,
    emit_runtime_hook_audits,
    ensure_resume_generation_compatible,
    optional_int,
)
from deepkeel.runtime_failure_handling import RuntimeFailureHandlingMixin
from deepkeel.runtime_graph_execution import execute_graph_turn
from deepkeel.runtime_lifecycle import run_start_lifecycle_hooks
from deepkeel.runtime_model_pipeline import build_runtime_model_gateway
from deepkeel.runtime_policy import (
    _budget_limits,
    _conservative_model_context_profile,
    _max_elapsed_seconds,
    _merge_skill_activation,
    _model_providers,
    _prior_budget_state,
    _prior_diagnostics,
    _resolved_model_policy,
)
from deepkeel.runtime_results import _failed_runtime_state, project_harness_result
from deepkeel.runtime_settlement import project_and_settle_runtime_result
from deepkeel.scope import RuntimeScope
from deepkeel.skills import SkillPolicy
from deepkeel.telemetry import TelemetryRecord
from deepkeel.tools import ToolExecutionContext
from deepkeel.turn_context import TurnExecutionContext
from deepkeel.type_narrowing import as_dict


async def _arestore_budget(ledger: Any, run_id: str, snapshot: Any) -> None:
    await run_sync_adapter(ledger.restore, run_id, snapshot)


async def _abudget_snapshot(ledger: Any, run_id: str) -> Any:
    return await run_sync_adapter(ledger.snapshot, run_id)


class RuntimeTurnExecutionMixin(RuntimeFailureHandlingMixin):
    """Internal orchestration for one claimed turn and terminal setup failures."""

    async def _arun_claimed(
        self: Any,
        request: RuntimeRequest,
        *,
        provider: Any = None,
        providers: dict[str, Any] | None = None,
        session: Any = None,
        event_sink: EventSink | None = None,
        execution_fence: ExecutionFence | None = None,
    ) -> RuntimeResult:
        """Execute one turn through the typed, product-neutral public API."""
        question = request.question
        runtime_scope = request.runtime_scope
        user_id = runtime_scope.user_id
        short_context = request.short_context
        context_bundle = request.context_bundle
        skill_activation = request.skill_activation
        model_policy = request.model_policy
        input_parts = list(request.input_parts)
        run_started_monotonic = time.monotonic()
        short = short_context if isinstance(short_context, dict) else {}
        bundle = dict(context_bundle) if isinstance(context_bundle, dict) else {}
        # Policy inputs must come from the typed request, not from host-specific
        # duplication inside context_bundle.
        if request.run_id:
            bundle.setdefault("run_id", request.run_id)
        if request.thread_id:
            bundle.setdefault("thread_id", request.thread_id)
        if runtime_scope.tenant_id:
            bundle.setdefault("tenant_id", runtime_scope.tenant_id)
        if user_id:
            bundle.setdefault("user_id", user_id)
        if skill_activation:
            bundle["skill_activation"] = dict(skill_activation)
        resolved_model_policy = _resolved_model_policy(
            model_policy,
            provider=provider,
            providers=providers,
            max_steps=self.max_steps,
        )
        model_providers = _model_providers(provider, providers, resolved_model_policy)
        model_context_profile = _conservative_model_context_profile(
            model_providers,
            resolved_model_policy,
        )
        configured_input_limit = int(
            as_dict(resolved_model_policy.get("budget")).get("max_input_tokens_per_call") or 0
        )
        context_window_diagnostics: dict[str, Any] = {}
        try:
            if self.memory_recall_coordinator is not None:
                bundle = await self.memory_recall_coordinator.prepare(question, short, bundle)
                if not isinstance(bundle, dict):
                    raise TypeError("memory recall coordinator must return a mapping")
            if self.context_builder is not None:
                bundle = self.context_builder(question, short, bundle)
                if not isinstance(bundle, dict):
                    raise TypeError("context builder must return a mapping")
            for contributor_id, contributor in self.capability_catalog.context_contributors.items():
                contributed = contributor(dict(bundle))
                if not isinstance(contributed, dict):
                    raise TypeError(f"context contributor {contributor_id} must return a mapping")
                bundle = contributed
            bundle["_model_context_profile"] = model_context_profile.as_dict()
            bundle["_configured_input_limit"] = configured_input_limit
            prepared_context = self.context_window_manager.prepare(
                question,
                short,
                bundle,
            )
            bundle = prepared_context.context_bundle
            context_window_diagnostics = dict(prepared_context.diagnostics)
        except Exception as exc:
            return await self._context_setup_failure(
                question,
                exc,
                short_context=short,
                context_bundle=bundle,
                user_id=user_id,
                skill_activation=skill_activation,
                model_policy=model_policy,
                session=session,
                event_sink=event_sink,
                execution_fence=execution_fence,
                scope=runtime_scope,
            )
        run_id = str(
            request.run_id
            or bundle.get("agent_session_id")
            or bundle.get("agent_run_id")
            or bundle.get("run_id")
            or uuid4()
        )
        if short.get("resume"):
            (
                durable_state,
                checkpoint_authority,
                checkpoint_load_errors,
            ) = await self._aload_authoritative_checkpoint(
                run_id,
                session=session,
                user_id=str(user_id or "local-device"),
                scope=runtime_scope,
            )
        else:
            durable_state, checkpoint_authority, checkpoint_load_errors = {}, "none", []
        try:
            ensure_resume_generation_compatible(
                self.runtime_generation,
                durable_state,
            )
        except CheckpointCompatibilityError as exc:
            return await self._context_setup_failure(
                question,
                exc,
                short_context=short,
                context_bundle=bundle,
                user_id=user_id,
                skill_activation=skill_activation,
                model_policy=model_policy,
                session=session,
                event_sink=event_sink,
                execution_fence=execution_fence,
                scope=runtime_scope,
            )
        max_elapsed_seconds = _max_elapsed_seconds(resolved_model_policy)
        deadline_monotonic = (
            time.monotonic() + max_elapsed_seconds if max_elapsed_seconds > 0 else None
        )
        operational_run_id = runtime_scope.qualify_identity(run_id)
        prior_budget = _prior_budget_state(durable_state, short)
        await _arestore_budget(self.budget_ledger, operational_run_id, prior_budget)
        previous_diagnostics = _prior_diagnostics(durable_state, short)
        resolved_skill = _merge_skill_activation(
            durable_state=durable_state,
            session_projection=short,
            explicit=skill_activation,
        )
        skill_policy = SkillPolicy.from_snapshot(resolved_skill)
        skill = skill_policy.runtime_snapshot()
        conversation_thread_id = str(
            request.thread_id
            or bundle.get("thread_id")
            or bundle.get("ask_thread_id")
            or short.get("ask_thread_id")
            or run_id
        )
        graph_thread_id = operational_run_id
        turn_id = str(
            request.turn_id or bundle.get("turn_id") or short.get("turn_id") or f"turn-{uuid4()}"
        )
        bundle["operational_run_id"] = operational_run_id
        bundle["tenant_id"] = runtime_scope.tenant_id
        bundle["namespace"] = runtime_scope.namespace
        event_sequence = await self._event_latest_sequence(
            run_id,
            scope=runtime_scope,
            fallback=max(
                0,
                optional_int(bundle.get("event_sequence")) or 0,
                optional_int(short.get("event_sequence")) or 0,
            ),
        )
        run_version = max(
            0,
            optional_int(bundle.get("run_version")) or 0,
            optional_int(short.get("run_version")) or 0,
        )
        emitter = RuntimeEventEmitter(
            run_id=run_id,
            thread_id=conversation_thread_id,
            turn_id=turn_id,
            run_version=run_version,
            initial_sequence=event_sequence,
            skill_id=str(skill.get("skill_id") or ""),
            scope=runtime_scope,
            run_control=self.run_control,
            telemetry=self.telemetry,
            event_journal=self.event_journal,
            async_event_journal=self.async_event_journal,
            event_sink=event_sink,
            execution_fence=execution_fence,
        )
        emit = emitter

        memory_recall = as_dict(bundle.get("memory_recall"))
        # Resume must preserve run.resumed as the first newly persisted event.
        # Recall is deliberately skipped on resume and remains available in
        # diagnostics, so emitting a second skip event adds no recovery value.
        emit_memory_recall = not bool(short.get("resume") or short.get("recover_interrupted"))
        if memory_recall and emit_memory_recall:
            emit(
                {
                    "event_type": "memory.recall.decided",
                    "title": "Memory recall policy evaluated",
                    "summary": str(
                        memory_recall.get("reason") or memory_recall.get("status") or ""
                    ),
                    "payload": dict(memory_recall),
                    "visible": False,
                }
            )
            status = str(memory_recall.get("status") or "")
            if status in {"completed", "failed", "skipped"}:
                emit(
                    {
                        "event_type": f"memory.recall.{status}",
                        "title": f"Memory recall {status}",
                        "summary": str(memory_recall.get("reason") or status),
                        "payload": dict(memory_recall),
                        "visible": False,
                    }
                )

        package_ids = tuple(
            contribution.package_id for contribution in self.capability_contributions
        )
        lifecycle_start = await run_start_lifecycle_hooks(
            hook_runner=self.hook_runner,
            emit=emit,
            emit_audits=emit_runtime_hook_audits,
            run_id=run_id,
            thread_id=conversation_thread_id,
            turn_id=turn_id,
            package_ids=package_ids,
            skill_id=str(skill.get("skill_id") or ""),
            question=question,
            context_window=context_window_diagnostics,
            user_id=str(user_id or "local-device"),
            resumed=bool(short.get("resume")),
        )
        lifecycle_audits = list(lifecycle_start.audits)
        if lifecycle_start.context_patch:
            bundle.update(lifecycle_start.context_patch)
        if lifecycle_start.stopped_reason:
            return await self._context_setup_failure(
                question,
                RuntimeError(lifecycle_start.stopped_reason),
                short_context=short,
                context_bundle=bundle,
                user_id=user_id,
                skill_activation=skill_activation,
                model_policy=model_policy,
                session=session,
                event_sink=event_sink,
                execution_fence=execution_fence,
                scope=runtime_scope,
            )
        if lifecycle_start.context_patch:
            bundle["_model_context_profile"] = model_context_profile.as_dict()
            bundle["_configured_input_limit"] = configured_input_limit
            prepared_context = self.context_window_manager.prepare(
                question,
                short,
                bundle,
            )
            bundle = prepared_context.context_bundle
            context_window_diagnostics = dict(prepared_context.diagnostics)

        model_gateway = build_runtime_model_gateway(
            model_providers,
            router=self.model_router,
            policy_engine=self.policy_engine,
            budget_ledger=self.budget_ledger,
            invocation_recorder=self.model_invocation_recorder,
            invocation_store=self.model_invocation_store,
            model_health_store=self.model_health_store,
        )
        if self.reuse_compiled_graph:
            graph = self._shared_compiled_graph()
        else:
            graph = create_harness_graph(
                model=model_gateway,
                tool_executor=self.tool_executor,
                tool_registry=self.tool_registry,
                system_prompt=self.system_prompt_factory(skill),
                max_steps=self.max_steps,
                checkpointer=compiler_checkpointer(self.checkpointer),
                supports_async_checkpointer=checkpointer_supports_async(self.checkpointer),
                budget_ledger=self.budget_ledger,
                deadline_monotonic=deadline_monotonic,
                run_control=self.run_control,
                durability=self.graph_durability,
            )
        budget_limits = _budget_limits(resolved_model_policy)
        tool_context = ToolExecutionContext(
            run_id=run_id,
            user_id=str(user_id or "local-device"),
            thread_id=conversation_thread_id,
            turn_id=turn_id,
            session=session,
            session_factory=self.session_factory,
            context_bundle=bundle,
            metadata={
                "skill_activation": skill,
                "tenant_id": str(bundle.get("tenant_id") or ""),
                "governance_scope": {
                    "tenant_id": runtime_scope.tenant_id,
                    "user_id": str(user_id or "local-device"),
                    "namespace": runtime_scope.namespace,
                    "skill_id": str(skill.get("skill_id") or ""),
                    "scopes": list(bundle.get("governance_scopes") or []),
                },
                "operational_run_id": operational_run_id,
                "model_providers": model_providers,
                "model_policy": resolved_model_policy,
                "event_sink": emit,
                "budget_ledger": self.budget_ledger,
                "capability_package_ids": [
                    contribution.package_id for contribution in self.capability_contributions
                ],
            },
            budget_limits=budget_limits,
            deadline_monotonic=deadline_monotonic,
            run_control=self.run_control,
            execution_fence=execution_fence,
            scope=runtime_scope,
        )
        turn_context = TurnExecutionContext(
            model=model_gateway,
            system_prompt=self.system_prompt_factory(skill),
            tool_context=tool_context,
            event_sink=emit,
            deadline_monotonic=deadline_monotonic,
            tool_view_mode=self.tool_view_mode,
            hook_runner=self.hook_runner,
            entry_tool_skill_activator=self.entry_tool_skill_activator,
        )

        initial_budget_snapshot = await _abudget_snapshot(self.budget_ledger, operational_run_id)
        try:
            graph_outcome = await execute_graph_turn(
                graph=graph,
                question=question,
                run_id=run_id,
                graph_thread_id=graph_thread_id,
                turn_id=turn_id,
                user_id=str(user_id or "local-device"),
                short_context=short,
                context_bundle=bundle,
                skill_activation=skill,
                model_policy=resolved_model_policy,
                budget_state=initial_budget_snapshot.as_dict(),
                input_parts=input_parts,
                durable_state=durable_state,
                state_migrations=self.state_migrations,
                tool_registry=self.tool_registry,
                tool_context=tool_context,
                turn_context=turn_context,
                emit=emit,
                has_graph_checkpoint=self._has_graph_checkpoint,
            )
            state = graph_outcome.state
            recovery_source = graph_outcome.recovery_source
            active_graph_thread_id = graph_outcome.active_graph_thread_id
        except AgentEventPersistenceError:
            raise
        except Exception as exc:
            recovery_source = ""
            active_graph_thread_id = graph_thread_id
            failure = classify_runtime_failure(exc)
            if failure.category != "canceled":
                emit(
                    {
                        "event_type": "agent.failed",
                        "title": "Agent run failed",
                        "summary": failure.user_message,
                        "payload": {
                            "error": failure.detail,
                            "error_type": failure.exception_type,
                            "error_code": failure.code,
                            "failure": failure.as_dict(),
                        },
                    }
                )
            failed_budget_snapshot = await _abudget_snapshot(self.budget_ledger, operational_run_id)
            state = _failed_runtime_state(
                question,
                exc,
                run_id=run_id,
                thread_id=graph_thread_id,
                turn_id=turn_id,
                user_id=str(user_id or "local-device"),
                short_context=short,
                context_bundle=bundle,
                skill_activation=skill,
                model_policy=resolved_model_policy,
                budget_state=failed_budget_snapshot.as_dict(),
                events=emitter.events,
                failure=failure,
            )
        return await project_and_settle_runtime_result(
            runtime=self,
            state=state,
            question=question,
            context_bundle=bundle,
            short_context=short,
            skill_activation=skill,
            model_policy=resolved_model_policy,
            previous_diagnostics=previous_diagnostics,
            context_window_diagnostics=context_window_diagnostics,
            emitter=emitter,
            lifecycle_audits=lifecycle_audits,
            package_ids=package_ids,
            recovery_source=recovery_source,
            checkpoint_authority=checkpoint_authority,
            checkpoint_load_errors=checkpoint_load_errors,
            run_started_monotonic=run_started_monotonic,
            run_id=run_id,
            conversation_thread_id=conversation_thread_id,
            turn_id=turn_id,
            user_id=str(user_id or "local-device"),
            runtime_scope=runtime_scope,
            active_graph_thread_id=active_graph_thread_id,
            session=session,
            execution_fence=execution_fence,
            emit_hook_audits=emit_runtime_hook_audits,
        )
