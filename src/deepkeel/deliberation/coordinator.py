from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from deepkeel.deliberation.contracts import (
    DeliberationArgument,
    DeliberationPhase,
    DeliberationResult,
    DeliberationSpec,
    DeliberationStatus,
)
from deepkeel.subagents import DelegationRequest, DelegationTask, SubAgentExecutor
from deepkeel.tools import ToolExecutionContext


EventSink = Callable[[dict[str, Any]], None]
CheckpointSink = Callable[[str, dict[str, Any]], None]


class DeliberationCoordinator:
    """Business-neutral, bounded multi-agent deliberation over one immutable fact packet."""

    def __init__(self, subagents: SubAgentExecutor) -> None:
        self.subagents = subagents

    def run(
        self,
        spec: DeliberationSpec,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: EventSink | None = None,
        should_stop: Callable[[], bool] | None = None,
        resume_state: dict[str, Any] | None = None,
        checkpoint_sink: CheckpointSink | None = None,
    ) -> DeliberationResult:
        stop_requested = should_stop or (lambda: False)
        state = self._restore_state(resume_state)
        self._emit(event_sink, "deliberation.started", "Deliberation started", spec, {
            "participants": [item.model_dump(mode="json") for item in spec.participants],
            "max_rounds": spec.max_rounds,
            "max_model_calls": spec.max_model_calls,
        })

        if "opening" not in state["completed_stages"]:
            self._emit_stage_started(
                event_sink,
                spec,
                phase="opening",
                round_index=1,
                participants=spec.participants,
            )
            opening, calls, retries = self._participant_batch(
                spec,
                phase="opening",
                round_index=1,
                context=context,
                providers=providers,
                event_sink=event_sink,
                prior_arguments=[],
                existing_arguments=state["arguments"],
                should_stop=stop_requested,
            )
            state["arguments"] = _merge_arguments(state["arguments"], opening)
            state["model_calls"] += calls
            state["retry_count"] += retries
            state["completed_stages"].append("opening")
            self._checkpoint(checkpoint_sink, "opening", state)
        opening = [item for item in state["arguments"] if item.phase == "opening"]
        completed_openings = sum(item.status == "completed" for item in opening)
        if completed_openings < spec.min_completed_participants:
            return self._finish(spec, state, {}, "no_opening_arguments", event_sink, checkpoint_sink)

        round_index = max(1, max((item.round_index for item in state["arguments"]), default=1))
        moderation_key = f"moderate:{round_index}"
        stopped_after_opening = stop_requested()
        moderator = (
            state["moderation_history"][-1]
            if state["moderation_history"]
            else {"decision": "synthesize", "status": "skipped"}
            if stopped_after_opening
            else {}
        )
        if moderation_key not in state["completed_stages"] and not stopped_after_opening:
            self._emit_stage_started(
                event_sink,
                spec,
                phase="moderate",
                round_index=round_index,
            )
            moderator, calls, retries = self._moderate(
                spec,
                state["arguments"],
                context,
                providers,
                event_sink,
                phase="moderate",
                round_index=round_index,
            )
            state["moderation_history"].append(moderator)
            state["model_calls"] += calls
            state["retry_count"] += retries
            state["completed_stages"].append(moderation_key)
            self._checkpoint(checkpoint_sink, "moderating", state)

        stop_reason = "user_stop_and_summarize" if stopped_after_opening else "moderator_converged"
        while True:
            decision = self._moderator_decision(moderator, round_index)
            targets = self._target_participants(spec, moderator)
            budget_needed = len(targets) + 1 + spec.synthesis_reserve_calls
            if stop_requested():
                stop_reason = "user_stop_and_summarize"
                break
            if decision == "synthesize":
                stop_reason = "moderator_converged"
                break
            if round_index >= spec.max_rounds or state["model_calls"] + budget_needed > spec.max_model_calls:
                stop_reason = "round_or_budget_limit"
                break

            round_index += 1
            rebuttal_key = f"rebuttal:{round_index}"
            if rebuttal_key not in state["completed_stages"]:
                self._emit_stage_started(
                    event_sink,
                    spec,
                    phase="rebuttal",
                    round_index=round_index,
                    participants=targets,
                    extra={
                        "unresolved_questions": moderator.get("unresolved_questions") or [],
                        "disagreement_graph": moderator.get("disagreement_graph") or [],
                    },
                )
                rebuttals, calls, retries = self._participant_batch(
                    spec,
                    phase="rebuttal",
                    round_index=round_index,
                    context=context,
                    providers=providers,
                    event_sink=event_sink,
                    prior_arguments=state["arguments"],
                    moderator=moderator,
                    participants=targets,
                    existing_arguments=state["arguments"],
                    should_stop=stop_requested,
                )
                state["arguments"] = _merge_arguments(state["arguments"], rebuttals)
                state["model_calls"] += calls
                state["retry_count"] += retries
                state["completed_stages"].append(rebuttal_key)
                self._checkpoint(checkpoint_sink, "rebuttal", state)
            if stop_requested():
                stop_reason = "user_stop_and_summarize"
                break

            moderation_key = f"moderate:{round_index}"
            if moderation_key not in state["completed_stages"]:
                self._emit_stage_started(
                    event_sink,
                    spec,
                    phase="moderate",
                    round_index=round_index,
                )
                moderator, calls, retries = self._moderate(
                    spec,
                    state["arguments"],
                    context,
                    providers,
                    event_sink,
                    phase="moderate",
                    round_index=round_index,
                )
                state["moderation_history"].append(moderator)
                state["model_calls"] += calls
                state["retry_count"] += retries
                state["completed_stages"].append(moderation_key)
                self._checkpoint(checkpoint_sink, "moderating", state)

        if "synthesize" not in state["completed_stages"]:
            self._emit_stage_started(
                event_sink,
                spec,
                phase="synthesize",
                round_index=round_index,
            )
            synthesis, calls, retries = self._moderate(
                spec,
                state["arguments"],
                context,
                providers,
                event_sink,
                phase="synthesize",
                round_index=round_index,
            )
            state["synthesis"] = synthesis
            state["model_calls"] += calls
            state["retry_count"] += retries
            state["completed_stages"].append("synthesize")
            self._checkpoint(checkpoint_sink, "synthesizing", state)
        return self._finish(spec, state, moderator, stop_reason, event_sink, checkpoint_sink)

    def _participant_batch(
        self,
        spec: DeliberationSpec,
        *,
        phase: DeliberationPhase,
        round_index: int,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: EventSink | None,
        prior_arguments: list[DeliberationArgument],
        moderator: dict[str, Any] | None = None,
        participants=None,
        existing_arguments: list[DeliberationArgument] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[list[DeliberationArgument], int, int]:
        selected = list(participants or spec.participants)
        completed_ids = {
            item.argument_id
            for item in (existing_arguments or [])
            if item.status == "completed"
        }
        selected = [
            item
            for item in selected
            if _argument_id(spec, phase, round_index, item.participant_instance_id)
            not in completed_ids
        ]
        stop_before_batch = (
            should_stop is not None
            and should_stop()
            and (phase != "opening" or bool(completed_ids))
        )
        if not selected or stop_before_batch:
            return [], 0, 0
        tasks = [self._participant_task(spec, item, phase, round_index, prior_arguments, moderator) for item in selected]
        results, calls, retries = self._execute_with_retry(
            spec,
            tasks,
            context=context,
            providers=providers,
            event_sink=event_sink,
            stage=f"{phase}:{round_index}",
            should_stop=should_stop,
        )
        by_agent = {item.agent_id: item for item in results}
        arguments = []
        for participant in selected:
            item = by_agent.get(participant.agent_id)
            argument = DeliberationArgument(
                argument_id=_argument_id(
                    spec,
                    phase,
                    round_index,
                    participant.participant_instance_id,
                ),
                round_index=round_index,
                phase=phase,
                participant_instance_id=participant.participant_instance_id,
                agent_id=participant.agent_id,
                display_name=participant.display_name,
                label=participant.label,
                status=item.status if item else "failed",
                conclusion=item.conclusion if item else "",
                evidence=item.evidence if item else [],
                evidence_refs=item.evidence_refs if item else [],
                risks=item.risks if item else [],
                recommendations=item.recommendations if item else [],
                warnings=item.warnings if item else [],
                tool_trace=(item.metadata.get("tool_trace") or []) if item else [],
                confidence=item.confidence if item else None,
                duration_ms=item.duration_ms if item else 0,
                child_run_id=item.child_run_id if item else "",
                model_role=item.model_role if item else "",
                model_id=item.model_id if item else "",
                outcome=item.outcome if item else "",
                error=item.error if item else "missing result",
            )
            arguments.append(argument)
            self._emit(event_sink, f"deliberation.{phase}.completed", participant.display_name, spec, {
                "participant": participant.model_dump(mode="json"),
                "argument": argument.model_dump(mode="json"),
            })
        return arguments, calls, retries

    @staticmethod
    def _participant_task(spec, participant, phase, round_index, prior_arguments, moderator):
        objective = (
            f"From the {participant.label} perspective, provide an independent opening judgment: {spec.question}"
            if phase == "opening"
            else f"From the {participant.label} perspective, address the moderator-selected disagreement: {spec.question}"
        )
        return DelegationTask(
            id=f"{phase}-{round_index}-{participant.participant_instance_id}",
            agent_id=participant.agent_id,
            objective=objective,
            input_data={
                "question": spec.question,
                "facts": _participant_facts(spec.facts, participant.fact_keys),
                "phase": phase,
                "round_index": round_index,
                "other_views": [item.model_dump(mode="json") for item in prior_arguments],
                "moderator": moderator or {},
            },
            constraints=[
                "All participants must use the same facts and must not fabricate external facts",
                "When facts are insufficient, use only allowed read-only tools and cite their source",
                "Address arguments rather than identities; do not decide by vote",
                *participant.instructions,
            ],
            metadata={
                "participant_instance_id": participant.participant_instance_id,
                "participant_label": participant.label,
                "fact_keys": list(participant.fact_keys),
                "deliberation_id": spec.deliberation_id,
                "deliberation_phase": phase,
                "deliberation_round": round_index,
                "deliberation_role": "participant",
            },
        )

    def _moderate(self, spec, arguments, context, providers, event_sink, *, phase: str, round_index: int):
        objective = (
            "Identify factual consensus and substantive disagreement, then continue, target a response, or synthesize."
            if phase == "moderate"
            else "Synthesize a direct conclusion, shared evidence, remaining disagreements, conditions, actions, and boundaries."
        )
        task = DelegationTask(
            id=f"{phase}-{round_index}-moderator",
            agent_id=spec.moderator_agent_id,
            objective=objective,
            input_data={
                "question": spec.question,
                "facts": spec.facts,
                "participant_views": [item.model_dump(mode="json") for item in arguments],
                "phase": phase,
                "round_index": round_index,
                "max_rounds": spec.max_rounds,
            },
            constraints=[
                "Do not decide by vote or introduce facts beyond the supplied facts and read-only tool results",
                "The moderation phase must state its decision, unresolved questions, and target agent IDs",
            ],
            metadata={
                "deliberation_id": spec.deliberation_id,
                "deliberation_phase": phase,
                "deliberation_round": round_index,
                "deliberation_role": "moderator",
            },
        )
        results, calls, retries = self._execute_with_retry(
            spec,
            [task],
            context=context,
            providers=providers,
            event_sink=event_sink,
            stage=f"{phase}:{round_index}",
        )
        item = next((entry for entry in results if entry.status == "completed"), None)
        output = dict(item.output) if item and isinstance(item.output, dict) else {}
        payload = {
            "status": item.status if item else "failed",
            "summary": item.conclusion if item else "",
            "shared_evidence": item.evidence if item else [],
            "disagreements": item.risks if item else [],
            "retained_positions": item.recommendations if item else [],
            "decision": str(output.get("decision") or ""),
            "unresolved_questions": _string_list(output.get("unresolved_questions")),
            "target_agent_ids": _string_list(output.get("target_agent_ids")),
            "disagreement_graph": _dict_list(output.get("disagreement_graph")),
            "convergence_score": _score(output.get("convergence_score")),
            "conditions": _string_list(output.get("conditions")),
            "action_recommendations": _string_list(output.get("action_recommendations")),
            "judgment_boundary": str(output.get("judgment_boundary") or "").strip(),
            "round_index": round_index,
            "error": item.error if item else "moderator result is missing",
        }
        self._emit(event_sink, f"deliberation.{phase}.completed", "The lead agent completed phase synthesis", spec, payload)
        return payload, calls, retries

    def _execute_with_retry(
        self,
        spec,
        tasks,
        *,
        context,
        providers,
        event_sink,
        stage,
        should_stop: Callable[[], bool] | None = None,
    ):
        batch = self.subagents.execute_many(
            DelegationRequest(
                delegation_id=f"{spec.deliberation_id}:{stage}",
                root_run_id=context.run_id,
                parent_run_id=context.run_id,
                tasks=tasks,
                max_concurrency=len(tasks),
            ),
            context=context,
            providers=providers,
            event_sink=event_sink,
        )
        by_task = {item.task_id: item for item in batch.results}
        failed = [task for task in tasks if by_task.get(task.id) is None or by_task[task.id].status != "completed"]
        retries = 0
        calls = len(tasks)
        if failed:
            retries = len(failed)
            # A parallel provider burst can fail as one unit. Retry failed specialists
            # serially so the recovery path does not reproduce the same upstream load.
            for index, task in enumerate(failed, start=1):
                if should_stop is not None and should_stop():
                    break
                retry_batch = self.subagents.execute_many(
                    DelegationRequest(
                        delegation_id=f"{spec.deliberation_id}:{stage}:retry-{index}",
                        root_run_id=context.run_id,
                        parent_run_id=context.run_id,
                        tasks=[task],
                        max_concurrency=1,
                    ),
                    context=context,
                    providers=providers,
                    event_sink=event_sink,
                )
                calls += 1
                for item in retry_batch.results:
                    if item.status == "completed" or item.task_id not in by_task:
                        by_task[item.task_id] = item
        return [by_task[task.id] for task in tasks if task.id in by_task], calls, retries

    @staticmethod
    def _moderator_decision(moderator: dict[str, Any], round_index: int) -> str:
        decision = str(moderator.get("decision") or "").strip().lower()
        if decision in {"continue", "targeted_rebuttal", "synthesize"}:
            return decision
        return "continue" if round_index < 2 else "synthesize"

    @staticmethod
    def _target_participants(spec, moderator):
        requested = set(_string_list(moderator.get("target_agent_ids")))
        selected = [item for item in spec.participants if item.agent_id in requested]
        return selected if selected else spec.participants

    def _finish(self, spec, state, moderator, reason, sink, checkpoint_sink):
        arguments = state["arguments"]
        synthesis = state.get("synthesis") or {}
        completed = sum(item.status == "completed" for item in arguments)
        status: DeliberationStatus = (
            "completed"
            if completed == len(arguments) and synthesis.get("status") == "completed"
            else "partial"
            if completed
            else "failed"
        )
        diagnostics = self._diagnostics(spec, state, status=status, stop_reason=reason)
        result = DeliberationResult(
            deliberation_id=spec.deliberation_id,
            status=status,
            stop_reason=reason,
            question=spec.question,
            participants=spec.participants,
            arguments=arguments,
            moderator=moderator,
            moderation_history=state["moderation_history"],
            synthesis=synthesis,
            model_calls=state["model_calls"],
            retry_count=state["retry_count"],
            diagnostics=diagnostics,
        )
        self._emit(sink, "deliberation.completed", "Deliberation completed", spec, {
            "status": status,
            "stop_reason": reason,
            "summary": synthesis.get("summary", ""),
            "diagnostics": diagnostics,
        })
        state["result"] = result.model_dump(mode="json")
        state["completed_stages"] = list(dict.fromkeys([*state["completed_stages"], "completed"]))
        self._checkpoint(checkpoint_sink, "completed", state)
        return result

    @staticmethod
    def _restore_state(raw):
        source = raw if isinstance(raw, dict) else {}
        arguments = []
        for item in source.get("arguments") or []:
            try:
                arguments.append(DeliberationArgument.model_validate(item))
            except Exception:
                continue
        return {
            "arguments": arguments,
            "moderation_history": list(source.get("moderation_history") or []),
            "synthesis": dict(source.get("synthesis") or {}),
            "model_calls": int(source.get("model_calls") or 0),
            "retry_count": int(source.get("retry_count") or 0),
            "completed_stages": list(source.get("completed_stages") or []),
            "profile": dict(source.get("profile") or {}),
            "resume_count": int(source.get("resume_count") or 0) + (1 if source else 0),
            "recovered_argument_count": len(arguments) if source else 0,
        }

    @staticmethod
    def _checkpoint(sink, phase, state):
        if sink is None:
            return
        payload = {
            **state,
            "arguments": [item.model_dump(mode="json") for item in state["arguments"]],
        }
        sink(phase, copy.deepcopy(payload))

    @staticmethod
    def _diagnostics(spec, state, *, status, stop_reason):
        arguments = list(state.get("arguments") or [])
        failed = [item for item in arguments if item.status != "completed"]
        routes = []
        seen_routes = set()
        for item in arguments:
            route = (item.agent_id, item.model_role, item.model_id)
            if route in seen_routes:
                continue
            seen_routes.add(route)
            routes.append({
                "agent_id": item.agent_id,
                "model_role": item.model_role,
                "model_id": item.model_id,
            })
        return {
            "status": status,
            "stop_reason": stop_reason,
            "completed_stages": list(state.get("completed_stages") or []),
            "completed_argument_count": len(arguments) - len(failed),
            "failed_argument_count": len(failed),
            "failed_arguments": [
                {
                    "argument_id": item.argument_id,
                    "agent_id": item.agent_id,
                    "phase": item.phase,
                    "round_index": item.round_index,
                    "error": item.error,
                }
                for item in failed
            ],
            "model_routes": routes,
            "model_calls": int(state.get("model_calls") or 0),
            "retry_count": int(state.get("retry_count") or 0),
            "budget": {
                "maximum": spec.max_model_calls,
                "remaining": max(0, spec.max_model_calls - int(state.get("model_calls") or 0)),
                "synthesis_reserve_calls": spec.synthesis_reserve_calls,
            },
            "recovery": {
                "resume_count": int(state.get("resume_count") or 0),
                "recovered_argument_count": int(state.get("recovered_argument_count") or 0),
            },
            "participants": _participant_diagnostics(spec, arguments),
        }

    def _emit_stage_started(
        self,
        sink,
        spec,
        *,
        phase,
        round_index,
        participants=None,
        extra=None,
    ):
        selected = list(participants or [])
        self._emit(sink, "deliberation.stage.started", "Deliberation stage started", spec, {
            "phase": phase,
            "round_index": round_index,
            "participant_ids": [item.agent_id for item in selected],
            "participant_labels": [item.display_name for item in selected],
            **(extra or {}),
        })

    @staticmethod
    def _emit(sink, event_type, title, spec, payload):
        if sink is None:
            return
        sink({
            "event_type": event_type,
            "title": title,
            "summary": str(payload.get("summary") or title),
            "payload": {"visible": True, "deliberation_id": spec.deliberation_id, **payload},
        })


def _string_list(value) -> list[str]:
    return [str(item).strip() for item in value] if isinstance(value, list) else []


def _dict_list(value) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _participant_facts(facts: dict[str, Any], fact_keys: list[str]) -> dict[str, Any]:
    if not fact_keys:
        return copy.deepcopy(facts)
    selected = {
        key: copy.deepcopy(facts[key])
        for key in fact_keys
        if key in facts
    }
    # Provenance and subject identity are shared safety context, not optional
    # analytical detail. Preserve them whenever the host supplied them.
    for key in ("subject", "provenance", "snapshot_version", "availability"):
        if key in facts and key not in selected:
            selected[key] = copy.deepcopy(facts[key])
    return selected


def _argument_id(
    spec: DeliberationSpec,
    phase: DeliberationPhase,
    round_index: int,
    participant_instance_id: str,
) -> str:
    return f"{spec.deliberation_id}:{phase}:{round_index}:{participant_instance_id}"


def _merge_arguments(
    current: list[DeliberationArgument],
    incoming: list[DeliberationArgument],
) -> list[DeliberationArgument]:
    by_id = {item.argument_id: item for item in current}
    order = [item.argument_id for item in current]
    for item in incoming:
        if item.argument_id not in by_id:
            order.append(item.argument_id)
        previous = by_id.get(item.argument_id)
        if previous is None or item.status == "completed" or previous.status != "completed":
            by_id[item.argument_id] = item
    return [by_id[argument_id] for argument_id in order]


def _participant_diagnostics(
    spec: DeliberationSpec,
    arguments: list[DeliberationArgument],
) -> list[dict[str, Any]]:
    rows = []
    for participant in spec.participants:
        views = [
            item
            for item in arguments
            if item.participant_instance_id == participant.participant_instance_id
        ]
        completed = [item for item in views if item.status == "completed"]
        latest = views[-1] if views else None
        rows.append(
            {
                "agent_id": participant.agent_id,
                "participant_instance_id": participant.participant_instance_id,
                "display_name": participant.display_name,
                "status": "completed" if completed else latest.status if latest else "pending",
                "completed_rounds": sorted({item.round_index for item in completed}),
                "model_role": latest.model_role if latest else "",
                "model_id": latest.model_id if latest else "",
                "duration_ms": sum(item.duration_ms for item in views),
                "error": latest.error if latest and latest.status != "completed" else "",
            }
        )
    return rows


def _score(value) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
