from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from harness_core.deliberation.contracts import (
    DeliberationArgument,
    DeliberationResult,
    DeliberationSpec,
)
from harness_core.subagents import DelegationRequest, DelegationTask, SubAgentExecutor
from harness_core.tools import ToolExecutionContext


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
        self._emit(event_sink, "deliberation.started", "多学派辩论已开始", spec, {
            "participants": [item.model_dump(mode="json") for item in spec.participants],
        })

        if "opening" not in state["completed_stages"]:
            opening, calls, retries = self._participant_batch(
                spec,
                phase="opening",
                round_index=1,
                context=context,
                providers=providers,
                event_sink=event_sink,
                prior_arguments=[],
            )
            state["arguments"].extend(opening)
            state["model_calls"] += calls
            state["retry_count"] += retries
            state["completed_stages"].append("opening")
            self._checkpoint(checkpoint_sink, "opening", state)
        opening = [item for item in state["arguments"] if item.phase == "opening"]
        if not any(item.status == "completed" for item in opening):
            return self._finish(spec, state, {}, "no_opening_arguments", event_sink, checkpoint_sink)

        round_index = max(1, max((item.round_index for item in state["arguments"]), default=1))
        moderation_key = f"moderate:{round_index}"
        moderator = state["moderation_history"][-1] if state["moderation_history"] else {}
        if moderation_key not in state["completed_stages"]:
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

        stop_reason = "moderator_converged"
        while True:
            decision = self._moderator_decision(moderator, round_index)
            budget_needed = len(spec.participants) + 2
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
                targets = self._target_participants(spec, moderator)
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
                )
                state["arguments"].extend(rebuttals)
                state["model_calls"] += calls
                state["retry_count"] += retries
                state["completed_stages"].append(rebuttal_key)
                self._checkpoint(checkpoint_sink, "rebuttal", state)
            if stop_requested():
                stop_reason = "user_stop_and_summarize"
                break

            moderation_key = f"moderate:{round_index}"
            if moderation_key not in state["completed_stages"]:
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
        phase: str,
        round_index: int,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: EventSink | None,
        prior_arguments: list[DeliberationArgument],
        moderator: dict[str, Any] | None = None,
        participants=None,
    ) -> tuple[list[DeliberationArgument], int, int]:
        selected = list(participants or spec.participants)
        tasks = [self._participant_task(spec, item, phase, round_index, prior_arguments, moderator) for item in selected]
        results, calls, retries = self._execute_with_retry(
            spec,
            tasks,
            context=context,
            providers=providers,
            event_sink=event_sink,
            stage=f"{phase}:{round_index}",
        )
        by_agent = {item.agent_id: item for item in results}
        arguments = []
        for participant in selected:
            item = by_agent.get(participant.agent_id)
            argument = DeliberationArgument(
                argument_id=f"{spec.deliberation_id}:{phase}:{round_index}:{participant.participant_instance_id}",
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
            f"以{participant.label}立场独立提出首轮判断：{spec.question}"
            if phase == "opening"
            else f"以{participant.label}立场回应主持人指定的未决分歧：{spec.question}"
        )
        return DelegationTask(
            id=f"{phase}-{round_index}-{participant.participant_instance_id}",
            agent_id=participant.agent_id,
            objective=objective,
            input_data={
                "question": spec.question,
                "fact_packet": spec.fact_packet,
                "phase": phase,
                "round_index": round_index,
                "other_views": [item.model_dump(mode="json") for item in prior_arguments],
                "moderator": moderator or {},
            },
            constraints=[
                "所有参与者以同一份事实包为基线，不得补造外部事实",
                "事实包缺少完成本轮判断所需的信息时，只能使用允许的只读工具补证，并明确工具来源",
                "回应观点而不是评价其他参与者身份，不以投票决定结论",
            ],
        )

    def _moderate(self, spec, arguments, context, providers, event_sink, *, phase: str, round_index: int):
        objective = (
            "识别事实共识、实质分歧并决定继续讨论、定向回应或直接总结。"
            if phase == "moderate"
            else "综合全部发言，形成直接主结论、共同依据、保留分歧、成立条件、行动建议和判断边界。"
        )
        task = DelegationTask(
            id=f"{phase}-{round_index}-moderator",
            agent_id=spec.moderator_agent_id,
            objective=objective,
            input_data={
                "question": spec.question,
                "fact_packet": spec.fact_packet,
                "participant_views": [item.model_dump(mode="json") for item in arguments],
                "phase": phase,
                "round_index": round_index,
                "max_rounds": spec.max_rounds,
            },
            constraints=[
                "不按票数裁决，不引入事实包或只读工具结果之外的新外部事实",
                "moderate 阶段必须明确 decision、未决问题和需要回应的 agent_id",
            ],
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
            "convergence_score": _score(output.get("convergence_score")),
            "conditions": _string_list(output.get("conditions")),
            "action_recommendations": _string_list(output.get("action_recommendations")),
            "judgment_boundary": str(output.get("judgment_boundary") or "").strip(),
            "round_index": round_index,
            "error": item.error if item else "moderator result is missing",
        }
        self._emit(event_sink, f"deliberation.{phase}.completed", "主 Agent 已完成阶段整理", spec, payload)
        return payload, calls, retries

    def _execute_with_retry(self, spec, tasks, *, context, providers, event_sink, stage):
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
        status = "completed" if completed == len(arguments) and synthesis.get("status") == "completed" else "partial" if completed else "failed"
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
        )
        self._emit(sink, "deliberation.completed", "多学派辩论已完成", spec, {
            "status": status,
            "stop_reason": reason,
            "summary": synthesis.get("summary", ""),
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


def _score(value) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
