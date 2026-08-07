from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from deepkeel.context_contracts import ContextCheckpoint
from deepkeel.context_contracts import ModelContextProfile
from deepkeel.context_planning import ContextBudgetPlanner


class TokenEstimatorLike(Protocol):
    def estimate(self, value: Any) -> int: ...


@dataclass(frozen=True, slots=True)
class ContextCompactionResult:
    retained_messages: list[dict[str, Any]]
    checkpoint: ContextCheckpoint | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelInputContextResult:
    messages: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class ContextInputBudgetError(RuntimeError):
    code = "CONTEXT_INPUT_BUDGET_EXCEEDED"


class WorkingContextCompactor(Protocol):
    def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        token_budget: int,
        thread_id: str = "",
        subject_id: str = "",
        previous_checkpoint: ContextCheckpoint | None = None,
        protect_marked_messages: bool = True,
    ) -> ContextCompactionResult: ...


class DeterministicWorkingContextCompactor:
    """Token-aware L2 reducer that never splits tool-call/result groups."""

    def __init__(self, estimator: TokenEstimatorLike | None = None) -> None:
        if estimator is None:
            from deepkeel.context_window import ConservativeTokenEstimator

            estimator = ConservativeTokenEstimator()
        self.estimator = estimator

    def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        token_budget: int,
        thread_id: str = "",
        subject_id: str = "",
        previous_checkpoint: ContextCheckpoint | None = None,
        protect_marked_messages: bool = True,
    ) -> ContextCompactionResult:
        valid = [copy.deepcopy(item) for item in messages if _valid_message(item)]
        original_tokens = self.estimator.estimate(valid) if valid else 0
        if original_tokens <= max(0, int(token_budget)):
            return ContextCompactionResult(
                retained_messages=valid,
                checkpoint=previous_checkpoint,
                diagnostics={
                    "triggered": False,
                    "original_tokens": original_tokens,
                    "final_tokens": original_tokens,
                    "omitted_count": 0,
                    "atomic_group_count": len(_atomic_message_groups(valid)),
                },
            )

        groups = _atomic_message_groups(valid)
        protected_indexes = {
            index
            for index, group in enumerate(groups)
            if protect_marked_messages and any(bool(item.get("_context_protected")) for item in group)
        }
        protected_tokens = sum(self.estimator.estimate(groups[index]) for index in protected_indexes)
        if protected_tokens > token_budget:
            raise ContextInputBudgetError(
                "protected context exceeds the selected model input budget"
            )
        selected_indexes = set(protected_indexes)
        used = protected_tokens
        for index in range(len(groups) - 1, -1, -1):
            if index in selected_indexes:
                continue
            group = groups[index]
            cost = self.estimator.estimate(group)
            if used + cost <= token_budget:
                selected_indexes.add(index)
                used += cost
                continue
            remaining = max(0, token_budget - used)
            compacted = _compact_oversized_group(group, remaining, self.estimator)
            if compacted:
                groups[index] = compacted
                selected_indexes.add(index)
                used += self.estimator.estimate(compacted)
            break
        retained_groups = [groups[index] for index in sorted(selected_indexes)]
        retained = [item for group in retained_groups for item in group]
        omitted = [
            item
            for index, group in enumerate(groups)
            if index not in selected_indexes
            for item in group
        ]
        omitted_count = len(omitted)
        checkpoint = previous_checkpoint
        if omitted:
            checkpoint = _checkpoint_from_messages(
                omitted,
                retained,
                thread_id=thread_id,
                subject_id=subject_id,
                previous_checkpoint=previous_checkpoint,
            )
        return ContextCompactionResult(
            retained_messages=retained,
            checkpoint=checkpoint,
            diagnostics={
                "triggered": True,
                "original_tokens": original_tokens,
                "final_tokens": self.estimator.estimate(retained) if retained else 0,
                "omitted_count": omitted_count,
                "atomic_group_count": len(groups),
                "retained_group_count": len(retained_groups),
                "protected_group_count": len(protected_indexes),
                "checkpoint_id": checkpoint.checkpoint_id if checkpoint is not None else "",
                "previous_checkpoint_id": (
                    checkpoint.previous_checkpoint_id if checkpoint is not None else ""
                ),
                "subject_id": checkpoint.subject_id if checkpoint is not None else subject_id,
                "covered_event_range": (
                    list(checkpoint.covered_event_range)
                    if checkpoint is not None
                    else []
                ),
                "first_kept_event_id": (
                    checkpoint.first_kept_event_id if checkpoint is not None else ""
                ),
            },
        )


def prepare_model_input_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    profile: ModelContextProfile,
    configured_input_limit: int | None = None,
    estimator: TokenEstimatorLike | None = None,
    thread_id: str = "",
    subject_id: str = "",
) -> ModelInputContextResult:
    """Apply the final model-specific budget after routing selected a provider."""

    if estimator is None:
        from deepkeel.context_window import ConservativeTokenEstimator

        estimator = ConservativeTokenEstimator()
    planner = ContextBudgetPlanner()
    tool_tokens = estimator.estimate(tools) if tools else 0
    l1_messages = [
        copy.deepcopy(item)
        for item in messages
        if str(item.get("role") or "") == "system"
        and str(item.get("_context_tier") or "") == "L1"
    ]
    l2_system = [
        copy.deepcopy(item)
        for item in messages
        if str(item.get("role") or "") == "system"
        and str(item.get("_context_tier") or "") == "L2"
    ]
    l3_system = [
        copy.deepcopy(item)
        for item in messages
        if str(item.get("role") or "") == "system"
        and str(item.get("_context_tier") or "") == "L3"
    ]
    conversation = [
        copy.deepcopy(item)
        for item in messages
        if not (
            str(item.get("role") or "") == "system"
            and str(item.get("_context_tier") or "") in {"L1", "L2", "L3"}
        )
    ]
    current_turn_start = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if str(conversation[index].get("role") or "") == "user"
        ),
        None,
    )
    if current_turn_start is not None:
        for message in conversation[current_turn_start:]:
            message["_context_protected"] = True
    l1_tokens = estimator.estimate(l1_messages) if l1_messages else 0
    plan = planner.plan(
        profile,
        l1_required_tokens=l1_tokens,
        tool_schema_tokens=tool_tokens,
        configured_input_limit=configured_input_limit,
    )
    if l1_tokens > plan.available_input_tokens:
        raise ContextInputBudgetError(
            "L1 control context exceeds the selected model input budget"
        )

    l2_system, l2_system_dropped = _fit_optional_system_messages(
        l2_system,
        max(0, plan.available_input_tokens - l1_tokens),
        estimator,
        compact=True,
    )
    base_tokens = l1_tokens + (estimator.estimate(l2_system) if l2_system else 0)
    conversation_budget = max(0, plan.available_input_tokens - base_tokens)
    compaction = DeterministicWorkingContextCompactor(estimator).compact(
        conversation,
        token_budget=conversation_budget,
        thread_id=thread_id,
        subject_id=subject_id,
    )
    retained_conversation = compaction.retained_messages
    checkpoint_message: list[dict[str, Any]] = []
    if compaction.checkpoint is not None:
        checkpoint_reserve = min(
            2_048,
            max(96, int(conversation_budget * 0.25)),
        )
        compaction = DeterministicWorkingContextCompactor(estimator).compact(
            conversation,
            token_budget=max(0, conversation_budget - checkpoint_reserve),
            thread_id=thread_id,
            subject_id=subject_id,
        )
        retained_conversation = compaction.retained_messages
        checkpoint_message = _checkpoint_prompt_message(
            compaction.checkpoint,
            checkpoint_reserve,
            estimator,
        )

    used_without_l3 = estimator.estimate(
        [*l1_messages, *l2_system, *checkpoint_message, *retained_conversation]
    )
    l3_budget = max(0, plan.available_input_tokens - used_without_l3)
    l3_system, l3_dropped = _fit_optional_system_messages(
        l3_system,
        l3_budget,
        estimator,
        compact=False,
    )
    final = [
        *l1_messages,
        *l2_system,
        *checkpoint_message,
        *l3_system,
        *retained_conversation,
    ]
    final = [_strip_internal_context_metadata(item) for item in final]
    original_tokens = estimator.estimate(
        [_strip_internal_context_metadata(item) for item in messages]
    )
    final_tokens = estimator.estimate(final)
    if final_tokens > plan.available_input_tokens:
        raise ContextInputBudgetError(
            "prepared context exceeds the selected model input budget"
        )
    return ModelInputContextResult(
        messages=final,
        diagnostics={
            "schema_version": "harness-model-context-v1",
            "budget_plan": plan.as_dict(),
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "over_budget": final_tokens > plan.available_input_tokens,
            "tiers": {
                "L1": {
                    "tokens": l1_tokens,
                    "message_count": len(l1_messages),
                },
                "L2": {
                    "tokens": estimator.estimate(
                        [*l2_system, *checkpoint_message, *retained_conversation]
                    ),
                    "message_count": len(l2_system)
                    + len(checkpoint_message)
                    + len(retained_conversation),
                    "compaction": compaction.diagnostics,
                    "optional_system_dropped": l2_system_dropped,
                },
                "L3": {
                    "tokens": estimator.estimate(l3_system) if l3_system else 0,
                    "message_count": len(l3_system),
                    "dropped": l3_dropped,
                },
            },
        },
    )


def _fit_optional_system_messages(
    messages: list[dict[str, Any]],
    budget: int,
    estimator: TokenEstimatorLike,
    *,
    compact: bool,
) -> tuple[list[dict[str, Any]], int]:
    retained: list[dict[str, Any]] = []
    remaining = max(0, int(budget))
    for message in messages:
        cost = estimator.estimate(message)
        if cost <= remaining:
            retained.append(message)
            remaining -= cost
            continue
        if compact and remaining >= 32:
            compacted = _compact_oversized_group([message], remaining, estimator)
            if compacted and estimator.estimate(compacted) <= remaining:
                retained.extend(compacted)
                remaining -= estimator.estimate(compacted)
        break
    return retained, max(0, len(messages) - len(retained))


def _checkpoint_prompt_message(
    checkpoint: ContextCheckpoint | None,
    token_budget: int,
    estimator: TokenEstimatorLike,
) -> list[dict[str, Any]]:
    if checkpoint is None or token_budget <= 0:
        return []
    prefix = "L2 working checkpoint (derived; raw events remain authoritative):\n"
    payload = checkpoint.as_dict()
    candidates = [
        payload,
        {
            **payload,
            "critical_facts": payload["critical_facts"][-6:],
            "progress": {
                "done": payload["progress"]["done"][-2:],
                "in_progress": payload["progress"]["in_progress"],
                "blocked": payload["progress"]["blocked"],
            },
        },
        {
            "checkpoint_id": payload["checkpoint_id"],
            "subject_id": payload["subject_id"],
            "goal": payload["goal"],
            "critical_facts": payload["critical_facts"][-3:],
            "covered_event_range": payload["covered_event_range"],
            "first_kept_event_id": payload["first_kept_event_id"],
            "source_fingerprint": payload["source_fingerprint"],
        },
    ]
    for candidate in candidates:
        message = {
            "role": "system",
            "content": prefix + json.dumps(candidate, ensure_ascii=False, default=str),
            "_context_tier": "L2",
        }
        if estimator.estimate([message]) <= token_budget:
            return [message]
    minimal = json.dumps(candidates[-1], ensure_ascii=False, default=str)
    compacted = _head_tail(minimal, max(1, token_budget - estimator.estimate(prefix)), estimator)
    message = {
        "role": "system",
        "content": prefix + compacted,
        "_context_tier": "L2",
    }
    return [message] if estimator.estimate([message]) <= token_budget else []


def _strip_internal_context_metadata(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in message.items()
        if not str(key).startswith("_context_")
    }


def _valid_message(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and str(value.get("role") or "") in {"system", "user", "assistant", "tool"}
        and (
            value.get("content") not in (None, "", [], {})
            or bool(value.get("tool_calls"))
        )
    )


def _atomic_message_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    open_tool_call_ids: set[str] = set()
    for message in messages:
        role = str(message.get("role") or "")
        if role == "tool" and groups and open_tool_call_ids:
            groups[-1].append(message)
            call_id = str(message.get("tool_call_id") or "")
            if call_id:
                open_tool_call_ids.discard(call_id)
            continue
        groups.append([message])
        open_tool_call_ids = {
            str(item.get("id") or "")
            for item in message.get("tool_calls") or []
            if isinstance(item, dict) and item.get("id")
        }
    return groups


def _compact_oversized_group(
    group: list[dict[str, Any]],
    budget: int,
    estimator: TokenEstimatorLike,
) -> list[dict[str, Any]]:
    result = copy.deepcopy(group)
    if not result or budget <= 0:
        return []
    # Preserve every envelope in an atomic group. Dropping the assistant tool
    # call while retaining its result creates provider-invalid history.
    structural = copy.deepcopy(result)
    for message in structural:
        if isinstance(message.get("content"), str):
            message["content"] = ""
    if estimator.estimate(structural) > budget:
        return []
    per_message = max(1, budget // len(result))
    for message in result:
        content = message.get("content")
        if not isinstance(content, str) or estimator.estimate(content) <= per_message:
            continue
        message["content"] = _head_tail(content, per_message, estimator)
        message["context_compacted"] = True
    if estimator.estimate(result) > budget:
        for message in result:
            if isinstance(message.get("content"), str) and message.get("content"):
                message["content"] = "[context compacted]"
                message["context_compacted"] = True
    return result if estimator.estimate(result) <= budget else []


def _head_tail(value: str, budget: int, estimator: TokenEstimatorLike) -> str:
    marker = "\n...[context compacted]...\n"
    if estimator.estimate(marker) >= budget:
        return marker.strip()
    low, high = 0, len(value) // 2
    best = marker
    while low <= high:
        size = (low + high) // 2
        candidate = f"{value[:size].rstrip()}{marker}{value[-size:].lstrip()}"
        if estimator.estimate(candidate) <= budget:
            best = candidate
            low = size + 1
        else:
            high = size - 1
    return best


def _checkpoint_from_messages(
    omitted: list[dict[str, Any]],
    retained: list[dict[str, Any]],
    *,
    thread_id: str,
    subject_id: str,
    previous_checkpoint: ContextCheckpoint | None,
) -> ContextCheckpoint:
    first_id = str(omitted[0].get("id") or "") if omitted else ""
    last_id = str(omitted[-1].get("id") or "") if omitted else ""
    first_kept_id = str(retained[0].get("id") or "") if retained else ""
    user_messages = [
        _message_excerpt(item)
        for item in omitted
        if str(item.get("role") or "") == "user"
    ]
    critical_facts = tuple(previous_checkpoint.critical_facts) if previous_checkpoint else ()
    critical_facts += tuple(
        {
            "value": _message_excerpt(item),
            "source_ref": str(item.get("id") or "conversation-history"),
            "role": str(item.get("role") or ""),
        }
        for item in omitted[-8:]
        if str(item.get("role") or "") in {"user", "tool"}
        if _message_excerpt(item)
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "previous": previous_checkpoint.source_fingerprint if previous_checkpoint else "",
                "events": omitted,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return ContextCheckpoint(
        checkpoint_id=f"context-{fingerprint[:20]}",
        thread_id=thread_id,
        subject_id=subject_id,
        goal=(previous_checkpoint.goal if previous_checkpoint and previous_checkpoint.goal else user_messages[-1] if user_messages else ""),
        constraints_and_preferences=(
            previous_checkpoint.constraints_and_preferences if previous_checkpoint else ()
        ),
        done=previous_checkpoint.done if previous_checkpoint else (),
        in_progress=previous_checkpoint.in_progress if previous_checkpoint else (),
        blocked=previous_checkpoint.blocked if previous_checkpoint else (),
        key_decisions=previous_checkpoint.key_decisions if previous_checkpoint else (),
        pending_actions=previous_checkpoint.pending_actions if previous_checkpoint else (),
        open_questions=previous_checkpoint.open_questions if previous_checkpoint else (),
        critical_facts=critical_facts[-16:],
        artifacts=previous_checkpoint.artifacts if previous_checkpoint else (),
        failed_attempts=previous_checkpoint.failed_attempts if previous_checkpoint else (),
        next_steps=previous_checkpoint.next_steps if previous_checkpoint else (),
        covered_event_range=(
            previous_checkpoint.covered_event_range[0]
            if previous_checkpoint and previous_checkpoint.covered_event_range[0]
            else first_id,
            last_id or (previous_checkpoint.covered_event_range[1] if previous_checkpoint else ""),
        ),
        first_kept_event_id=first_kept_id,
        source_fingerprint=fingerprint,
        previous_checkpoint_id=(
            previous_checkpoint.checkpoint_id if previous_checkpoint is not None else ""
        ),
    )


def _message_excerpt(message: dict[str, Any], limit: int = 240) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    normalized = " ".join(content.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 3].rstrip()}..."
