from __future__ import annotations

import pytest

from harness_core.context import build_initial_messages
from harness_core.context_compaction import (
    DeterministicWorkingContextCompactor,
    prepare_model_input_context,
)
from harness_core.context_contracts import ContextCheckpoint, ContextItem, ModelContextProfile
from harness_core.context_compaction import ContextInputBudgetError
from harness_core.context_validation import validate_context_items
from harness_core.context_planning import ContextBudgetPlanner
from harness_core.context_window import (
    ConservativeTokenEstimator,
    ContextWindowPolicy,
    DeterministicContextWindowManager,
)


def test_context_attributes_are_orthogonal_and_subjects_are_validated() -> None:
    item = ContextItem(
        key="friend_chart",
        value={"day_master": "jia"},
        tier="L2",
        scope="thread",
        visibility="both",
        retention="protected",
        representation="digest",
        authority="derived",
        subject_id="friend",
        source_ref="artifact:chart-1",
    )

    result = validate_context_items([item], active_subject_id="self")

    assert item.model_visible is True
    assert item.protected is True
    assert result.valid is False
    assert result.errors == ("subject mismatch for friend_chart",)


def test_configured_input_limit_caps_input_without_double_reserving_output() -> None:
    plan = ContextBudgetPlanner().plan(
        ModelContextProfile(
            context_window_tokens=32_000,
            max_output_tokens=8_000,
        ),
        configured_input_limit=10_000,
    )

    assert plan.context_window_tokens == 32_000
    assert plan.available_input_tokens == 10_000


def test_context_window_uses_token_budget_instead_of_a_fixed_message_count() -> None:
    manager = DeterministicContextWindowManager(
        ContextWindowPolicy(
            max_input_tokens=20_000,
            reserved_output_tokens=1_000,
            history_limit=0,
            working_memory_ratio=0.8,
        )
    )
    recent = [
        {"id": f"m-{index}", "role": "user", "content": f"message {index}"}
        for index in range(20)
    ]

    result = manager.prepare("current", {}, {"recent_messages": recent})

    assert len(result.context_bundle["recent_messages"]) == 20
    messages = build_initial_messages("current", {}, result.context_bundle)
    assert sum(message.metadata.get("history") is True for message in messages) == 20


def test_runtime_only_context_is_not_injected_into_model_payload() -> None:
    manager = DeterministicContextWindowManager()

    result = manager.prepare(
        "hello",
        {},
        {
            "context_segments": [
                {
                    "key": "secret_handle",
                    "value": "runtime-secret-reference",
                    "tier": "L1",
                    "visibility": "runtime",
                },
                {
                    "key": "policy",
                    "value": "answer safely",
                    "tier": "L1",
                    "visibility": "model",
                },
            ]
        },
    )

    assert result.context_bundle["runtime_only_context"] == {
        "secret_handle": "runtime-secret-reference"
    }
    assert "secret_handle" not in result.context_bundle["runtime_context"]
    assert result.diagnostics["context_manifest"]["decisions"][0]["action"] in {
        "retained",
        "runtime_only",
    }


def test_subject_mismatch_is_quarantined_before_model_input() -> None:
    manager = DeterministicContextWindowManager()

    result = manager.prepare(
        "continue",
        {},
        {
            "subject_context": {"subject_id": "self"},
            "context_segments": [
                {
                    "key": "friend_chart",
                    "value": {"day_master": "yi"},
                    "tier": "L1",
                    "subject_id": "friend",
                    "source_ref": "chart:friend",
                }
            ],
        },
    )

    assert "friend_chart" not in result.context_bundle["runtime_context"]
    assert result.context_bundle["quarantined_context"]["friend_chart"] == {
        "day_master": "yi"
    }
    assert result.diagnostics["validation"]["valid"] is False


def test_working_context_compaction_keeps_tool_exchange_atomic_and_stable() -> None:
    estimator = ConservativeTokenEstimator()
    tool_group = [
        {
            "id": "assistant-tool",
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "name": "search", "arguments": {}}],
        },
        {
            "id": "tool-result",
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "authoritative observation",
        },
        {"id": "latest", "role": "user", "content": "continue"},
    ]
    messages = [
        {"id": "old", "role": "user", "content": "x" * 800},
        *tool_group,
    ]
    budget = estimator.estimate(tool_group) + 8
    compactor = DeterministicWorkingContextCompactor(estimator)

    first = compactor.compact(messages, token_budget=budget, thread_id="thread-1")
    second = compactor.compact(messages, token_budget=budget, thread_id="thread-1")

    retained_ids = [item["id"] for item in first.retained_messages]
    assert retained_ids == ["assistant-tool", "tool-result", "latest"]
    assert first.checkpoint is not None
    assert second.checkpoint is not None
    assert first.checkpoint.checkpoint_id == second.checkpoint.checkpoint_id


def test_model_specific_context_drops_l3_before_recent_l2() -> None:
    estimator = ConservativeTokenEstimator()
    messages = [
        {"role": "system", "content": "core policy", "_context_tier": "L1"},
        {"role": "system", "content": "memory " * 600, "_context_tier": "L3"},
        {"id": "u-1", "role": "user", "content": "old question " * 120},
        {"id": "a-1", "role": "assistant", "content": "old answer " * 120},
        {"id": "u-2", "role": "user", "content": "current question"},
    ]

    result = prepare_model_input_context(
        messages,
        [],
        profile=ModelContextProfile(
            model_id="small",
            context_window_tokens=900,
            max_output_tokens=160,
            source="test",
        ),
        estimator=estimator,
        thread_id="thread-1",
    )

    assert result.diagnostics["tiers"]["L3"]["dropped"] == 1
    assert result.messages[-1]["content"] == "current question"
    assert all("_context_tier" not in item for item in result.messages)
    assert result.diagnostics["final_tokens"] <= result.diagnostics["budget_plan"][
        "available_input_tokens"
    ]


def test_dynamic_system_repair_message_keeps_chronological_position() -> None:
    messages = [
        {"role": "system", "content": "constitution", "_context_tier": "L1"},
        {"id": "user", "role": "user", "content": "build report"},
        {"id": "answer", "role": "assistant", "content": "premature answer"},
        {"id": "repair", "role": "system", "content": "call required tool"},
    ]

    result = prepare_model_input_context(
        messages,
        [],
        profile=ModelContextProfile(context_window_tokens=8_000, max_output_tokens=1_000),
    )

    assert result.messages[-1]["content"] == "call required tool"


def test_oversized_tool_exchange_never_leaves_orphan_tool_result() -> None:
    estimator = ConservativeTokenEstimator()
    group = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "name": "search", "arguments": {"query": "x" * 800}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "result " * 800,
        },
    ]

    result = DeterministicWorkingContextCompactor(estimator).compact(
        group,
        token_budget=120,
    )

    assert result.retained_messages == [] or [item["role"] for item in result.retained_messages] == [
        "assistant",
        "tool",
    ]


def test_protected_current_request_is_never_silently_truncated() -> None:
    messages = [
        {"role": "system", "content": "policy", "_context_tier": "L1"},
        {
            "role": "user",
            "content": "important request " * 2_000,
            "_context_protected": True,
        },
    ]

    try:
        prepare_model_input_context(
            messages,
            [],
            profile=ModelContextProfile(context_window_tokens=1_000, max_output_tokens=200),
        )
    except ContextInputBudgetError as exc:
        assert "protected context" in str(exc)
    else:
        raise AssertionError("protected request must fail instead of being truncated")


def test_prepare_model_input_protects_current_turn_tool_observations() -> None:
    messages = [
        {"role": "user", "content": "current request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "name": "search", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 400},
    ]

    with pytest.raises(ContextInputBudgetError):
        prepare_model_input_context(
            messages,
            [],
            profile=ModelContextProfile(
                context_window_tokens=180,
                max_output_tokens=32,
            ),
        )


def test_context_window_removes_current_request_from_history_before_budgeting() -> None:
    manager = DeterministicContextWindowManager()

    result = manager.prepare(
        "current question",
        {},
        {
            "recent_messages": [
                {"id": "old", "role": "assistant", "content": "previous answer"},
                {"id": "current", "role": "user", "content": "current question"},
            ]
        },
    )

    assert [item["id"] for item in result.context_bundle["recent_messages"]] == ["old"]
    assert result.diagnostics["current_message_removed_from_history"] is True


def test_checkpoint_compaction_chains_previous_checkpoint_without_marking_answers_done() -> None:
    previous = ContextCheckpoint(
        checkpoint_id="checkpoint-1",
        thread_id="thread-1",
        subject_id="subject-1",
        goal="finish the report",
        done=("validated input",),
        covered_event_range=("m-1", "m-2"),
        source_fingerprint="previous-fingerprint",
    )
    messages = [
        {"id": "m-3", "role": "assistant", "content": "unverified draft"},
        {"id": "m-4", "role": "user", "content": "keep the original scope"},
        {"id": "m-5", "role": "user", "content": "continue", "_context_protected": True},
    ]

    result = DeterministicWorkingContextCompactor().compact(
        messages,
        token_budget=20,
        thread_id="thread-1",
        subject_id="subject-1",
        previous_checkpoint=previous,
    )

    assert result.checkpoint is not None
    assert result.checkpoint.previous_checkpoint_id == "checkpoint-1"
    assert result.checkpoint.goal == "finish the report"
    assert result.checkpoint.done == ("validated input",)
