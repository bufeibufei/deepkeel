from __future__ import annotations

from harness_core.context import build_initial_messages
from harness_core.context_compaction import (
    DeterministicWorkingContextCompactor,
    prepare_model_input_context,
)
from harness_core.context_contracts import ContextItem, ModelContextProfile
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
