from deepkeel.budget import BudgetPolicy, BudgetSnapshot, OUTPUT_TOKENS
from deepkeel.model import (
    _context_output_capacity,
    _reasoning_effort,
    _remaining_output_tokens,
)
from deepkeel.model_capabilities import ModelCapabilities


def test_generation_limit_uses_smallest_physical_and_user_boundary() -> None:
    policy = BudgetPolicy(
        max_output_tokens_total=20_000,
        max_output_tokens_per_call=12_000,
    )
    snapshot = BudgetSnapshot(
        run_id="run-1",
        usage={OUTPUT_TOKENS: 3_000},
    )
    capabilities = ModelCapabilities(
        context_window_tokens=16_000,
        max_output_tokens=10_000,
    )

    assert _remaining_output_tokens(
        policy,
        snapshot,
        "reasoning",
        capabilities=capabilities,
        estimated_input_tokens=7_000,
    ) == 6_952


def test_unlimited_user_budget_uses_safe_reasoning_call_default() -> None:
    capabilities = ModelCapabilities(
        context_window_tokens=256_000,
        max_output_tokens=128_000,
    )

    assert _remaining_output_tokens(
        BudgetPolicy(),
        BudgetSnapshot(run_id="run-1"),
        "reasoning",
        capabilities=capabilities,
        estimated_input_tokens=32_000,
    ) == 16_384


def test_unlimited_user_budget_uses_smaller_fast_call_default() -> None:
    assert _remaining_output_tokens(
        BudgetPolicy(),
        BudgetSnapshot(run_id="run-1"),
        "fast",
        capabilities=ModelCapabilities(
            context_window_tokens=1_024_000,
            max_output_tokens=128_000,
        ),
        estimated_input_tokens=44_516,
    ) == 8_192


def test_context_capacity_keeps_a_safety_reserve() -> None:
    assert _context_output_capacity(16_000, 7_000) == 6_952


def test_reasoning_effort_is_only_emitted_when_supported() -> None:
    supported = ModelCapabilities(supports_reasoning_effort=True)
    unknown = ModelCapabilities()

    assert _reasoning_effort(supported, "fast") == "low"
    assert _reasoning_effort(supported, "reasoning") == "high"
    assert _reasoning_effort(unknown, "reasoning") is None
