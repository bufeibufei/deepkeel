from __future__ import annotations

from dataclasses import dataclass

from harness_core.context_contracts import ContextBudgetPlan, ModelContextProfile


@dataclass(frozen=True, slots=True)
class ContextPlanningPolicy:
    fallback_context_window_tokens: int = 24_000
    fallback_output_reserve_tokens: int = 4_000
    tool_loop_reserve_tokens: int = 1_024
    minimum_safety_margin_tokens: int = 512
    safety_margin_ratio: float = 0.03
    l2_minimum_tokens: int = 2_048
    maximum_output_reserve_ratio: float = 0.35


class ContextBudgetPlanner:
    """Derive a model-aware input budget without coupling to a provider SDK."""

    def __init__(self, policy: ContextPlanningPolicy | None = None) -> None:
        self.policy = policy or ContextPlanningPolicy()

    def plan(
        self,
        profile: ModelContextProfile | None = None,
        *,
        l1_required_tokens: int = 0,
        tool_schema_tokens: int = 0,
        configured_input_limit: int | None = None,
    ) -> ContextBudgetPlan:
        profile = profile or ModelContextProfile()
        policy = self.policy
        context_window = int(
            profile.context_window_tokens or policy.fallback_context_window_tokens
        )
        max_output_reserve = max(1, int(context_window * policy.maximum_output_reserve_ratio))
        output_reserve = min(
            max_output_reserve,
            int(profile.max_output_tokens or policy.fallback_output_reserve_tokens),
        )
        tool_loop_reserve = min(
            int(policy.tool_loop_reserve_tokens),
            max(0, int(context_window * 0.10)),
            max(0, context_window - output_reserve - 1),
        )
        safety_margin = min(
            max(
                int(policy.minimum_safety_margin_tokens),
                int(context_window * policy.safety_margin_ratio),
            ),
            max(1, int(context_window * 0.10)),
        )
        available = max(
            1,
            context_window
            - output_reserve
            - tool_loop_reserve
            - safety_margin
            - max(0, int(tool_schema_tokens)),
        )
        if configured_input_limit is not None and configured_input_limit > 0:
            available = min(available, int(configured_input_limit))
        l1_required = max(0, int(l1_required_tokens))
        l2_minimum = min(
            max(0, int(policy.l2_minimum_tokens)),
            max(0, available - l1_required),
        )
        return ContextBudgetPlan(
            context_window_tokens=context_window,
            output_reserve_tokens=output_reserve,
            tool_loop_reserve_tokens=tool_loop_reserve,
            safety_margin_tokens=safety_margin,
            available_input_tokens=available,
            l1_required_tokens=l1_required,
            l2_minimum_tokens=l2_minimum,
            l3_available_tokens=max(0, available - l1_required - l2_minimum),
            model_profile=profile,
        )
