"""Step-routing tests owned by the standalone runtime package."""

from deepkeel.model_routing import AdaptiveStepModelRouter, ModelStepContext


def _context(**overrides):
    values = {
        "run_id": "run-a",
        "user_id": "user-a",
        "thread_id": "thread-a",
        "turn_id": "turn-a",
        "step_index": 0,
        "message_count": 1,
        "observation_count": 0,
        "tool_result_count": 0,
        "available_roles": ("fast", "reasoning"),
        "model_policy": {"mode": "adaptive"},
        "skill_activation": {},
        "policy_phase": "",
    }
    values.update(overrides)
    return ModelStepContext(**values)


def test_single_model_policy_uses_primary_role():
    decision = AdaptiveStepModelRouter().route(
        _context(model_policy={"mode": "single", "primary_role": "reasoning"})
    )
    assert decision.role == "reasoning"


def test_adaptive_initial_step_uses_fast_model():
    assert AdaptiveStepModelRouter().route(_context()).role == "fast"


def test_workflow_initial_planning_uses_fast_model():
    decision = AdaptiveStepModelRouter().route(
        _context(skill_activation={"skill_id": "guided-workflow", "kind": "workflow"})
    )
    assert decision.role == "fast"
    assert decision.reason == "workflow initial planning uses fast model"


def test_workflow_observation_synthesis_uses_reasoning_model():
    decision = AdaptiveStepModelRouter().route(
        _context(
            step_index=1,
            observation_count=1,
            tool_result_count=1,
            skill_activation={"skill_id": "guided-workflow", "kind": "workflow"},
        )
    )
    assert decision.role == "reasoning"


def test_tool_discovery_only_continuation_stays_on_fast_model():
    decision = AdaptiveStepModelRouter().route(
        _context(
            step_index=1,
            observation_count=1,
            tool_result_count=1,
            observation_sources=("runtime.discover_tools",),
            tool_result_names=("runtime.discover_tools",),
        )
    )
    assert decision.role == "fast"
    assert decision.reason == "tool discovery continuation uses fast model"


def test_business_tool_observation_uses_reasoning_model():
    decision = AdaptiveStepModelRouter().route(
        _context(
            step_index=1,
            observation_count=1,
            tool_result_count=1,
            observation_sources=("profile.read_current",),
            tool_result_names=("profile.read_current",),
        )
    )
    assert decision.role == "reasoning"
    assert decision.reason == "tool observations require synthesis"


def test_incomplete_observation_identity_does_not_downgrade_to_fast_model():
    decision = AdaptiveStepModelRouter().route(
        _context(
            step_index=1,
            observation_count=1,
            tool_result_count=1,
            observation_sources=("runtime.discover_tools",),
        )
    )
    assert decision.role == "reasoning"


def test_contract_repair_always_uses_reasoning_model():
    assert AdaptiveStepModelRouter().route(_context(policy_phase="repair")).role == "reasoning"
