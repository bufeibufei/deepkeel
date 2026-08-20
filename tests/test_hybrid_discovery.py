from __future__ import annotations

import pytest

from deepkeel.discovery_sdk import (
    HybridDiscoveryPolicy,
    HybridSkillRanker,
    HybridToolRanker,
    PreservingSkillReranker,
    PreservingToolReranker,
)
from deepkeel.skill_disclosure import SkillDescriptor
from deepkeel.tool_disclosure import ToolDescriptor


def test_preserving_rerankers_keep_upstream_order() -> None:
    skills = (
        SkillDescriptor("second", "Second", "Second skill", "prompt", ("model",), ("second.run",)),
        SkillDescriptor("first", "First", "First skill", "prompt", ("model",), ("first.run",)),
    )
    tools = (
        ToolDescriptor("second.run", "Second tool", "discoverable", ()),
        ToolDescriptor("first.run", "First tool", "discoverable", ()),
    )

    assert PreservingSkillReranker().rerank(
        query="ignored", candidates=skills, limit=1
    ) == skills[:1]
    assert PreservingToolReranker().rerank(
        query="ignored", candidates=tools, limit=1
    ) == tools[:1]


class _Similarity:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores

    def score(self, *, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        del query, documents
        return self.scores


def test_hybrid_skill_ranker_uses_semantics_but_abstains_below_threshold() -> None:
    candidates = (
        SkillDescriptor("calendar", "Calendar", "Select a date", "workflow", ("model",), ("date.run",)),
        SkillDescriptor("naming", "Naming", "Create a name", "workflow", ("model",), ("name.run",)),
    )
    ranker = HybridSkillRanker(
        _Similarity((0.1, 0.95)),
        HybridDiscoveryPolicy(semantic_weight=1, lexical_weight=0, minimum_score=0.5),
    )

    assert ranker.discover(query="help me", candidates=candidates, limit=3) == (candidates[1],)

    abstaining = HybridSkillRanker(
        _Similarity((0.1, 0.2)),
        HybridDiscoveryPolicy(semantic_weight=1, lexical_weight=0, minimum_score=0.5),
    )
    assert abstaining.discover(query="unrelated", candidates=candidates, limit=3) == ()


def test_hybrid_tool_ranker_preserves_candidates_and_validates_adapter_shape() -> None:
    candidates = (
        ToolDescriptor("weather.read", "Read weather", "discoverable", ("weather",)),
        ToolDescriptor("calendar.write", "Create event", "discoverable", ("calendar",)),
    )
    ranker = HybridToolRanker(
        _Similarity((0.2, 0.9)),
        HybridDiscoveryPolicy(semantic_weight=1, lexical_weight=0, minimum_score=0.1),
    )

    assert ranker.rerank(query="event", candidates=candidates, limit=1) == (candidates[1],)

    invalid = HybridToolRanker(_Similarity((0.5,)))
    with pytest.raises(ValueError, match="score count"):
        invalid.discover(query="anything", candidates=candidates, limit=2)


def test_hybrid_policy_rejects_invalid_thresholds_and_weights() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridDiscoveryPolicy(semantic_weight=0, lexical_weight=0)
    with pytest.raises(ValueError, match="minimum_score"):
        HybridDiscoveryPolicy(minimum_score=1.1)
