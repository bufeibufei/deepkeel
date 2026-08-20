from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence, TypeVar

from deepkeel.skill_disclosure import SkillDescriptor
from deepkeel.tool_disclosure import ToolDescriptor


class SimilarityPort(Protocol):
    """Provider-neutral semantic similarity boundary used during catalog recall."""

    def score(self, *, query: str, documents: tuple[str, ...]) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class HybridDiscoveryPolicy:
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    minimum_score: float = 0.2

    def __post_init__(self) -> None:
        if self.semantic_weight < 0 or self.lexical_weight < 0:
            raise ValueError("discovery weights must be non-negative")
        if self.semantic_weight + self.lexical_weight <= 0:
            raise ValueError("at least one discovery weight must be positive")
        if not 0 <= self.minimum_score <= 1:
            raise ValueError("minimum_score must be between zero and one")


class HybridSkillRanker:
    """Two-stage Skill adapter combining semantic recall with lexical evidence."""

    def __init__(
        self,
        similarity: SimilarityPort,
        policy: HybridDiscoveryPolicy | None = None,
    ) -> None:
        self.similarity = similarity
        self.policy = policy or HybridDiscoveryPolicy()

    def discover(
        self,
        *,
        query: str,
        candidates: tuple[SkillDescriptor, ...],
        limit: int,
    ) -> tuple[SkillDescriptor, ...]:
        return _rank(
            query=query,
            candidates=candidates,
            documents=tuple(_skill_document(item) for item in candidates),
            identities=tuple(item.skill_id for item in candidates),
            similarity=self.similarity,
            policy=self.policy,
            limit=limit,
        )

    rerank = discover


class HybridToolRanker:
    """Two-stage Tool adapter that never widens the permission-filtered catalog."""

    def __init__(
        self,
        similarity: SimilarityPort,
        policy: HybridDiscoveryPolicy | None = None,
    ) -> None:
        self.similarity = similarity
        self.policy = policy or HybridDiscoveryPolicy()

    def discover(
        self,
        *,
        query: str,
        candidates: tuple[ToolDescriptor, ...],
        limit: int,
    ) -> tuple[ToolDescriptor, ...]:
        return _rank(
            query=query,
            candidates=candidates,
            documents=tuple(_tool_document(item) for item in candidates),
            identities=tuple(item.name for item in candidates),
            similarity=self.similarity,
            policy=self.policy,
            limit=limit,
        )

    rerank = discover


class PreservingSkillReranker:
    """Keep the order produced by an upstream Skill discovery adapter."""

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[SkillDescriptor, ...],
        limit: int,
    ) -> tuple[SkillDescriptor, ...]:
        del query
        return candidates[: max(1, int(limit))]


class PreservingToolReranker:
    """Keep the order produced by an upstream Tool discovery adapter."""

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[ToolDescriptor, ...],
        limit: int,
    ) -> tuple[ToolDescriptor, ...]:
        del query
        return candidates[: max(1, int(limit))]


DescriptorT = TypeVar("DescriptorT")


def _rank(
    *,
    query: str,
    candidates: tuple[DescriptorT, ...],
    documents: tuple[str, ...],
    identities: tuple[str, ...],
    similarity: SimilarityPort,
    policy: HybridDiscoveryPolicy,
    limit: int,
) -> tuple[DescriptorT, ...]:
    if not candidates or not query.strip():
        return ()
    semantic_scores = similarity.score(query=query, documents=documents)
    if len(semantic_scores) != len(candidates):
        raise ValueError("similarity adapter returned a score count that does not match candidates")
    lexical_scores = tuple(_lexical_score(query, document) for document in documents)
    weight_total = policy.semantic_weight + policy.lexical_weight
    ranked: list[tuple[float, str, DescriptorT]] = []
    for candidate, identity, semantic, lexical in zip(
        candidates,
        identities,
        semantic_scores,
        lexical_scores,
        strict=True,
    ):
        semantic_score = _bounded_score(semantic)
        combined = (
            policy.semantic_weight * semantic_score
            + policy.lexical_weight * lexical
        ) / weight_total
        if combined >= policy.minimum_score:
            ranked.append((combined, identity, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    bounded = max(1, int(limit))
    return tuple(item[2] for item in ranked[:bounded])


def _bounded_score(value: float) -> float:
    score = float(value)
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(score, 1.0))


def _lexical_score(query: str, document: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    document_tokens = _tokens(document)
    overlap = len(query_tokens & document_tokens) / len(query_tokens)
    exact = 1.0 if query.strip().lower() in document.lower() else 0.0
    return min(1.0, overlap * 0.75 + exact * 0.25)


def _tokens(value: str) -> set[str]:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in value)
    words = {word for word in normalized.split() if word}
    chinese = "".join(character for character in normalized if "\u4e00" <= character <= "\u9fff")
    words.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    return words


def _skill_document(descriptor: SkillDescriptor) -> str:
    return " ".join(
        (
            descriptor.skill_id,
            descriptor.label,
            descriptor.description,
            descriptor.kind,
            descriptor.package_id,
            *descriptor.tags,
        )
    )


def _tool_document(descriptor: ToolDescriptor) -> str:
    return " ".join((descriptor.name, descriptor.description, *descriptor.tags))


__all__ = [
    "HybridDiscoveryPolicy",
    "HybridSkillRanker",
    "HybridToolRanker",
    "PreservingSkillReranker",
    "PreservingToolReranker",
    "SimilarityPort",
]
