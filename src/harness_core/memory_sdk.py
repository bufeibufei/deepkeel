"""Portable memory contracts for Harness Agent hosts and capability packages."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


MemoryScope = Literal["global", "profile", "thread", "temporary"]
MemoryStatus = Literal["active", "inactive", "superseded", "archived"]
MemoryMutationAction = Literal["create", "reinforce", "update", "archive", "noop"]


class MemoryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = ""
    source_type: str
    source_id: str = ""
    source_message_id: str = ""
    source_role: str = "user"
    text: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryClaim(BaseModel):
    """A durable, attributable claim; storage and embedding remain host concerns."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = ""
    schema_version: int = 1
    tenant_id: str = ""
    user_id: str
    subject_type: str = "user"
    subject_id: str = ""
    domain: str = "general"
    predicate: str
    value: str
    scope: MemoryScope = "global"
    profile_id: str = ""
    status: MemoryStatus = "active"
    confidence: float = Field(default=0.7, ge=0, le=1)
    observation_count: int = Field(default=1, ge=1)
    sensitivity: str = "normal"
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: MemoryMutationAction
    claim: MemoryClaim | None = None
    target_claim_id: str = ""
    evidence: list[MemoryEvidence] = Field(default_factory=list)
    reason: str = ""
    idempotency_key: str = ""


class MemoryMutationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: MemoryMutationAction
    claim_id: str = ""
    applied: bool = False
    version: int = 0
    reason: str = ""


class MemoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = ""
    user_id: str
    subject_type: str = ""
    subject_id: str = ""
    profile_id: str = ""
    thread_id: str = ""
    text: str = ""
    domains: list[str] = Field(default_factory=list)
    predicates: list[str] = Field(default_factory=list)
    scopes: list[MemoryScope] = Field(default_factory=list)
    include_sensitive: bool = False
    limit: int = Field(default=8, ge=1, le=50)


class MemorySearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: MemoryClaim
    score: float = 0
    structured_score: float = 0
    lexical_score: float = 0
    semantic_score: float = 0
    rerank_score: float = 0
    reasons: list[str] = Field(default_factory=list)


class MemorySearchPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[MemorySearchHit] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    trace: dict[str, Any] = Field(default_factory=dict)


class MemoryPort(Protocol):
    """Host-provided source-of-truth memory persistence and retrieval port."""

    def apply(self, mutation: MemoryMutation) -> MemoryMutationReceipt: ...

    def search(self, query: MemoryQuery) -> MemorySearchPage: ...

    def get(self, claim_id: str) -> MemoryClaim | None: ...


MEMORY_SDK_API = (
    "MemoryClaim",
    "MemoryEvidence",
    "MemoryMutation",
    "MemoryMutationAction",
    "MemoryMutationReceipt",
    "MemoryPort",
    "MemoryQuery",
    "MemoryScope",
    "MemorySearchHit",
    "MemorySearchPage",
    "MemoryStatus",
)

__all__ = list(MEMORY_SDK_API)
