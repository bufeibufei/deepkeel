from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, Protocol

from deepkeel.context_contracts import (
    ContextAuthority,
    ContextRepresentation,
    ContextRetention,
    ContextScope,
    ContextTier,
    ContextVisibility,
)


@dataclass(frozen=True, slots=True)
class ContextWindowPolicy:
    max_input_tokens: int = 24_000
    reserved_output_tokens: int = 4_000
    history_limit: int = 0
    max_message_tokens: int = 0
    minimum_recent_history_tokens: int = 2_048
    working_memory_ratio: float = 0.45
    tool_loop_reserve_tokens: int = 0
    minimum_safety_margin_tokens: int = 0
    safety_margin_ratio: float = 0.0
    minimum_section_tokens: int = 64
    priority_sections: tuple[str, ...] = (
        "current_time",
        "subject",
        "facts",
        "memories",
    )
    required_sections: tuple[str, ...] = ("current_time", "subject")
    protected_sections: tuple[str, ...] = (
        "current_goal",
        "skill_activation",
        "pending_action",
        "business_object",
        "confirmed_facts",
        "pending_tools",
        "artifact_refs",
        "policy_constraints",
        "budget_constraints",
    )
    policy_id: str = "tiered-context-window-v3"


ContextLayer = Literal[
    "runtime_constitution",
    "turn_context",
    "working_memory",
    "retrieved_context",
]


@dataclass(frozen=True, slots=True)
class ContextSegment:
    """A product-neutral prompt section with explicit retention metadata."""

    key: str
    value: Any
    priority: int = 0
    required: bool = False
    source: str = ""
    summary: Any = None
    summary_version: str = ""
    cache_key: str = ""
    source_fingerprint: str = ""
    max_tokens: int = 0
    layer: ContextLayer = "turn_context"
    retention: ContextRetention = "normal"
    tier: ContextTier | None = None
    scope: ContextScope = "run"
    visibility: ContextVisibility = "model"
    representation: ContextRepresentation = "raw"
    authority: ContextAuthority = "canonical"
    subject_id: str = ""
    source_ref: str = ""


@dataclass(frozen=True, slots=True)
class ContextWindowResult:
    context_bundle: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ContextWindowManager(Protocol):
    def prepare(
        self,
        question: str,
        short_context: dict[str, Any],
        context_bundle: dict[str, Any],
    ) -> ContextWindowResult: ...


@dataclass(frozen=True, slots=True)
class ContextSummaryRecord:
    cache_key: str
    source_fingerprint: str
    summary: Any
    summary_version: str = ""


class ContextSummaryCache(Protocol):
    def get(self, cache_key: str, source_fingerprint: str) -> ContextSummaryRecord | None: ...

    def put(self, record: ContextSummaryRecord) -> None: ...

    def invalidate(self, cache_key: str) -> None: ...


class InMemoryContextSummaryCache:
    """Thread-safe reference cache that rejects stale source fingerprints."""

    def __init__(self) -> None:
        self._records: dict[str, ContextSummaryRecord] = {}
        self._lock = Lock()

    def get(self, cache_key: str, source_fingerprint: str) -> ContextSummaryRecord | None:
        with self._lock:
            record = self._records.get(str(cache_key or ""))
            if record is None or record.source_fingerprint != source_fingerprint:
                return None
            return copy.deepcopy(record)

    def put(self, record: ContextSummaryRecord) -> None:
        if not record.cache_key or not record.source_fingerprint:
            return
        with self._lock:
            self._records[record.cache_key] = copy.deepcopy(record)

    def invalidate(self, cache_key: str) -> None:
        with self._lock:
            self._records.pop(str(cache_key or ""), None)
