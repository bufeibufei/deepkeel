from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ContextTier = Literal["L1", "L2", "L3"]
ContextScope = Literal["step", "run", "thread", "user", "tenant"]
ContextVisibility = Literal["runtime", "model", "both"]
ContextRetention = Literal["pinned", "protected", "normal", "ephemeral"]
ContextRepresentation = Literal["raw", "digest", "pointer"]
ContextAuthority = Literal["canonical", "derived"]


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One context candidate with assembly metadata independent of its payload."""

    key: str
    value: Any
    tier: ContextTier = "L2"
    scope: ContextScope = "run"
    visibility: ContextVisibility = "model"
    retention: ContextRetention = "normal"
    representation: ContextRepresentation = "raw"
    authority: ContextAuthority = "canonical"
    subject_id: str = ""
    source_ref: str = ""
    source: str = ""
    priority: int = 0
    required: bool = False
    max_tokens: int = 0
    summary: Any = None
    summary_version: str = ""
    cache_key: str = ""
    source_fingerprint: str = ""

    @property
    def model_visible(self) -> bool:
        return self.visibility in {"model", "both"}

    @property
    def protected(self) -> bool:
        return self.required or self.retention in {"pinned", "protected"}


@dataclass(frozen=True, slots=True)
class ModelContextProfile:
    """Provider-neutral limits used to plan one concrete model invocation."""

    model_id: str = ""
    model_role: str = ""
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    source: str = "unknown"

    @classmethod
    def from_mapping(cls, value: Any) -> "ModelContextProfile":
        raw = value if isinstance(value, dict) else {}
        return cls(
            model_id=str(raw.get("model_id") or ""),
            model_role=str(raw.get("model_role") or raw.get("role") or ""),
            context_window_tokens=_positive_int_or_none(raw.get("context_window_tokens")),
            max_output_tokens=_positive_int_or_none(raw.get("max_output_tokens")),
            source=str(raw.get("source") or "unknown"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_role": self.model_role,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ContextBudgetPlan:
    context_window_tokens: int
    output_reserve_tokens: int
    tool_loop_reserve_tokens: int
    safety_margin_tokens: int
    available_input_tokens: int
    l1_required_tokens: int = 0
    l2_minimum_tokens: int = 0
    l3_available_tokens: int = 0
    model_profile: ModelContextProfile = field(default_factory=ModelContextProfile)

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_window_tokens": self.context_window_tokens,
            "output_reserve_tokens": self.output_reserve_tokens,
            "tool_loop_reserve_tokens": self.tool_loop_reserve_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "available_input_tokens": self.available_input_tokens,
            "l1_required_tokens": self.l1_required_tokens,
            "l2_minimum_tokens": self.l2_minimum_tokens,
            "l3_available_tokens": self.l3_available_tokens,
            "model_profile": self.model_profile.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ContextCheckpoint:
    """Derived L2 working checkpoint. Raw events remain the source of truth."""

    checkpoint_id: str
    thread_id: str = ""
    subject_id: str = ""
    goal: str = ""
    constraints_and_preferences: tuple[str, ...] = ()
    done: tuple[str, ...] = ()
    in_progress: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    key_decisions: tuple[str, ...] = ()
    pending_actions: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    critical_facts: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    failed_attempts: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    covered_event_range: tuple[str, str] = ("", "")
    first_kept_event_id: str = ""
    source_fingerprint: str = ""
    previous_checkpoint_id: str = ""
    summary_version: str = "context-checkpoint-v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "subject_id": self.subject_id,
            "goal": self.goal,
            "constraints_and_preferences": list(self.constraints_and_preferences),
            "progress": {
                "done": list(self.done),
                "in_progress": list(self.in_progress),
                "blocked": list(self.blocked),
            },
            "key_decisions": list(self.key_decisions),
            "pending_actions": list(self.pending_actions),
            "open_questions": list(self.open_questions),
            "critical_facts": [dict(item) for item in self.critical_facts],
            "artifacts": [dict(item) for item in self.artifacts],
            "failed_attempts": list(self.failed_attempts),
            "next_steps": list(self.next_steps),
            "covered_event_range": list(self.covered_event_range),
            "first_kept_event_id": self.first_kept_event_id,
            "source_fingerprint": self.source_fingerprint,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "summary_version": self.summary_version,
        }


@dataclass(frozen=True, slots=True)
class ContextDecision:
    key: str
    tier: ContextTier
    action: Literal["retained", "summarized", "truncated", "dropped", "runtime_only"]
    reason: str
    tokens: int = 0
    source_ref: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tier": self.tier,
            "action": self.action,
            "reason": self.reason,
            "tokens": self.tokens,
            "source_ref": self.source_ref,
        }


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
