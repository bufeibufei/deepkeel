from __future__ import annotations

import hashlib
import json
from typing import Any

from deepkeel.context_contracts import (
    ContextAuthority,
    ContextCheckpoint,
    ContextDecision,
    ContextItem,
    ContextRepresentation,
    ContextRetention,
    ContextScope,
    ContextTier,
    ContextVisibility,
)
from deepkeel.context_window_contracts import ContextLayer, ContextSegment
from deepkeel.token_estimation import ConservativeTokenEstimator
from deepkeel.type_narrowing import as_dict


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _default_context_layer(key: str) -> ContextLayer:
    if key in {"runtime_constitution", "policy_constraints", "budget_constraints"}:
        return "runtime_constitution"
    if key in {
        "recent_messages",
        "observations",
        "tool_summaries",
        "working_memory",
    }:
        return "working_memory"
    if key in {
        "memories",
        "retrieved_context",
        "references",
        "artifact_refs",
        "search_results",
    }:
        return "retrieved_context"
    return "turn_context"


def _context_layer(value: Any, default: ContextLayer) -> ContextLayer:
    normalized = str(value or "").strip()
    if normalized in {
        "runtime_constitution",
        "turn_context",
        "working_memory",
        "retrieved_context",
    }:
        return normalized  # type: ignore[return-value]
    return default


def _context_retention(value: Any, default: ContextRetention) -> ContextRetention:
    normalized = str(value or "").strip()
    if normalized in {"pinned", "protected", "normal", "ephemeral"}:
        return normalized  # type: ignore[return-value]
    return default


def _default_context_tier(key: str) -> ContextTier:
    if key in {
        "runtime_constitution",
        "current_time",
        "subject",
        "profile",
        "chart_facts",
        "consultation_policy",
        "artifact_reuse",
        "skill_activation",
        "pending_action",
        "policy_constraints",
        "budget_constraints",
    }:
        return "L1"
    if key in {
        "conversation_summary",
        "working_context_checkpoint",
        "recent_messages",
        "observations",
        "resume_observations",
        "tool_summaries",
        "working_memory",
        "latest_bazi_reading",
        "active_matters",
    }:
        return "L2"
    return "L3"


def _default_context_scope(key: str) -> ContextScope:
    if key in {"memories", "available_profiles"}:
        return "user"
    if key in {"runtime_constitution", "policy_constraints"}:
        return "tenant"
    if key in {"conversation_summary", "recent_messages", "working_context_checkpoint"}:
        return "thread"
    return "run"


def _default_context_representation(key: str) -> ContextRepresentation:
    if key in {"conversation_summary", "working_context_checkpoint", "tool_summaries"}:
        return "digest"
    if key in {"artifact_refs", "references"}:
        return "pointer"
    return "raw"


def _default_context_authority(key: str) -> ContextAuthority:
    if key in {
        "conversation_summary",
        "working_context_checkpoint",
        "context_selection",
        "context_budget",
        "tool_summaries",
    }:
        return "derived"
    return "canonical"


def _context_tier(value: Any, default: ContextTier) -> ContextTier:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"L1", "L2", "L3"} else default  # type: ignore[return-value]


def _context_scope(value: Any, default: ContextScope) -> ContextScope:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"step", "run", "thread", "user", "tenant"} else default  # type: ignore[return-value]


def _context_visibility(value: Any, default: ContextVisibility) -> ContextVisibility:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"runtime", "model", "both"} else default  # type: ignore[return-value]


def _context_representation(
    value: Any,
    default: ContextRepresentation,
) -> ContextRepresentation:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"raw", "digest", "pointer"} else default  # type: ignore[return-value]


def _context_authority(value: Any, default: ContextAuthority) -> ContextAuthority:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"canonical", "derived"} else default  # type: ignore[return-value]


def _segment_tier(segment: ContextSegment) -> ContextTier:
    if segment.tier is not None:
        return segment.tier
    mapping: dict[ContextLayer, ContextTier] = {
        "runtime_constitution": "L1",
        "turn_context": "L1" if segment.required else "L2",
        "working_memory": "L2",
        "retrieved_context": "L3",
    }
    return mapping[segment.layer]


def _tier_order(tier: ContextTier) -> int:
    return {"L1": 0, "L2": 1, "L3": 2}[tier]


def _context_item(segment: ContextSegment) -> ContextItem:
    return ContextItem(
        key=segment.key,
        value=segment.value,
        tier=_segment_tier(segment),
        scope=segment.scope,
        visibility=segment.visibility,
        retention=segment.retention,
        representation=segment.representation,
        authority=segment.authority,
        subject_id=segment.subject_id,
        source_ref=segment.source_ref,
        source=segment.source,
        priority=segment.priority,
        required=segment.required,
        max_tokens=segment.max_tokens,
        summary=segment.summary,
        summary_version=segment.summary_version,
        cache_key=segment.cache_key,
        source_fingerprint=segment.source_fingerprint,
    )


def _active_subject_id(bundle: dict[str, Any], runtime_context: dict[str, Any]) -> str:
    subject_context = as_dict(bundle.get("subject_context"))
    if subject_context.get("subject_id"):
        return str(subject_context["subject_id"])
    snapshot = as_dict(bundle.get("context_snapshot"))
    snapshot_subject = as_dict(snapshot.get("subject"))
    if snapshot_subject.get("subject_id"):
        return str(snapshot_subject["subject_id"])
    subject = as_dict(runtime_context.get("subject"))
    if subject.get("subject_id"):
        return str(subject["subject_id"])
    profile = as_dict(runtime_context.get("profile"))
    return str(profile.get("subject_id") or "")


def _context_decisions(
    segments: list[ContextSegment],
    bounded_context: dict[str, Any],
    diagnostics: dict[str, Any],
) -> list[ContextDecision]:
    dropped = set(diagnostics.get("dropped_sections") or [])
    summarized = set(diagnostics.get("summarized_sections") or [])
    truncated = set(diagnostics.get("truncated_sections") or [])
    decisions: list[ContextDecision] = []
    estimator = ConservativeTokenEstimator()
    for segment in segments:
        if segment.key in dropped or segment.key not in bounded_context:
            action = "dropped"
            reason = "lower-priority context exceeded the model input budget"
            tokens = 0
        elif segment.key in summarized:
            action = "summarized"
            reason = "host-provided digest replaced the raw value"
            tokens = estimator.estimate(bounded_context[segment.key])
        elif segment.key in truncated:
            action = "truncated"
            reason = "value was deterministically bounded to the allocated tier budget"
            tokens = estimator.estimate(bounded_context[segment.key])
        else:
            action = "retained"
            reason = "item fit its tier budget"
            tokens = estimator.estimate(bounded_context[segment.key])
        decisions.append(
            ContextDecision(
                key=segment.key,
                tier=_segment_tier(segment),
                action=action,  # type: ignore[arg-type]
                reason=reason,
                tokens=tokens,
                source_ref=segment.source_ref,
            )
        )
    return decisions


def context_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _checkpoint_from_runtime_context(runtime_context: dict[str, Any]) -> ContextCheckpoint | None:
    for key in ("conversation_summary", "working_context_checkpoint"):
        checkpoint = ContextCheckpoint.from_mapping(runtime_context.get(key))
        if checkpoint is not None:
            return checkpoint
    return None
