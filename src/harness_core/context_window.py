from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, Protocol

from harness_core.context_compaction import DeterministicWorkingContextCompactor
from harness_core.context_contracts import (
    ContextCheckpoint,
    ContextAuthority,
    ContextDecision,
    ContextItem,
    ContextRepresentation,
    ContextRetention,
    ContextScope,
    ContextTier,
    ContextVisibility,
    ModelContextProfile,
)
from harness_core.context_planning import (
    ContextBudgetPlanner,
    ContextPlanningPolicy,
)
from harness_core.context_validation import validate_context_items
from harness_core.type_narrowing import as_dict


class TokenEstimator(Protocol):
    estimator_id: str

    def estimate(self, value: Any) -> int: ...


class ConservativeTokenEstimator:
    """Dependency-free estimator that is conservative for CJK and JSON text."""

    estimator_id = "conservative-cjk-v1"

    def estimate(self, value: Any) -> int:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
        ascii_count = sum(1 for char in value if ord(char) < 128)
        non_ascii_count = len(value) - ascii_count
        return max(1, math.ceil(ascii_count / 4) + non_ascii_count)


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


class DeterministicContextWindowManager:
    """Build a bounded prompt context without model calls or hidden summaries."""

    def __init__(
        self,
        policy: ContextWindowPolicy | None = None,
        estimator: TokenEstimator | None = None,
        summary_cache: ContextSummaryCache | None = None,
    ) -> None:
        self.policy = policy or ContextWindowPolicy()
        self.estimator = estimator or ConservativeTokenEstimator()
        self.summary_cache = summary_cache
        self.compactor = DeterministicWorkingContextCompactor(self.estimator)

    def prepare(
        self,
        question: str,
        short_context: dict[str, Any],
        context_bundle: dict[str, Any],
    ) -> ContextWindowResult:
        policy = self.policy
        bundle = dict(context_bundle)
        profile = ModelContextProfile.from_mapping(
            bundle.pop("_model_context_profile", None)
            or bundle.get("model_context_profile")
        )
        configured_input_limit = _safe_int(
            bundle.pop("_configured_input_limit", None),
            0,
        )
        budget_plan = ContextBudgetPlanner(
            ContextPlanningPolicy(
                fallback_context_window_tokens=int(policy.max_input_tokens),
                fallback_output_reserve_tokens=int(policy.reserved_output_tokens),
                tool_loop_reserve_tokens=int(policy.tool_loop_reserve_tokens),
                minimum_safety_margin_tokens=int(policy.minimum_safety_margin_tokens),
                safety_margin_ratio=float(policy.safety_margin_ratio),
                l2_minimum_tokens=int(policy.minimum_recent_history_tokens),
            )
        ).plan(
            profile,
            configured_input_limit=(
                configured_input_limit if configured_input_limit > 0 else None
            ),
        )
        runtime_context = copy.deepcopy(as_dict(bundle.get("runtime_context")))
        if short_context.get("current_time") not in (None, "", [], {}):
            runtime_context.setdefault("current_time", copy.deepcopy(short_context["current_time"]))

        raw_history = bundle.get("recent_messages")
        if not isinstance(raw_history, list):
            raw_history = runtime_context.get("recent_messages")
        raw_history = raw_history if isinstance(raw_history, list) else []
        current_message_removed = False
        if raw_history:
            latest = raw_history[-1]
            if (
                isinstance(latest, dict)
                and str(latest.get("role") or "") == "user"
                and str(latest.get("content") or "").strip() == str(question or "").strip()
            ):
                raw_history = raw_history[:-1]
                current_message_removed = True
        # History is represented as messages, never duplicated inside the system context JSON.
        runtime_context.pop("recent_messages", None)
        explicit_segments = bundle.pop("context_segments", None)

        question_tokens = self.estimator.estimate(question)
        history_budget = min(
            max(0, budget_plan.available_input_tokens - question_tokens),
            max(
                int(policy.minimum_recent_history_tokens),
                int(budget_plan.available_input_tokens * float(policy.working_memory_ratio)),
            ),
        )
        history, history_checkpoint, history_diagnostics = self._bounded_history(
            raw_history,
            token_budget=history_budget,
            thread_id=str(bundle.get("thread_id") or bundle.get("ask_thread_id") or ""),
            subject_id=_active_subject_id(bundle, runtime_context),
            previous_checkpoint=_checkpoint_from_runtime_context(runtime_context),
        )
        if history_checkpoint is not None:
            runtime_context["conversation_summary"] = history_checkpoint.as_dict()
        history_tokens = self.estimator.estimate(history) if history else 0
        context_budget = max(
            0,
            int(budget_plan.available_input_tokens)
            - question_tokens
            - history_tokens,
        )
        segments = self._context_segments(runtime_context, explicit_segments)
        context_items = [_context_item(segment) for segment in segments]
        active_subject_id = _active_subject_id(bundle, runtime_context)
        validation = validate_context_items(
            context_items,
            active_subject_id=active_subject_id,
        )
        mismatched_segments = [
            segment
            for segment in segments
            if active_subject_id
            and segment.subject_id
            and segment.subject_id != active_subject_id
            and _segment_tier(segment) in {"L1", "L2"}
        ]
        mismatched_keys = {segment.key for segment in mismatched_segments}
        runtime_only_context = {
            segment.key: copy.deepcopy(segment.value)
            for segment in segments
            if segment.visibility == "runtime"
        }
        model_segments = [
            segment
            for segment in segments
            if segment.visibility in {"model", "both"}
            and segment.key not in mismatched_keys
        ]
        bounded_context, context_diagnostics = self._bounded_context(
            model_segments,
            context_budget,
        )

        bundle["runtime_context"] = bounded_context
        if runtime_only_context:
            bundle["runtime_only_context"] = runtime_only_context
        if mismatched_segments:
            bundle["quarantined_context"] = {
                segment.key: copy.deepcopy(segment.value)
                for segment in mismatched_segments
            }
        bundle["recent_messages"] = history
        original_tokens = (
            question_tokens
            + self.estimator.estimate(runtime_context)
            + (self.estimator.estimate(raw_history) if raw_history else 0)
        )
        final_tokens = (
            question_tokens
            + self.estimator.estimate(bounded_context)
            + (self.estimator.estimate(history) if history else 0)
        )
        input_budget_tokens = budget_plan.available_input_tokens
        tiers = self._tier_diagnostics(
            model_segments,
            bounded_context,
            history_tokens=history_tokens,
        )
        decisions = _context_decisions(model_segments, bounded_context, context_diagnostics)
        decisions.extend(
            ContextDecision(
                key=segment.key,
                tier=_segment_tier(segment),
                action="runtime_only",
                reason="runtime visibility excludes item from model input",
                tokens=self.estimator.estimate(segment.value),
                source_ref=segment.source_ref,
            )
            for segment in segments
            if segment.visibility == "runtime"
        )
        decisions.extend(
            ContextDecision(
                key=segment.key,
                tier=_segment_tier(segment),
                action="dropped",
                reason="subject mismatch quarantined before model input",
                tokens=self.estimator.estimate(segment.value),
                source_ref=segment.source_ref,
            )
            for segment in mismatched_segments
        )
        diagnostics = {
            "schema_version": "harness-context-window-v3",
            "policy_id": policy.policy_id,
            "estimator_id": self.estimator.estimator_id,
            "max_input_tokens": budget_plan.context_window_tokens,
            "reserved_output_tokens": budget_plan.output_reserve_tokens,
            "input_budget_tokens": input_budget_tokens,
            "question_tokens": question_tokens,
            "current_message_removed_from_history": current_message_removed,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "context_budget_tokens": context_budget,
            "over_budget": final_tokens > input_budget_tokens,
            "layers": self._layer_diagnostics(
                model_segments,
                bounded_context,
                history_tokens=history_tokens,
            ),
            "tiers": tiers,
            "budget_plan": budget_plan.as_dict(),
            "validation": validation.as_dict(),
            "context_manifest": {
                "schema_version": "harness-context-manifest-v1",
                "tiers": tiers,
                "decisions": [decision.as_dict() for decision in decisions],
                "validation": validation.as_dict(),
            },
            "injection_sources": sorted(
                {
                    segment.source
                    for segment in model_segments
                    if segment.source and segment.key in bounded_context
                }
            ),
            "summary_versions": {
                segment.key: segment.summary_version
                for segment in model_segments
                if segment.summary_version and segment.key in bounded_context
            },
            **history_diagnostics,
            **context_diagnostics,
        }
        bundle["context_tier_payloads"] = {
            tier: {
                segment.key: copy.deepcopy(bounded_context[segment.key])
                for segment in model_segments
                if _segment_tier(segment) == tier and segment.key in bounded_context
            }
            for tier in ("L1", "L2", "L3")
        }
        bundle["context_manifest"] = copy.deepcopy(diagnostics["context_manifest"])
        return ContextWindowResult(context_bundle=bundle, diagnostics=diagnostics)

    def _bounded_history(
        self,
        raw_history: list[Any],
        *,
        token_budget: int,
        thread_id: str,
        subject_id: str,
        previous_checkpoint: ContextCheckpoint | None = None,
    ) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
        valid = [
            item
            for item in raw_history
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant", "tool"}
            and (str(item.get("content") or "").strip() or item.get("tool_calls"))
        ]
        history_limit = int(self.policy.history_limit)
        selected = valid[-history_limit:] if history_limit > 0 else valid
        bounded: list[dict[str, Any]] = []
        truncated_messages = 0
        for item in selected:
            copied = dict(item)
            content = str(copied.get("content") or "").strip()
            bounded_content = (
                self._truncate_text(content, self.policy.max_message_tokens)
                if int(self.policy.max_message_tokens) > 0
                else content
            )
            if bounded_content != content:
                truncated_messages += 1
            copied["content"] = bounded_content
            bounded.append(copied)
        compacted = self.compactor.compact(
            bounded,
            token_budget=max(0, int(token_budget)),
            thread_id=thread_id,
            subject_id=subject_id,
            previous_checkpoint=previous_checkpoint,
        )
        return compacted.retained_messages, compacted.checkpoint, {
            "history_original_count": len(valid),
            "history_retained_count": len(compacted.retained_messages),
            "history_dropped_count": max(0, len(valid) - len(compacted.retained_messages)),
            "history_truncated_count": truncated_messages,
            "history_token_budget": int(token_budget),
            "history_compaction": compacted.diagnostics,
        }

    def _bounded_context(
        self,
        segments: list[ContextSegment],
        budget: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime_context = {segment.key: segment.value for segment in segments}
        original_tokens = self.estimator.estimate(runtime_context) if runtime_context else 0
        if not runtime_context or original_tokens <= budget:
            if self.summary_cache is not None:
                for segment in segments:
                    if segment.cache_key and segment.summary not in (None, "", [], {}):
                        self.summary_cache.put(
                            ContextSummaryRecord(
                                cache_key=segment.cache_key,
                                source_fingerprint=(
                                    segment.source_fingerprint
                                    or context_fingerprint(segment.value)
                                ),
                                summary=segment.summary,
                                summary_version=segment.summary_version,
                            )
                        )
            return runtime_context, {
                "context_original_tokens": original_tokens,
                "context_final_tokens": original_tokens,
                "dropped_sections": [],
                "truncated_sections": [],
                "summarized_sections": [],
                "summary_cache_hits": [],
                "summary_cache_misses": [],
                "required_sections_retained": [
                    segment.key for segment in segments if segment.required
                ],
                "protected_sections_retained": [
                    segment.key
                    for segment in segments
                    if segment.retention in {"pinned", "protected"}
                ],
            }

        ordered_segments = sorted(
            enumerate(segments),
            key=lambda item: (
                not (
                    item[1].required
                    or item[1].retention in {"pinned", "protected"}
                ),
                _tier_order(_segment_tier(item[1])),
                -int(item[1].priority),
                item[0],
            ),
        )
        bounded: dict[str, Any] = {}
        dropped: list[str] = []
        truncated: list[str] = []
        summarized: list[str] = []
        summary_cache_hits: list[str] = []
        summary_cache_misses: list[str] = []
        retained_required: list[str] = []
        retained_protected: list[str] = []
        remaining = max(0, int(budget))
        for index, (_, segment) in enumerate(ordered_segments):
            key = segment.key
            value = segment.value
            value_tokens = self.estimator.estimate(value)
            sections_left = len(ordered_segments) - index
            fair_share = remaining if sections_left <= 1 else max(
                int(self.policy.minimum_section_tokens),
                remaining // sections_left,
            )
            required_left = sum(
                1
                for _, candidate in ordered_segments[index:]
                if candidate.required or candidate.retention in {"pinned", "protected"}
            )
            segment_required = (
                segment.required or segment.retention in {"pinned", "protected"}
            )
            allocation_cap = (
                max(1, remaining // max(1, required_left))
                if segment_required
                else fair_share
            )
            allowance = min(
                remaining,
                value_tokens,
                int(segment.max_tokens) if int(segment.max_tokens) > 0 else value_tokens,
                allocation_cap,
            )
            if allowance <= 0 or (
                not segment_required
                and allowance < int(self.policy.minimum_section_tokens)
            ):
                if segment.retention in {"pinned", "protected"}:
                    fallback = (
                        copy.deepcopy(segment.summary)
                        if segment.summary not in (None, "", [], {})
                        else copy.deepcopy(value)
                    )
                    bounded[key] = fallback
                    retained_required.append(key)
                    retained_protected.append(key)
                    truncated.append(key)
                    continue
                dropped.append(key)
                continue
            compacted = None
            summary = segment.summary
            fingerprint = segment.source_fingerprint or context_fingerprint(value)
            if segment.cache_key and self.summary_cache is not None:
                cached = self.summary_cache.get(segment.cache_key, fingerprint)
                if cached is not None:
                    summary = cached.summary
                    summary_cache_hits.append(key)
                elif summary not in (None, "", [], {}):
                    summary_cache_misses.append(key)
                    self.summary_cache.put(
                        ContextSummaryRecord(
                            cache_key=segment.cache_key,
                            source_fingerprint=fingerprint,
                            summary=summary,
                            summary_version=segment.summary_version,
                        )
                    )
                else:
                    summary_cache_misses.append(key)
            if summary not in (None, "", [], {}):
                summary_tokens = self.estimator.estimate(summary)
                if value_tokens > allowance and summary_tokens <= allowance:
                    compacted = copy.deepcopy(summary)
                    summarized.append(key)
            if compacted is None:
                compacted = self._compact_value(value, allowance)
            if compacted in (None, "", [], {}):
                dropped.append(key)
                continue
            compacted_tokens = self.estimator.estimate(compacted)
            if compacted_tokens > remaining:
                dropped.append(key)
                continue
            bounded[key] = compacted
            remaining -= compacted_tokens
            if segment.required:
                retained_required.append(key)
            if segment.retention in {"pinned", "protected"}:
                retained_protected.append(key)
            if compacted_tokens < value_tokens:
                truncated.append(key)
        while bounded and self.estimator.estimate(bounded) > budget:
            removable = [
                segment.key
                for _, segment in reversed(ordered_segments)
                if segment.key in bounded
                and not segment.required
                and segment.retention not in {"pinned", "protected"}
            ]
            if not removable:
                break
            removed_key = removable[0]
            bounded.pop(removed_key)
            if removed_key not in dropped:
                dropped.append(removed_key)
            if removed_key in truncated:
                truncated.remove(removed_key)
            if removed_key in summarized:
                summarized.remove(removed_key)
            if removed_key in retained_required:
                retained_required.remove(removed_key)
        return bounded, {
            "context_original_tokens": original_tokens,
            "context_final_tokens": self.estimator.estimate(bounded) if bounded else 0,
            "dropped_sections": dropped,
            "truncated_sections": truncated,
            "summarized_sections": summarized,
            "required_sections_retained": retained_required,
            "protected_sections_retained": retained_protected,
            "protected_over_budget": (
                self.estimator.estimate(bounded) > budget
                and bool(retained_protected)
            ),
            "summary_cache_hits": summary_cache_hits,
            "summary_cache_misses": summary_cache_misses,
        }

    def _context_segments(
        self,
        runtime_context: dict[str, Any],
        explicit_segments: Any,
    ) -> list[ContextSegment]:
        priority = {
            key: len(self.policy.priority_sections) - index
            for index, key in enumerate(self.policy.priority_sections)
        }
        segments = {
            key: ContextSegment(
                key=key,
                value=value,
                priority=priority.get(key, 0),
                required=(
                    key in self.policy.required_sections
                    or key in self.policy.protected_sections
                ),
                source="runtime_context",
                layer=_default_context_layer(key),
                retention=(
                    "protected"
                    if key in self.policy.protected_sections
                    else "normal"
                ),
                tier=_default_context_tier(key),
                scope=_default_context_scope(key),
                visibility="model",
                representation=_default_context_representation(key),
                authority=_default_context_authority(key),
                source_ref=key,
            )
            for key, value in runtime_context.items()
        }
        if isinstance(explicit_segments, list):
            for raw in explicit_segments:
                if isinstance(raw, ContextSegment):
                    segment = raw
                elif isinstance(raw, dict):
                    key = str(raw.get("key") or raw.get("section") or "").strip()
                    if not key:
                        continue
                    inherited = segments.get(key)
                    value = raw.get(
                        "value",
                        inherited.value if inherited is not None else None,
                    )
                    segment = ContextSegment(
                        key=key,
                        value=value,
                        priority=_safe_int(raw.get("priority"), priority.get(key, 0)),
                        required=bool(
                            raw.get(
                                "required",
                                inherited.required if inherited is not None else False,
                            )
                        ),
                        source=str(raw.get("source") or "host"),
                        summary=raw.get("summary"),
                        summary_version=str(raw.get("summary_version") or ""),
                        cache_key=str(raw.get("cache_key") or ""),
                        source_fingerprint=str(raw.get("source_fingerprint") or ""),
                        max_tokens=max(0, _safe_int(raw.get("max_tokens"), 0)),
                        layer=_context_layer(
                            raw.get("layer"),
                            inherited.layer
                            if inherited is not None
                            else _default_context_layer(key),
                        ),
                        retention=_context_retention(
                            raw.get("retention"),
                            inherited.retention
                            if inherited is not None
                            else (
                                "protected"
                                if key in self.policy.protected_sections
                                else "normal"
                            ),
                        ),
                        tier=_context_tier(
                            raw.get("tier"),
                            inherited.tier
                            if inherited is not None and inherited.tier is not None
                            else _default_context_tier(key),
                        ),
                        scope=_context_scope(
                            raw.get("scope"),
                            inherited.scope if inherited is not None else _default_context_scope(key),
                        ),
                        visibility=_context_visibility(
                            raw.get("visibility"),
                            inherited.visibility if inherited is not None else "model",
                        ),
                        representation=_context_representation(
                            raw.get("representation"),
                            inherited.representation
                            if inherited is not None
                            else _default_context_representation(key),
                        ),
                        authority=_context_authority(
                            raw.get("authority"),
                            inherited.authority
                            if inherited is not None
                            else _default_context_authority(key),
                        ),
                        subject_id=str(
                            raw.get("subject_id")
                            or (inherited.subject_id if inherited is not None else "")
                        ),
                        source_ref=str(
                            raw.get("source_ref")
                            or (inherited.source_ref if inherited is not None else key)
                        ),
                    )
                else:
                    continue
                if segment.key and segment.value not in (None, "", [], {}):
                    segments[segment.key] = segment
        return list(segments.values())

    def _layer_diagnostics(
        self,
        segments: list[ContextSegment],
        bounded_context: dict[str, Any],
        *,
        history_tokens: int,
    ) -> dict[str, dict[str, Any]]:
        layers: dict[str, dict[str, Any]] = {
            name: {"tokens": 0, "sections": [], "sources": []}
            for name in (
                "runtime_constitution",
                "turn_context",
                "working_memory",
                "retrieved_context",
            )
        }
        for segment in segments:
            if segment.key not in bounded_context:
                continue
            item = layers[segment.layer]
            item["tokens"] = int(item["tokens"]) + self.estimator.estimate(
                bounded_context[segment.key]
            )
            item["sections"].append(segment.key)
            if segment.source and segment.source not in item["sources"]:
                item["sources"].append(segment.source)
        layers["working_memory"]["tokens"] = (
            int(layers["working_memory"]["tokens"]) + history_tokens
        )
        if history_tokens:
            layers["working_memory"]["sections"].append("recent_messages")
            layers["working_memory"]["sources"].append("conversation_history")
        return layers

    def _tier_diagnostics(
        self,
        segments: list[ContextSegment],
        bounded_context: dict[str, Any],
        *,
        history_tokens: int,
    ) -> dict[str, dict[str, Any]]:
        tiers: dict[str, dict[str, Any]] = {
            tier: {"tokens": 0, "sections": [], "sources": []}
            for tier in ("L1", "L2", "L3")
        }
        for segment in segments:
            if segment.key not in bounded_context:
                continue
            tier = _segment_tier(segment)
            item = tiers[tier]
            item["tokens"] = int(item["tokens"]) + self.estimator.estimate(
                bounded_context[segment.key]
            )
            item["sections"].append(segment.key)
            if segment.source and segment.source not in item["sources"]:
                item["sources"].append(segment.source)
        tiers["L2"]["tokens"] = int(tiers["L2"]["tokens"]) + history_tokens
        if history_tokens:
            tiers["L2"]["sections"].append("recent_messages")
            tiers["L2"]["sources"].append("conversation_history")
        return tiers

    def _compact_value(self, value: Any, budget: int) -> Any:
        if budget <= 0:
            return None
        if self.estimator.estimate(value) <= budget:
            return copy.deepcopy(value)
        if isinstance(value, str):
            return self._truncate_text(value, budget)
        if isinstance(value, list):
            list_result: list[Any] = []
            remaining = budget
            for item in value:
                compacted = self._compact_value(item, remaining)
                if compacted in (None, "", [], {}):
                    break
                cost = self.estimator.estimate(compacted)
                if cost > remaining:
                    break
                list_result.append(compacted)
                remaining -= cost
            return list_result
        if isinstance(value, dict):
            dict_result: dict[str, Any] = {}
            remaining = budget
            for key, item in value.items():
                key_cost = self.estimator.estimate(str(key))
                if key_cost >= remaining:
                    break
                compacted = self._compact_value(item, remaining - key_cost)
                if compacted in (None, "", [], {}):
                    continue
                cost = key_cost + self.estimator.estimate(compacted)
                if cost > remaining:
                    continue
                dict_result[str(key)] = compacted
                remaining -= cost
            return dict_result
        return None

    def _truncate_text(self, value: str, max_tokens: int) -> str:
        if self.estimator.estimate(value) <= max_tokens:
            return value
        if max_tokens <= 1:
            return ""
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = value[:middle].rstrip() + "..."
            if self.estimator.estimate(candidate) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        return value[:low].rstrip() + "..." if low else ""


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
