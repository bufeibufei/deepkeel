from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, Protocol

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
    history_limit: int = 8
    max_message_tokens: int = 2_000
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
    policy_id: str = "deterministic-context-window-v2"


ContextLayer = Literal[
    "runtime_constitution",
    "turn_context",
    "working_memory",
    "retrieved_context",
]
ContextRetention = Literal["protected", "normal", "ephemeral"]


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

    def prepare(
        self,
        question: str,
        short_context: dict[str, Any],
        context_bundle: dict[str, Any],
    ) -> ContextWindowResult:
        policy = self.policy
        bundle = dict(context_bundle)
        runtime_context = copy.deepcopy(as_dict(bundle.get("runtime_context")))
        if short_context.get("current_time") not in (None, "", [], {}):
            runtime_context.setdefault("current_time", copy.deepcopy(short_context["current_time"]))

        raw_history = bundle.get("recent_messages")
        if not isinstance(raw_history, list):
            raw_history = runtime_context.get("recent_messages")
        raw_history = raw_history if isinstance(raw_history, list) else []
        # History is represented as messages, never duplicated inside the system context JSON.
        runtime_context.pop("recent_messages", None)
        explicit_segments = bundle.pop("context_segments", None)

        history, history_diagnostics = self._bounded_history(raw_history)
        question_tokens = self.estimator.estimate(question)
        history_tokens = self.estimator.estimate(history) if history else 0
        context_budget = max(
            0,
            int(policy.max_input_tokens)
            - int(policy.reserved_output_tokens)
            - question_tokens
            - history_tokens,
        )
        segments = self._context_segments(runtime_context, explicit_segments)
        bounded_context, context_diagnostics = self._bounded_context(
            segments,
            context_budget,
        )

        bundle["runtime_context"] = bounded_context
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
        input_budget_tokens = max(
            0,
            int(policy.max_input_tokens) - int(policy.reserved_output_tokens),
        )
        diagnostics = {
            "schema_version": "harness-context-window-v2",
            "policy_id": policy.policy_id,
            "estimator_id": self.estimator.estimator_id,
            "max_input_tokens": int(policy.max_input_tokens),
            "reserved_output_tokens": int(policy.reserved_output_tokens),
            "input_budget_tokens": input_budget_tokens,
            "question_tokens": question_tokens,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "context_budget_tokens": context_budget,
            "over_budget": final_tokens > input_budget_tokens,
            "layers": self._layer_diagnostics(
                segments,
                bounded_context,
                history_tokens=history_tokens,
            ),
            "injection_sources": sorted(
                {
                    segment.source
                    for segment in segments
                    if segment.source and segment.key in bounded_context
                }
            ),
            "summary_versions": {
                segment.key: segment.summary_version
                for segment in segments
                if segment.summary_version and segment.key in bounded_context
            },
            **history_diagnostics,
            **context_diagnostics,
        }
        return ContextWindowResult(context_bundle=bundle, diagnostics=diagnostics)

    def _bounded_history(self, raw_history: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        valid = [
            item
            for item in raw_history
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and str(item.get("content") or "").strip()
        ]
        selected = valid[-max(1, int(self.policy.history_limit)) :]
        bounded: list[dict[str, Any]] = []
        truncated_messages = 0
        for item in selected:
            copied = dict(item)
            content = str(copied.get("content") or "").strip()
            bounded_content = self._truncate_text(content, self.policy.max_message_tokens)
            if bounded_content != content:
                truncated_messages += 1
            copied["content"] = bounded_content
            bounded.append(copied)
        return bounded, {
            "history_original_count": len(valid),
            "history_retained_count": len(bounded),
            "history_dropped_count": max(0, len(valid) - len(bounded)),
            "history_truncated_count": truncated_messages,
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
                    if segment.retention == "protected"
                ],
            }

        ordered_segments = sorted(
            enumerate(segments),
            key=lambda item: (
                not (
                    item[1].required
                    or item[1].retention == "protected"
                ),
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
                if candidate.required or candidate.retention == "protected"
            )
            segment_required = (
                segment.required or segment.retention == "protected"
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
                if segment.retention == "protected":
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
            if segment.retention == "protected":
                retained_protected.append(key)
            if compacted_tokens < value_tokens:
                truncated.append(key)
        while bounded and self.estimator.estimate(bounded) > budget:
            removable = [
                segment.key
                for _, segment in reversed(ordered_segments)
                if segment.key in bounded
                and not segment.required
                and segment.retention != "protected"
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
    if normalized in {"protected", "normal", "ephemeral"}:
        return normalized  # type: ignore[return-value]
    return default


def context_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
