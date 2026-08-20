"""Selective memory recall orchestration for product-neutral runtimes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from deepkeel.memory_sdk import AsyncMemoryPort, MemoryPort, MemoryQuery, MemorySearchPage
from deepkeel.type_narrowing import as_dict, as_list


MemoryRecallMode = Literal["skip", "prefetch", "agent_decide"]
MemoryRecallEnforcement = Literal["legacy", "shadow", "enforced"]


class MemoryRecallRequest(BaseModel):
    """Small, serializable policy input built before expensive L3 retrieval."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    question: str
    thread_id: str = ""
    subject_type: str = ""
    subject_id: str = ""
    profile_id: str = ""
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    conversation_summary: dict[str, Any] = Field(default_factory=dict)
    skill_activation: dict[str, Any] = Field(default_factory=dict)
    pending_action: dict[str, Any] = Field(default_factory=dict)
    context_inventory: list[str] = Field(default_factory=list)
    context_hints: dict[str, Any] = Field(default_factory=dict)
    resuming: bool = False

    @classmethod
    def from_context(
        cls,
        question: str,
        short_context: Mapping[str, Any] | None,
        context_bundle: Mapping[str, Any] | None,
    ) -> "MemoryRecallRequest":
        short = dict(short_context or {})
        bundle = dict(context_bundle or {})
        subject = as_dict(bundle.get("subject_context"))
        profile = as_dict(bundle.get("active_profile"))
        skill = as_dict(bundle.get("skill_activation"))
        recent = [
            dict(item)
            for item in as_list(bundle.get("recent_messages"))
            if isinstance(item, dict)
        ]
        compressed = as_dict(bundle.get("compressed_history"))
        return cls(
            run_id=str(
                bundle.get("agent_session_id")
                or bundle.get("agent_run_id")
                or bundle.get("run_id")
                or ""
            ),
            tenant_id=str(bundle.get("tenant_id") or short.get("tenant_id") or ""),
            user_id=str(bundle.get("user_id") or short.get("user_id") or ""),
            question=str(question or ""),
            thread_id=str(bundle.get("thread_id") or bundle.get("ask_thread_id") or ""),
            subject_type=str(subject.get("subject_kind") or subject.get("subject_type") or ""),
            subject_id=str(subject.get("subject_id") or profile.get("subject_id") or ""),
            profile_id=str(
                subject.get("profile_id")
                or profile.get("birth_profile_id")
                or bundle.get("birth_profile_id")
                or ""
            ),
            recent_messages=recent[-8:],
            conversation_summary=compressed,
            skill_activation=skill,
            pending_action=as_dict(short.get("pending_action") or bundle.get("pending_action")),
            context_inventory=sorted(str(key) for key in bundle if str(key)),
            context_hints={
                "subject_context": subject,
                "consultation_context": as_dict(bundle.get("consultation_context")),
                "memory_policy": as_dict(bundle.get("memory_policy")),
            },
            resuming=bool(short.get("resume") or short.get("recover_interrupted")),
        )


class MemoryRecallDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: MemoryRecallMode = "agent_decide"
    query: str = ""
    domains: list[str] = Field(default_factory=list)
    predicates: list[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=50)
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    allow_runtime_search: bool = True


class MemoryRecallPolicy(Protocol):
    def decide(self, request: MemoryRecallRequest) -> MemoryRecallDecision: ...


class AsyncMemoryRecallPolicy(Protocol):
    async def adecide(self, request: MemoryRecallRequest) -> MemoryRecallDecision: ...


class MemoryRecallCoordinator(Protocol):
    async def prepare(
        self,
        question: str,
        short_context: Mapping[str, Any] | None,
        context_bundle: Mapping[str, Any] | None,
    ) -> dict[str, Any]: ...


MemoryRecallProjector = Callable[[MemorySearchPage], Mapping[str, Any]]


class DefaultMemoryRecallCoordinator:
    """Run a host policy and prefetch only when it makes a high-confidence request."""

    def __init__(
        self,
        *,
        policy: MemoryRecallPolicy | AsyncMemoryRecallPolicy,
        memory_port: MemoryPort | AsyncMemoryPort,
        projector: MemoryRecallProjector | None = None,
        enforcement: MemoryRecallEnforcement = "enforced",
        runtime_search_tool_name: str = "memory.search",
        cache_ttl_seconds: float = 300.0,
        cache_max_entries: int = 1_024,
    ) -> None:
        self.policy = policy
        self.memory_port = memory_port
        self.projector = projector or _portable_memory_page
        self.enforcement = enforcement
        self.runtime_search_tool_name = str(runtime_search_tool_name or "memory.search")
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.cache_max_entries = max(1, int(cache_max_entries))
        self._cache: OrderedDict[str, tuple[float, MemorySearchPage]] = OrderedDict()
        self._cache_lock = Lock()

    async def prepare(
        self,
        question: str,
        short_context: Mapping[str, Any] | None,
        context_bundle: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        bundle = dict(context_bundle or {})
        request = MemoryRecallRequest.from_context(question, short_context, bundle)
        started = time.perf_counter()
        try:
            decision = await self._decide(request)
        except Exception as exc:
            decision = MemoryRecallDecision(
                mode="skip",
                query=request.question,
                reason=f"policy_error:{type(exc).__name__}",
                confidence=0,
                allow_runtime_search=False,
            )

        effective_mode = decision.mode
        if self.enforcement in {"legacy", "shadow"} and request.user_id and not request.resuming:
            effective_mode = "prefetch"
        trace: dict[str, Any] = {
            "schema_version": "harness-memory-recall-v1",
            "mode": decision.mode,
            "effective_mode": effective_mode,
            "enforcement": self.enforcement,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "allow_runtime_search": decision.allow_runtime_search,
            "query": decision.query or request.question,
            "status": "skipped",
            "selected_count": 0,
            "cache_hit": False,
        }
        if not decision.allow_runtime_search:
            disabled = {
                str(name)
                for name in as_list(bundle.get("disabled_tool_names"))
                if str(name).strip()
            }
            disabled.add(self.runtime_search_tool_name)
            bundle["disabled_tool_names"] = sorted(disabled)

        if effective_mode == "prefetch" and request.user_id and not request.resuming:
            query = MemoryQuery(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                subject_type=request.subject_type,
                subject_id=request.subject_id,
                profile_id=request.profile_id,
                thread_id=request.thread_id,
                text=decision.query or request.question,
                domains=list(decision.domains),
                predicates=list(decision.predicates),
                limit=decision.limit,
            )
            try:
                page, cache_hit = await self._search(request.run_id, query)
                projected = dict(self.projector(page))
                memories = [
                    dict(item)
                    for item in as_list(projected.get("memories"))
                    if isinstance(item, dict)
                ]
                bundle["long_term_memories"] = memories
                trace.update(
                    {
                        "status": "completed",
                        "selected_count": len(memories),
                        "candidate_count": len(page.hits),
                        "retrieval_mode": page.retrieval_mode,
                        "cache_hit": cache_hit,
                        "search_trace": dict(projected.get("trace") or page.trace),
                    }
                )
            except Exception as exc:
                # Memory enrichment is optional; a provider outage must not fail the turn.
                trace.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                    }
                )
        trace["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        bundle["memory_recall"] = trace
        bundle["memory_retrieval"] = dict(trace)
        return bundle

    async def _search(
        self,
        run_id: str,
        query: MemoryQuery,
    ) -> tuple[MemorySearchPage, bool]:
        key = _recall_cache_key(run_id, query)
        cached = self._cache_get(key)
        if cached is not None:
            return cached, True
        native_search = getattr(self.memory_port, "asearch", None)
        if callable(native_search):
            page = await native_search(query)
        else:
            search = getattr(self.memory_port, "search", None)
            if not callable(search):
                raise TypeError("memory port must implement search() or asearch()")
            page = await asyncio.to_thread(search, query)
        validated = MemorySearchPage.model_validate(page)
        self._cache_put(key, validated)
        return validated.model_copy(deep=True), False

    async def _decide(self, request: MemoryRecallRequest) -> MemoryRecallDecision:
        native_decide = getattr(self.policy, "adecide", None)
        if callable(native_decide):
            return MemoryRecallDecision.model_validate(await native_decide(request))
        decide = getattr(self.policy, "decide", None)
        if not callable(decide):
            raise TypeError("memory recall policy must implement decide() or adecide()")
        return MemoryRecallDecision.model_validate(await asyncio.to_thread(decide, request))

    def _cache_get(self, key: str) -> MemorySearchPage | None:
        if self.cache_ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            expires_at, page = cached
            if expires_at <= now:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return page.model_copy(deep=True)

    def _cache_put(self, key: str, page: MemorySearchPage) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (
                time.monotonic() + self.cache_ttl_seconds,
                page.model_copy(deep=True),
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)


def _recall_cache_key(run_id: str, query: MemoryQuery) -> str:
    payload = {
        "run_id": str(run_id or ""),
        **query.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _portable_memory_page(page: MemorySearchPage) -> Mapping[str, Any]:
    memories: list[dict[str, Any]] = []
    for hit in page.hits:
        claim = hit.claim
        memories.append(
            {
                "id": claim.claim_id,
                "subject_type": claim.subject_type,
                "subject_id": claim.subject_id,
                "domain": claim.domain,
                "predicate": claim.predicate,
                "content": claim.value,
                "scope": claim.scope,
                "birth_profile_id": claim.profile_id,
                "confidence": claim.confidence,
                "sensitivity": claim.sensitivity,
                "retrieval_score": hit.score,
                "retrieval_components": {
                    "structured_score": hit.structured_score,
                    "lexical_score": hit.lexical_score,
                    "semantic_score": hit.semantic_score,
                    "rerank_score": hit.rerank_score,
                },
            }
        )
    return {"memories": memories, "trace": dict(page.trace)}


MEMORY_RECALL_API = (
    "AsyncMemoryRecallPolicy",
    "DefaultMemoryRecallCoordinator",
    "MemoryRecallCoordinator",
    "MemoryRecallDecision",
    "MemoryRecallEnforcement",
    "MemoryRecallMode",
    "MemoryRecallPolicy",
    "MemoryRecallProjector",
    "MemoryRecallRequest",
)

__all__ = list(MEMORY_RECALL_API)
