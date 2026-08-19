from __future__ import annotations

import asyncio

from deepkeel.composition import HarnessRuntimeBuilder, RuntimePorts
from deepkeel.graph_state import _allowed_tool_names
from deepkeel.memory_recall import (
    DefaultMemoryRecallCoordinator,
    MemoryRecallDecision,
    MemoryRecallRequest,
)
from deepkeel.memory_sdk import MemoryClaim, MemorySearchHit, MemorySearchPage
from deepkeel.runtime_sdk import RuntimeRequest
from deepkeel.tool_registry import ToolRegistry, ToolSpec


class StaticPolicy:
    def __init__(self, decision: MemoryRecallDecision) -> None:
        self.decision = decision
        self.requests: list[MemoryRecallRequest] = []

    def decide(self, request: MemoryRecallRequest) -> MemoryRecallDecision:
        self.requests.append(request)
        return self.decision


class BrokenPolicy:
    def decide(self, _request):
        raise RuntimeError("policy unavailable")


class RecordingMemoryPort:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("vector store unavailable")
        claim = MemoryClaim(
            claim_id="memory-1",
            user_id=query.user_id,
            subject_id=query.subject_id,
            domain="career",
            predicate="career.goal",
            value="Build reusable agent runtimes.",
        )
        return MemorySearchPage(
            hits=[MemorySearchHit(claim=claim, score=0.9, semantic_score=0.9)],
            trace={"candidate_count": 1},
        )

    def apply(self, _mutation):  # pragma: no cover - protocol completeness
        raise NotImplementedError

    def get(self, _claim_id):  # pragma: no cover - protocol completeness
        raise NotImplementedError


class NativeAsyncPolicy:
    def __init__(self) -> None:
        self.called = False

    async def adecide(self, request):
        self.called = True
        await asyncio.sleep(0)
        return MemoryRecallDecision(
            mode="prefetch",
            query=request.question,
            reason="native_async_policy",
            confidence=1,
        )


class NativeAsyncMemoryPort:
    def __init__(self) -> None:
        self.called = False

    async def asearch(self, query):
        self.called = True
        await asyncio.sleep(0)
        claim = MemoryClaim(
            claim_id="async-memory-1",
            user_id=query.user_id,
            predicate="preference.response_style",
            value="Prefer concise answers.",
        )
        return MemorySearchPage(hits=[MemorySearchHit(claim=claim, score=1)])

    async def aapply(self, _mutation):  # pragma: no cover - protocol completeness
        raise NotImplementedError

    async def aget(self, _claim_id):  # pragma: no cover - protocol completeness
        raise NotImplementedError


class ScriptedProvider:
    model = "scripted-model"
    model_role = "reasoning"

    def complete_chat(self, _messages, **_kwargs):
        return {
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
            "model": self.model,
        }


def _bundle() -> dict:
    return {
        "agent_session_id": "run-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "subject_context": {"subject_id": "subject-1", "subject_kind": "user"},
        "active_profile": {"birth_profile_id": "profile-1"},
        "recent_messages": [{"role": "user", "content": "Earlier context"}],
    }


def test_skip_does_not_touch_memory_port() -> None:
    policy = StaticPolicy(
        MemoryRecallDecision(mode="skip", reason="simple_greeting", confidence=0.99)
    )
    port = RecordingMemoryPort()
    coordinator = DefaultMemoryRecallCoordinator(policy=policy, memory_port=port)

    result = asyncio.run(coordinator.prepare("hello", {}, _bundle()))

    assert port.queries == []
    assert result["memory_recall"]["status"] == "skipped"
    assert result["memory_recall"]["mode"] == "skip"


def test_prefetch_injects_l3_payload_and_replays_same_run_query() -> None:
    policy = StaticPolicy(
        MemoryRecallDecision(
            mode="prefetch",
            query="long-term career goal",
            domains=["career"],
            reason="explicit_long_term_reference",
            confidence=0.95,
        )
    )
    port = RecordingMemoryPort()
    coordinator = DefaultMemoryRecallCoordinator(policy=policy, memory_port=port)

    first = asyncio.run(coordinator.prepare("consider my long-term goal", {}, _bundle()))
    second = asyncio.run(coordinator.prepare("consider my long-term goal", {}, _bundle()))

    assert len(port.queries) == 1
    assert first["long_term_memories"][0]["content"] == "Build reusable agent runtimes."
    assert first["memory_recall"]["cache_hit"] is False
    assert second["memory_recall"]["cache_hit"] is True


def test_native_async_policy_and_memory_port_do_not_require_sync_facades() -> None:
    policy = NativeAsyncPolicy()
    port = NativeAsyncMemoryPort()
    coordinator = DefaultMemoryRecallCoordinator(policy=policy, memory_port=port)

    result = asyncio.run(coordinator.prepare("remember my style", {}, _bundle()))

    assert policy.called is True
    assert port.called is True
    assert result["long_term_memories"][0]["content"] == "Prefer concise answers."


def test_recall_failure_is_non_fatal_and_keeps_runtime_search_available() -> None:
    policy = StaticPolicy(MemoryRecallDecision(mode="prefetch", reason="required"))
    coordinator = DefaultMemoryRecallCoordinator(
        policy=policy,
        memory_port=RecordingMemoryPort(fail=True),
    )

    result = asyncio.run(coordinator.prepare("remember this", {}, _bundle()))

    assert result["memory_recall"]["status"] == "failed"
    assert result["memory_recall"]["error_type"] == "RuntimeError"
    assert "disabled_tool_names" not in result


def test_policy_can_disable_runtime_memory_search() -> None:
    policy = StaticPolicy(
        MemoryRecallDecision(
            mode="skip",
            reason="user_opted_out",
            allow_runtime_search=False,
        )
    )
    coordinator = DefaultMemoryRecallCoordinator(policy=policy, memory_port=RecordingMemoryPort())

    result = asyncio.run(coordinator.prepare("do not use memory", {}, _bundle()))

    assert result["disabled_tool_names"] == ["memory.search"]


def test_policy_failure_denies_runtime_memory_search() -> None:
    coordinator = DefaultMemoryRecallCoordinator(
        policy=BrokenPolicy(),
        memory_port=RecordingMemoryPort(),
    )

    result = asyncio.run(coordinator.prepare("remember this", {}, _bundle()))

    assert result["memory_recall"]["mode"] == "skip"
    assert result["memory_recall"]["reason"] == "policy_error:RuntimeError"
    assert result["disabled_tool_names"] == ["memory.search"]


def test_runtime_supplies_typed_request_identity_and_skill_to_recall_policy() -> None:
    policy = StaticPolicy(
        MemoryRecallDecision(mode="prefetch", reason="required", confidence=1.0)
    )
    port = RecordingMemoryPort()
    coordinator = DefaultMemoryRecallCoordinator(policy=policy, memory_port=port)
    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(RuntimePorts(memory_recall_coordinator=coordinator))
        .build()
    )

    result = runtime.run(
        RuntimeRequest(
            question="Use my long-term plan",
            tenant_id="tenant-1",
            user_id="user-1",
            run_id="run-typed-request",
            thread_id="thread-1",
            skill_activation={"skill_id": "career-planning"},
        ),
        provider=ScriptedProvider(),
    )

    assert result.final_answer.markdown == "done"
    request = policy.requests[0]
    assert request.tenant_id == "tenant-1"
    assert request.user_id == "user-1"
    assert request.run_id == "run-typed-request"
    assert request.thread_id == "thread-1"
    assert request.skill_activation["skill_id"] == "career-planning"
    assert port.queries[0].tenant_id == "tenant-1"
    assert any(event.event_type == "memory.recall.completed" for event in result.events)


def test_disabled_runtime_search_is_removed_from_the_model_tool_view() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(name="memory.search"),
            ToolSpec(name="general.read"),
        ]
    )
    state = {
        "skill_activation": {},
        "metadata": {"disabled_tool_names": ["memory.search"]},
    }

    assert _allowed_tool_names(state, registry) == {"general.read"}
