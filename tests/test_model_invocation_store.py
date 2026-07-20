from __future__ import annotations

import pytest

from harness_core.composition import HarnessRuntimeBuilder, RuntimePorts
from harness_core.model import (
    InMemoryModelInvocationStore,
    ModelInvocation,
    ModelInvocationConflict,
    ModelInvocationEnvelope,
    ModelProviderInfo,
    ModelTurn,
)
from harness_core.runtime_api import RuntimeRequest


class CountingProvider:
    info = ModelProviderInfo(
        provider_id="example.counting",
        model_id="counting-v1",
        model_role="fast",
    )

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        self.calls += 1
        if on_text_delta is not None:
            on_text_delta("durable ")
            on_text_delta("answer")
        return ModelTurn(
            content="durable answer",
            finish_reason="stop",
            model_id=self.info.model_id,
            model_role=self.info.model_role,
        )


def _envelope() -> ModelInvocationEnvelope:
    return ModelInvocationEnvelope(
        invocation_id="run-1:model:0:attempt:1",
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        request=ModelInvocation(messages=[{"role": "user", "content": "hello"}]),
    )


def test_model_invocation_store_claims_settles_and_replays_result():
    store = InMemoryModelInvocationStore()
    envelope = _envelope()

    claim = store.claim(envelope)
    assert claim.outcome == "acquired"
    assert claim.claim_token
    assert store.claim(envelope).outcome == "in_progress"

    result = ModelTurn(content="answer", finish_reason="stop")
    store.complete(envelope.invocation_id, claim_token=claim.claim_token, result=result)
    replay = store.claim(envelope)

    assert replay.outcome == "replay"
    assert replay.result == result


def test_model_invocation_store_rejects_changed_request_and_stale_settlement():
    store = InMemoryModelInvocationStore()
    envelope = _envelope()
    claim = store.claim(envelope)

    changed = envelope.model_copy(
        update={
            "request": ModelInvocation(
                messages=[{"role": "user", "content": "different"}]
            )
        }
    )
    with pytest.raises(ModelInvocationConflict, match="different request"):
        store.claim(changed)
    with pytest.raises(ModelInvocationConflict, match="claim token"):
        store.complete(
            envelope.invocation_id,
            claim_token="stale",
            result=ModelTurn(content="answer"),
        )


def test_runtime_replays_completed_model_invocation_without_second_provider_call():
    store = InMemoryModelInvocationStore()
    provider = CountingProvider()
    request = RuntimeRequest(
        question="hello",
        run_id="run-replay",
        thread_id="thread-replay",
        turn_id="turn-replay",
        model_policy={"mode": "single", "primary_role": "fast"},
    )

    first_runtime = HarnessRuntimeBuilder().with_ports(
        RuntimePorts(model_invocation_store=store)
    ).build()
    second_runtime = HarnessRuntimeBuilder().with_ports(
        RuntimePorts(model_invocation_store=store)
    ).build()

    first = first_runtime.run(request, provider=provider)
    second = second_runtime.run(request, provider=provider)

    assert first.final_answer.markdown == "durable answer"
    assert second.final_answer.markdown == "durable answer"
    assert provider.calls == 1
    route = next(item for item in second.trace if item["action"] == "model.route.selected")
    assert route["invocation"]["claim_outcome"] == "replay"
    assert route["invocation"]["replayed"] is True
