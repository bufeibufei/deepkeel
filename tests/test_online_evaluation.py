from __future__ import annotations

from deepkeel.composition import RuntimePorts
from deepkeel.golden_path import AgentHarness
from deepkeel.model import ModelInvocation, ModelProviderInfo, ModelTurn
from deepkeel.online_evaluation import (
    InMemoryOnlineEvalStore,
    OnlineEvalPipeline,
    OnlineEvalPolicy,
)


class _Provider:
    info = ModelProviderInfo(provider_id="test", model_id="test-v1", model_role="fast")

    def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        del request
        if on_text_delta is not None:
            on_text_delta("answer")
        return ModelTurn(
            content="answer",
            finish_reason="stop",
            model_id=self.info.model_id,
            model_role=self.info.model_role,
        )


class _BrokenOnlineEvalPort:
    def submit(self, sample) -> None:
        del sample
        raise RuntimeError("evaluation backend unavailable")


def test_runtime_submits_privacy_bounded_online_eval_sample() -> None:
    store = InMemoryOnlineEvalStore()
    pipeline = OnlineEvalPipeline(store=store)
    harness = AgentHarness.create(
        provider=_Provider(),
        ports=RuntimePorts(
            online_eval_port=pipeline,
            online_eval_policy=OnlineEvalPolicy(content_mode="digest"),
        ),
    )

    result = harness.run("private question", thread_id="thread-eval")

    records = store.snapshot()
    assert len(records) == 1
    sample = records[0].sample
    assert sample.run_id == result.run_id
    assert sample.answer == ""
    assert sample.answer_digest
    assert records[0].scores[0].metric == "runtime_success"
    assert result.diagnostics["online_eval"]["status"] == "submitted"


def test_online_eval_sampling_is_deterministic_and_can_be_disabled() -> None:
    disabled = OnlineEvalPolicy(sample_rate=0)
    partial = OnlineEvalPolicy(sample_rate=0.5)

    assert disabled.should_sample(run_id="run-1", status="completed") is False
    assert partial.should_sample(run_id="stable", status="completed") == partial.should_sample(
        run_id="stable", status="completed"
    )


def test_online_eval_failure_degrades_without_failing_the_run() -> None:
    harness = AgentHarness.create(
        provider=_Provider(),
        ports=RuntimePorts(online_eval_port=_BrokenOnlineEvalPort()),
    )

    result = harness.run("question", thread_id="thread-eval-degraded")

    assert result.status.value == "completed"
    assert result.diagnostics["online_eval"] == {
        "status": "degraded",
        "error_type": "RuntimeError",
    }
