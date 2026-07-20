from __future__ import annotations

import asyncio
import time

from harness_core.adapter_sdk import HarnessRuntimeBuilder, RuntimePorts
from harness_core.runtime_sdk import InMemoryRunControl
from harness_core.runtime_sdk import RuntimeRequest


class StreamingProvider:
    model = "async-stream-model"
    model_role = "fast"

    def stream_chat(self, _messages, **_kwargs):
        for delta, finish_reason in (("async ", None), ("answer", "stop")):
            yield {
                "choices": [
                    {
                        "delta": {"content": delta},
                        "finish_reason": finish_reason,
                    }
                ]
            }


def test_arun_uses_the_same_canonical_runtime_contract() -> None:
    async def scenario():
        runtime = HarnessRuntimeBuilder().build()
        return await runtime.arun(
            RuntimeRequest(question="hello", run_id="async-run"),
            provider=StreamingProvider(),
        )

    result = asyncio.run(scenario())
    assert result.run_id == "async-run"
    assert result.final_answer.markdown == "async answer"
    assert result.answer_delta_streamed is True


def test_astream_emits_deltas_and_one_typed_terminal_result() -> None:
    async def scenario():
        runtime = HarnessRuntimeBuilder().build()
        return [
            event
            async for event in runtime.astream(
                RuntimeRequest(question="hello", run_id="stream-run"),
                provider=StreamingProvider(),
            )
        ]

    events = asyncio.run(scenario())
    deltas = [event.payload.get("delta", "") for event in events if event.event_type == "answer.delta"]
    terminal = [event for event in events if event.event_type == "runtime.result"]

    assert "".join(deltas) == "async answer"
    assert len(terminal) == 1
    assert terminal[0].payload["result"]["status"] == "completed"


def test_astream_applies_backpressure_and_cooperatively_cancels_on_disconnect() -> None:
    class CountingProvider:
        model = "backpressure-model"
        model_role = "fast"

        def __init__(self) -> None:
            self.produced = 0

        def stream_chat(self, _messages, **_kwargs):
            for index in range(40):
                self.produced += 1
                yield {
                    "choices": [
                        {
                            "delta": {"content": str(index)},
                            "finish_reason": None,
                        }
                    ]
                }

    class TrackingControl(InMemoryRunControl):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_calls = 0

        def cancel(self, run_id: str) -> None:
            self.cancel_calls += 1
            super().cancel(run_id)

    async def scenario():
        provider = CountingProvider()
        control = TrackingControl()
        runtime = (
            HarnessRuntimeBuilder()
            .with_ports(
                RuntimePorts(
                    run_control=control,
                    async_stream_buffer_size=1,
                    async_cancel_timeout_seconds=1,
                )
            )
            .build()
        )
        stream = runtime.astream(
            RuntimeRequest(question="hello", run_id="disconnect-run"),
            provider=provider,
        )
        await anext(stream)
        await asyncio.sleep(0.05)
        produced_before_close = provider.produced
        await stream.aclose()
        return control.cancel_calls, produced_before_close

    cancel_calls, produced_before_close = asyncio.run(scenario())
    assert cancel_calls == 1
    assert produced_before_close < 40
