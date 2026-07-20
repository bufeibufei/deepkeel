from __future__ import annotations

import asyncio

from harness_core.adapter_sdk import HarnessRuntimeBuilder
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
