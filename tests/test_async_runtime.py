from __future__ import annotations

import asyncio
import time

from harness_core.adapter_sdk import HarnessRuntimeBuilder, RuntimePorts
from harness_core.contracts import AgentMessage, RunContext, ToolCall, ToolResult
from harness_core.graph import HarnessGraph
from harness_core.model import ModelInvocation, ModelProviderInfo, ModelTurn
from harness_core.runtime_sdk import InMemoryRunControl
from harness_core.runtime_sdk import RuntimeRequest
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tools import ToolExecutionContext, ToolExecutor


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


def test_async_graph_falls_back_to_sync_for_sync_only_checkpointer() -> None:
    class SyncOnlyCompiledGraph:
        def __init__(self) -> None:
            self.sync_calls = 0

        def invoke(self, value, *, config):
            del config
            self.sync_calls += 1
            return value

        async def ainvoke(self, value, *, config):
            del value, config
            raise AssertionError("sync-only checkpointer must not use ainvoke")

    async def scenario():
        compiled = SyncOnlyCompiledGraph()
        graph = HarnessGraph(
            compiled_graph=compiled,
            supports_async_checkpointer=False,
        )
        context = RunContext(
            run_id="sync-checkpointer-run",
            thread_id="sync-checkpointer-thread",
            turn_id="sync-checkpointer-turn",
            user_id="user-1",
            messages=[AgentMessage(id="message-1", role="user", content="hello")],
        )
        state = await graph.ainvoke(
            context,
            tool_context=ToolExecutionContext(
                run_id=context.run_id,
                user_id=context.user_id,
            ),
        )
        return compiled, state

    compiled, state = asyncio.run(scenario())
    assert compiled.sync_calls == 1
    assert state["run_id"] == "sync-checkpointer-run"


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


def test_async_tool_handler_runs_in_host_loop_and_propagates_cancellation() -> None:
    async def scenario() -> bool:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="async.wait",
                parameters_schema={"type": "object", "properties": {}},
                read_only=True,
                parallel_safe=True,
            )
        )
        executor = ToolExecutor(registry)
        started = asyncio.Event()

        async def wait_handler(call, _context):
            started.set()
            await asyncio.sleep(60)
            return ToolResult(call=call, status="succeeded")

        executor.register("async.wait", wait_handler)
        task = asyncio.create_task(
            executor.aexecute(
                ToolCall(id="async-call", name="async.wait", arguments={}),
                ToolExecutionContext(run_id="async-tool-run", user_id="user-1"),
            )
        )
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(scenario()) is True


def test_async_tool_executor_preserves_parallel_result_order() -> None:
    async def scenario() -> tuple[list[str], float]:
        registry = ToolRegistry()
        executor = ToolExecutor(registry, max_parallel_tools=2)
        for name in ("async.first", "async.second"):
            registry.register(
                ToolSpec(
                    name=name,
                    parameters_schema={"type": "object", "properties": {}},
                    read_only=True,
                    parallel_safe=True,
                )
            )

            async def handler(call, _context):
                await asyncio.sleep(0.05)
                return ToolResult(call=call, status="succeeded", summary=call.name)

            executor.register(name, handler)
        calls = [
            ToolCall(id="first", name="async.first", arguments={}),
            ToolCall(id="second", name="async.second", arguments={}),
        ]
        started_at = time.perf_counter()
        results = await executor.aexecute_many(
            calls,
            ToolExecutionContext(run_id="parallel-run", user_id="user-1"),
        )
        return [result.name for result in results], time.perf_counter() - started_at

    names, elapsed = asyncio.run(scenario())
    assert names == ["async.first", "async.second"]
    assert elapsed < 0.09


def test_native_async_model_provider_runs_in_host_loop() -> None:
    class AsyncProvider:
        info = ModelProviderInfo(
            provider_id="example.async",
            model_id="async-v1",
            model_role="fast",
        )

        def __init__(self) -> None:
            self.loop: asyncio.AbstractEventLoop | None = None

        async def ainvoke(self, _request: ModelInvocation, *, on_text_delta=None):
            self.loop = asyncio.get_running_loop()
            if on_text_delta is not None:
                on_text_delta("native ")
                await asyncio.sleep(0)
                on_text_delta("async")
            return ModelTurn(
                content="native async",
                finish_reason="stop",
                model_id=self.info.model_id,
            )

    async def scenario():
        provider = AsyncProvider()
        host_loop = asyncio.get_running_loop()
        result = await HarnessRuntimeBuilder().build().arun(
            RuntimeRequest(
                question="hello",
                run_id="native-async-run",
                model_policy={"mode": "single", "primary_role": "fast"},
            ),
            provider=provider,
        )
        return result, provider.loop is host_loop

    result, used_host_loop = asyncio.run(scenario())
    assert result.final_answer.markdown == "native async"
    assert used_host_loop is True


def test_native_async_model_provider_receives_task_cancellation() -> None:
    class WaitingProvider:
        info = ModelProviderInfo(
            provider_id="example.waiting",
            model_id="waiting-v1",
            model_role="fast",
        )

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.canceled = False

        async def ainvoke(self, _request: ModelInvocation, *, on_text_delta=None):
            del on_text_delta
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.canceled = True
                raise

    async def scenario() -> bool:
        provider = WaitingProvider()
        runtime = HarnessRuntimeBuilder().build()
        task = asyncio.create_task(
            runtime.arun(
                RuntimeRequest(
                    question="hello",
                    run_id="cancel-model-run",
                    model_policy={"mode": "single", "primary_role": "fast"},
                ),
                provider=provider,
            )
        )
        await provider.started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return provider.canceled

    assert asyncio.run(scenario()) is True
