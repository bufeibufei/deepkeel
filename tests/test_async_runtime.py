from __future__ import annotations

import asyncio
import time
from threading import Event as ThreadEvent

import pytest

from deepkeel.adapter_sdk import HarnessRuntimeBuilder, RuntimePorts
from deepkeel.contracts import AgentMessage, RunContext, ToolCall, ToolResult
from deepkeel.event_journal import InMemoryRuntimeEventJournal
from deepkeel.graph import HarnessGraph
from deepkeel.leases import InMemoryRunLeaseStore
from deepkeel.model import ModelInvocation, ModelProviderInfo, ModelTurn
from deepkeel.persistence import InMemoryDurableCheckpointStore
from deepkeel.scope import RuntimeScope
from deepkeel.state_store import InMemoryRuntimeStateStore
from deepkeel.runtime_streaming import BoundedRuntimeStreamBridge
from deepkeel.runtime_sdk import InMemoryRunControl
from deepkeel.runtime_sdk import RuntimeRequest
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor


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


class NativeAsyncStateStore:
    terminal_settlement_owner = "runtime"

    def __init__(self) -> None:
        self.store = InMemoryRuntimeStateStore()
        self.loops: list[asyncio.AbstractEventLoop] = []

    async def commit_scoped(self, mutation, *, scope, session=None):
        self.loops.append(asyncio.get_running_loop())
        return self.store.commit_scoped(mutation, scope=scope, session=session)

    async def load_snapshot_scoped(self, run_id, *, scope, session=None):
        self.loops.append(asyncio.get_running_loop())
        return self.store.load_snapshot_scoped(run_id, scope=scope, session=session)

    async def list_snapshots_scoped(
        self,
        *,
        scope,
        session=None,
        statuses=(),
        limit=100,
    ):
        self.loops.append(asyncio.get_running_loop())
        return self.store.list_snapshots_scoped(
            scope=scope,
            session=session,
            statuses=statuses,
            limit=limit,
        )


class NativeAsyncCheckpointStore:
    def __init__(self) -> None:
        self.store = InMemoryDurableCheckpointStore()
        self.operations: list[tuple[str, asyncio.AbstractEventLoop]] = []

    async def load(self, run_id, **kwargs):
        self.operations.append(("load", asyncio.get_running_loop()))
        return self.store.load(run_id, **kwargs)

    async def save(self, run_id, state, **kwargs):
        self.operations.append(("save", asyncio.get_running_loop()))
        self.store.save(run_id, state, **kwargs)

    async def delete(self, run_id, **kwargs):
        self.operations.append(("delete", asyncio.get_running_loop()))
        self.store.delete(run_id, **kwargs)


class NativeAsyncEventJournal:
    def __init__(self) -> None:
        self.journal = InMemoryRuntimeEventJournal()
        self.loops: list[asyncio.AbstractEventLoop] = []

    async def append(self, event):
        self.loops.append(asyncio.get_running_loop())
        return self.journal.append(event)

    async def latest_sequence(self, run_id):
        self.loops.append(asyncio.get_running_loop())
        return self.journal.latest_sequence(run_id)

    async def read_after(self, run_id, *, after_sequence=0, limit=100):
        self.loops.append(asyncio.get_running_loop())
        return self.journal.read_after(
            run_id,
            after_sequence=after_sequence,
            limit=limit,
        )


class NativeAsyncLeaseStore:
    def __init__(self) -> None:
        self.store = InMemoryRunLeaseStore()
        self.loops: list[asyncio.AbstractEventLoop] = []

    async def claim(self, run_id, *, owner_id, ttl_seconds):
        self.loops.append(asyncio.get_running_loop())
        return self.store.claim(run_id, owner_id=owner_id, ttl_seconds=ttl_seconds)

    async def renew(self, lease, *, ttl_seconds):
        self.loops.append(asyncio.get_running_loop())
        return self.store.renew(lease, ttl_seconds=ttl_seconds)

    async def release(self, lease):
        self.loops.append(asyncio.get_running_loop())
        self.store.release(lease)

    async def inspect(self, run_id):
        self.loops.append(asyncio.get_running_loop())
        return self.store.inspect(run_id)


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


def test_arun_uses_native_async_persistence_and_lease_ports_on_host_loop() -> None:
    async def scenario():
        host_loop = asyncio.get_running_loop()
        state_store = NativeAsyncStateStore()
        checkpoint_store = NativeAsyncCheckpointStore()
        event_journal = NativeAsyncEventJournal()
        lease_store = NativeAsyncLeaseStore()
        runtime = (
            HarnessRuntimeBuilder()
            .with_ports(
                RuntimePorts(
                    async_runtime_state_store=state_store,
                    async_checkpoint_store=checkpoint_store,
                    async_event_journal=event_journal,
                    async_run_lease_store=lease_store,
                    run_lease_owner_id="async-worker",
                )
            )
            .build()
        )
        scope = RuntimeScope(user_id="async-user")
        result = await runtime.arun(
            RuntimeRequest(
                question="hello",
                run_id="native-async-ports",
                scope=scope,
            ),
            provider=StreamingProvider(),
        )
        replay = await runtime.areplay_events(result.run_id)
        snapshot = await state_store.load_snapshot_scoped(
            result.run_id,
            scope=scope,
        )
        lease = await lease_store.inspect(result.run_id)
        used_loops = [
            *state_store.loops,
            *(loop for _, loop in checkpoint_store.operations),
            *event_journal.loops,
            *lease_store.loops,
        ]
        return result, replay, snapshot, lease, used_loops, host_loop, checkpoint_store

    result, replay, snapshot, lease, used_loops, host_loop, checkpoint_store = asyncio.run(
        scenario()
    )

    assert result.status.value == "completed"
    assert replay
    assert snapshot.status == "completed"
    assert result.diagnostics["recovery"][
        "checkpoint_cleanup"
    ]["durable"] == "deleted"
    assert any(
        event.event_type == "runtime.checkpoint.cleanup.recorded" for event in replay
    )
    assert lease is None
    assert used_loops and all(loop is host_loop for loop in used_loops)
    assert [operation for operation, _ in checkpoint_store.operations] == ["delete"]


def test_arun_consults_async_checkpoint_fallback_for_resume_without_state_store() -> None:
    async def scenario():
        checkpoint_store = NativeAsyncCheckpointStore()
        runtime = HarnessRuntimeBuilder().with_ports(
            RuntimePorts(async_checkpoint_store=checkpoint_store)
        ).build()
        result = await runtime.arun(
            RuntimeRequest(
                question="hello",
                run_id="async-checkpoint-fallback",
                short_context={"resume": True},
            ),
            provider=StreamingProvider(),
        )
        return result, checkpoint_store

    result, checkpoint_store = asyncio.run(scenario())

    assert result.status.value == "failed"
    assert [operation for operation, _ in checkpoint_store.operations] == [
        "load",
        "save",
        "delete",
    ]


def test_runtime_rejects_sync_and_async_versions_of_the_same_port() -> None:
    journal = InMemoryRuntimeEventJournal()

    with pytest.raises(ValueError, match="either event_journal or async_event_journal"):
        (
            HarnessRuntimeBuilder()
            .with_ports(
                RuntimePorts(
                    event_journal=journal,
                    async_event_journal=NativeAsyncEventJournal(),
                )
            )
            .build()
        )


def test_async_journal_persists_durable_events_before_publishing_them() -> None:
    class OrderedJournal(NativeAsyncEventJournal):
        def __init__(self) -> None:
            super().__init__()
            self.order: list[tuple[str, str]] = []

        async def append(self, event):
            await asyncio.sleep(0)
            self.order.append(("append", event.event_id))
            return await super().append(event)

    async def scenario():
        journal = OrderedJournal()

        def publish(event):
            journal.order.append(("publish", str(event.get("event_id") or "")))

        runtime = HarnessRuntimeBuilder().with_ports(
            RuntimePorts(async_event_journal=journal)
        ).build()
        result = await runtime.arun(
            RuntimeRequest(question="hello", run_id="ordered-async-journal"),
            provider=StreamingProvider(),
            event_sink=publish,
        )
        persisted = await runtime.areplay_events(result.run_id)
        return journal.order, persisted

    order, persisted = asyncio.run(scenario())

    for event in persisted:
        if event.event_type == "runtime.checkpoint.cleanup.recorded":
            assert ("publish", event.event_id) not in order
            continue
        assert order.index(("append", event.event_id)) < order.index(
            ("publish", event.event_id)
        )


def test_async_graph_falls_back_to_sync_for_sync_only_checkpointer() -> None:
    class SyncOnlyCompiledGraph:
        def __init__(self) -> None:
            self.sync_calls = 0

        def invoke(self, value, *, config, durability):
            del config
            assert durability == "exit"
            self.sync_calls += 1
            return value

        async def ainvoke(self, value, *, config, durability):
            del value, config, durability
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


def test_same_loop_stream_backpressure_is_bounded_and_preserves_delta_text() -> None:
    async def scenario():
        bridge = BoundedRuntimeStreamBridge(
            loop=asyncio.get_running_loop(),
            maxsize=1,
            closed=ThreadEvent(),
        )
        for index in range(100):
            bridge.offer_event(
                {
                    "event_type": "answer.delta",
                    "ephemeral": True,
                    "payload": {"delta": str(index)},
                }
            )
        buffered = bridge.buffered_items
        first = await bridge.get()
        second = await bridge.get()
        await bridge.close()
        return buffered, first, second

    buffered, first, second = asyncio.run(scenario())
    events = [first[1], second[1]]

    assert buffered == 2
    assert "".join(str(event["payload"]["delta"]) for event in events) == "".join(
        str(index) for index in range(100)
    )
    assert second[1]["payload"]["merged_count"] == 99


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
