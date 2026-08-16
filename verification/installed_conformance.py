from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import deepkeel
from deepkeel.mcp_sdk import (
    McpCallResult,
    McpClientPool,
    McpNormalizedResult,
    McpRemoteTool,
    McpServerSpec,
    McpToolBinding,
    McpToolProvider,
)
from deepkeel.adapter_sdk import RuntimePorts
from deepkeel.extension_sdk import (
    ArtifactTypeSpec,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPackSpec,
    DefaultReferenceProjector,
    SkillPackageManifest,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)
from deepkeel.orchestration_sdk import (
    DelegationRequest,
    DelegationTask,
    SubAgentExecutor,
    SubAgentRegistry,
    SubAgentSpec,
)
from deepkeel.runtime_sdk import (
    Artifact,
    HarnessRuntimeBuilder,
    InMemoryRunControl,
    Observation,
    PendingAction,
    RuntimeRequest,
    RuntimeScope,
    ToolCall,
    ToolResult,
)


def _tool_turn(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def _parallel_turn(*calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _tool_turn(call_id, name, arguments)["tool_calls"][0]
            for call_id, name, arguments in calls
        ],
    }


class CompletionProvider:
    model = "conformance-model"
    model_role = "reasoning"

    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)

    def complete_chat(self, _messages, **_kwargs):
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        message = (
            {"role": "assistant", "content": turn}
            if isinstance(turn, str)
            else turn
        )
        return {
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            "model": self.model,
        }


class StreamingProvider:
    model = "conformance-stream-model"
    model_role = "fast"

    def stream_chat(self, _messages, **_kwargs):
        for delta, finish_reason in (("streamed ", None), ("answer", "stop")):
            yield {
                "choices": [
                    {
                        "delta": {"content": delta},
                        "finish_reason": finish_reason,
                    }
                ]
            }


def _result(
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    data: dict[str, Any],
    summary: str,
) -> ToolResult:
    return ToolResult(
        call=call,
        status="succeeded",
        summary=summary,
        data=data,
        observation=Observation(
            id=f"{call.id}:observation",
            run_id=context.run_id,
            tool_call_id=call.id,
            source=call.name,
            status="succeeded",
            outcome="completed",
            summary=summary,
            data=data,
        ),
    )


class InventoryPack:
    spec = CapabilityPackSpec(
        package_id="conformance.inventory",
        package_version="1.0.0",
        declared_tools=("inventory.lookup", "inventory.reserve"),
        declared_skills=("inventory-review",),
        declared_artifact_types=("inventory_record",),
    )

    def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
        lookup = ToolSpec(
            name="inventory.lookup",
            parameters_schema={
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
                "additionalProperties": False,
            },
            required_args=["item"],
            read_only=True,
            parallel_safe=True,
        )
        reserve = ToolSpec(
            name="inventory.reserve",
            parameters_schema={
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
                "additionalProperties": False,
            },
            required_args=["item"],
            read_only=False,
        )
        context.register_tool(lookup, self.lookup)
        context.register_tool(reserve, self.reserve)
        context.register_skill("inventory-review", {"label": "Inventory review"})
        context.register_artifact_type(
            ArtifactTypeSpec(
                artifact_type="inventory_record",
                schema={
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
            )
        )
        return CapabilityContribution(package_id=self.spec.package_id)

    @staticmethod
    def lookup(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        item = str(call.arguments["item"])
        return _result(
            call,
            context,
            data={
                "item": item,
                "quantity": 7,
                "evidence": [{"id": f"record:{item}", "title": "Inventory ledger"}],
                "results": [{"url": "https://example.test/inventory", "title": "Inventory API"}],
            },
            summary=f"{item}: 7 available",
        )

    @staticmethod
    def reserve(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            call=call,
            status="requires_user_action",
            summary="Reservation confirmation required.",
            pending_action=PendingAction(
                id=f"{call.id}:confirm",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="confirmation",
                title="Confirm reservation",
                prompt="Continue?",
                payload={"item": call.arguments["item"]},
            ),
        )


def _request(run_id: str, question: str, **updates: Any) -> RuntimeRequest:
    return RuntimeRequest(
        question=question,
        user_id="conformance-user",
        short_context=dict(updates.pop("short_context", {})),
        context_bundle={"agent_session_id": run_id, "thread_id": f"thread:{run_id}"},
        skill_activation=dict(updates.pop("skill_activation", {})),
        model_policy=dict(updates.pop("model_policy", {})),
    )


def verify_runtime_and_streaming() -> None:
    runtime = HarnessRuntimeBuilder().add_capability_pack(InventoryPack()).build()
    normal = runtime.run(
        _request("normal", "Hello"),
        provider=CompletionProvider(["normal answer"]),
    )
    assert normal.status.value == "completed"
    assert normal.final_answer.markdown == "normal answer"

    deltas: list[str] = []
    streamed = runtime.run(
        _request("stream", "Stream"),
        provider=StreamingProvider(),
        event_sink=lambda event: deltas.append(
            str((event.get("payload") or {}).get("delta") or "")
        )
        if event.get("event_type") == "answer.delta"
        else None,
    )
    assert streamed.final_answer.markdown == "streamed answer"
    assert "".join(deltas) == "streamed answer"
    assert streamed.answer_delta_streamed is True

    async def verify_async_entrypoints() -> None:
        async_result = await runtime.arun(
            _request("async", "Async"),
            provider=CompletionProvider(["async answer"]),
        )
        assert async_result.final_answer.markdown == "async answer"
        events = [
            event
            async for event in runtime.astream(
                _request("async-stream", "Async stream"),
                provider=StreamingProvider(),
            )
        ]
        assert events[-1].event_type == "runtime.result"
        assert events[-1].payload["result"]["status"] == "completed"

    asyncio.run(verify_async_entrypoints())


def verify_tools_parallel_failure_and_references() -> None:
    runtime = HarnessRuntimeBuilder().add_capability_pack(InventoryPack()).build()
    result = runtime.run(
        _request("tools", "Check two items"),
        provider=CompletionProvider(
            [
                _parallel_turn(
                    ("lookup-a", "inventory.lookup", {"item": "bearing"}),
                    ("lookup-b", "inventory.lookup", {"item": "spring"}),
                ),
                "Both items are available.",
            ]
        ),
    )
    assert result.status.value == "completed"
    assert len(result.tool_results) == 2
    assert {item.data["item"] for item in result.tool_results} == {"bearing", "spring"}
    assert result.references and result.evidence

    failed = runtime.run(
        _request("failure", "Use a missing tool"),
        provider=CompletionProvider(
            [_tool_turn("missing", "missing.tool", {}), "Recovered from tool failure."]
        ),
    )
    assert failed.status.value == "completed"
    assert failed.tool_results[0].status == "failed"

    terminal = runtime.run(
        _request("provider-failure", "Fail safely"),
        provider=CompletionProvider([RuntimeError("provider unavailable")]),
    )
    assert terminal.status.value == "failed"
    assert terminal.error is not None


def verify_wait_resume_async_and_cancel() -> None:
    runtime = HarnessRuntimeBuilder().add_capability_pack(InventoryPack()).build()
    waiting = runtime.run(
        _request("wait", "Reserve a bearing"),
        provider=CompletionProvider(
            [_tool_turn("reserve", "inventory.reserve", {"item": "bearing"})]
        ),
    )
    assert waiting.status.value == "waiting_user_action"
    assert waiting.pending_action is not None
    resumed = runtime.run(
        _request(
            "wait",
            "Reserve a bearing",
            short_context={
                "resume": True,
                "resume_observation": {
                    "status": "succeeded",
                    "summary": "Reservation confirmed.",
                    "data": {"confirmed": True},
                },
                "previous_runtime": waiting.model_dump(mode="json"),
            },
        ),
        provider=CompletionProvider(["Reservation resumed."]),
    )
    assert resumed.status.value == "completed"

    registry = ToolRegistry(
        [
            ToolSpec(
                name="report.start",
                parameters_schema={"type": "object", "properties": {}},
                async_tool=True,
                read_only=False,
                task_kind="report",
            )
        ]
    )
    executor = ToolExecutor(registry)

    def start_report(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            call=call,
            status="waiting_async",
            summary="Report started.",
            data={"task_id": "report-1"},
            observation=Observation(
                id=f"{call.id}:observation",
                run_id=context.run_id,
                tool_call_id=call.id,
                source=call.name,
                status="pending",
                summary="Report started.",
                data={"task_id": "report-1"},
            ),
        )

    executor.register("report.start", start_report)
    async_runtime = HarnessRuntimeBuilder(registry, executor).build()
    running = async_runtime.run(
        _request("async", "Build report"),
        provider=CompletionProvider([_tool_turn("report", "report.start", {})]),
    )
    assert running.status.value == "task_running"
    async_resumed = async_runtime.run(
        _request(
            "async",
            "Build report",
            short_context={
                "resume": True,
                "resume_observation": {
                    "status": "succeeded",
                    "summary": "Report completed.",
                    "data": {"task_id": "report-1"},
                },
                "previous_runtime": running.model_dump(mode="json"),
            },
        ),
        provider=CompletionProvider(["Async report completed."]),
    )
    assert async_resumed.status.value == "completed"

    control = InMemoryRunControl()
    control.cancel(
        RuntimeScope(user_id="conformance-user").qualify_identity("canceled")
    )
    canceled_runtime = HarnessRuntimeBuilder().with_ports(
        RuntimePorts(run_control=control)
    ).build()
    canceled = canceled_runtime.run(
        _request("canceled", "Do not run"),
        provider=CompletionProvider(["not emitted"]),
    )
    assert canceled.status.value == "canceled"


def verify_skill_artifact_and_reference_contracts() -> None:
    manifest = SkillPackageManifest(
        package_id="conformance.inventory-review",
        capability_pack="conformance.inventory",
        entry_tool="inventory.lookup",
        entry_tools=["inventory.lookup"],
        required_tools=["inventory.lookup"],
        artifact_types=["inventory_record"],
        skill_spec={
            "id": "inventory-review",
                "version": "1.0.0",
                "label": "Inventory review",
                "description": "Review inventory facts and produce a record.",
                "icon_key": "inventory",
                "kind": "prompt",
            "prompt_instructions": "Review inventory facts.",
            "allowed_tools": ["inventory.lookup"],
            "required_tools": ["inventory.lookup"],
            "output_contract": {"requires_artifact": "inventory_record"},
        },
    )
    assert manifest.skill_id == "inventory-review"

    runtime = HarnessRuntimeBuilder().add_capability_pack(InventoryPack()).build()
    skilled = runtime.run(
        _request(
            "skill",
            "Review inventory",
            skill_activation={
                "skill_id": "inventory-review",
                "kind": "prompt",
                "prompt": "Review inventory facts.",
            },
        ),
        provider=CompletionProvider(["Skill response."]),
    )
    assert skilled.skill_activation["skill_id"] == "inventory-review"

    artifact = Artifact(
        id="artifact-1",
        run_id="artifact-run",
        artifact_type="inventory_record",
        data={"item": "bearing"},
    )
    assert artifact.artifact_type == "inventory_record"
    projection = DefaultReferenceProjector()(
        [
            {
                "name": "inventory.lookup",
                "data": {
                    "evidence": [{"id": "record-1", "title": "Ledger"}],
                    "results": [{"url": "https://example.test", "title": "API"}],
                },
            }
        ],
        {},
    )
    assert len(projection.references) == 2
    assert len(projection.evidence) == 1


class FakeMcpClient:
    def __init__(self, spec: McpServerSpec) -> None:
        self.server_id = spec.id
        self.generation = 1

    def list_tools(self, *, timeout_seconds=None):
        return [
            McpRemoteTool(
                name="inventory_lookup",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    def call_tool(self, name, arguments, *, timeout_seconds=None):
        return McpCallResult(
            structured_content={"item": arguments["item"], "quantity": 7},
            metadata={"transport": "stdio"},
        )

    def diagnostics(self):
        return {"id": self.server_id, "running": True}

    def close(self):
        return None


def verify_mcp_and_subagent() -> None:
    server = McpServerSpec(id="inventory-mcp", command="fake")
    pool = McpClientPool([server], client_factory=FakeMcpClient)
    provider = McpToolProvider(pool)
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    provider.register(
        McpToolBinding(
            server_id="inventory-mcp",
            remote_name="inventory_lookup",
            local_spec=ToolSpec(
                name="mcp.inventory_lookup",
                parameters_schema={
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
            ),
            normalize_result=lambda raw, _args: McpNormalizedResult(
                data=raw.structured_content,
                summary="MCP lookup completed.",
            ),
        ),
        registry=registry,
        executor=executor,
    )
    mcp_result = executor.execute(
        ToolCall(
            id="mcp-call",
            name="mcp.inventory_lookup",
            arguments={"item": "bearing"},
        ),
        ToolExecutionContext(run_id="mcp-run", user_id="conformance-user"),
    )
    assert mcp_result.status == "succeeded"
    assert mcp_result.data["quantity"] == 7
    provider.close()

    class SpecialistProvider:
        model = "specialist-model"

        def complete(self, _system_prompt, user_prompt, **_kwargs):
            objective = json.loads(user_prompt)["objective"]
            return json.dumps(
                {
                    "conclusion": f"Reviewed: {objective}",
                    "evidence": ["inventory ledger"],
                    "risks": [],
                    "recommendations": ["continue"],
                }
            )

    subagents = SubAgentExecutor(
        SubAgentRegistry(
            [SubAgentSpec(id="inventory.reviewer", label="Inventory reviewer", model_role="fast")]
        )
    )
    batch = subagents.execute_many(
        DelegationRequest(
            root_run_id="subagent-run",
            parent_run_id="subagent-run",
            tasks=[
                DelegationTask(
                    id="review",
                    agent_id="inventory.reviewer",
                    objective="Review bearing availability",
                )
            ],
        ),
        context=ToolExecutionContext(
            run_id="subagent-run",
            user_id="conformance-user",
        ),
        providers={"fast": SpecialistProvider()},
    )
    assert batch.status == "completed"
    assert batch.results[0].conclusion.startswith("Reviewed:")


def verify_installation_isolation() -> None:
    package_path = Path(deepkeel.__file__).resolve()
    assert "packages/deepkeel/src" not in package_path.as_posix()
    assert importlib.util.find_spec("app") is None
    assert deepkeel.DEEPKEEL_VERSION == "4.1.0rc2"
    assert deepkeel.DEEPKEEL_CONTRACT_VERSION == "harness-core-v3"
    assert tuple(deepkeel.__all__) == (
        "DEEPKEEL_CONTRACT_VERSION",
        "DEEPKEEL_VERSION",
        "adapter_sdk",
        "extension_sdk",
        "mcp_sdk",
        "memory_sdk",
        "orchestration_sdk",
        "runtime_sdk",
    )


def main() -> None:
    verify_installation_isolation()
    verify_runtime_and_streaming()
    verify_tools_parallel_failure_and_references()
    verify_wait_resume_async_and_cancel()
    verify_skill_artifact_and_reference_contracts()
    verify_mcp_and_subagent()
    print("installed deepkeel conformance: passed")


if __name__ == "__main__":
    main()
