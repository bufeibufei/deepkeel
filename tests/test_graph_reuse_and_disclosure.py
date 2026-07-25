from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from harness_core.adapter_sdk import (
    HarnessRuntimeBuilder,
    LangGraphCheckpointerAdapter,
    RuntimePorts,
)
from harness_core.contracts import Artifact, ToolCall, ToolResult
from harness_core.extension_sdk import ToolExecutor, ToolRegistry, ToolSpec
from harness_core.policy import DefaultPolicyEngine, PolicyRequest
from harness_core.runtime_sdk import RuntimeRequest
from harness_core.skills import SkillPolicy
from harness_core.graph_state import _allowed_tool_names, _upsert_artifact
from harness_core.tool_disclosure import (
    TOOL_DISCOVERY_NAME,
    install_tool_discovery,
    resolve_tool_view,
)
from harness_core.tools import ToolExecutionContext


class PromptEchoProvider:
    model = "prompt-echo"
    model_role = "fast"

    def stream_chat(self, messages, **_kwargs):
        system = next(
            (str(message.get("content") or "") for message in messages if message.get("role") == "system"),
            "missing-system",
        )
        yield {
            "choices": [
                {
                    "delta": {"content": system},
                    "finish_reason": "stop",
                }
            ]
        }


class CountingSaver(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self.put_count = 0

    def put(self, config, checkpoint, metadata, new_versions):
        self.put_count += 1
        return super().put(config, checkpoint, metadata, new_versions)


class WorkflowFinalizationProvider:
    model = "workflow-finalization"
    model_role = "reasoning"

    def __init__(self) -> None:
        self.tool_views: list[list[str]] = []

    def complete_chat(self, _messages, *, tools=None, **_kwargs):
        names = [
            str(item.get("function", {}).get("name") or "")
            for item in tools or []
        ]
        self.tool_views.append(names)
        if len(self.tool_views) == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "build-report",
                            "type": "function",
                            "function": {
                                "name": "workflow.build_report",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "model": self.model,
            }
        return {
            "message": {
                "role": "assistant",
                "content": "The completed report is ready.",
            },
            "finish_reason": "stop",
            "model": self.model,
        }


def test_repeated_artifact_observations_update_one_durable_identity() -> None:
    state = {"artifacts": []}
    _upsert_artifact(
        state,
        {
            "id": "artifact-1",
            "run_id": "run-1",
            "artifact_type": "report",
            "summary": "生成中",
            "data": {"status": "running"},
            "metadata": {"source": "first"},
            "created_at": "2026-07-25T00:00:00Z",
        },
    )
    _upsert_artifact(
        state,
        {
            "id": "artifact-1",
            "run_id": "run-1",
            "artifact_type": "report",
            "summary": "已完成",
            "data": {"status": "completed", "result_id": "result-1"},
            "metadata": {"source": "second"},
            "created_at": "2026-07-25T00:01:00Z",
        },
    )

    assert len(state["artifacts"]) == 1
    assert state["artifacts"][0]["summary"] == "已完成"
    assert state["artifacts"][0]["data"] == {
        "status": "completed",
        "result_id": "result-1",
    }
    assert state["artifacts"][0]["metadata"] == {"source": "second"}
    assert state["artifacts"][0]["created_at"] == "2026-07-25T00:00:00Z"


def test_completed_workflow_enters_tool_free_answer_finalization() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="workflow.build_report",
                exposure_mode="skill_entry",
                read_only=False,
            )
        ]
    )
    executor = ToolExecutor(registry)

    def build_report(call, context):
        return ToolResult(
            call=call,
            status="succeeded",
            summary="Report completed.",
            artifacts=[
                Artifact(
                    id="report-1",
                    run_id=context.run_id,
                    artifact_type="report",
                    summary="Report completed.",
                    data={"status": "completed"},
                )
            ],
        )

    executor.register("workflow.build_report", build_report)
    runtime = HarnessRuntimeBuilder(registry, executor).build()
    executor.configure_artifact_schemas({"report": {"type": "object"}})
    provider = WorkflowFinalizationProvider()

    result = runtime.run(
        RuntimeRequest(
            question="Build a report.",
            run_id="workflow-finalization",
            skill_activation={
                "skill_id": "report",
                "kind": "workflow",
                "tool_scope_mode": "allowlist",
                "allowed_tools": ["workflow.build_report"],
                "required_tools": ["workflow.build_report"],
                "output_contract": {"requires_artifact": "report"},
            },
        ),
        provider=provider,
    )

    assert result.status == "completed", result.model_dump(mode="json")
    assert provider.tool_views == [["workflow.build_report"], []]
    assert [item.id for item in result.artifacts] == ["report-1"]


def test_default_graph_durability_checkpoints_only_at_runtime_exit() -> None:
    saver = CountingSaver()
    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(
            RuntimePorts(
                checkpointer=LangGraphCheckpointerAdapter(saver),
            )
        )
        .build()
    )

    result = runtime.run(
        RuntimeRequest(question="hello", run_id="exit-durability"),
        provider=PromptEchoProvider(),
    )

    assert result.status == "completed"
    assert saver.put_count == 1
    assert result.diagnostics["execution_contract"]["graph_durability"] == "exit"


def test_runtime_reuses_one_graph_without_leaking_turn_prompt() -> None:
    async def scenario():
        runtime = (
            HarnessRuntimeBuilder()
            .with_ports(
                RuntimePorts(
                    system_prompt_factory=lambda skill: f"prompt:{skill.get('skill_id', '')}",
                    reuse_compiled_graph=True,
                )
            )
            .build()
        )
        first, second = await asyncio.gather(
            runtime.arun(
                RuntimeRequest(
                    question="first",
                    run_id="reuse-first",
                    skill_activation={"skill_id": "first"},
                ),
                provider=PromptEchoProvider(),
            ),
            runtime.arun(
                RuntimeRequest(
                    question="second",
                    run_id="reuse-second",
                    skill_activation={"skill_id": "second"},
                ),
                provider=PromptEchoProvider(),
            ),
        )
        return runtime, first, second

    runtime, first, second = asyncio.run(scenario())

    assert first.final_answer.markdown == "prompt:first"
    assert second.final_answer.markdown == "prompt:second"
    assert runtime.graph_compile_count == 1
    assert first.diagnostics["execution_contract"]["graph_reused"] is True


def test_shadow_tool_view_observes_progressive_selection_without_changing_exposure() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(name="general.read", exposure_mode="baseline"),
            ToolSpec(name="catalog.search", exposure_mode="discoverable"),
            ToolSpec(name="skill.run", exposure_mode="skill_entry"),
            ToolSpec(name="runtime.internal", exposure_mode="internal"),
        ]
    )
    skill = SkillPolicy.from_snapshot(
        {
            "skill_id": "demo",
            "allowed_tools": ["general.read", "catalog.search", "skill.run"],
            "required_tools": ["catalog.search"],
        }
    )

    view = resolve_tool_view(
        registry=registry,
        allowed_names=set(skill.allowed_tools),
        skill=skill,
        mode="shadow",
    )

    assert view.exposed_names == skill.allowed_tools
    assert view.proposed_names == frozenset(
        {"general.read", "catalog.search", "skill.run"}
    )
    assert view.catalog_version == registry.catalog_version()


def test_active_skill_with_empty_allowlist_denies_tool_execution() -> None:
    skill = SkillPolicy.from_snapshot(
        {
            "skill_id": "no-tools",
            "tool_scope_mode": "allowlist",
            "allowed_tools": [],
        }
    )
    decision = DefaultPolicyEngine().evaluate(
        PolicyRequest(
            action="tool.invoke",
            resource_type="tool",
            resource_id="general.read",
            run_id="run-1",
            user_id="user-1",
            context={"skill_activation": skill.runtime_snapshot()},
        )
    )

    assert skill.allows_tool("general.read") is False
    assert skill.allows_tool(TOOL_DISCOVERY_NAME) is False
    assert decision.allowed is False
    assert decision.metadata["rule"] == "skill_tool_allowlist"


def test_nonempty_skill_allowlist_permits_catalog_discovery_without_expanding_scope() -> None:
    skill = SkillPolicy.from_snapshot(
        {
            "skill_id": "search-only",
            "tool_scope_mode": "allowlist",
            "allowed_tools": ["web.search"],
        }
    )

    decision = DefaultPolicyEngine().evaluate(
        PolicyRequest(
            action="tool.invoke",
            resource_type="tool",
            resource_id=TOOL_DISCOVERY_NAME,
            run_id="run-1",
            user_id="user-1",
            context={"skill_activation": skill.runtime_snapshot()},
        )
    )

    assert skill.allows_tool(TOOL_DISCOVERY_NAME) is True
    assert decision.allowed is True


def test_discovery_tool_grants_only_ranked_permitted_tools_to_enforced_view() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(name="general.read", exposure_mode="baseline"),
            ToolSpec(
                name="web.search",
                description="Search current internet sources and news.",
                exposure_mode="discoverable",
                discovery_tags=["web", "search", "current"],
            ),
            ToolSpec(
                name="private.write",
                description="Mutate a private record.",
                exposure_mode="skill_only",
                discovery_tags=["private", "write"],
            ),
        ]
    )
    executor = ToolExecutor(registry)
    install_tool_discovery(registry, executor)
    context = ToolExecutionContext(run_id="run-discovery", user_id="user-1")

    result = executor.execute(
        ToolCall(
            id="call-discovery",
            name=TOOL_DISCOVERY_NAME,
            arguments={"query": "current web search"},
            idempotency_key="discover:current-web",
        ),
        context,
    )

    assert result.status == "succeeded"
    assert result.data["discovered_names"] == ["web.search"]
    view = resolve_tool_view(
        registry=registry,
        allowed_names={spec.name for spec in registry.list_tools()},
        skill=SkillPolicy.from_snapshot({}),
        mode="enforced",
        discovered_names=set(result.data["discovered_names"]),
    )
    assert view.exposed_names == frozenset(
        {"general.read", TOOL_DISCOVERY_NAME, "web.search"}
    )
    assert "private.write" not in view.exposed_names


def test_tool_discovery_is_hidden_after_two_attempts() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(name=TOOL_DISCOVERY_NAME, exposure_mode="baseline"),
            ToolSpec(name="web.search", exposure_mode="discoverable"),
        ]
    )
    state = {
        "skill_activation": {
            "skill_id": "research",
            "tool_scope_mode": "allowlist",
            "allowed_tools": ["web.search"],
        },
        "tool_results": [
            {"name": TOOL_DISCOVERY_NAME, "status": "succeeded"},
            {"name": TOOL_DISCOVERY_NAME, "status": "succeeded"},
        ],
        "metadata": {},
    }

    assert _allowed_tool_names(state, registry) == {"web.search"}
