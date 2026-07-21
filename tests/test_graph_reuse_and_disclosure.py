from __future__ import annotations

import asyncio

from harness_core.adapter_sdk import HarnessRuntimeBuilder, RuntimePorts
from harness_core.contracts import ToolCall
from harness_core.extension_sdk import ToolExecutor, ToolRegistry, ToolSpec
from harness_core.policy import DefaultPolicyEngine, PolicyRequest
from harness_core.runtime_sdk import RuntimeRequest
from harness_core.skills import SkillPolicy
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
    assert decision.allowed is False
    assert decision.metadata["rule"] == "skill_tool_allowlist"


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
