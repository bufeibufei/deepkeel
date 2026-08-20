from __future__ import annotations

import asyncio

import pytest

from deepkeel.composition import HarnessRuntimeBuilder
from deepkeel.entrypoints import (
    AgentEntrypointSpec,
    narrow_capability_view,
    resolve_capability_view,
)
from deepkeel.extension_sdk import (
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityManifest,
    CapabilityPackSpec,
)
from deepkeel.runtime_sdk import RuntimeRequest
from deepkeel.subagents.contracts import SubAgentSpec
from deepkeel.tool_registry import ToolSpec


class RecordingProvider:
    model = "entrypoint-recorder"
    model_role = "reasoning"

    def __init__(self) -> None:
        self.system_prompts: list[str] = []
        self.tool_views: list[list[str]] = []

    def complete_chat(self, messages, *, tools=None, **_kwargs):
        self.system_prompts.append(
            next(
                (
                    str(message.get("content") or "")
                    for message in messages
                    if message.get("role") == "system"
                ),
                "",
            )
        )
        self.tool_views.append(
            sorted(str(item.get("function", {}).get("name") or "") for item in tools or [])
        )
        return {
            "message": {"role": "assistant", "content": "completed"},
            "finish_reason": "stop",
            "model": self.model,
        }


class FoundationPack:
    spec = CapabilityPackSpec(
        package_id="demo.foundation",
        package_version="1.0.0",
        declared_tools=("foundation.lookup",),
        declared_context_contributors=("foundation.context",),
    )

    def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
        context.register_tool(
            ToolSpec(name="foundation.lookup", read_only=True),
            lambda _arguments, _context: {"ok": True},
        )
        context.register_context_contributor(
            "foundation.context",
            # A badly behaved pack must not be able to erase the runtime scope.
            lambda _current: {"foundation": "ready"},
        )
        return CapabilityContribution(
            package_id=self.spec.package_id,
            tools=("foundation.lookup",),
            context_contributors=("foundation.context",),
        )


class SpecialistPack:
    spec = CapabilityPackSpec(
        package_id="demo.specialist",
        package_version="1.0.0",
        declared_tools=("specialist.inspect",),
        declared_skills=("specialist.analysis",),
        declared_subagents=("specialist.reviewer",),
        declared_context_contributors=("specialist.context",),
        declared_agent_entrypoints=("specialist",),
    )

    def __init__(self, *, invalid_tool: bool = False) -> None:
        self.invalid_tool = invalid_tool

    def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
        context.register_tool(
            ToolSpec(name="specialist.inspect", read_only=True),
            lambda _arguments, _context: {"finding": "stable"},
        )
        context.register_skill("specialist.analysis", {"label": "Analysis"})
        context.register_subagent(
            SubAgentSpec(
                id="specialist.reviewer",
                label="Reviewer",
                tool_allowlist=["specialist.inspect"],
            )
        )
        context.register_context_contributor(
            "specialist.context",
            lambda current: {**current, "specialist": "ready"},
        )
        context.register_agent_entrypoint(
            AgentEntrypointSpec(
                id="specialist",
                version="1.0.0",
                label="Specialist Agent",
                description="A directly addressable specialist.",
                skill_allowlist=("specialist.analysis",),
                tool_allowlist=(
                    "specialist.inspect",
                    "missing.tool" if self.invalid_tool else "foundation.lookup",
                ),
                subagent_allowlist=("specialist.reviewer",),
                system_prompt="You are the specialist entrypoint.",
                model_policy={"roles": {"reasoning": {"preferred": True}}},
                memory_policy={"recall": "specialist_only"},
            )
        )
        return CapabilityContribution(
            package_id=self.spec.package_id,
            tools=("specialist.inspect",),
            skills=("specialist.analysis",),
            subagents=("specialist.reviewer",),
            context_contributors=("specialist.context",),
            agent_entrypoints=("specialist",),
        )


class UnrelatedPack:
    spec = CapabilityPackSpec(
        package_id="demo.unrelated",
        package_version="1.0.0",
        declared_tools=("unrelated.write",),
        declared_skills=("unrelated.skill",),
    )

    def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
        context.register_tool(
            ToolSpec(name="unrelated.write", read_only=False),
            lambda _arguments, _context: {"ok": True},
        )
        context.register_skill("unrelated.skill", {"label": "Unrelated"})
        return CapabilityContribution(
            package_id=self.spec.package_id,
            tools=("unrelated.write",),
            skills=("unrelated.skill",),
        )


FOUNDATION_MANIFEST = CapabilityManifest(
    id="demo.foundation",
    version="1.0.0",
    core_version="*",
    entrypoint="tests.test_agent_entrypoints:FoundationPack",
    tools=("foundation.lookup",),
    context_contributors=("foundation.context",),
    memory_namespaces=("foundation",),
    permissions=("foundation:read",),
)

SPECIALIST_MANIFEST = CapabilityManifest(
    id="demo.specialist",
    version="1.0.0",
    core_version="*",
    entrypoint="tests.test_agent_entrypoints:SpecialistPack",
    dependencies={"demo.foundation": ">=1.0.0"},
    tools=("specialist.inspect",),
    skills=("specialist.analysis",),
    subagents=("specialist.reviewer",),
    context_contributors=("specialist.context",),
    agent_entrypoints=("specialist",),
    memory_namespaces=("specialist",),
    permissions=("specialist:read",),
)

UNRELATED_MANIFEST = CapabilityManifest(
    id="demo.unrelated",
    version="1.0.0",
    core_version="*",
    entrypoint="tests.test_agent_entrypoints:UnrelatedPack",
    tools=("unrelated.write",),
    skills=("unrelated.skill",),
)


def _runtime(*, invalid_tool: bool = False):
    return (
        HarnessRuntimeBuilder()
        .add_capability_pack(FoundationPack(), manifest=FOUNDATION_MANIFEST)
        .add_capability_pack(UnrelatedPack(), manifest=UNRELATED_MANIFEST)
        .add_capability_pack(
            SpecialistPack(invalid_tool=invalid_tool),
            manifest=SPECIALIST_MANIFEST,
        )
        .build()
    )


def test_entrypoint_resolves_dependency_closed_capability_view() -> None:
    runtime = _runtime()

    view = resolve_capability_view(
        entrypoint_id="specialist",
        entrypoint_version="1.0.0",
        runtime_generation=runtime.runtime_generation,
        contributions=runtime.capability_contributions,
        catalog=runtime.capability_catalog,
        installed_tool_names=(tool.name for tool in runtime.tool_registry.list_tools()),
    )

    assert view.restricted is True
    assert view.package_ids == ("demo.foundation", "demo.specialist")
    assert view.skill_ids == ("specialist.analysis",)
    assert view.tool_names == ("foundation.lookup", "specialist.inspect")
    assert view.subagent_ids == ("specialist.reviewer",)
    assert view.context_contributor_ids == (
        "foundation.context",
        "specialist.context",
    )
    assert view.memory_namespaces == ("foundation", "specialist")
    assert view.permission_scopes == ("foundation:read", "specialist:read")
    assert view.scope_hash


def test_entrypoint_rejects_out_of_scope_declaration_atomically() -> None:
    with pytest.raises(ValueError, match="out-of-scope tools: missing.tool"):
        _runtime(invalid_tool=True)


def test_entrypoint_scope_filters_tools_and_reuses_one_compiled_graph() -> None:
    async def scenario():
        runtime = _runtime()
        specialist_provider = RecordingProvider()
        default_provider = RecordingProvider()
        specialist_result, default_result = await asyncio.gather(
            runtime.arun(
                RuntimeRequest(
                    question="inspect",
                    run_id="entrypoint-specialist",
                    agent_entrypoint_id="specialist",
                    agent_entrypoint_version="1.0.0",
                ),
                provider=specialist_provider,
            ),
            runtime.arun(
                RuntimeRequest(question="general", run_id="entrypoint-default"),
                provider=default_provider,
            ),
        )
        return runtime, specialist_provider, default_provider, specialist_result, default_result

    runtime, specialist, default, specialist_result, default_result = asyncio.run(scenario())

    assert specialist_result.status == "completed"
    assert default_result.status == "completed"
    assert specialist.tool_views == [["foundation.lookup", "specialist.inspect"]]
    assert "unrelated.write" in default.tool_views[0]
    assert "You are the specialist entrypoint." in specialist.system_prompts[0]
    assert "You are the specialist entrypoint." not in default.system_prompts[0]
    assert runtime.graph_compile_count == 1
    assert any(
        event.event_type == "agent.entrypoint.resolved" for event in specialist_result.events
    )


def test_entrypoint_rejects_explicit_skill_outside_scope_as_runtime_failure() -> None:
    runtime = _runtime()

    result = runtime.run(
        RuntimeRequest(
            question="escape",
            run_id="entrypoint-skill-boundary",
            agent_entrypoint_id="specialist",
            skill_activation={"skill_id": "unrelated.skill"},
        ),
        provider=RecordingProvider(),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error["code"] == "RUNTIME_CONTRACT_INVALID"


def test_child_capability_view_can_only_narrow_parent_scope() -> None:
    runtime = _runtime()
    parent = resolve_capability_view(
        entrypoint_id="specialist",
        runtime_generation=runtime.runtime_generation,
        contributions=runtime.capability_contributions,
        catalog=runtime.capability_catalog,
        installed_tool_names=(tool.name for tool in runtime.tool_registry.list_tools()),
    )

    child = narrow_capability_view(
        parent,
        tool_names=("specialist.inspect",),
        subagent_ids=(),
    )

    assert child.tool_names == ("specialist.inspect",)
    assert child.subagent_ids == ()
    assert child.scope_hash != parent.scope_hash
    with pytest.raises(ValueError, match="cannot add tool_names"):
        narrow_capability_view(parent, tool_names=("unrelated.write",))
