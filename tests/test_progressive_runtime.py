from __future__ import annotations

import asyncio

import pytest

from harness_core.capability_manifest import (
    CapabilityManifest,
    RuntimeGeneration,
    validate_manifest_set,
)
from harness_core.context_window import (
    ContextSegment,
    ContextWindowPolicy,
    DeterministicContextWindowManager,
)
from harness_core.contracts import ToolCall, ToolResult
from harness_core.hooks import (
    HookDecision,
    HookPoint,
    HookRunner,
    HookSpec,
)
from harness_core.skills import SkillPolicy
from harness_core.tool_disclosure import (
    ToolDescriptor,
    ToolDiscoveryPort,
    discover_tools,
    resolve_tool_view,
)
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tools import ToolExecutionContext, ToolExecutor


def _tool(name: str, *, exposure_mode: str = "discoverable") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Capability for {name}",
        parameters_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        },
        required_args=["count"],
        exposure_mode=exposure_mode,
        read_only=True,
    )


def test_tool_hook_arguments_are_revalidated_before_handler_execution() -> None:
    registry = ToolRegistry([_tool("demo.count")])
    called = False

    def handler(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        nonlocal called
        called = True
        return ToolResult(call=call, status="succeeded", summary="ok")

    hooks = HookRunner()
    hooks.register(
        HookSpec(
            id="rewrite-invalid",
            point=HookPoint.TOOL_BEFORE,
            handler=lambda _: HookDecision(tool_arguments={"count": "not-an-int"}),
        )
    )
    executor = ToolExecutor(registry, hook_runner=hooks)
    executor.register("demo.count", handler)

    result = asyncio.run(
        executor.aexecute(
            ToolCall(id="call-1", name="demo.count", arguments={"count": 1}),
            ToolExecutionContext(run_id="run-1", user_id="user-1"),
        )
    )

    assert result.status == "failed"
    assert "integer" in result.error
    assert called is False


def test_enforced_tool_view_never_fails_open_for_large_catalog() -> None:
    registry = ToolRegistry([_tool(f"demo.tool-{index}") for index in range(12)])

    view = resolve_tool_view(
        registry=registry,
        allowed_names=None,
        skill=SkillPolicy.from_snapshot({}),
        mode="enforced",
    )

    assert view.exposed_names == frozenset()
    assert view.fail_open is False
    assert view.direct_injection is False
    assert len(view.filtered_names) == 12


class _ReverseDiscovery(ToolDiscoveryPort):
    def discover(
        self,
        *,
        query: str,
        candidates: tuple[ToolDescriptor, ...],
        limit: int,
    ) -> tuple[ToolDescriptor, ...]:
        del query
        return tuple(reversed(candidates))[:limit]


def test_discovery_port_is_bounded_and_cannot_escape_candidate_scope() -> None:
    registry = ToolRegistry([_tool(f"demo.tool-{index}") for index in range(8)])

    discovered = discover_tools(
        registry,
        query="anything",
        limit=8,
        allowed_names={f"demo.tool-{index}" for index in range(6)},
        discovery_port=_ReverseDiscovery(),
    )

    assert len(discovered) == 5
    assert {item.name for item in discovered}.issubset(
        {f"demo.tool-{index}" for index in range(6)}
    )


def test_context_layers_keep_protected_state_and_report_sources() -> None:
    manager = DeterministicContextWindowManager(
        ContextWindowPolicy(
            max_input_tokens=180,
            reserved_output_tokens=40,
            minimum_section_tokens=8,
        )
    )
    result = manager.prepare(
        "question",
        {},
        {
            "runtime_context": {
                "current_goal": "finish the durable workflow",
                "search_results": "x" * 800,
            },
            "context_segments": [
                ContextSegment(
                    key="current_goal",
                    value="finish the durable workflow",
                    source="runtime_state",
                    layer="turn_context",
                    retention="protected",
                ),
                ContextSegment(
                    key="search_results",
                    value="x" * 800,
                    source="retrieval",
                    layer="retrieved_context",
                    retention="ephemeral",
                ),
            ],
        },
    )

    runtime_context = result.context_bundle["runtime_context"]
    assert runtime_context["current_goal"] == "finish the durable workflow"
    assert "current_goal" in result.diagnostics["protected_sections_retained"]
    assert "runtime_state" in result.diagnostics["injection_sources"]
    assert result.diagnostics["layers"]["turn_context"]["tokens"] > 0


def _manifest(
    package_id: str,
    *,
    version: str = "1.0.0",
    dependencies: dict[str, str] | None = None,
    tools: tuple[str, ...] = (),
) -> CapabilityManifest:
    return CapabilityManifest(
        id=package_id,
        version=version,
        core_version=">=3.16.0,<4.0.0",
        entrypoint=f"{package_id}:Pack",
        dependencies=dependencies or {},
        tools=tools,
    )


def test_runtime_generation_validates_dependencies_and_is_deterministic() -> None:
    foundation = _manifest("demo.foundation", version="2.0.0")
    planning = _manifest(
        "demo.planning",
        dependencies={"demo.foundation": ">=2.0.0"},
        tools=("planning.run",),
    )

    first = RuntimeGeneration.create(
        (planning, foundation),
        catalog_version="catalog-1",
    )
    second = RuntimeGeneration.create(
        (foundation, planning),
        catalog_version="catalog-1",
    )

    assert first.generation_id == second.generation_id
    assert first.package_versions()["demo.planning"] == "1.0.0"

    incompatible = _manifest(
        "demo.broken",
        dependencies={"demo.foundation": ">=3.0.0"},
    )
    with pytest.raises(ValueError, match="requires demo.foundation"):
        validate_manifest_set((foundation, incompatible))
