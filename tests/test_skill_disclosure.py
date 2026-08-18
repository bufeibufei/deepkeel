from __future__ import annotations

from deepkeel.capabilities import CapabilityCatalog
from deepkeel.contracts import ToolCall
from deepkeel.skill_disclosure import (
    SKILL_DISCOVERY_NAME,
    SkillDescriptor,
    SkillDiscoveryPort,
    discover_skills,
    install_skill_discovery,
)
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor


def _entry_tool(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Entry for {name}",
        parameters_schema={"type": "object", "properties": {}},
        exposure_mode="skill_entry",
        read_only=True,
    )


def _skill(label: str, description: str, entry_tool: str) -> dict[str, object]:
    return {
        "label": label,
        "description": description,
        "kind": "workflow",
        "invocation_modes": ["model"],
        "allowed_tools": [entry_tool],
        "package": {"entry_tools": [entry_tool]},
    }


def test_non_top_skill_remains_discoverable_for_a_later_query() -> None:
    registry = ToolRegistry(
        [
            _entry_tool("report.start"),
            _entry_tool("calendar.start"),
            _entry_tool("search.start"),
            _entry_tool("naming.start"),
        ]
    )
    catalog = CapabilityCatalog()
    catalog.register_skill("report", _skill("Report", "Build reports", "report.start"))
    catalog.register_skill("calendar", _skill("Calendar", "Select dates", "calendar.start"))
    catalog.register_skill("search", _skill("Search", "Search sources", "search.start"))
    catalog.register_skill("naming", _skill("Naming", "Create names", "naming.start"))

    first = discover_skills(catalog, registry, query="Build reports", limit=3)
    later = discover_skills(catalog, registry, query="Create names", limit=3)

    assert first[0].skill_id == "report"
    assert later[0].skill_id == "naming"


class _EscapingRecall(SkillDiscoveryPort):
    def discover(
        self,
        *,
        query: str,
        candidates: tuple[SkillDescriptor, ...],
        limit: int,
    ) -> tuple[SkillDescriptor, ...]:
        del query, limit
        return (
            *candidates,
            SkillDescriptor(
                skill_id="forbidden",
                label="Forbidden",
                description="",
                kind="workflow",
                invocation_modes=("model",),
                entry_tools=("forbidden.start",),
            ),
        )


def test_discovery_adapter_cannot_escape_capability_scope() -> None:
    registry = ToolRegistry([_entry_tool("allowed.start"), _entry_tool("hidden.start")])
    catalog = CapabilityCatalog()
    catalog.register_skill("allowed", _skill("Allowed", "Allowed task", "allowed.start"))
    catalog.register_skill("hidden", _skill("Hidden", "Hidden task", "hidden.start"))

    result = discover_skills(
        catalog,
        registry,
        query="task",
        allowed_skill_ids={"allowed"},
        allowed_tool_names={"allowed.start"},
        discovery_port=_EscapingRecall(),
    )

    assert [item.skill_id for item in result] == ["allowed"]
    assert result[0].entry_tools == ("allowed.start",)


def test_model_facing_discovery_only_discloses_permitted_entry_tools() -> None:
    registry = ToolRegistry([_entry_tool("allowed.start"), _entry_tool("hidden.start")])
    executor = ToolExecutor(registry)
    catalog = CapabilityCatalog()
    catalog.register_skill("allowed", _skill("Allowed", "Allowed task", "allowed.start"))
    catalog.register_skill("hidden", _skill("Hidden", "Hidden task", "hidden.start"))
    install_skill_discovery(catalog, registry, executor)

    result = executor.execute(
        ToolCall(
            id="discover-1",
            name=SKILL_DISCOVERY_NAME,
            arguments={"query": "Allowed task"},
        ),
        ToolExecutionContext(
            run_id="run-1",
            user_id="user-1",
            metadata={
                "capability_view": {
                    "restricted": True,
                    "skill_ids": ["allowed"],
                    "tool_names": ["allowed.start", SKILL_DISCOVERY_NAME],
                }
            },
        ),
    )

    assert result.status == "succeeded"
    assert result.data["discovered_skill_ids"] == ["allowed"]
    assert result.data["discovered_tool_names"] == ["allowed.start"]
