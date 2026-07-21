from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from harness_core.contracts import Observation, ToolCall, ToolResult
from harness_core.skills import SkillPolicy
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tools import ToolExecutionContext, ToolExecutor
from harness_core.turn_context import ToolViewMode


TOOL_DISCOVERY_NAME = "runtime.discover_tools"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    description: str
    exposure_mode: str
    tags: tuple[str, ...]

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> "ToolDescriptor":
        exposure_mode = spec.exposure_mode
        if str(spec.runtime_policy.get("model_exposure") or "") == "skill_only":
            exposure_mode = "skill_only"
        return cls(
            name=spec.name,
            description=spec.description,
            exposure_mode=exposure_mode,
            tags=tuple(sorted(str(tag) for tag in spec.discovery_tags if str(tag).strip())),
        )


@dataclass(frozen=True, slots=True)
class ToolView:
    mode: ToolViewMode
    catalog_version: str
    allowed_names: frozenset[str]
    exposed_names: frozenset[str]
    proposed_names: frozenset[str]
    fail_open: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "catalog_version": self.catalog_version,
            "allowed_names": sorted(self.allowed_names),
            "exposed_names": sorted(self.exposed_names),
            "proposed_names": sorted(self.proposed_names),
            "fail_open": self.fail_open,
        }


def resolve_tool_view(
    *,
    registry: ToolRegistry,
    allowed_names: set[str] | None,
    skill: SkillPolicy,
    mode: ToolViewMode,
    discovered_names: set[str] | None = None,
) -> ToolView:
    installed = {spec.name for spec in registry.list_tools()}
    allowed = installed if allowed_names is None else installed & set(allowed_names)
    proposed: set[str] = set()
    for spec in registry.list_tools():
        if spec.name not in allowed:
            continue
        descriptor = ToolDescriptor.from_spec(spec)
        if descriptor.exposure_mode == "baseline":
            proposed.add(spec.name)
        elif skill.active and descriptor.exposure_mode in {"skill_entry", "skill_only"}:
            proposed.add(spec.name)
    proposed.update(skill.required_tools & allowed)
    for group in skill.required_tool_groups:
        proposed.update(group & allowed)
    proposed.update(set(discovered_names or ()) & allowed)

    fail_open = False
    if mode == "legacy":
        exposed = allowed
    elif mode == "shadow":
        exposed = allowed
    else:
        exposed = proposed
        # Until semantic discovery is configured, never strand a turn that has
        # executable tools but no discoverable entry point.
        if allowed and not exposed:
            exposed = allowed
            fail_open = True
    return ToolView(
        mode=mode,
        catalog_version=registry.catalog_version(),
        allowed_names=frozenset(allowed),
        exposed_names=frozenset(exposed),
        proposed_names=frozenset(proposed),
        fail_open=fail_open,
    )


def tool_discovery_spec() -> ToolSpec:
    """Return the model-facing catalog search tool used for progressive disclosure."""

    return ToolSpec(
        name=TOOL_DISCOVERY_NAME,
        description=(
            "Search the available tool catalog when the currently visible tools cannot "
            "complete the request. Describe the capability you need, not a tool name."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Capability or action needed for the current task.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        required_args=["query"],
        read_only=True,
        parallel_safe=True,
        visible_label="Discover tools",
        exposure_mode="baseline",
        discovery_tags=["tool", "catalog", "capability", "discovery"],
        runtime_policy={"internal_runtime_tool": True},
    )


def install_tool_discovery(registry: ToolRegistry, executor: ToolExecutor) -> None:
    """Install the portable discovery entrypoint without coupling it to a product pack."""

    if TOOL_DISCOVERY_NAME not in {spec.name for spec in registry.list_tools()}:
        registry.register(tool_discovery_spec())

    def discover(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        query = str(call.arguments.get("query") or "").strip()
        try:
            limit = max(1, min(int(call.arguments.get("limit") or 5), 8))
        except (TypeError, ValueError):
            limit = 5
        descriptors = discover_tools(
            registry,
            query=query,
            limit=limit,
            allowed_names=_context_allowed_names(registry, context),
        )
        names = [item.name for item in descriptors]
        summary = (
            f"Discovered {len(names)} relevant tool(s)."
            if names
            else "No additional relevant tools were found."
        )
        data = {
            "query": query,
            "discovered_names": names,
            "tools": [
                {
                    "name": item.name,
                    "description": item.description,
                    "tags": list(item.tags),
                }
                for item in descriptors
            ],
            "catalog_version": registry.catalog_version(),
        }
        return ToolResult(
            call=call,
            status="succeeded",
            summary=summary,
            data=data,
            observation=Observation(
                id=f"{call.id}:tool-discovery",
                run_id=context.run_id,
                tool_call_id=call.id,
                source=TOOL_DISCOVERY_NAME,
                status="succeeded",
                summary=summary,
                data=data,
                metadata={"visible": False},
            ),
            metadata={"visible": False, "runtime_internal": True},
        )

    executor.register(TOOL_DISCOVERY_NAME, discover)


def discover_tools(
    registry: ToolRegistry,
    *,
    query: str,
    limit: int = 5,
    allowed_names: set[str] | None = None,
) -> tuple[ToolDescriptor, ...]:
    """Deterministically rank discoverable tools by description, tags, and name."""

    tokens = _search_tokens(query)
    ranked: list[tuple[int, str, ToolDescriptor]] = []
    for spec in registry.list_tools():
        if allowed_names is not None and spec.name not in allowed_names:
            continue
        descriptor = ToolDescriptor.from_spec(spec)
        if descriptor.name == TOOL_DISCOVERY_NAME or descriptor.exposure_mode not in {
            "discoverable",
            "skill_entry",
        }:
            continue
        haystack = " ".join(
            [descriptor.name.replace(".", " "), descriptor.description, *descriptor.tags]
        ).lower()
        score = sum(3 if token in descriptor.tags else 1 for token in tokens if token in haystack)
        if query.strip() and query.strip().lower() in haystack:
            score += 4
        if score > 0:
            ranked.append((score, descriptor.name, descriptor))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in ranked[: max(1, min(int(limit), 8))])


def _context_allowed_names(
    registry: ToolRegistry,
    context: ToolExecutionContext,
) -> set[str]:
    skill = SkillPolicy.from_snapshot(context.metadata.get("skill_activation"))
    raw = context.metadata.get("skill_activation")
    snapshot = raw if isinstance(raw, dict) else {}
    if skill.active and skill.tool_scope_mode == "allowlist":
        return {str(name) for name in snapshot.get("allowed_tools", []) if str(name).strip()}
    return {
        spec.name
        for spec in registry.list_tools()
        if str(spec.runtime_policy.get("model_exposure") or "always") != "skill_only"
    }


def _search_tokens(value: str) -> set[str]:
    normalized = str(value or "").strip().lower()
    tokens = {token for token in re.split(r"[^\w\u4e00-\u9fff]+", normalized) if token}
    # Chinese capability descriptions often have no whitespace. Character n-grams
    # retain deterministic matching without introducing an embedding dependency.
    compact = "".join(char for char in normalized if "\u4e00" <= char <= "\u9fff")
    tokens.update(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
    return tokens
