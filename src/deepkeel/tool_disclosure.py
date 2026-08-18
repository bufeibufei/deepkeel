from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol

from deepkeel.contracts import Observation, ToolCall, ToolResult
from deepkeel.skills import SkillPolicy
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor
from deepkeel.turn_context import ToolViewMode


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
    direct_injection: bool = False
    filtered_names: frozenset[str] = frozenset()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "catalog_version": self.catalog_version,
            "allowed_names": sorted(self.allowed_names),
            "exposed_names": sorted(self.exposed_names),
            "proposed_names": sorted(self.proposed_names),
            "fail_open": self.fail_open,
            "direct_injection": self.direct_injection,
            "filtered_names": sorted(self.filtered_names),
        }


class ToolDiscoveryPort(Protocol):
    """Host-replaceable broad-recall boundary."""

    def discover(
        self,
        *,
        query: str,
        candidates: tuple[ToolDescriptor, ...],
        limit: int,
    ) -> tuple[ToolDescriptor, ...]: ...


class ToolRerankerPort(Protocol):
    """Optional second-stage ranker applied only to permission-filtered candidates."""

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[ToolDescriptor, ...],
        limit: int,
    ) -> tuple[ToolDescriptor, ...]: ...


class LexicalToolDiscovery:
    """Portable deterministic fallback that does not require an embedding provider."""

    def discover(
        self,
        *,
        query: str,
        candidates: tuple[ToolDescriptor, ...],
        limit: int,
    ) -> tuple[ToolDescriptor, ...]:
        tokens = _search_tokens(query)
        ranked: list[tuple[int, str, ToolDescriptor]] = []
        for descriptor in candidates:
            haystack = " ".join(
                [
                    descriptor.name.replace(".", " "),
                    descriptor.description,
                    *descriptor.tags,
                ]
            ).lower()
            score = sum(
                3 if token in descriptor.tags else 1
                for token in tokens
                if token in haystack
            )
            if query.strip() and query.strip().lower() in haystack:
                score += 4
            if score > 0:
                ranked.append((score, descriptor.name, descriptor))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in ranked[: max(1, min(int(limit), 20))])


class LexicalToolReranker:
    """Deterministic final-stage fallback for portable deployments."""

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[ToolDescriptor, ...],
        limit: int,
    ) -> tuple[ToolDescriptor, ...]:
        ranked = LexicalToolDiscovery().discover(
            query=query,
            candidates=candidates,
            limit=max(1, min(int(limit), 5)),
        )
        selected = {item.name for item in ranked}
        ordered = list(ranked)
        ordered.extend(item for item in candidates if item.name not in selected)
        return tuple(ordered[: max(1, min(int(limit), 5))])


def resolve_tool_view(
    *,
    registry: ToolRegistry,
    allowed_names: set[str] | None,
    skill: SkillPolicy,
    mode: ToolViewMode,
    discovered_names: set[str] | None = None,
    direct_injection_max_tools: int = 10,
) -> ToolView:
    installed = {spec.name for spec in registry.list_tools()}
    allowed = installed if allowed_names is None else installed & set(allowed_names)
    eligible = {
        spec.name
        for spec in registry.list_tools()
        if spec.name in allowed
        and ToolDescriptor.from_spec(spec).exposure_mode != "internal"
        and (
            ToolDescriptor.from_spec(spec).exposure_mode != "skill_only"
            or skill.active
        )
    }
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
    direct_injection = False
    if mode == "legacy":
        exposed = allowed
    elif mode == "shadow":
        exposed = allowed
    else:
        if len(eligible) <= max(0, int(direct_injection_max_tools)):
            exposed = eligible
            direct_injection = True
        else:
            exposed = proposed & eligible
    return ToolView(
        mode=mode,
        catalog_version=registry.catalog_version(),
        allowed_names=frozenset(allowed),
        exposed_names=frozenset(exposed),
        proposed_names=frozenset(proposed),
        fail_open=fail_open,
        direct_injection=direct_injection,
        filtered_names=frozenset(allowed - exposed),
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
                    "maximum": 5,
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


def install_tool_discovery(
    registry: ToolRegistry,
    executor: ToolExecutor,
    *,
    discovery_port: ToolDiscoveryPort | None = None,
    reranker_port: ToolRerankerPort | None = None,
) -> None:
    """Install the portable discovery entrypoint without coupling it to a product pack."""

    if TOOL_DISCOVERY_NAME not in {spec.name for spec in registry.list_tools()}:
        registry.register(tool_discovery_spec())

    def discover(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        query = str(call.arguments.get("query") or "").strip()
        try:
            limit = max(1, min(int(call.arguments.get("limit") or 5), 5))
        except (TypeError, ValueError):
            limit = 5
        descriptors = discover_tools(
            registry,
            query=query,
            limit=limit,
            allowed_names=_context_allowed_names(registry, context),
            discovery_port=discovery_port,
            reranker_port=reranker_port,
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
            "retry_allowed": not names,
            "permission_filtered_count": max(
                0,
                len(registry.list_tools()) - len(_context_allowed_names(registry, context)),
            ),
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
    discovery_port: ToolDiscoveryPort | None = None,
    reranker_port: ToolRerankerPort | None = None,
    recall_limit: int = 20,
) -> tuple[ToolDescriptor, ...]:
    """Recall broadly, then rerank a bounded permission-filtered tool view."""

    candidates: list[ToolDescriptor] = []
    for spec in registry.list_tools():
        if allowed_names is not None and spec.name not in allowed_names:
            continue
        descriptor = ToolDescriptor.from_spec(spec)
        if descriptor.name == TOOL_DISCOVERY_NAME or descriptor.exposure_mode not in {
            "discoverable",
            "skill_entry",
        }:
            continue
        candidates.append(descriptor)
    port = discovery_port or LexicalToolDiscovery()
    recalled = port.discover(
        query=query,
        candidates=tuple(candidates),
        limit=max(1, min(max(int(recall_limit), int(limit)), 20)),
    )
    candidate_names = {item.name for item in candidates}
    recalled_unique: dict[str, ToolDescriptor] = {}
    for descriptor in recalled:
        if descriptor.name in candidate_names and descriptor.name not in recalled_unique:
            recalled_unique[descriptor.name] = descriptor
        if len(recalled_unique) >= 20:
            break
    reranker = reranker_port or LexicalToolReranker()
    reranked = reranker.rerank(
        query=query,
        candidates=tuple(recalled_unique.values()),
        limit=max(1, min(int(limit), 5)),
    )
    allowed_recalled = set(recalled_unique)
    final: dict[str, ToolDescriptor] = {}
    for descriptor in reranked:
        if descriptor.name in allowed_recalled and descriptor.name not in final:
            final[descriptor.name] = descriptor
        if len(final) >= max(1, min(int(limit), 5)):
            break
    return tuple(final.values())


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


__all__ = [
    "LexicalToolDiscovery",
    "LexicalToolReranker",
    "TOOL_DISCOVERY_NAME",
    "ToolDescriptor",
    "ToolDiscoveryPort",
    "ToolRerankerPort",
    "ToolView",
    "discover_tools",
    "install_tool_discovery",
    "resolve_tool_view",
    "tool_discovery_spec",
]
