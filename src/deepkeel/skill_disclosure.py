from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Protocol

from deepkeel.capabilities import CapabilityCatalog
from deepkeel.contracts import Observation, ToolCall, ToolResult
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor


SKILL_DISCOVERY_NAME = "runtime.discover_skills"


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    skill_id: str
    label: str
    description: str
    kind: str
    invocation_modes: tuple[str, ...]
    entry_tools: tuple[str, ...]
    tags: tuple[str, ...] = ()
    package_id: str = ""


class SkillDiscoveryPort(Protocol):
    """Host-replaceable broad recall for a permission-filtered Skill catalog."""

    def discover(
        self,
        *,
        query: str,
        candidates: tuple[SkillDescriptor, ...],
        limit: int,
    ) -> tuple[SkillDescriptor, ...]: ...


class SkillRerankerPort(Protocol):
    """Optional second-stage Skill ranker; it cannot introduce new candidates."""

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[SkillDescriptor, ...],
        limit: int,
    ) -> tuple[SkillDescriptor, ...]: ...


class LexicalSkillDiscovery:
    """Deterministic broad-recall fallback with Chinese bigram matching."""

    def discover(
        self,
        *,
        query: str,
        candidates: tuple[SkillDescriptor, ...],
        limit: int,
    ) -> tuple[SkillDescriptor, ...]:
        tokens = _search_tokens(query)
        ranked: list[tuple[int, str, SkillDescriptor]] = []
        for descriptor in candidates:
            haystack = " ".join(
                (
                    descriptor.skill_id.replace(".", " "),
                    descriptor.label,
                    descriptor.description,
                    descriptor.kind,
                    descriptor.package_id,
                    *descriptor.tags,
                )
            ).lower()
            score = sum(3 if token in descriptor.tags else 1 for token in tokens if token in haystack)
            normalized_query = query.strip().lower()
            if normalized_query and normalized_query in haystack:
                score += 4
            if score > 0:
                ranked.append((score, descriptor.skill_id, descriptor))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in ranked[: max(1, min(int(limit), 12))])


class LexicalSkillReranker:
    """Portable final ranker that preserves semantic recall ordering on lexical ties."""

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[SkillDescriptor, ...],
        limit: int,
    ) -> tuple[SkillDescriptor, ...]:
        bounded = max(1, min(int(limit), 3))
        ranked = LexicalSkillDiscovery().discover(
            query=query,
            candidates=candidates,
            limit=bounded,
        )
        selected = {item.skill_id for item in ranked}
        ordered = list(ranked)
        ordered.extend(item for item in candidates if item.skill_id not in selected)
        return tuple(ordered[:bounded])


def skill_discovery_spec() -> ToolSpec:
    return ToolSpec(
        name=SKILL_DISCOVERY_NAME,
        description=(
            "Search the available Skill catalog when no visible capability clearly matches "
            "the request. Describe the user outcome needed, not a Skill identifier."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User outcome or domain capability needed for the task.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 3,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        required_args=["query"],
        read_only=True,
        parallel_safe=True,
        visible_label="Discover skills",
        exposure_mode="baseline",
        discovery_tags=["skill", "catalog", "capability", "discovery"],
        runtime_policy={"internal_runtime_tool": True},
    )


def install_skill_discovery(
    catalog: CapabilityCatalog,
    registry: ToolRegistry,
    executor: ToolExecutor,
    *,
    discovery_port: SkillDiscoveryPort | None = None,
    reranker_port: SkillRerankerPort | None = None,
) -> None:
    """Install model-facing Skill discovery without activating or authorizing a Skill."""

    if SKILL_DISCOVERY_NAME not in {spec.name for spec in registry.list_tools()}:
        registry.register(skill_discovery_spec())

    def discover(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        query = str(call.arguments.get("query") or "").strip()
        try:
            limit = max(1, min(int(call.arguments.get("limit") or 3), 3))
        except (TypeError, ValueError):
            limit = 3
        allowed_skill_ids, allowed_tool_names = _context_capability_scope(context)
        descriptors = discover_skills(
            catalog,
            registry,
            query=query,
            limit=limit,
            allowed_skill_ids=allowed_skill_ids,
            allowed_tool_names=allowed_tool_names,
            discovery_port=discovery_port,
            reranker_port=reranker_port,
        )
        skill_ids = [item.skill_id for item in descriptors]
        entry_tools = list(dict.fromkeys(tool for item in descriptors for tool in item.entry_tools))
        summary = (
            f"Discovered {len(skill_ids)} relevant Skill(s)."
            if skill_ids
            else "No additional relevant Skills were found."
        )
        data = {
            "query": query,
            "discovered_skill_ids": skill_ids,
            "discovered_tool_names": entry_tools,
            "skills": [
                {
                    "skill_id": item.skill_id,
                    "label": item.label,
                    "description": item.description,
                    "kind": item.kind,
                    "entry_tools": list(item.entry_tools),
                    "tags": list(item.tags),
                }
                for item in descriptors
            ],
            "retry_allowed": not skill_ids,
        }
        return ToolResult(
            call=call,
            status="succeeded",
            summary=summary,
            data=data,
            observation=Observation(
                id=f"{call.id}:skill-discovery",
                run_id=context.run_id,
                tool_call_id=call.id,
                source=SKILL_DISCOVERY_NAME,
                status="succeeded",
                summary=summary,
                data=data,
                metadata={"visible": False},
            ),
            metadata={"visible": False, "runtime_internal": True},
        )

    executor.register(SKILL_DISCOVERY_NAME, discover)


def discover_skills(
    catalog: CapabilityCatalog,
    registry: ToolRegistry,
    *,
    query: str,
    limit: int = 3,
    allowed_skill_ids: set[str] | None = None,
    allowed_tool_names: set[str] | None = None,
    discovery_port: SkillDiscoveryPort | None = None,
    reranker_port: SkillRerankerPort | None = None,
    recall_limit: int = 12,
) -> tuple[SkillDescriptor, ...]:
    """Recall and rerank model-invocable Skills without widening capability scope."""

    candidates: list[SkillDescriptor] = []
    for skill_id, spec in sorted(catalog.skills.items()):
        if allowed_skill_ids is not None and skill_id not in allowed_skill_ids:
            continue
        descriptor = _descriptor_from_spec(skill_id, spec, registry, allowed_tool_names)
        if "model" not in descriptor.invocation_modes or not descriptor.entry_tools:
            continue
        candidates.append(descriptor)
    recall = discovery_port or LexicalSkillDiscovery()
    recalled = recall.discover(
        query=query,
        candidates=tuple(candidates),
        limit=max(1, min(max(int(recall_limit), int(limit)), 12)),
    )
    candidate_ids = {item.skill_id for item in candidates}
    recalled_unique: dict[str, SkillDescriptor] = {}
    for descriptor in recalled:
        if descriptor.skill_id in candidate_ids and descriptor.skill_id not in recalled_unique:
            recalled_unique[descriptor.skill_id] = descriptor
        if len(recalled_unique) >= 12:
            break
    reranker = reranker_port or LexicalSkillReranker()
    reranked = reranker.rerank(
        query=query,
        candidates=tuple(recalled_unique.values()),
        limit=max(1, min(int(limit), 3)),
    )
    allowed_recalled = set(recalled_unique)
    final: dict[str, SkillDescriptor] = {}
    for descriptor in reranked:
        if descriptor.skill_id in allowed_recalled and descriptor.skill_id not in final:
            final[descriptor.skill_id] = descriptor
        if len(final) >= max(1, min(int(limit), 3)):
            break
    return tuple(final.values())


def _descriptor_from_spec(
    skill_id: str,
    spec: object,
    registry: ToolRegistry,
    allowed_tool_names: set[str] | None,
) -> SkillDescriptor:
    raw = _mapping(spec)
    package = _mapping(raw.get("package"))
    allowed_tools = _strings(raw.get("allowed_tools"))
    declared_entries = (
        _strings(package.get("entry_tools"))
        or _strings(raw.get("entry_tools"))
        or _strings(raw.get("entry_tool"))
    )
    installed = {item.name: item for item in registry.list_tools()}
    inferred_entries = tuple(
        name
        for name in allowed_tools
        if name in installed and installed[name].exposure_mode == "skill_entry"
    )
    entries = declared_entries or inferred_entries
    entries = tuple(
        name
        for name in entries
        if name in installed and (allowed_tool_names is None or name in allowed_tool_names)
    )
    modes = _strings(raw.get("invocation_modes")) or ("composer", "model")
    tags = tuple(
        dict.fromkeys(
            (
                *_strings(raw.get("discovery_tags")),
                *_strings(raw.get("tags")),
                str(raw.get("kind") or "prompt"),
                str(package.get("capability_pack") or ""),
            )
        )
    )
    return SkillDescriptor(
        skill_id=skill_id,
        label=str(raw.get("label") or skill_id),
        description=str(raw.get("description") or raw.get("context_hint") or ""),
        kind=str(raw.get("kind") or "prompt"),
        invocation_modes=modes,
        entry_tools=entries,
        tags=tuple(item for item in tags if item),
        package_id=str(package.get("package_id") or ""),
    )


def _context_capability_scope(
    context: ToolExecutionContext,
) -> tuple[set[str] | None, set[str] | None]:
    raw = context.metadata.get("capability_view")
    view = raw if isinstance(raw, Mapping) else {}
    if not bool(view.get("restricted")):
        return None, None
    return (
        {str(item) for item in view.get("skill_ids", ()) if str(item).strip()},
        {str(item) for item in view.get("tool_names", ()) if str(item).strip()},
    )


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        raw = dump(mode="json")
        return dict(raw) if isinstance(raw, Mapping) else {}
    return {
        name: getattr(value, name)
        for name in (
            "label",
            "description",
            "kind",
            "invocation_modes",
            "allowed_tools",
            "entry_tools",
            "entry_tool",
            "discovery_tags",
            "tags",
            "context_hint",
            "package",
        )
        if hasattr(value, name)
    }


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _search_tokens(value: str) -> set[str]:
    normalized = str(value or "").strip().lower()
    tokens = {token for token in re.split(r"[^\w\u4e00-\u9fff]+", normalized) if token}
    compact = "".join(char for char in normalized if "\u4e00" <= char <= "\u9fff")
    tokens.update(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
    return tokens


__all__ = [
    "LexicalSkillDiscovery",
    "LexicalSkillReranker",
    "SKILL_DISCOVERY_NAME",
    "SkillDescriptor",
    "SkillDiscoveryPort",
    "SkillRerankerPort",
    "discover_skills",
    "install_skill_discovery",
    "skill_discovery_spec",
]
