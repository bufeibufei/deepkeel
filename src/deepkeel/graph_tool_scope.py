from __future__ import annotations

from copy import deepcopy
from typing import Any

from deepkeel.entrypoints import _restrict_subagent_ids
from deepkeel.skills import DelegationPolicy
from deepkeel.tool_registry import ToolRegistry
from deepkeel.type_narrowing import as_dict


def _skill_tool_parameter_overrides(
    state: dict[str, Any],
    registry: ToolRegistry,
) -> dict[str, dict[str, Any]]:
    policy = DelegationPolicy.from_snapshot(as_dict(state.get("skill_activation")))
    if not policy.enabled:
        return {}
    try:
        spec = registry.get("agent.delegate")
    except KeyError:
        return {}
    formal_schema = getattr(spec, "formal_parameters_schema", None)
    schema = formal_schema() if callable(formal_schema) else {}
    if (
        not isinstance(schema, dict)
        or schema.get("type") != "object"
        or not isinstance(schema.get("properties"), dict)
    ):
        return {}
    schema = deepcopy(schema)
    properties = as_dict(schema.get("properties"))
    concurrency = properties.get("max_concurrency")
    if isinstance(concurrency, dict):
        concurrency["maximum"] = policy.max_concurrency
        concurrency["default"] = min(
            int(concurrency.get("default") or policy.max_concurrency),
            policy.max_concurrency,
        )
    tasks = properties.get("tasks")
    if isinstance(tasks, dict):
        tasks["maxItems"] = policy.max_tasks
        agent_id = as_dict(as_dict(tasks.get("items")).get("properties")).get(
            "agent_id"
        )
        if isinstance(agent_id, dict):
            view = as_dict(as_dict(state.get("metadata")).get("capability_view"))
            allowed_agents = _restrict_subagent_ids(view, policy.allowed_agents)
            if bool(view.get("restricted")) or allowed_agents:
                agent_id["enum"] = sorted(allowed_agents)
    return {"agent.delegate": schema}


__all__ = ["_skill_tool_parameter_overrides"]
