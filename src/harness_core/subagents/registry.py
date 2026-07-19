from __future__ import annotations

from harness_core.subagents.contracts import SubAgentSpec


class SubAgentRegistry:
    """Registry boundary kept independent from LangGraph and business packages."""

    def __init__(self, agents: list[SubAgentSpec] | None = None) -> None:
        self._agents = {agent.id: agent for agent in agents or []}

    def register(self, agent: SubAgentSpec, *, replace: bool = False) -> None:
        if agent.id in self._agents and not replace:
            raise ValueError(f"subagent is already registered: {agent.id}")
        self._agents[agent.id] = agent

    def register_many(self, agents: list[SubAgentSpec], *, replace: bool = False) -> None:
        for agent in agents:
            self.register(agent, replace=replace)

    def get(self, agent_id: str) -> SubAgentSpec:
        return self._agents[agent_id]

    def list_agents(self) -> list[SubAgentSpec]:
        return list(self._agents.values())

    def public_list(self) -> list[dict]:
        return [
            {
                "id": agent.id,
                "version": agent.version,
                "label": agent.label,
                "description": agent.description,
                "model_role": agent.model_role,
                "capabilities": list(agent.capabilities),
                "tool_allowlist": list(agent.tool_allowlist),
                "read_only": agent.read_only,
            }
            for agent in self.list_agents()
        ]
