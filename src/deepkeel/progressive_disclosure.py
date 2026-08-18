from __future__ import annotations

from deepkeel.capabilities import CapabilityCatalog
from deepkeel.skill_disclosure import (
    SkillDiscoveryPort,
    SkillRerankerPort,
    install_skill_discovery,
)
from deepkeel.tool_disclosure import (
    ToolDiscoveryPort,
    ToolRerankerPort,
    install_tool_discovery,
)
from deepkeel.tools import ToolExecutor
from deepkeel.tool_registry import ToolRegistry


def install_progressive_disclosure(
    catalog: CapabilityCatalog,
    registry: ToolRegistry,
    executor: ToolExecutor,
    *,
    tool_discovery: ToolDiscoveryPort | None = None,
    tool_reranker: ToolRerankerPort | None = None,
    skill_discovery: SkillDiscoveryPort | None = None,
    skill_reranker: SkillRerankerPort | None = None,
) -> None:
    install_tool_discovery(
        registry,
        executor,
        discovery_port=tool_discovery,
        reranker_port=tool_reranker,
    )
    install_skill_discovery(
        catalog,
        registry,
        executor,
        discovery_port=skill_discovery,
        reranker_port=skill_reranker,
    )


__all__ = ["install_progressive_disclosure"]
