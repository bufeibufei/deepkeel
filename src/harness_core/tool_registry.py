from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    argument_examples: list[dict[str, Any]] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    requires_user_action: bool = False
    visible_label: str = ""
    handoff_view: str = ""
    read_only: bool = True
    async_tool: bool = False
    parallel_safe: bool = False
    required_args: list[str] = Field(default_factory=list)
    required_arg_groups: list[list[str]] = Field(default_factory=list)
    usage_policy: dict[str, Any] = Field(default_factory=dict)
    runtime_policy: dict[str, Any] = Field(default_factory=dict)
    observation_contract: dict[str, Any] = Field(default_factory=dict)
    argument_contract: dict[str, Any] = Field(default_factory=dict)
    task_kind: str = ""
    exposure_mode: Literal[
        "baseline", "discoverable", "skill_entry", "skill_only", "internal"
    ] = "baseline"
    discovery_tags: list[str] = Field(default_factory=list)

    def formal_parameters_schema(self) -> dict[str, Any]:
        return self.parameters_schema


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec] | None = None):
        self._tools = {tool.name: tool for tool in tools or []}

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def register(self, tool: ToolSpec, *, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ValueError(f"tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def snapshot(self) -> dict[str, ToolSpec]:
        return dict(self._tools)

    def restore(self, snapshot: dict[str, ToolSpec]) -> None:
        self._tools = dict(snapshot)

    def as_public_list(self) -> list[dict[str, Any]]:
        return [tool.model_dump(mode="json") for tool in self.list_tools()]

    def catalog_version(self) -> str:
        payload = [
            {
                "name": tool.name,
                "description": tool.description,
                "exposure_mode": tool.exposure_mode,
                "discovery_tags": sorted(tool.discovery_tags),
                "parameters_schema": tool.parameters_schema,
            }
            for tool in sorted(self.list_tools(), key=lambda item: item.name)
        ]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
