from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from deepkeel.tool_registry import ToolRegistry
from deepkeel.tools import ToolExecutor


@dataclass(frozen=True, slots=True)
class ToolProviderSpec:
    """Protocol-neutral declaration for a source of runtime tools."""

    provider_id: str
    tool_names: tuple[str, ...] = ()
    provider_kind: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider_id = self.provider_id.strip()
        if not provider_id:
            raise ValueError("tool provider id must not be blank")
        tool_names = tuple(
            dict.fromkeys(name.strip() for name in self.tool_names if name.strip())
        )
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "tool_names", tool_names)
        object.__setattr__(self, "provider_kind", self.provider_kind.strip() or "custom")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "tool_names": list(self.tool_names),
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ToolProvider(Protocol):
    """Adapter boundary for MCP, OpenAPI, RPC, or in-process tool sources."""

    @property
    def spec(self) -> ToolProviderSpec: ...

    def install(self, *, registry: ToolRegistry, executor: ToolExecutor) -> None: ...

    def diagnostics(self) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


def verify_tool_provider(provider: object) -> ToolProviderSpec:
    spec = getattr(provider, "spec", None)
    if not isinstance(spec, ToolProviderSpec):
        raise TypeError("tool provider must declare a ToolProviderSpec as spec")
    for method_name in ("install", "diagnostics", "close"):
        if not callable(getattr(provider, method_name, None)):
            raise TypeError(f"tool provider must implement {method_name}()")
    return spec
