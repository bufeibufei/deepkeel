from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


ApiStability = Literal["stable", "advanced", "experimental"]


@dataclass(frozen=True, slots=True)
class PublicApiLayer:
    name: str
    module: str
    stability: ApiStability
    audience: str
    symbols: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "module": self.module,
            "stability": self.stability,
            "audience": self.audience,
            "symbols": list(self.symbols),
        }


PUBLIC_API_LAYER_POLICY: Mapping[str, tuple[ApiStability, str]] = {
    "runtime": ("stable", "host runtime consumers"),
    "extension": ("stable", "capability package authors"),
    "memory": ("stable", "host memory adapters and capability packages"),
    "adapter": ("advanced", "host infrastructure integrators"),
    "mcp": ("advanced", "MCP transport and ToolProvider integrators"),
    "orchestration": ("experimental", "bounded SubAgent workflow authors"),
}


def build_public_api_manifest(
    api_by_layer: Mapping[str, Sequence[str]],
) -> tuple[PublicApiLayer, ...]:
    unknown = sorted(set(api_by_layer) - set(PUBLIC_API_LAYER_POLICY))
    missing = sorted(set(PUBLIC_API_LAYER_POLICY) - set(api_by_layer))
    if unknown or missing:
        raise ValueError(
            f"public API layer mismatch: unknown={unknown!r}, missing={missing!r}"
        )

    owners: dict[str, str] = {}
    layers: list[PublicApiLayer] = []
    for name, symbols in api_by_layer.items():
        duplicates = sorted(symbol for symbol in symbols if symbol in owners)
        if duplicates:
            detail = ", ".join(
                f"{symbol} ({owners[symbol]} and {name})" for symbol in duplicates
            )
            raise ValueError(f"public API symbols must have one canonical layer: {detail}")
        for symbol in symbols:
            owners[symbol] = name
        stability, audience = PUBLIC_API_LAYER_POLICY[name]
        layers.append(
            PublicApiLayer(
                name=name,
                module=f"deepkeel.{name}_sdk",
                stability=stability,
                audience=audience,
                symbols=tuple(symbols),
            )
        )
    return tuple(layers)


__all__ = [
    "ApiStability",
    "PUBLIC_API_LAYER_POLICY",
    "PublicApiLayer",
    "build_public_api_manifest",
]
