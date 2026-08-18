from __future__ import annotations

import dataclasses
import inspect
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence, cast

from pydantic import BaseModel


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
    "a2a": ("experimental", "A2A remote specialist integrators"),
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


def build_semantic_contract(
    targets: Mapping[str, tuple[object, Sequence[str]]],
) -> dict[str, dict[str, Any]]:
    """Describe signatures and data shapes whose meaning is frozen for one API line."""

    return {
        name: _semantic_descriptor(value, members)
        for name, (value, members) in sorted(targets.items())
    }


def _semantic_descriptor(value: object, members: Sequence[str]) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "kind": "class" if inspect.isclass(value) else "callable",
    }
    if inspect.isclass(value) and issubclass(value, BaseModel):
        descriptor["model_fields"] = {
            name: {
                "annotation": _annotation_name(field.annotation),
                "alias": field.alias or name,
                "required": field.is_required(),
                "default": _stable_default(field.default),
            }
            for name, field in value.model_fields.items()
        }
    elif inspect.isclass(value) and dataclasses.is_dataclass(value):
        descriptor["dataclass_fields"] = {
            field.name: {
                "annotation": _annotation_name(field.type),
                "required": (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                ),
                "default": _stable_default(field.default),
            }
            for field in dataclasses.fields(value)
        }
    elif inspect.isclass(value) and issubclass(value, Enum):
        descriptor["enum_values"] = [member.value for member in value]
    if callable(value):
        descriptor["signature"] = _signature_descriptor(value)
    descriptor["members"] = {
        member: _signature_descriptor(getattr(value, member))
        for member in members
    }
    return descriptor


def _signature_descriptor(value: object) -> dict[str, Any]:
    try:
        signature = inspect.signature(cast(Callable[..., Any], value))
    except (TypeError, ValueError):
        return {"available": False}
    return {
        "available": True,
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "required": parameter.default is inspect.Parameter.empty,
                "default": _stable_default(parameter.default),
                "annotation": _annotation_name(parameter.annotation),
            }
            for parameter in signature.parameters.values()
        ],
        "return": _annotation_name(signature.return_annotation),
    }


def _annotation_name(value: object) -> str:
    if value is inspect.Parameter.empty or value is inspect.Signature.empty:
        return ""
    rendered = str(value).replace("typing.", "")
    # CPython 3.14 exposes extra ForwardRef implementation metadata and renders
    # Optional aliases with PEP 604 syntax. Neither changes the SDK contract.
    rendered = re.sub(
        r"ForwardRef\((['\"].*?['\"]), is_class=True\)",
        r"ForwardRef(\1)",
        rendered,
    )
    if rendered.endswith(" | None"):
        rendered = f"Optional[{rendered[:-7]}]"
    if rendered.startswith("None | "):
        rendered = f"Optional[{rendered[7:]}]"
    return rendered


def _stable_default(value: object) -> Any:
    if value is inspect.Parameter.empty or value is dataclasses.MISSING:
        return "<required>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return f"<{type(value).__name__}>"


__all__ = [
    "ApiStability",
    "PUBLIC_API_LAYER_POLICY",
    "PublicApiLayer",
    "build_public_api_manifest",
    "build_semantic_contract",
]
