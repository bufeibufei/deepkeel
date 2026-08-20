from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

from deepkeel.mcp.protocol import McpProtocolError


_HEADER_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(frozen=True, slots=True)
class McpHeaderBinding:
    name: str
    path: tuple[str, ...]
    value_type: str


def tool_header_bindings(input_schema: dict[str, Any]) -> list[McpHeaderBinding]:
    """Validate and collect statically reachable ``x-mcp-header`` annotations."""

    bindings: list[McpHeaderBinding] = []
    seen_names: set[str] = set()
    _walk_reachable(input_schema, (), bindings, seen_names)
    return bindings


def tool_parameter_headers(
    input_schema: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, str]:
    headers: dict[str, str] = {}
    for binding in tool_header_bindings(input_schema):
        found, value = _path_value(arguments, binding.path)
        if not found or value is None:
            continue
        text = _primitive_text(value, binding.value_type, binding.path)
        headers[f"Mcp-Param-{binding.name}"] = encode_mcp_header_value(text)
    return headers


def encode_mcp_header_value(value: str) -> str:
    """Encode an MCP mirrored value using the protocol's Base64 sentinel."""

    plain_ascii = bool(value) and all(
        character == "\t" or 0x20 <= ord(character) <= 0x7E
        for character in value
    )
    sentinel = value.startswith("=?base64?") and value.endswith("?=")
    if plain_ascii and value == value.strip(" \t") and not sentinel:
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"=?base64?{encoded}?="


def _walk_reachable(
    node: Any,
    path: tuple[str, ...],
    bindings: list[McpHeaderBinding],
    seen_names: set[str],
) -> None:
    if not isinstance(node, dict):
        return
    if "x-mcp-header" in node:
        _append_binding(node, path, bindings, seen_names)
    properties = node.get("properties")
    if isinstance(properties, dict):
        for property_name, child in properties.items():
            _walk_reachable(
                child,
                (*path, str(property_name)),
                bindings,
                seen_names,
            )
    for key, value in node.items():
        if key in {"x-mcp-header", "properties"}:
            continue
        _reject_unreachable_annotations(value)


def _reject_unreachable_annotations(value: Any) -> None:
    if isinstance(value, dict):
        if "x-mcp-header" in value:
            raise McpProtocolError(
                "x-mcp-header must be reachable only through properties"
            )
        for child in value.values():
            _reject_unreachable_annotations(child)
    elif isinstance(value, list):
        for child in value:
            _reject_unreachable_annotations(child)


def _append_binding(
    node: dict[str, Any],
    path: tuple[str, ...],
    bindings: list[McpHeaderBinding],
    seen_names: set[str],
) -> None:
    name = str(node.get("x-mcp-header") or "")
    if not path:
        raise McpProtocolError("x-mcp-header cannot be applied to the schema root")
    if not name or not _HEADER_TOKEN.fullmatch(name):
        raise McpProtocolError("x-mcp-header must be a valid HTTP field-name token")
    normalized = name.lower()
    if normalized in seen_names:
        raise McpProtocolError("x-mcp-header names must be case-insensitively unique")
    value_type = str(node.get("type") or "")
    if value_type not in {"string", "integer", "boolean"}:
        raise McpProtocolError(
            "x-mcp-header is only valid for string, integer, or boolean properties"
        )
    seen_names.add(normalized)
    bindings.append(McpHeaderBinding(name=name, path=path, value_type=value_type))


def _path_value(arguments: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = arguments
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _primitive_text(value: Any, value_type: str, path: tuple[str, ...]) -> str:
    location = ".".join(path)
    if value_type == "string" and isinstance(value, str):
        return value
    if value_type == "boolean" and isinstance(value, bool):
        return "true" if value else "false"
    if value_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
        if -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            return str(value)
        raise McpProtocolError(
            f"x-mcp-header integer exceeds the IEEE754 safe range: {location}"
        )
    raise McpProtocolError(
        f"x-mcp-header argument does not match its primitive schema type: {location}"
    )
