from __future__ import annotations

import json
from typing import Any


MCP_PROTOCOL_VERSION = "2025-03-26"
SUPPORTED_MCP_PROTOCOL_VERSIONS = frozenset(
    {"2024-11-05", MCP_PROTOCOL_VERSION}
)
MAX_TOOL_LIST_PAGES = 20


class McpTransportError(RuntimeError):
    pass


class McpTimeoutError(McpTransportError):
    pass


class McpProtocolError(RuntimeError):
    pass


def structured_content_from_text(content: list[dict[str, Any]]) -> dict[str, Any]:
    for item in content:
        if item.get("type") != "text" or not isinstance(item.get("text"), str):
            continue
        try:
            value = json.loads(item["text"])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def safe_server_info(value: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(value.get("name") or ""),
        "version": str(value.get("version") or ""),
    }


def validate_session_id(value: str) -> str:
    """Accept only the visible ASCII session identifiers required by MCP."""
    session_id = str(value or "")
    if session_id and any(ord(char) < 0x21 or ord(char) > 0x7E for char in session_id):
        raise McpProtocolError("MCP session id contains invalid characters")
    return session_id
