from __future__ import annotations

import json
from typing import Any


MCP_PROTOCOL_VERSION = "2026-07-28"
LEGACY_MCP_PROTOCOL_VERSION = "2025-11-25"
LEGACY_MCP_PROTOCOL_VERSIONS = frozenset(
    {"2024-11-05", "2025-03-26", "2025-06-18", LEGACY_MCP_PROTOCOL_VERSION}
)
SUPPORTED_MCP_PROTOCOL_VERSIONS = frozenset(
    {*LEGACY_MCP_PROTOCOL_VERSIONS, MCP_PROTOCOL_VERSION}
)
MAX_TOOL_LIST_PAGES = 20
UNSUPPORTED_PROTOCOL_VERSION_ERROR = -32022
MCP_TASKS_EXTENSION = "io.modelcontextprotocol/tasks"


class McpTransportError(RuntimeError):
    pass


class McpTimeoutError(McpTransportError):
    pass


class McpProtocolError(RuntimeError):
    pass


class McpRemoteError(McpProtocolError):
    """Typed JSON-RPC error used for protocol-era negotiation."""

    def __init__(
        self,
        method: str,
        *,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.code = code
        self.data = dict(data or {})
        super().__init__(f"MCP {method} failed ({code}): {message}")

    @property
    def is_modern_version_error(self) -> bool:
        return self.code == UNSUPPORTED_PROTOCOL_VERSION_ERROR


def default_client_capabilities() -> dict[str, Any]:
    """Advertise only modern extensions that DeepKeel can project durably."""

    return {"extensions": {MCP_TASKS_EXTENSION: {}}}


def modern_request_metadata(
    *,
    client_name: str,
    client_version: str,
    client_capabilities: dict[str, Any],
    protocol_version: str = MCP_PROTOCOL_VERSION,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "io.modelcontextprotocol/protocolVersion": protocol_version,
        "io.modelcontextprotocol/clientInfo": {
            "name": client_name,
            "version": client_version,
        },
        "io.modelcontextprotocol/clientCapabilities": dict(client_capabilities),
    }
    metadata.update(trace_context_metadata())
    return metadata


def trace_context_metadata() -> dict[str, str]:
    """Inject W3C trace context without requiring OpenTelemetry at runtime."""

    carrier: dict[str, str] = {}
    try:
        from opentelemetry.propagate import inject
    except ImportError:
        return carrier
    try:
        inject(carrier)
    except Exception:
        return {}
    return {
        key: value
        for key, value in carrier.items()
        if key in {"traceparent", "tracestate", "baggage"} and value
    }


def remote_error(method: str, value: dict[str, Any]) -> McpRemoteError:
    raw_code = value.get("code")
    try:
        code = int(str(raw_code))
    except (TypeError, ValueError):
        code = -32000
    data = value.get("data")
    return McpRemoteError(
        method,
        code=code,
        message=str(value.get("message") or "remote error"),
        data=data if isinstance(data, dict) else {},
    )


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
