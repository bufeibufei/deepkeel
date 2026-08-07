from __future__ import annotations

import json
import threading
from itertools import count
from typing import Any

import httpx

from deepkeel.mcp.contracts import McpCallResult, McpRemoteTool, McpServerSpec
from deepkeel.mcp.protocol import (
    MAX_TOOL_LIST_PAGES,
    MCP_PROTOCOL_VERSION,
    SUPPORTED_MCP_PROTOCOL_VERSIONS,
    McpProtocolError,
    McpTimeoutError,
    McpTransportError,
    safe_server_info,
    structured_content_from_text,
    validate_session_id,
)
from deepkeel.type_narrowing import as_dict, as_list


class _McpSessionExpired(McpTransportError):
    pass


class StreamableHttpMcpClient:
    """Persistent MCP client for the Streamable HTTP transport."""

    def __init__(self, spec: McpServerSpec):
        if spec.transport != "streamable_http":
            raise ValueError(f"unsupported MCP transport: {spec.transport}")
        self.spec = spec
        self._state_lock = threading.RLock()
        self._request_ids = count(1)
        self._client = httpx.Client(
            headers=dict(spec.headers),
            timeout=spec.request_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )
        self._session_id = ""
        self._protocol_version = ""
        self._server_info: dict[str, Any] = {}
        self._closed = False
        self._generation = 0
        self._in_flight = 0
        self._timeout_count = 0
        self._last_transport_error = ""

    @property
    def server_id(self) -> str:
        return self.spec.id

    @property
    def generation(self) -> int:
        return self._generation

    def start(self, *, timeout_seconds: float | None = None) -> None:
        with self._state_lock:
            if self._closed:
                raise McpTransportError(f"MCP client is closed: {self.server_id}")
            if self._protocol_version:
                return
            request_id = next(self._request_ids)
            result, headers = self._post_request(
                request_id,
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": self.spec.client_name,
                        "version": self.spec.client_version,
                    },
                },
                timeout_seconds=timeout_seconds or self.spec.startup_timeout_seconds,
                include_session=False,
            )
            negotiated = str(result.get("protocolVersion") or MCP_PROTOCOL_VERSION)
            if negotiated not in SUPPORTED_MCP_PROTOCOL_VERSIONS:
                raise McpProtocolError(
                    f"unsupported MCP protocol version: {negotiated}"
                )
            self._protocol_version = negotiated
            self._session_id = validate_session_id(
                str(headers.get("mcp-session-id") or "")
            )
            self._server_info = as_dict(result.get("serverInfo"))
            self._generation += 1
            try:
                self._post_notification(
                    "notifications/initialized",
                    {},
                    timeout_seconds=timeout_seconds,
                )
            except Exception:
                self._reset_session()
                raise

    def list_tools(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> list[McpRemoteTool]:
        deadline = _deadline(timeout_seconds or self.spec.request_timeout_seconds)
        self.start(timeout_seconds=_remaining(deadline))
        tools: list[McpRemoteTool] = []
        cursor = ""
        seen_cursors: set[str] = set()
        for _page in range(MAX_TOOL_LIST_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = self._request_with_session_recovery(
                "tools/list",
                params,
                timeout_seconds=_remaining(deadline),
            )
            for item in as_list(result.get("tools")):
                if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                    continue
                tools.append(
                    McpRemoteTool(
                        name=str(item["name"]),
                        description=str(item.get("description") or ""),
                        input_schema=as_dict(item.get("inputSchema")),
                    )
                )
            cursor = str(result.get("nextCursor") or "")
            if not cursor:
                return tools
            if cursor in seen_cursors:
                raise McpProtocolError("MCP tools/list returned a repeated cursor")
            seen_cursors.add(cursor)
        raise McpProtocolError("MCP tools/list exceeded the pagination limit")

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> McpCallResult:
        timeout = timeout_seconds or self.spec.request_timeout_seconds
        self.start(timeout_seconds=timeout)
        result = self._request_with_session_recovery(
            "tools/call",
            {"name": name, "arguments": dict(arguments)},
            timeout_seconds=timeout,
        )
        content = [
            self._redact_value(item)
            for item in as_list(result.get("content"))
            if isinstance(item, dict)
        ]
        structured = (
            self._redact_value(as_dict(result.get("structuredContent")))
            if isinstance(result.get("structuredContent"), dict)
            else structured_content_from_text(content)
        )
        return McpCallResult(
            content=content,
            structured_content=structured,
            is_error=bool(result.get("isError")),
            metadata={
                "server_id": self.server_id,
                "transport": "streamable_http",
                "protocol_version": self._protocol_version,
                "server_info": safe_server_info(self._server_info),
            },
        )

    def diagnostics(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                **self.spec.public_snapshot(),
                "running": bool(self._protocol_version and not self._closed),
                "protocol_version": self._protocol_version,
                "server_info": safe_server_info(self._server_info),
                "session_active": bool(self._session_id),
                "generation": self._generation,
                "restart_count": max(0, self._generation - 1),
                "in_flight_requests": self._in_flight,
                "timeout_count": self._timeout_count,
                "last_transport_error": self._redact(self._last_transport_error),
            }

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            if self._session_id:
                try:
                    self._client.delete(
                        self.spec.url,
                        headers=self._request_headers(include_session=True),
                        timeout=self.spec.shutdown_timeout_seconds,
                    )
                except httpx.HTTPError:
                    pass
            self._reset_session()
            self._client.close()

    def _request_with_session_recovery(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        request_id = next(self._request_ids)
        try:
            result, _headers = self._post_request(
                request_id,
                method,
                params,
                timeout_seconds=timeout_seconds,
                include_session=True,
            )
            return result
        except _McpSessionExpired:
            with self._state_lock:
                self._reset_session()
            self.start(timeout_seconds=timeout_seconds)
            result, _headers = self._post_request(
                next(self._request_ids),
                method,
                params,
                timeout_seconds=timeout_seconds,
                include_session=True,
            )
            return result

    def _post_request(
        self,
        request_id: int,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        include_session: bool,
    ) -> tuple[dict[str, Any], httpx.Headers]:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        response = self._post(
            payload,
            timeout_seconds=timeout_seconds,
            include_session=include_session,
        )
        message = _response_message(response, request_id)
        if isinstance(message.get("error"), dict):
            error = message["error"]
            raise McpProtocolError(
                self._redact(
                    f"MCP {method} failed ({error.get('code')}): {error.get('message')}"
                )
            )
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError(f"MCP {method} returned an invalid result")
        return result, response.headers

    def _post_notification(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float | None,
    ) -> None:
        self._post(
            {"jsonrpc": "2.0", "method": method, "params": params},
            timeout_seconds=timeout_seconds or self.spec.request_timeout_seconds,
            include_session=True,
        )

    def _post(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
        include_session: bool,
    ) -> httpx.Response:
        with self._state_lock:
            if self._closed:
                raise McpTransportError(f"MCP client is closed: {self.server_id}")
            self._in_flight += 1
        try:
            response = self._client.post(
                self.spec.url,
                json=payload,
                headers=self._request_headers(include_session=include_session),
                timeout=max(0.001, float(timeout_seconds)),
            )
        except httpx.TimeoutException as exc:
            self._timeout_count += 1
            timeout_error = McpTimeoutError(f"MCP request timed out: {self.server_id}")
            self._last_transport_error = str(timeout_error)
            raise timeout_error from exc
        except httpx.HTTPError as exc:
            transport_error = McpTransportError(
                self._redact(f"MCP HTTP transport failed: {self.server_id}: {exc}")
            )
            self._last_transport_error = str(transport_error)
            raise transport_error from exc
        finally:
            with self._state_lock:
                self._in_flight = max(0, self._in_flight - 1)
        if response.status_code == 404 and include_session and self._session_id:
            raise _McpSessionExpired(f"MCP session expired: {self.server_id}")
        if response.status_code >= 400:
            detail = self._redact(response.text[:500])
            raise McpTransportError(
                f"MCP HTTP {response.status_code}: {self.server_id}: {detail}"
            )
        return response

    def _request_headers(self, *, include_session: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._protocol_version:
            headers["MCP-Protocol-Version"] = self._protocol_version
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _reset_session(self) -> None:
        self._session_id = ""
        self._protocol_version = ""
        self._server_info = {}

    def _redact(self, value: str) -> str:
        clean = value
        for secret in (*self.spec.headers.values(), *self.spec.environment.values()):
            if secret:
                clean = clean.replace(secret, "***")
        return clean

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._redact_value(item) for key, item in value.items()}
        return value


def _response_message(response: httpx.Response, request_id: int) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "").lower()
    if response.status_code == 202 and not response.content:
        return {}
    if "text/event-stream" in content_type:
        messages = _sse_messages(response.text)
    else:
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise McpProtocolError("MCP HTTP response is not valid JSON") from exc
        messages = payload if isinstance(payload, list) else [payload]
    for message in messages:
        if isinstance(message, dict) and message.get("id") == request_id:
            return message
    raise McpProtocolError("MCP HTTP response did not contain the requested JSON-RPC id")


def _sse_messages(body: str) -> list[Any]:
    messages: list[Any] = []
    data_lines: list[str] = []
    for line in [*body.splitlines(), ""]:
        if not line:
            if data_lines:
                try:
                    messages.append(json.loads("\n".join(data_lines)))
                except json.JSONDecodeError as exc:
                    raise McpProtocolError("MCP SSE event contains invalid JSON") from exc
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return messages


def _deadline(timeout_seconds: float) -> float:
    import time

    return time.monotonic() + max(0.001, float(timeout_seconds))


def _remaining(deadline: float) -> float:
    import time

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise McpTimeoutError("MCP request deadline exceeded")
    return remaining
