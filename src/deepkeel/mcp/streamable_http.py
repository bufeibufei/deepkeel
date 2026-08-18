from __future__ import annotations

import json
import ipaddress
import socket
import threading
import time
import warnings
from itertools import count
from typing import Any
from urllib.parse import urlsplit

import httpx

from deepkeel.mcp.contracts import McpCallResult, McpRemoteTool, McpServerSpec, McpTask
from deepkeel.mcp.header_projection import (
    encode_mcp_header_value,
    tool_header_bindings,
    tool_parameter_headers,
)
from deepkeel.mcp.protocol import (
    LEGACY_MCP_PROTOCOL_VERSION,
    LEGACY_MCP_PROTOCOL_VERSIONS,
    MAX_TOOL_LIST_PAGES,
    MCP_PROTOCOL_VERSION,
    SUPPORTED_MCP_PROTOCOL_VERSIONS,
    McpProtocolError,
    McpRemoteError,
    McpTimeoutError,
    McpTransportError,
    modern_request_metadata,
    remote_error,
    safe_server_info,
    validate_session_id,
)
from deepkeel.mcp.result_parsing import call_result, remote_tools, task_result
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
        self._protocol_era = ""
        self._server_info: dict[str, Any] = {}
        self._server_capabilities: dict[str, Any] = {}
        self._tool_cache: list[McpRemoteTool] = []
        self._tool_cache_expires_at = 0.0
        self._tool_cache_scope = "private"
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
            startup_timeout = timeout_seconds or self.spec.startup_timeout_seconds
            if self.spec.protocol_version in LEGACY_MCP_PROTOCOL_VERSIONS:
                self._start_legacy(self.spec.protocol_version, startup_timeout)
                return
            try:
                self._start_modern(startup_timeout)
            except (McpProtocolError, McpTimeoutError, McpTransportError) as exc:
                if not self._can_fallback_to_legacy(exc):
                    raise
                self._start_legacy(LEGACY_MCP_PROTOCOL_VERSION, startup_timeout)

    def _start_modern(self, timeout_seconds: float) -> None:
        result, _headers = self._post_request(
            next(self._request_ids),
            "server/discover",
            {},
            timeout_seconds=timeout_seconds,
            include_session=False,
            modern=True,
        )
        supported = {
            str(value)
            for value in as_list(result.get("supportedVersions"))
            if str(value) in SUPPORTED_MCP_PROTOCOL_VERSIONS
        }
        if MCP_PROTOCOL_VERSION not in supported:
            raise McpProtocolError(
                "MCP server does not advertise the modern protocol revision"
            )
        self._protocol_version = MCP_PROTOCOL_VERSION
        self._protocol_era = "modern"
        self._server_capabilities = as_dict(result.get("capabilities"))
        self._server_info = _server_info_from_result(result)
        self._generation += 1

    def _start_legacy(self, version: str, timeout_seconds: float) -> None:
        result, headers = self._post_request(
            next(self._request_ids),
            "initialize",
            {
                "protocolVersion": version,
                "capabilities": dict(self.spec.client_capabilities),
                "clientInfo": {
                    "name": self.spec.client_name,
                    "version": self.spec.client_version,
                },
            },
            timeout_seconds=timeout_seconds,
            include_session=False,
        )
        negotiated = str(result.get("protocolVersion") or version)
        if negotiated not in LEGACY_MCP_PROTOCOL_VERSIONS:
            raise McpProtocolError(
                f"unsupported legacy MCP protocol version: {negotiated}"
            )
        self._protocol_version = negotiated
        self._protocol_era = "legacy"
        self._session_id = validate_session_id(str(headers.get("mcp-session-id") or ""))
        self._server_info = as_dict(result.get("serverInfo"))
        self._server_capabilities = as_dict(result.get("capabilities"))
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

    def _can_fallback_to_legacy(self, exc: BaseException) -> bool:
        if self.spec.protocol_version or not self.spec.allow_legacy_fallback:
            return False
        if isinstance(exc, McpRemoteError) and exc.is_modern_version_error:
            supported = {
                str(value) for value in as_list(exc.data.get("supported"))
            }
            return bool(supported & LEGACY_MCP_PROTOCOL_VERSIONS)
        return True

    def list_tools(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> list[McpRemoteTool]:
        deadline = _deadline(timeout_seconds or self.spec.request_timeout_seconds)
        self.start(timeout_seconds=_remaining(deadline))
        if self._tool_cache and time.monotonic() < self._tool_cache_expires_at:
            return list(self._tool_cache)
        tools: list[McpRemoteTool] = []
        ttl_seconds = self.spec.tool_cache_ttl_seconds
        cursor = ""
        seen_cursors: set[str] = set()
        for _page in range(MAX_TOOL_LIST_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = self._request_with_session_recovery(
                "tools/list",
                params,
                timeout_seconds=_remaining(deadline),
            )
            page_tools = remote_tools(result)
            if self._protocol_era == "modern":
                page_tools = self._valid_http_tools(page_tools)
            tools.extend(page_tools)
            ttl_seconds = min(ttl_seconds, _cache_ttl_seconds(result, ttl_seconds))
            self._tool_cache_scope = str(result.get("cacheScope") or "private")
            cursor = str(result.get("nextCursor") or "")
            if not cursor:
                self._tool_cache = sorted(tools, key=lambda item: item.name)
                self._tool_cache_expires_at = time.monotonic() + ttl_seconds
                return list(self._tool_cache)
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
        deadline = _deadline(timeout)
        self.start(timeout_seconds=_remaining(deadline))
        self._ensure_modern_tool_schema(name, timeout_seconds=_remaining(deadline))
        return self._call_tool(name, arguments, timeout_seconds=_remaining(deadline))

    def continue_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        input_responses: dict[str, Any],
        request_state: Any,
        timeout_seconds: float | None = None,
    ) -> McpCallResult:
        timeout = timeout_seconds or self.spec.request_timeout_seconds
        deadline = _deadline(timeout)
        self.start(timeout_seconds=_remaining(deadline))
        self._ensure_modern_tool_schema(name, timeout_seconds=_remaining(deadline))
        return self._call_tool(
            name,
            arguments,
            input_responses=input_responses,
            request_state=request_state,
            timeout_seconds=_remaining(deadline),
        )

    def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        input_responses: dict[str, Any] | None = None,
        request_state: Any = None,
        timeout_seconds: float,
    ) -> McpCallResult:
        params: dict[str, Any] = {"name": name, "arguments": dict(arguments)}
        if input_responses is not None:
            params["inputResponses"] = dict(input_responses)
            params["requestState"] = request_state
        result = self._request_with_session_recovery(
            "tools/call",
            params,
            timeout_seconds=timeout_seconds,
        )
        return call_result(
            result,
            redact=self._redact_value,
            metadata={
                "server_id": self.server_id,
                "transport": "streamable_http",
                "protocol_version": self._protocol_version,
                "protocol_era": self._protocol_era,
                "server_info": safe_server_info(self._server_info),
            },
        )

    def get_task(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> McpTask:
        task = self._task_request("tasks/get", {"taskId": task_id}, timeout_seconds)
        if task is None:
            raise McpProtocolError("MCP tasks/get returned an empty result")
        return task

    def update_task(
        self,
        task_id: str,
        input_responses: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> McpTask | None:
        return self._task_request(
            "tasks/update",
            {"taskId": task_id, "inputResponses": dict(input_responses)},
            timeout_seconds,
            allow_empty=True,
        )

    def cancel_task(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> McpTask | None:
        return self._task_request(
            "tasks/cancel",
            {"taskId": task_id},
            timeout_seconds,
            allow_empty=True,
        )

    def _task_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None,
        *,
        allow_empty: bool = False,
    ) -> McpTask | None:
        self.start(timeout_seconds=timeout_seconds)
        result = self._request_with_session_recovery(
            method,
            params,
            timeout_seconds=timeout_seconds or self.spec.request_timeout_seconds,
        )
        if allow_empty and not result:
            return None
        return task_result(result)

    def diagnostics(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                **self.spec.public_snapshot(),
                "running": bool(self._protocol_version and not self._closed),
                "protocol_version": self._protocol_version,
                "protocol_era": self._protocol_era,
                "server_info": safe_server_info(self._server_info),
                "server_capabilities": dict(self._server_capabilities),
                "session_active": bool(self._session_id),
                "tool_cache_scope": self._tool_cache_scope,
                "tool_cache_remaining_seconds": round(
                    max(0.0, self._tool_cache_expires_at - time.monotonic()), 3
                ),
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
            if self._protocol_era == "legacy" and self._session_id:
                try:
                    self._validate_egress_target()
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
        include_session = self._protocol_era == "legacy"
        try:
            result, _headers = self._post_request(
                request_id,
                method,
                params,
                timeout_seconds=timeout_seconds,
                include_session=include_session,
            )
            return result
        except _McpSessionExpired:
            if not include_session:
                raise
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
        modern: bool | None = None,
    ) -> tuple[dict[str, Any], httpx.Headers]:
        request_params = dict(params)
        if modern is True or (modern is None and self._protocol_era == "modern"):
            request_params = _modern_params(self.spec, request_params)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": request_params,
        }
        response = self._post(
            payload,
            timeout_seconds=timeout_seconds,
            include_session=include_session,
        )
        message = _response_message(response, request_id)
        if isinstance(message.get("error"), dict):
            error = remote_error(method, message["error"])
            error.args = (self._redact(str(error)),)
            raise error
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError(f"MCP {method} returned an invalid result")
        server_info = _server_info_from_result(result)
        if server_info:
            self._server_info = server_info
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
        encoded_size = len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if encoded_size > self.spec.max_request_bytes:
            raise McpTransportError(
                f"MCP request exceeds max_request_bytes: {self.server_id}"
            )
        with self._state_lock:
            if self._closed:
                raise McpTransportError(f"MCP client is closed: {self.server_id}")
            self._in_flight += 1
        try:
            self._validate_egress_target()
            with self._client.stream(
                "POST",
                self.spec.url,
                json=payload,
                headers=self._request_headers(
                    include_session=include_session,
                    payload=payload,
                ),
                timeout=max(0.001, float(timeout_seconds)),
            ) as streamed:
                content = bytearray()
                for chunk in streamed.iter_bytes():
                    content.extend(chunk)
                    if len(content) > self.spec.max_response_bytes:
                        raise McpTransportError(
                            f"MCP response exceeds max_response_bytes: {self.server_id}"
                        )
                response = httpx.Response(
                    streamed.status_code,
                    headers=streamed.headers,
                    content=bytes(content),
                    request=streamed.request,
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

    def _validate_egress_target(self) -> None:
        parsed = urlsplit(self.spec.url)
        hostname = str(parsed.hostname or "").strip().lower()
        if not hostname:
            raise McpTransportError(f"MCP endpoint has no hostname: {self.server_id}")
        if self.spec.allowed_hosts and hostname not in {
            host.strip().lower() for host in self.spec.allowed_hosts if host.strip()
        }:
            raise McpTransportError(f"MCP endpoint host is not allowed: {self.server_id}")
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise McpTransportError(
                f"MCP endpoint DNS resolution failed: {self.server_id}"
            ) from exc
        if self.spec.allow_private_network:
            return
        for value in addresses:
            address = ipaddress.ip_address(value)
            if not address.is_global:
                raise McpTransportError(
                    f"MCP endpoint resolved to a non-public address: {self.server_id}"
                )

    def _request_headers(
        self,
        *,
        include_session: bool,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        method = str((payload or {}).get("method") or "")
        params = as_dict((payload or {}).get("params"))
        metadata = as_dict(params.get("_meta"))
        request_version = str(
            metadata.get("io.modelcontextprotocol/protocolVersion")
            or self._protocol_version
            or ""
        )
        if request_version:
            headers["MCP-Protocol-Version"] = request_version
        if request_version == MCP_PROTOCOL_VERSION and method:
            headers["Mcp-Method"] = method
            if method in {"tools/call", "resources/read", "prompts/get"}:
                name = str(params.get("name") or params.get("uri") or "").strip()
                if name:
                    headers["Mcp-Name"] = encode_mcp_header_value(name)
            if method == "tools/call":
                arguments = as_dict(params.get("arguments"))
                tool = next(
                    (item for item in self._tool_cache if item.name == params.get("name")),
                    None,
                )
                if tool is not None:
                    headers.update(
                        tool_parameter_headers(tool.input_schema, arguments)
                    )
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _valid_http_tools(
        self,
        tools: list[McpRemoteTool],
    ) -> list[McpRemoteTool]:
        valid: list[McpRemoteTool] = []
        for tool in tools:
            try:
                tool_header_bindings(tool.input_schema)
            except McpProtocolError as exc:
                warnings.warn(
                    f"Ignoring invalid MCP tool {tool.name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            valid.append(tool)
        return valid

    def _ensure_modern_tool_schema(
        self,
        name: str,
        *,
        timeout_seconds: float,
    ) -> None:
        if self._protocol_era != "modern":
            return
        if not any(tool.name == name for tool in self._tool_cache):
            self.list_tools(timeout_seconds=timeout_seconds)
        if not any(tool.name == name for tool in self._tool_cache):
            raise McpProtocolError(
                f"MCP tool {name} is not exposed by {self.server_id}"
            )

    def _reset_session(self) -> None:
        self._session_id = ""
        self._protocol_version = ""
        self._protocol_era = ""
        self._server_info = {}
        self._server_capabilities = {}
        self._tool_cache = []
        self._tool_cache_expires_at = 0.0

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
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise McpTimeoutError("MCP request deadline exceeded")
    return remaining


def _modern_params(spec: McpServerSpec, params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(params)
    existing_meta = as_dict(merged.get("_meta"))
    merged["_meta"] = {
        **modern_request_metadata(
            client_name=spec.client_name,
            client_version=spec.client_version,
            client_capabilities=spec.client_capabilities,
        ),
        **existing_meta,
    }
    return merged


def _server_info_from_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = as_dict(result.get("_meta"))
    return as_dict(metadata.get("io.modelcontextprotocol/serverInfo")) or as_dict(
        result.get("serverInfo")
    )


def _cache_ttl_seconds(result: dict[str, Any], fallback: float) -> float:
    raw_ttl = result.get("ttlMs")
    try:
        return max(0.0, float(str(raw_ttl)) / 1000.0)
    except (TypeError, ValueError):
        return fallback
