from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness_core.contracts import Observation, ToolCall, ToolResult
from harness_core.deadlines import deadline_with_timeout
from harness_core.governance import DenySecretProvider, GovernanceScope, SecretProvider, SecretRequest
from harness_core.mcp.contracts import (
    McpCallResult,
    McpClient,
    McpServerSpec,
)
from harness_core.mcp.protocol import (
    McpProtocolError,
    McpTimeoutError,
    McpTransportError,
)
from harness_core.mcp.stdio import (
    StdioMcpClient,
)
from harness_core.mcp.streamable_http import StreamableHttpMcpClient
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tools import ToolExecutionContext, ToolExecutor


ArgumentMapper = Callable[[dict[str, Any]], dict[str, Any]]
ResultNormalizer = Callable[[McpCallResult, dict[str, Any]], "McpNormalizedResult"]
McpClientFactory = Callable[[McpServerSpec], McpClient]


@dataclass(slots=True)
class McpNormalizedResult:
    data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str = ""
    retryable: bool = False


@dataclass(slots=True)
class McpToolBinding:
    server_id: str
    remote_name: str
    local_spec: ToolSpec
    map_arguments: ArgumentMapper = field(default=lambda arguments: dict(arguments))
    normalize_result: ResultNormalizer = field(default=lambda result, _: _default_normalize(result))
    trusted_content: bool = False


class McpClientPool:
    def __init__(
        self,
        specs: list[McpServerSpec] | None = None,
        *,
        client_factory: McpClientFactory | None = None,
        secret_provider: SecretProvider | None = None,
        governance_scope: GovernanceScope | None = None,
    ) -> None:
        self._specs = {spec.id: spec for spec in specs or []}
        self._client_factory = client_factory or _default_client_factory
        self._secret_provider = secret_provider or DenySecretProvider()
        self._governance_scope = governance_scope or GovernanceScope()
        self._clients: dict[str, McpClient] = {}
        self._lock = threading.Lock()
        self._closed = False

    def get(self, server_id: str) -> McpClient:
        with self._lock:
            if self._closed:
                raise McpTransportError("MCP client pool is closed")
            if server_id not in self._specs:
                raise KeyError(f"unknown MCP server: {server_id}")
            client = self._clients.get(server_id)
            if client is None:
                client = self._client_factory(self._resolved_spec(self._specs[server_id]))
                self._clients[server_id] = client
            return client

    def _resolved_spec(self, spec: McpServerSpec) -> McpServerSpec:
        scope = self._governance_scope
        missing_scopes = set(spec.required_scopes) - set(scope.scopes)
        if missing_scopes:
            raise PermissionError(
                f"MCP server {spec.id} requires scopes: {', '.join(sorted(missing_scopes))}"
            )
        if not spec.secret_environment and not spec.secret_headers:
            return spec
        resolved_environment = dict(spec.environment)
        for environment_key, secret_name in spec.secret_environment.items():
            resolved_environment[environment_key] = self._secret_provider.resolve(
                SecretRequest(
                    name=secret_name,
                    tenant_id=scope.tenant_id,
                    user_id=scope.user_id,
                    resource_type="mcp_server",
                    resource_id=spec.id,
                    required_scopes=tuple(spec.required_scopes),
                )
            )
        resolved_headers = dict(spec.headers)
        for header_name, secret_name in spec.secret_headers.items():
            resolved_headers[header_name] = self._secret_provider.resolve(
                SecretRequest(
                    name=secret_name,
                    tenant_id=scope.tenant_id,
                    user_id=scope.user_id,
                    resource_type="mcp_server",
                    resource_id=spec.id,
                    required_scopes=tuple(spec.required_scopes),
                )
            )
        return spec.model_copy(
            update={
                "environment": resolved_environment,
                "headers": resolved_headers,
            }
        )

    def diagnostics(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._closed:
                return []
            clients = dict(self._clients)
            specs = dict(self._specs)
        return [
            clients[server_id].diagnostics()
            if server_id in clients
            else {**spec.public_snapshot(), "running": False}
            for server_id, spec in specs.items()
        ]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()


class McpToolProvider:
    """Bridges provider MCP tools into stable Harness Runtime contracts."""

    def __init__(self, clients: McpClientPool):
        self.clients = clients
        self._discovered: dict[str, tuple[int, set[str]]] = {}
        self._discovery_lock = threading.Lock()

    def register(
        self,
        binding: McpToolBinding,
        *,
        registry: ToolRegistry,
        executor: ToolExecutor,
    ) -> None:
        registry.register(binding.local_spec)
        executor.register(binding.local_spec.name, self._handler(binding))

    def close(self) -> None:
        self.clients.close()

    def diagnostics(self) -> list[dict[str, Any]]:
        return self.clients.diagnostics()

    def _handler(self, binding: McpToolBinding):
        def execute(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
            started_at = time.perf_counter()
            remote_arguments: dict[str, Any] = {}
            try:
                deadline = deadline_with_timeout(
                    context.deadline_monotonic,
                    _binding_timeout(binding),
                )
                client = self.clients.get(binding.server_id)
                self._ensure_remote_tool(
                    client,
                    binding.remote_name,
                    timeout_seconds=_remaining_timeout(deadline),
                )
                remote_arguments = binding.map_arguments(call.arguments)
                raw = client.call_tool(
                    binding.remote_name,
                    remote_arguments,
                    timeout_seconds=_remaining_timeout(deadline),
                )
                normalized = binding.normalize_result(raw, call.arguments)
                if raw.is_error and not normalized.error:
                    normalized.error = normalized.summary or "MCP tool returned an error"
                status = "failed" if normalized.error else "succeeded"
                summary = normalized.summary or normalized.error
                metadata = {
                    "mcp": {
                        "server_id": binding.server_id,
                        "transport": str(raw.metadata.get("transport") or "stdio"),
                        "protocol_version": str(raw.metadata.get("protocol_version") or ""),
                        "remote_tool": binding.remote_name,
                        "local_tool": binding.local_spec.name,
                        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                        "result_count": _result_count(normalized.data),
                        "untrusted_external_content": not binding.trusted_content,
                    }
                }
                observation = Observation(
                    id=f"{call.id}:observation",
                    run_id=context.run_id,
                    tool_call_id=call.id,
                    source=binding.local_spec.name,
                    status=status,
                    summary=summary,
                    data=normalized.data,
                    error=normalized.error,
                    metadata=metadata,
                )
                return ToolResult(
                    call=call,
                    status=status,
                    summary=summary,
                    data=normalized.data,
                    error=normalized.error,
                    retryable=normalized.retryable,
                    observation=observation,
                    metadata=metadata,
                )
            except McpProtocolError as exc:
                return _failed_mcp_result(
                    call,
                    context,
                    binding,
                    str(exc),
                    started_at=started_at,
                    retryable=False,
                )
            except (McpTimeoutError, McpTransportError) as exc:
                return _failed_mcp_result(
                    call,
                    context,
                    binding,
                    str(exc),
                    started_at=started_at,
                    retryable=True,
                )
            except (KeyError, TypeError, ValueError) as exc:
                return _failed_mcp_result(
                    call,
                    context,
                    binding,
                    str(exc),
                    started_at=started_at,
                    retryable=False,
                )

        return execute

    def _ensure_remote_tool(
        self,
        client: McpClient,
        remote_name: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        with self._discovery_lock:
            generation = int(getattr(client, "generation", 0) or 0)
            cached = self._discovered.get(client.server_id)
            if cached is None or cached[0] != generation:
                discovered = {
                    tool.name
                    for tool in client.list_tools(timeout_seconds=timeout_seconds)
                }
                generation = int(getattr(client, "generation", generation) or generation)
                self._discovered[client.server_id] = (generation, discovered)
            else:
                discovered = cached[1]
            if remote_name not in discovered:
                raise McpProtocolError(
                    f"MCP tool {remote_name} is not exposed by {client.server_id}"
                )


def _default_normalize(result: McpCallResult) -> McpNormalizedResult:
    data = dict(result.structured_content)
    if not data:
        data = {"content": result.content}
    return McpNormalizedResult(
        data=data,
        summary="MCP tool completed.",
        error="MCP tool returned an error" if result.is_error else "",
    )


def _default_client_factory(spec: McpServerSpec) -> McpClient:
    if spec.transport == "stdio":
        return StdioMcpClient(spec)
    if spec.transport == "streamable_http":
        return StreamableHttpMcpClient(spec)
    raise McpTransportError(f"unsupported MCP transport: {spec.transport}")


def _binding_timeout(binding: McpToolBinding) -> float | None:
    value = binding.local_spec.runtime_policy.get("timeout_seconds")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _result_count(data: dict[str, Any]) -> int:
    try:
        return max(0, int(data.get("result_count") or 0))
    except (TypeError, ValueError):
        return 0


def _remaining_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise McpTimeoutError("MCP tool deadline exceeded")
    return remaining


def _failed_mcp_result(
    call: ToolCall,
    context: ToolExecutionContext,
    binding: McpToolBinding,
    error: str,
    *,
    started_at: float,
    retryable: bool,
) -> ToolResult:
    metadata = {
        "mcp": {
            "server_id": binding.server_id,
            "transport": "stdio",
            "remote_tool": binding.remote_name,
            "local_tool": binding.local_spec.name,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "failed": True,
        }
    }
    observation = Observation(
        id=f"{call.id}:observation",
        run_id=context.run_id,
        tool_call_id=call.id,
        source=binding.local_spec.name,
        status="failed",
        summary=error,
        error=error,
        metadata=metadata,
    )
    return ToolResult(
        call=call,
        status="failed",
        summary=error,
        error=error,
        retryable=retryable,
        observation=observation,
        metadata=metadata,
    )
