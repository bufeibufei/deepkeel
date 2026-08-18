from __future__ import annotations

import base64
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import pytest

from deepkeel.contracts import ToolCall
from deepkeel.mcp_sdk import (
    MCP_PROTOCOL_VERSION,
    MCP_TASKS_EXTENSION,
    McpClientPool,
    McpServerSpec,
    McpToolBinding,
    McpToolProvider,
)
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor


class _ModernMcpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        type(self).requests.append(
            {
                "method": payload.get("method"),
                "params": params,
                "mcp_method": self.headers.get("Mcp-Method"),
                "mcp_name": self.headers.get("Mcp-Name"),
                "mcp_param_region": self.headers.get("Mcp-Param-Region"),
                "mcp_param_tenant": self.headers.get("Mcp-Param-Tenant"),
                "session_id": self.headers.get("Mcp-Session-Id"),
            }
        )
        method = str(payload.get("method") or "")
        if method == "server/discover":
            result = {
                "resultType": "complete",
                "supportedVersions": [MCP_PROTOCOL_VERSION],
                "capabilities": {
                    "tools": {},
                    "extensions": {MCP_TASKS_EXTENSION: {}},
                },
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "modern-contract-server",
                        "version": "2",
                    }
                },
            }
        elif method == "tools/list":
            result = {
                "resultType": "complete",
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Modern lookup",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string"},
                                "region": {
                                    "type": "string",
                                    "x-mcp-header": "Region",
                                },
                                "routing": {
                                    "type": "object",
                                    "properties": {
                                        "tenant": {
                                            "type": "string",
                                            "x-mcp-header": "Tenant",
                                        }
                                    },
                                },
                            },
                        },
                        "outputSchema": {
                            "oneOf": [
                                {"type": "object"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                        "taskSupport": "optional",
                    },
                    {
                        "name": "invalid-headers",
                        "description": "Must be filtered before use",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "first": {
                                    "type": "string",
                                    "x-mcp-header": "Tenant",
                                },
                                "second": {
                                    "type": "string",
                                    "x-mcp-header": "tenant",
                                },
                            },
                        },
                    },
                ],
                "ttlMs": 60_000,
                "cacheScope": "private",
            }
        elif method == "tools/call":
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            mode = str(arguments.get("mode") or "complete")
            if params.get("inputResponses"):
                result = {
                    "resultType": "complete",
                    "structuredContent": {"continued": True},
                    "content": [],
                }
            elif mode == "input":
                result = {
                    "resultType": "input_required",
                    "inputRequests": {
                        "destination": {
                            "method": "elicitation/create",
                            "params": {"message": "Where should the result be sent?"},
                        }
                    },
                    "requestState": "opaque-state",
                }
            elif mode == "task":
                result = {
                    "resultType": "task",
                    "task": {
                        "taskId": "task-1",
                        "status": "working",
                        "statusMessage": "Indexing records",
                        "ttlMs": 120_000,
                        "pollIntervalMs": 250,
                    },
                }
            elif mode == "invalid_output":
                result = {
                    "resultType": "complete",
                    "structuredContent": 42,
                    "content": [],
                }
            else:
                result = {
                    "resultType": "complete",
                    "structuredContent": ["one", "two"],
                    "content": [],
                }
        elif method == "tasks/get":
            result = {
                "task": {
                    "taskId": params["taskId"],
                    "status": "working",
                    "statusMessage": "Still working",
                    "ttlMs": 120_000,
                    "pollIntervalMs": 250,
                }
            }
        elif method in {"tasks/update", "tasks/cancel"}:
            result = {}
        else:
            self._respond(
                payload,
                error={"code": -32601, "message": f"unknown method: {method}"},
            )
            return
        self._respond(payload, result=result)

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _respond(
        self,
        request: dict[str, Any],
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request.get("id")}
        payload["error" if error is not None else "result"] = error or result or {}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _modern_server() -> Iterator[str]:
    _ModernMcpHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModernMcpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/mcp"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _pool(url: str) -> McpClientPool:
    return McpClientPool(
        [
            McpServerSpec(
                id="modern",
                transport="streamable_http",
                url=url,
                allow_insecure_http=True,
                allow_private_network=True,
            )
        ]
    )


def test_modern_transport_is_stateless_cacheable_and_task_aware() -> None:
    with _modern_server() as url:
        pool = _pool(url)
        client = pool.get("modern")

        with pytest.warns(RuntimeWarning, match="Ignoring invalid MCP tool"):
            first_tools = client.list_tools()
        second_tools = client.list_tools()
        complete = client.call_tool(
            "lookup",
            {
                "mode": "complete",
                "region": " 华北 ",
                "routing": {"tenant": "tenant-a"},
            },
        )
        continued = client.continue_tool(
            "lookup",
            {"mode": "input"},
            input_responses={"destination": {"action": "accept"}},
            request_state="opaque-state",
        )
        task = client.call_tool("lookup", {"mode": "task"}).task
        polled = client.get_task("task-1")
        assert client.update_task("task-1", {"approval": {"action": "accept"}}) is None
        assert client.cancel_task("task-1") is None
        diagnostics = client.diagnostics()
        pool.close()

    assert [item.name for item in first_tools] == ["lookup"]
    assert [item.name for item in second_tools] == ["lookup"]
    assert first_tools[0].task_support == "optional"
    assert complete.structured_content == ["one", "two"]
    assert continued.structured_content == {"continued": True}
    assert task is not None and task.poll_interval_ms == 250
    assert polled.task_id == "task-1"
    assert diagnostics["protocol_era"] == "modern"
    assert diagnostics["session_active"] is False
    methods = [str(item["method"]) for item in _ModernMcpHandler.requests]
    assert methods.count("server/discover") == 1
    assert methods.count("tools/list") == 1
    assert "initialize" not in methods
    assert all(item["session_id"] is None for item in _ModernMcpHandler.requests)
    assert all(
        item["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"]
        == MCP_PROTOCOL_VERSION
        for item in _ModernMcpHandler.requests
    )
    assert all(
        MCP_TASKS_EXTENSION
        in item["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"][
            "extensions"
        ]
        for item in _ModernMcpHandler.requests
    )
    assert all(
        item["mcp_method"] == item["method"] for item in _ModernMcpHandler.requests
    )
    tool_calls = [item for item in _ModernMcpHandler.requests if item["method"] == "tools/call"]
    assert all(item["mcp_name"] == "lookup" for item in tool_calls)
    complete_call = next(
        item
        for item in tool_calls
        if item["params"].get("arguments", {}).get("mode") == "complete"
    )
    encoded_region = base64.b64encode(" 华北 ".encode()).decode()
    assert complete_call["mcp_param_region"] == f"=?base64?{encoded_region}?="
    assert complete_call["mcp_param_tenant"] == "tenant-a"


def test_mcp_provider_projects_input_and_tasks_into_runtime_lifecycle() -> None:
    with _modern_server() as url:
        pool = _pool(url)
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        provider = McpToolProvider(pool)
        provider.register(
            McpToolBinding(
                server_id="modern",
                remote_name="lookup",
                local_spec=ToolSpec(
                    name="records.lookup",
                    parameters_schema={"type": "object"},
                ),
            ),
            registry=registry,
            executor=executor,
        )
        context = ToolExecutionContext(run_id="run-1", user_id="user-1")

        with pytest.warns(RuntimeWarning, match="Ignoring invalid MCP tool"):
            needs_input = executor.execute(
                ToolCall(
                    id="call-input",
                    name="records.lookup",
                    arguments={"mode": "input"},
                    idempotency_key="input-1",
                ),
                context,
            )
        waiting_task = executor.execute(
            ToolCall(
                id="call-task",
                name="records.lookup",
                arguments={"mode": "task"},
                idempotency_key="task-1",
            ),
            context,
        )
        invalid_output = executor.execute(
            ToolCall(
                id="call-invalid-output",
                name="records.lookup",
                arguments={"mode": "invalid_output"},
                idempotency_key="invalid-output-1",
            ),
            context,
        )
        provider.close()

    assert needs_input.status == "requires_user_action"
    assert needs_input.pending_action is not None
    assert needs_input.pending_action.action_type == "mcp_input_required"
    assert needs_input.pending_action.payload["request_state"] == "opaque-state"
    assert needs_input.pending_action.handoff_view == "mcp_elicitation"
    assert waiting_task.status == "waiting_async"
    assert waiting_task.observation is not None
    assert waiting_task.observation.status == "pending"
    assert waiting_task.data["task_id"] == "task-1"
    assert waiting_task.data["poll_interval_ms"] == 250
    assert invalid_output.status == "failed"
    assert invalid_output.retryable is False
    assert "outside outputSchema" in invalid_output.error
