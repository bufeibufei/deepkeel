from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from deepkeel.adapter_sdk import GovernanceScope, MappingSecretProvider
from deepkeel.mcp import McpClientPool, McpServerSpec, McpTransportError


class _McpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict] = []
    deleted = False
    session_id = "session-contract-test"
    initialize_count = 0
    expire_next_call = False

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(
            {
                "payload": payload,
                "accept": self.headers.get("Accept"),
                "authorization": self.headers.get("Authorization"),
                "session_id": self.headers.get("Mcp-Session-Id"),
                "protocol_version": self.headers.get("MCP-Protocol-Version"),
            }
        )
        method = str(payload.get("method") or "")
        if method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if method != "initialize" and self.headers.get("Mcp-Session-Id") != self.session_id:
            self._json_response(400, {"error": "missing session"})
            return
        if method == "initialize":
            type(self).initialize_count += 1
            self._json_response(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "contract-server", "version": "1"},
                    },
                },
                session_id=self.session_id,
            )
            return
        if method == "tools/call" and type(self).expire_next_call:
            type(self).expire_next_call = False
            self._json_response(404, {"error": "expired session"})
            return
        if method == "tools/list":
            body = (
                "event: message\n"
                "data: "
                + json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "search",
                                    "description": "Search records",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"query": {"type": "string"}},
                                    },
                                }
                            ]
                        },
                    }
                )
                + "\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if method == "tools/call":
            query = payload["params"]["arguments"]["query"]
            self._json_response(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "content": [{"type": "text", "text": f"result:{query}"}],
                        "structuredContent": {"query": query, "count": 1},
                        "isError": False,
                    },
                },
            )
            return
        self._json_response(404, {"error": "unknown method"})

    def do_DELETE(self) -> None:
        type(self).deleted = True
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args) -> None:
        return

    def _json_response(
        self,
        status: int,
        payload: dict,
        *,
        session_id: str = "",
    ) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _mcp_server() -> Iterator[str]:
    _McpHandler.requests = []
    _McpHandler.deleted = False
    _McpHandler.initialize_count = 0
    _McpHandler.expire_next_call = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _McpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/mcp"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_streamable_http_transport_negotiates_session_and_redacts_secrets() -> None:
    with _mcp_server() as url:
        spec = McpServerSpec(
            id="remote-search",
            transport="streamable_http",
            url=url,
            allow_insecure_http=True,
            allow_private_network=True,
            protocol_version="2025-03-26",
            secret_headers={"Authorization": "remote-search-token"},
            required_scopes=["search.read"],
        )
        pool = McpClientPool(
            [spec],
            secret_provider=MappingSecretProvider(
                {"remote-search-token": "Bearer contract-secret"}
            ),
            governance_scope=GovernanceScope(scopes=frozenset({"search.read"})),
        )
        client = pool.get("remote-search")

        tools = client.list_tools()
        result = client.call_tool("search", {"query": "bearing"})
        diagnostics = client.diagnostics()
        pool.close()

    assert [tool.name for tool in tools] == ["search"]
    assert result.structured_content == {"query": "bearing", "count": 1}
    assert result.metadata["transport"] == "streamable_http"
    assert diagnostics["session_active"] is True
    assert diagnostics["server_info"]["name"] == "contract-server"
    assert "contract-secret" not in str(diagnostics)
    assert all(
        request["accept"] == "application/json, text/event-stream"
        for request in _McpHandler.requests
    )
    assert all(
        request["authorization"] == "Bearer contract-secret"
        for request in _McpHandler.requests
    )
    assert _McpHandler.requests[0]["session_id"] is None
    assert all(
        request["session_id"] == _McpHandler.session_id
        for request in _McpHandler.requests[1:]
    )
    assert all(
        request["protocol_version"] == "2025-03-26"
        for request in _McpHandler.requests[1:]
    )
    assert _McpHandler.deleted is True


def test_streamable_http_transport_recovers_an_expired_session_once() -> None:
    with _mcp_server() as url:
        pool = McpClientPool(
            [
                McpServerSpec(
                    id="recovering-server",
                    transport="streamable_http",
                    url=f"{url}?access_token=must-not-leak",
                    allow_insecure_http=True,
                    allow_private_network=True,
                    protocol_version="2025-03-26",
                )
            ]
        )
        client = pool.get("recovering-server")
        client.list_tools()
        _McpHandler.expire_next_call = True

        result = client.call_tool("search", {"query": "recovered"})
        diagnostics = client.diagnostics()
        pool.close()

    assert result.structured_content["query"] == "recovered"
    assert _McpHandler.initialize_count == 2
    assert diagnostics["generation"] == 2
    assert diagnostics["restart_count"] == 1
    assert "access_token" not in diagnostics["url"]
    assert "must-not-leak" not in str(diagnostics)


def test_mcp_pool_enforces_server_scopes_even_without_secrets() -> None:
    spec = McpServerSpec(
        id="scope-protected",
        transport="streamable_http",
        url="https://mcp.example.test/service",
        required_scopes=["records.read"],
    )
    pool = McpClientPool([spec], governance_scope=GovernanceScope())

    with pytest.raises(PermissionError, match="records.read"):
        pool.get("scope-protected")

    pool.close()


def test_streamable_http_blocks_private_network_without_explicit_opt_in() -> None:
    with _mcp_server() as url:
        pool = McpClientPool(
            [
                McpServerSpec(
                    id="private-network-blocked",
                    transport="streamable_http",
                    url=url,
                    allow_insecure_http=True,
                )
            ]
        )
        client = pool.get("private-network-blocked")

        with pytest.raises(McpTransportError, match="non-public address"):
            client.list_tools()

        pool.close()


def test_streamable_http_enforces_request_and_response_size_limits() -> None:
    with _mcp_server() as url:
        request_limited = McpClientPool(
            [
                McpServerSpec(
                    id="request-limited",
                    transport="streamable_http",
                    url=url,
                    allow_insecure_http=True,
                    allow_private_network=True,
                    protocol_version="2025-03-26",
                    max_request_bytes=32,
                )
            ]
        )
        with pytest.raises(McpTransportError, match="max_request_bytes"):
            request_limited.get("request-limited").list_tools()
        request_limited.close()

        response_limited = McpClientPool(
            [
                McpServerSpec(
                    id="response-limited",
                    transport="streamable_http",
                    url=url,
                    allow_insecure_http=True,
                    allow_private_network=True,
                    protocol_version="2025-03-26",
                    max_response_bytes=32,
                )
            ]
        )
        with pytest.raises(McpTransportError, match="max_response_bytes"):
            response_limited.get("response-limited").list_tools()
        response_limited.close()
