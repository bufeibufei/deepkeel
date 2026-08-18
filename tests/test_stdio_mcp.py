from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from deepkeel.mcp_sdk import (
    McpProtocolError,
    McpServerSpec,
    McpTimeoutError,
    McpTransportError,
    StdioMcpClient,
)


FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _server(**updates) -> McpServerSpec:
    values = {
        "id": "fake.tools",
        "command": sys.executable,
        "args": ["-u", str(FAKE_SERVER)],
        "environment": {"MCP_FAKE_SECRET": "private-value"},
        "startup_timeout_seconds": 5,
        "request_timeout_seconds": 2,
        "protocol_version": "2025-03-26",
    }
    values.update(updates)
    return McpServerSpec(**values)


def test_stdio_client_negotiates_calls_and_redacts_secrets(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_UNRELATED_SECRET", "must-not-be-inherited")
    client = StdioMcpClient(_server())
    try:
        assert [tool.name for tool in client.list_tools()] == ["lookup"]
        result = client.call_tool("lookup", {"query": "inventory"})
        time.sleep(0.05)
        diagnostics = client.diagnostics()
    finally:
        client.close()

    assert result.structured_content["results"][0]["title"] == "Result for inventory"
    assert "private-value" not in str(result.structured_content)
    assert "must-not-be-inherited" not in str(result.structured_content)
    assert diagnostics["protocol_version"] == "2025-03-26"
    assert diagnostics["server_info"] == {"name": "fake-tools", "version": "1.0"}
    assert "private-value" not in str(diagnostics)
    assert "***" in str(diagnostics["stderr_tail"])


def test_stdio_client_redacts_protocol_errors() -> None:
    client = StdioMcpClient(_server())
    try:
        with pytest.raises(McpProtocolError) as raised:
            client.call_tool("lookup", {"query": "protocol_error"})
    finally:
        client.close()

    assert "private-value" not in str(raised.value)
    assert "***" in str(raised.value)


def test_stdio_timeout_does_not_poison_later_calls() -> None:
    client = StdioMcpClient(_server(request_timeout_seconds=0.03))
    try:
        with pytest.raises(McpTimeoutError):
            client.call_tool("lookup", {"query": "timeout"})
        time.sleep(0.25)
        recovered = client.call_tool("lookup", {"query": "recovered"}, timeout_seconds=1)
    finally:
        client.close()

    assert recovered.structured_content["results"][0]["title"] == "Result for recovered"


def test_stdio_timeout_isolated_from_concurrent_request() -> None:
    client = StdioMcpClient(_server())
    completed: list[str] = []

    def slow_call() -> None:
        result = client.call_tool("lookup", {"query": "concurrent_slow"}, timeout_seconds=1)
        completed.append(result.structured_content["results"][0]["title"])

    worker = threading.Thread(target=slow_call)
    try:
        client.start()
        generation = client.generation
        worker.start()
        deadline = time.monotonic() + 0.5
        while client.diagnostics()["in_flight_requests"] < 1:
            if time.monotonic() >= deadline:
                pytest.fail("concurrent request did not start")
            time.sleep(0.005)
        with pytest.raises(McpTimeoutError):
            client.call_tool("lookup", {"query": "timeout"}, timeout_seconds=0.03)
        worker.join(timeout=1)
        diagnostics = client.diagnostics()
    finally:
        client.close()

    assert completed == ["Result for concurrent_slow"]
    assert client.generation == generation
    assert diagnostics["timeout_count"] == 1
    assert diagnostics["restart_count"] == 0


def test_closed_stdio_client_cannot_restart() -> None:
    client = StdioMcpClient(_server())
    client.close()
    with pytest.raises(McpTransportError, match="closed"):
        client.list_tools()
