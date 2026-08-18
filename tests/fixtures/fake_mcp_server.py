from __future__ import annotations

import json
import os
import sys
import threading
import time


write_lock = threading.Lock()


def write(payload: dict) -> None:
    with write_lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def write_tool_result(request_id: int, query: str, secret: str) -> None:
    if query in {"timeout", "concurrent_slow"}:
        time.sleep(0.2)
    payload = {
        "results": [{"title": f"Result for {query}", "summary": secret}],
        "inherited_secret": os.getenv("HARNESS_UNRELATED_SECRET", ""),
    }
    write(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
                "isError": False,
            },
        }
    )


secret = os.getenv("MCP_FAKE_SECRET", "")
if secret:
    sys.stderr.write(f"server secret={secret}\n")
    sys.stderr.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "id" not in request:
        continue
    request_id = request["id"]
    method = request.get("method")
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    if method == "server/discover":
        delay = float(os.getenv("MCP_DISCOVERY_DELAY", "0") or 0)
        if delay > 0:
            time.sleep(delay)
    if method == "initialize":
        result = {
            "protocolVersion": request.get("params", {}).get("protocolVersion"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-tools", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "lookup",
                    "description": "fake lookup",
                    "inputSchema": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ]
        }
    elif method == "tools/call":
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        query = str(arguments.get("query") or "")
        if query == "protocol_error":
            write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": f"rejected secret={secret}"},
                }
            )
            continue
        threading.Thread(
            target=write_tool_result,
            args=(request_id, query, secret),
            daemon=True,
        ).start()
        continue
    else:
        write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "method not found"},
            }
        )
        continue
    write({"jsonrpc": "2.0", "id": request_id, "result": result})
