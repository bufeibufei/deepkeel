from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from itertools import count
from typing import Any

from harness_core.mcp.contracts import McpCallResult, McpRemoteTool, McpServerSpec
from harness_core.mcp.protocol import (
    MAX_TOOL_LIST_PAGES,
    MCP_PROTOCOL_VERSION,
    SUPPORTED_MCP_PROTOCOL_VERSIONS,
    McpProtocolError,
    McpTimeoutError,
    McpTransportError,
    safe_server_info,
    structured_content_from_text,
)


logger = logging.getLogger(__name__)


class StdioMcpClient:
    """Persistent, concurrent JSON-RPC client for stdio MCP servers."""

    def __init__(self, spec: McpServerSpec):
        if spec.transport != "stdio":
            raise ValueError(f"unsupported MCP transport: {spec.transport}")
        self.spec = spec
        self._process: subprocess.Popen[str] | None = None
        self._state_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any] | BaseException]] = {}
        self._request_ids = count(1)
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._protocol_version = ""
        self._server_info: dict[str, Any] = {}
        self._closed = False
        self._generation = 0
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
            if self._process is not None and self._process.poll() is None and self._protocol_version:
                return
            self._stop_locked()
            self._launch_locked()
            try:
                initialized = self._request_started(
                    "initialize",
                    {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {
                            "name": self.spec.client_name,
                            "version": self.spec.client_version,
                        },
                    },
                    timeout_seconds=_bounded_timeout(
                        timeout_seconds,
                        self.spec.startup_timeout_seconds,
                    ),
                )
                negotiated_version = str(
                    initialized.get("protocolVersion") or MCP_PROTOCOL_VERSION
                )
                if negotiated_version not in SUPPORTED_MCP_PROTOCOL_VERSIONS:
                    raise McpProtocolError(
                        f"unsupported MCP protocol version: {negotiated_version}"
                    )
                self._protocol_version = negotiated_version
                self._server_info = (
                    dict(initialized.get("serverInfo"))
                    if isinstance(initialized.get("serverInfo"), dict)
                    else {}
                )
                self._write_message(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    }
                )
            except Exception:
                self._stop_locked()
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
            result = self._request_started(
                "tools/list",
                params,
                timeout_seconds=_remaining(deadline),
            )
            for item in result.get("tools", []) if isinstance(result.get("tools"), list) else []:
                if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                    continue
                tools.append(
                    McpRemoteTool(
                        name=str(item["name"]),
                        description=str(item.get("description") or ""),
                        input_schema=(
                            dict(item.get("inputSchema"))
                            if isinstance(item.get("inputSchema"), dict)
                            else {}
                        ),
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
        deadline = _deadline(timeout_seconds or self.spec.request_timeout_seconds)
        self.start(timeout_seconds=_remaining(deadline))
        result = self._request_started(
            "tools/call",
            {"name": name, "arguments": dict(arguments)},
            timeout_seconds=_remaining(deadline),
        )
        content = self._redact_value(
            [item for item in result.get("content", []) if isinstance(item, dict)]
        )
        structured = (
            self._redact_value(dict(result.get("structuredContent")))
            if isinstance(result.get("structuredContent"), dict)
            else structured_content_from_text(content)
        )
        return McpCallResult(
            content=content,
            structured_content=structured,
            is_error=bool(result.get("isError")),
            metadata={
                "server_id": self.server_id,
                "transport": "stdio",
                "protocol_version": self._protocol_version,
                "server_info": safe_server_info(self._server_info),
            },
        )

    def diagnostics(self) -> dict[str, Any]:
        process = self._process
        with self._pending_lock:
            in_flight_requests = len(self._pending)
        return {
            **self.spec.public_snapshot(),
            "running": bool(process is not None and process.poll() is None),
            "protocol_version": self._protocol_version,
            "server_info": safe_server_info(self._server_info),
            "stderr_tail": list(self._stderr_tail),
            "generation": self._generation,
            "restart_count": max(0, self._generation - 1),
            "in_flight_requests": in_flight_requests,
            "timeout_count": self._timeout_count,
            "last_transport_error": self._redact(self._last_transport_error),
        }

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            self._stop_locked()

    def _launch_locked(self) -> None:
        environment = {
            key: os.environ[key]
            for key in self.spec.inherited_environment_keys
            if key in os.environ
        }
        environment.update(self.spec.environment)
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                [self.spec.command, *self.spec.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise McpTransportError(
                f"failed to start MCP server {self.server_id}: {exc}"
            ) from exc
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process,),
            name=f"mcp-{self.server_id}-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._process,),
            name=f"mcp-{self.server_id}-stderr",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()
        self._generation += 1

    def _request_started(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        request_id = next(self._request_ids)
        response_queue: queue.Queue[dict[str, Any] | BaseException] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue
        try:
            self._write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
            timeout = timeout_seconds or self.spec.request_timeout_seconds
            try:
                response = response_queue.get(timeout=max(0.001, float(timeout)))
            except queue.Empty as exc:
                error = McpTimeoutError(
                    f"MCP request timed out: {self.server_id}.{method}"
                )
                self._timeout_count += 1
                self._last_transport_error = str(error)
                self._cancel_request(request_id, str(error))
                self._reset_after_timeout(request_id)
                raise error from exc
            if isinstance(response, BaseException):
                raise response
            if isinstance(response.get("error"), dict):
                error = response["error"]
                raise McpProtocolError(
                    self._redact(
                        f"MCP {method} failed ({error.get('code')}): {error.get('message')}"
                    )
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise McpProtocolError(f"MCP {method} returned an invalid result")
            return result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def _write_message(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise McpTransportError(f"MCP server is not running: {self.server_id}")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            try:
                process.stdin.write(line + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise McpTransportError(
                    f"MCP server pipe is unavailable: {self.server_id}"
                ) from exc

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    error = McpProtocolError(
                        f"MCP server emitted invalid JSON on stdout: {self.server_id}"
                    )
                    self._fail_pending(error)
                    if process.poll() is None:
                        process.terminate()
                    return
                messages = payload if isinstance(payload, list) else [payload]
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    if "method" in message:
                        self._handle_server_message(process, message)
                        continue
                    if "id" not in message or not (
                        "result" in message or "error" in message
                    ):
                        continue
                    try:
                        request_id = int(message["id"])
                    except (TypeError, ValueError):
                        continue
                    with self._pending_lock:
                        target = self._pending.get(request_id)
                    if target is not None:
                        try:
                            target.put_nowait(message)
                        except queue.Full:
                            logger.debug(
                                "ignored duplicate MCP response from %s for request %s",
                                self.server_id,
                                request_id,
                            )
        finally:
            if self._process is process:
                self._fail_pending(
                    McpTransportError(f"MCP server exited: {self.server_id}")
                )

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            clean = self._redact(line.strip())
            if clean:
                self._stderr_tail.append(clean[:1000])

    def _redact(self, value: str) -> str:
        clean = value
        for secret in self.spec.environment.values():
            if secret:
                clean = clean.replace(secret, "***")
        return clean

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._redact_value(item)
                for key, item in value.items()
            }
        return value

    def _handle_server_message(
        self,
        process: subprocess.Popen[str],
        message: dict[str, Any],
    ) -> None:
        if "id" not in message or self._process is not process:
            return
        method = str(message.get("method") or "")
        if method == "ping":
            response = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        else:
            response = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": "method not supported"},
            }
        self._write_message(response)

    def _cancel_request(self, request_id: int, reason: str) -> None:
        try:
            self._write_message(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": request_id, "reason": reason},
                }
            )
        except McpTransportError:
            pass

    def _reset_after_timeout(self, request_id: int) -> None:
        # A stdio server may support concurrent requests. Do not destroy healthy
        # in-flight work just because one request exceeded its own deadline.
        with self._pending_lock:
            has_other_requests = any(
                pending_id != request_id for pending_id in self._pending
            )
        if has_other_requests:
            return
        with self._state_lock:
            if not self._closed:
                self._stop_locked()

    def _fail_pending(self, error: BaseException) -> None:
        self._last_transport_error = str(error)
        with self._pending_lock:
            targets = list(self._pending.values())
        for target in targets:
            try:
                target.put_nowait(error)
            except queue.Full:
                pass

    def _stop_locked(self) -> None:
        process = self._process
        reader_thread = self._reader_thread
        stderr_thread = self._stderr_thread
        self._process = None
        self._reader_thread = None
        self._stderr_thread = None
        self._protocol_version = ""
        self._server_info = {}
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.spec.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.spec.shutdown_timeout_seconds)
        self._fail_pending(McpTransportError(f"MCP server closed: {self.server_id}"))
        current = threading.current_thread()
        for thread in (reader_thread, stderr_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=min(0.5, self.spec.shutdown_timeout_seconds))


def _deadline(timeout_seconds: float) -> float:
    return time.monotonic() + max(0.001, float(timeout_seconds))


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise McpTimeoutError("MCP request deadline exceeded")
    return remaining


def _bounded_timeout(requested: float | None, maximum: float) -> float:
    if requested is None:
        return maximum
    return max(0.001, min(float(requested), maximum))
