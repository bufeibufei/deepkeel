from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from deepkeel.contrib.a2a.contracts import (
    A2A_PROTOCOL_VERSION,
    A2AAgentCard,
    A2AMessage,
    A2AProtocolError,
    A2ASendResponse,
    A2ATask,
    A2ATransportError,
)


class HttpJsonA2AClient:
    """Small HTTP+JSON/REST A2A 1.0 client with explicit egress controls."""

    def __init__(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        protocol_version: str = A2A_PROTOCOL_VERSION,
        request_timeout_seconds: float = 45.0,
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
        allowed_hosts: list[str] | None = None,
        max_response_bytes: int = 4_194_304,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.protocol_version = protocol_version
        self.request_timeout_seconds = max(0.001, float(request_timeout_seconds))
        self.allow_insecure_http = allow_insecure_http
        self.allow_private_network = allow_private_network
        self.allowed_hosts = {item.lower().strip() for item in allowed_hosts or [] if item.strip()}
        self.max_response_bytes = max(1, int(max_response_bytes))
        self._headers = dict(headers or {})
        self._client = client or httpx.Client()
        self._owns_client = client is None
        self._validate_endpoint()

    def send_message(
        self,
        message: A2AMessage,
        *,
        accepted_output_modes: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> A2ASendResponse:
        payload: dict[str, Any] = {
            "message": message.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                exclude_defaults=True,
            )
        }
        if accepted_output_modes:
            payload["configuration"] = {
                "acceptedOutputModes": list(accepted_output_modes)
            }
        value = self._request(
            "POST",
            "/message:send",
            json=payload,
            timeout_seconds=timeout_seconds,
        )
        return A2ASendResponse.model_validate(value)

    def get_task(
        self,
        task_id: str,
        *,
        history_length: int = 5,
        timeout_seconds: float | None = None,
    ) -> A2ATask:
        value = self._request(
            "GET",
            f"/tasks/{quote(task_id, safe='')}",
            params={"historyLength": max(0, int(history_length))},
            timeout_seconds=timeout_seconds,
        )
        return A2ATask.model_validate(value.get("task") or value)

    def cancel_task(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> A2ATask:
        value = self._request(
            "POST",
            f"/tasks/{quote(task_id, safe='')}:cancel",
            json={},
            timeout_seconds=timeout_seconds,
        )
        return A2ATask.model_validate(value.get("task") or value)

    def get_agent_card(
        self,
        card_url: str,
        *,
        timeout_seconds: float | None = None,
    ) -> A2AAgentCard:
        value = self._request_absolute(
            "GET",
            card_url,
            timeout_seconds=timeout_seconds,
        )
        return A2AAgentCard.model_validate(value)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._request_absolute(
            method,
            f"{self.endpoint}{path}",
            json=json,
            params=params,
            timeout_seconds=timeout_seconds,
        )

    def _request_absolute(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._validate_url(url)
        headers = {
            "Accept": "application/a2a+json, application/json",
            "Content-Type": "application/a2a+json",
            "A2A-Version": self.protocol_version,
            **self._headers,
        }
        try:
            with self._client.stream(
                method,
                url,
                headers=headers,
                json=json,
                params=params,
                timeout=timeout_seconds or self.request_timeout_seconds,
            ) as response:
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > self.max_response_bytes:
                        raise A2ATransportError("A2A response exceeds max_response_bytes")
                status_code = response.status_code
                content_type = response.headers.get("content-type", "")
        except httpx.TimeoutException as exc:
            raise A2ATransportError("A2A request timed out") from exc
        except httpx.HTTPError as exc:
            raise A2ATransportError(f"A2A transport failed: {exc}") from exc
        try:
            value = httpx.Response(
                status_code,
                content=bytes(content),
                headers={"content-type": content_type},
            ).json()
        except ValueError as exc:
            raise A2AProtocolError("A2A response is not valid JSON") from exc
        if status_code >= 400:
            error = value.get("error") if isinstance(value, dict) else None
            message = error.get("message") if isinstance(error, dict) else str(value)
            raise A2ATransportError(f"A2A HTTP {status_code}: {message}")
        if not isinstance(value, dict):
            raise A2AProtocolError("A2A response must be a JSON object")
        return value

    def _validate_endpoint(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("A2A endpoint must be an absolute http(s) URL")
        if parsed.scheme == "http" and not self.allow_insecure_http:
            raise ValueError("A2A endpoint requires https unless explicitly allowed")
        if parsed.username or parsed.password:
            raise ValueError("A2A endpoint cannot contain credentials")
        self._validate_url(self.endpoint)

    def _validate_url(self, value: str) -> None:
        parsed = urlsplit(value)
        endpoint = urlsplit(self.endpoint)
        hostname = str(parsed.hostname or "").lower()
        if not hostname or hostname != str(endpoint.hostname or "").lower():
            raise A2ATransportError("A2A request cannot leave the configured endpoint host")
        if self.allowed_hosts and hostname not in self.allowed_hosts:
            raise A2ATransportError("A2A endpoint host is not allowed")
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
            raise A2ATransportError("A2A endpoint DNS resolution failed") from exc
        if not self.allow_private_network and any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise A2ATransportError("A2A endpoint resolved to a non-public address")

    def public_endpoint(self) -> str:
        parsed = urlsplit(self.endpoint)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
