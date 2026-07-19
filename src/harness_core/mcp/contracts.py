from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MCP_INHERITED_ENVIRONMENT_KEYS = [
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LOCALAPPDATA",
    "APPDATA",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "UV_CACHE_DIR",
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
]


class McpServerSpec(BaseModel):
    """Provider-neutral MCP server configuration owned by the runtime."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    transport: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict, exclude=True)
    secret_environment: dict[str, str] = Field(default_factory=dict, exclude=True)
    required_scopes: list[str] = Field(default_factory=list)
    trust_level: Literal["untrusted", "trusted"] = "untrusted"
    inherited_environment_keys: list[str] = Field(
        default_factory=lambda: list(DEFAULT_MCP_INHERITED_ENVIRONMENT_KEYS)
    )
    startup_timeout_seconds: float = Field(default=120.0, gt=0)
    request_timeout_seconds: float = Field(default=45.0, gt=0)
    shutdown_timeout_seconds: float = Field(default=3.0, gt=0)

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "environment_keys": sorted(self.environment),
            "secret_environment_keys": sorted(self.secret_environment),
            "required_scopes": sorted(self.required_scopes),
            "trust_level": self.trust_level,
            "inherited_environment_keys": sorted(self.inherited_environment_keys),
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
        }


class McpRemoteTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class McpCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: list[dict[str, Any]] = Field(default_factory=list)
    structured_content: dict[str, Any] = Field(default_factory=dict)
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class McpClient(Protocol):
    @property
    def server_id(self) -> str: ...

    @property
    def generation(self) -> int: ...

    def list_tools(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> list[McpRemoteTool]: ...

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> McpCallResult: ...

    def diagnostics(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
