from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping, Protocol
from uuid import uuid4

from deepkeel.contracts import ToolCall
from deepkeel.scope import RuntimeScope


class WorkspaceRetention(StrEnum):
    EPHEMERAL = "ephemeral"
    RUN = "run"
    DURABLE = "durable"


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    wall_time_seconds: float | None = None
    cpu_time_seconds: float | None = None
    memory_bytes: int | None = None
    output_bytes: int | None = None
    file_count: int | None = None
    process_count: int | None = None
    network_access: bool = False

    def __post_init__(self) -> None:
        for name in (
            "wall_time_seconds",
            "cpu_time_seconds",
            "memory_bytes",
            "output_bytes",
            "file_count",
            "process_count",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"sandbox limit {name} must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "memory_bytes": self.memory_bytes,
            "output_bytes": self.output_bytes,
            "file_count": self.file_count,
            "process_count": self.process_count,
            "network_access": self.network_access,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceRequest:
    run_id: str
    thread_id: str
    turn_id: str
    tool_call: ToolCall
    scope: RuntimeScope = field(default_factory=RuntimeScope)
    retention: WorkspaceRetention = WorkspaceRetention.EPHEMERAL
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class WorkspaceLease:
    workspace_id: str
    root_path: str = ""
    retention: WorkspaceRetention = WorkspaceRetention.EPHEMERAL
    writable: bool = True
    available: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workspace_id.strip():
            raise ValueError("workspace_id must not be blank")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "root_path": self.root_path,
            "retention": self.retention.value,
            "writable": self.writable,
            "available": self.available,
            "metadata": dict(self.metadata),
        }


class WorkspacePort(Protocol):
    def allocate(self, request: WorkspaceRequest) -> WorkspaceLease: ...

    def release(self, lease: WorkspaceLease, *, status: str) -> None: ...


@dataclass(frozen=True, slots=True)
class SandboxRequest:
    run_id: str
    thread_id: str
    turn_id: str
    tool_call: ToolCall
    profile: str = "default"
    scope: RuntimeScope = field(default_factory=RuntimeScope)
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    workspace: WorkspaceLease | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SandboxLease:
    sandbox_id: str
    backend: str
    enforced: bool
    workspace: WorkspaceLease | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sandbox_id.strip():
            raise ValueError("sandbox_id must not be blank")
        if not self.backend.strip():
            raise ValueError("sandbox backend must not be blank")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "backend": self.backend,
            "enforced": self.enforced,
            "workspace": self.workspace.as_dict() if self.workspace is not None else None,
            "metadata": dict(self.metadata),
        }


class SandboxPort(Protocol):
    def acquire(self, request: SandboxRequest) -> SandboxLease: ...

    def release(self, lease: SandboxLease, *, status: str) -> None: ...


class NoopSandboxPort:
    """Explicit development adapter that declares isolation is not enforced."""

    def acquire(self, request: SandboxRequest) -> SandboxLease:
        return SandboxLease(
            sandbox_id=f"noop-{request.tool_call.id}",
            backend="none",
            enforced=False,
            workspace=request.workspace,
        )

    def release(self, lease: SandboxLease, *, status: str) -> None:
        del lease, status


class LocalWorkspacePort:
    """Development workspace adapter with bounded roots and deterministic cleanup."""

    def __init__(self, base_directory: str | Path | None = None) -> None:
        root = Path(base_directory) if base_directory is not None else Path(tempfile.gettempdir())
        self.base_directory = (root / "deepkeel-workspaces").resolve()
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self._owned: set[Path] = set()
        self._lock = RLock()

    def allocate(self, request: WorkspaceRequest) -> WorkspaceLease:
        prefix = f"{_safe_component(request.run_id)}-{_safe_component(request.tool_call.id)}-"
        root = Path(tempfile.mkdtemp(prefix=prefix, dir=self.base_directory)).resolve()
        if self.base_directory not in root.parents:
            raise RuntimeError("allocated workspace escaped the configured base directory")
        with self._lock:
            self._owned.add(root)
        return WorkspaceLease(
            workspace_id=f"workspace-{uuid4().hex}",
            root_path=str(root),
            retention=request.retention,
            metadata={"adapter": "local-temporary"},
        )

    def release(self, lease: WorkspaceLease, *, status: str) -> None:
        del status
        if lease.retention != WorkspaceRetention.EPHEMERAL or not lease.root_path:
            return
        root = Path(lease.root_path).resolve()
        with self._lock:
            if root not in self._owned:
                return
            self._owned.remove(root)
        if self.base_directory not in root.parents:
            raise RuntimeError("refusing to remove a workspace outside the configured base")
        if root.exists():
            shutil.rmtree(root)


class SandboxUnavailable(RuntimeError):
    code = "SANDBOX_UNAVAILABLE"


def _safe_component(value: str) -> str:
    normalized = "".join(character for character in value if character.isalnum() or character in "-_")
    return (normalized or "run")[:32]


__all__ = [
    "LocalWorkspacePort",
    "NoopSandboxPort",
    "SandboxLease",
    "SandboxLimits",
    "SandboxPort",
    "SandboxRequest",
    "SandboxUnavailable",
    "WorkspaceLease",
    "WorkspacePort",
    "WorkspaceRequest",
    "WorkspaceRetention",
]
