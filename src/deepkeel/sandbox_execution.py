from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepkeel.async_ports import run_sync_adapter
from deepkeel.contracts import ToolCall
from deepkeel.sandbox import (
    SandboxLease,
    SandboxLimits,
    SandboxPort,
    SandboxRequest,
    SandboxUnavailable,
    WorkspaceLease,
    WorkspacePort,
    WorkspaceRequest,
    WorkspaceRetention,
)
from deepkeel.tool_execution import ToolExecutionContext
from deepkeel.tool_registry import ToolSpec
from deepkeel.type_narrowing import as_dict


@dataclass(slots=True)
class PreparedToolEnvironment:
    context: ToolExecutionContext
    sandbox_port: SandboxPort | None = None
    sandbox: SandboxLease | None = None
    workspace_port: WorkspacePort | None = None
    workspace: WorkspaceLease | None = None

    async def close(self, *, status: str) -> None:
        errors: list[str] = []
        if self.sandbox_port is not None and self.sandbox is not None:
            try:
                await run_sync_adapter(self.sandbox_port.release, self.sandbox, status=status)
            except Exception as exc:
                errors.append(f"sandbox release failed: {exc}")
        if self.workspace_port is not None and self.workspace is not None:
            try:
                await run_sync_adapter(self.workspace_port.release, self.workspace, status=status)
            except Exception as exc:
                errors.append(f"workspace release failed: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))


async def prepare_tool_environment(
    *,
    sandbox_port: SandboxPort | None,
    workspace_port: WorkspacePort | None,
    call: ToolCall,
    context: ToolExecutionContext,
    spec: ToolSpec,
) -> PreparedToolEnvironment:
    policy = as_dict(spec.runtime_policy.get("sandbox"))
    sandbox_required = bool(policy.get("required"))
    workspace_required = bool(policy.get("workspace_required"))
    sandbox_enabled = sandbox_required or bool(policy.get("enabled"))
    workspace_enabled = workspace_required or bool(policy.get("workspace"))
    workspace: WorkspaceLease | None = None
    if workspace_enabled:
        if workspace_port is None:
            raise SandboxUnavailable(f"tool {call.name} requires a WorkspacePort")
        retention = _retention(policy.get("workspace_retention"))
        workspace = await run_sync_adapter(
            workspace_port.allocate,
            WorkspaceRequest(
                run_id=context.run_id,
                thread_id=context.thread_id,
                turn_id=context.turn_id,
                tool_call=call,
                scope=context.scope,
                retention=retention,
                metadata={"tool_name": call.name},
            ),
        )
        if workspace_required and not workspace.available:
            await run_sync_adapter(workspace_port.release, workspace, status="rejected")
            raise SandboxUnavailable(f"tool {call.name} requires an available workspace")

    sandbox: SandboxLease | None = None
    if sandbox_enabled:
        if sandbox_port is None:
            if workspace_port is not None and workspace is not None:
                await run_sync_adapter(workspace_port.release, workspace, status="failed")
            raise SandboxUnavailable(f"tool {call.name} requires a SandboxPort")
        try:
            limits = _limits(policy)
            sandbox = await run_sync_adapter(
                sandbox_port.acquire,
                SandboxRequest(
                    run_id=context.run_id,
                    thread_id=context.thread_id,
                    turn_id=context.turn_id,
                    tool_call=call,
                    profile=str(policy.get("profile") or "default"),
                    scope=context.scope,
                    limits=limits,
                    workspace=workspace,
                    metadata={"tool_name": call.name},
                ),
            )
        except Exception:
            if workspace_port is not None and workspace is not None:
                await run_sync_adapter(workspace_port.release, workspace, status="failed")
            raise
        if sandbox_required and not sandbox.enforced:
            await run_sync_adapter(sandbox_port.release, sandbox, status="rejected")
            if workspace_port is not None and workspace is not None:
                await run_sync_adapter(workspace_port.release, workspace, status="rejected")
            raise SandboxUnavailable(
                f"tool {call.name} requires enforced isolation; backend {sandbox.backend!r} "
                "did not provide it"
            )

    scoped_context = context.fork(session=context.session)
    if workspace is not None:
        scoped_context.metadata["workspace"] = workspace.as_dict()
    if sandbox is not None:
        scoped_context.metadata["sandbox"] = sandbox.as_dict()
    return PreparedToolEnvironment(
        context=scoped_context,
        sandbox_port=sandbox_port,
        sandbox=sandbox,
        workspace_port=workspace_port,
        workspace=workspace,
    )


def _limits(policy: dict[str, Any]) -> SandboxLimits:
    raw = as_dict(policy.get("limits"))
    return SandboxLimits(
        wall_time_seconds=_optional_float(raw.get("wall_time_seconds")),
        cpu_time_seconds=_optional_float(raw.get("cpu_time_seconds")),
        memory_bytes=_optional_int(raw.get("memory_bytes")),
        output_bytes=_optional_int(raw.get("output_bytes")),
        file_count=_optional_int(raw.get("file_count")),
        process_count=_optional_int(raw.get("process_count")),
        network_access=bool(raw.get("network_access", False)),
    )


def _retention(value: Any) -> WorkspaceRetention:
    try:
        return WorkspaceRetention(str(value or WorkspaceRetention.EPHEMERAL.value))
    except ValueError as exc:
        raise ValueError(f"unsupported workspace retention: {value!r}") from exc


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


__all__ = ["PreparedToolEnvironment", "prepare_tool_environment"]
