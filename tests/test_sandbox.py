from __future__ import annotations

from pathlib import Path

from deepkeel.adapter_sdk import (
    LocalWorkspacePort,
    NoopSandboxPort,
    SandboxLease,
    SandboxRequest,
)
from deepkeel.composition import HarnessRuntimeBuilder, RuntimePorts
from deepkeel.contracts import ToolCall, ToolResult
from deepkeel.tool_execution import ToolExecutionContext
from deepkeel.tool_executor import ToolExecutor
from deepkeel.tool_registry import ToolRegistry, ToolSpec


class RecordingSandboxPort:
    def __init__(self) -> None:
        self.requests: list[SandboxRequest] = []
        self.releases: list[tuple[str, str]] = []

    def acquire(self, request: SandboxRequest) -> SandboxLease:
        self.requests.append(request)
        return SandboxLease(
            sandbox_id=f"sandbox-{request.tool_call.id}",
            backend="test-isolator",
            enforced=True,
            workspace=request.workspace,
        )

    def release(self, lease: SandboxLease, *, status: str) -> None:
        self.releases.append((lease.sandbox_id, status))


class FailingWorkspacePort:
    def allocate(self, request):
        del request
        from deepkeel.adapter_sdk import WorkspaceLease

        return WorkspaceLease(workspace_id="cleanup-failure")

    def release(self, lease, *, status: str) -> None:
        del lease, status
        raise RuntimeError("cleanup unavailable")


def _tool(*, sandbox: dict | None = None) -> ToolSpec:
    return ToolSpec(
        name="demo.code",
        parameters_schema={"type": "object", "properties": {}},
        runtime_policy={"sandbox": sandbox or {}},
    )


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        run_id="run-sandbox",
        user_id="user-sandbox",
        thread_id="thread-sandbox",
        turn_id="turn-sandbox",
    )


def _call() -> ToolCall:
    return ToolCall(id="call-sandbox", name="demo.code", arguments={})


def test_required_sandbox_fails_closed_before_handler_runs() -> None:
    registry = ToolRegistry([_tool(sandbox={"required": True})])
    executor = ToolExecutor(registry)
    called = False

    def handler(call, context):
        nonlocal called
        del context
        called = True
        return ToolResult(call=call, status="succeeded")

    executor.register("demo.code", handler)
    result = executor.execute(_call(), _context())

    assert result.status == "failed"
    assert result.retryable is False
    assert result.metadata["error_code"] == "SANDBOX_UNAVAILABLE"
    assert called is False


def test_required_sandbox_rejects_non_enforcing_adapter() -> None:
    registry = ToolRegistry([_tool(sandbox={"required": True})])
    executor = ToolExecutor(registry, sandbox_port=NoopSandboxPort())
    executor.register(
        "demo.code",
        lambda call, context: ToolResult(call=call, status="succeeded"),
    )

    result = executor.execute(_call(), _context())

    assert result.status == "failed"
    assert result.metadata["error_code"] == "SANDBOX_UNAVAILABLE"


def test_sandbox_and_ephemeral_workspace_are_injected_and_cleaned(tmp_path: Path) -> None:
    sandbox = RecordingSandboxPort()
    workspace = LocalWorkspacePort(tmp_path)
    registry = ToolRegistry(
        [
            _tool(
                sandbox={
                    "enabled": True,
                    "required": True,
                    "workspace": True,
                    "workspace_required": True,
                    "limits": {
                        "wall_time_seconds": 15,
                        "memory_bytes": 1024,
                        "network_access": False,
                    },
                }
            )
        ]
    )
    executor = ToolExecutor(
        registry,
        sandbox_port=sandbox,
        workspace_port=workspace,
    )
    observed_path: Path | None = None

    def handler(call, context):
        nonlocal observed_path
        observed_path = Path(context.metadata["workspace"]["root_path"])
        assert observed_path.exists()
        assert context.metadata["sandbox"]["backend"] == "test-isolator"
        (observed_path / "result.txt").write_text("isolated", encoding="utf-8")
        return ToolResult(call=call, status="succeeded", summary="done")

    executor.register("demo.code", handler)
    result = executor.execute(_call(), _context())

    assert result.status == "succeeded"
    assert observed_path is not None and not observed_path.exists()
    assert sandbox.requests[0].limits.wall_time_seconds == 15
    assert sandbox.requests[0].limits.memory_bytes == 1024
    assert sandbox.requests[0].limits.network_access is False
    assert sandbox.releases == [("sandbox-call-sandbox", "succeeded")]
    assert result.metadata["execution_environment"]["sandbox"]["enforced"] is True


def test_configured_ports_do_not_change_tools_without_sandbox_policy(tmp_path: Path) -> None:
    sandbox = RecordingSandboxPort()
    registry = ToolRegistry([_tool()])
    executor = ToolExecutor(
        registry,
        sandbox_port=sandbox,
        workspace_port=LocalWorkspacePort(tmp_path),
    )

    def handler(call, context):
        assert "sandbox" not in context.metadata
        assert "workspace" not in context.metadata
        return ToolResult(call=call, status="succeeded")

    executor.register("demo.code", handler)
    result = executor.execute(_call(), _context())

    assert result.status == "succeeded"
    assert sandbox.requests == []


def test_cleanup_failure_is_reported_as_retryable_environment_failure() -> None:
    registry = ToolRegistry([_tool(sandbox={"workspace": True})])
    executor = ToolExecutor(registry, workspace_port=FailingWorkspacePort())
    executor.register(
        "demo.code",
        lambda call, context: ToolResult(call=call, status="succeeded"),
    )

    result = executor.execute(_call(), _context())

    assert result.status == "failed"
    assert result.retryable is True
    assert result.metadata["error_code"] == "SANDBOX_CLEANUP_FAILED"


def test_builder_wires_sandbox_and_workspace_ports_into_executor(tmp_path: Path) -> None:
    sandbox = RecordingSandboxPort()
    workspace = LocalWorkspacePort(tmp_path)

    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(RuntimePorts(sandbox_port=sandbox, workspace_port=workspace))
        .build()
    )

    assert runtime.tool_executor.sandbox_port is sandbox
    assert runtime.tool_executor.workspace_port is workspace
