from __future__ import annotations

import ast
from pathlib import Path

import pytest

from deepkeel.contracts import AgentMessage, RunContext, RunStatus, ToolCall
from deepkeel.control import InMemoryRunControl
from deepkeel.failures import RunCanceledError
from deepkeel.graph_nodes import GraphNodes
from deepkeel.graph_model_step import (
    build_model_metrics,
    build_model_step_context,
    partition_model_tool_calls,
)
from deepkeel.graph_state import _state_from_context
from deepkeel.graph_workflow import (
    _route_after_model,
    _route_after_tools,
    _route_from_start,
)
from deepkeel.runtime_api import RuntimeResultStatus
from deepkeel.runtime_policy import _budget_limits, _resolved_model_policy
from deepkeel.runtime_results import project_harness_result
from deepkeel.model import (
    InMemoryModelInvocationStore as PublicModelInvocationStore,
    ModelInvocation as PublicModelInvocation,
)
from deepkeel.model_invocations import (
    InMemoryModelInvocationStore,
    ModelInvocation,
    ModelTurn,
)
from deepkeel.tool_execution import (
    InMemoryToolExecutionStore,
    ToolExecutionContext,
)
from deepkeel.tools import (
    InMemoryToolExecutionStore as PublicToolExecutionStore,
    ToolExecutionContext as PublicToolExecutionContext,
)
from deepkeel.subagents.contracts import DelegationTask
from deepkeel.subagents.execution_support import _child_run_id
from deepkeel.subagents.execution_types import _DelegationQuota
from deepkeel.subagents.output_validation import (
    _confidence,
    _json_object,
    _validate_input,
)


class _Provider:
    model_role = "fast"


def test_split_model_and_tool_execution_contracts_keep_public_imports_stable() -> None:
    assert PublicModelInvocation is ModelInvocation
    assert PublicModelInvocationStore is InMemoryModelInvocationStore
    assert PublicToolExecutionContext is ToolExecutionContext
    assert PublicToolExecutionStore is InMemoryToolExecutionStore


def test_runtime_policy_resolves_provider_roles_and_budget_limits() -> None:
    policy = _resolved_model_policy(
        {"budget": {"max_model_calls": 3, "max_parallel_tools": 2}},
        provider=_Provider(),
        providers=None,
        max_steps=12,
    )

    assert policy["mode"] == "adaptive"
    assert policy["primary_role"] == "reasoning"
    assert policy["available_roles"] == ["fast", "reasoning"]
    assert _budget_limits(policy)["model_calls"] == 3
    assert _budget_limits(policy)["tool_concurrency"] == 2


def test_runtime_results_projects_a_completed_typed_result() -> None:
    state = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "status": "completed",
        "step_count": 1,
        "messages": [],
        "observations": [],
        "artifacts": [],
        "tool_results": [],
        "pending_tool_calls": [],
        "final_answer": {
            "markdown": "The task is complete.",
            "summary": "Complete",
            "status": "completed",
        },
        "metadata": {},
    }

    result = project_harness_result(
        state,
        question="Complete the task",
        context_bundle={},
        short_context={},
        skill_activation={},
        streamed_events=[],
        user_id="user-1",
    )

    assert result.status is RuntimeResultStatus.COMPLETED
    assert result.final_answer.markdown == "The task is complete."
    assert result.checkpoint["schema_version"] == "harness-checkpoint-v2"
    assert result.run_context is not None
    assert result.run_context.run_id == result.run_id == "run-1"
    assert result.run_context.thread_id == result.thread_id == "thread-1"
    assert result.run_context.turn_id == result.turn_id == "turn-1"
    assert result.run_context.user_id == "user-1"


def test_graph_state_serializes_typed_context_without_losing_identity() -> None:
    context = RunContext(
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        user_id="user-1",
        status=RunStatus.PREPARING,
        messages=[AgentMessage(id="message-1", role="user", content="hello")],
        skill_activation={"kind": "workflow"},
    )

    state = _state_from_context(context)

    assert state["run_id"] == "run-1"
    assert state["messages"][0]["content"] == "hello"
    assert state["policy_phase"] == "pending"
    assert state["tool_results"] == []


@pytest.mark.parametrize(
    ("router", "state", "expected"),
    [
        (_route_from_start, {"pending_tool_calls": [{"id": "call-1"}]}, "tools"),
        (_route_from_start, {"pending_tool_calls": []}, "model"),
        (_route_after_tools, {"status": "waiting_user"}, "await_user"),
        (_route_after_tools, {"status": "waiting_async"}, "await_async"),
        (_route_after_model, {"pending_tool_calls": [{"id": "call-1"}]}, "tools"),
        (_route_after_model, {"status": "completed"}, "end"),
    ],
)
def test_graph_workflow_routes_each_runtime_phase(router, state, expected) -> None:
    assert router(state) == expected


def test_graph_nodes_honors_cooperative_cancellation() -> None:
    control = InMemoryRunControl()
    control.cancel("run-1")
    nodes = object.__new__(GraphNodes)
    nodes.control = control
    nodes.deadline_monotonic = None

    with pytest.raises(RunCanceledError):
        nodes.ensure_active({"run_id": "run-1"})


def test_graph_model_step_helpers_preserve_context_metrics_and_disclosure() -> None:
    state = {
        "run_id": "run-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "messages": [{"role": "user", "content": "hello"}],
        "observations": [{"source": "tool-a"}],
        "tool_results": [{"name": "tool-a"}],
        "model_policy": {"mode": "adaptive"},
        "metadata": {"governance_scope": {"tenant_id": "tenant-1"}},
    }
    context = build_model_step_context(
        state,
        available_roles=("fast", "reasoning"),
        forced_tool_name="tool-a",
        deadline_monotonic=123.0,
    )
    assert context.available_roles == ("fast", "reasoning")
    assert context.observation_sources == ("tool-a",)
    assert context.governance_scope == {"tenant_id": "tenant-1"}

    accepted, rejected = partition_model_tool_calls(
        [
            ToolCall(id="allowed", name="tool-a", arguments={}),
            ToolCall(id="hidden", name="tool-b", arguments={}),
        ],
        workflow_is_finalizing=True,
        exposed_tool_names={"tool-a"},
        registered_tool_names={"tool-a", "tool-b"},
    )
    assert [call.name for call in accepted] == ["tool-a"]
    assert [call.name for call in rejected] == ["tool-b"]

    metrics = build_model_metrics(
        ModelTurn(content="ready", model_id="m1", model_role="fast"),
        {"usage": {"input_tokens": 4}, "budget_metrics": {"model_calls": {}}},
        latency_ms=50,
        first_token_latency_ms=10,
        delta_count=2,
        delta_chars=5,
        forced_tool_name="",
    )
    assert metrics["content_chars"] == 5
    assert metrics["usage"] == {"input_tokens": 4}


@pytest.mark.parametrize(
    ("path", "class_name", "method_name", "maximum_lines"),
    [
        ("src/deepkeel/model_gateway.py", "RoutedModelGateway", "arun_turn", 400),
        ("src/deepkeel/graph_model_node.py", "GraphModelNodeMixin", "amodel_node", 620),
        (
            "src/deepkeel/subagents/batch_execution.py",
            "SubAgentBatchExecutionMixin",
            "execute_many",
            300,
        ),
        ("src/deepkeel/subagents/executor.py", "SubAgentExecutor", "_execute_one", 300),
    ],
)
def test_central_execution_methods_stay_within_ratcheted_size_budget(
    path: str,
    class_name: str,
    method_name: str,
    maximum_lines: int,
) -> None:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    target = next(
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name == method_name
    )
    assert target.end_lineno is not None
    assert target.end_lineno - target.lineno + 1 <= maximum_lines


@pytest.mark.parametrize(
    ("path", "maximum_lines"),
    [
        ("src/deepkeel/runtime.py", 875),
        ("src/deepkeel/runtime_results.py", 900),
        ("src/deepkeel/graph_state.py", 800),
        ("src/deepkeel/composition.py", 800),
        ("src/deepkeel/model.py", 250),
        ("src/deepkeel/tools.py", 80),
        ("src/deepkeel/graph_nodes.py", 300),
        ("src/deepkeel/subagents/executor.py", 700),
        ("src/deepkeel/subagents/batch_execution.py", 400),
        ("src/deepkeel/context_window.py", 800),
        ("src/deepkeel/runtime_turn_execution.py", 750),
        ("src/deepkeel/model_gateway.py", 750),
        ("src/deepkeel/tool_executor.py", 750),
        ("src/deepkeel/graph_model_node.py", 750),
        ("src/deepkeel/subagents/bounded_execution.py", 750),
    ],
)
def test_execution_modules_stay_within_ratcheted_size_budget(
    path: str,
    maximum_lines: int,
) -> None:
    assert len(Path(path).read_text(encoding="utf-8").splitlines()) <= maximum_lines


def test_internal_module_dependency_graph_is_acyclic() -> None:
    root = Path("src/deepkeel")
    modules = {
        ".".join(path.relative_to("src").with_suffix("").parts): path
        for path in root.rglob("*.py")
    }
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = (
                    [node.module]
                    if not node.level
                    else [
                        ".".join(
                            module.split(".")[:-node.level] + [node.module]
                        )
                    ]
                )
            for imported_module in imported:
                candidate = imported_module
                while candidate and candidate not in modules:
                    candidate = candidate.rpartition(".")[0]
                if candidate in modules and candidate != module:
                    graph[module].add(candidate)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, path: tuple[str, ...]) -> None:
        if module in visiting:
            cycle_start = path.index(module)
            pytest.fail("internal dependency cycle: " + " -> ".join(path[cycle_start:]))
        if module in visited:
            return
        visiting.add(module)
        for dependency in sorted(graph[module]):
            visit(dependency, (*path, dependency))
        visiting.remove(module)
        visited.add(module)

    for module in sorted(graph):
        visit(module, (module,))


def test_subagent_quota_reserves_calls_atomically() -> None:
    quota = _DelegationQuota(max_model_calls=1, max_tool_calls=2)

    quota.reserve_model_call()
    quota.reserve_tool_calls(2)

    with pytest.raises(RuntimeError, match="model call budget"):
        quota.reserve_model_call()
    with pytest.raises(RuntimeError, match="tool call budget"):
        quota.reserve_tool_calls(1)


def test_subagent_output_validation_normalizes_structured_output() -> None:
    assert _json_object('```json\n{"conclusion":"ready"}\n```') == {"conclusion": "ready"}
    assert _confidence("0.75") == 0.75
    with pytest.raises(RuntimeError, match="missing required field"):
        _validate_input({}, {"required": ["record_id"]})


def test_subagent_child_run_id_is_stable_and_task_scoped() -> None:
    first = DelegationTask(id="task-1", agent_id="reviewer", objective="Review")
    second = DelegationTask(id="task-2", agent_id="reviewer", objective="Review")

    assert _child_run_id("parent", "delegation", first) == _child_run_id(
        "parent", "delegation", first
    )
    assert _child_run_id("parent", "delegation", first) != _child_run_id(
        "parent", "delegation", second
    )
