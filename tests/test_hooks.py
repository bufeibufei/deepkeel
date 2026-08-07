from __future__ import annotations

import asyncio
import time

import pytest

from deepkeel.hooks import (
    HookAction,
    HookDecision,
    HookExecutionError,
    HookInvocation,
    HookPoint,
    HookRunner,
    HookScope,
    HookSpec,
)


def _invocation(
    point: HookPoint = HookPoint.TOOL_BEFORE,
    *,
    operation_id: str = "op-1",
) -> HookInvocation:
    return HookInvocation(
        point=point,
        operation_id=operation_id,
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        package_ids=("pack-a",),
        skill_id="skill-a",
        payload={"arguments": {"query": "before"}},
    )


def test_runner_orders_hooks_and_merges_typed_effects() -> None:
    calls: list[str] = []
    runner = HookRunner()
    runner.register(
        HookSpec(
            id="second",
            point=HookPoint.TOOL_BEFORE,
            priority=20,
            handler=lambda _: (
                calls.append("second")
                or HookDecision(context_patch={"b": 2}, tool_arguments={"query": "after"})
            ),
        )
    )
    runner.register(
        HookSpec(
            id="first",
            point=HookPoint.TOOL_BEFORE,
            priority=10,
            handler=lambda _: (
                calls.append("first") or HookDecision(context_patch={"a": 1})
            ),
        )
    )

    result = asyncio.run(runner.arun(_invocation()))

    assert calls == ["first", "second"]
    assert dict(result.decision.context_patch) == {"a": 1, "b": 2}
    assert dict(result.decision.tool_arguments or {}) == {"query": "after"}
    assert [audit.status for audit in result.audits] == ["completed", "completed"]


def test_runner_replays_successful_hook_by_operation_id() -> None:
    calls = 0

    def handler(_: HookInvocation) -> HookDecision:
        nonlocal calls
        calls += 1
        return HookDecision(diagnostics={"calls": calls})

    runner = HookRunner()
    runner.register(HookSpec(id="once", point=HookPoint.RUN_STARTED, handler=handler))

    first = asyncio.run(runner.arun(_invocation(HookPoint.RUN_STARTED)))
    replay = asyncio.run(runner.arun(_invocation(HookPoint.RUN_STARTED)))

    assert calls == 1
    assert first.audits[0].status == "completed"
    assert replay.audits[0].status == "replayed"
    assert replay.audits[0].replayed is True


def test_optional_hook_failure_is_isolated_but_required_failure_stops_run() -> None:
    optional = HookRunner()
    optional.register(
        HookSpec(
            id="optional",
            point=HookPoint.MODEL_BEFORE,
            handler=lambda _: (_ for _ in ()).throw(RuntimeError("optional failed")),
        )
    )
    result = asyncio.run(
        optional.arun(_invocation(HookPoint.MODEL_BEFORE))
    )
    assert result.decision.action == HookAction.CONTINUE
    assert result.audits[0].status == "failed"

    required = HookRunner()
    required.register(
        HookSpec(
            id="required",
            point=HookPoint.MODEL_BEFORE,
            required=True,
            handler=lambda _: (_ for _ in ()).throw(RuntimeError("required failed")),
        )
    )
    with pytest.raises(HookExecutionError, match="required failed"):
        asyncio.run(required.arun(_invocation(HookPoint.MODEL_BEFORE)))


def test_hook_scope_and_terminal_decision() -> None:
    calls: list[str] = []
    runner = HookRunner()
    runner.register(
        HookSpec(
            id="other-package",
            point=HookPoint.TOOL_BEFORE,
            scope=HookScope.PACKAGE,
            selector="pack-b",
            handler=lambda _: calls.append("wrong"),
        )
    )
    runner.register(
        HookSpec(
            id="matching-skill",
            point=HookPoint.TOOL_BEFORE,
            scope=HookScope.SKILL,
            selector="skill-a",
            handler=lambda _: HookDecision(
                action=HookAction.DENY,
                reason="blocked",
            ),
        )
    )
    runner.register(
        HookSpec(
            id="never-runs",
            point=HookPoint.TOOL_BEFORE,
            priority=200,
            handler=lambda _: calls.append("late"),
        )
    )

    result = asyncio.run(runner.arun(_invocation()))

    assert result.decision.action == HookAction.DENY
    assert result.decision.reason == "blocked"
    assert calls == []


def test_sync_hook_timeout_is_enforced_without_breaking_optional_run() -> None:
    def slow(_: HookInvocation) -> None:
        time.sleep(0.05)

    runner = HookRunner()
    runner.register(
        HookSpec(
            id="slow",
            point=HookPoint.RUN_SETTLED,
            handler=slow,
            timeout_seconds=0.01,
        )
    )

    result = asyncio.run(runner.arun(_invocation(HookPoint.RUN_SETTLED)))

    assert result.audits[0].status == "failed"
    assert result.decision.action == HookAction.CONTINUE
