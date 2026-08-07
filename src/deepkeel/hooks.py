from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Protocol


class HookPoint(StrEnum):
    RUN_STARTED = "run.started"
    TURN_STARTED = "turn.started"
    CONTEXT_PREPARED = "context.prepared"
    MODEL_BEFORE = "model.before"
    MODEL_AFTER = "model.after"
    TOOL_BEFORE = "tool.before"
    TOOL_AFTER = "tool.after"
    TOOL_FAILED = "tool.failed"
    RUN_SUSPENDING = "run.suspending"
    RUN_RESUMED = "run.resumed"
    ANSWER_BEFORE_FINALIZE = "answer.before_finalize"
    RUN_SETTLED = "run.settled"


class HookAction(StrEnum):
    CONTINUE = "continue"
    DENY = "deny"
    WAIT_FOR_CONFIRMATION = "wait_for_confirmation"


class HookScope(StrEnum):
    GLOBAL = "global"
    PACKAGE = "package"
    SKILL = "skill"
    RUN = "run"


@dataclass(frozen=True, slots=True)
class HookInvocation:
    point: HookPoint
    operation_id: str
    run_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    package_ids: tuple[str, ...] = ()
    skill_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation_id = self.operation_id.strip()
        if not operation_id:
            raise ValueError("hook operation_id must not be blank")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(
            self,
            "package_ids",
            tuple(dict.fromkeys(value.strip() for value in self.package_ids if value.strip())),
        )
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class HookDecision:
    action: HookAction = HookAction.CONTINUE
    reason: str = ""
    context_patch: Mapping[str, Any] = field(default_factory=dict)
    model_input_patch: Mapping[str, Any] = field(default_factory=dict)
    tool_arguments: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    confirmation_title: str = ""
    confirmation_message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_patch", MappingProxyType(dict(self.context_patch)))
        object.__setattr__(
            self,
            "model_input_patch",
            MappingProxyType(dict(self.model_input_patch)),
        )
        if self.tool_arguments is not None:
            object.__setattr__(
                self,
                "tool_arguments",
                MappingProxyType(dict(self.tool_arguments)),
            )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True, slots=True)
class HookAudit:
    hook_id: str
    point: HookPoint
    operation_id: str
    status: str
    duration_ms: float = 0.0
    replayed: bool = False
    required: bool = False
    error: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


HookHandler = Callable[
    [HookInvocation],
    HookDecision | Mapping[str, Any] | None | Awaitable[HookDecision | Mapping[str, Any] | None],
]
HookAuditSink = Callable[[HookAudit], None | Awaitable[None]]


@dataclass(frozen=True, slots=True)
class HookSpec:
    id: str
    point: HookPoint
    handler: HookHandler
    priority: int = 100
    scope: HookScope = HookScope.GLOBAL
    selector: str = ""
    timeout_seconds: float = 2.0
    required: bool = False

    def __post_init__(self) -> None:
        hook_id = self.id.strip()
        if not hook_id:
            raise ValueError("hook id must not be blank")
        if not callable(self.handler):
            raise TypeError("hook handler must be callable")
        if self.timeout_seconds <= 0:
            raise ValueError("hook timeout_seconds must be positive")
        selector = self.selector.strip()
        if self.scope != HookScope.GLOBAL and not selector:
            raise ValueError(f"{self.scope.value} hook must declare selector")
        object.__setattr__(self, "id", hook_id)
        object.__setattr__(self, "selector", selector)


@dataclass(frozen=True, slots=True)
class HookRunResult:
    decision: HookDecision
    audits: tuple[HookAudit, ...] = ()


class HookExecutionStore(Protocol):
    def get(
        self,
        *,
        hook_id: str,
        point: HookPoint,
        operation_id: str,
    ) -> HookDecision | None: ...

    def put(
        self,
        *,
        hook_id: str,
        point: HookPoint,
        operation_id: str,
        decision: HookDecision,
    ) -> None: ...


class InMemoryHookExecutionStore:
    def __init__(self) -> None:
        self._values: dict[tuple[str, HookPoint, str], HookDecision] = {}
        self._lock = RLock()

    def get(
        self,
        *,
        hook_id: str,
        point: HookPoint,
        operation_id: str,
    ) -> HookDecision | None:
        with self._lock:
            return self._values.get((hook_id, point, operation_id))

    def put(
        self,
        *,
        hook_id: str,
        point: HookPoint,
        operation_id: str,
        decision: HookDecision,
    ) -> None:
        with self._lock:
            self._values[(hook_id, point, operation_id)] = decision


class HookExecutionError(RuntimeError):
    def __init__(self, *, hook_id: str, point: HookPoint, message: str) -> None:
        super().__init__(f"required hook {hook_id} failed at {point.value}: {message}")
        self.hook_id = hook_id
        self.point = point


class HookRunner:
    """Runs scoped lifecycle hooks with timeout, replay safety, and audit isolation."""

    def __init__(
        self,
        *,
        store: HookExecutionStore | None = None,
        audit_sink: HookAuditSink | None = None,
    ) -> None:
        self._hooks: dict[str, HookSpec] = {}
        self._store = store or InMemoryHookExecutionStore()
        self._audit_sink = audit_sink
        self._lock = RLock()

    @property
    def registered_hooks(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._hooks)

    def register(self, spec: HookSpec) -> None:
        with self._lock:
            if spec.id in self._hooks:
                raise ValueError(f"hook is already registered: {spec.id}")
            self._hooks[spec.id] = spec

    def unregister(self, hook_id: str) -> None:
        with self._lock:
            self._hooks.pop(hook_id, None)

    def snapshot(self) -> dict[str, HookSpec]:
        with self._lock:
            return dict(self._hooks)

    def restore(self, snapshot: Mapping[str, HookSpec]) -> None:
        with self._lock:
            self._hooks = dict(snapshot)

    async def arun(self, invocation: HookInvocation) -> HookRunResult:
        with self._lock:
            hooks = sorted(
                (
                    spec
                    for spec in self._hooks.values()
                    if spec.point == invocation.point and _matches_scope(spec, invocation)
                ),
                key=lambda spec: (spec.priority, spec.id),
            )
        combined = HookDecision()
        audits: list[HookAudit] = []
        for spec in hooks:
            cached = self._store.get(
                hook_id=spec.id,
                point=spec.point,
                operation_id=invocation.operation_id,
            )
            if cached is not None:
                audit = HookAudit(
                    hook_id=spec.id,
                    point=spec.point,
                    operation_id=invocation.operation_id,
                    status="replayed",
                    replayed=True,
                    required=spec.required,
                    diagnostics=cached.diagnostics,
                )
                audits.append(audit)
                await self._publish_audit(audit)
                combined = _merge_decisions(combined, cached)
                if combined.action != HookAction.CONTINUE:
                    break
                continue

            started = time.perf_counter()
            try:
                if inspect.iscoroutinefunction(spec.handler):
                    result = await asyncio.wait_for(
                        spec.handler(invocation),
                        timeout=spec.timeout_seconds,
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(spec.handler, invocation),
                        timeout=spec.timeout_seconds,
                    )
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, timeout=spec.timeout_seconds)
                decision = _coerce_decision(result)
                self._store.put(
                    hook_id=spec.id,
                    point=spec.point,
                    operation_id=invocation.operation_id,
                    decision=decision,
                )
                audit = HookAudit(
                    hook_id=spec.id,
                    point=spec.point,
                    operation_id=invocation.operation_id,
                    status="completed",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    required=spec.required,
                    diagnostics=decision.diagnostics,
                )
                audits.append(audit)
                await self._publish_audit(audit)
                combined = _merge_decisions(combined, decision)
                if combined.action != HookAction.CONTINUE:
                    break
            except Exception as exc:
                audit = HookAudit(
                    hook_id=spec.id,
                    point=spec.point,
                    operation_id=invocation.operation_id,
                    status="failed",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    required=spec.required,
                    error=str(exc),
                )
                audits.append(audit)
                await self._publish_audit(audit)
                if spec.required:
                    raise HookExecutionError(
                        hook_id=spec.id,
                        point=spec.point,
                        message=str(exc),
                    ) from exc
        return HookRunResult(decision=combined, audits=tuple(audits))

    def run(self, invocation: HookInvocation) -> HookRunResult:
        return asyncio.run(self.arun(invocation))

    async def _publish_audit(self, audit: HookAudit) -> None:
        if self._audit_sink is None:
            return
        result = self._audit_sink(audit)
        if inspect.isawaitable(result):
            await result


def _matches_scope(spec: HookSpec, invocation: HookInvocation) -> bool:
    if spec.scope == HookScope.GLOBAL:
        return True
    if spec.scope == HookScope.PACKAGE:
        return spec.selector in invocation.package_ids
    if spec.scope == HookScope.SKILL:
        return spec.selector == invocation.skill_id
    if spec.scope == HookScope.RUN:
        return spec.selector == invocation.run_id
    return False


def _coerce_decision(value: HookDecision | Mapping[str, Any] | None) -> HookDecision:
    if value is None:
        return HookDecision()
    if isinstance(value, HookDecision):
        return value
    if isinstance(value, Mapping):
        return HookDecision(**dict(value))
    raise TypeError("hook handler must return HookDecision, mapping, or None")


def _merge_decisions(current: HookDecision, incoming: HookDecision) -> HookDecision:
    context_patch = dict(current.context_patch)
    context_patch.update(incoming.context_patch)
    model_input_patch = dict(current.model_input_patch)
    model_input_patch.update(incoming.model_input_patch)
    diagnostics = dict(current.diagnostics)
    diagnostics.update(incoming.diagnostics)
    action = (
        incoming.action
        if incoming.action != HookAction.CONTINUE
        else current.action
    )
    return HookDecision(
        action=action,
        reason=incoming.reason or current.reason,
        context_patch=context_patch,
        model_input_patch=model_input_patch,
        tool_arguments=(
            incoming.tool_arguments
            if incoming.tool_arguments is not None
            else current.tool_arguments
        ),
        diagnostics=diagnostics,
        confirmation_title=(
            incoming.confirmation_title or current.confirmation_title
        ),
        confirmation_message=(
            incoming.confirmation_message or current.confirmation_message
        ),
    )


__all__ = [
    "HookAction",
    "HookAudit",
    "HookDecision",
    "HookExecutionError",
    "HookExecutionStore",
    "HookInvocation",
    "HookPoint",
    "HookRunResult",
    "HookRunner",
    "HookScope",
    "HookSpec",
    "InMemoryHookExecutionStore",
]
