from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Protocol

from deepkeel.contracts import DataProvenance


class GuardrailStage(StrEnum):
    """Semantic boundaries where untrusted data may enter or leave the runtime."""

    INPUT = "input"
    MODEL_INPUT = "model.input"
    MODEL_OUTPUT = "model.output"
    TOOL_INPUT = "tool.input"
    TOOL_OUTPUT = "tool.output"
    ARTIFACT_OUTPUT = "artifact.output"
    FINAL_OUTPUT = "final.output"


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    TRANSFORM = "transform"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


class GuardrailScope(StrEnum):
    GLOBAL = "global"
    PACKAGE = "package"
    SKILL = "skill"
    TOOL = "tool"
    RUN = "run"


@dataclass(frozen=True, slots=True)
class GuardrailRequest:
    stage: GuardrailStage
    operation_id: str
    run_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    package_ids: tuple[str, ...] = ()
    skill_id: str = ""
    tool_name: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[DataProvenance, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation_id = self.operation_id.strip()
        if not operation_id:
            raise ValueError("guardrail operation_id must not be blank")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(
            self,
            "package_ids",
            tuple(dict.fromkeys(value.strip() for value in self.package_ids if value.strip())),
        )
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    action: GuardrailAction = GuardrailAction.ALLOW
    reason: str = ""
    code: str = ""
    payload_patch: Mapping[str, Any] = field(default_factory=dict)
    redactions: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    approval_title: str = ""
    approval_prompt: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_patch", MappingProxyType(dict(self.payload_patch)))
        object.__setattr__(
            self,
            "redactions",
            tuple(dict.fromkeys(value.strip() for value in self.redactions if value.strip())),
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def terminal(self) -> bool:
        return self.action in {GuardrailAction.BLOCK, GuardrailAction.REQUIRE_APPROVAL}

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "code": self.code,
            "payload_patch": dict(self.payload_patch),
            "redactions": list(self.redactions),
            "diagnostics": dict(self.diagnostics),
            "approval_title": self.approval_title,
            "approval_prompt": self.approval_prompt,
        }


@dataclass(frozen=True, slots=True)
class GuardrailAudit:
    guardrail_id: str
    stage: GuardrailStage
    operation_id: str
    status: str
    action: GuardrailAction = GuardrailAction.ALLOW
    duration_ms: float = 0.0
    replayed: bool = False
    required: bool = True
    reason: str = ""
    error: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


GuardrailHandler = Callable[
    [GuardrailRequest],
    GuardrailDecision
    | Mapping[str, Any]
    | None
    | Awaitable[GuardrailDecision | Mapping[str, Any] | None],
]
GuardrailAuditSink = Callable[[GuardrailAudit], None | Awaitable[None]]


@dataclass(frozen=True, slots=True)
class GuardrailSpec:
    id: str
    stage: GuardrailStage
    handler: GuardrailHandler
    priority: int = 100
    scope: GuardrailScope = GuardrailScope.GLOBAL
    selector: str = ""
    timeout_seconds: float = 2.0
    required: bool = True

    def __post_init__(self) -> None:
        guardrail_id = self.id.strip()
        if not guardrail_id:
            raise ValueError("guardrail id must not be blank")
        if not callable(self.handler):
            raise TypeError("guardrail handler must be callable")
        if self.timeout_seconds <= 0:
            raise ValueError("guardrail timeout_seconds must be positive")
        selector = self.selector.strip()
        if self.scope != GuardrailScope.GLOBAL and not selector:
            raise ValueError(f"{self.scope.value} guardrail must declare selector")
        object.__setattr__(self, "id", guardrail_id)
        object.__setattr__(self, "selector", selector)


@dataclass(frozen=True, slots=True)
class GuardrailRunResult:
    decision: GuardrailDecision
    audits: tuple[GuardrailAudit, ...] = ()


class GuardrailExecutionStore(Protocol):
    def get(
        self,
        *,
        guardrail_id: str,
        stage: GuardrailStage,
        operation_id: str,
    ) -> GuardrailDecision | None: ...

    def put(
        self,
        *,
        guardrail_id: str,
        stage: GuardrailStage,
        operation_id: str,
        decision: GuardrailDecision,
    ) -> None: ...


class InMemoryGuardrailExecutionStore:
    def __init__(self) -> None:
        self._values: dict[tuple[str, GuardrailStage, str], GuardrailDecision] = {}
        self._lock = RLock()

    def get(
        self,
        *,
        guardrail_id: str,
        stage: GuardrailStage,
        operation_id: str,
    ) -> GuardrailDecision | None:
        with self._lock:
            return self._values.get((guardrail_id, stage, operation_id))

    def put(
        self,
        *,
        guardrail_id: str,
        stage: GuardrailStage,
        operation_id: str,
        decision: GuardrailDecision,
    ) -> None:
        with self._lock:
            self._values[(guardrail_id, stage, operation_id)] = decision


class GuardrailExecutionError(RuntimeError):
    def __init__(self, *, guardrail_id: str, stage: GuardrailStage, message: str) -> None:
        super().__init__(f"required guardrail {guardrail_id} failed at {stage.value}: {message}")
        self.guardrail_id = guardrail_id
        self.stage = stage


class GuardrailRunner:
    """Runs ordered trust decisions with replay safety and fail-closed defaults."""

    def __init__(
        self,
        *,
        store: GuardrailExecutionStore | None = None,
        audit_sink: GuardrailAuditSink | None = None,
    ) -> None:
        self._guardrails: dict[str, GuardrailSpec] = {}
        self._store = store or InMemoryGuardrailExecutionStore()
        self._audit_sink = audit_sink
        self._lock = RLock()

    @property
    def registered_guardrails(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._guardrails)

    @property
    def execution_store(self) -> GuardrailExecutionStore:
        return self._store

    def has_stage(self, stage: GuardrailStage) -> bool:
        with self._lock:
            return any(spec.stage == stage for spec in self._guardrails.values())

    def register(self, spec: GuardrailSpec) -> None:
        with self._lock:
            if spec.id in self._guardrails:
                raise ValueError(f"guardrail is already registered: {spec.id}")
            self._guardrails[spec.id] = spec

    def unregister(self, guardrail_id: str) -> None:
        with self._lock:
            self._guardrails.pop(guardrail_id, None)

    def snapshot(self) -> dict[str, GuardrailSpec]:
        with self._lock:
            return dict(self._guardrails)

    def restore(self, snapshot: Mapping[str, GuardrailSpec]) -> None:
        with self._lock:
            self._guardrails = dict(snapshot)

    async def arun(self, request: GuardrailRequest) -> GuardrailRunResult:
        with self._lock:
            guardrails = sorted(
                (
                    spec
                    for spec in self._guardrails.values()
                    if spec.stage == request.stage and _matches_scope(spec, request)
                ),
                key=lambda spec: (spec.priority, spec.id),
            )
        combined = GuardrailDecision()
        audits: list[GuardrailAudit] = []
        for spec in guardrails:
            cached = self._store.get(
                guardrail_id=spec.id,
                stage=spec.stage,
                operation_id=request.operation_id,
            )
            if cached is not None:
                audit = _audit(
                    spec,
                    request,
                    status="replayed",
                    decision=cached,
                    replayed=True,
                )
                audits.append(audit)
                await self._publish_audit(audit)
                combined = _merge_decisions(combined, cached)
                if combined.terminal:
                    break
                continue

            started = time.perf_counter()
            try:
                if inspect.iscoroutinefunction(spec.handler):
                    value = await asyncio.wait_for(
                        spec.handler(request), timeout=spec.timeout_seconds
                    )
                else:
                    value = await asyncio.wait_for(
                        asyncio.to_thread(spec.handler, request),
                        timeout=spec.timeout_seconds,
                    )
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=spec.timeout_seconds)
                decision = _coerce_decision(value)
                self._store.put(
                    guardrail_id=spec.id,
                    stage=spec.stage,
                    operation_id=request.operation_id,
                    decision=decision,
                )
                audit = _audit(
                    spec,
                    request,
                    status="completed",
                    decision=decision,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                audits.append(audit)
                await self._publish_audit(audit)
                combined = _merge_decisions(combined, decision)
                if combined.terminal:
                    break
            except Exception as exc:
                audit = GuardrailAudit(
                    guardrail_id=spec.id,
                    stage=spec.stage,
                    operation_id=request.operation_id,
                    status="failed",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    required=spec.required,
                    error=str(exc),
                )
                audits.append(audit)
                await self._publish_audit(audit)
                if spec.required:
                    raise GuardrailExecutionError(
                        guardrail_id=spec.id,
                        stage=spec.stage,
                        message=str(exc),
                    ) from exc
        return GuardrailRunResult(decision=combined, audits=tuple(audits))

    def run(self, request: GuardrailRequest) -> GuardrailRunResult:
        return asyncio.run(self.arun(request))

    async def _publish_audit(self, audit: GuardrailAudit) -> None:
        if self._audit_sink is None:
            return
        result = self._audit_sink(audit)
        if inspect.isawaitable(result):
            await result


def _matches_scope(spec: GuardrailSpec, request: GuardrailRequest) -> bool:
    if spec.scope == GuardrailScope.GLOBAL:
        return True
    if spec.scope == GuardrailScope.PACKAGE:
        return spec.selector in request.package_ids
    if spec.scope == GuardrailScope.SKILL:
        return spec.selector == request.skill_id
    if spec.scope == GuardrailScope.TOOL:
        return spec.selector == request.tool_name
    if spec.scope == GuardrailScope.RUN:
        return spec.selector == request.run_id
    return False


def _coerce_decision(
    value: GuardrailDecision | Mapping[str, Any] | None,
) -> GuardrailDecision:
    if value is None:
        return GuardrailDecision()
    if isinstance(value, GuardrailDecision):
        return value
    if isinstance(value, Mapping):
        return GuardrailDecision(**dict(value))
    raise TypeError("guardrail handler must return GuardrailDecision, mapping, or None")


def _merge_decisions(
    current: GuardrailDecision,
    incoming: GuardrailDecision,
) -> GuardrailDecision:
    payload_patch = dict(current.payload_patch)
    payload_patch.update(incoming.payload_patch)
    diagnostics = dict(current.diagnostics)
    diagnostics.update(incoming.diagnostics)
    action = _stronger_action(current.action, incoming.action)
    return GuardrailDecision(
        action=action,
        reason=incoming.reason or current.reason,
        code=incoming.code or current.code,
        payload_patch=payload_patch,
        redactions=tuple((*current.redactions, *incoming.redactions)),
        diagnostics=diagnostics,
        approval_title=incoming.approval_title or current.approval_title,
        approval_prompt=incoming.approval_prompt or current.approval_prompt,
    )


def _stronger_action(
    current: GuardrailAction,
    incoming: GuardrailAction,
) -> GuardrailAction:
    rank = {
        GuardrailAction.ALLOW: 0,
        GuardrailAction.TRANSFORM: 1,
        GuardrailAction.REQUIRE_APPROVAL: 2,
        GuardrailAction.BLOCK: 3,
    }
    return incoming if rank[incoming] >= rank[current] else current


def _audit(
    spec: GuardrailSpec,
    request: GuardrailRequest,
    *,
    status: str,
    decision: GuardrailDecision,
    duration_ms: float = 0.0,
    replayed: bool = False,
) -> GuardrailAudit:
    return GuardrailAudit(
        guardrail_id=spec.id,
        stage=request.stage,
        operation_id=request.operation_id,
        status=status,
        action=decision.action,
        duration_ms=duration_ms,
        replayed=replayed,
        required=spec.required,
        reason=decision.reason,
        diagnostics=decision.diagnostics,
    )


__all__ = [
    "GuardrailAction",
    "GuardrailAudit",
    "GuardrailAuditSink",
    "GuardrailDecision",
    "GuardrailExecutionError",
    "GuardrailExecutionStore",
    "GuardrailHandler",
    "GuardrailRequest",
    "GuardrailRunResult",
    "GuardrailRunner",
    "GuardrailScope",
    "GuardrailSpec",
    "GuardrailStage",
    "InMemoryGuardrailExecutionStore",
]
