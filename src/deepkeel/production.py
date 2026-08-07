from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


class ProductionRuntimePorts(Protocol):
    """Minimal structural view required by the production-readiness gate."""

    @property
    def checkpointer(self) -> Any: ...

    @property
    def checkpoint_store(self) -> Any: ...

    @property
    def async_checkpoint_store(self) -> Any: ...

    @property
    def runtime_state_store(self) -> Any: ...

    @property
    def async_runtime_state_store(self) -> Any: ...

    @property
    def event_journal(self) -> Any: ...

    @property
    def async_event_journal(self) -> Any: ...

    @property
    def run_lease_store(self) -> Any: ...

    @property
    def async_run_lease_store(self) -> Any: ...

    @property
    def model_invocation_store(self) -> Any: ...

    @property
    def tool_execution_store(self) -> Any: ...

    @property
    def async_tool_execution_store(self) -> Any: ...

    @property
    def budget_ledger(self) -> Any: ...

    @property
    def model_health_store(self) -> Any: ...

    @property
    def run_control(self) -> Any: ...

    @property
    def telemetry(self) -> Any: ...

    @property
    def tool_view_mode(self) -> str: ...


ReadinessSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ProductionReadinessIssue:
    code: str
    message: str
    severity: ReadinessSeverity
    port: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "port": self.port,
        }


@dataclass(frozen=True, slots=True)
class ProductionReadinessReport:
    issues: tuple[ProductionReadinessIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ProductionReadinessIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ProductionReadinessIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "issues": [issue.as_dict() for issue in self.issues],
        }

    def require_ready(self) -> None:
        if not self.ready:
            raise ProductionConfigurationError(self)


class ProductionConfigurationError(RuntimeError):
    code = "PRODUCTION_CONFIGURATION_INVALID"

    def __init__(self, report: ProductionReadinessReport) -> None:
        self.report = report
        summary = "; ".join(
            f"{issue.port or issue.code}: {issue.message}" for issue in report.errors
        )
        super().__init__(f"DeepKeel production configuration is not ready: {summary}")


def assess_production_readiness(ports: ProductionRuntimePorts) -> ProductionReadinessReport:
    """Evaluate whether explicit Host ports are safe for multi-worker use."""

    issues: list[ProductionReadinessIssue] = []
    _require_explicit(
        issues,
        ports.checkpointer,
        port="checkpointer",
        message="configure a durable LangGraph checkpointer",
    )
    _require_one(
        issues,
        ports.runtime_state_store,
        ports.async_runtime_state_store,
        port="runtime_state_store",
        message="configure a durable canonical runtime state store",
    )
    _require_one(
        issues,
        ports.event_journal,
        ports.async_event_journal,
        port="event_journal",
        message="configure a durable runtime event journal",
    )
    _require_one(
        issues,
        ports.run_lease_store,
        ports.async_run_lease_store,
        port="run_lease_store",
        message="configure a distributed run lease store",
    )
    for port, value, message in (
        (
            "model_invocation_store",
            ports.model_invocation_store,
            "configure durable model invocation idempotency",
        ),
        ("budget_ledger", ports.budget_ledger, "configure a shared budget ledger"),
        (
            "model_health_store",
            ports.model_health_store,
            "configure shared model health state",
        ),
        (
            "run_control",
            ports.run_control,
            "configure a cancellable shared run control",
        ),
        ("telemetry", ports.telemetry, "configure durable telemetry or trace export"),
    ):
        _require_explicit(issues, value, port=port, message=message)
    _require_one(
        issues,
        ports.tool_execution_store,
        ports.async_tool_execution_store,
        port="tool_execution_store",
        message="configure durable tool execution idempotency",
    )
    _reject_known_local_ports(issues, ports)
    _warn_blocking_async_ports(issues, ports)
    if ports.tool_view_mode != "enforced":
        issues.append(
            ProductionReadinessIssue(
                code="TOOL_DISCLOSURE_NOT_ENFORCED",
                message="production requires enforced progressive tool disclosure",
                severity="error",
                port="tool_view_mode",
            )
        )
    return ProductionReadinessReport(tuple(issues))


def _require_explicit(
    issues: list[ProductionReadinessIssue],
    value: object | None,
    *,
    port: str,
    message: str,
) -> None:
    if value is None:
        issues.append(
            ProductionReadinessIssue(
                code="MISSING_PRODUCTION_PORT",
                message=message,
                severity="error",
                port=port,
            )
        )


def _require_one(
    issues: list[ProductionReadinessIssue],
    synchronous: object | None,
    asynchronous: object | None,
    *,
    port: str,
    message: str,
) -> None:
    if synchronous is None and asynchronous is None:
        _require_explicit(issues, None, port=port, message=message)
    if synchronous is not None and asynchronous is not None:
        issues.append(
            ProductionReadinessIssue(
                code="AMBIGUOUS_PRODUCTION_PORT",
                message="configure either the synchronous or asynchronous port, not both",
                severity="error",
                port=port,
            )
        )


def _reject_known_local_ports(
    issues: list[ProductionReadinessIssue],
    ports: ProductionRuntimePorts,
) -> None:
    candidates = {
        "checkpointer": ports.checkpointer,
        "runtime_state_store": ports.async_runtime_state_store or ports.runtime_state_store,
        "event_journal": ports.async_event_journal or ports.event_journal,
        "run_lease_store": ports.async_run_lease_store or ports.run_lease_store,
        "model_invocation_store": ports.model_invocation_store,
        "tool_execution_store": (
            ports.async_tool_execution_store or ports.tool_execution_store
        ),
        "budget_ledger": ports.budget_ledger,
        "model_health_store": ports.model_health_store,
        "run_control": ports.run_control,
        "telemetry": ports.telemetry,
    }
    for port, value in candidates.items():
        if value is None:
            continue
        implementation = _underlying_implementation(value)
        type_name = type(implementation).__name__
        saver_name = type(getattr(implementation, "saver", None)).__name__
        if (
            type_name.startswith("InMemory")
            or type_name in {"NoopRunControl", "NoopTelemetry"}
            or saver_name == "InMemorySaver"
        ):
            issues.append(
                ProductionReadinessIssue(
                    code="PROCESS_LOCAL_PRODUCTION_PORT",
                    message=f"{type_name} is process-local and unsafe for multi-worker use",
                    severity="error",
                    port=port,
                )
            )


def _warn_blocking_async_ports(
    issues: list[ProductionReadinessIssue],
    ports: ProductionRuntimePorts,
) -> None:
    for port, synchronous, asynchronous in (
        ("checkpoint_store", ports.checkpoint_store, ports.async_checkpoint_store),
        (
            "runtime_state_store",
            ports.runtime_state_store,
            ports.async_runtime_state_store,
        ),
        ("event_journal", ports.event_journal, ports.async_event_journal),
        ("run_lease_store", ports.run_lease_store, ports.async_run_lease_store),
        (
            "tool_execution_store",
            ports.tool_execution_store,
            ports.async_tool_execution_store,
        ),
    ):
        if synchronous is not None and asynchronous is None:
            issues.append(
                ProductionReadinessIssue(
                    code="BLOCKING_ASYNC_PATH",
                    message=(
                        "arun() requires a thread bridge for this synchronous adapter; "
                        "prefer a native async port for sustained concurrency"
                    ),
                    severity="warning",
                    port=port,
                )
            )


def _underlying_implementation(value: object) -> object:
    """Unwrap the explicit sync-to-async bridges used by the Adapter SDK."""

    for attribute in ("store", "journal"):
        wrapped = getattr(value, attribute, None)
        if wrapped is not None:
            return wrapped
    return value
