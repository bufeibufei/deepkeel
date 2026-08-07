from __future__ import annotations

from collections.abc import Callable, Iterable
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from deepkeel.runtime_api import RuntimeRequest, RuntimeResult, RuntimeResultStatus
from deepkeel.telemetry import TelemetryRecord


class EvalExpectation(BaseModel):
    """Deterministic assertions portable across business Capability Packs."""

    model_config = ConfigDict(extra="forbid")

    allowed_statuses: frozenset[RuntimeResultStatus] = Field(
        default_factory=lambda: frozenset({RuntimeResultStatus.COMPLETED})
    )
    required_tools: frozenset[str] = Field(default_factory=frozenset)
    forbidden_tools: frozenset[str] = Field(default_factory=frozenset)
    required_artifact_types: frozenset[str] = Field(default_factory=frozenset)
    forbidden_error_codes: frozenset[str] = Field(default_factory=frozenset)
    required_trace_events: tuple[str, ...] = ()
    forbidden_trace_events: frozenset[str] = Field(default_factory=frozenset)
    ordered_trace_events: tuple[str, ...] = ()
    max_steps: int | None = Field(default=None, ge=0)
    require_nonempty_answer: bool = True


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    request: RuntimeRequest
    expectation: EvalExpectation = Field(default_factory=EvalExpectation)
    tags: frozenset[str] = Field(default_factory=frozenset)


class EvalViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    actual: str = ""
    expected: str = ""


class EvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    passed: bool
    runtime_status: str = ""
    duration_ms: float = Field(default=0.0, ge=0)
    violations: list[EvalViolation] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class EvalSuiteReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    cases: list[EvalCaseResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.cases) and all(case.passed for case in self.cases)

    @property
    def pass_rate(self) -> float:
        return (
            sum(1 for case in self.cases if case.passed) / len(self.cases)
            if self.cases
            else 0.0
        )


def evaluate_runtime_result(
    case: EvalCase,
    result: RuntimeResult,
    *,
    trace: Iterable[TelemetryRecord] = (),
    duration_ms: float = 0.0,
) -> EvalCaseResult:
    expectation = case.expectation
    violations: list[EvalViolation] = []
    if result.status not in expectation.allowed_statuses:
        violations.append(
            EvalViolation(
                code="unexpected_status",
                message="runtime status is outside the allowed set",
                actual=result.status.value,
                expected=",".join(sorted(status.value for status in expectation.allowed_statuses)),
            )
        )
    tool_names = {tool.name for tool in result.tool_results}
    _require_members(
        violations,
        code="missing_tool",
        message="required tool was not executed",
        required=expectation.required_tools,
        actual=tool_names,
    )
    _forbid_members(
        violations,
        code="forbidden_tool",
        message="forbidden tool was executed",
        forbidden=expectation.forbidden_tools,
        actual=tool_names,
    )
    artifact_types = {artifact.artifact_type for artifact in result.artifacts}
    _require_members(
        violations,
        code="missing_artifact",
        message="required artifact type was not produced",
        required=expectation.required_artifact_types,
        actual=artifact_types,
    )
    error_code = str(result.error["code"] if result.error is not None else "")
    if error_code and error_code in expectation.forbidden_error_codes:
        violations.append(
            EvalViolation(
                code="forbidden_error",
                message="runtime returned a forbidden error code",
                actual=error_code,
            )
        )
    if expectation.max_steps is not None and result.step_count > expectation.max_steps:
        violations.append(
            EvalViolation(
                code="step_budget_exceeded",
                message="runtime used more reasoning steps than allowed",
                actual=str(result.step_count),
                expected=str(expectation.max_steps),
            )
        )
    if expectation.require_nonempty_answer and not result.final_answer.markdown.strip():
        violations.append(
            EvalViolation(
                code="empty_answer",
                message="runtime did not produce a user-visible answer",
            )
        )

    trace_records = tuple(trace)
    trace_names = tuple(record.event_name for record in trace_records)
    _require_members(
        violations,
        code="missing_trace_event",
        message="required trace event was not observed",
        required=frozenset(expectation.required_trace_events),
        actual=set(trace_names),
    )
    _forbid_members(
        violations,
        code="forbidden_trace_event",
        message="forbidden trace event was observed",
        forbidden=expectation.forbidden_trace_events,
        actual=set(trace_names),
    )
    if expectation.ordered_trace_events and not _is_ordered_subsequence(
        expectation.ordered_trace_events,
        trace_names,
    ):
        violations.append(
            EvalViolation(
                code="trace_order_mismatch",
                message="required trace events did not occur in order",
                actual=" > ".join(trace_names),
                expected=" > ".join(expectation.ordered_trace_events),
            )
        )
    return EvalCaseResult(
        case_id=case.case_id,
        passed=not violations,
        runtime_status=result.status.value,
        duration_ms=max(0.0, float(duration_ms)),
        violations=violations,
        metrics={
            "step_count": float(result.step_count),
            "tool_count": float(len(result.tool_results)),
            "artifact_count": float(len(result.artifacts)),
            "trace_event_count": float(len(trace_records)),
        },
    )


class EvalSuiteRunner:
    """Runs deterministic cases through any callable exposing RuntimeRequest."""

    def __init__(
        self,
        execute: Callable[[RuntimeRequest], RuntimeResult],
        *,
        trace_loader: Callable[[str], Iterable[TelemetryRecord]] | None = None,
    ) -> None:
        self.execute = execute
        self.trace_loader = trace_loader

    def run(self, suite_id: str, cases: Iterable[EvalCase]) -> EvalSuiteReport:
        results: list[EvalCaseResult] = []
        for case in cases:
            started = perf_counter()
            try:
                runtime_result = self.execute(case.request)
            except Exception as exc:
                results.append(
                    EvalCaseResult(
                        case_id=case.case_id,
                        passed=False,
                        duration_ms=(perf_counter() - started) * 1000,
                        violations=[
                            EvalViolation(
                                code="execution_exception",
                                message="runtime execution raised an exception",
                                actual=f"{type(exc).__name__}: {exc}",
                            )
                        ],
                    )
                )
                continue
            trace = (
                tuple(self.trace_loader(runtime_result.run_id))
                if self.trace_loader is not None
                else ()
            )
            results.append(
                evaluate_runtime_result(
                    case,
                    runtime_result,
                    trace=trace,
                    duration_ms=(perf_counter() - started) * 1000,
                )
            )
        return EvalSuiteReport(suite_id=suite_id, cases=results)


def _require_members(
    violations: list[EvalViolation],
    *,
    code: str,
    message: str,
    required: frozenset[str],
    actual: set[str],
) -> None:
    for value in sorted(required - actual):
        violations.append(
            EvalViolation(code=code, message=message, actual="", expected=value)
        )


def _forbid_members(
    violations: list[EvalViolation],
    *,
    code: str,
    message: str,
    forbidden: frozenset[str],
    actual: set[str],
) -> None:
    for value in sorted(forbidden & actual):
        violations.append(
            EvalViolation(code=code, message=message, actual=value, expected="")
        )


def _is_ordered_subsequence(expected: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    cursor = iter(actual)
    return all(any(candidate == item for candidate in cursor) for item in expected)
