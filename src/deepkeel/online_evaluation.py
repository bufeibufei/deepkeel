from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from deepkeel.runtime_api import RuntimeResult


OnlineEvalContentMode = Literal["none", "digest", "full"]


class OnlineEvalPolicy(BaseModel):
    """Deterministic sampling and privacy policy for production evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    statuses: frozenset[str] = Field(
        default_factory=lambda: frozenset(
            {
                "completed",
                "failed",
                "waiting_user_action",
                "waiting_user_input",
                "task_running",
            }
        )
    )
    content_mode: OnlineEvalContentMode = "digest"
    policy_id: str = "online-eval-v1"

    def should_sample(self, *, run_id: str, status: str) -> bool:
        if status not in self.statuses or self.sample_rate <= 0:
            return False
        if self.sample_rate >= 1:
            return True
        digest = hashlib.sha256(str(run_id).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return bucket < self.sample_rate


class OnlineEvalSample(BaseModel):
    """Privacy-bounded runtime projection submitted to online evaluators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "harness-online-eval-sample-v1"
    sample_id: str
    policy_id: str
    run_id: str
    thread_id: str = ""
    turn_id: str = ""
    tenant_id: str = ""
    namespace: str = "default"
    skill_id: str = ""
    status: str
    stop_reason: str = ""
    step_count: int = 0
    tool_names: tuple[str, ...] = ()
    failed_tool_names: tuple[str, ...] = ()
    artifact_types: tuple[str, ...] = ()
    error_code: str = ""
    answer_digest: str = ""
    answer: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OnlineEvalScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_id: str
    metric: str
    score: float
    passed: bool | None = None
    label: str = ""
    reason: str = ""


class OnlineEvalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: OnlineEvalSample
    scores: tuple[OnlineEvalScore, ...] = ()
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OnlineEvalPort(Protocol):
    """Fast submission boundary; production adapters should enqueue durably."""

    def submit(self, sample: OnlineEvalSample) -> None: ...


class OnlineEvaluator(Protocol):
    def evaluate(self, sample: OnlineEvalSample) -> tuple[OnlineEvalScore, ...]: ...


class OnlineEvalStore(Protocol):
    def append(self, record: OnlineEvalRecord) -> None: ...


class InMemoryOnlineEvalStore:
    def __init__(self) -> None:
        self._records: list[OnlineEvalRecord] = []
        self._lock = Lock()

    def append(self, record: OnlineEvalRecord) -> None:
        with self._lock:
            if any(item.sample.sample_id == record.sample.sample_id for item in self._records):
                return
            self._records.append(record.model_copy(deep=True))

    def snapshot(self) -> tuple[OnlineEvalRecord, ...]:
        with self._lock:
            return tuple(item.model_copy(deep=True) for item in self._records)


class RuntimeContractOnlineEvaluator:
    """Cheap deterministic health signals; domain judges remain Host adapters."""

    evaluator_id = "runtime-contract-v1"

    def evaluate(self, sample: OnlineEvalSample) -> tuple[OnlineEvalScore, ...]:
        terminal_ok = sample.status not in {"failed", "canceled"}
        tool_ok = not sample.failed_tool_names
        answer_ok = sample.status != "completed" or bool(sample.answer_digest)
        return (
            OnlineEvalScore(
                evaluator_id=self.evaluator_id,
                metric="runtime_success",
                score=1.0 if terminal_ok else 0.0,
                passed=terminal_ok,
                label="healthy" if terminal_ok else "failed",
            ),
            OnlineEvalScore(
                evaluator_id=self.evaluator_id,
                metric="tool_success",
                score=1.0 if tool_ok else 0.0,
                passed=tool_ok,
                label="healthy" if tool_ok else "tool_failure",
            ),
            OnlineEvalScore(
                evaluator_id=self.evaluator_id,
                metric="answer_present",
                score=1.0 if answer_ok else 0.0,
                passed=answer_ok,
                label="present" if answer_ok else "empty",
            ),
        )


class OnlineEvalPipeline:
    """Reference in-process pipeline; queue-backed Hosts implement OnlineEvalPort directly."""

    def __init__(
        self,
        *,
        evaluators: tuple[OnlineEvaluator, ...] = (RuntimeContractOnlineEvaluator(),),
        store: OnlineEvalStore | None = None,
    ) -> None:
        self._evaluators = evaluators
        self._store = store or InMemoryOnlineEvalStore()

    @property
    def store(self) -> OnlineEvalStore:
        return self._store

    def submit(self, sample: OnlineEvalSample) -> None:
        scores = tuple(score for evaluator in self._evaluators for score in evaluator.evaluate(sample))
        self._store.append(OnlineEvalRecord(sample=sample, scores=scores))


def online_eval_sample(
    result: RuntimeResult,
    *,
    policy: OnlineEvalPolicy,
    thread_id: str = "",
    turn_id: str = "",
    tenant_id: str = "",
    namespace: str = "default",
    skill_id: str = "",
) -> OnlineEvalSample:
    answer = result.final_answer.markdown
    answer_digest = hashlib.sha256(answer.encode("utf-8")).hexdigest() if answer else ""
    included_answer = answer if policy.content_mode == "full" else ""
    failed_tools = tuple(
        item.name for item in result.tool_results if item.status not in {"succeeded", "waiting_async"}
    )
    error_code = result.error["code"] if result.error is not None else ""
    sample_identity = f"{result.run_id}:{turn_id}:{result.status.value}:{result.step_count}"
    return OnlineEvalSample(
        sample_id=hashlib.sha256(sample_identity.encode("utf-8")).hexdigest(),
        policy_id=policy.policy_id,
        run_id=result.run_id,
        thread_id=thread_id,
        turn_id=turn_id,
        tenant_id=tenant_id,
        namespace=namespace,
        skill_id=skill_id,
        status=result.status.value,
        stop_reason=result.stop_reason,
        step_count=result.step_count,
        tool_names=tuple(item.name for item in result.tool_results),
        failed_tool_names=failed_tools,
        artifact_types=tuple(item.artifact_type for item in result.artifacts),
        error_code=str(error_code),
        answer_digest=answer_digest if policy.content_mode != "none" else "",
        answer=included_answer,
    )


__all__ = [
    "InMemoryOnlineEvalStore",
    "OnlineEvalContentMode",
    "OnlineEvalPipeline",
    "OnlineEvalPolicy",
    "OnlineEvalPort",
    "OnlineEvalRecord",
    "OnlineEvalSample",
    "OnlineEvalScore",
    "OnlineEvalStore",
    "OnlineEvaluator",
    "RuntimeContractOnlineEvaluator",
    "online_eval_sample",
]
