from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from deepkeel.budget import (
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    BudgetDecision,
    BudgetLedger,
    BudgetPolicy,
    BudgetRequest,
    BudgetSnapshot,
    UsageReport,
    preview_budget,
)
from deepkeel.context_window import ConservativeTokenEstimator
from deepkeel.deadlines import ensure_time_remaining
from deepkeel.model_health import ModelHealthStore
from deepkeel.model_invocations import (
    ModelInvocation,
    ModelInvocationEnvelope,
    ModelInvocationRecorder,
    ModelInvocationStore,
    ModelInvocationUnavailable,
    ModelTurn,
)
from deepkeel.model_routing import ModelRouteDecision, ModelStepContext


ProviderInvoker = Callable[..., Awaitable[ModelTurn]]
TurnValidator = Callable[[ModelTurn, ModelStepContext | None], None]


@dataclass(frozen=True, slots=True)
class ModelAttemptOutcome:
    turn: ModelTurn
    visible_delta_emitted: bool
    streamed_output: str


class ModelAttemptExecutionError(RuntimeError):
    """Internal carrier that preserves attempt telemetry without replacing the cause."""

    def __init__(
        self,
        cause: Exception,
        *,
        visible_delta_emitted: bool,
        streamed_output: str,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.visible_delta_emitted = visible_delta_emitted
        self.streamed_output = streamed_output


async def execute_model_attempt(
    *,
    provider: Any,
    route: ModelRouteDecision,
    step_context: ModelStepContext,
    attempt_index: int,
    retry_kind: str,
    provider_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
    request_timeout: int,
    max_output_tokens: int | None,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    estimated_input_tokens: int,
    route_payload: dict[str, Any],
    token_estimator: ConservativeTokenEstimator,
    invocation_recorder: ModelInvocationRecorder | None,
    invocation_store: ModelInvocationStore | None,
    model_health_store: ModelHealthStore,
    invoke_provider: ProviderInvoker,
    validate_turn: TurnValidator,
    on_text_delta: Callable[[str], None] | None,
) -> ModelAttemptOutcome:
    visible_delta_emitted = False
    streamed_output_parts: list[str] = []
    forced_tool_name = str(step_context.forced_tool_name or "").strip()

    def tracked_delta(delta: str) -> None:
        nonlocal visible_delta_emitted
        ensure_time_remaining(step_context.deadline_monotonic)
        if delta:
            if on_text_delta is not None and not forced_tool_name:
                visible_delta_emitted = True
            streamed_output_parts.append(delta)
            if (
                max_output_tokens is not None
                and token_estimator.estimate("".join(streamed_output_parts))
                > max_output_tokens
            ):
                decision = preview_budget(
                    BudgetSnapshot(run_id=step_context.run_id),
                    BudgetRequest(
                        run_id=step_context.run_id,
                        metric=OUTPUT_TOKENS,
                        amount=token_estimator.estimate("".join(streamed_output_parts)),
                        limit=max_output_tokens,
                    ),
                )
                from deepkeel.budget import BudgetExceededError

                raise BudgetExceededError(decision)
        if on_text_delta is not None and not forced_tool_name:
            on_text_delta(delta)

    invocation = ModelInvocation(
        messages=provider_messages,
        tools=tools,
        tool_choice=tool_choice,
        request_timeout=request_timeout,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    envelope = ModelInvocationEnvelope(
        invocation_id=(
            f"{step_context.run_id}:turn:{step_context.turn_id}:"
            f"model:{step_context.step_index}:attempt:{attempt_index}"
        ),
        run_id=step_context.run_id,
        thread_id=step_context.thread_id,
        turn_id=step_context.turn_id,
        step_index=step_context.step_index,
        attempt_index=attempt_index,
        retry_kind=retry_kind,
        provider_id=provider.info.provider_id,
        model_id=provider.info.model_id,
        model_role=route.role,
        router_id=route.router_id,
        route_reason=route.reason,
        estimated_input_tokens=estimated_input_tokens,
        request=invocation,
    )
    route_payload["invocation"] = envelope.public_snapshot()
    if invocation_recorder is not None:
        try:
            invocation_recorder.record(envelope)
            route_payload["invocation"]["recorded"] = True
        except Exception as recorder_error:
            route_payload["invocation"]["recorded"] = False
            route_payload["invocation"]["recorder_error"] = type(recorder_error).__name__

    claim_token = ""
    try:
        ensure_time_remaining(step_context.deadline_monotonic)
        if not provider.info.supports_native_tools:
            raise RuntimeError("provider does not support native tool calls")
        if invocation_store is not None:
            claim = invocation_store.claim(
                envelope,
                lease_seconds=max(30.0, float(request_timeout) + 30.0),
            )
            route_payload["invocation"]["claim_outcome"] = claim.outcome
            if claim.outcome == "replay" and claim.result is not None:
                turn = claim.result.model_copy(deep=True)
                route_payload["invocation"]["replayed"] = True
            elif claim.outcome == "acquired":
                claim_token = claim.claim_token
                turn = await invoke_provider(
                    provider,
                    invocation,
                    on_text_delta=tracked_delta,
                )
            else:
                detail = claim.failure_message or (
                    "the model invocation is already executing"
                    if claim.outcome == "in_progress"
                    else "the model invocation cannot be safely replayed"
                )
                raise ModelInvocationUnavailable(
                    f"model invocation {claim.outcome}: {detail}"
                )
        else:
            turn = await invoke_provider(
                provider,
                invocation,
                on_text_delta=tracked_delta,
            )
        validate_turn(turn, step_context)
        ensure_time_remaining(step_context.deadline_monotonic)
        model_health_store.record_success(
            provider.info.provider_id,
            provider.info.model_id,
        )
        if invocation_store is not None and claim_token:
            invocation_store.complete(
                envelope.invocation_id,
                claim_token=claim_token,
                result=turn,
            )
        return ModelAttemptOutcome(
            turn=turn,
            visible_delta_emitted=visible_delta_emitted,
            streamed_output="".join(streamed_output_parts),
        )
    except Exception as exc:
        if invocation_store is not None and claim_token:
            try:
                invocation_store.fail(
                    envelope.invocation_id,
                    claim_token=claim_token,
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                )
            except Exception as settlement_error:
                route_payload["invocation"]["settlement_error"] = type(
                    settlement_error
                ).__name__
        raise ModelAttemptExecutionError(
            exc,
            visible_delta_emitted=visible_delta_emitted,
            streamed_output="".join(streamed_output_parts),
        ) from exc


def record_failed_attempt_usage(
    *,
    ledger: BudgetLedger,
    policy: BudgetPolicy,
    step_context: ModelStepContext,
    role: str,
    attempt_index: int,
    estimated_input_tokens: int,
    streamed_output: str,
    token_estimator: ConservativeTokenEstimator,
    route_payload: dict[str, Any],
) -> tuple[BudgetDecision, BudgetDecision | None]:
    failed_output_tokens = token_estimator.estimate(streamed_output) if streamed_output else 0
    usage = UsageReport(
        input_tokens=estimated_input_tokens,
        output_tokens=failed_output_tokens,
        total_tokens=estimated_input_tokens + failed_output_tokens,
        source="estimated_failure",
    )
    input_budget = ledger.consume(
        BudgetRequest(
            run_id=step_context.run_id,
            metric=INPUT_TOKENS,
            amount=estimated_input_tokens,
            limit=policy.limit("max_input_tokens_total"),
            operation_id=f"model-input:{step_context.step_index}:attempt:{attempt_index}",
            metadata={"role": role, "usage_source": usage.source},
        )
    )
    output_budget = None
    if failed_output_tokens:
        output_budget = ledger.consume(
            BudgetRequest(
                run_id=step_context.run_id,
                metric=OUTPUT_TOKENS,
                amount=failed_output_tokens,
                limit=policy.limit("max_output_tokens_total"),
                operation_id=(
                    f"model-output:{step_context.step_index}:attempt:{attempt_index}"
                ),
                metadata={"role": role, "usage_source": usage.source},
            )
        )
    route_payload["budget_metrics"][INPUT_TOKENS] = input_budget.as_dict()
    if output_budget is not None:
        route_payload["budget_metrics"][OUTPUT_TOKENS] = output_budget.as_dict()
    route_payload["usage"] = usage.as_dict()
    return input_budget, output_budget


def record_successful_attempt_usage(
    *,
    ledger: BudgetLedger,
    policy: BudgetPolicy,
    step_context: ModelStepContext,
    role: str,
    attempt_index: int,
    estimated_input_tokens: int,
    turn: ModelTurn,
    provider_usage: dict[str, Any],
    token_estimator: ConservativeTokenEstimator,
    route_payload: dict[str, Any],
) -> tuple[BudgetDecision, BudgetDecision]:
    estimated_output_tokens = token_estimator.estimate(
        {
            "content": turn.content,
            "tool_calls": [call.model_dump() for call in turn.tool_calls],
        }
    )
    usage = UsageReport.from_provider(
        provider_usage,
        estimated_input=estimated_input_tokens,
        estimated_output=estimated_output_tokens,
    )
    input_budget = ledger.consume(
        BudgetRequest(
            run_id=step_context.run_id,
            metric=INPUT_TOKENS,
            amount=usage.input_tokens,
            limit=policy.limit("max_input_tokens_total"),
            operation_id=f"model-input:{step_context.step_index}:attempt:{attempt_index}",
            metadata={"role": role, "usage_source": usage.source},
        )
    )
    output_budget = ledger.consume(
        BudgetRequest(
            run_id=step_context.run_id,
            metric=OUTPUT_TOKENS,
            amount=usage.output_tokens,
            limit=policy.limit("max_output_tokens_total"),
            operation_id=f"model-output:{step_context.step_index}:attempt:{attempt_index}",
            metadata={"role": role, "usage_source": usage.source},
        )
    )
    route_payload["budget_metrics"][INPUT_TOKENS] = input_budget.as_dict()
    route_payload["budget_metrics"][OUTPUT_TOKENS] = output_budget.as_dict()
    route_payload["usage"] = usage.as_dict()
    return input_budget, output_budget
