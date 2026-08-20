from __future__ import annotations

from typing import Any

from deepkeel.online_evaluation import online_eval_sample
from deepkeel.runtime_api import RuntimeResult
from deepkeel.scope import RuntimeScope


def submit_runtime_online_evaluation(
    *,
    runtime: Any,
    result: RuntimeResult,
    runtime_scope: RuntimeScope,
) -> dict[str, str] | None:
    """Submit a sampled result without making evaluation part of run correctness."""
    try:
        port = runtime.online_eval_port
        policy = runtime.online_eval_policy
        if port is None or not policy.should_sample(
            run_id=f"{result.run_id}:{result.turn_id}",
            status=result.status.value,
        ):
            return None
        sample = online_eval_sample(
            result,
            policy=policy,
            thread_id=result.thread_id,
            turn_id=result.turn_id,
            tenant_id=runtime_scope.tenant_id,
            namespace=runtime_scope.namespace,
            skill_id=str(result.skill_activation.get("skill_id") or ""),
        )
        port.submit(sample)
        return {
            "status": "submitted",
            "sample_id": sample.sample_id,
            "policy_id": sample.policy_id,
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "error_type": type(exc).__name__,
        }


__all__ = ["submit_runtime_online_evaluation"]
