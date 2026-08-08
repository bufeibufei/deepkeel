from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from deepkeel.runtime_sdk import HarnessRuntimeBuilder
from deepkeel.runtime_sdk import RuntimeRequest


class BenchmarkProvider:
    """Stateless provider used to measure runtime overhead without network noise."""

    model = "benchmark-model"
    model_role = "fast"

    def complete_chat(self, _messages: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:
        return {
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
            "model": self.model,
        }


@dataclass(frozen=True)
class ConcurrencyBenchmarkReport:
    requests: int
    workers: int
    succeeded: int
    failed: int
    elapsed_seconds: float
    throughput_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.requests if self.requests else 1.0


def run_concurrency_benchmark(
    *,
    requests: int = 300,
    workers: int = 32,
) -> ConcurrencyBenchmarkReport:
    request_count = max(1, int(requests))
    worker_count = max(1, min(int(workers), request_count))
    runtime = HarnessRuntimeBuilder().build()

    def execute(index: int) -> tuple[bool, float]:
        started = time.perf_counter()
        result = runtime.run(
            RuntimeRequest(
                question="Return ok.",
                user_id=f"benchmark-user-{index % worker_count}",
                run_id=f"benchmark-run-{index}",
                thread_id=f"benchmark-thread-{index}",
                turn_id=f"benchmark-turn-{index}",
            ),
            provider=BenchmarkProvider(),
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000
        succeeded = (
            result.status.value == "completed"
            and result.final_answer.markdown.strip() == "ok"
        )
        return succeeded, elapsed_ms

    started = time.perf_counter()
    outcomes: list[tuple[bool, float]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(execute, index) for index in range(request_count)]
        for future in as_completed(futures):
            outcomes.append(future.result())
    elapsed = max(time.perf_counter() - started, 1e-9)

    latencies = sorted(latency for _, latency in outcomes)
    succeeded = sum(1 for success, _ in outcomes if success)
    return ConcurrencyBenchmarkReport(
        requests=request_count,
        workers=worker_count,
        succeeded=succeeded,
        failed=request_count - succeeded,
        elapsed_seconds=round(elapsed, 6),
        throughput_per_second=round(request_count / elapsed, 3),
        latency_p50_ms=round(_percentile(latencies, 0.50), 3),
        latency_p95_ms=round(_percentile(latencies, 0.95), 3),
        latency_max_ms=round(max(latencies), 3),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(len(values) * quantile) - 1))
    return values[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DeepKeel concurrency baseline.")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--max-p95-ms", type=float, default=2_000.0)
    args = parser.parse_args()

    report = run_concurrency_benchmark(
        requests=args.requests,
        workers=args.workers,
    )
    payload = asdict(report)
    payload["success_rate"] = report.success_rate
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return int(
        report.success_rate < args.min_success_rate
        or report.latency_p95_ms > args.max_p95_ms
    )


if __name__ == "__main__":
    raise SystemExit(main())
