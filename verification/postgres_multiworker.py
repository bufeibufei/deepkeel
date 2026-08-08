from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from statistics import quantiles
from time import perf_counter
from uuid import uuid4

from deepkeel.runtime_sdk import RuntimeStateMutation
from deepkeel.contrib.postgres import (
    PostgresDatabase,
    PostgresDurableCheckpointStore,
    PostgresRunLeaseStore,
    PostgresRuntimeStateStore,
)


def run_baseline(
    dsn: str,
    *,
    requests: int,
    workers: int,
    max_p95_ms: float,
) -> dict[str, float | int]:
    schema = f"deepkeel_baseline_{uuid4().hex[:12]}"
    database = PostgresDatabase(dsn, schema=schema)
    database.initialize()
    started = perf_counter()

    def execute(index: int) -> float:
        item_started = perf_counter()
        run_id = f"baseline-{index}-{uuid4().hex}"
        owner_id = f"worker-{index % workers}"
        lease_store = PostgresRunLeaseStore(database)
        lease = lease_store.claim(run_id, owner_id=owner_id, ttl_seconds=30)
        PostgresRuntimeStateStore(database).commit(
            RuntimeStateMutation(
                mutation_id=f"{run_id}:started",
                run_id=run_id,
                event_type="run.started",
                target_status="task_running",
                checkpoint_state={"phase": "running", "request": index},
                fence_token=lease.token,
                fence_generation=lease.generation,
            ),
            user_id="baseline-user",
        )
        PostgresDurableCheckpointStore(database).save(
            run_id,
            {
                "schema_version": "harness-durable-checkpoint-v2",
                "phase": "running",
                "request": index,
            },
            user_id="baseline-user",
        )
        recovered = PostgresDurableCheckpointStore(database).load(
            run_id,
            user_id="baseline-user",
        )
        if recovered is None or recovered.get("request") != index:
            raise AssertionError("checkpoint did not survive worker recreation")
        lease_store.release(lease)
        return (perf_counter() - item_started) * 1000

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(execute, range(requests)))
        elapsed = perf_counter() - started
        p95_ms = quantiles(latencies, n=100, method="inclusive")[94]
        result: dict[str, float | int] = {
            "requests": requests,
            "workers": workers,
            "success_rate": 1.0,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_per_second": round(requests / elapsed, 3),
            "p95_ms": round(p95_ms, 3),
        }
        if p95_ms > max_p95_ms:
            raise RuntimeError(
                f"PostgreSQL multi-worker p95 {p95_ms:.3f}ms exceeds {max_p95_ms:.3f}ms"
            )
        return result
    finally:
        database.drop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the product-neutral PostgreSQL multi-worker recovery baseline."
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DEEPKEEL_TEST_POSTGRES_DSN", ""),
    )
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-p95-ms", type=float, default=1500.0)
    args = parser.parse_args()
    if not str(args.dsn or "").strip():
        raise SystemExit("DEEPKEEL_TEST_POSTGRES_DSN or --dsn is required")
    print(
        json.dumps(
            run_baseline(
                args.dsn,
                requests=max(2, args.requests),
                workers=max(1, args.workers),
                max_p95_ms=max(1.0, args.max_p95_ms),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
