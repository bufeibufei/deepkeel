from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from deepkeel.adapter_sdk import CompositeTelemetry, TelemetryPort
from deepkeel.contrib.postgres import (
    PostgresRuntimeBundle,
    PostgresSchemaStatus,
)
from deepkeel.runtime_sdk import HarnessRuntime, HarnessRuntimeBuilder


@dataclass(frozen=True, slots=True)
class ProductionWorker:
    runtime: HarnessRuntime
    postgres: PostgresRuntimeBundle
    schema_status: PostgresSchemaStatus


def build_production_worker(
    checkpointer: Any,
    *,
    dsn: str | None = None,
    schema: str = "deepkeel",
    worker_id: str | None = None,
    telemetry: TelemetryPort | None = None,
) -> ProductionWorker:
    """Compose one production worker entirely through public DeepKeel APIs."""

    resolved_dsn = str(dsn or os.environ.get("DEEPKEEL_POSTGRES_DSN") or "").strip()
    if not resolved_dsn:
        raise RuntimeError("DEEPKEEL_POSTGRES_DSN is required")
    resolved_worker_id = str(
        worker_id or os.environ.get("DEEPKEEL_WORKER_ID") or "worker-01"
    ).strip()
    postgres = PostgresRuntimeBundle.create(resolved_dsn, schema=schema)
    ports = postgres.runtime_ports(
        checkpointer=checkpointer,
        run_lease_owner_id=resolved_worker_id,
    )
    builder = HarnessRuntimeBuilder(profile="production").with_ports(ports)
    if telemetry is not None:
        builder.configure_ports(
            telemetry=CompositeTelemetry((postgres.trace_store, telemetry))
        )
    report = builder.production_readiness()
    if not report.ready:  # pragma: no cover - build() repeats this fail-closed gate
        raise RuntimeError(f"production runtime is not ready: {report.as_dict()}")
    return ProductionWorker(
        runtime=builder.build(),
        postgres=postgres,
        schema_status=postgres.database.migration_status(),
    )
