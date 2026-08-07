# Production worker composition

This example assembles DeepKeel's production profile using only public SDKs.
The packaged PostgreSQL bundle owns canonical state, durable checkpoints,
events, leases, model/tool idempotency, governance and traces. The Host still
owns its LangGraph saver and model Provider.

```python
from examples.production_worker import build_production_worker

worker = build_production_worker(
    langgraph_postgres_saver,
    worker_id="agent-worker-01",
)
result = await worker.runtime.arun(request, provider=model_provider)
```

Set `DEEPKEEL_POSTGRES_DSN` before startup. Run `deepkeel doctor`, then
`deepkeel postgres status`; apply pending migrations explicitly with
`deepkeel postgres upgrade --yes` as a deployment step. Application startup is
still idempotent, but migrations should not be hidden inside every worker's
readiness probe.

To export OpenTelemetry spans, configure the OpenTelemetry SDK in the Host and
pass `deepkeel.contrib.otel.OpenTelemetryTelemetry()` as `telemetry`. The
example fans it out alongside the durable PostgreSQL trace store.
