# 生产 Worker 装配

[English](README.md) | [简体中文](README.zh-CN.md)

该示例只使用公开 SDK 装配 DeepKeel Production Profile。内置 PostgreSQL Bundle 负责
Canonical State、Durable Checkpoint、Event、Lease、Model/Tool Idempotency、Governance
与 Trace；Host 仍负责 LangGraph Saver 和 Model Provider。

```python
from examples.production_worker import build_production_worker

worker = build_production_worker(
    langgraph_postgres_saver,
    worker_id="agent-worker-01",
)
result = await worker.runtime.arun(request, provider=model_provider)
```

启动前设置 `DEEPKEEL_POSTGRES_DSN`。先运行 `deepkeel doctor` 与
`deepkeel postgres status`，再把 `deepkeel postgres upgrade --yes` 作为显式部署步骤。
应用启动仍应幂等，但 Migration 不应隐藏在每个 Worker 的 Readiness Probe 中。

导出 OTel Span 时，由 Host 配置 OTel SDK，并把
`deepkeel.contrib.otel.OpenTelemetryTelemetry()` 作为 `telemetry` 传入。示例会将其
与 Durable PostgreSQL Trace Store 一起 Fan-out。
