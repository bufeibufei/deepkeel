# Reference Host

This example shows the Host boundary that DeepKeel intentionally does not own:
HTTP request validation, SSE delivery, run inspection and cancellation. It uses
process-local adapters so it can be started without infrastructure.

```bash
pip install fastapi uvicorn
uvicorn your_module:host.app
```

Create `host` with `create_reference_host(model_provider)`. For production,
replace the in-memory `RuntimePorts` with `PostgresRuntimeBundle.runtime_ports()`
as demonstrated by `examples/production_worker`, and configure the Host-owned
model credentials, authentication, queue and OpenTelemetry exporter. The HTTP
surface does not change when the adapters change.
