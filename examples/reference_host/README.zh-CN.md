# 参考 Host

[English](README.md) | [简体中文](README.zh-CN.md)

该示例展示 DeepKeel 有意不负责的 Host 边界：HTTP Request Validation、SSE Delivery、
Run Inspection 与 Cancellation。为方便直接启动，它使用进程本地 Adapter。

```bash
pip install fastapi uvicorn
uvicorn your_module:host.app
```

通过 `create_reference_host(model_provider)` 创建 `host`。生产环境应按
`examples/production_worker` 替换为 `PostgresRuntimeBundle.runtime_ports()`，并由
Host 配置模型凭据、认证、队列与 OTel Exporter。更换 Adapter 不改变 HTTP Surface。
