# 持久化审批与恢复

[English](README.md) | [简体中文](README.zh-CN.md)

该示例返回类型化 `PendingAction`，挂起 Run，再用 Host 提供的 Observation 恢复同一 Run：

```bash
python examples/durable_approval/main.py
```

示例为清晰起见使用 In-memory State。生产 Host 必须通过 Durable Adapter 持久化 Run、
PendingAction、Checkpoint 与 Observation。
