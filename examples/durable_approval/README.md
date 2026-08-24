# Durable approval and resume

[English](README.md) | [简体中文](README.zh-CN.md)

This example returns a typed `PendingAction`, suspends the run, and resumes the
same run with a Host-provided observation:

```bash
python examples/durable_approval/main.py
```

The example uses in-memory state for clarity. Production Hosts must persist the
run, pending action, checkpoint and observation through durable adapters.
