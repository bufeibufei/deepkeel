# Quickstart

[English](README.md) | [简体中文](README.zh-CN.md)

From a cloned DeepKeel repository, install the package and run the offline
deterministic provider:

```bash
pip install -e .
python examples/quickstart/main.py
```

Replace `LocalProvider` with a Host adapter for the model service you operate.
The runtime remains vendor-neutral.

The Golden Path is intentionally small:

```python
from deepkeel.runtime_sdk import AgentHarness

harness = AgentHarness.create(provider=my_provider)
result = harness.run("Inspect this incident", user_id="operator-42")
print(result.final_answer.markdown)
```

For production composition, build a `HarnessRuntime` with explicit durable
Ports and pass it to `AgentHarness.from_runtime(...)`.
