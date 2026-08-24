# 快速开始

[English](README.md) | [简体中文](README.zh-CN.md)

在已克隆的 DeepKeel 仓库中安装 Package，并运行离线确定性 Provider：

```bash
pip install -e .
python examples/quickstart/main.py
```

将 `LocalProvider` 替换为 Host 所运营模型服务的 Adapter；Runtime 保持厂商中立。

Golden Path 刻意保持精简：

```python
from deepkeel.runtime_sdk import AgentHarness

harness = AgentHarness.create(provider=my_provider)
result = harness.run("Inspect this incident", user_id="operator-42")
print(result.final_answer.markdown)
```

生产装配请使用显式 Durable Port 构造 `HarnessRuntime`，再传给
`AgentHarness.from_runtime(...)`。
