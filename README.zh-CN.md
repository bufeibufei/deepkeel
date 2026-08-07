# DeepKeel

[English](README.md) | [简体中文](README.zh-CN.md)

**面向生产级 AI Agent 的持久化、可观测、能力包驱动 Harness Runtime。**

DeepKeel 负责模型与工具的单循环执行、类型化运行契约、中断与恢复、
上下文工程、治理端口、ToolProvider、Capability Pack、MCP 适配和有界
SubAgent 编排。它不负责产品 API、业务数据库模型、领域提示词、宿主工具
或前端页面。

当前发布候选版本为 `4.0.0rc2`，安装命令如下：

```bash
pip install "deepkeel @ git+https://github.com/bufeibufei/deepkeel.git@v4.0.0-rc.2"
python examples/quickstart/main.py
```

PyPI 发布使用 Trusted Publishing，并在仓库侧完成发布者配置后单独启用。
在此之前，Git 标签与 GitHub Release 制品是发布候选版本的事实来源。

## 从这里开始

- [整体架构](docs/architecture.md)
- [运行生命周期](docs/runtime-lifecycle.md)
- [Capability Package](docs/capability-package-v1.md)
- [上下文管理](docs/context-management.md)
- [持久化执行](docs/durable-execution.md)
- [模型 Provider](docs/model-provider.md)
- [可观测性](docs/observability.md)
- [生产就绪](docs/production-readiness.md)
- [PostgreSQL 参考适配](docs/postgresql-reference.md)
- [API 稳定性](docs/api-stability.md)
- [发布流程](docs/releasing.md)

可运行的 [quickstart](examples/quickstart) 展示最小模型 Provider；
[inventory_pack](examples/inventory_pack) 展示不依赖具体业务的工具和 Artifact；
[durable_approval](examples/durable_approval) 展示中断、人工确认与恢复。

## 架构边界

DeepKeel 将系统分为三个稳定层次：

- **Host** 负责身份认证、API、数据库连接、队列、密钥、部署和前端。
- **Core Runtime** 负责 ReAct 循环、状态机、事件、恢复、策略、预算和调度。
- **Capability Pack** 负责领域工具、Skill、Artifact、Handoff、SubAgent 与上下文贡献。

Host 通过 `RuntimePorts` 注入基础设施，通过
`HarnessRuntimeBuilder` 安装能力包并构建 Runtime。Capability Pack 只依赖公开
SDK，不应直接访问 Host 的 ORM、路由对象或私有运行状态。

`RuntimePorts` 可按持久化、治理、可观测、执行四组 bundle 进行组合，同时保留
原有扁平字段接口。这样宿主可以分领域装配基础设施，又不会让 Capability Pack
或业务代码依赖具体数据库和模型 Provider。

## 快速开始

```python
from deepkeel.runtime_sdk import HarnessRuntimeBuilder, RuntimeRequest

runtime = HarnessRuntimeBuilder().build()
result = runtime.run(
    RuntimeRequest(
        question="检查当前库存",
        user_id="user-1",
        run_id="run-1",
        thread_id="thread-1",
    ),
    provider=model_provider,
)

print(result.status, result.final_answer.markdown)
```

异步 Host 使用同一套执行语义：

```python
result = await runtime.arun(request, provider=model_provider)

async for event in runtime.astream(request, provider=model_provider):
    if event.event_type == "answer.delta":
        publish(event.payload["delta"])
```

`run()` 是同步边界适配器；`arun()` 是规范的异步执行入口。两者不会维护两套
Agent 逻辑。

## 运行契约

`RuntimeRequest` 提供问题、输入部件、运行身份、`RuntimeScope`、短上下文、
Skill 激活和模型策略。`RuntimeResult` 返回最终状态、答案、Observation、
Artifact、引用、诊断、检查点、事件和稳定的 `ui_state`。

运行事件使用带 `run_id`、`turn_id`、序号和游标的类型化 Envelope。持久事件
可通过 Event Journal 重放；`answer.delta` 等临时事件用于实时呈现，不作为恢复
事实来源。

`RuntimeScope` 是租户、命名空间和用户的统一所有权边界。若同时传入 Scope 和
标量身份字段，二者必须一致；Core 会拒绝冲突身份，避免跨租户状态串用。

## 异步与流式

网络型存储应直接实现以下原生异步端口：

- `AsyncRuntimeStateStore`
- `AsyncDurableCheckpointStore`
- `AsyncRuntimeEventJournal`
- `AsyncRunLeaseStore`
- `AsyncTraceStore`
- `AsyncMemoryPort` 与 `AsyncMemoryRecallPolicy`
- `AsyncToolExecutionStore`
- `AsyncDelegationExecutor` 与 `AsyncDelegationDispatcher`

Adapter SDK 提供的 `Async*Adapter` 使用线程卸载，只适合明确线程安全的同步
实现。生产数据库驱动应实现原生异步协议，不应把线程绑定的 ORM Session 交给
线程池。

`astream()` 使用有界队列和有界同循环 backlog。连续文本增量可以合并但不会
丢字，因此慢消费者不会造成无限任务堆积。消费者断开时，Runtime 会请求协作式
取消，并在配置的确认超时后终止后台任务。

## 持久化、恢复与长任务

生产 Host 应提供以下能力：

- `RuntimeStateStore` 保存权威运行投影和可移植 checkpoint。
- LangGraph Checkpointer 保存图内部状态。
- `RuntimeEventJournal` 保存可重放事件。
- `RunLeaseStore` 提供单所有者租约和 fencing token。
- `ModelInvocationStore` 与 `ToolExecutionStore` 提供副作用幂等。
- `RunControl` 提供取消、恢复和操作控制。

恢复时先读取用户 Scope 下的权威状态，再获取租约，从 checkpoint 继续，并重放
已经结算的模型或工具调用。Runtime 会先原子提交终态，再清理兼容 checkpoint，
并把清理结果写入可重放的持久事件；因此终态提交失败时仍保留恢复点，运维人员也
能够追踪 checkpoint 删除失败等诊断。该清理事件仅用于重放和调试，不进入实时
消息流，因此面向用户的终态事件仍然是本轮最后一个流式事件。

Capability Pack 的安装、启用、禁用、升级和回滚通过不可变
`RuntimeGeneration` 与运行实例解耦。旧任务可以继续使用启动时捕获的代际，新任务
使用新代际。

## Capability Pack

能力包使用 Manifest-first V1 契约，声明版本、Core 约束、工具、Skill、Artifact、
Handoff、MCP 与 SubAgent。安装过程是原子的：任一注册失败都会反向清理本轮已经
注册的资源。

```python
from deepkeel.extension_sdk import CapabilityPackSpec

class InventoryPack:
    spec = CapabilityPackSpec(
        package_id="example.inventory",
        package_version="1.0.0",
        declared_tools=("inventory.lookup",),
        declared_skills=("inventory-assistant",),
        declared_artifact_types=("inventory_record",),
    )

    def install(self, context):
        context.register_tool(tool_spec, lookup_handler)
        context.register_skill("inventory-assistant", skill_manifest)
        context.register_artifact_type(artifact_type_spec)
        return contribution
```

完整契约与发布门禁见
[Capability Package V1](docs/capability-package-v1.md)。

## 工具、Skill 与模型

Skill 约束任务上下文、允许工具、预算和输出方式，但工具能力不依赖硬编码业务
路由。渐进式工具披露可以先向模型暴露目录，再按需加载完整 Schema，降低上下文
成本。

模型路由按步骤执行，可根据角色、上下文窗口、结构化输出要求、健康状态、失败
策略和预算选择 Provider。模型 Provider、路由策略和健康存储均通过公开端口注入，
Capability Pack 不绑定具体云厂商。

## 上下文与记忆

Core 提供 L1/L2/L3 上下文规划、token 预算、原子消息组保护、确定性压缩和工作
checkpoint。当前用户输入、工具调用与工具结果不会被拆散或静默截断。

Memory SDK 定义记忆候选、主题、权威性和读取协议；业务事实的提取、审批和持久化
策略由 Host 或 Capability Pack 决定。Core 不把任意助手文本自动视为长期事实。

## MCP、SubAgent 与安全

本地 stdio 和远程 Streamable HTTP MCP 都进入同一受治理工具网关。MCP 工具不能
绕过 ToolSpec、权限、预算、超时、审计和密钥边界。

SubAgent 编排是有界的，受并发、深度、预算和父运行取消控制。Handoff 使用类型化
待办动作和恢复契约，不依赖前端猜测状态。

密钥通过 `SecretProvider` 获取，不进入 Prompt、公开事件或 Artifact。生产 Host
还应在 API 边界执行租户授权，并对外部写操作配置确认策略。

## 生产就绪门禁

`build()` 适合本地开发和测试。生产 Worker 应使用可执行门禁：

```python
builder = HarnessRuntimeBuilder(profile="production").with_ports(production_ports)
report = builder.production_readiness()
runtime = builder.build_production()
```

门禁会检查必需 Host 端口、同步/异步端口歧义、已知 `InMemory*`/`Noop*` 实现和
异步路径中的阻塞适配器。错误会阻止构建，告警会保留在报告中。自定义数据库
适配器仍需运行 `deepkeel.adapter_sdk` 中对应的 conformance verifier。

详细部署清单见 [生产就绪文档](docs/production-readiness.md)。

随包提供的 PostgreSQL 适配器使用带校验和、事务级 advisory lock 的前向迁移。
部署前可通过 `database.migration_status()` 和
`database.migration_registry().plan()` 检查版本；迁移历史或物理列发生漂移、以及
自动降级请求都会 fail-closed。

## 可观测性与评测

Runtime 输出类型化 Trace、事件、失败分类、路由选择、工具生命周期、预算和恢复
诊断。公开记录不会包含完整 Prompt 或敏感工具参数。

`EvalSuiteRunner` 用于确定性回归，包括状态、工具选择、Artifact 合约、错误码、
步骤预算和 Trace 顺序。业务答案质量由 Capability Pack 自己的数据集和评测器
负责；DeepEval 等语义评分器可以消费同一个 `RuntimeResult` 与 Trace。

## 公开 SDK

- `deepkeel.runtime_sdk`：请求、结果、事件、运行与状态契约。
- `deepkeel.extension_sdk`：工具、Skill、Artifact 和 Capability Pack。
- `deepkeel.adapter_sdk`：模型、存储、策略、预算、租约和生产门禁。
- `deepkeel.memory_sdk`：记忆端口与主题契约。
- `deepkeel.orchestration_sdk`：有界 SubAgent 编排。
- `deepkeel.mcp_sdk`：MCP Client、传输和工具映射。

当前 Capability Pack 持久协议仍为 `harness-core-v3`，公开 SDK 版本为 `4.0.0`。
公开符号清单位于 `deepkeel.public_api`，测试会冻结 API 指纹；任何变更都需要显式
兼容性审阅。

## 开发与发布

```powershell
uv sync --extra test
uv run ruff check src tests verification
uv run mypy src/deepkeel
uv run pytest -q --cov=deepkeel --cov-fail-under=80
uv build --clear
uv run python verification/verify_distributions.py dist
```

PR 与主分支 CI 负责跨平台正确性。固定 p95 阈值由独立 Linux 性能基线工作流
执行，避免把 Windows 调度抖动误判为功能回归。

贡献规范、安全策略、支持范围和版本历史分别见
[CONTRIBUTING.md](CONTRIBUTING.md)、[SECURITY.md](SECURITY.md)、
[SUPPORT.md](SUPPORT.md) 与 [CHANGELOG.md](CHANGELOG.md)。

DeepKeel 使用 [Apache License 2.0](LICENSE)。
