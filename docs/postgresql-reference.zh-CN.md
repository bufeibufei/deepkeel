# PostgreSQL 参考适配

[English](postgresql-reference.md) | [简体中文](postgresql-reference.zh-CN.md)

DeepKeel 不强制使用 PostgreSQL、ORM 或特定队列。生产 Host 通过 Port 提供基础设施。
安装 `deepkeel[postgres]`，从 `deepkeel.contrib.postgres` 导入受支持 Adapter。它实现
多 Worker 场景中最容易出错的持久化语义，但不会把 PostgreSQL 强加给 Core。

## 覆盖边界

内置集成实现并验证：

- `RuntimeStateStore` 原子事务提交、幂等 Mutation Receipt、乐观版本、User/Tenant/
  Namespace 隔离与 Execution Fence 拒绝；
- Append-only `RuntimeEventJournal`、稳定 Event Identity、逐 Run 单调 Cursor 与精确
  Replay；
- 使用数据库时间的 `RunLeaseStore` Owner、过期与续租，以及 Release 后仍单调递增的
  Fencing Generation；
- 带防御性 JSON 副本和所有权隔离的 `DurableCheckpointStore`；
- `ModelInvocationStore`、`ToolExecutionStore` 的 Claim、Settlement 与重投精确重放；
- 跨 Worker 的 `BudgetLedger`、`ModelHealthStore`、`CancellableRunControl`；
- Scoped `TraceStore`、确定性查询与有界 Retention Cleanup；
- 进程共享 Package Catalog、绑定 Fingerprint 的 Context Summary、幂等类型化 Memory
  Claim，以及 Durable SubAgent Lineage/Checkpoint/Suspension；
- 带 Checksum 的 Schema Registry、Advisory-lock 串行 Migration、Plan/Status、Drift
  Detection 与 Forward-only Upgrade。

LangGraph Saver 仍由 Host 管理，因为它的生命周期和部署拓扑与 Worker State 独立。
Package、Summary、Memory 与 SubAgent Store 是参考实现，可以由 Host 替换。连接成功
不代表语义正确，必须运行每个公开 Conformance Verifier。

## 装配 Worker

```python
from deepkeel.contrib.postgres import PostgresRuntimeBundle
from deepkeel.runtime_sdk import HarnessRuntimeBuilder

postgres = PostgresRuntimeBundle.create(
    "postgresql://deepkeel:secret@postgres/deepkeel",
    schema="deepkeel",
)
ports = postgres.runtime_ports(
    checkpointer=langgraph_postgres_saver,
    run_lease_owner_id="worker-01",
)
runtime = HarnessRuntimeBuilder(profile="production").with_ports(ports).build()

# 同一 Bundle 暴露的可选 Control-plane 引用。
package_store = postgres.capability_package_store
summary_cache = postgres.context_summary_cache
memory = postgres.memory_store
subagents = postgres.subagent_store
```

`PostgresRuntimeBundle` 不创建或管理 LangGraph Saver；Host 按自身进程生命周期初始化和
关闭它。`create(initialize=True)` 在暴露 Port 前升级 DeepKeel Adapter Schema，不会
检查或修改产品表。

## Schema 生命周期

`PostgresDatabase.initialize()` 是兼容入口，内部委托给
`PostgresSchemaRegistry`。运维人员也可显式查看和应用：

```python
from deepkeel.contrib.postgres import PostgresDatabase

database = PostgresDatabase(dsn, schema="deepkeel")
status = database.migration_status()
pending = database.migration_registry().plan()
upgraded = database.migrate()
```

每个已应用版本都在 `schema_migrations` 中记录不可变 Name 与 Checksum。Checksum
变化、未知未来版本、历史缺口、运行表缺失或必需列缺失都会以 Schema Drift
Fail-closed。并发 Worker 使用事务级 PostgreSQL Advisory Lock 串行 Bootstrap 与
Upgrade。只支持向前迁移；Rollback 由运维 Restore 或后续补偿 Migration 完成。

部署自动化使用内置 CLI。DSN 只能从环境变量读取，不能作为命令参数传递：

```powershell
$env:DEEPKEEL_POSTGRES_DSN = "postgresql://..."
deepkeel postgres status
deepkeel postgres plan
deepkeel postgres upgrade --yes
```

Schema 落后时 `status` 非零退出；`plan` 只读；`upgrade` 必须显式确认，可用
`--target-version` 升到中间版本。Migration 前运行 `deepkeel doctor` 检查 Runtime
和可选集成。

## 运行契约测试

将测试指向一次性数据库。Suite 会创建并删除唯一 Schema，不使用产品表或 Migration：

```powershell
uv sync --extra test --extra postgres
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://user:password@localhost:5432/deepkeel_test"
uv run pytest -q -m postgres tests/test_postgres_reference.py
```

测试使用独立 Adapter Object 模拟不同 Worker，验证 Lease 与乐观 State 竞争只有一个
Winner、跨 Worker Checkpoint/Event 恢复、Scope 隔离、注入崩溃后的 Rollback、旧
Schema Upgrade、Drift 拒绝、Migration 幂等和并发 Migrator 串行化。

重复运行恢复基线：

```powershell
uv run python -m verification.postgres_multiworker `
  --requests 64 `
  --workers 12 `
  --max-p95-ms 1500
```

报告的 p95 只表示本地数据库与 Adapter 行为，不含模型或外部工具延迟。生产 Host
应保留该门禁，并增加使用真实连接池、队列、模型 Gateway 与部署拓扑的 Load Test。

## 采用规则

直接导入 Adapter，不要复制实现。业务 ORM 与 Authorization 保持在 Core 外；用
`RuntimeScope` 表示所有权；Lease 使用数据库时间；模型/工具副作用独立幂等；Lease
被接管后绝不能用过期 Fencing Generation 提交。
