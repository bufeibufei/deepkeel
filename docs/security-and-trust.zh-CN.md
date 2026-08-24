# 安全与信任

[English](security-and-trust.md) | [简体中文](security-and-trust.zh-CN.md)

DeepKeel 将用户输入、模型输出、Tool Argument/Result、MCP Response、A2A Payload、
Checkpoint 与 Capability Package Metadata 全部视为不可信数据。安全策略通过类型化
Runtime Boundary 强制执行，不能只依赖 Prompt 文本。

## Guardrail Pipeline

`GuardrailRunner` 在输入、模型输出和工具边界按顺序执行 Policy。Decision 可以 Allow、
Transform、Redact、Require Confirmation 或 Deny。Required Guardrail 超时或抛错时
Fail-closed；Optional Guardrail 遵循声明的 Failure Policy。Decision 以 Guardrail、
Stage 与 Operation 为键，Durable Host Store 可以重放，避免重复调用外部审核服务。

Guardrail 是纵深防御，不能替代 Host Authentication、Tenant Authorization、ToolSpec
Permission、Budget Policy 与 Schema Validation。Audit Sink 应持久化 Decision Metadata
和 Digest，不保存敏感原始 Payload。

## Provenance 与外部内容

`DataProvenance` 区分可信配置、用户内容、模型输出、检索证据与外部 Tool Data。外部
内容始终是数据，不能提升为 System Instruction。调用模型前，Context Quality Check
可以隔离 Subject 或 Authority 不匹配的内容。

## Sandbox 与 Workspace Port

`ToolSpec` 可以声明 Required Sandbox、执行限制、Network Policy 与 Workspace 要求。
Core 在 Handler 前获取资源，只注入类型化 Lease Metadata，并在所有终态路径释放。
必需 Adapter 缺失或返回 `enforced=False` 时，Handler 不执行。

`NoopSandboxPort` 和 `LocalWorkspacePort` 是开发 Adapter。生产 Host 应使用真正限制
Wall-time、CPU、Memory、Process、Output、Filesystem 与 Network 的 OS、Container、
VM 或 Remote-execution Adapter。Workspace Root 必须按 Tenant 隔离，Cleanup 必须
拒绝分配根目录外的路径。

## 远程互操作

MCP 与 A2A Endpoint 使用 Egress Control、Scoped Secret、有界 Payload、Timeout、
Protocol Validation 与脱敏 Diagnostic。Streamable HTTP MCP 默认拒绝 Private Network；
自定义 A2A Client 应采用同等 Allowlist 与 DNS-rebinding 防护。

## Host 检查清单

1. 构造请求前对 `RuntimeScope` 授权；
2. Secret 只通过 `SecretProvider` 获取，绝不写入 Context 或 Event；
3. 强制 Tool/Skill Disclosure，拒绝未知 Capability；
4. 多 Worker 使用 Durable Replay、State、Lease 与 Audit Store；
5. 外部写入和不可逆操作必须确认；
6. 只导出隐私受限 Telemetry 与 Online Eval Sample；
7. 部署前验证 Release Provenance 与 SBOM。
