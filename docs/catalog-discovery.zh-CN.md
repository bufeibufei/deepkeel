# 能力目录发现

[English](catalog-discovery.md) | [简体中文](catalog-discovery.zh-CN.md)

DeepKeel 在发现前先按 Runtime Scope 与 Policy 过滤 Skill 和 Tool。模型只看到有界
结果集；Discovery Adapter 无权把已被权限过滤移除的条目重新加入候选集。

默认实现是可移植、确定性的词法匹配。较大的目录应从 `deepkeel.discovery_sdk`
安装 `HybridSkillRanker` 与 `HybridToolRanker`，同时作为 Discovery 和 Reranker
Port。它们融合与 Provider 无关的语义相似度和词法证据；最佳候选低于
`HybridDiscoveryPolicy.minimum_score` 时会主动 Abstain。

Host 负责实现 `SimilarityPort`，可以使用 Embedding、Rerank 服务或离线索引。
Core 不依赖特定模型厂商或向量数据库。语义 Adapter 必须为每个输入文档返回
`[0, 1]` 内的有限分数，并且只能看到权限过滤后的候选集。

生产评测建议记录 recall@k、选择精度、拒选准确率、延迟和 Catalog Version。可以先
宽召回 10 至 20 项，再重排到 3 个 Skill 或 5 个 Tool；阈值应基于领域场景调优，
而不是强迫每个查询都命中能力。
