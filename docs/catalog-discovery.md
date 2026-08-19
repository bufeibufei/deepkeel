# Catalog discovery

DeepKeel filters Skills and tools by runtime scope and policy before discovery.
The model sees only a bounded result set; discovery adapters cannot add entries
that were not in the permission-filtered candidate set.

The portable default is deterministic lexical matching. Larger catalogs should
install `HybridSkillRanker` and `HybridToolRanker` from
`deepkeel.hybrid_discovery` as both the discovery and reranker ports. They combine
provider-neutral semantic similarity with lexical evidence and abstain when the
best candidates do not reach `HybridDiscoveryPolicy.minimum_score`.

Hosts own the `SimilarityPort` implementation and may use embeddings, a rerank
service, or an offline index. Core does not depend on a model vendor or vector
database. A semantic adapter must return one finite score in `[0, 1]` per input
document. Permission filtering always happens before the adapter is invoked.

Recommended production evaluation records recall@k, selection precision,
abstention accuracy, latency, and the selected catalog version. Start with broad
recall (10-20 entries), rerank to 3 Skills or 5 tools, and tune the threshold on
domain scenarios instead of forcing a match for every query.
