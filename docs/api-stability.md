# API stability

[English](api-stability.md) | [简体中文](api-stability.zh-CN.md)

`RuntimeResultSummary` and `RuntimeResult.to_summary()` are additive v4.1
projections. The full `RuntimeResult` remains canonical for recovery and
diagnostics, so existing Hosts do not need to migrate. New product read paths
should prefer the summary projection and fetch traces or checkpoints from their
dedicated operational interfaces only when needed.

Stable and candidate releases are installable from immutable Git tags and
GitHub release artifacts. PyPI publication is enabled only after this
repository's Trusted Publisher has been configured.

DeepKeel follows Semantic Versioning. The v4 public SDK is frozen by
`tests/public_api_v4.sha256`. A public symbol or signature change requires an
intentional API review, changelog entry, snapshot update and migration note.
`tests/public_api_semantics_v4.sha256` additionally freezes every symbol in the
stable runtime, extension, and memory layers: callable signatures,
model/dataclass fields, enum values, and selected behavioral entry methods.
This prevents a name-preserving semantic break from bypassing the release gate.

The package root and the versioned `*_sdk` modules are public. Other modules are
implementation details unless explicitly documented. Persisted schema versions
and Capability Pack contracts evolve independently from the package version.

Every public symbol has exactly one canonical SDK layer. The machine-readable
manifest is available from `deepkeel.public_api.PUBLIC_API_MANIFEST`; new
cross-layer re-exports are rejected by the release tests. Layers use three
stability levels:

- `stable`: runtime, extension, and memory contracts covered by the v4
  compatibility policy;
- `advanced`: adapter and MCP integration contracts intended for infrastructure
  authors and changed only through an explicit compatibility review;
- `experimental`: bounded orchestration and A2A adapter contracts that remain
  eligible for refinement in minor releases.

`PUBLIC_API_BY_STABILITY` provides the corresponding machine-readable facade.
Ordinary applications should stay in the stable layers; infrastructure code
may opt into advanced adapters, while experimental orchestration must be
version-pinned and isolated behind a Capability Package.

Release candidates are intended for Host compatibility testing. Stable v4
releases preserve documented SDK and persisted-contract behavior; deprecations
must provide a migration path before removal in a later major version.

The v4 distribution rename keeps the persisted `harness-core-v3` contract.
Capability generations that accepted the final v3 package line remain
loadable through an explicit compatibility bridge; newly authored packages
should declare a v4 package range.

During the v4 release-candidate cycle, `HarnessRuntimeBuilder` moved from
`deepkeel.adapter_sdk` to `deepkeel.runtime_sdk` because application authors need
it for ordinary runtime construction. The former module keeps a transitional
attribute for source compatibility, but new code must import the canonical
Runtime SDK symbol.

The 4.1 line also adds additive async counterparts for
Memory recall, tool-execution idempotency, and experimental bounded delegation.
Existing synchronous contracts remain source compatible; async Hosts should
prefer the native protocols and use the supplied bridges only for thread-safe
legacy adapters.

DeepKeel 4.1 adds the stable `AgentHarness` Golden Path and additive Guardrail,
Sandbox, context-quality, Skill/Tool discovery, and online-evaluation contracts.
The optional `a2a_sdk` is explicitly experimental. The persisted Capability
Pack and runtime contract remains `harness-core-v3`; see
[Migrating to 4.1](migrating-to-4.1.md).
