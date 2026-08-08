# API stability

Release candidates are installable from their immutable Git tags and GitHub
release artifacts. PyPI publication is enabled only after this repository's
Trusted Publisher has been configured.

DeepKeel follows Semantic Versioning. The v4 public SDK is frozen by
`tests/public_api_v4.sha256`. A public symbol or signature change requires an
intentional API review, changelog entry, snapshot update and migration note.
`tests/public_api_semantics_v4.sha256` additionally freezes callable
signatures, model/dataclass fields, enum values and selected runtime methods,
so a name-preserving semantic break cannot bypass the release gate.

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
- `experimental`: bounded orchestration contracts that remain eligible for
  refinement during release candidates.

Release candidates are intended for Host compatibility testing. Stable v4
releases preserve documented SDK and persisted-contract behavior; deprecations
must provide a migration path before removal in a later major version.

The v4 distribution rename keeps the persisted `harness-core-v3` contract.
Capability generations that accepted the final v3 package line remain
loadable through an explicit compatibility bridge; newly authored packages
should declare a v4 package range.

During the final v4 release-candidate cycle, `HarnessRuntimeBuilder` moved from
`deepkeel.adapter_sdk` to `deepkeel.runtime_sdk` because application authors need
it for ordinary runtime construction. The former module keeps a transitional
attribute for source compatibility, but new code must import the canonical
Runtime SDK symbol.

The final release-candidate cycle also adds additive async counterparts for
Memory recall, tool-execution idempotency, and experimental bounded delegation.
Existing synchronous contracts remain source compatible; async Hosts should
prefer the native protocols and use the supplied bridges only for thread-safe
legacy adapters.
