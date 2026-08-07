# API stability

DeepKeel follows Semantic Versioning. The v4 public SDK is frozen by
`tests/public_api_v4.sha256`. A public symbol or signature change requires an
intentional API review, changelog entry, snapshot update and migration note.

The package root and the versioned `*_sdk` modules are public. Other modules are
implementation details unless explicitly documented. Persisted schema versions
and Capability Pack contracts evolve independently from the package version.

Release candidates are intended for Host compatibility testing. Stable v4
releases preserve documented SDK and persisted-contract behavior; deprecations
must provide a migration path before removal in a later major version.
