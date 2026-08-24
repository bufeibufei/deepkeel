# Supply-chain controls

[English](supply-chain.md) | [简体中文](supply-chain.zh-CN.md)

DeepKeel releases are built from immutable semantic-version tags. The release
workflow verifies the tag against `pyproject.toml` and `deepkeel.version`, runs
the production regression matrix, builds wheel and sdist once, verifies both in
clean environments, and publishes those exact artifacts.

## Release evidence

Every GA release should provide:

- wheel and source distribution with matching metadata;
- an SPDX JSON software bill of materials;
- a deterministic `SHA256SUMS` manifest covering every uploaded artifact;
- GitHub build-provenance attestations for all release artifacts;
- frozen public API and semantic-contract fingerprints;
- synchronized English and Simplified Chinese READMEs and linked documentation;
- changelog and migration notes for the tagged version;
- successful deterministic, fault-injection, concurrency, PostgreSQL, and
  clean-install conformance gates.

PyPI publication uses OIDC Trusted Publishing and does not store a long-lived
upload token. The GitHub release job is idempotent for an existing immutable tag
and replaces only uploaded assets, never the source tag.

## Consumer verification

1. Download artifacts only from the tagged GitHub Release or configured PyPI
   project.
2. Verify GitHub artifact attestations against the repository and workflow.
3. Verify each artifact against `SHA256SUMS` before installation.
4. Inspect the SBOM and apply the organization's vulnerability policy.
5. Pin the exact version or immutable commit in production lockfiles.
6. Run Host adapter and Capability Pack tests against the installed wheel.

An SBOM describes components; it is not a vulnerability verdict. Provenance
shows where an artifact was built; it does not replace code review, dependency
policy, or runtime isolation.
