# Capability Package V1

## Developer workflow

Use `deepkeel pack init` to create a manifest-first skeleton, `pack inspect` to
inspect its normalized contract, and `pack validate --factory` to compose the
executable package through the public SDK and detect declaration drift. Release
certification adds executable scenario evaluations through
`certify_capability_package()`.

Package permissions govern tools and resources after installation; they do not
sandbox imported Python. Hosts must apply the provenance policy described in
[`capability-trust.md`](capability-trust.md) before importing third-party code.

A Capability Package is the only supported boundary for adding business
behavior to DeepKeel. The package owns domain tools, Skills,
artifacts, handoffs, prompts, UI projection metadata, and evaluation cases.
It must not import a Host API layer, ORM model, web framework, or private Core
module.

## Required files

```text
my_capability/
├── manifest.json
├── package.py
├── eval_cases.py
└── tests/
    └── test_certification.py
```

`manifest.json` is the single source of truth. `package.py` derives its
`CapabilityPackSpec` with `capability_pack_spec_from_manifest`; duplicating
tool or Skill declarations in Python is not supported.

Skills may declare a `planning_policy` when their work benefits from several
dependent capabilities. The supported modes are `disabled`, `allowed`,
`preferred`, and `required`, with bounded `max_steps`, `max_revisions`,
`max_parallel_steps`, and `max_attempts_per_step`. Planning never expands the
Skill allowlist: every plan step must still reference a tool already available
to the active Skill.

The manifest must declare:

- package identity, semantic version, Core contract, and entrypoint;
- tools, Skills, artifact types, handoffs, and optional MCP providers;
- governance permissions, per-tool permission mappings, and portable budget ceilings;
- state schema and resume-compatible package versions;
- memory namespaces and UI surfaces owned by the package.

## Installation boundary

The package receives only `CapabilityInstallContext`. It registers public
extension objects and returns a `CapabilityContribution` that exactly matches
the manifest. It must not obtain the Host database session, mutate another
package registry, or create a second agent loop.

All write tools must use idempotency keys and declare side-effect policy.
Tool permissions must be present both in the manifest and in the relevant
`ToolSpec.runtime_policy.required_scopes`.

## Certification gate

Every release runs `certify_capability_package`. Certification fails unless:

1. Manifest, package spec, installed contribution, schemas, permissions, and
   handlers are aligned.
2. The package survives install, discovery, disable, enable, upgrade, rollback,
   and runtime-generation resume compatibility checks.
3. The package supplies executable evaluations covering `tool_selection`,
   `argument_generation`, `task_completion`, `recovery`, and `answer_quality`.
4. Every evaluation passes through the public `HarnessRuntime` API.

Packages that consume dependency tools must pass both `dependency_manifests`
and `dependency_packs` to the certification API. Manifest-only dependencies
validate lifecycle metadata, while real dependency packs provide the runtime
tools, handlers, and other contributions used during composition.

Semantic quality graders may be added by the Host, but deterministic contract
checks remain mandatory and cannot be replaced by an LLM score.

## Versioning

Increment the package version for every behavior or contract change. Increment
`state_schema_version` when persisted package state changes. A new package
version may resume an older run only when the old version appears in
`resume_compatible_versions` and the old schema is identical or has a declared
migration.

Workers execute an immutable `RuntimeGeneration`. Installing a new package
version affects new runs only; suspended runs resume only on a compatible
generation.
