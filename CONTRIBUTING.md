# Contributing

Harness Agent Core is developed inside the kuitianjiandi repository until its
standalone extraction. Changes to the Core must preserve product neutrality and
must not import application modules, database models, web handlers or business
capability implementations.

Before submitting a change, run from the repository root:

```powershell
.\scripts\verify_harness_core_package.ps1
```

Public SDK changes must be intentional. Update the API version, compatibility
notes, public API snapshot and changelog together. Runtime behavior changes
must include package-owned tests; product-specific expectations belong in host
contract tests rather than the Core package.

New integrations should enter through Ports, ToolProvider, Capability Pack or
the versioned SDK modules. Avoid adding product vocabulary or compatibility
aliases to the kernel.
