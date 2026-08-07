# Contributing

DeepKeel is an independent product-neutral runtime. Changes must not
import host application modules, database models, web handlers or business
capability implementations.

Before submitting a change, run from the repository root:

```powershell
.\scripts\verify.ps1
```

On macOS or Linux, run the equivalent commands from `.github/workflows/ci.yml`
with `uv`. Pull requests must target `main`; official tags are created from
`main` only.

Public SDK changes must be intentional. Update the API version, compatibility
notes, public API snapshot and changelog together. Runtime behavior changes
must include package-owned tests; product-specific expectations belong in each
consumer repository's host contract tests rather than the Core package.

New integrations should enter through Ports, ToolProvider, Capability Pack or
the versioned SDK modules. Avoid adding product vocabulary or compatibility
aliases to the kernel.
