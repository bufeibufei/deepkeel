# Capability Package trust model

A Capability Manifest declares runtime permissions; it is not a Python sandbox.
Importing a Python Capability Pack executes code in the Host process. DeepKeel
therefore distinguishes two deployment modes:

- `trusted_in_process`: the Host verifies an allowlisted SHA-256 digest before
  import. Local development may explicitly opt into unverified packages.
- `isolated`: the implementation runs behind MCP or A2A. The Host allowlists
  endpoint prefixes and the normal egress, authentication and policy controls
  still apply.

`CapabilityPackageSource`, `CapabilityTrustPolicy` and
`evaluate_capability_trust()` provide the portable decision contract. The Host
must perform this check before importing an untrusted entrypoint. Package
permissions, Tool policy and runtime Guardrails remain mandatory after trust is
established; provenance does not grant business permissions.

The CLI supports the local development loop:

```bash
deepkeel pack init ./my_pack --package-id company.my-pack
deepkeel pack inspect ./my_pack/manifest.json
deepkeel pack digest ./my_pack/manifest.json ./my_pack/package.py
deepkeel pack validate ./my_pack/manifest.json \
  --factory my_pack.package:MyPack
```
