# Harness Agent Core

`kuitianjiandi-harness-core` is the product-neutral runtime kernel currently
validated by Kuitianjiandi. It owns the model/tool loop, runtime contracts,
governance ports, interruption, recovery, MCP gateway primitives, sub-agent
execution and Capability Pack composition.

The package does not contain Kuitianjiandi tools, database models, API routes or
product prompts. Consumers integrate through `HarnessRuntimeBuilder`,
`RuntimePorts` and a versioned `CapabilityPack`.

```python
from harness_core import HarnessRuntimeBuilder

runtime = HarnessRuntimeBuilder().add_capability_pack(pack).build()
```

The frozen public contract is `harness-core-v1`; the current package version is
`1.0.0`.

