"""Public entrypoint for DeepKeel.

Consumers integrate through the versioned Runtime, Extension, and Adapter SDKs.
The package root intentionally stays small so internal modules can evolve without
silently expanding the compatibility surface.
"""

from deepkeel import (
    a2a_sdk,
    adapter_sdk,
    extension_sdk,
    mcp_sdk,
    memory_sdk,
    orchestration_sdk,
    runtime_sdk,
)
from deepkeel.version import DEEPKEEL_CONTRACT_VERSION, DEEPKEEL_VERSION

__all__ = [
    "DEEPKEEL_CONTRACT_VERSION",
    "DEEPKEEL_VERSION",
    "a2a_sdk",
    "adapter_sdk",
    "extension_sdk",
    "mcp_sdk",
    "memory_sdk",
    "orchestration_sdk",
    "runtime_sdk",
]
