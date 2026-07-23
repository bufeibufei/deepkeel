"""Machine-readable manifest for the frozen Harness Core v3 public surface."""

from harness_core.adapter_sdk import ADAPTER_SDK_API
from harness_core.extension_sdk import EXTENSION_SDK_API
from harness_core.mcp_sdk import MCP_SDK_API
from harness_core.memory_sdk import MEMORY_SDK_API
from harness_core.orchestration_sdk import ORCHESTRATION_SDK_API
from harness_core.runtime_sdk import RUNTIME_SDK_API

PUBLIC_API_VERSION = "3.4.0"
PUBLIC_API_BY_LAYER = {
    "runtime": RUNTIME_SDK_API,
    "extension": EXTENSION_SDK_API,
    "orchestration": ORCHESTRATION_SDK_API,
    "mcp": MCP_SDK_API,
    "memory": MEMORY_SDK_API,
    "adapter": ADAPTER_SDK_API,
}
PUBLIC_API_SYMBOLS = frozenset(
    symbol
    for symbols in PUBLIC_API_BY_LAYER.values()
    for symbol in symbols
)

__all__ = ["PUBLIC_API_BY_LAYER", "PUBLIC_API_SYMBOLS", "PUBLIC_API_VERSION"]
