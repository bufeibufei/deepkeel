"""Machine-readable manifest for the frozen DeepKeel v4 public surface."""

from deepkeel.adapter_sdk import ADAPTER_SDK_API
from deepkeel.extension_sdk import EXTENSION_SDK_API
from deepkeel.mcp_sdk import MCP_SDK_API
from deepkeel.memory_sdk import MEMORY_SDK_API
from deepkeel.orchestration_sdk import ORCHESTRATION_SDK_API
from deepkeel.runtime_sdk import RUNTIME_SDK_API
from deepkeel.version import SDK_API_VERSION

PUBLIC_API_VERSION = SDK_API_VERSION
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
