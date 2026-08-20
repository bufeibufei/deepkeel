"""Machine-readable manifest for the frozen DeepKeel v4 public surface."""

import deepkeel.adapter_sdk as adapter_sdk
import deepkeel.discovery_sdk as discovery_sdk
import deepkeel.extension_sdk as extension_sdk
import deepkeel.memory_sdk as memory_sdk
import deepkeel.runtime_sdk as runtime_sdk

from deepkeel.adapter_sdk import ADAPTER_SDK_API
from deepkeel.discovery_sdk import DISCOVERY_SDK_API
from deepkeel.a2a_sdk import A2A_SDK_API
from deepkeel.api_stability import build_public_api_manifest, build_semantic_contract
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
    "a2a": A2A_SDK_API,
    "mcp": MCP_SDK_API,
    "memory": MEMORY_SDK_API,
    "adapter": ADAPTER_SDK_API,
    "discovery": DISCOVERY_SDK_API,
}
PUBLIC_API_SYMBOLS = frozenset(
    symbol for symbols in PUBLIC_API_BY_LAYER.values() for symbol in symbols
)
PUBLIC_API_MANIFEST = build_public_api_manifest(PUBLIC_API_BY_LAYER)
PUBLIC_API_CANONICAL_LAYER = {
    symbol: layer.name for layer in PUBLIC_API_MANIFEST for symbol in layer.symbols
}
PUBLIC_API_BY_STABILITY = {
    stability: tuple(
        symbol
        for layer in PUBLIC_API_MANIFEST
        if layer.stability == stability
        for symbol in layer.symbols
    )
    for stability in ("stable", "advanced", "experimental")
}
_EXPLICIT_SEMANTIC_MEMBERS = {
    "adapter.ModelProviderAdapter": (
        adapter_sdk.ModelProviderAdapter,
        ("invoke",),
    ),
    "adapter.AsyncModelProviderAdapter": (
        adapter_sdk.AsyncModelProviderAdapter,
        ("ainvoke",),
    ),
    "adapter.RuntimePorts": (adapter_sdk.RuntimePorts, ()),
    "extension.ToolExecutionContext": (
        extension_sdk.ToolExecutionContext,
        ("fork",),
    ),
    "extension.ToolExecutor": (
        extension_sdk.ToolExecutor,
        ("execute", "aexecute", "execute_many", "aexecute_many"),
    ),
    "extension.ToolSpec": (extension_sdk.ToolSpec, ()),
    "runtime.HarnessRuntime": (
        runtime_sdk.HarnessRuntime,
        (
            "run",
            "arun",
            "astream",
            "replay_events",
            "cleanup_run",
        ),
    ),
    "runtime.RuntimeRequest": (runtime_sdk.RuntimeRequest, ()),
    "runtime.RuntimeResult": (runtime_sdk.RuntimeResult, ()),
    "runtime.RuntimeScope": (runtime_sdk.RuntimeScope, ("qualify_identity",)),
}
_STABLE_MODULES = {
    "runtime": runtime_sdk,
    "extension": extension_sdk,
    "memory": memory_sdk,
}
PUBLIC_API_SEMANTIC_TARGETS = {
    f"{layer}.{symbol}": (
        getattr(module, symbol),
        _EXPLICIT_SEMANTIC_MEMBERS.get(f"{layer}.{symbol}", (None, ()))[1],
    )
    for layer, module in _STABLE_MODULES.items()
    for symbol in PUBLIC_API_BY_LAYER[layer]
}
PUBLIC_API_SEMANTIC_MANIFEST = build_semantic_contract(PUBLIC_API_SEMANTIC_TARGETS)

__all__ = [
    "PUBLIC_API_BY_LAYER",
    "PUBLIC_API_BY_STABILITY",
    "PUBLIC_API_CANONICAL_LAYER",
    "PUBLIC_API_MANIFEST",
    "PUBLIC_API_SYMBOLS",
    "PUBLIC_API_SEMANTIC_MANIFEST",
    "PUBLIC_API_SEMANTIC_TARGETS",
    "PUBLIC_API_VERSION",
]
