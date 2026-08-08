from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Machine-readable operational guarantees declared by a runtime adapter."""

    durable: bool = False
    process_shared: bool = False
    runtime_scope: bool = False
    native_async: bool = False
    cancellation_safe: bool = False
    transactional: bool = False
    source: str = "adapter_declaration"

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "durable": self.durable,
            "process_shared": self.process_shared,
            "runtime_scope": self.runtime_scope,
            "native_async": self.native_async,
            "cancellation_safe": self.cancellation_safe,
            "transactional": self.transactional,
            "source": self.source,
        }


def declared_adapter_capabilities(value: object | None) -> AdapterCapabilities | None:
    """Resolve a declaration through supported adapter wrapper layers."""

    current: object | None = value
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        declared = getattr(current, "adapter_capabilities", None)
        if isinstance(declared, AdapterCapabilities):
            return declared
        if isinstance(declared, dict):
            return AdapterCapabilities(
                durable=bool(declared.get("durable")),
                process_shared=bool(declared.get("process_shared")),
                runtime_scope=bool(declared.get("runtime_scope")),
                native_async=bool(declared.get("native_async")),
                cancellation_safe=bool(declared.get("cancellation_safe")),
                transactional=bool(declared.get("transactional")),
                source=str(declared.get("source") or "adapter_declaration"),
            )
        current = _wrapped_adapter(current)
    return None


def _wrapped_adapter(value: object) -> Any:
    for attribute in ("store", "journal", "database", "saver"):
        wrapped = getattr(value, attribute, None)
        if wrapped is not None:
            return wrapped
    return None
