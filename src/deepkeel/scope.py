from __future__ import annotations

import hashlib
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class RuntimeScopeUnsupported(RuntimeError):
    """Raised when a legacy adapter cannot safely represent a runtime scope."""

    code = "RUNTIME_SCOPE_UNSUPPORTED"


class RuntimeScope(BaseModel):
    """Stable ownership boundary for runtime and operational persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = ""
    user_id: str = "local-device"
    namespace: str = "default"
    attributes: dict[str, str] = Field(default_factory=dict)

    @property
    def is_legacy_compatible(self) -> bool:
        return not self.tenant_id and self.namespace in {"", "default"}

    @property
    def storage_key(self) -> tuple[str, str, str]:
        return (
            str(self.tenant_id or ""),
            str(self.namespace or "default"),
            str(self.user_id or "local-device"),
        )

    @property
    def identity_fingerprint(self) -> str:
        payload = "\x1f".join(self.storage_key)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def qualify_identity(self, value: str) -> str:
        """Create an opaque storage identity while preserving legacy defaults."""

        normalized = str(value or "")
        if self.storage_key == RuntimeScope().storage_key:
            return normalized
        return f"scope:{self.identity_fingerprint}:{normalized}"

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


def resolve_runtime_scope(
    scope: RuntimeScope | None = None,
    *,
    tenant_id: str = "",
    user_id: str = "",
    namespace: str = "",
) -> RuntimeScope:
    if scope is not None:
        provided = {
            "tenant_id": str(tenant_id) if tenant_id else "",
            "user_id": str(user_id) if user_id else "",
            "namespace": str(namespace) if namespace else "",
        }
        expected = {
            "tenant_id": scope.tenant_id,
            "user_id": scope.user_id,
            "namespace": scope.namespace,
        }
        conflicts = [
            field for field, value in provided.items() if value and value != expected[field]
        ]
        if conflicts:
            raise ValueError(
                "runtime scope conflicts with scalar identity fields: " + ", ".join(conflicts)
            )
        return scope
    return RuntimeScope(
        tenant_id=str(tenant_id or ""),
        user_id=str(user_id or "local-device"),
        namespace=str(namespace or "default"),
    )


def require_legacy_compatible_scope(
    scope: RuntimeScope,
    *,
    adapter_name: str,
) -> str:
    if not scope.is_legacy_compatible:
        raise RuntimeScopeUnsupported(
            f"{adapter_name} does not support tenant or namespace isolation"
        )
    return scope.user_id


def scoped_adapter_operation(
    adapter: Any,
    operation: str,
    scope: RuntimeScope,
) -> Callable[..., Any]:
    """Resolve a scope-aware adapter operation without unsafe fallback.

    New adapters expose ``<operation>_scoped``. Legacy adapters remain usable
    for the default single-tenant scope, but cannot silently collapse tenant or
    namespace ownership into a bare ``run_id``.
    """

    scoped = getattr(adapter, f"{operation}_scoped", None)
    if callable(scoped):
        return scoped
    require_legacy_compatible_scope(scope, adapter_name=type(adapter).__name__)
    legacy = getattr(adapter, operation, None)
    if not callable(legacy):
        raise TypeError(f"{type(adapter).__name__} does not implement {operation}")
    return legacy
