from __future__ import annotations

from typing import Any

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
