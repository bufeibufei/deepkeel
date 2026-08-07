from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from deepkeel.budget import BudgetLedger, InMemoryBudgetLedger
from deepkeel.policy import DefaultPolicyEngine, PolicyEngine


@dataclass(frozen=True, slots=True)
class SecretRequest:
    name: str
    tenant_id: str = ""
    user_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    required_scopes: tuple[str, ...] = ()


class SecretProvider(Protocol):
    """Resolves credentials without exposing storage details to capabilities."""

    def resolve(self, request: SecretRequest) -> str: ...


class SecretNotFoundError(LookupError):
    code = "SECRET_NOT_FOUND"


class DenySecretProvider:
    """Safe default that never falls back to ambient credentials."""

    def resolve(self, request: SecretRequest) -> str:
        raise SecretNotFoundError(f"secret is unavailable: {request.name}")


class MappingSecretProvider:
    def __init__(self, secrets: Mapping[str, str] | None = None) -> None:
        self._secrets = {str(key): str(value) for key, value in (secrets or {}).items()}

    def resolve(self, request: SecretRequest) -> str:
        value = self._secrets.get(request.name, "")
        if not value:
            raise SecretNotFoundError(f"secret is unavailable: {request.name}")
        return value


class EnvironmentSecretProvider:
    """Infrastructure adapter for explicitly allowlisted environment secrets."""

    def __init__(self, allowed_names: set[str] | frozenset[str] | None = None) -> None:
        self.allowed_names = frozenset(str(name) for name in (allowed_names or set()))

    def resolve(self, request: SecretRequest) -> str:
        if request.name not in self.allowed_names:
            raise SecretNotFoundError(f"secret is not allowlisted: {request.name}")
        value = os.environ.get(request.name, "")
        if not value:
            raise SecretNotFoundError(f"secret is unavailable: {request.name}")
        return value


@dataclass(frozen=True, slots=True)
class GovernanceScope:
    tenant_id: str = ""
    user_id: str = ""
    skill_id: str = ""
    scopes: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class GovernanceBundle:
    """One injection unit for all decisions made around external operations."""

    policy_engine: PolicyEngine = field(default_factory=DefaultPolicyEngine)
    budget_ledger: BudgetLedger = field(default_factory=InMemoryBudgetLedger)
    secret_provider: SecretProvider = field(default_factory=DenySecretProvider)

