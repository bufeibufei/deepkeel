from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


PolicyEffect = Literal["allow", "deny", "confirm"]


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    action: str
    resource_type: str
    resource_id: str
    run_id: str
    user_id: str
    tenant_id: str = ""
    risk_level: str = "low"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str
    policy_id: str = "default"
    requires_confirmation: bool = False
    effect: PolicyEffect | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        effect: PolicyEffect = self.effect or (
            "confirm" if self.requires_confirmation else "allow" if self.allowed else "deny"
        )
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "allowed", effect == "allow")
        object.__setattr__(self, "requires_confirmation", effect == "confirm")

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "effect": self.effect,
            "requires_confirmation": self.requires_confirmation,
            "metadata": dict(self.metadata),
        }


class PolicyEngine(Protocol):
    """Authorization port evaluated before every model or tool operation."""

    def evaluate(self, request: PolicyRequest) -> PolicyDecision: ...


class DefaultPolicyEngine:
    """Safe default policy that preserves existing product behavior."""

    policy_id = "harness-default-v1"

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        usage_policy = (
            request.context.get("usage_policy")
            if isinstance(request.context.get("usage_policy"), dict)
            else {}
        )
        if usage_policy.get("enabled") is False:
            return PolicyDecision(
                allowed=False,
                reason="resource is disabled by its usage policy",
                policy_id=self.policy_id,
                metadata={"rule": "resource_disabled"},
            )

        skill = (
            request.context.get("skill_activation")
            if isinstance(request.context.get("skill_activation"), dict)
            else {}
        )
        allowed_tools = skill.get("allowed_tools") if isinstance(skill.get("allowed_tools"), list) else []
        if (
            request.action == "tool.invoke"
            and skill.get("skill_id")
            and allowed_tools
            and request.resource_id not in {str(name) for name in allowed_tools}
        ):
            return PolicyDecision(
                allowed=False,
                reason="tool is outside the active skill allowlist",
                policy_id=self.policy_id,
                metadata={
                    "rule": "skill_tool_allowlist",
                    "skill_id": str(skill.get("skill_id") or ""),
                },
            )

        runtime_policy = (
            request.context.get("runtime_policy")
            if isinstance(request.context.get("runtime_policy"), dict)
            else {}
        )
        allowed_tenants = {
            str(value)
            for value in runtime_policy.get("allowed_tenants") or []
            if str(value or "").strip()
        }
        if allowed_tenants and request.tenant_id not in allowed_tenants:
            return PolicyDecision(
                allowed=False,
                reason="tenant is not allowed to use this resource",
                policy_id=self.policy_id,
                metadata={"rule": "tenant_allowlist"},
            )
        governance_scope = (
            request.context.get("governance_scope")
            if isinstance(request.context.get("governance_scope"), dict)
            else {}
        )
        granted_scopes = {
            str(value)
            for value in governance_scope.get("scopes") or []
            if str(value or "").strip()
        }
        required_scopes = {
            str(value)
            for value in runtime_policy.get("required_scopes") or []
            if str(value or "").strip()
        }
        if required_scopes - granted_scopes:
            return PolicyDecision(
                allowed=False,
                reason="required governance scopes are missing",
                policy_id=self.policy_id,
                metadata={
                    "rule": "required_scopes",
                    "missing_scopes": sorted(required_scopes - granted_scopes),
                },
            )
        if (
            request.action == "tool.invoke"
            and runtime_policy.get("confirmation_required") is True
            and request.context.get("legacy_confirmation_passthrough") is not True
        ):
            grant = (
                request.context.get("confirmation_grant")
                if isinstance(request.context.get("confirmation_grant"), dict)
                else {}
            )
            if not _confirmation_grant_matches(request, grant):
                return PolicyDecision(
                    allowed=False,
                    effect="confirm",
                    reason=str(
                        runtime_policy.get("confirmation_prompt")
                        or "This tool requires user confirmation before execution."
                    ),
                    policy_id=self.policy_id,
                    metadata={
                        "rule": "tool_confirmation_required",
                        "side_effect": str(runtime_policy.get("side_effect") or ""),
                    },
                )

        return PolicyDecision(
            allowed=True,
            reason="allowed by default harness policy",
            policy_id=self.policy_id,
            metadata={"rule": "default_allow"},
        )


@dataclass(frozen=True, slots=True)
class PolicyRule:
    id: str
    effect: PolicyEffect
    actions: frozenset[str] = field(default_factory=frozenset)
    resource_types: frozenset[str] = field(default_factory=frozenset)
    resource_ids: frozenset[str] = field(default_factory=frozenset)
    tenant_ids: frozenset[str] = field(default_factory=frozenset)
    user_ids: frozenset[str] = field(default_factory=frozenset)
    skill_ids: frozenset[str] = field(default_factory=frozenset)
    risk_levels: frozenset[str] = field(default_factory=frozenset)
    reason: str = ""

    def matches(self, request: PolicyRequest) -> bool:
        skill = request.context.get("skill_activation")
        skill_id = str(skill.get("skill_id") or "") if isinstance(skill, dict) else ""
        return all(
            (
                not allowed or value in allowed
                for allowed, value in (
                    (self.actions, request.action),
                    (self.resource_types, request.resource_type),
                    (self.resource_ids, request.resource_id),
                    (self.tenant_ids, request.tenant_id),
                    (self.user_ids, request.user_id),
                    (self.skill_ids, skill_id),
                    (self.risk_levels, request.risk_level),
                )
            )
        )


class RuleBasedPolicyEngine:
    """Ordered policy overlay with the compatible default engine as fallback."""

    def __init__(
        self,
        rules: list[PolicyRule] | tuple[PolicyRule, ...],
        *,
        fallback: PolicyEngine | None = None,
    ) -> None:
        self.rules = tuple(rules)
        self.fallback = fallback or DefaultPolicyEngine()

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        for rule in self.rules:
            if not rule.matches(request):
                continue
            return PolicyDecision(
                allowed=rule.effect == "allow",
                requires_confirmation=rule.effect == "confirm",
                effect=rule.effect,
                reason=rule.reason or f"matched policy rule {rule.id}",
                policy_id=rule.id,
                metadata={"rule": "configured_policy"},
            )
        return self.fallback.evaluate(request)


class PolicyDeniedError(RuntimeError):
    code = "POLICY_DENIED"

    def __init__(self, decision: PolicyDecision):
        super().__init__(decision.reason)
        self.decision = decision


def _confirmation_grant_matches(request: PolicyRequest, grant: dict[str, Any]) -> bool:
    return (
        grant.get("confirmed") is True
        and str(grant.get("run_id") or "") == request.run_id
        and str(grant.get("tool_call_id") or "")
        == str(request.context.get("tool_call_id") or "")
        and str(grant.get("tool_name") or "") == request.resource_id
    )
