from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelStepContext:
    run_id: str
    user_id: str
    thread_id: str
    turn_id: str
    step_index: int
    message_count: int
    observation_count: int
    tool_result_count: int
    available_roles: tuple[str, ...]
    observation_sources: tuple[str, ...] = ()
    tool_result_names: tuple[str, ...] = ()
    model_policy: dict[str, Any] = field(default_factory=dict)
    skill_activation: dict[str, Any] = field(default_factory=dict)
    policy_phase: str = ""
    forced_tool_name: str = ""
    governance_scope: dict[str, Any] = field(default_factory=dict)
    deadline_monotonic: float | None = None
    operational_run_id: str = ""

    @property
    def accounting_run_id(self) -> str:
        """Opaque identity for shared budget, control, and accounting stores."""

        return self.operational_run_id or self.run_id


@dataclass(frozen=True, slots=True)
class ModelRouteDecision:
    role: str
    reason: str
    router_id: str = "adaptive-step-router-v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "reason": self.reason,
            "router_id": self.router_id,
            "metadata": dict(self.metadata),
        }


class ModelRouter(Protocol):
    """Selects a model role for each individual reasoning step."""

    def route(self, context: ModelStepContext) -> ModelRouteDecision: ...


class AdaptiveStepModelRouter:
    router_id = "adaptive-step-router-v1"

    def route(self, context: ModelStepContext) -> ModelRouteDecision:
        available = tuple(dict.fromkeys(role for role in context.available_roles if role))
        if not available:
            raise RuntimeError("provider does not support native tool calls")

        policy = context.model_policy if isinstance(context.model_policy, dict) else {}
        mode = str(policy.get("mode") or "single").strip().lower()
        primary = str(policy.get("primary_role") or "reasoning")
        if mode != "adaptive":
            return self._decision(
                _available_role(primary, available),
                "single model policy",
                context,
            )

        skill = context.skill_activation if isinstance(context.skill_activation, dict) else {}
        if context.policy_phase == "repair":
            return self._decision(
                _available_role("reasoning", available),
                "workflow contract repair requires reasoning",
                context,
            )
        if _is_tool_discovery_continuation(context) and "fast" in available:
            return self._decision(
                "fast",
                "tool discovery continuation uses fast model",
                context,
            )
        is_workflow = str(skill.get("kind") or "") == "workflow"
        if (
            is_workflow
            and context.step_index == 0
            and context.observation_count == 0
            and context.tool_result_count == 0
            and "fast" in available
        ):
            return self._decision(
                "fast",
                "workflow initial planning uses fast model",
                context,
            )
        if is_workflow:
            return self._decision(
                _available_role("reasoning", available),
                "workflow observations require reasoning",
                context,
            )
        if context.observation_count > 0 or context.tool_result_count > 0:
            return self._decision(
                _available_role("reasoning", available),
                "tool observations require synthesis",
                context,
            )
        if context.step_index == 0 and "fast" in available:
            return self._decision("fast", "initial lightweight planning step", context)
        return self._decision(
            _available_role("reasoning", available),
            "continued reasoning step",
            context,
        )

    def _decision(
        self,
        role: str,
        reason: str,
        context: ModelStepContext,
    ) -> ModelRouteDecision:
        return ModelRouteDecision(
            role=role,
            reason=reason,
            router_id=self.router_id,
            metadata={
                "step_index": context.step_index,
                "observation_count": context.observation_count,
                "tool_result_count": context.tool_result_count,
                "observation_sources": list(context.observation_sources),
                "tool_result_names": list(context.tool_result_names),
                "policy_phase": context.policy_phase,
                "skill_id": str(context.skill_activation.get("skill_id") or ""),
            },
        )


def _is_tool_discovery_continuation(context: ModelStepContext) -> bool:
    if context.observation_count <= 0 and context.tool_result_count <= 0:
        return False
    sources = tuple(source for source in context.observation_sources if source)
    result_names = tuple(name for name in context.tool_result_names if name)
    if len(sources) != context.observation_count or len(result_names) != context.tool_result_count:
        return False
    return all(name == "runtime.discover_tools" for name in (*sources, *result_names))


def _available_role(preferred: str, available: tuple[str, ...]) -> str:
    if preferred in available:
        return preferred
    if "reasoning" in available:
        return "reasoning"
    if "fast" in available:
        return "fast"
    return available[0]
