from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from deepkeel.contracts import ToolCall
from deepkeel.hooks import HookAudit, HookInvocation, HookPoint
from deepkeel.model import ModelTurn
from deepkeel.model_failures import ModelToolContractError
from deepkeel.skills import SkillPolicy
from deepkeel.tool_registry import ToolRegistry
from deepkeel.turn_context import TurnExecutionContext
from deepkeel.type_narrowing import as_dict
from deepkeel.graph_workflow import _emit


def _latest_user_question(messages: list[Any]) -> str:
    for item in reversed(messages):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("role") or "") == "user":
            return str(item.get("content") or "").strip()
    return ""


def _model_hook_invocation(
    point: HookPoint,
    state: Mapping[str, Any],
    turn_context: TurnExecutionContext,
    *,
    payload: Mapping[str, Any],
) -> HookInvocation:
    skill = as_dict(state.get("skill_activation"))
    metadata = as_dict(state.get("metadata"))
    package_ids = tuple(
        str(value)
        for value in turn_context.tool_context.metadata.get("capability_package_ids", ())
        if str(value).strip()
    )
    return HookInvocation(
        point=point,
        operation_id=(
            f"{state.get('run_id')}:{state.get('turn_id')}:"
            f"model:{int(state.get('step_count') or 0)}:{point.value}"
        ),
        run_id=str(state.get("run_id") or ""),
        thread_id=str(state.get("thread_id") or ""),
        turn_id=str(state.get("turn_id") or ""),
        package_ids=package_ids,
        skill_id=str(skill.get("skill_id") or ""),
        payload=dict(payload),
        metadata={
            "governance_scope": dict(metadata.get("governance_scope") or {}),
        },
    )


def _forced_tool_clarification_fallback(
    registry: ToolRegistry,
    forced_tool_name: str,
    error: ModelToolContractError,
    *,
    state: Mapping[str, Any] | None = None,
    turn_context: TurnExecutionContext | None = None,
) -> ModelTurn | None:
    try:
        spec = registry.get(forced_tool_name)
    except KeyError:
        return None
    current = state if isinstance(state, Mapping) else {}
    skill = SkillPolicy.from_snapshot(current.get("skill_activation"))
    fallback = as_dict(spec.runtime_policy.get("forced_tool_contract_fallback"))
    if (
        fallback.get("enabled") is True
        and skill.active
        and (not fallback.get("explicit_skill_only", True) or skill.explicit)
        and skill.required_tools == frozenset({forced_tool_name})
        and not skill.required_tool_groups
    ):
        arguments = _forced_tool_fallback_arguments(
            spec=spec,
            fallback=fallback,
            state=current,
            turn_context=turn_context,
        )
        if arguments is not None:
            return ModelTurn(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"contract-fallback-{uuid4()}",
                        name=forced_tool_name,
                        arguments=arguments,
                    )
                ],
                finish_reason="tool_calls",
                raw={
                    "synthetic": True,
                    "recovery": "forced_tool_arguments",
                    "error": str(error),
                },
            )
    clarification = as_dict(as_dict(spec.argument_contract).get("clarification"))
    if not clarification or not (spec.required_args or spec.required_arg_groups):
        return None
    return ModelTurn(
        content="",
        tool_calls=[
            ToolCall(
                id=f"contract-clarification-{uuid4()}",
                name=forced_tool_name,
                arguments={},
            )
        ],
        finish_reason="tool_calls",
        raw={
            "synthetic": True,
            "recovery": "forced_tool_clarification",
            "error": str(error),
        },
    )


def _forced_tool_fallback_arguments(
    *,
    spec: Any,
    fallback: Mapping[str, Any],
    state: Mapping[str, Any],
    turn_context: TurnExecutionContext | None,
) -> dict[str, Any] | None:
    """Resolve only Host-declared arguments for an opt-in forced-tool recovery."""
    context_bundle = (
        turn_context.tool_context.context_bundle
        if turn_context is not None
        and isinstance(turn_context.tool_context.context_bundle, Mapping)
        else {}
    )
    argument_sources = as_dict(fallback.get("argument_sources"))
    context_bindings = as_dict(spec.runtime_policy.get("context_argument_bindings"))
    arguments: dict[str, Any] = {}
    for name in set(spec.required_args).union(
        field for group in spec.required_arg_groups for field in group
    ):
        configured_sources = argument_sources.get(name)
        sources = (
            configured_sources
            if isinstance(configured_sources, list)
            else [configured_sources]
            if configured_sources
            else []
        )
        value = next(
            (
                resolved
                for source in sources
                if (
                    resolved := _forced_tool_argument_source(
                        str(source or ""),
                        state=state,
                        context_bundle=context_bundle,
                    )
                )
                not in (None, "")
            ),
            None,
        )
        if value in (None, ""):
            configured_paths = context_bindings.get(name)
            paths = (
                configured_paths
                if isinstance(configured_paths, list)
                else [configured_paths]
                if configured_paths
                else []
            )
            value = next(
                (
                    resolved
                    for path in paths
                    if (
                        resolved := _mapping_path_value(
                            context_bundle,
                            str(path or ""),
                        )
                    )
                    not in (None, "")
                ),
                None,
            )
        if value not in (None, ""):
            arguments[str(name)] = value
    if any(arguments.get(name) in (None, "") for name in spec.required_args):
        return None
    if any(
        not any(arguments.get(name) not in (None, "") for name in group)
        for group in spec.required_arg_groups
    ):
        return None
    return arguments


def _forced_tool_argument_source(
    source: str,
    *,
    state: Mapping[str, Any],
    context_bundle: Mapping[str, Any],
) -> Any:
    if source == "latest_user_message":
        return _latest_user_question(list(state.get("messages") or []))
    if source.startswith("context."):
        return _mapping_path_value(context_bundle, source.removeprefix("context."))
    return None


def _mapping_path_value(document: Mapping[str, Any], path: str) -> Any:
    current: Any = document
    for segment in (part for part in path.split(".") if part):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _emit_hook_audits(
    state: dict[str, Any],
    config: RunnableConfig,
    audits: tuple[HookAudit, ...],
) -> None:
    for audit in audits:
        _emit(
            state,
            config,
            "hook.executed",
            "Lifecycle hook",
            f"{audit.point.value}: {audit.status}",
            {
                "hook_id": audit.hook_id,
                "hook_point": audit.point.value,
                "operation_id": audit.operation_id,
                "status": audit.status,
                "duration_ms": audit.duration_ms,
                "replayed": audit.replayed,
                "required": audit.required,
                "error": audit.error,
                "diagnostics": dict(audit.diagnostics),
                "visible": False,
            },
        )
