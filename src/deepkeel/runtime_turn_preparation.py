from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from deepkeel.persistence import CheckpointCompatibilityError
from deepkeel.runtime_api import RuntimeRequest
from deepkeel.runtime_execution_support import optional_int
from deepkeel.runtime_policy import (
    _conservative_model_context_profile,
    _model_providers,
    _resolved_model_policy,
)
from deepkeel.type_narrowing import as_dict


@dataclass(frozen=True, slots=True)
class PreparedTurnInputs:
    short: dict[str, Any]
    bundle: dict[str, Any]
    resolved_model_policy: dict[str, Any]
    model_providers: dict[str, Any]
    model_context_profile: Any
    configured_input_limit: int
    context_window_diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedTurnIdentity:
    run_id: str
    operational_run_id: str
    conversation_thread_id: str
    graph_thread_id: str
    turn_id: str
    durable_state: dict[str, Any]
    checkpoint_authority: str
    checkpoint_load_errors: list[str]
    event_sequence: int
    run_version: int


async def prepare_turn_inputs(
    runtime: Any,
    request: RuntimeRequest,
    *,
    provider: Any = None,
    providers: dict[str, Any] | None = None,
) -> PreparedTurnInputs:
    short = request.short_context if isinstance(request.short_context, dict) else {}
    bundle = dict(request.context_bundle) if isinstance(request.context_bundle, dict) else {}
    scope = request.runtime_scope
    for key, value in (
        ("run_id", request.run_id),
        ("thread_id", request.thread_id),
        ("tenant_id", scope.tenant_id),
        ("user_id", scope.user_id),
    ):
        if value:
            bundle.setdefault(key, value)
    if request.skill_activation:
        bundle["skill_activation"] = dict(request.skill_activation)

    resolved_model_policy = _resolved_model_policy(
        request.model_policy,
        provider=provider,
        providers=providers,
        max_steps=runtime.max_steps,
    )
    model_providers = _model_providers(provider, providers, resolved_model_policy)
    model_context_profile = _conservative_model_context_profile(
        model_providers,
        resolved_model_policy,
    )
    configured_input_limit = int(
        as_dict(resolved_model_policy.get("budget")).get("max_input_tokens_per_call") or 0
    )
    if runtime.memory_recall_coordinator is not None:
        bundle = await runtime.memory_recall_coordinator.prepare(request.question, short, bundle)
        if not isinstance(bundle, dict):
            raise TypeError("memory recall coordinator must return a mapping")
    if runtime.context_builder is not None:
        bundle = runtime.context_builder(request.question, short, bundle)
        if not isinstance(bundle, dict):
            raise TypeError("context builder must return a mapping")
    for contributor_id, contributor in runtime.capability_catalog.context_contributors.items():
        contributed = contributor(dict(bundle))
        if not isinstance(contributed, dict):
            raise TypeError(f"context contributor {contributor_id} must return a mapping")
        bundle = contributed
    bundle["_model_context_profile"] = model_context_profile.as_dict()
    bundle["_configured_input_limit"] = configured_input_limit
    prepared_context = runtime.context_window_manager.prepare(
        request.question,
        short,
        bundle,
    )
    return PreparedTurnInputs(
        short=short,
        bundle=prepared_context.context_bundle,
        resolved_model_policy=resolved_model_policy,
        model_providers=model_providers,
        model_context_profile=model_context_profile,
        configured_input_limit=configured_input_limit,
        context_window_diagnostics=dict(prepared_context.diagnostics),
    )


async def prepare_turn_identity(
    runtime: Any,
    request: RuntimeRequest,
    inputs: PreparedTurnInputs,
    *,
    session: Any = None,
) -> PreparedTurnIdentity:
    scope = request.runtime_scope
    short = inputs.short
    bundle = inputs.bundle
    run_id = str(
        request.run_id
        or bundle.get("agent_session_id")
        or bundle.get("agent_run_id")
        or bundle.get("run_id")
        or uuid4()
    )
    if short.get("resume"):
        durable_state, authority, load_errors = await runtime._aload_authoritative_checkpoint(
            run_id,
            session=session,
            user_id=str(scope.user_id or "local-device"),
            scope=scope,
        )
    else:
        durable_state, authority, load_errors = {}, "none", []
    try:
        from deepkeel.runtime_execution_support import ensure_resume_generation_compatible

        ensure_resume_generation_compatible(runtime.runtime_generation, durable_state)
    except CheckpointCompatibilityError:
        raise

    operational_run_id = scope.qualify_identity(run_id)
    conversation_thread_id = str(
        request.thread_id
        or bundle.get("thread_id")
        or bundle.get("ask_thread_id")
        or short.get("ask_thread_id")
        or run_id
    )
    turn_id = str(
        request.turn_id or bundle.get("turn_id") or short.get("turn_id") or f"turn-{uuid4()}"
    )
    bundle.update(
        {
            "operational_run_id": operational_run_id,
            "tenant_id": scope.tenant_id,
            "namespace": scope.namespace,
        }
    )
    event_sequence = await runtime._event_latest_sequence(
        run_id,
        scope=scope,
        fallback=max(
            0,
            optional_int(bundle.get("event_sequence")) or 0,
            optional_int(short.get("event_sequence")) or 0,
        ),
    )
    run_version = max(
        0,
        optional_int(bundle.get("run_version")) or 0,
        optional_int(short.get("run_version")) or 0,
    )
    return PreparedTurnIdentity(
        run_id=run_id,
        operational_run_id=operational_run_id,
        conversation_thread_id=conversation_thread_id,
        graph_thread_id=operational_run_id,
        turn_id=turn_id,
        durable_state=durable_state,
        checkpoint_authority=authority,
        checkpoint_load_errors=list(load_errors),
        event_sequence=event_sequence,
        run_version=run_version,
    )
