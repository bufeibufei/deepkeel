from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from deepkeel.context_snapshot import normalize_agent_context_snapshot
from deepkeel.contracts import AgentMessage, MessageContentPart


def build_initial_messages(
    question: str,
    short_context: dict[str, Any] | None,
    context_bundle: dict[str, Any] | None,
    *,
    history_limit: int = 0,
    input_parts: list[MessageContentPart] | None = None,
) -> list[AgentMessage]:
    short = short_context if isinstance(short_context, dict) else {}
    bundle = context_bundle if isinstance(context_bundle, dict) else {}
    messages: list[AgentMessage] = []
    runtime_context = runtime_context_payload(short, bundle)
    tier_payloads = bundle.get("context_tier_payloads")
    tier_payloads = tier_payloads if isinstance(tier_payloads, dict) else {}
    if tier_payloads:
        preamble = str(
            bundle.get("context_preamble")
            or "Available runtime context (facts and history only):"
        ).strip()
        for tier in ("L1", "L2", "L3"):
            payload = tier_payloads.get(tier)
            if not isinstance(payload, dict) or not payload:
                continue
            messages.append(
                AgentMessage(
                    id=f"context-{tier.lower()}-{uuid4()}",
                    role="system",
                    content=(
                        f"{preamble}\nContext tier {tier}:\n"
                        + json.dumps(payload, ensure_ascii=False, default=str)
                    ),
                    metadata={
                        "context_tier": tier,
                        "context_authority": "canonical" if tier == "L1" else "derived",
                    },
                )
            )
    elif runtime_context:
        preamble = str(
            bundle.get("context_preamble")
            or "Available runtime context (facts and history only):"
        ).strip()
        messages.append(
            AgentMessage(
                id=f"context-{uuid4()}",
                role="system",
                content=f"{preamble}\n"
                + json.dumps(runtime_context, ensure_ascii=False, default=str),
                metadata={"context_tier": "L1"},
            )
        )
    recent = bundle.get("recent_messages")
    if not isinstance(recent, list):
        envelope = bundle.get("runtime_context")
        recent = envelope.get("recent_messages") if isinstance(envelope, dict) else []
    if not isinstance(recent, list):
        recent = []
    selected_recent = recent[-int(history_limit) :] if int(history_limit) > 0 else recent
    for raw in selected_recent:
        if not isinstance(raw, dict) or raw.get("role") not in {"user", "assistant"}:
            continue
        content = str(raw.get("content") or "").strip()
        content_parts = _validated_content_parts(raw.get("content_parts"))
        if not content and not content_parts:
            continue
        messages.append(
            AgentMessage(
                id=str(raw.get("id") or f"history-{uuid4()}"),
                role="user" if raw["role"] == "user" else "assistant",
                content=content,
                content_parts=content_parts,
                metadata={
                    "history": True,
                    "created_at": raw.get("created_at"),
                    "context_tier": "L2",
                    "context_retention": "normal",
                    "source_ref": str(raw.get("id") or ""),
                },
            )
        )
    current_parts = list(input_parts or [])
    if not messages or messages[-1].role != "user" or messages[-1].content.strip() != question.strip():
        messages.append(
            AgentMessage(
                id=f"user-{uuid4()}",
                role="user",
                content=question,
                content_parts=current_parts,
                metadata={
                    "context_tier": "L2",
                    "context_retention": "protected",
                    "context_authority": "canonical",
                },
            )
        )
    else:
        metadata = {
            **dict(messages[-1].metadata),
            "context_tier": "L2",
            "context_retention": "protected",
            "context_authority": "canonical",
        }
        update: dict[str, Any] = {"metadata": metadata}
        if current_parts:
            update["content_parts"] = current_parts
        messages[-1] = messages[-1].model_copy(update=update)
    return messages


def _validated_content_parts(value: Any) -> list[MessageContentPart]:
    if not isinstance(value, list):
        return []
    result: list[MessageContentPart] = []
    for item in value:
        if isinstance(item, MessageContentPart):
            result.append(item)
            continue
        if not isinstance(item, dict):
            continue
        try:
            result.append(MessageContentPart.model_validate(item))
        except (TypeError, ValueError):
            continue
    return result


def runtime_context_payload(
    short_context: dict[str, Any] | None,
    context_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    short = short_context if isinstance(short_context, dict) else {}
    bundle = context_bundle if isinstance(context_bundle, dict) else {}
    supplied = bundle.get("runtime_context")
    context = dict(supplied) if isinstance(supplied, dict) else {}
    # Recent messages are injected as role messages and must not be duplicated
    # inside the system context envelope.
    context.pop("recent_messages", None)
    if short.get("current_time") not in (None, "", [], {}):
        context.setdefault("current_time", short["current_time"])
    return context


def build_context_snapshot(
    question: str,
    context_bundle: dict[str, Any] | None,
    short_context: dict[str, Any] | None,
    skill_activation: dict[str, Any] | None,
) -> dict[str, Any]:
    bundle = context_bundle if isinstance(context_bundle, dict) else {}
    short = short_context if isinstance(short_context, dict) else {}
    supplied = bundle.get("context_snapshot")
    snapshot = dict(supplied) if isinstance(supplied, dict) else {}
    snapshot.update(
        {
            "question": question,
            "thread_id": str(bundle.get("thread_id") or snapshot.get("thread_id") or ""),
            "current_time": short.get("current_time")
            if isinstance(short.get("current_time"), dict)
            else {},
            "skill_activation": dict(skill_activation or {}),
        }
    )
    return normalize_agent_context_snapshot(snapshot)
