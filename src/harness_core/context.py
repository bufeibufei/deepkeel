from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from harness_core.contracts import AgentMessage
from harness_core.context_snapshot import normalize_agent_context_snapshot


def build_initial_messages(
    question: str,
    short_context: dict[str, Any] | None,
    context_bundle: dict[str, Any] | None,
    *,
    history_limit: int = 8,
) -> list[AgentMessage]:
    short = short_context if isinstance(short_context, dict) else {}
    bundle = context_bundle if isinstance(context_bundle, dict) else {}
    messages: list[AgentMessage] = []
    runtime_context = runtime_context_payload(short, bundle)
    if runtime_context:
        messages.append(
            AgentMessage(
                id=f"context-{uuid4()}",
                role="system",
                content="本轮可用上下文（仅作事实与历史参考）：\n"
                + json.dumps(runtime_context, ensure_ascii=False, default=str),
            )
        )
    recent = bundle.get("recent_messages") if isinstance(bundle.get("recent_messages"), list) else []
    for raw in recent[-max(1, int(history_limit)) :]:
        if not isinstance(raw, dict) or raw.get("role") not in {"user", "assistant"}:
            continue
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        messages.append(
            AgentMessage(
                id=str(raw.get("id") or f"history-{uuid4()}"),
                role=str(raw["role"]),
                content=content,
                metadata={"history": True, "created_at": raw.get("created_at")},
            )
        )
    if not messages or messages[-1].role != "user" or messages[-1].content.strip() != question.strip():
        messages.append(AgentMessage(id=f"user-{uuid4()}", role="user", content=question))
    return messages


def runtime_context_payload(
    short_context: dict[str, Any] | None,
    context_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    short = short_context if isinstance(short_context, dict) else {}
    bundle = context_bundle if isinstance(context_bundle, dict) else {}
    keys = (
        "subject_context",
        "active_profile",
        "available_profiles",
        "chart_facts",
        "latest_bazi_reading",
        "compressed_history",
        "long_term_memories",
        "retrieved_evidence",
        "response_policy",
        "source",
    )
    context = {key: bundle[key] for key in keys if bundle.get(key) not in (None, "", [], {})}
    memories = context.get("long_term_memories")
    if isinstance(memories, list):
        sanitized, suppressed = _sanitize_runtime_memories(
            memories,
            active_profile=context.get("active_profile"),
            subject_context=context.get("subject_context"),
        )
        if sanitized:
            context["long_term_memories"] = sanitized
        else:
            context.pop("long_term_memories", None)
        context["context_policy"] = {
            "authoritative_sources": ["subject_context", "active_profile", "chart_facts"],
            "supplementary_sources": ["latest_bazi_reading", "long_term_memories", "compressed_history"],
            "rule": "Current entity facts override memory and conversation summaries; never ask the user to resolve a conflict already settled by an authoritative source.",
            "suppressed_memory_count": suppressed,
        }
    for key in ("current_time", "profile", "profile_context", "latest_reading_summary"):
        if short.get(key) not in (None, "", [], {}):
            context[key] = short[key]
    return context


def _sanitize_runtime_memories(
    memories: list[Any],
    *,
    active_profile: Any,
    subject_context: Any,
) -> tuple[list[dict[str, Any]], int]:
    has_authoritative_entity = bool(
        isinstance(active_profile, dict)
        and any(active_profile.get(key) not in (None, "") for key in ("birth_profile_id", "birth_datetime", "name"))
    ) or bool(isinstance(subject_context, dict) and subject_context.get("mode"))
    sanitized: list[dict[str, Any]] = []
    suppressed = 0
    for item in memories:
        if not isinstance(item, dict) or item.get("runtime_eligible") is False:
            suppressed += 1
            continue
        canonical_key = str(item.get("canonical_key") or "").strip().lower()
        conflicts_with_entity = has_authoritative_entity and canonical_key.startswith(
            ("profile.", "chart.", "bazi.", "liuyao.")
        )
        if conflicts_with_entity:
            suppressed += 1
            continue
        sanitized.append(item)
    return sanitized, suppressed


def build_context_snapshot(
    question: str,
    context_bundle: dict[str, Any] | None,
    short_context: dict[str, Any] | None,
    skill_activation: dict[str, Any] | None,
) -> dict[str, Any]:
    bundle = context_bundle if isinstance(context_bundle, dict) else {}
    short = short_context if isinstance(short_context, dict) else {}
    skill = skill_activation if isinstance(skill_activation, dict) else {}
    return normalize_agent_context_snapshot({
        "question": question,
        "thread_id": str(bundle.get("thread_id") or ""),
        "source": str(bundle.get("source") or "runtime"),
        "profile": bundle.get("active_profile") if isinstance(bundle.get("active_profile"), dict) else {},
        "available_profiles": bundle.get("available_profiles")
        if isinstance(bundle.get("available_profiles"), list)
        else [],
        "subject_context": bundle.get("subject_context") if isinstance(bundle.get("subject_context"), dict) else {},
        "chart_facts": bundle.get("chart_facts") if isinstance(bundle.get("chart_facts"), dict) else {},
        "latest_bazi_reading": bundle.get("latest_bazi_reading")
        if isinstance(bundle.get("latest_bazi_reading"), dict)
        else {},
        "current_time": short.get("current_time") if isinstance(short.get("current_time"), dict) else {},
        "skill_activation": skill,
    })
