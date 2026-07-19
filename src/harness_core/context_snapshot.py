from __future__ import annotations

import hashlib
import json
from typing import Any


CONTEXT_SNAPSHOT_VERSION = "agent-context-v4"
FACT_PACKET_VERSION = "agent-facts-v1"


def normalize_agent_context_snapshot(value: dict[str, Any] | None) -> dict[str, Any]:
    """Build a stable, portable context envelope while preserving legacy fields."""
    raw = dict(value) if isinstance(value, dict) else {}
    profile = raw.get("profile") if isinstance(raw.get("profile"), dict) else {}
    subject_context = raw.get("subject_context") if isinstance(raw.get("subject_context"), dict) else {}
    chart_facts = raw.get("chart_facts") if isinstance(raw.get("chart_facts"), dict) else {}
    latest_reading = raw.get("latest_bazi_reading") if isinstance(raw.get("latest_bazi_reading"), dict) else {}
    subject = _subject_ref(subject_context, profile, raw)
    fact_packet = {
        "schema_version": FACT_PACKET_VERSION,
        "subject": subject,
        "chart": _stable_chart_facts(chart_facts, subject_context),
        "latest_reading": _stable_latest_reading(latest_reading),
    }
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    provenance = {
        **provenance,
        "source": str(provenance.get("source") or raw.get("source") or "runtime"),
        "thread_id": str(provenance.get("thread_id") or raw.get("thread_id") or raw.get("ask_thread_id") or ""),
        "profile_id": subject["profile_id"],
    }
    return {
        **raw,
        "schema_version": CONTEXT_SNAPSHOT_VERSION,
        "subject": subject,
        "fact_packet": fact_packet,
        "fact_packet_hash": _stable_hash(fact_packet),
        "provenance": provenance,
    }


def context_snapshot_subject(value: dict[str, Any] | None) -> dict[str, str]:
    snapshot = normalize_agent_context_snapshot(value)
    subject = snapshot.get("subject") if isinstance(snapshot.get("subject"), dict) else {}
    return {
        "mode": str(subject.get("mode") or "none"),
        "subject_id": str(subject.get("subject_id") or ""),
        "profile_id": str(subject.get("profile_id") or ""),
        "name": str(subject.get("name") or ""),
        "relationship": str(subject.get("relationship") or ""),
        "fact_packet_hash": str(snapshot.get("fact_packet_hash") or ""),
    }


def _subject_ref(subject_context: dict, profile: dict, raw: dict) -> dict[str, str]:
    mode = str(subject_context.get("mode") or "").strip()
    profile_id = str(
        subject_context.get("profile_id")
        or profile.get("birth_profile_id")
        or profile.get("id")
        or raw.get("birth_profile_id")
        or ""
    ).strip()
    if mode not in {"saved_profile", "temporary"}:
        mode = "saved_profile" if profile_id else "none"
    name = str(subject_context.get("name") or profile.get("name") or "").strip()
    relationship = str(
        subject_context.get("relationship")
        or ("self" if mode == "saved_profile" else "other" if mode == "temporary" else "")
    ).strip()
    if profile_id:
        subject_id = f"profile:{profile_id}"
    elif mode == "temporary":
        identity = {
            "name": name,
            "relationship": relationship,
            "profile": subject_context.get("profile") if isinstance(subject_context.get("profile"), dict) else {},
            "pillars": subject_context.get("pillars") if isinstance(subject_context.get("pillars"), dict) else {},
        }
        subject_id = f"temporary:{_stable_hash(identity)[:20]}"
    else:
        subject_id = ""
    return {
        "mode": mode,
        "subject_id": subject_id,
        "profile_id": profile_id,
        "name": name,
        "relationship": relationship,
    }


def _stable_chart_facts(chart_facts: dict, subject_context: dict) -> dict[str, Any]:
    pillars = chart_facts.get("pillars") if isinstance(chart_facts.get("pillars"), dict) else {}
    if not pillars and isinstance(subject_context.get("pillars"), dict):
        pillars = subject_context["pillars"]
    return {
        "pillars": {
            "year": str(pillars.get("year") or ""),
            "month": str(pillars.get("month") or ""),
            "day": str(pillars.get("day") or ""),
            "hour": str(pillars.get("hour") or ""),
        },
        "day_master": str(chart_facts.get("day_master") or subject_context.get("day_master") or ""),
        "calculation_method": str(chart_facts.get("calculation_method") or ""),
    }


def _stable_latest_reading(latest_reading: dict) -> dict[str, Any]:
    domains = latest_reading.get("domains") if isinstance(latest_reading.get("domains"), list) else []
    return {
        "session_id": str(latest_reading.get("session_id") or ""),
        "summary": str(latest_reading.get("summary") or ""),
        "domains": [dict(item) for item in domains if isinstance(item, dict)],
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
