from __future__ import annotations

import hashlib
import json
from typing import Any


CONTEXT_SNAPSHOT_VERSION = "harness-context-v1"
FACT_PACKET_VERSION = "harness-facts-v1"


def normalize_agent_context_snapshot(value: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a domain-neutral, immutable context snapshot envelope."""

    raw = dict(value) if isinstance(value, dict) else {}
    supplied_subject = raw.get("subject")
    if not isinstance(supplied_subject, dict):
        supplied_subject = raw.get("subject_context")
    subject = _subject_ref(supplied_subject)
    supplied_facts = raw.get("facts")
    if not isinstance(supplied_facts, dict):
        supplied_facts = raw.get("fact_packet")
    facts = dict(supplied_facts) if isinstance(supplied_facts, dict) else {}
    facts.setdefault("schema_version", FACT_PACKET_VERSION)
    facts.setdefault("subject", subject)
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    provenance = {
        **provenance,
        "source": str(provenance.get("source") or raw.get("source") or "runtime"),
        "thread_id": str(provenance.get("thread_id") or raw.get("thread_id") or ""),
        "subject_id": subject["subject_id"],
        "profile_id": subject["profile_id"],
    }
    facts_hash = _stable_hash(facts)
    return {
        **raw,
        "schema_version": CONTEXT_SNAPSHOT_VERSION,
        "subject": subject,
        "facts": facts,
        # Compatibility aliases retained while product persistence migrates.
        "fact_packet": facts,
        "facts_hash": facts_hash,
        "fact_packet_hash": facts_hash,
        "provenance": provenance,
    }


def context_snapshot_subject(value: dict[str, Any] | None) -> dict[str, str]:
    snapshot = normalize_agent_context_snapshot(value)
    subject = snapshot["subject"]
    return {
        "mode": subject["mode"],
        "subject_id": subject["subject_id"],
        "profile_id": subject["profile_id"],
        "name": subject["name"],
        "relationship": subject["relationship"],
        "fact_packet_hash": str(snapshot.get("facts_hash") or ""),
    }


def _subject_ref(value: Any) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    profile_id = str(source.get("profile_id") or "")
    subject_id = str(source.get("subject_id") or "")
    if not subject_id and profile_id:
        subject_id = f"profile:{profile_id}"
    return {
        "mode": str(source.get("mode") or "none"),
        "subject_id": subject_id,
        "profile_id": profile_id,
        "name": str(source.get("name") or ""),
        "relationship": str(source.get("relationship") or ""),
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
