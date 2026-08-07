from __future__ import annotations

import copy
from datetime import UTC, datetime
import hashlib
import json
from typing import Any

from deepkeel.adapter_sdk import RunLease
from deepkeel.runtime_sdk import (
    RunAggregate,
    RunStateSnapshot,
    RuntimeStateConflict,
    RuntimeStateMutation,
)


def aggregate_from_row(row: dict[str, Any]) -> RunAggregate:
    return RunAggregate(
        run_id=str(row["run_id"]),
        version=int(row["version"]),
        sequence=int(row["sequence"]),
        status=str(row["status"]),
        settled=bool(row["settled"]),
        settlement_status=str(row["settlement_status"]),
        last_event_type=str(row["last_event_type"]),
        checkpoint_type=str(row["checkpoint_type"]),
        checkpoint_state=copy.deepcopy(row["checkpoint_state"] or {}),
        resume_token=str(row["resume_token"]),
        fence_token=str(row["fence_token"]),
        fence_generation=int(row["fence_generation"]),
        updated_at=aware(row["updated_at"]),
    )


def snapshot_from_row(run_id: str, row: dict[str, Any] | None) -> RunStateSnapshot:
    if row is None:
        return RunStateSnapshot(run_id=run_id)
    return aggregate_from_row(row).snapshot()


def assert_expected_state(row: dict[str, Any], mutation: RuntimeStateMutation) -> None:
    if mutation.expected_version is not None and mutation.expected_version != int(
        row["version"]
    ):
        raise RuntimeStateConflict(
            f"runtime version changed: expected {mutation.expected_version}, found {row['version']}"
        )
    if mutation.expected_sequence is not None and mutation.expected_sequence != int(
        row["sequence"]
    ):
        raise RuntimeStateConflict(
            "runtime sequence changed: "
            f"expected {mutation.expected_sequence}, found {row['sequence']}"
        )


def assert_current_fence(row: dict[str, Any], mutation: RuntimeStateMutation) -> None:
    generation = int(mutation.fence_generation or 0)
    token = str(mutation.fence_token or "")
    current_generation = int(row["fence_generation"] or 0)
    current_token = str(row["fence_token"] or "")
    if generation < 0:
        raise ValueError("fence_generation must be non-negative")
    if generation and not token:
        raise ValueError("fence_token is required when fence_generation is set")
    if current_generation and not generation:
        raise RuntimeStateConflict("fenced run requires an execution fence")
    if generation < current_generation:
        raise RuntimeStateConflict(
            f"stale execution fence: expected generation {current_generation}, found {generation}"
        )
    if generation == current_generation and generation and token != current_token:
        raise RuntimeStateConflict("execution fence token does not match current owner")


def lease_from_row(row: dict[str, Any]) -> RunLease:
    return RunLease(
        run_id=str(row["run_id"]),
        owner_id=str(row["owner_id"]),
        token=str(row["token"]),
        acquired_at=aware(row["acquired_at"]),
        expires_at=aware(row["expires_at"]),
        generation=int(row["generation"]),
    )


def advisory_lock(cursor: Any, value: str) -> None:
    cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (value,))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json_value(value).encode("utf-8")).hexdigest()


def json_value(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return aware(value).isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def validated_ttl(value: float) -> float:
    ttl = float(value)
    if ttl <= 0:
        raise ValueError("ttl_seconds must be positive")
    return ttl
