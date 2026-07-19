from __future__ import annotations

import math
import time

from harness_core.failures import RunDeadlineExceededError


def ensure_time_remaining(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise RunDeadlineExceededError()


def remaining_timeout_ceiling(
    deadline_monotonic: float | None,
    *,
    maximum: int,
) -> int:
    if deadline_monotonic is None:
        return max(1, int(maximum))
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise RunDeadlineExceededError()
    return max(1, min(int(maximum), int(math.ceil(remaining))))


def deadline_with_timeout(
    parent_deadline_monotonic: float | None,
    timeout_seconds: float | None,
) -> float | None:
    local_deadline = (
        time.monotonic() + max(0.001, timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    candidates = [
        value
        for value in (parent_deadline_monotonic, local_deadline)
        if value is not None
    ]
    return min(candidates) if candidates else None
