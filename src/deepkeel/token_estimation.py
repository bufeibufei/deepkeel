from __future__ import annotations

import json
import math
from typing import Any, Protocol


class TokenEstimator(Protocol):
    estimator_id: str

    def estimate(self, value: Any) -> int: ...


class ConservativeTokenEstimator:
    """Dependency-free estimator that is conservative for CJK and JSON text."""

    estimator_id = "conservative-cjk-v1"

    def estimate(self, value: Any) -> int:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
        ascii_count = sum(1 for char in value if ord(char) < 128)
        non_ascii_count = len(value) - ascii_count
        return max(1, math.ceil(ascii_count / 4) + non_ascii_count)


__all__ = ["ConservativeTokenEstimator", "TokenEstimator"]
