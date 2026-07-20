from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def as_dict(value: object) -> dict[str, Any]:
    """Narrow an untrusted JSON-like value to a string-keyed dictionary."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def as_optional_dict(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    return as_dict(value) if isinstance(value, Mapping) else None


def as_list(value: object) -> list[Any]:
    """Narrow an untrusted JSON-like value to a list without accepting text."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(value)


def as_dict_list(value: object) -> list[dict[str, Any]]:
    return [as_dict(item) for item in as_list(value) if isinstance(item, Mapping)]
