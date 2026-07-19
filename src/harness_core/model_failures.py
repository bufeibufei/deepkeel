from __future__ import annotations

from dataclasses import dataclass
import socket
from urllib.error import URLError


@dataclass(frozen=True, slots=True)
class ModelFailureInfo:
    category: str
    retryable: bool
    status_code: int | None
    retry_after_seconds: float
    public_message: str


def classify_model_failure(exc: BaseException) -> ModelFailureInfo:
    status_code = _status_code(exc)
    message = str(exc or "").lower()
    retry_after = _retry_after_seconds(exc)
    if status_code == 429 or "too many requests" in message or "rate limit" in message:
        return ModelFailureInfo(
            "rate_limited",
            True,
            status_code or 429,
            retry_after,
            "The model is rate limited; a fallback was attempted.",
        )
    if isinstance(exc, (TimeoutError, socket.timeout)) or any(
        token in message for token in ("timed out", "timeout")
    ):
        return ModelFailureInfo(
            "timeout",
            True,
            status_code,
            retry_after,
            "The model timed out; a fallback was attempted.",
        )
    if status_code in {400, 401, 403, 404, 409, 422}:
        return ModelFailureInfo(
            "invalid_request",
            False,
            status_code,
            0.0,
            "The model request was rejected. Check model configuration.",
        )
    if (
        status_code in {500, 502, 503, 504}
        or isinstance(exc, URLError)
        or any(
            token in message
            for token in (
                "connection refused",
                "connection reset",
                "temporarily unavailable",
                "remote disconnected",
            )
        )
    ):
        return ModelFailureInfo(
            "provider_unavailable",
            True,
            status_code,
            retry_after,
            "The model is unavailable; a fallback was attempted.",
        )
    return ModelFailureInfo(
        "provider_error",
        False,
        status_code,
        0.0,
        "The model call failed. Try again later.",
    )


def provider_fingerprint(provider) -> tuple[str, str, str]:
    if provider is None:
        return ("", "", "")
    return (
        provider.__class__.__qualname__,
        str(getattr(provider, "base_url", "") or ""),
        str(getattr(provider, "model", "") or ""),
    )


def _status_code(exc: BaseException) -> int | None:
    values = (
        getattr(exc, "code", None),
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _retry_after_seconds(exc: BaseException) -> float:
    headers = getattr(exc, "headers", None) or getattr(
        getattr(exc, "response", None), "headers", None
    )
    if headers is None:
        return 0.0
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
        return max(0.0, float(value or 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0
