from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field


CapabilityExecutionMode = Literal["trusted_in_process", "isolated"]


class CapabilityPackageSource(BaseModel):
    """Host-supplied provenance for an installable capability package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str = Field(min_length=1)
    execution_mode: CapabilityExecutionMode = "trusted_in_process"
    source_uri: str = ""
    content_sha256: str = ""


class CapabilityTrustPolicy(BaseModel):
    """Host policy for code that may execute inside the runtime process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_in_process_digests: frozenset[str] = frozenset()
    allowed_isolated_sources: tuple[str, ...] = ()
    allow_unverified_in_process: bool = False


class CapabilityTrustReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    execution_mode: CapabilityExecutionMode
    trusted: bool
    reason: str
    content_sha256: str = ""


def capability_source_digest(*paths: str | Path) -> str:
    """Hash named package files deterministically, independent of absolute paths."""

    sources = sorted((Path(path) for path in paths), key=lambda item: item.name)
    if not sources:
        raise ValueError("at least one capability source path is required")
    digest = hashlib.sha256()
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_capability_trust(
    source: CapabilityPackageSource,
    policy: CapabilityTrustPolicy,
) -> CapabilityTrustReport:
    """Evaluate provenance before a Host imports or connects to a package."""

    digest = source.content_sha256.strip().lower()
    if source.execution_mode == "trusted_in_process":
        trusted = policy.allow_unverified_in_process or (
            bool(digest) and digest in policy.allowed_in_process_digests
        )
        reason = (
            "in-process package digest is trusted"
            if trusted and digest
            else "Host explicitly permits unverified in-process packages"
            if trusted
            else "in-process Python packages require an allowlisted SHA-256 digest"
        )
    else:
        trusted = any(
            _uri_is_within_prefix(source.source_uri, prefix)
            for prefix in policy.allowed_isolated_sources
        )
        reason = (
            "isolated package source is allowed"
            if trusted
            else "isolated package source is not allowed by Host policy"
        )
    return CapabilityTrustReport(
        package_id=source.package_id,
        execution_mode=source.execution_mode,
        trusted=trusted,
        reason=reason,
        content_sha256=digest,
    )


def _uri_is_within_prefix(source_uri: str, allowed_prefix: str) -> bool:
    source = urlsplit(str(source_uri or ""))
    allowed = urlsplit(str(allowed_prefix or ""))
    if not source.scheme or not source.hostname or not allowed.scheme or not allowed.hostname:
        return False
    source_port = source.port or (443 if source.scheme.lower() == "https" else 80)
    allowed_port = allowed.port or (443 if allowed.scheme.lower() == "https" else 80)
    if (
        source.scheme.lower() != allowed.scheme.lower()
        or source.hostname.lower() != allowed.hostname.lower()
        or source_port != allowed_port
    ):
        return False
    allowed_path = "/" + str(allowed.path or "").strip("/")
    source_path = "/" + str(source.path or "").strip("/")
    if allowed_path == "/":
        return True
    return source_path == allowed_path or source_path.startswith(f"{allowed_path}/")


__all__ = [
    "CapabilityExecutionMode",
    "CapabilityPackageSource",
    "CapabilityTrustPolicy",
    "CapabilityTrustReport",
    "capability_source_digest",
    "evaluate_capability_trust",
]
