from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deepkeel.context_contracts import ContextItem


ContextQualityMode = Literal["audit", "enforce"]
ContextQualitySeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class ContextQualityPolicy:
    """Tier-aware provenance policy for model-visible context."""

    mode: ContextQualityMode = "audit"
    require_l1_source: bool = True
    require_derived_source_ref: bool = True
    require_summary_fingerprint: bool = True
    quarantine_severities: tuple[ContextQualitySeverity, ...] = ("error",)
    policy_id: str = "tiered-context-quality-v1"


@dataclass(frozen=True, slots=True)
class ContextQualityIssue:
    code: str
    key: str
    tier: str
    severity: ContextQualitySeverity
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "key": self.key,
            "tier": self.tier,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ContextQualityReport:
    policy_id: str
    mode: ContextQualityMode
    issues: tuple[ContextQualityIssue, ...] = ()
    quarantined_keys: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "harness-context-quality-v1",
            "policy_id": self.policy_id,
            "mode": self.mode,
            "valid": self.valid,
            "issues": [item.as_dict() for item in self.issues],
            "quarantined_keys": list(self.quarantined_keys),
        }


class ContextQualityGate:
    """Evaluate context quality without owning or mutating payloads."""

    def __init__(self, policy: ContextQualityPolicy | None = None) -> None:
        self.policy = policy or ContextQualityPolicy()

    def evaluate(
        self,
        items: list[ContextItem],
        *,
        active_subject_id: str = "",
    ) -> ContextQualityReport:
        issues: list[ContextQualityIssue] = []
        seen: set[str] = set()
        for item in items:
            if item.key in seen:
                issues.append(_issue(item, "duplicate_key", "error", "Context key is duplicated."))
            seen.add(item.key)
            if (
                active_subject_id
                and item.subject_id
                and item.subject_id != active_subject_id
                and item.tier in {"L1", "L2"}
            ):
                issues.append(
                    _issue(
                        item,
                        "subject_mismatch",
                        "error",
                        "Context belongs to a different active subject.",
                    )
                )
            if (
                self.policy.require_l1_source
                and item.tier == "L1"
                and item.model_visible
                and not (item.source or item.source_ref)
            ):
                issues.append(
                    _issue(item, "l1_source_missing", "error", "L1 context has no source.")
                )
            if (
                self.policy.require_derived_source_ref
                and item.authority == "derived"
                and item.tier in {"L1", "L2"}
                and not item.source_ref
            ):
                severity: ContextQualitySeverity = "error" if item.tier == "L1" else "warning"
                issues.append(
                    _issue(
                        item,
                        "derived_source_missing",
                        severity,
                        "Derived context has no source reference.",
                    )
                )
            if item.representation == "pointer" and not item.source_ref:
                issues.append(
                    _issue(
                        item,
                        "pointer_target_missing",
                        "error",
                        "Pointer context has no target reference.",
                    )
                )
            if (
                self.policy.require_summary_fingerprint
                and (item.summary not in (None, "", [], {}) or item.cache_key)
                and not item.source_fingerprint
            ):
                issues.append(
                    _issue(
                        item,
                        "summary_fingerprint_missing",
                        "warning",
                        "Cached or summarized context has no source fingerprint.",
                    )
                )
            if item.summary not in (None, "", [], {}) and not item.summary_version:
                issues.append(
                    _issue(
                        item,
                        "summary_version_missing",
                        "warning",
                        "Summarized context has no summary version.",
                    )
                )
            if item.tier == "L3" and item.protected:
                issues.append(
                    _issue(
                        item,
                        "l3_overprotected",
                        "warning",
                        "Retrieved L3 context should not normally be protected or required.",
                    )
                )
        quarantine: tuple[str, ...] = ()
        if self.policy.mode == "enforce":
            severities = set(self.policy.quarantine_severities)
            quarantine = tuple(
                dict.fromkeys(item.key for item in issues if item.severity in severities)
            )
        return ContextQualityReport(
            policy_id=self.policy.policy_id,
            mode=self.policy.mode,
            issues=tuple(issues),
            quarantined_keys=quarantine,
        )


def _issue(
    item: ContextItem,
    code: str,
    severity: ContextQualitySeverity,
    message: str,
) -> ContextQualityIssue:
    return ContextQualityIssue(
        code=code,
        key=item.key,
        tier=item.tier,
        severity=severity,
        message=message,
    )


__all__ = [
    "ContextQualityGate",
    "ContextQualityIssue",
    "ContextQualityMode",
    "ContextQualityPolicy",
    "ContextQualityReport",
    "ContextQualitySeverity",
]
