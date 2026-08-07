from __future__ import annotations

from dataclasses import dataclass

from deepkeel.context_contracts import ContextItem


@dataclass(frozen=True, slots=True)
class ContextValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_context_items(
    items: list[ContextItem],
    *,
    active_subject_id: str = "",
) -> ContextValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    keys: set[str] = set()
    for item in items:
        if not item.key:
            errors.append("context item key is required")
            continue
        if item.key in keys:
            errors.append(f"duplicate context item: {item.key}")
        keys.add(item.key)
        if item.authority == "derived" and not item.source_ref:
            warnings.append(f"derived context item lacks source_ref: {item.key}")
        if (
            active_subject_id
            and item.subject_id
            and item.subject_id != active_subject_id
            and item.tier in {"L1", "L2"}
        ):
            errors.append(f"subject mismatch for {item.key}")
        if item.visibility == "runtime" and item.model_visible:
            errors.append(f"runtime-only item is model visible: {item.key}")
    return ContextValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
