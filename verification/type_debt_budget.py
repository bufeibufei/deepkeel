from __future__ import annotations

import re
from pathlib import Path


MAX_ANY_REFERENCES = 1657
MAX_TYPE_IGNORES = 15


def main() -> int:
    source_root = Path("src/deepkeel")
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(source_root.rglob("*.py"))
    )
    any_references = len(re.findall(r"\bAny\b", source))
    type_ignores = len(re.findall(r"#\s*type:\s*ignore", source))
    failures: list[str] = []
    if any_references > MAX_ANY_REFERENCES:
        failures.append(
            f"Any references {any_references} exceed the ratchet {MAX_ANY_REFERENCES}"
        )
    if type_ignores > MAX_TYPE_IGNORES:
        failures.append(
            f"type ignores {type_ignores} exceed the ratchet {MAX_TYPE_IGNORES}"
        )
    if failures:
        for failure in failures:
            print(f"TYPE_DEBT_BUDGET_FAILED: {failure}")
        return 1
    print(
        "DeepKeel type debt budget passed: "
        f"Any={any_references}/{MAX_ANY_REFERENCES}, "
        f"type-ignore={type_ignores}/{MAX_TYPE_IGNORES}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
