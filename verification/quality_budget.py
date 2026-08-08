from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CRITICAL_COVERAGE = {
    "src/deepkeel/async_ports.py": 60.0,
    "src/deepkeel/event_journal.py": 90.0,
    "src/deepkeel/events.py": 95.0,
    "src/deepkeel/leases.py": 80.0,
    "src/deepkeel/mcp/contracts.py": 80.0,
    "src/deepkeel/mcp/provider.py": 70.0,
    "src/deepkeel/mcp/streamable_http.py": 80.0,
    "src/deepkeel/model_invocations.py": 85.0,
    "src/deepkeel/model_step_execution.py": 80.0,
    "src/deepkeel/operations.py": 85.0,
    "src/deepkeel/production.py": 90.0,
    "src/deepkeel/runtime.py": 70.0,
    "src/deepkeel/runtime_events.py": 75.0,
    "src/deepkeel/runtime_persistence.py": 70.0,
    "src/deepkeel/runtime_settlement.py": 90.0,
    "src/deepkeel/runtime_turn_execution.py": 85.0,
    "src/deepkeel/scope.py": 90.0,
    "src/deepkeel/tool_execution.py": 80.0,
    "src/deepkeel/tool_executor.py": 70.0,
}

CRITICAL_COMPLEXITY_FILES = tuple(CRITICAL_COVERAGE)
MAX_CYCLOMATIC_COMPLEXITY = 25
MAX_FUNCTION_LINES = 180
FUNCTION_BUDGET_OVERRIDES = {
    "execute_model_attempt": (28, MAX_FUNCTION_LINES),
    "persist_runtime_snapshot": (33, MAX_FUNCTION_LINES),
    "project_and_settle_runtime_result": (MAX_CYCLOMATIC_COMPLEXITY, 183),
    "RuntimeTurnExecutionMixin._arun_claimed": (63, 418),
    "ToolExecutor._aexecute_core": (29, 248),
}


@dataclass(frozen=True, slots=True)
class FunctionComplexity:
    path: str
    name: str
    line: int
    lines: int
    complexity: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DeepKeel quality ratchets")
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    failures: list[str] = []
    if args.coverage is not None:
        failures.extend(verify_critical_coverage(args.coverage, root=args.root))
    failures.extend(verify_complexity(root=args.root))
    if failures:
        for failure in failures:
            print(f"QUALITY_BUDGET_FAILED: {failure}")
        return 1
    print("DeepKeel critical coverage and complexity budgets passed.")
    return 0


def verify_critical_coverage(coverage_path: Path, *, root: Path) -> list[str]:
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else {}
    normalized = {
        _normalized_path(path, root): item
        for path, item in (files.items() if isinstance(files, dict) else [])
    }
    failures: list[str] = []
    for path, minimum in CRITICAL_COVERAGE.items():
        item = normalized.get(path)
        if not isinstance(item, dict):
            failures.append(f"critical coverage is missing for {path}")
            continue
        raw_summary = item.get("summary")
        summary = raw_summary if isinstance(raw_summary, dict) else {}
        covered = float(summary.get("percent_covered") or 0.0)
        if covered + 1e-9 < minimum:
            failures.append(f"{path} coverage {covered:.2f}% is below {minimum:.2f}%")
    return failures


def verify_complexity(*, root: Path) -> list[str]:
    failures: list[str] = []
    for result in iter_function_complexities(root, CRITICAL_COMPLEXITY_FILES):
        complexity_limit, line_limit = FUNCTION_BUDGET_OVERRIDES.get(
            result.name,
            (MAX_CYCLOMATIC_COMPLEXITY, MAX_FUNCTION_LINES),
        )
        if result.complexity > complexity_limit:
            failures.append(
                f"{result.path}:{result.line} {result.name} complexity "
                f"{result.complexity} exceeds {complexity_limit}"
            )
        if result.lines > line_limit:
            failures.append(
                f"{result.path}:{result.line} {result.name} length "
                f"{result.lines} exceeds {line_limit} lines"
            )
    return failures


def iter_function_complexities(
    root: Path,
    paths: Iterable[str],
) -> Iterable[FunctionComplexity]:
    for relative in paths:
        source_path = root / relative
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative)
        parents: list[str] = []

        def walk(node: ast.AST) -> Iterable[FunctionComplexity]:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                parents.append(node.name)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end_line = int(node.end_lineno or node.lineno)
                    yield FunctionComplexity(
                        path=relative,
                        name=".".join(parents),
                        line=node.lineno,
                        lines=end_line - node.lineno + 1,
                        complexity=_cyclomatic_complexity(node),
                    )
                for child in ast.iter_child_nodes(node):
                    yield from walk(child)
                parents.pop()
                return
            for child in ast.iter_child_nodes(node):
                yield from walk(child)

        yield from walk(tree)


def _cyclomatic_complexity(node: ast.AST) -> int:
    complexity = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.ExceptHandler,
                ast.IfExp,
                ast.Assert,
                ast.comprehension,
            ),
        ):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(0, len(child.values) - 1)
        elif isinstance(child, ast.Match):
            complexity += max(0, len(child.cases) - 1)
    return complexity


def _normalized_path(path: str, root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError:
            pass
    return candidate.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
