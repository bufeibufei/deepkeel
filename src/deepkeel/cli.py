from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
import platform
import re
import sys
from typing import Any, Sequence

from deepkeel.public_api import PUBLIC_API_VERSION
from deepkeel.version import PACKAGE_VERSION


class CliError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "handler", None)
    if not callable(handler):
        parser.print_help()
        return 2
    try:
        return int(handler(args))
    except Exception as exc:
        code = exc.code if isinstance(exc, CliError) else _error_code(exc)
        _emit(
            {
                "ok": False,
                "error": {
                    "code": code,
                    "type": type(exc).__name__,
                    "message": _safe_error(exc),
                },
            }
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepkeel",
        description="DeepKeel runtime diagnostics and migration operations",
    )
    commands = parser.add_subparsers(dest="command")

    doctor = commands.add_parser("doctor", help="inspect runtime and optional dependencies")
    doctor.set_defaults(handler=_doctor)

    postgres = commands.add_parser("postgres", help="operate the packaged PostgreSQL schema")
    postgres_commands = postgres.add_subparsers(dest="postgres_command")
    for name, handler in (
        ("status", _postgres_status),
        ("plan", _postgres_plan),
        ("upgrade", _postgres_upgrade),
    ):
        command = postgres_commands.add_parser(name)
        command.add_argument("--schema", default="deepkeel")
        command.add_argument("--dsn-env", default="DEEPKEEL_POSTGRES_DSN")
        if name == "upgrade":
            command.add_argument(
                "--yes",
                action="store_true",
                help="apply pending forward migrations",
            )
            command.add_argument("--target-version", type=int)
        command.set_defaults(handler=handler)
    return parser


def _doctor(_args: argparse.Namespace) -> int:
    mandatory = {
        name: _distribution_version(name)
        for name in ("httpx", "jsonschema", "langgraph", "pydantic")
    }
    optional = {
        "postgres": _distribution_version("psycopg"),
        "otel": _distribution_version("opentelemetry-api"),
    }
    python_ok = sys.version_info >= (3, 12)
    ok = python_ok and all(mandatory.values())
    _emit(
        {
            "ok": ok,
            "deepkeel_version": PACKAGE_VERSION,
            "public_api_version": PUBLIC_API_VERSION,
            "python": platform.python_version(),
            "mandatory_dependencies": mandatory,
            "optional_integrations": optional,
        }
    )
    return 0 if ok else 2


def _postgres_status(args: argparse.Namespace) -> int:
    database = _postgres_database(args)
    status = database.migration_status()
    _emit({"ok": status.up_to_date, "schema": database.schema, **_status_payload(status)})
    return 0 if status.up_to_date else 1


def _postgres_plan(args: argparse.Namespace) -> int:
    database = _postgres_database(args)
    status = database.migration_status()
    pending = database.migration_registry().plan()
    _emit(
        {
            "ok": True,
            "schema": database.schema,
            **_status_payload(status),
            "plan": [_migration_payload(item) for item in pending],
        }
    )
    return 0


def _postgres_upgrade(args: argparse.Namespace) -> int:
    database = _postgres_database(args)
    registry = database.migration_registry()
    pending = registry.plan(target_version=args.target_version)
    if pending and not args.yes:
        raise CliError(
            "MIGRATION_CONFIRMATION_REQUIRED",
            "pending migrations require the explicit --yes flag",
        )
    status = registry.upgrade(target_version=args.target_version)
    effective_target = (
        args.target_version
        if args.target_version is not None
        else status.target_version
    )
    reached_target = status.current_version == effective_target
    _emit(
        {
            "ok": reached_target,
            "schema": database.schema,
            "requested_migrations": [_migration_payload(item) for item in pending],
            "requested_target_version": effective_target,
            **_status_payload(status),
        }
    )
    return 0 if reached_target else 1


def _postgres_database(args: argparse.Namespace) -> Any:
    from deepkeel.contrib.postgres import PostgresDatabase

    variable = str(args.dsn_env or "").strip()
    if not variable:
        raise CliError("POSTGRES_DSN_ENV_REQUIRED", "--dsn-env must not be empty")
    dsn = str(os.environ.get(variable) or "").strip()
    if not dsn:
        raise CliError(
            "POSTGRES_DSN_MISSING",
            f"PostgreSQL DSN environment variable is not set: {variable}",
        )
    return PostgresDatabase(dsn, schema=args.schema)


def _status_payload(status: Any) -> dict[str, Any]:
    return {
        "current_version": status.current_version,
        "target_version": status.target_version,
        "up_to_date": status.up_to_date,
        "applied": [
            {
                "version": item.version,
                "name": item.name,
                "checksum": item.checksum,
                "applied_at": item.applied_at.isoformat(),
                "duration_ms": item.duration_ms,
            }
            for item in status.applied
        ],
        "pending": [_migration_payload(item) for item in status.pending],
    }


def _migration_payload(migration: Any) -> dict[str, Any]:
    return {
        "version": migration.version,
        "name": migration.name,
        "checksum": migration.checksum,
    }


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _error_code(exc: Exception) -> str:
    name = type(exc).__name__.upper()
    if "DRIFT" in name:
        return "POSTGRES_SCHEMA_DRIFT"
    if "SCHEMA" in name:
        return "POSTGRES_SCHEMA_ERROR"
    return "DEEPKEEL_CLI_ERROR"


def _safe_error(exc: Exception) -> str:
    message = str(exc or "operation failed")
    message = re.sub(r"postgres(?:ql)?://[^\s]+", "postgresql://***", message)
    message = re.sub(r"(?i)(password\s*=\s*)[^\s]+", r"\1***", message)
    return message[:1000]


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
