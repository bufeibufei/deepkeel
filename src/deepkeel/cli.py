from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
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

    pack = commands.add_parser("pack", help="scaffold and validate Capability Packages")
    pack_commands = pack.add_subparsers(dest="pack_command")

    pack_init = pack_commands.add_parser("init", help="create a minimal package skeleton")
    pack_init.add_argument("path")
    pack_init.add_argument("--package-id", required=True)
    pack_init.add_argument("--force", action="store_true")
    pack_init.set_defaults(handler=_pack_init)

    pack_inspect = pack_commands.add_parser("inspect", help="inspect a package manifest")
    pack_inspect.add_argument("manifest")
    pack_inspect.set_defaults(handler=_pack_inspect)

    pack_validate = pack_commands.add_parser(
        "validate",
        help="validate a manifest and optionally its executable pack",
    )
    pack_validate.add_argument("manifest")
    pack_validate.add_argument("--factory", help="Python reference in module:attribute form")
    pack_validate.set_defaults(handler=_pack_validate)

    pack_digest = pack_commands.add_parser(
        "digest",
        help="compute a deterministic SHA-256 digest for package source files",
    )
    pack_digest.add_argument("paths", nargs="+")
    pack_digest.set_defaults(handler=_pack_digest)
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


def _pack_init(args: argparse.Namespace) -> int:
    from deepkeel.version import DEEPKEEL_CONTRACT_VERSION, DEEPKEEL_VERSION

    destination = Path(args.path).resolve()
    manifest_path = destination / "manifest.json"
    package_path = destination / "package.py"
    readme_path = destination / "README.md"
    existing = [path for path in (manifest_path, package_path, readme_path) if path.exists()]
    if existing and not args.force:
        raise CliError(
            "CAPABILITY_PACK_EXISTS",
            "capability package files already exist; pass --force to replace them",
        )
    destination.mkdir(parents=True, exist_ok=True)
    package_id = str(args.package_id).strip()
    if not package_id:
        raise CliError("CAPABILITY_PACKAGE_ID_REQUIRED", "--package-id must not be blank")
    class_name = "".join(part.title() for part in re.split(r"[^A-Za-z0-9]+", package_id))
    class_name = f"{class_name or 'Example'}Pack"
    manifest = {
        "schema_version": "harness-capability-manifest-v1",
        "id": package_id,
        "version": "0.1.0",
        "core_contract": DEEPKEEL_CONTRACT_VERSION,
        "core_version": f">={DEEPKEEL_VERSION},<5.0.0",
        "entrypoint": f"package:{class_name}",
        "skills": [],
        "tools": [],
        "artifact_types": [],
        "permissions": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    package_path.write_text(_pack_template(class_name, package_id), encoding="utf-8")
    readme_path.write_text(
        f"# {package_id}\n\nValidate with `deepkeel pack validate manifest.json "
        f"--factory package:{class_name}`.\n",
        encoding="utf-8",
    )
    _emit(
        {
            "ok": True,
            "package_id": package_id,
            "path": str(destination),
            "files": ["manifest.json", "package.py", "README.md"],
        }
    )
    return 0


def _pack_inspect(args: argparse.Namespace) -> int:
    from deepkeel.capability_manifest import load_capability_manifest

    manifest = load_capability_manifest(args.manifest)
    _emit({"ok": True, "manifest": manifest.model_dump(mode="json")})
    return 0


def _pack_validate(args: argparse.Namespace) -> int:
    from deepkeel.capability_manifest import load_capability_manifest
    from deepkeel.conformance import validate_capability_pack

    manifest = load_capability_manifest(args.manifest)
    if not args.factory:
        _emit({"ok": True, "manifest": manifest.model_dump(mode="json")})
        return 0
    pack = _load_pack_factory(args.factory)
    report = validate_capability_pack(pack, manifest=manifest)
    _emit({"ok": report.passed, "report": report.model_dump(mode="json")})
    return 0 if report.passed else 1


def _pack_digest(args: argparse.Namespace) -> int:
    from deepkeel.capability_trust import capability_source_digest

    digest = capability_source_digest(*(Path(path) for path in args.paths))
    _emit({"ok": True, "sha256": digest, "files": sorted(args.paths)})
    return 0


def _load_pack_factory(reference: str) -> Any:
    module_name, separator, attribute_name = str(reference).partition(":")
    if not separator or not module_name or not attribute_name:
        raise CliError(
            "CAPABILITY_FACTORY_INVALID",
            "--factory must use module:attribute syntax",
        )
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    if isinstance(value, type):
        return value()
    if hasattr(value, "spec"):
        return value
    if callable(value):
        return value()
    raise CliError("CAPABILITY_FACTORY_INVALID", "factory did not return a capability pack")


def _pack_template(class_name: str, package_id: str) -> str:
    return f'''from dataclasses import dataclass

from deepkeel.extension_sdk import (
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPackSpec,
)


@dataclass(frozen=True, slots=True)
class {class_name}:
    spec = CapabilityPackSpec(
        package_id={package_id!r},
        package_version="0.1.0",
    )

    def install(
        self,
        context: CapabilityInstallContext,
    ) -> CapabilityContribution:
        return CapabilityContribution(package_id=self.spec.package_id)
'''


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
