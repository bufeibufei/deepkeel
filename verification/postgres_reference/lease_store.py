from __future__ import annotations

from uuid import uuid4

from deepkeel.adapter_sdk import RunLease, RunLeaseConflict, RunLeaseLost
from verification.postgres_reference.database import PostgresReferenceDatabase
from verification.postgres_reference.support import lease_from_row, validated_ttl


class PostgresRunLeaseStore:
    """Database-time lease ownership with monotonic fencing generations."""

    def __init__(self, database: PostgresReferenceDatabase) -> None:
        self.database = database

    def claim(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_seconds: float,
    ) -> RunLease:
        normalized_run_id = str(run_id or "").strip()
        normalized_owner = str(owner_id or "").strip()
        ttl = validated_ttl(ttl_seconds)
        if not normalized_run_id or not normalized_owner:
            raise ValueError("run_id and owner_id are required")
        token = uuid4().hex
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {schema}.run_leases (
                        run_id, owner_id, token, acquired_at, expires_at, generation
                    ) VALUES (
                        %s, %s, %s, CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'), 1
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        owner_id = EXCLUDED.owner_id,
                        token = EXCLUDED.token,
                        acquired_at = CURRENT_TIMESTAMP,
                        expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                        generation = {schema}.run_leases.generation + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE {schema}.run_leases.expires_at IS NULL
                       OR {schema}.run_leases.expires_at <= CURRENT_TIMESTAMP
                       OR {schema}.run_leases.owner_id = ''
                    RETURNING *
                    """,
                    (normalized_run_id, normalized_owner, token, ttl, ttl),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RunLeaseConflict(
                        f"run {normalized_run_id} already has a live owner"
                    )
        return lease_from_row(row)

    def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        ttl = validated_ttl(ttl_seconds)
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {schema}.run_leases
                    SET expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s AND owner_id = %s AND token = %s
                      AND generation = %s AND expires_at > CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (ttl, lease.run_id, lease.owner_id, lease.token, lease.generation),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RunLeaseLost(f"run lease was lost for {lease.run_id}")
        return lease_from_row(row)

    def release(self, lease: RunLease) -> None:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE {schema}.run_leases
                    SET owner_id = '', token = '', expires_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s AND owner_id = %s AND token = %s AND generation = %s
                    """,
                    (lease.run_id, lease.owner_id, lease.token, lease.generation),
                )
                if cursor.rowcount == 1:
                    return
                cursor.execute(
                    f"SELECT run_id FROM {schema}.run_leases WHERE run_id = %s",
                    (lease.run_id,),
                )
                if cursor.fetchone() is not None:
                    raise RunLeaseLost(f"run lease was replaced for {lease.run_id}")

    def inspect(self, run_id: str) -> RunLease | None:
        schema = self.database.schema
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT * FROM {schema}.run_leases
                    WHERE run_id = %s AND owner_id <> '' AND token <> ''
                      AND expires_at > CURRENT_TIMESTAMP
                    """,
                    (str(run_id or ""),),
                )
                row = cursor.fetchone()
        return lease_from_row(row) if row is not None else None
