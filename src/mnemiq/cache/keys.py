from __future__ import annotations

import hashlib

import sqlglot


def canonical_plan(plan_sql: str, dialect: str = "duckdb") -> str:
    """Normalize the SQL so formatting cannot fork the cache.

    Never raises: on the cache path a miss is acceptable and a crash is not.
    """
    try:
        return sqlglot.parse_one(plan_sql, read=dialect).sql(dialect=dialect, normalize=True)
    except Exception:
        return " ".join(plan_sql.split())


def cache_key(
    plan_sql: str,
    grant_fingerprint: str,
    enrichment_version: str | None,
    dialect: str = "duckdb",
) -> str:
    """plan + grants + enrichment version.

    The grant fingerprint is not an optimization -- it is the authorization boundary. Keying
    on the question alone would let a narrow identity be served a broad identity's rows
    without a single guard running. Identical grants share entries; different grants cannot
    collide. The enrichment version does the same job across time: re-enrich and every stale
    answer is invalidated by construction.
    """
    material = "\x00".join(
        [canonical_plan(plan_sql, dialect), grant_fingerprint, enrichment_version or "none"]
    )
    return hashlib.sha256(material.encode()).hexdigest()
