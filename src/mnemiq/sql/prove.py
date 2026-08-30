"""Prove an approved plan against the source before running it, in the way that source understands.

Both deciders do this, and for the same reason: a snapshot can be stale, and only the source knows
whether the objects a plan names are really there. Both did it by issuing `EXPLAIN <sql>` and
treating any exception as `EXPLAIN_FAILED`.

**That is Postgres and DuckDB syntax, and it is not universal.** Measured against a live Oracle
23ai instance: `EXPLAIN SELECT id FROM t` raises ORA-02000, "missing PLAN keyword". So every
otherwise-valid Oracle plan was refused before execution, on a step whose entire purpose is to
distinguish a valid plan from an invalid one. Nothing caught it because the adapter's own tests
exercise the adapter and the decider's tests use adapters that speak `EXPLAIN` -- the failure lived
exactly in the gap between two well-tested things, which is where **M26** lived too.

The seam is optional rather than a required method on every adapter. Adapters here are structurally
typed -- there is no ABC and no Protocol for them -- so requiring `validate` would mean editing four
adapters plus every test double that stands in for one, and a test double that silently lacks the
method would then fail in a way that looks like a source error. An adapter that knows a better proof
offers one; the rest keep the behaviour they already had, unchanged.
"""

from __future__ import annotations

from typing import Any

from mnemiq.sql.verdict import Refusal, RefusalCode


def prove(adapter: Any, target_sql: str) -> Refusal | None:
    """`None` when the source accepts the statement, a `Refusal` when it does not.

    The refusal carries the source's own message. That is deliberate and load-bearing: it is what
    a correction loop reads to repair the SQL, so replacing it with a generic sentence would cost
    the repair its only evidence.
    """
    validate = getattr(adapter, "validate", None)
    try:
        if validate is not None:
            validate(target_sql)
        else:
            # `EXPLAIN` plans without executing on Postgres, DuckDB and SQLite.
            adapter.execute(f"EXPLAIN {target_sql}")
    except Exception as exc:
        return Refusal(
            code=RefusalCode.EXPLAIN_FAILED,
            message=f"The source rejected this query: {exc}",
        )
    return None
