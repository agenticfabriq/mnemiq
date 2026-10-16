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

import logging

from typing import Any

from mnemiq.sql.verdict import Refusal, RefusalCode

logger = logging.getLogger(__name__)


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
        # Logged here rather than where the loop gives up, because a refusal on an attempt that
        # a later one repairs is exactly the one an operator wants to read, and by then it has
        # been overwritten. This is the only record of it: `source_detail` is withheld from the
        # caller, so without this line an EXPLAIN failure would be visible nowhere at all.
        logger.warning("EXPLAIN refused a plan; the source said: %s", exc)
        return Refusal(
            code=RefusalCode.EXPLAIN_FAILED,
            message="The source rejected this query.",
            source_detail=str(exc),
        )
    return None


def prove_governed(adapter: Any, target_sql: str, asked_sql: str) -> Refusal | None:
    """`prove` the statement as it will run, and where the source refuses only what the access
    policy added, say so instead (M158).

    A row filter gone stale against its source -- a column renamed under it -- fails the proof of
    every statement that reads its table. EXPLAIN_FAILED is repairable, so each attempt went back
    to the model to repair a statement that was never wrong, and the caller was told the model had
    failed. When the policy's rewrite is refused and the statement as asked is accepted, the fault
    is in what the engine added: the policy, or the snapshot it was checked against. The cost is
    one or two more proofs, and only for a refused statement that a policy rewrote.

    The second proof tells the caller one thing: that their own statement is valid. A working
    policy would have told them the same. A proof reads the statement and the schema, never rows:
    EXPLAIN plans without running, and Oracle's `validate` parses.
    """
    refused = prove(adapter, target_sql)
    if refused is None or asked_sql == target_sql:
        return refused
    logger.info("Proving the statement as asked, without the access policy's rewrite.")
    if prove(adapter, asked_sql) is not None:
        return refused  # the statement itself is refused: the model's to repair, as before
    # Once more before the policy is blamed: `prove` reads every exception as a refusal, so a
    # dropped connection or a lock timeout on the first proof would otherwise land here, and a
    # policy fault is not retried.
    if prove(adapter, target_sql) is None:
        return None
    # `prove` logged the source's words for the rewrite; this says whose they were.
    logger.warning("The source accepts the statement as asked and refuses the access policy's "
                   "rewrite of it: a row filter, or the snapshot it was checked against, no "
                   "longer matches the source.")
    return Refusal(
        code=RefusalCode.POLICY_REWRITE_REJECTED,
        message=("The source accepts this query as written but rejects it once your access policy "
                 "is applied, so it cannot be run for you. A row filter, or the schema this "
                 "deployment checked the policy against, no longer matches the source. This is a "
                 "configuration fault for the operator, not a limit on your access; the server log "
                 "has the source's error."),
    )
