from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.cls import check_cls
from mnemiq.sql.guard import MAX_ROWS, check_shape
from mnemiq.sql.lint import lint
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import expand_tables, object_key
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.values_check import check_values
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode, Verdict


def decide(
    sql: str,
    visible: dict[str, set[str]],
    adapter=None,
    dialect: str = "duckdb",
    target: str = "postgres",
    max_rows: int = MAX_ROWS,
    values=None,
    policy: AccessPolicy | None = None,
    registry: dict[str, str] | None = None,
) -> Verdict:
    """The deterministic decider: shape, then access, then proof against the real source.

    Order matters. Shape first, because a DROP is a DROP regardless of what it names. Access
    second, because an unauthorized query must never reach the database -- the DB's own
    permission error is not the control: by the time it fires, we have already confirmed to
    the model that the table exists.
    """
    policy = policy or AccessPolicy()
    shaped = check_shape(sql, dialect=dialect, max_rows=max_rows)
    if isinstance(shaped, Refusal):
        return shaped

    refusal = check_access(shaped, visible)
    if refusal is not None:
        return refusal

    cls = check_cls(shaped, policy)  # column deny / mask-in-predicate
    if cls is not None:
        return cls

    violation = lint(shaped)
    if violation is not None:
        # silently-wrong SQL: runs fine, wrong answer. Repairable -- the corrector fixes it.
        return Refusal(code=RefusalCode.LOGIC_LINT, message=violation.message)

    if values is not None:
        # a filter literal that does not exist in its column runs fine and answers wrong --
        # another silently-wrong class the corrector fixes, given the real values to pick from.
        # M5: the index is unfiltered and this runs before the RLS rewrite below, so the
        # refusal must know which tables this identity only sees a slice of.
        grounding = check_values(shaped, visible, values, row_filtered=set(policy.row_filters))
        if grounding is not None:
            return grounding

    if not policy.empty:  # RLS + source-side mask rewrite (filtered/masked tables -> derived)
        shaped = apply_row_and_mask(shaped, policy, visible, dialect=dialect)
        if isinstance(shaped, Refusal):
            return shaped

    # Provenance uses the qualified id and must be read BEFORE expansion rewrites the nodes.
    cte_names = {cte.alias_or_name for cte in shaped.find_all(exp.CTE)}
    tables = sorted({object_key(t) for t in shaped.find_all(exp.Table)} - cte_names)
    columns = sorted({c.name for c in shaped.find_all(exp.Column)})

    # Federation: 'catalog.table' -> 'catalog.schema.table' so DuckDB resolves it. No-op single-source.
    expand_tables(shaped, registry or {})

    plan_sql = shaped.sql(dialect=dialect)
    target_sql = sqlglot.transpile(plan_sql, read=dialect, write=target)[0]

    if adapter is not None:
        try:
            adapter.execute(f"EXPLAIN {target_sql}")
        except Exception as exc:
            # the snapshot can be stale, and only the source knows the truth
            return Refusal(
                code=RefusalCode.EXPLAIN_FAILED,
                message=f"The source rejected this query: {exc}",
            )

    return Approved(plan_sql=plan_sql, target_sql=target_sql, tables=tables, columns=columns)
