from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.guard import MAX_ROWS, check_shape
from mnemiq.sql.lint import lint
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode, Verdict


def decide(
    sql: str,
    visible: dict[str, set[str]],
    adapter=None,
    dialect: str = "duckdb",
    target: str = "postgres",
    max_rows: int = MAX_ROWS,
) -> Verdict:
    """The deterministic decider: shape, then access, then proof against the real source.

    Order matters. Shape first, because a DROP is a DROP regardless of what it names. Access
    second, because an unauthorized query must never reach the database -- the DB's own
    permission error is not the control: by the time it fires, we have already confirmed to
    the model that the table exists.
    """
    shaped = check_shape(sql, dialect=dialect, max_rows=max_rows)
    if isinstance(shaped, Refusal):
        return shaped

    refusal = check_access(shaped, visible)
    if refusal is not None:
        return refusal

    violation = lint(shaped)
    if violation is not None:
        # silently-wrong SQL: runs fine, wrong answer. Repairable -- the corrector fixes it.
        return Refusal(code=RefusalCode.LOGIC_LINT, message=violation.message)

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

    cte_names = {cte.alias_or_name for cte in shaped.find_all(exp.CTE)}
    tables = sorted({t.name for t in shaped.find_all(exp.Table)} - cte_names)
    columns = sorted({c.name for c in shaped.find_all(exp.Column)})

    return Approved(plan_sql=plan_sql, target_sql=target_sql, tables=tables, columns=columns)
