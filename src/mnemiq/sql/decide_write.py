from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode

_WRITE_ROOTS = (exp.Insert, exp.Update, exp.Delete)


def _target_table(ast: exp.Expression) -> str | None:
    this = ast.this
    if isinstance(this, exp.Schema):  # INSERT INTO t (cols)
        this = this.this
    if isinstance(this, exp.Table):
        return this.name
    found = ast.find(exp.Table)
    return found.name if found else None


def check_write_shape(sql: str, dialect: str = "duckdb") -> exp.Expression | Refusal:
    """One bounded DML statement, or a Refusal. UPDATE/DELETE must be bounded by a WHERE."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception:
        return Refusal(code=RefusalCode.PARSE_ERROR, message="That is not valid SQL.")
    if len(statements) != 1:
        return Refusal(
            code=RefusalCode.NOT_A_SINGLE_STATEMENT,
            message="Return exactly one statement. Multiple statements are never executed.",
        )
    ast = statements[0]
    if not isinstance(ast, _WRITE_ROOTS):
        return Refusal(
            code=RefusalCode.NOT_A_WRITE,
            message="Only INSERT, UPDATE, or DELETE is allowed here. DDL is never executed.",
            subject=type(ast).__name__.upper(),
        )
    if isinstance(ast, (exp.Update, exp.Delete)) and ast.args.get("where") is None:
        return Refusal(
            code=RefusalCode.UNBOUNDED_WRITE,
            message="An UPDATE or DELETE must have a WHERE clause; an unbounded mutation is refused.",
        )
    return ast


def decide_write(
    sql: str,
    visible: dict[str, set[str]],
    grants: GrantSet,
    adapter=None,
    dialect: str = "duckdb",
    target: str | None = None,
) -> ApprovedWrite | Refusal:
    """The deterministic write decider: shape -> table/column authz -> target-writable -> proof."""
    target = target or dialect
    shaped = check_write_shape(sql, dialect=dialect)
    if isinstance(shaped, Refusal):
        return shaped

    refusal = check_access(shaped, visible)  # every referenced table readable, columns exist
    if refusal is not None:
        return refusal

    tgt = _target_table(shaped)
    if tgt is None or not grants.allows_write(tgt):
        return Refusal(
            code=RefusalCode.UNAUTHORIZED_WRITE,
            message=f"You may not write to {tgt!r}.",
            subject=tgt,
        )

    plan_sql = shaped.sql(dialect=dialect)
    target_sql = sqlglot.transpile(plan_sql, read=dialect, write=target)[0]
    if adapter is not None:
        try:
            adapter.execute(f"EXPLAIN {target_sql}")  # plans without executing; proves the SQL
        except Exception as exc:
            return Refusal(
                code=RefusalCode.EXPLAIN_FAILED,
                message=f"The source rejected this query: {exc}",
            )

    cte_names = {c.alias_or_name for c in shaped.find_all(exp.CTE)}
    tables = sorted({t.name for t in shaped.find_all(exp.Table)} - cte_names)
    return ApprovedWrite(plan_sql=plan_sql, target_sql=target_sql, target=tgt, tables=tables)
