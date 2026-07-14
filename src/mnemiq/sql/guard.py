from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.verdict import Refusal, RefusalCode

MAX_ROWS = 1000

# Everything the engine is willing to run is one of these. Not a blocklist: a blocklist is a
# bet that you thought of every dangerous statement, and that bet is always lost eventually.
_ALLOWED_ROOTS = (exp.Select, exp.Union)


def check_shape(
    sql: str, dialect: str = "duckdb", max_rows: int = MAX_ROWS
) -> exp.Expression | Refusal:
    """Parse and enforce the shape of the query. Returns the AST, with a LIMIT guaranteed.

    The limit is *injected*, not requested: asking a model to add LIMIT is a request, and
    rewriting the tree is a guarantee.
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception:
        return Refusal(
            code=RefusalCode.PARSE_ERROR,
            message="That is not valid SQL. Return a single SELECT statement.",
        )

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        # sqlglot.parse_one() would fold "SELECT 1; DROP TABLE t" into a Block and return it
        # without raising -- so we count statements ourselves.
        return Refusal(
            code=RefusalCode.NOT_A_SINGLE_STATEMENT,
            message="Return exactly one statement. Multiple statements are never executed.",
        )

    ast = statements[0]
    if not isinstance(ast, _ALLOWED_ROOTS):
        return Refusal(
            code=RefusalCode.NOT_SELECT_ONLY,
            message="Only SELECT queries are allowed. This engine never modifies data.",
            subject=type(ast).__name__.upper(),
        )

    if _has_projection_star(ast):
        return Refusal(
            code=RefusalCode.SELECT_STAR,
            message=(
                "Do not use SELECT *. List the explicit columns you need -- the engine must "
                "know which columns it returns."
            ),
        )

    return _with_limit(ast, max_rows)


def _has_projection_star(ast: exp.Expression) -> bool:
    """A star in the SELECT list -- not a star anywhere in the tree.

    COUNT(*) contains an exp.Star, so a naive find_all(exp.Star) would reject `SELECT
    count(*) FROM claim`: the single most common analytics query there is. Only a star that
    is being *projected* hides columns from us.
    """
    for select in [ast, *ast.find_all(exp.Select)]:
        if not isinstance(select, exp.Select):
            continue
        for projection in select.expressions:
            if isinstance(projection, exp.Star):
                return True  # SELECT *
            if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                return True  # SELECT c.*
    return False


def _with_limit(ast: exp.Expression, max_rows: int) -> exp.Expression:
    existing = ast.args.get("limit")
    if existing is None:
        return ast.limit(max_rows)

    try:
        requested = int(existing.expression.name)
    except (AttributeError, ValueError):
        return ast.limit(max_rows)  # unreadable limit -> impose our own

    return ast if requested <= max_rows else ast.limit(max_rows)
