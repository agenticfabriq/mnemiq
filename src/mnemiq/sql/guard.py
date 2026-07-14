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


def _output_selects(ast: exp.Expression) -> list[exp.Select]:
    """The SELECTs whose projections actually leave the engine.

    For a UNION that is every branch; for a plain query it is just the root. Inner SELECTs
    (subqueries, CTE bodies, EXISTS) do not return columns to the caller -- whatever they
    project is bounded by the explicit projection that wraps them.
    """
    if isinstance(ast, exp.Select):
        return [ast]
    if isinstance(ast, exp.Union):
        return [*_output_selects(ast.left), *_output_selects(ast.right)]
    return []


def _sources(select: exp.Select) -> dict[str, exp.Expression]:
    """Everything the SELECT reads from, keyed by the name a star could be qualified with."""
    found: list[exp.Expression] = []
    # sqlglot spells this arg "from" in older versions and "from_" from v30 on. Reading the
    # wrong key silently yields no sources -- which would make every star look safe.
    from_ = select.args.get("from") or select.args.get("from_")
    if from_ is not None:
        found.append(from_.this)
    for join in select.args.get("joins") or []:
        found.append(join.this)
    return {source.alias_or_name: source for source in found}


def _expands_a_base_table(select: exp.Select, star: exp.Expression, cte_names: set[str]) -> bool:
    """Would this star pull in the columns of a real source table?

    Over a subquery or a CTE the columns are already determined by an explicit projection --
    and were themselves authorized. Over a base table they are unbounded: we would not know
    what we are returning, and column-level authorization would have nothing to check.
    """
    sources = _sources(select)
    if not sources:
        return True  # a star over sources we cannot see is not a star we can vouch for

    qualifier = star.table if isinstance(star, exp.Column) else None
    candidates = (
        [sources[qualifier]] if qualifier and qualifier in sources else list(sources.values())
    )

    for source in candidates:
        if isinstance(source, exp.Subquery):
            continue  # derived table: its projection is explicit
        if isinstance(source, exp.Table) and source.name in cte_names:
            continue  # a CTE: same
        return True
    return False


def _has_projection_star(ast: exp.Expression) -> bool:
    """A star that would return columns we cannot name.

    Three things this deliberately does not reject -- a guard that refuses legitimate
    analytics SQL is broken, not safe (all 99 TPC-DS queries must pass it):

    - `COUNT(*)` contains an exp.Star, so a naive find_all(exp.Star) would refuse `SELECT
      count(*) FROM claim`, the most common analytics query there is.
    - An *inner* star returns nothing to the caller: `EXISTS (SELECT * FROM t)` discards its
      projection, and `SELECT a FROM (SELECT * FROM t)` still returns exactly `a`.
    - A star over a *derived table or CTE* expands an explicit projection, so the output
      columns are known: `SELECT * FROM (SELECT a, b FROM t)` returns exactly a and b.

    What is refused is a star that expands a **base table** -- there the column set is
    unbounded, and neither we nor column-level authorization can say what comes back.
    """
    cte_names = {cte.alias_or_name for cte in ast.find_all(exp.CTE)}

    for select in _output_selects(ast):
        for projection in select.expressions:
            is_star = isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
            )
            if is_star and _expands_a_base_table(select, projection, cte_names):
                return True
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
