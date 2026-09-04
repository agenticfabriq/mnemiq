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
    # SetOperation, not Union: EXCEPT and INTERSECT are its SIBLINGS in sqlglot, not its
    # subclasses, so keying on Union read `... EXCEPT ...` as having no output projection at all.
    if isinstance(ast, exp.SetOperation):
        return [*_output_selects(ast.left), *_output_selects(ast.right)]
    if isinstance(ast, exp.Subquery):
        return _output_selects(ast.this)  # redundant parentheses nest Subquery in Subquery
    return []


def _is_star(projection: exp.Expression) -> bool:
    """`COUNT(*)` contains an `exp.Star` too, so this asks about the PROJECTION, not the tree."""
    return isinstance(projection, exp.Star) or (
        isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
    )


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


# A CTE name resolves to its body AND to the scope that body was written in -- never to the
# scope it is read from. `Scope` is that pair, and pairing them is the whole point: a bare
# `name -> body` map has to borrow some caller's names to interpret the body, and the caller's
# names are not the ones the body was written against.
Scope = dict[str, tuple[exp.Expression, "Scope"]]


def _visible_ctes(node: exp.Expression, outer: Scope) -> Scope:
    """The CTE names visible inside `node`, each bound to the scope ITS OWN body sees.

    Lexical, and lexical at the definition site. Two rules make that different from "the names
    around the reference", and both were live bypasses:

      * a non-recursive CTE cannot see a LATER sibling, so in
        `WITH a AS (SELECT * FROM claim), claim AS (...) SELECT * FROM a` the `claim` inside `a`
        is the base table, not the sibling defined after it;
      * an inner WITH at the reference site cannot rebind a name inside an outer CTE's body, so
        `WITH a AS (SELECT * FROM claim) SELECT * FROM (WITH claim AS (...) SELECT * FROM a) z`
        does not make `a`'s star explicit either.

    Both returned every column of `claim`, `ssn` included. Carrying one flat map -- or merging
    the caller's -- lets a projection vouch for a star it has no relationship to; `cls.py` records
    the same defect in the CLS resolver, where one alias meant two tables and the map kept
    whichever was seen last.
    """
    # "with" in older sqlglot, "with_" from v30 on -- the same rename `_sources` guards against
    # for "from". Reading only the old key made every CTE invisible: `source.name in ctes` was
    # never true, so a CTE source fell through to the base-table branch. That direction refuses
    # legitimate SQL rather than permitting a star, which is the only reason the tests caught it.
    with_ = node.args.get("with") or node.args.get("with_")
    if with_ is None:
        return outer

    recursive = bool(with_.args.get("recursive"))
    scope: Scope = dict(outer)
    for cte in with_.expressions:
        name = cte.alias_or_name
        # snapshot BEFORE binding this name: the body sees the outer scope and its preceding
        # siblings, plus itself only when the WITH is RECURSIVE
        body_scope: Scope = dict(scope)
        entry = (cte.this, body_scope)
        if recursive:
            body_scope[name] = entry
        scope[name] = entry
    return scope


def _star_reaches_base(node: exp.Expression, ctes: Scope,
                       memo: dict[int, bool], stack: frozenset[int]) -> bool:
    """Does this expression's own OUTPUT projection reach a base table's unbounded column set?

    The recursion is the point. "Its projection is explicit" is a claim about the derived table,
    and it is only true when that projection names columns -- `(SELECT a, b FROM t)` yes,
    `(SELECT * FROM t)` no. Answering it by looking one level down let a base-table star through
    inside a wrapper, so the column set was unbounded after all.

    Unreadable is not safe. A node `_output_selects` cannot decompose has no known projection,
    so it is treated as reaching a base table rather than as reaching nothing -- the same rule
    `_expands_a_base_table` already applies to sources it cannot see. Returning False there made
    every gap in that function a silent hole, which is how EXCEPT and doubled parentheses got
    through.

    `memo` keeps this linear, and is only sound BECAUSE the scopes above are lexical: each node
    then sits in exactly one scope, so its answer cannot depend on the path that reached it. While
    a CTE body was interpreted against its caller's names, the memo cached one caller's answer for
    all of them and two union branches gave opposite verdicts depending on their order. Without
    it, a CTE named twice per level is re-walked once per path -- 8s of pre-execution gate on
    1.2KB of model-written SQL.
    """
    if id(node) in stack:
        return False  # a recursive CTE names itself; the base case is the branch that ends
    if id(node) in memo:
        return memo[id(node)]

    selects = _output_selects(node)
    answer = True  # cannot be read -> cannot be vouched for
    if selects:
        inner, within = _visible_ctes(node, ctes), stack | {id(node)}
        answer = any(
            _expands_a_base_table(select, projection, inner, memo, within)
            for select in selects
            for projection in select.expressions
            if _is_star(projection)
        )
    memo[id(node)] = answer
    return answer


def _expands_a_base_table(select: exp.Select, star: exp.Expression, ctes: Scope,
                          memo: dict[int, bool], stack: frozenset[int]) -> bool:
    """Would this star pull in the columns of a real source table?

    Over a base table the columns are unbounded: we would not know what we are returning, and
    column-level authorization would have nothing to check. Over a subquery or a CTE they are
    determined by that source's projection -- so that projection is what gets asked, rather than
    assumed. A wrapper does not make a star explicit, and treating it as if it did returned a
    DENIED column in full: `SELECT claim.ssn FROM claim` was refused `unauthorized_column` while
    `SELECT * FROM (SELECT * FROM claim) t` came back with the same values, masks and denials
    alike passing through untouched because neither `check_cls` nor `referenced_masked` sees an
    `exp.Column` for a star.
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
            if _star_reaches_base(source.this, ctes, memo, stack):
                return True
            continue  # derived table whose projection really is explicit
        if isinstance(source, exp.Table) and source.name in ctes:
            body, body_scope = ctes[source.name]
            # `body_scope`, not `ctes`: the body is interpreted where it was written
            if _star_reaches_base(body, body_scope, memo, stack):
                return True
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

    What is refused is a star that expands a **base table**, directly or through a wrapper the
    walker can follow -- there the column set is unbounded, and neither we nor column-level
    authorization can say what comes back. The third exemption above is the narrow one: it holds
    when the derived table or CTE projects named columns, and not when it projects a star of its
    own. "A wrapper the walker can follow" is the honest limit rather than a universal: what it
    follows is what `_output_selects` decomposes, and a shape it cannot read is refused rather
    than waved through, so the gap costs a false refusal instead of a leak.
    """
    # The top level keeps its own fail-open reading: a statement whose shape `_output_selects`
    # does not recognise returns no columns to a caller here, and refusing every such statement
    # would refuse writes and DDL this function is not the gate for. Fail-CLOSED starts one level
    # down, where the question is whether a wrapper can be vouched for.
    if not _output_selects(ast):
        return False
    return _star_reaches_base(ast, {}, {}, frozenset())


def _with_limit(ast: exp.Expression, max_rows: int) -> exp.Expression:
    existing = ast.args.get("limit")
    if existing is None:
        return ast.limit(max_rows)

    try:
        requested = int(existing.expression.name)
    except (AttributeError, ValueError):
        return ast.limit(max_rows)  # unreadable limit -> impose our own

    return ast if requested <= max_rows else ast.limit(max_rows)
