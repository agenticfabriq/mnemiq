"""A bare date compared with a timestamp column is read as the whole day it names.

`ts <= '2024-03-09'` compares with midnight at the START of the 9th, so every row later that day
falls out, where the range a person means by "as of", "through" or "up to" the 9th ends before the
10th. Measured on saved runs of a local 32B model: it wrote this bound on most runs of the questions
that turn on it, every such answer was wrong, and ending the range at the next day turned them
right and turned no right answer wrong.

Every comparison of the column with the day is read by day, as `CAST(col AS DATE) op D` reads it,
so a statement's complementary predicates stay complementary (moving `<=` alone made `<= D` and
`> D` buckets overlap and count a noon row twice, Codex's review):

    col <= D   ->  col < D+1               col > D    ->  col >= D+1
    col = D    ->  (col >= D AND col < D+1)
    col <> D   ->  NOT (col >= D AND col < D+1)
    col BETWEEN x AND D  ->  (col >= x AND col < D+1)

each with the literal on either side, D+1 written in the literal's own spelling so the statement
stays valid in its dialect. `col < D` and `col >= D` already mean the day's boundary and stay as
written; so do IN and NOT IN, together, and BETWEEN SYMMETRIC.

D is a date-only ISO literal naming a real day, spelled plain, cast or typed (an allow-list: `_day`).
`col` is a bare column bound directly to a base table of its own scope (no CTE, derived table, set
operation or outer reference between them), placed by the values resolver's own-scope answer on a
column declared a naive timestamp, or a DATE on Oracle, where a DATE holds a time of day. The
resolver's wider readings are not used: past its own scope it can place a set operation's text
column on a timestamp column of the same name (Codex's review). Nothing moves on a DATE elsewhere
(whole days make both readings the same rows), a zone-aware timestamp (its midnight is the
session's zone's), text, an expression, or a bound with a time of day, which says where it ends.
"""
from __future__ import annotations

from datetime import date, timedelta

from sqlglot import exp

from mnemiq.sql.values_check import _in_scope, _iso_date, _scopes, _temporal_kind

_DAY_FORMATS = frozenset({"%Y-%m-%d"})  # a parse format naming the day alone (Oracle's TO_DATE 'YYYY-MM-DD')


def _day(node: exp.Expression) -> exp.Literal | None:
    """The string literal of a date-only bound, in one of the spellings that mean midnight of that
    day, else None: a plain string; a CAST or TRY_CAST of one to a date or a naive timestamp; a
    DATE literal as Oracle parses it; a parse whose format names the day alone."""
    inner = None
    if isinstance(node, exp.Literal):
        inner = node
    elif isinstance(node, (exp.Cast, exp.TryCast)):
        if _temporal_kind(node.args.get("to"), None) in ("date", "timestamp"):
            inner = node.this
    elif isinstance(node, exp.DateStrToDate):
        inner = node.this
    elif isinstance(node, (exp.StrToDate, exp.StrToTime)):
        fmt = node.args.get("format")
        if isinstance(fmt, exp.Literal) and fmt.is_string and fmt.this in _DAY_FORMATS:
            inner = node.this
    if isinstance(inner, exp.Literal) and inner.is_string and _iso_date(inner.this):
        return inner
    return None


def _next_day(bound: exp.Expression) -> exp.Expression:
    """`bound`, its date literal the day after, in its own spelling."""
    moved = bound.copy()
    literal = _day(moved)
    day = exp.Literal.string((date.fromisoformat(literal.this) + timedelta(days=1)).isoformat())
    if literal is moved:  # a plain string is the bound itself, with no parent to replace it in
        return day
    literal.replace(day)
    return moved


def _the_day(column: exp.Expression, bound: exp.Expression) -> exp.Paren:
    """`(column >= bound AND column < the next day)`: the whole day `bound` names."""
    return exp.Paren(this=exp.And(this=exp.GTE(this=column.copy(), expression=bound.copy()),
                                  expression=exp.LT(this=column.copy(), expression=_next_day(bound))))


def close_day_bounds(ast: exp.Expression, visible: dict[str, set[str]],
                     types: dict[str, dict[str, str]] | None, dialect: str | None) -> int:
    """Rewrites every comparison described above, in place, and returns how many it rewrote.

    `visible` and `dialect` place a column as the values resolver's own scope places it; `types` is
    each column's declared type by table, None reading every column as of unknown type (nothing
    moves).
    """
    scopes = _scopes(ast)
    oracle = (dialect or "").lower() == "oracle"

    def holds_times(node: exp.Expression) -> bool:
        if not isinstance(node, exp.Column):
            return False
        scope = scopes.get(id(node))
        if scope is None:
            return False
        # Bound directly to a base table of its own scope: the qualifier's source, or, unqualified,
        # every source of the scope. A CTE, a derived table or an enclosing scope's source is a
        # Scope here, and its output column's type is not the base column's.
        sources = [scope.sources.get(node.table)] if node.table else list(scope.sources.values())
        if not sources or not all(isinstance(s, exp.Table) for s in sources):
            return False
        placed = _in_scope(node, scope, visible, dialect)
        if not isinstance(placed, tuple):
            return False
        declared = (types or {}).get(placed[0], {}).get(placed[1])
        if declared is None:
            return False
        kind = _temporal_kind(declared, dialect)
        return kind == "timestamp" or (oracle and kind == "date")

    def operands(node: exp.Binary) -> tuple[exp.Expression, exp.Expression, bool] | None:
        """(column, bound, the column on the left) where one side is a timestamp column and the
        other a bare day."""
        if holds_times(node.this) and _day(node.expression):
            return node.this, node.expression, True
        if holds_times(node.expression) and _day(node.this):
            return node.expression, node.this, False
        return None

    moved = 0
    for node in list(ast.find_all(exp.LTE, exp.GTE, exp.GT, exp.LT, exp.EQ, exp.NEQ, exp.Between)):
        if isinstance(node, exp.Between):
            if node.args.get("symmetric") or not (holds_times(node.this) and _day(node.args["high"])):
                continue
            column = node.this
            node.replace(exp.Paren(this=exp.And(
                this=exp.GTE(this=column.copy(), expression=node.args["low"].copy()),
                expression=exp.LT(this=column.copy(), expression=_next_day(node.args["high"])))))
            moved += 1
            continue
        found = operands(node)
        if found is None:
            continue
        column, bound, left = found
        # Normalized to `column op bound`: the literal on the left reverses the operator.
        op = type(node) if left else {exp.LTE: exp.GTE, exp.GTE: exp.LTE, exp.GT: exp.LT,
                                      exp.LT: exp.GT}.get(type(node), type(node))
        if op is exp.LTE:
            node.replace(exp.LT(this=column.copy(), expression=_next_day(bound)))
        elif op is exp.GT:
            node.replace(exp.GTE(this=column.copy(), expression=_next_day(bound)))
        elif op is exp.EQ:
            node.replace(_the_day(column, bound))
        elif op is exp.NEQ:
            node.replace(exp.Not(this=_the_day(column, bound)))
        else:  # `<` and `>=` already mean the day's boundary
            continue
        moved += 1
    return moved
