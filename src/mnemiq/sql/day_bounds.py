"""A range written to end on a bare date against a timestamp column ends at the next day's start.

`ts <= '2024-03-09'` compares with midnight at the START of the 9th, so every row later that day
falls out, where the range a person means by "as of", "through" or "up to" the 9th ends before the
10th. Measured on saved runs of a local 32B model: it wrote this bound on most runs of the questions
that turn on it, every such answer was wrong, and ending the range at the next day turned them
right and turned no right answer wrong.

Moved: `col <= D` and its mirror `D >= col` become `col < D+1` and `D+1 > col`, and `col BETWEEN x
AND D` becomes `(col >= x AND col < D+1)`, where D is a date-only ISO literal naming a real day,
spelled plain, cast or typed (an allow-list: `_day`), and `col` is a bare column the values
resolver places on a column declared a naive timestamp, or a DATE on Oracle, where a DATE holds a
time of day. The next day is written in the literal's own spelling, so the statement stays valid
in the dialect it was written in.

Not moved: a DATE elsewhere, whose whole days make the two bounds the same rows; a zone-aware
timestamp, whose midnight is the session's zone's; text, which compares as text; an expression or
a column the resolver cannot place or whose type is not declared; a bound with a time of day,
which says where it ends; and `>`, `=` and `BETWEEN SYMMETRIC`, which no measurement covers.
"""
from __future__ import annotations

from datetime import date, timedelta

from sqlglot import exp

from mnemiq.sql.values_check import _iso_date, _resolver, _temporal_kind

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


def close_day_bounds(ast: exp.Expression, visible: dict[str, set[str]],
                     types: dict[str, dict[str, str]] | None, dialect: str | None) -> int:
    """Moves every bound described above, in place, and returns how many it moved.

    `visible` and `dialect` place a column as `ground_values` places it (`_resolver`), so a column
    reached through a CTE or an alias is the base column the engine reads; `types` is each
    column's declared type by table, None reading every column as of unknown type (nothing moves).
    """
    resolve = _resolver(ast, visible, dialect)
    oracle = (dialect or "").lower() == "oracle"

    def holds_times(node: exp.Expression) -> bool:
        if not isinstance(node, exp.Column):
            return False
        placed = resolve(node)
        if placed is None:
            return False
        declared = (types or {}).get(placed[0], {}).get(placed[1])
        if declared is None:
            return False
        kind = _temporal_kind(declared, dialect)
        return kind == "timestamp" or (oracle and kind == "date")

    moved = 0
    for node in list(ast.find_all(exp.LTE, exp.GTE, exp.Between)):
        if isinstance(node, exp.LTE) and holds_times(node.this) and _day(node.expression):
            node.replace(exp.LT(this=node.this.copy(), expression=_next_day(node.expression)))
        elif isinstance(node, exp.GTE) and holds_times(node.expression) and _day(node.this):
            node.replace(exp.GT(this=_next_day(node.this), expression=node.expression.copy()))
        elif (isinstance(node, exp.Between) and not node.args.get("symmetric")
              and holds_times(node.this) and _day(node.args["high"])):
            column = node.this
            node.replace(exp.Paren(this=exp.And(
                this=exp.GTE(this=column.copy(), expression=node.args["low"].copy()),
                expression=exp.LT(this=column.copy(), expression=_next_day(node.args["high"])))))
        else:
            continue
        moved += 1
    return moved
