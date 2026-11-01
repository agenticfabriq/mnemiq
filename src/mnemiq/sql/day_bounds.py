"""A bare date compared with a timestamp column is read as the whole day it names.

`ts <= '2024-03-09'` compares with midnight at the START of the 9th, so every row later that day
falls out, where the range a person means by "as of", "through" or "up to" the 9th ends before the
10th. Measured on saved runs of a local 32B model: it wrote this bound on most runs of the questions
that turn on it, every such answer was wrong, and ending the range at the next day turned them
right and turned no right answer wrong.

Every comparison of such a column with a bare day is read by day, as `CAST(col AS DATE) op D`
reads it, the literal on either side and the column written first:

    col <= D   ->  col < D+1                 col > D    ->  col >= D+1
    col = D    ->  (col >= D AND col < D+1)  col <> D   ->  NOT (col >= D AND col < D+1)
    col IN (D1, D2)              ->  ((col >= D1 AND col < D1+1) OR (col >= D2 AND col < D2+1))
    col IS NOT DISTINCT FROM D   ->  (col IS NOT NULL AND col >= D AND col < D+1)
    col BETWEEN x AND D          ->  (col >= x AND col < D+1)

with NOT IN, IS DISTINCT FROM and NOT BETWEEN the negations, NULL kept as each form keeps it, and
D+1 written in the literal's own spelling so the statement stays valid in its dialect. `col < D`
and `col >= D` already mean the day's boundary and stay as written.

One statement, one reading. A statement's predicates must keep meaning together what they meant
together: moving `=` beside an unmoved `IS DISTINCT FROM`, or `<=` beside `>`, let a contradiction
return rows and complementary buckets count a row twice (Codex's reviews). So the statement is left
exactly as written wherever any bare day is compared with something this cannot read by day -- a
column read through a CTE's output, a derived table, a set operation or an enclosing scope; an
expression of no known type; a simple CASE's WHEN; BETWEEN SYMMETRIC; an IN list holding anything
but bare days; a day reaching a comparison inside a function, an array or ANY. A bare day compared
with a column declared DATE (whole days, where both readings are the same rows), text, a number or
a zone-aware timestamp, or with a cast to such a type, is no obstacle and is not moved.

D is a date-only ISO literal naming a real day, spelled plain, cast or typed (an allow-list:
`_day`). The column is a bare column bound directly to a base table of its own scope, placed by the
values resolver's own-scope answer on a column declared a naive timestamp, or a DATE on Oracle,
where a DATE holds a time of day. The resolver's wider readings are not used: past its own scope it
can place a set operation's text column on a timestamp column of the same name (Codex's review).
"""
from __future__ import annotations

from datetime import date, timedelta

from sqlglot import exp

from mnemiq.sql.values_check import _in_scope, _iso_date, _scopes, _temporal_kind

_DAY_FORMATS = frozenset({"%Y-%m-%d"})  # a parse format naming the day alone (Oracle's TO_DATE 'YYYY-MM-DD')
_WRAPPERS = (exp.Cast, exp.TryCast, exp.DateStrToDate, exp.StrToDate, exp.StrToTime)
_BINARY = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ)
# The comparison a day can reach through something other than a direct operand: any of these above
# it, short of the query it sits in, means a value compared in a way this does not read.
_COMPARING = (*_BINARY, exp.In, exp.Between, exp.Any, exp.All)
# `bound op col` read as `col op' bound`.
_MIRROR = {exp.LT: exp.GT, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.GTE: exp.LTE}


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
    """Reads the statement's bare-day comparisons by day, in place, as described above, and returns
    how many comparisons it rewrote: 0, the statement untouched, wherever one cannot be read so.

    `visible` and `dialect` place a column as the values resolver's own scope places it; `types` is
    each column's declared type by table, None reading every column as of unknown type.
    """
    scopes = _scopes(ast)
    oracle = (dialect or "").lower() == "oracle"

    def kind_of_column(node: exp.Column) -> str | None:
        """The declared temporal kind of a column bound directly to a base table of its own scope
        ("" for a declared column of no temporal kind), or None where it is not so bound or placed,
        or its type is not declared."""
        scope = scopes.get(id(node))
        if scope is None:
            return None
        # The qualifier's source, or, unqualified, every source of the scope, must be a base table:
        # a CTE, a derived table or an enclosing scope's source is a Scope here, and its output
        # column's type is not the base column's.
        sources = [scope.sources.get(node.table)] if node.table else list(scope.sources.values())
        if not sources or not all(isinstance(s, exp.Table) for s in sources):
            return None
        placed = _in_scope(node, scope, visible, dialect)
        if not isinstance(placed, tuple):
            return None
        declared = (types or {}).get(placed[0], {}).get(placed[1])
        if declared is None:
            return None
        return _temporal_kind(declared, dialect) or ""

    def read(node: exp.Expression) -> str:
        """"times" for a column whose bare-day comparisons are read by day; "settled" for an operand
        whose comparison with a bare day means the same in both readings or is not this one's to
        read; "unknown" for anything else, which leaves the statement as written."""
        if isinstance(node, exp.Column):
            kind = kind_of_column(node)
            if kind is None:
                return "unknown"
            if kind == "timestamp" or (oracle and kind == "date"):
                return "times"
            return "settled"  # a DATE's whole days, text, a number, a zone-aware timestamp
        if isinstance(node, (exp.Literal, exp.Null, exp.Boolean)):
            return "settled"
        if isinstance(node, (exp.Cast, exp.TryCast)):
            kind = _temporal_kind(node.args.get("to"), dialect)
            if kind is None or (kind == "date" and not oracle) or kind == "timestamptz":
                return "settled"
        return "unknown"

    rewrites: list[tuple[exp.Expression, exp.Expression]] = []
    for literal in list(ast.find_all(exp.Literal)):
        if not (literal.is_string and _iso_date(literal.this)):
            continue
        bound = literal
        if isinstance(literal.parent, _WRAPPERS) and _day(literal.parent) is literal:
            bound = literal.parent
        parent = bound.parent
        if isinstance(parent, _BINARY):
            left = parent.expression is bound
            other = parent.this if left else parent.expression
            seen = read(other)
            if seen == "unknown":
                return 0
            if seen == "times":
                op = type(parent) if left else _MIRROR.get(type(parent), type(parent))
                if op is exp.LTE:
                    rewrites.append((parent, exp.LT(this=other.copy(), expression=_next_day(bound))))
                elif op is exp.GT:
                    rewrites.append((parent, exp.GTE(this=other.copy(), expression=_next_day(bound))))
                elif op is exp.EQ:
                    rewrites.append((parent, _the_day(other, bound)))
                elif op is exp.NEQ:
                    rewrites.append((parent, exp.Not(this=_the_day(other, bound))))
                elif op in (exp.NullSafeEQ, exp.NullSafeNEQ):
                    held = exp.Paren(this=exp.And(this=exp.Not(this=exp.Is(this=other.copy(), expression=exp.Null())),
                                                  expression=_the_day(other, bound).this))
                    rewrites.append((parent, held if op is exp.NullSafeEQ else exp.Not(this=held)))
                # `<` and `>=` already mean the day's boundary
        elif isinstance(parent, exp.Between) and bound is not parent.this:
            seen = read(parent.this)
            if seen == "unknown" or (seen == "times" and parent.args.get("symmetric")):
                return 0
            if seen == "times" and bound is parent.args.get("high"):
                column = parent.this
                rewrites.append((parent, exp.Paren(this=exp.And(
                    this=exp.GTE(this=column.copy(), expression=parent.args["low"].copy()),
                    expression=exp.LT(this=column.copy(), expression=_next_day(bound))))))
            # a day as the low end is the inclusive start, the day's own boundary
        elif isinstance(parent, exp.In) and bound is not parent.this:
            seen = read(parent.this)
            if seen == "unknown":
                return 0
            if seen == "times":
                items = list(parent.expressions)
                if parent.args.get("query") or not items or not all(_day(i) for i in items):
                    return 0
                if items[0] is bound:  # the list rewritten once, by its first item
                    days = [_the_day(parent.this, i) for i in items]
                    either = days[0]
                    for d in days[1:]:
                        either = exp.Or(this=either, expression=d)
                    rewrites.append((parent, exp.Paren(this=either) if len(days) > 1 else days[0]))
        elif (isinstance(parent, exp.If) and isinstance(parent.parent, exp.Case)
              and parent.parent.this is not None and parent.this is bound):
            if read(parent.parent.this) != "settled":
                return 0  # `CASE col WHEN D`: an equality this does not rewrite
        elif any(isinstance(a, _COMPARING) for a in _up_to_query(bound)):
            return 0  # compared, but not as a direct operand: a simple CASE, BETWEEN's subject, ANY
    for node, replacement in rewrites:
        node.replace(replacement)
    return len(rewrites)


def _up_to_query(node: exp.Expression):
    """`node`'s ancestors, short of the query it sits in."""
    node = node.parent
    while node is not None and not isinstance(node, exp.Query):
        yield node
        node = node.parent
