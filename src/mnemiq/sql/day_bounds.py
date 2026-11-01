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

One statement, one reading. Moving some of a statement's comparisons and not others let a
contradiction return rows and complementary buckets count a row twice (Codex's reviews: `<=` moved
beside `>`, `=` beside IS DISTINCT FROM, a day inside a subquery, a cast of the same column). So
every comparison in the statement, at any depth, is classified by its operands (`_classify`), and
the statement moves only where every one is read the same way in both readings or is one this
rewrites:

- a timestamp column meets only a bare day (rewritten), a moment written with its time of day in
  ISO form (which means that moment), NULL, or another timestamp column -- never text, a string in
  any other form (an engine reads `'2024-3-9'` as that day's midnight too), a DATE, or an opaque
  operand;
- an opaque operand -- an expression, a function, a subquery, a column read through a CTE's output,
  a derived table, a set operation or an enclosing scope, a cast of anything but a literal to
  anything but DATE -- meets only a number, a boolean, NULL, or a column of a type neither temporal
  nor text: an engine converts a string to a timestamp where an opaque timestamp meets one (Codex's
  fourth review), and a number never carries a day;
- a simple CASE's WHEN, BETWEEN SYMMETRIC or a mixed IN list over a timestamp column, LIKE or a
  pattern match on one, and a USING or NATURAL join (a comparison nothing here reads), leave it as
  written too. SQLite's `IS` and `IS NOT` are the NULL-safe pair.

A bare day buried in an operand (a function's argument, an array, ANY, a subquery) makes that
operand opaque, so it never meets a timestamp column or a bare day unread. The cost, accepted: a
statement comparing an expression with a string (`upper(region) = 'WEST'`) is left as written.

Otherwise the statement is left exactly as written. A bare day compared with a column declared
DATE (whole days, where both readings are the same rows), a zone-aware timestamp, text or a
number, or with a cast to DATE, is not moved and stops nothing.

D is a date-only ISO literal naming a real day, spelled plain, cast or typed (an allow-list:
`_day`). A timestamp column is a bare column bound directly to a base table of its own scope,
placed by the values resolver's own-scope answer on a column declared a naive timestamp, or a DATE
on Oracle, where a DATE holds a time of day.
"""
from __future__ import annotations

from datetime import date, timedelta

from sqlglot import exp

from sqlglot.errors import SqlglotError

from mnemiq.sql.values_check import _in_scope, _iso_date, _moment, _scopes, _temporal_kind

_DAY_FORMATS = frozenset({"%Y-%m-%d"})  # a parse format naming the day alone (Oracle's TO_DATE 'YYYY-MM-DD')
_WRAPPERS = (exp.Cast, exp.TryCast, exp.DateStrToDate, exp.StrToDate, exp.StrToTime)
_BINARY = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ, exp.Is)
_PATTERNS = (exp.Like, exp.ILike, exp.Glob, exp.SimilarTo, exp.RegexpLike, exp.RegexpILike)
_MIRROR = {exp.LT: exp.GT, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.GTE: exp.LTE}  # `bound op col` as `col op' bound`

# Operand classes (`_classify`): a bare day; a moment (an ISO timestamp string with its time of day);
# a number, boolean or NULL; any other string; a typed column bound directly to a base table (a naive
# timestamp, a DATE, a zone-aware timestamp, text, anything else); opaque.
DAY, MOMENT, NUMBER, STRING = "day", "moment", "number", "string"
TIMES, DATE, ZONED, TEXT, OTHER, OPAQUE = "times", "date", "zoned", "text", "other", "opaque"


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


def _is_text(declared: str, dialect: str | None) -> bool:
    """Whether a declared type is a string type; an unreadable one counts as text, the reading that
    leaves more statements as written."""
    try:
        return exp.DataType.build(declared, dialect=dialect, udt=True).is_type(*exp.DataType.TEXT_TYPES)
    except (SqlglotError, ValueError):
        return True


def _agree(a: str, b: str) -> bool:
    """Whether a comparison of operands of classes `a` and `b` reads the same in both readings, or
    is one this rewrites (a timestamp column and a bare day)."""
    pair = {a, b}
    if TIMES in pair:
        return pair - {TIMES} <= {DAY, MOMENT, NUMBER, TIMES, ZONED}
    if OPAQUE in pair:
        return len(pair) == 2 and pair - {OPAQUE} <= {NUMBER, OTHER}
    return True


def close_day_bounds(ast: exp.Expression, visible: dict[str, set[str]],
                     types: dict[str, dict[str, str]] | None, dialect: str | None) -> int:
    """Reads the statement's bare-day comparisons by day, in place, as described above, and returns
    how many comparisons it rewrote: 0, the statement untouched, wherever any comparison in it
    cannot be read the same way.

    `visible` and `dialect` place a column as the values resolver's own scope places it; `types` is
    each column's declared type by table, None reading every column as of unknown type.
    """
    scopes = _scopes(ast)
    oracle = (dialect or "").lower() == "oracle"

    def _literal(node: exp.Literal) -> str:
        if not node.is_string:
            return NUMBER
        return MOMENT if _moment(node.this) == "timestamp" else STRING

    def _classify(node: exp.Expression) -> str:
        if _day(node):
            return DAY
        if isinstance(node, (exp.Null, exp.Boolean)):
            return NUMBER
        if isinstance(node, exp.Literal):
            return _literal(node)
        if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and not node.this.is_string:
            return NUMBER
        if isinstance(node, _WRAPPERS) and isinstance(node.this, exp.Literal):
            # A constant keeps its literal's class: a day in a spelling `_day` does not read (a cast
            # to text, a format that is not the day alone) is a string, which no timestamp meets.
            return _literal(node.this)
        if isinstance(node, exp.Column):
            scope = scopes.get(id(node))
            if scope is None:
                return OPAQUE
            # Bound directly to a base table of its own scope: the qualifier's source, or every
            # source when unqualified. A CTE, a derived table or an enclosing scope's source is a
            # Scope here, and its output column's type is not the base column's.
            sources = [scope.sources.get(node.table)] if node.table else list(scope.sources.values())
            if not sources or not all(isinstance(s, exp.Table) for s in sources):
                return OPAQUE
            placed = _in_scope(node, scope, visible, dialect)
            declared = (types or {}).get(placed[0], {}).get(placed[1]) if isinstance(placed, tuple) else None
            if declared is None:
                return OPAQUE
            kind = _temporal_kind(declared, dialect)
            if kind == "timestamp" or (oracle and kind == "date"):
                return TIMES
            if kind is not None:
                return {"date": DATE, "timestamptz": ZONED}.get(kind, OPAQUE)  # a TIME: opaque
            return TEXT if _is_text(declared, dialect) else OTHER
        if (isinstance(node, (exp.Cast, exp.TryCast)) and not oracle
                and _temporal_kind(node.args.get("to"), dialect) == "date"):
            return DATE  # a cast to DATE already reads its value by day
        return OPAQUE

    rewrites: list[tuple[exp.Expression, exp.Expression]] = []

    for join in ast.find_all(exp.Join):
        if join.args.get("using") or str(join.args.get("method") or "").upper() == "NATURAL":
            return 0
    for match in ast.find_all(*_PATTERNS):
        if TIMES in (_classify(match.this), _classify(match.expression)):
            return 0
    for node in list(ast.find_all(*_BINARY, exp.In, exp.Between, exp.Case)):
        if isinstance(node, exp.Case):
            if node.this is None:
                continue  # a searched CASE compares nothing itself; its WHENs are comparisons of their own
            subject = _classify(node.this)
            for branch in node.args.get("ifs") or []:
                value = _classify(branch.this)
                if not _agree(subject, value) or {subject, value} == {TIMES, DAY}:
                    return 0  # `CASE col WHEN D`: an equality this does not rewrite
            continue
        if isinstance(node, exp.Between):
            subject, low, high = node.this, node.args["low"], node.args["high"]
            classes = [_classify(subject), _classify(low), _classify(high)]
            if DAY in (classes[0],) or not (_agree(classes[0], classes[1]) and _agree(classes[0], classes[2])):
                return 0  # a day as BETWEEN's subject, or a pair that disagrees
            if classes[0] == TIMES and DAY in classes[1:]:
                if node.args.get("symmetric"):
                    return 0
                if classes[2] == DAY:
                    rewrites.append((node, exp.Paren(this=exp.And(
                        this=exp.GTE(this=subject.copy(), expression=low.copy()),
                        expression=exp.LT(this=subject.copy(), expression=_next_day(high))))))
                # a day as the low end is the inclusive start, the day's own boundary
            continue
        if isinstance(node, exp.In):
            subject = _classify(node.this)
            items = list(node.expressions)
            if node.args.get("query") is not None:
                items = [node.args["query"]]
            classes = [_classify(i) for i in items]
            if not items or not all(_agree(subject, c) for c in classes):
                return 0
            if subject == TIMES and DAY in classes:
                if not all(c == DAY for c in classes):
                    return 0  # a list mixing days and moments
                days = [_the_day(node.this, i) for i in items]
                either = days[0]
                for d in days[1:]:
                    either = exp.Or(this=either, expression=d)
                rewrites.append((node, exp.Paren(this=either) if len(days) > 1 else days[0]))
            continue
        left, right = node.this, node.expression
        a, b = _classify(left), _classify(right)
        if not _agree(a, b):
            return 0
        if {a, b} != {TIMES, DAY}:
            continue
        column, bound = (left, right) if a == TIMES else (right, left)
        op = type(node) if a == TIMES else _MIRROR.get(type(node), type(node))
        if op is exp.LTE:
            rewrites.append((node, exp.LT(this=column.copy(), expression=_next_day(bound))))
        elif op is exp.GT:
            rewrites.append((node, exp.GTE(this=column.copy(), expression=_next_day(bound))))
        elif op is exp.EQ:
            rewrites.append((node, _the_day(column, bound)))
        elif op is exp.NEQ:
            rewrites.append((node, exp.Not(this=_the_day(column, bound))))
        elif op in (exp.NullSafeEQ, exp.NullSafeNEQ, exp.Is):  # SQLite's `IS` is IS NOT DISTINCT FROM
            held = exp.Paren(this=exp.And(this=exp.Not(this=exp.Is(this=column.copy(), expression=exp.Null())),
                                          expression=_the_day(column, bound).this))
            rewrites.append((node, exp.Not(this=held) if op is exp.NullSafeNEQ else held))
        # `<` and `>=` already mean the day's boundary

    for node, replacement in rewrites:
        node.replace(replacement)
    return len(rewrites)
