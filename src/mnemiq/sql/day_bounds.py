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
contradiction return rows and complementary buckets count a row twice; twelve reviews by Codex each
found a path by which the same timestamp reached a midnight value unmoved -- `<=` beside `>`, `=`
beside IS DISTINCT FROM, a day inside a subquery, a cast of the column, SQLite's IS, a text or BLOB
column, a zone-aware copy, an epoch number, a function comparing internally, an aggregate's result,
a cast to DATE turned back into a timestamp. So the rule is closed by construction, on two parts.

Comparisons (`_classify`, `_agree`). Every comparison in the statement, at any depth, is between
bare columns and literals, and

- a timestamp column meets only a bare day (rewritten), a moment written with its time of day in
  ISO form (which means that moment), NULL, a number, or another timestamp column -- never a DATE,
  text, a zone-aware timestamp, any other column, a string in any other form (an engine reads
  `'2024-3-9'` as that day's midnight too), or anything opaque;
- nothing in any comparison is opaque: no expression, function, aggregate, subquery, column read
  through a CTE's output, a derived table, a set operation or an enclosing scope, or cast of
  anything but a literal (a cast to DATE, outside Oracle, is a DATE);
- a simple CASE's WHEN, BETWEEN SYMMETRIC or a mixed IN list over a timestamp column, a day as the
  subject of a BETWEEN or of an IN listing one, `IS` with a value, a bound whose day or next day no
  engine's timestamp holds, and a USING or NATURAL join (a comparison nothing here reads) are absent.

Where a time goes (`_CONSUMES`, `_CARRIES_ON`). A value computed from a timestamp column or a column
of unknown type ends only where the comparisons above read it, in what the statement shows or sorts,
or in a container whose outputs are read again as columns (a derived table, a CTE, UNION ALL -- not
INTERSECT, EXCEPT or UNION's dedup, which compare rows), passing on through an alias, MIN, MAX, a
cast to DATE or a subquery; COUNT, EXISTS and comparisons consume it. Anywhere else -- a function,
arithmetic, a CASE, a pattern match -- it may be compared unread.

Never on SQLite, which has no temporal type: a column declared TIMESTAMP may hold ISO text, a number
or anything, and a cast to TIMESTAMP is numeric affinity there.

Otherwise the statement is left exactly as written. The cost, accepted, and none on the saved runs
it was measured on: a statement with any expression in a comparison (`COUNT(*) > 5`, an IN over a
subquery, `upper(region) = 'WEST'`) or a timestamp inside a function or arithmetic (`ts + INTERVAL
1 DAY` projected) is not moved. A bare day compared with a column declared DATE (whole days, where
both readings are the same rows), a zone-aware timestamp, text or a number, or with a cast to DATE,
is not moved and stops nothing.

D is a date-only ISO literal naming a real day, spelled plain, cast or typed (an allow-list:
`_day`). A timestamp column is a bare column bound directly to a base table of its own scope,
placed by the values resolver's own-scope answer on a column declared a naive timestamp, or a DATE
on Oracle, where a DATE holds a time of day.
"""
from __future__ import annotations

from datetime import date, timedelta

from sqlglot import exp

from mnemiq.sql.values_check import _in_scope, _iso_date, _moment, _scopes, _temporal_kind

_DAY_FORMATS = frozenset({"%Y-%m-%d"})  # a parse format naming the day alone (Oracle's TO_DATE 'YYYY-MM-DD')
_WRAPPERS = (exp.Cast, exp.TryCast, exp.DateStrToDate, exp.StrToDate, exp.StrToTime)
_BINARY = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE, exp.NullSafeEQ, exp.NullSafeNEQ, exp.Is)
# The days every engine's timestamp holds whole: DuckDB's TIMESTAMP_NS runs from 1677-09-21 00:12:43
# to 2262-04-11. A bound outside them is NULL under TRY_CAST and an error under CAST, as written
# (rewritten into a NULL-safe range it turns true into unknown: Codex's ninth review) or once moved
# (Codex's eighth). A day before the first, a next day after the last, or past the calendar
# ('9999-12-31') leaves the statement as written.
_FIRST_DAY, _LAST_DAY = date(1677, 9, 22), date(2262, 4, 11)
# What consumes a time without passing it on: a comparison (read, or left as written, by the pair rules),
# COUNT, EXISTS and the connectives over comparisons.
_CONSUMES = (*_BINARY, exp.In, exp.Between, exp.Count, exp.Exists, exp.And, exp.Or, exp.Not)
# Where a time may go, or pass on from (`close_day_bounds`): what a statement shows or sorts, what
# carries it on to be checked again (an alias, MIN, MAX, a subquery), and the containers whose
# outputs are read again as columns of unknown type. A set operation only as UNION ALL: UNION's
# dedup, INTERSECT and EXCEPT compare rows, a day's midnight among them (Codex's thirteenth review).
_CARRIES_ON = (exp.Select, exp.Alias, exp.Group, exp.Order, exp.Ordered, exp.Window, exp.Distinct,
               exp.Count, exp.Min, exp.Max, exp.Subquery, exp.CTE, exp.With, exp.From, exp.Join,
               exp.Paren, exp.Exists)
_MIRROR = {exp.LT: exp.GT, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.GTE: exp.LTE}  # `bound op col` as `col op' bound`

# Operand classes (`_classify`): a bare day; a moment (an ISO timestamp string with its time of day);
# a number, boolean or NULL; any other string; a column bound directly to a base table and declared a
# naive timestamp (or an Oracle DATE); any other such column, or a cast to DATE outside Oracle; opaque.
DAY, MOMENT, NUMBER, STRING = "day", "moment", "number", "string"
TIMES, COLUMN, OPAQUE = "times", "column", "opaque"


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


def _movable(bound: exp.Expression) -> bool:
    """Whether `bound`'s day and the day after it are days every engine's timestamp holds whole."""
    day = date.fromisoformat(_day(bound).this)
    try:
        return _FIRST_DAY <= day and day + timedelta(days=1) <= _LAST_DAY
    except OverflowError:
        return False


def _the_day(column: exp.Expression, bound: exp.Expression) -> exp.Paren:
    """`(column >= bound AND column < the next day)`: the whole day `bound` names."""
    return exp.Paren(this=exp.And(this=exp.GTE(this=column.copy(), expression=bound.copy()),
                                  expression=exp.LT(this=column.copy(), expression=_next_day(bound))))


def _agree(a: str, b: str) -> bool:
    """Whether a comparison of operands of classes `a` and `b` is one of the two this allows: nothing
    opaque, and a timestamp column met only by a bare day (rewritten), a moment, a number or NULL,
    or another timestamp column."""
    pair = {a, b}
    if OPAQUE in pair:
        return False
    if TIMES in pair:
        return pair - {TIMES} <= {DAY, MOMENT, NUMBER, TIMES}
    return True


def close_day_bounds(ast: exp.Expression, visible: dict[str, set[str]],
                     types: dict[str, dict[str, str]] | None, dialect: str | None) -> int:
    """Reads the statement's bare-day comparisons by day, in place, as described above, and returns
    how many comparisons it rewrote: 0, the statement untouched, wherever any comparison in it
    cannot be read the same way.

    `visible` and `dialect` place a column as the values resolver's own scope places it; `types` is
    each column's declared type by table, None reading every column as of unknown type.
    """
    if (dialect or "").lower() == "sqlite":
        # No temporal type: a column declared TIMESTAMP may hold ISO text, a number or anything, and
        # `CAST('2024-03-09' AS TIMESTAMP)` is the integer 2024 there (Codex's seventh review), so no
        # reading of its bounds is proven.
        return 0
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
            return COLUMN
        if (isinstance(node, (exp.Cast, exp.TryCast)) and not oracle
                and _temporal_kind(node.args.get("to"), dialect) == "date"):
            return COLUMN  # a cast to DATE already reads its value by day
        return OPAQUE

    rewrites: list[tuple[exp.Expression, exp.Expression]] = []
    direct: set[int] = set()  # the operands of the comparisons read below

    for join in ast.find_all(exp.Join):
        if join.args.get("using") or str(join.args.get("method") or "").upper() == "NATURAL":
            return 0
    for node in list(ast.find_all(*_BINARY, exp.In, exp.Between, exp.Case)):
        if isinstance(node, exp.Case):
            if node.this is None:
                continue  # a searched CASE compares nothing itself; its WHENs are comparisons of their own
            subject = _classify(node.this)
            direct.add(id(node.this))
            for branch in node.args.get("ifs") or []:
                value = _classify(branch.this)
                direct.add(id(branch.this))
                if not _agree(subject, value) or {subject, value} == {TIMES, DAY}:
                    return 0  # `CASE col WHEN D`: an equality this does not rewrite
            continue
        if isinstance(node, exp.Between):
            subject, low, high = node.this, node.args["low"], node.args["high"]
            classes = [_classify(subject), _classify(low), _classify(high)]
            direct.update((id(subject), id(low), id(high)))
            if DAY in (classes[0],) or not (_agree(classes[0], classes[1]) and _agree(classes[0], classes[2])):
                return 0  # a day as BETWEEN's subject, or a pair that disagrees
            if classes[0] == TIMES and DAY in classes[1:]:
                if node.args.get("symmetric"):
                    return 0
                if classes[2] == DAY:
                    if not _movable(high):
                        return 0
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
            direct.update((id(node.this), *(id(i) for i in items)))
            if not items or not all(_agree(subject, c) for c in classes):
                return 0
            if subject == DAY and TIMES in classes:
                return 0  # a day as the subject, a timestamp in the list: a membership this does not rewrite
            if subject == TIMES and DAY in classes:
                if not all(c == DAY for c in classes):
                    return 0  # a list mixing days and moments
                if not all(_movable(i) for i in items):
                    return 0
                days = [_the_day(node.this, i) for i in items]
                either = days[0]
                for d in days[1:]:
                    either = exp.Or(this=either, expression=d)
                rewrites.append((node, exp.Paren(this=either) if len(days) > 1 else days[0]))
            continue
        left, right = node.this, node.expression
        a, b = _classify(left), _classify(right)
        direct.update((id(left), id(right)))
        if not _agree(a, b):
            return 0
        if {a, b} != {TIMES, DAY}:
            continue
        column, bound = (left, right) if a == TIMES else (right, left)
        op = type(node) if a == TIMES else _MIRROR.get(type(node), type(node))
        if op not in (exp.LT, exp.GTE) and not _movable(bound):
            return 0
        if op is exp.LTE:
            rewrites.append((node, exp.LT(this=column.copy(), expression=_next_day(bound))))
        elif op is exp.GT:
            rewrites.append((node, exp.GTE(this=column.copy(), expression=_next_day(bound))))
        elif op is exp.EQ:
            rewrites.append((node, _the_day(column, bound)))
        elif op is exp.NEQ:
            rewrites.append((node, exp.Not(this=_the_day(column, bound))))
        elif op in (exp.NullSafeEQ, exp.NullSafeNEQ):
            held = exp.Paren(this=exp.And(this=exp.Not(this=exp.Is(this=column.copy(), expression=exp.Null())),
                                          expression=_the_day(column, bound).this))
            rewrites.append((node, exp.Not(this=held) if op is exp.NullSafeNEQ else held))
        elif op is exp.Is:
            return 0  # `IS` with a value: SQLite's NULL-safe equality, which no other engine has
        # `<` and `>=` already mean the day's boundary

    # Where a time goes. A value computed from a timestamp column or a column of unknown type --
    # through anything but a comparison, COUNT or EXISTS, which consume it -- carries that time, and
    # may end only where the classifier read it (a comparison's direct operand), in what the
    # statement shows (a projection, GROUP BY, ORDER BY, a window's PARTITION BY or ORDER BY), or in
    # a container whose outputs are read again as columns (a derived table, a CTE, a set operation),
    # passing on through an alias, MIN or MAX, a cast to DATE outside Oracle, or a subquery. Reaching
    # anything else -- a function, arithmetic, a CASE -- it may be compared there with a day unread
    # (`list_contains([ts], TIMESTAMP D)`, `NULLIF(MIN(ts), TIMESTAMP D)`, `NULLIF(CAST(ts AS DATE) +
    # INTERVAL '12 hours', TIMESTAMP D)`: Codex's tenth to twelfth reviews), and the statement is
    # left as written.
    carries: set[int] = set()
    for node in reversed(list(ast.walk())):  # every node after its descendants
        if isinstance(node, exp.Column):
            if _classify(node) in (TIMES, OPAQUE):
                carries.add(id(node))
        elif not isinstance(node, _CONSUMES) and any(id(c) in carries for c in node.iter_expressions()):
            carries.add(id(node))
    for node in ast.walk():
        if id(node) in carries and node.parent is not None and id(node) not in direct:
            parent = node.parent
            if not (isinstance(parent, _CARRIES_ON) or _day_cast(parent, oracle)
                    or (isinstance(parent, exp.Union) and parent.args.get("distinct") is False)):
                return 0

    for node, replacement in rewrites:
        node.replace(replacement)
    return len(rewrites)


def _day_cast(node: exp.Expression, oracle: bool) -> bool:
    """A cast to DATE outside Oracle (whose DATE keeps the time of day): its value is its operand's day."""
    return (isinstance(node, (exp.Cast, exp.TryCast)) and not oracle
            and _temporal_kind(node.args.get("to"), None) == "date")
