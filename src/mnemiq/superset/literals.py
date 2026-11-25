"""A plan's filter on a value no row the builder can read holds, found with the ask path's own grounding.

The planner never sees a column's values, so a plan can filter on a value its column never holds (`'Won'` where the
column holds `WON`): the part then reads the wrong rows, and nothing checks it, since mnemiq's own statement for the
part repeats the same filter. So each string a plan's statement compares a column with is asked of the source, through
the request's adapter, as the builder: Superset's row rules apply to every statement it runs.

Nothing here reads a statement on its own terms. `ground_values` finds the values and the column each is compared
with, exactly as it does for an empty answer on the ask path; `confirm_by_source` asks the source about each, at most
`PROBE_LIMIT` of them per statement; `source_holds` asks. Only a value compared by equality or IN with a bare column
the resolver places on a table the builder may read is ever asked: one compared with an expression (`LOWER(status) =
'won'`), one on a masked or row-filtered column, and one on a table the grants hide are left unchecked, and refuse
nothing. Neither does a value the source does not answer for.

For a value the source rules out, the values its column does hold are read the same way, as the builder, so a builder
whose row rules narrow the column is told only the values they may read. Those are the only values a plan is ever told,
and only through the refusal of its own part. A log line names a column at most, never a value.

Every probe and read asks Superset to run it afresh (`SupersetSqlAdapter.execute_arrow_afresh`), never to answer from
its result cache: a cached empty answer could refuse a value the column holds now, and cached values could tell the
repair what the column used to hold.

One check of a plan shares one `Probes`: the time every probe and read of the plan may take together, the ask path's
`PROBE_BUDGET_S` for a turn's probes, since an equality on a column with no index, and a read of a column's distinct
values, can each scan the table; and what the source already said, so a value two parts filter on is asked once and a
column's values are read once. A value not asked once the time is spent stays unchecked, and refuses nothing.

**Whether a column holds NULL** is asked the same way (`holds_null`), for a chart filter that negates (`!=`, `NOT IN`,
`NOT LIKE`): SQL reads a comparison with NULL as unknown, never true, so such a filter leaves out every row where its
column is NULL as well as the rows it names. The probe answers only whether one row the builder can read holds NULL
there, never a value; it shares the plan's `Probes`, once per column, and is asked of no column the grounding would not
place: one masked or denied, one on a table the grants hide or filter by row. A probe that fails, or is not sent, says
nothing, and refuses nothing.

**Whether a joined table's key holds each value once** is asked the same way (`unique_live`), for a dataset the build
makes that LEFT JOINs a table by columns enrichment's profile says nothing of: the rows holding a key against their
distinct keys, two counts, never a value. It shares the plan's `Probes`, once per key, and is asked of no column or
table `holds_null` would not ask of. Unlike the probes above, a probe that fails or is not sent says nothing, and the
plan check then joins nothing on it: a key is joined only where it is shown to hold each value once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from mnemiq.agent.loop import PROBE_BUDGET_S
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot
from mnemiq.semantic.values import shown
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import column_types, visible_schema
from mnemiq.sql.values_check import (
    ABSENT,
    PROBE_LIMIT,
    UNCHECKED,
    confirm_by_source,
    ground_values,
    held_values_sql,
    source_holds,
)

__all__ = ["HELD_NAMED", "NEGATED_OPERATORS", "Absent", "Probes", "absent_literals", "filter_statement", "holds_null",
           "null_statement", "unique_live", "unique_statement"]

logger = logging.getLogger(__name__)

# The most values a refusal names of a column's; one more is read, to tell whether there are others.
HELD_NAMED = 20

# A chart filter's operators whose string operand is a value the column must hold for the filter to mean anything: `=`
# and `IN` select no row otherwise, and `!=` and `NOT IN` select every row, which is no filter at all.
_EQUALITY_OPERATORS = frozenset({"=", "!=", "IN", "NOT IN"})

# A chart filter's operators that leave out the rows where the column is NULL beside the rows they name: SQL reads
# `NULL != 'x'`, `NULL NOT IN (...)` and `NULL NOT LIKE '...'` as unknown, never true. The planner's operators have no
# `NOT ILIKE`.
NEGATED_OPERATORS: frozenset[str] = frozenset({"!=", "NOT IN", "NOT LIKE"})


@dataclass(frozen=True)
class Absent:
    """A value a statement filters `column` (`table.column`) on that no row the builder can read holds, with the first
    values the column does hold (`held`, at most `HELD_NAMED`, each as a prompt shows a value) and whether it holds
    more. `held` is empty where the source said the builder can read no value of the column, and None where it did not
    say: the read failed, or the plan's probe time was spent before it."""

    column: str
    literal: str
    held: tuple[str, ...] | None
    more: bool


@dataclass
class Probes:
    """What one check of a plan's probes share: `budget_s`, the time they may take together, `spent_s` of it taken as
    `clock` measures it, each probe's answer by `(table, column, text)`, each column's values by `(table, column)`,
    whether each column holds NULL by `(table, column)`, and whether each key holds each value once by `(table, its
    columns sorted)`. A call is given at most the time left, and none is made once it is spent. The budget is read
    between calls, so one call can overrun it by the transport's own margin, as a probe on the ask path can."""

    budget_s: float = PROBE_BUDGET_S
    clock: Callable[[], float] = time.monotonic
    spent_s: float = 0.0
    answers: dict[tuple[str, str, str], bool | None] = field(default_factory=dict)
    held: dict[tuple[str, str], tuple[tuple[str, ...], bool] | None] = field(default_factory=dict)
    nulls: dict[tuple[str, str], bool | None] = field(default_factory=dict)
    unique: dict[tuple[str, tuple[str, ...]], bool | None] = field(default_factory=dict)

    def left_s(self) -> float:
        return self.budget_s - self.spent_s

    def timed(self, call: Callable[[float], object], timeout_s: float):
        """`call(limit)` with `limit` the smaller of `timeout_s` and the time left, its time counted as spent."""
        started = self.clock()
        try:
            return call(min(timeout_s, self.left_s()))
        finally:
            self.spent_s += self.clock() - started


class _Afresh:
    """`adapter` as the ask path's `source_holds` and the read of a column's values use it, every statement run afresh
    (`execute_arrow_afresh`). An adapter without that method fails each read, which leaves the value unchecked."""

    def __init__(self, adapter) -> None:
        self._adapter = adapter

    def execute_arrow(self, sql: str, timeout_s: float | None = None):
        return self._adapter.execute_arrow_afresh(sql, timeout_s=timeout_s)


def filter_statement(table: str, column: str, op: str, operand, dialect: str | None) -> str | None:
    """A chart filter as a statement `absent_literals` reads: `SELECT 1 FROM <table> WHERE <column> IN (<its
    strings>)`, the names quoted as stored. None for an operator other than `=`, `!=`, `IN` and `NOT IN`, and for an
    operand that holds no string."""
    if op not in _EQUALITY_OPERATORS:
        return None
    strings = [item for item in (operand if isinstance(operand, list) else [operand]) if isinstance(item, str)]
    if not strings:
        return None
    return (exp.select(exp.Literal.number(1))
            .from_(exp.Table(this=exp.to_identifier(table, quoted=True)))
            .where(exp.In(this=exp.Column(this=exp.to_identifier(column, quoted=True)),
                          expressions=[exp.Literal.string(item) for item in strings]))
            .sql(dialect=dialect))


def absent_literals(sql: str, *, adapter, snapshot: Snapshot, grants: GrantSet, timeout_s: float,
                    probes: Probes | None = None) -> list[Absent]:
    """Each value `sql` filters a column on that the source, asked through `adapter`, says no row holds, in the order
    the statement holds them, once each. Grounded as the decider grounds a statement (`ground_values`, with what
    `grants` leave visible, their masks and row filters, the snapshot's column types and no value index), and asked of
    the source as the ask path asks it (`confirm_by_source`, `source_holds`), within `probes` (a plan's shared budget and
    answers; a fresh one when None). A statement that does not parse, or a grounding that cannot be read, refuses
    nothing, and nor does a value not asked once the time is spent."""
    if getattr(adapter, "terminal_refusal", None) is not None:
        return []  # Superset refused this request for good: every later call meets the same refusal
    dialect = adapter.dialect
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        policy = build_access_policy(snapshot, grants)

        def withheld(table: str, column: str) -> bool:
            # Whether a value exists is what a mask withholds; a denied column is never the builder's to ask about.
            return policy.masks(table, column) or policy.denies(table, column)

        grounding = ground_values(tree, visible_schema(snapshot, grants), None, masks=withheld, dialect=dialect,
                                  types=column_types(snapshot), row_filtered=set(policy.row_filters))
    except Exception as exc:  # noqa: BLE001 -- a statement or a grounding unread refuses nothing
        # The kind only: a parser's words quote the statement, values and all.
        logger.info("a plan's filter values could not be read (%s); none is checked", type(exc).__name__)
        return []
    probes = Probes() if probes is None else probes
    afresh = _Afresh(adapter)
    physical = {binding.object_id: binding.source_object for binding in snapshot.source_bindings}
    asked: set[tuple[str, str, str]] = set()
    late: set[tuple[str, str, str]] = set()

    def holds(table: str, column: str, text: str) -> bool | None:
        key = (table, column, text)
        asked.add(key)
        if key not in probes.answers:
            if probes.left_s() <= 0:
                late.add(key)
                return None  # not kept: unasked is not unanswered
            probes.answers[key] = probes.timed(lambda limit: source_holds(afresh, dialect, physical, limit)(*key),
                                               timeout_s)
        return probes.answers[key]

    confirmed = confirm_by_source(grounding, holds)
    probeable = {(*value.placed, value.text) for value in grounding.values
                 if value.outcome == UNCHECKED and value.why == "not_indexed" and value.placed is not None}
    if left := len(probeable - asked):
        logger.info("%d filter values were left unprobed past the limit of %d per statement; they stay unchecked and "
                    "refuse nothing", left, PROBE_LIMIT)
    if late:
        logger.info("%d filter values were left unprobed: the plan's %s s for probes was spent; they stay unchecked "
                    "and refuse nothing", len(late), probes.budget_s)
    found: list[Absent] = []
    seen: set[tuple[str, str, str]] = set()
    for value in confirmed.values:
        if value.outcome != ABSENT or value.why != "probed" or value.placed is None:
            continue
        key = (*value.placed, value.text)
        if key in seen:
            continue
        seen.add(key)
        if value.placed not in probes.held:
            if probes.left_s() <= 0:
                logger.info("the values %s.%s holds were not read: the plan's %s s for probes was spent",
                            *value.placed, probes.budget_s)
                read = None  # not kept, as an unasked probe is not
            else:
                read = probes.held[value.placed] = probes.timed(
                    lambda limit, placed=value.placed: _held(afresh, physical, *placed, dialect=dialect,
                                                             timeout_s=limit), timeout_s)
        else:
            read = probes.held[value.placed]
        values, more = read if read is not None else (None, False)
        found.append(Absent(f"{value.placed[0]}.{value.placed[1]}", value.text, values, more))
    return found


def null_statement(table: str, column: str, dialect: str | None) -> str:
    """Whether `table.column` holds NULL in a row the reader can read: `SELECT 1 FROM <table> WHERE <column> IS NULL
    LIMIT 1`, the names quoted as stored, as a probe of a value quotes them (`probe_sql`)."""
    return (exp.select(exp.Literal.number(1))
            .from_(exp.Table(this=exp.to_identifier(table, quoted=True)))
            .where(exp.Is(this=exp.Column(this=exp.to_identifier(column, quoted=True)), expression=exp.Null()))
            .limit(1)
            .sql(dialect=dialect))


def holds_null(table: str, column: str, *, adapter, snapshot: Snapshot, grants: GrantSet, timeout_s: float,
               probes: Probes) -> bool | None:
    """Whether a row the builder can read holds NULL in `table.column` (a snapshot table and column), asked of the
    source through `adapter`, afresh, within `probes`, once per column: True for a row back, False for none.

    None where the source said nothing. Not asked, and not remembered: a column the builder's grants hide, mask or
    deny, a table they filter by row (or any table while a row filter names one the snapshot does not hold, which may
    be this one misspelled), a request Superset refused for good, and a plan whose probe time is spent. Asked and
    failed, which is remembered, so the plan does not ask again, and logged by the failure's kind alone: a source's
    words can quote a value."""
    if getattr(adapter, "terminal_refusal", None) is not None:
        return None  # Superset refused this request for good: every later call meets the same refusal
    if _withheld(table, (column,), snapshot=snapshot, grants=grants):
        return None
    key = (table, column)
    if key in probes.nulls:
        return probes.nulls[key]
    if probes.left_s() <= 0:
        logger.info("whether %s.%s holds NULL was not asked: the plan's %s s for probes was spent", table, column,
                    probes.budget_s)
        return None  # not kept: unasked is not unanswered
    physical = {binding.object_id: binding.source_object for binding in snapshot.source_bindings}
    sql = null_statement(physical.get(table, table), column, adapter.dialect)

    def ask(limit: float) -> bool | None:
        try:
            return _Afresh(adapter).execute_arrow(sql, timeout_s=limit).num_rows > 0
        except Exception as exc:  # noqa: BLE001 -- unanswered, the column's negated filters refuse nothing
            logger.info("whether %s.%s holds NULL could not be read (%s); its negated filters refuse nothing", table,
                        column, type(exc).__name__)
            return None

    probes.nulls[key] = probes.timed(ask, timeout_s)
    return probes.nulls[key]


def _withheld(table: str, columns: Sequence[str], *, snapshot: Snapshot, grants: GrantSet) -> bool:
    """Whether a probe of `columns` of `table` is not the builder's to send: a column the grants hide, mask or deny, a
    table they filter by row (a probe runs outside mnemiq's own row filter), or any table while a row filter names one
    the snapshot does not hold, which may be this one misspelled."""
    policy = build_access_policy(snapshot, grants)
    visible = visible_schema(snapshot, grants).get(table, ())
    return policy.row_filter_for(table) is not None or bool(policy.unknown_filter_targets) or any(
        column not in visible or policy.masks(table, column) or policy.denies(table, column) for column in columns)


def unique_statement(table: str, columns: Sequence[str], dialect: str | None) -> str:
    """Whether `columns` of `table` hold each value once together: `SELECT COUNT(*), COUNT(DISTINCT <columns>) FROM
    <table> WHERE <each column> IS NOT NULL`, the names quoted as stored, several columns counted as one row value. A
    NULL in a key meets no row in a join, so both counts are of the rows whose key holds no NULL; each is named
    (`keyed_rows`, `distinct_keys`), so the answer holds two columns of two names."""
    named = [exp.Column(this=exp.to_identifier(column, quoted=True)) for column in columns]
    key = named[0] if len(named) == 1 else exp.Tuple(expressions=named)
    return (exp.select(exp.alias_(exp.Count(this=exp.Star()), "keyed_rows", quoted=True),
                       exp.alias_(exp.Count(this=exp.Distinct(expressions=[key.copy()])), "distinct_keys", quoted=True))
            .from_(exp.Table(this=exp.to_identifier(table, quoted=True)))
            .where(exp.and_(*(exp.Not(this=exp.Is(this=column.copy(), expression=exp.Null())) for column in named)))
            .sql(dialect=dialect))


def unique_live(table: str, columns: Sequence[str], *, adapter, snapshot: Snapshot, grants: GrantSet,
                probes: Probes, timeout_s: float) -> bool | None:
    """Whether `columns` of `table` (a snapshot table and its columns, a joined table's key) hold each value once
    together among the rows the builder can read, asked of the source through `adapter`, afresh, within `probes`, once
    per key (its columns in any order): True where the rows holding a key are as many as their distinct keys, False
    where they are more.

    None where the source said nothing. Not asked, and not remembered: a key of no column, one `holds_null` would not
    ask of (a column the grants hide, mask or deny, a table they filter by row, any table while a row filter names one
    the snapshot does not hold), a request Superset refused for good, and a plan whose probe time is spent. Asked and
    failed, or answered with no two counts, which is remembered, so the plan does not ask again; a failure is logged by
    its kind alone: a source's words can quote a value."""
    named = tuple(sorted(set(columns)))
    if not named or getattr(adapter, "terminal_refusal", None) is not None \
            or _withheld(table, named, snapshot=snapshot, grants=grants):
        return None
    key = (table, named)
    if key in probes.unique:
        return probes.unique[key]
    shown = ", ".join(f"{table}.{column}" for column in named)
    if probes.left_s() <= 0:
        logger.info("whether %s holds each value once was not asked: the plan's %s s for probes was spent", shown,
                    probes.budget_s)
        return None  # not kept: unasked is not unanswered
    physical = {binding.object_id: binding.source_object for binding in snapshot.source_bindings}
    sql = unique_statement(physical.get(table, table), named, adapter.dialect)

    def ask(limit: float) -> bool | None:
        try:
            result = _Afresh(adapter).execute_arrow(sql, timeout_s=limit)
            counts = [result.column(0)[0].as_py(), result.column(1)[0].as_py()] \
                if result.num_rows == 1 and result.num_columns == 2 else None
        except Exception as exc:  # noqa: BLE001 -- unanswered, the key is not shown to hold each value once
            logger.info("whether %s holds each value once could not be read (%s); it is not joined on", shown,
                        type(exc).__name__)
            return None
        if counts is None or not all(type(count) is int for count in counts):
            logger.info("whether %s holds each value once was answered with no two counts; it is not joined on", shown)
            return None
        return counts[0] == counts[1]

    probes.unique[key] = probes.timed(ask, timeout_s)
    return probes.unique[key]


def _held(adapter, physical: dict[str, str], table: str, column: str, *, dialect: str | None,
          timeout_s: float) -> tuple[tuple[str, ...], bool] | None:
    """The first `HELD_NAMED` values `table.column` holds, as text, and whether it holds more, read through `adapter`;
    None where the read fails, which is logged by the failure's kind: a source's words can quote a value."""
    sql = held_values_sql(physical.get(table, table), column, HELD_NAMED + 1, dialect)
    try:
        result = adapter.execute_arrow(sql, timeout_s=timeout_s)
        values = result.column(0).to_pylist() if result.num_columns else []
    except Exception as exc:  # noqa: BLE001 -- the value stays absent; only the values it is told are lost
        logger.info("the values %s.%s holds could not be read (%s); the refusal does not name them", table, column,
                    type(exc).__name__)
        return None
    return tuple(shown(str(item)) for item in values[:HELD_NAMED]), len(values) > HELD_NAMED
