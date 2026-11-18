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
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
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

__all__ = ["HELD_NAMED", "Absent", "Probes", "absent_literals", "filter_statement"]

logger = logging.getLogger(__name__)

# The most values a refusal names of a column's; one more is read, to tell whether there are others.
HELD_NAMED = 20

# A chart filter's operators whose string operand is a value the column must hold for the filter to mean anything: `=`
# and `IN` select no row otherwise, and `!=` and `NOT IN` select every row, which is no filter at all.
_EQUALITY_OPERATORS = frozenset({"=", "!=", "IN", "NOT IN"})


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
    `clock` measures it, each probe's answer by `(table, column, text)` and each column's values by `(table, column)`.
    A call is given at most the time left, and none is made once it is spent. The budget is read between calls, so one
    call can overrun it by the transport's own margin, as a probe on the ask path can."""

    budget_s: float = PROBE_BUDGET_S
    clock: Callable[[], float] = time.monotonic
    spent_s: float = 0.0
    answers: dict[tuple[str, str, str], bool | None] = field(default_factory=dict)
    held: dict[tuple[str, str], tuple[tuple[str, ...], bool] | None] = field(default_factory=dict)

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
