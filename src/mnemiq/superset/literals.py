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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

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

__all__ = ["HELD_NAMED", "Absent", "absent_literals", "filter_statement"]

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
    more. `held` is empty where the source did not say."""

    column: str
    literal: str
    held: tuple[str, ...]
    more: bool


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


def absent_literals(sql: str, *, adapter, snapshot: Snapshot, grants: GrantSet, timeout_s: float) -> list[Absent]:
    """Each value `sql` filters a column on that the source, asked through `adapter`, says no row holds, in the order
    the statement holds them, once each. Grounded as the decider grounds a statement (`ground_values`, with what
    `grants` leave visible, their masks and row filters, the snapshot's column types and no value index), and asked of
    the source as the ask path asks it (`confirm_by_source`, `source_holds`). A statement that does not parse, or a
    grounding that cannot be read, refuses nothing."""
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
    physical = {binding.object_id: binding.source_object for binding in snapshot.source_bindings}
    ask = source_holds(adapter, dialect, physical, timeout_s)
    asked: set[tuple[str, str, str]] = set()

    def holds(table: str, column: str, text: str) -> bool | None:
        asked.add((table, column, text))
        return ask(table, column, text)

    confirmed = confirm_by_source(grounding, holds)
    probeable = {(*value.placed, value.text) for value in grounding.values
                 if value.outcome == UNCHECKED and value.why == "not_indexed" and value.placed is not None}
    if left := len(probeable - asked):
        logger.info("%d filter values were left unprobed past the limit of %d per statement; they stay unchecked and "
                    "refuse nothing", left, PROBE_LIMIT)
    found: list[Absent] = []
    seen: set[tuple[str, str, str]] = set()
    held: dict[tuple[str, str], tuple[tuple[str, ...], bool]] = {}
    for value in confirmed.values:
        if value.outcome != ABSENT or value.why != "probed" or value.placed is None:
            continue
        key = (*value.placed, value.text)
        if key in seen:
            continue
        seen.add(key)
        if value.placed not in held:
            held[value.placed] = _held(adapter, physical, *value.placed, dialect=dialect, timeout_s=timeout_s)
        values, more = held[value.placed]
        found.append(Absent(f"{value.placed[0]}.{value.placed[1]}", value.text, values, more))
    return found


def _held(adapter, physical: dict[str, str], table: str, column: str, *, dialect: str | None,
          timeout_s: float) -> tuple[tuple[str, ...], bool]:
    """The first `HELD_NAMED` values `table.column` holds, as text, and whether it holds more, read through `adapter`;
    nothing where the read fails, which is logged by the failure's kind: a source's words can quote a value."""
    sql = held_values_sql(physical.get(table, table), column, HELD_NAMED + 1, dialect)
    try:
        result = adapter.execute_arrow(sql, timeout_s=timeout_s)
        values = result.column(0).to_pylist() if result.num_columns else []
    except Exception as exc:  # noqa: BLE001 -- the value stays absent; only the values it is told are lost
        logger.info("the values %s.%s holds could not be read (%s); the refusal does not name them", table, column,
                    type(exc).__name__)
        return (), False
    return tuple(shown(str(item)) for item in values[:HELD_NAMED]), len(values) > HELD_NAMED
