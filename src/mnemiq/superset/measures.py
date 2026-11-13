"""A measure: one named aggregate expression, with the kind that decides how its number is shown.

A measure is a definition, never a value. Its statement is decided under the asker's grants with the deciders every
ask-path statement meets (authorization, and the fan-out guard where it is on), and its value is computed through
Superset as whoever asks, each time, and handed back without being kept.

Whether an expression aggregates is read as Superset reads it (an aggregate function not directly under a window),
and fails closed: a function mnemiq does not model, or SQL that does not parse, is "cannot tell", and the measure is
refused rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, StrEnum

import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot
from mnemiq.execute.runner import run
from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.history import Decided, decide_statement, run_decided
from mnemiq.sql.verdict import Approved, Refusal

__all__ = ["Aggregation", "Decided", "Measure", "MeasureKind", "MeasureRefused", "aggregates", "check_measure",
           "display_format", "measure_statement", "measure_value", "run_decided"]

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
MAX_LABEL = 120


class MeasureKind(StrEnum):
    CURRENCY = "currency"
    PERCENT = "percent"
    COUNT = "count"
    NUMBER = "number"


@dataclass(frozen=True)
class Measure:
    name: str
    label: str
    expression: str
    kind: MeasureKind
    dataset_id: int
    description: str
    asked_by: str
    currency: str | None = None


class Aggregation(Enum):
    AGGREGATES = "aggregates"
    ROW_LEVEL = "row_level"
    CANNOT_TELL = "cannot_tell"


@dataclass(frozen=True)
class MeasureRefused:
    reason: str


_FORMATS = {
    MeasureKind.CURRENCY: ",.2f",
    MeasureKind.PERCENT: ",.1%",
    MeasureKind.COUNT: ",d",
    MeasureKind.NUMBER: ",.2f",
}


def aggregates(expression: str) -> Aggregation:
    """Whether `expression` aggregates: some aggregate function not directly under a window. `SUM(x) OVER ()` is a
    row-level value, `SUM(SUM(x)) OVER ()` aggregates. A function mnemiq does not model, SQL that does not parse and
    anything but one bare expression are `CANNOT_TELL`."""
    try:
        tree = sqlglot.parse_one(f"SELECT {expression}", read="postgres")
    except sqlglot.errors.SqlglotError:
        return Aggregation.CANNOT_TELL
    if not isinstance(tree, exp.Select) or len(tree.expressions) != 1 \
            or {key for key, value in tree.args.items() if value} - {"expressions"}:
        return Aggregation.CANNOT_TELL
    if any(tree.find_all(exp.Anonymous)):
        return Aggregation.CANNOT_TELL
    if any(not isinstance(agg.parent, exp.Window) for agg in tree.find_all(exp.AggFunc)):
        return Aggregation.AGGREGATES
    return Aggregation.ROW_LEVEL


def display_format(kind: MeasureKind, currency: str | None) -> tuple[str, dict | None]:
    """The d3 format Superset shows `kind` in, and its currency when it has one. Only a currency measure carries a
    currency: a percent never does."""
    fmt = _FORMATS[kind]
    if kind is MeasureKind.CURRENCY and currency:
        return fmt, {"symbol": currency, "symbolPosition": "prefix"}
    return fmt, None


def measure_statement(measure: Measure, source: str, dimensions: tuple[str, ...] = (),
                      filters: tuple[str, ...] = ()) -> str:
    """The statement that computes `measure` over `source` (a quoted table name or a parenthesized join), grouped by
    `dimensions` and narrowed by `filters`."""
    select = ", ".join([*dimensions, f"{measure.expression} AS {measure.name}"])
    sql = f"SELECT {select} FROM {source}"
    if filters:
        sql += " WHERE " + " AND ".join(f"({f})" for f in filters)
    if dimensions:
        sql += " GROUP BY " + ", ".join(str(i) for i in range(1, len(dimensions) + 1))
    return sql


def _marker(*texts: str) -> str | None:
    return next((d for text in texts for d in JINJA_DELIMITERS if d in text), None)


def check_measure(measure: Measure, *, source: str, dimension_names: tuple[str, ...] = (), adapter,
                  snapshot: Snapshot, grants: GrantSet, guard_fanout: bool | None) -> Approved | MeasureRefused:
    """The measure's statement, approved, or the reason it is refused. Nothing is sent to Superset except the
    deciders' own probe."""
    if not _NAME.match(measure.name):
        return MeasureRefused(f"the measure name {measure.name!r} must be lower case letters, digits and _, starting "
                              "with a letter, at most 63 characters")
    if not measure.label.strip() or len(measure.label) > MAX_LABEL:
        return MeasureRefused(f"the measure's label must be 1 to {MAX_LABEL} characters")
    if (found := _marker(measure.label, measure.expression, measure.description)) is not None:
        return MeasureRefused(f"the measure holds {found}, which Superset reads as a template marker; write it "
                              f"without {found}")
    if measure.currency is not None and not _CURRENCY.match(measure.currency):
        return MeasureRefused(f"the currency {measure.currency!r} must be an ISO 4217 code such as USD, never a "
                              "symbol")
    if measure.label.casefold() in {name.casefold() for name in dimension_names}:
        return MeasureRefused(f"the label {measure.label!r} is also the name of a dimension; name the measure "
                              "differently")
    shape = aggregates(measure.expression)
    if shape is Aggregation.CANNOT_TELL:
        return MeasureRefused("cannot tell whether the function aggregates: the expression holds a function mnemiq "
                              "does not model, or is not one SQL expression")
    if shape is Aggregation.ROW_LEVEL:
        return MeasureRefused("the expression does not aggregate: it gives one value per row, not one per group")
    verdict = decide_statement(measure_statement(measure, source), adapter=adapter, snapshot=snapshot, grants=grants,
                               guard_fanout=guard_fanout)
    if isinstance(verdict, Refusal):
        return MeasureRefused(f"{verdict.code.value}: {verdict.message}")
    return verdict


def measure_value(approved: Approved, adapter, timeout_s: float) -> float | None:
    """The one number `approved` returns when run as whoever `adapter` acts for, or None when it returns no row or a
    null. It is returned, never stored. A failed run raises `ExecutionError`."""
    table = run(adapter, approved.target_sql, timeout_s=timeout_s).table
    if table.num_rows == 0 or table.num_columns == 0:
        return None
    value = table.column(0)[0].as_py()
    return None if value is None else float(value)
