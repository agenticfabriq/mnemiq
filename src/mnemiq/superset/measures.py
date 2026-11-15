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
           "display_format", "measure_statement", "measure_value", "run_decided", "superset_text_problem"]

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
    """Why a measure is refused. `repairable` is False where no rewrite of the measure would pass: the deciders'
    refusal is not one the model can act on (`Refusal.repairable`), such as Superset not answering the proof."""

    reason: str
    repairable: bool = True


_FORMATS = {
    MeasureKind.CURRENCY: ",.2f",
    MeasureKind.PERCENT: ",.1%",
    MeasureKind.COUNT: ",d",
    MeasureKind.NUMBER: ",.2f",
}


# What sqlglot puts between a window and its own function: `SUM(x) FILTER (WHERE y > 0) OVER ()` parses as a Window over
# a Filter over the Sum, and `FIRST_VALUE(x) IGNORE NULLS OVER ()` as a Window over an IgnoreNulls.
_WINDOW_WRAPPERS = (exp.Filter, exp.IgnoreNulls, exp.RespectNulls)


def _is_windowed(agg: exp.Expression) -> bool:
    """Whether `agg` is the function a window computes over, looking through the wrappers sqlglot places between them."""
    node = agg
    while isinstance(node.parent, _WINDOW_WRAPPERS) and node.parent.this is node:
        node = node.parent
    return isinstance(node.parent, exp.Window)


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
    if any(not _is_windowed(agg) for agg in tree.find_all(exp.AggFunc)):
        return Aggregation.AGGREGATES
    return Aggregation.ROW_LEVEL


def display_format(kind: MeasureKind, currency: str | None) -> tuple[str, dict | None]:
    """The d3 format Superset shows `kind` in, and its currency when it has one. Only a currency measure carries a
    currency: a percent never does."""
    fmt = _FORMATS[kind]
    if kind is MeasureKind.CURRENCY and currency:
        return fmt, {"symbol": currency, "symbolPosition": "prefix"}
    return fmt, None


def measure_statement(measure: Measure | tuple[Measure, ...], source: str, dimensions: tuple[str, ...] = (),
                      filters: tuple[str, ...] = ()) -> str:
    """The statement that computes `measure` over `source` (a quoted table name or an unparenthesized join,
    `a JOIN b ON ...`: the deciders refuse a parenthesized one as an unmodelled source), grouped by
    `dimensions` and narrowed by `filters`. A tuple of measures computes each, in order, after the dimensions, as a
    chart that plots several does."""
    measures = measure if isinstance(measure, tuple) else (measure,)
    select = ", ".join([*dimensions, *(f"{m.expression} AS {m.name}" for m in measures)])
    sql = f"SELECT {select} FROM {source}"
    if filters:
        sql += " WHERE " + " AND ".join(f"({f})" for f in filters)
    if dimensions:
        sql += " GROUP BY " + ", ".join(str(i) for i in range(1, len(dimensions) + 1))
    return sql


def _marker(*texts: str) -> str | None:
    return next((d for text in texts for d in JINJA_DELIMITERS if d in text), None)


# What rc2's MCP does to every chart name (`GenerateChartRequest.sanitize_chart_name`) and every SQL metric's label
# (`ColumnRef.sanitize_label`, both in `superset/mcp_service/chart/schemas.py`): `sanitize_user_input`
# (`superset/mcp_service/utils/sanitization.py`) strips the ends, removes zero-width, control and line-separator
# characters (`_remove_dangerous_unicode`), unescapes entities and cleans out HTML with nh3, which escapes `<`, `>`, `&`
# and a no-break space and restores only `&amp;` (`_strip_html_tags`), and refuses a script scheme or an event handler
# (`_check_dangerous_patterns`). A text it changes is saved as other text: a chart a build then cannot find again by its
# name, or a metric whose format a build set, and checks, under a label the chart does not show. One it refuses makes no
# chart. So neither holds any of these.
_RC2_CHANGED_CHARACTERS = frozenset("<>&\u00a0")
_RC2_REMOVED = re.compile(r"[\u0000-\u001f\u007f-\u009f\u200b-\u200d\u2028\u2029\ufeff]")
_RC2_REFUSED = (re.compile(r"\b(javascript|vbscript|data):", re.IGNORECASE), re.compile(r"on\w+\s*=", re.IGNORECASE))


def superset_text_problem(text: str, what: str) -> str | None:
    """Why rc2's MCP would save `text`, a chart's title or a measure's label (`what` names which, in the reason), as
    other text, or refuse it; None when it keeps it as written. The reason is mnemiq's and does not repeat the text."""
    if text != text.strip():
        return f"{what} starts or ends with a space, which Superset removes; write it without one"
    if any(character in _RC2_CHANGED_CHARACTERS for character in text):
        return f"{what} holds <, >, & or a no-break space, which Superset saves as other text; write it without them"
    if _RC2_REMOVED.search(text):
        return (f"{what} holds a control, zero-width or line-separator character, which Superset removes; write it "
                "with plain spaces")
    if any(pattern.search(text) for pattern in _RC2_REFUSED):
        return (f"Superset refuses {what} that reads as a script scheme (javascript:, vbscript:, data:) or an event "
                "handler (a word starting with on, then =); reword it")
    return None


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
    if (problem := superset_text_problem(measure.label, "the measure's label")) is not None:
        return MeasureRefused(problem)
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
        return MeasureRefused(f"{verdict.code.value}: {verdict.message}", repairable=verdict.repairable)
    return verdict


def measure_value(approved: Approved, adapter, timeout_s: float) -> float | None:
    """The one number `approved` returns when run as whoever `adapter` acts for, or None when it returns no row or a
    null. It is returned, never stored. A failed run raises `ExecutionError`."""
    table = run(adapter, approved.target_sql, timeout_s=timeout_s).table
    if table.num_rows == 0 or table.num_columns == 0:
        return None
    value = table.column(0)[0].as_py()
    return None if value is None else float(value)
