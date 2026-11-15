"""The value check: whether a saved chart shows the numbers mnemiq computes for it, in a format that fits each measure.

What to compare is read from the chart's current `form_data`, never from mnemiq's plan, so a chart changed by hand is
checked as it now is. Both sides are read as the builder: the chart's rows through MCP's `get_chart_data`, and mnemiq's
own through the request's adapter, its statement decided as every ask-path statement is. The values ride back in the
`ChartCheck` for the response and are kept nowhere, logged nowhere.

Whatever mnemiq cannot reproduce faithfully is `NOT_CHECKED`, with mnemiq's reason, and never `CHECKED`. Three facts of
Superset 7.0.0rc2 decide most of it:

- `get_chart_data` (`superset/mcp_service/chart/tool/get_chart_data.py`) runs a chart's saved query context when it has
  one, and otherwise rebuilds one from `form_data` (`build_query_dicts_from_form_data` in
  `superset/mcp_service/chart/chart_helpers.py`) with no post-processing. Every chart MCP's `generate_chart` saves has
  none until mnemiq's extension claims it and saves a query context built as the chart's plugin builds its query,
  without post-processing. So a chart whose shown numbers come after the query (a rolling window, a time comparison, a
  contribution, a forecast, quartiles, bins) is not checked, whichever query its data tool runs.
- Superset's rows are read as the chart's groups only on evidence in the answer itself. Each result used carries every
  grouping column the chart's settings name, and none carries the `GROUPING()` markers of a grouping-sets query.
  Under a time grain every x-axis value sits on the grain: the rebuild puts a string x-axis among the query's
  columns as it is, and `time_grain_sqla` reaches a query only through a `BASE_AXIS` column
  (`apply_time_grain_to_base_axis` in `superset/common/utils/time_grain_utils.py`; `adhoc_column_to_sqla` in
  `superset/connectors/sqla/models.py`), as the saved query context sends it. Rows that are not the chart's groups
  (the frontend's own query context pivots a grouped time series into a column per group; the rebuild drops a pivot's
  rows and columns, a waterfall's x-axis and the grain) are not compared, and the chart is not checked, saying so.
- Superset's text never enters a reason: a refusal can name a table or a user the caller may not see. It goes to the
  log, as the client logs it.

A virtual dataset is never checked with mnemiq's own SQL: a row rule attached to the virtual dataset alone applies to
queries of that dataset, not to SQL over its tables. A saved metric on one is checked through `query_dataset`
(`superset/mcp_service/dataset/tool/query_dataset.py`), which takes saved metric names only, and a measure written on
the chart is not checked.

What the chart's query computes is read from its `form_data` by `chart_form.py`; this module reads Superset, runs
mnemiq's own statement and compares.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import StrEnum

from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot
from mnemiq.execute.runner import DEFAULT_TIMEOUT_S, ExecutionError
from mnemiq.sql.guard import MAX_ROWS
from mnemiq.sql.verdict import Refusal
from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.chart_form import (
    _COMPARISONS,
    _GRAINS,
    _LISTS,
    _NO_VALUE,
    _PATTERNS,
    DatasetFacts,
    Expectation,
    NotCheckable,
    _metric_items,
    _quoted,
    _row_limit,
    expectation_from_form_data,
)
from mnemiq.superset.charts import Filter
from mnemiq.superset.client import SupersetToolError, SupersetUnavailable
from mnemiq.superset.display_format import CANNOT_TELL, Displayed, displayed, fits, metric_label, shows_currency
from mnemiq.superset.measures import Measure, MeasureKind, measure_statement, run_decided
from mnemiq.superset.saved_metrics import SavedMetric

__all__ = ["ChartCheck", "ChartStatus", "DatasetFacts", "Expectation", "NotCheckable", "check_chart",
           "expectation_from_form_data", "read_dataset_facts"]

logger = logging.getLogger(__name__)


class ChartStatus(StrEnum):
    CHECKED = "checked"
    MISMATCH = "mismatch"
    EMPTY = "empty"
    CUT = "cut"
    FORMAT_WRONG = "format_wrong"
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class ChartCheck:
    """One chart's check. `chart_values` and `mnemiq_values` are the rows each side gave, each a tuple of the grouping
    values then the metric values, for the response only."""

    chart_id: int
    status: ChartStatus
    reason: str
    chart_values: tuple | None
    mnemiq_values: tuple | None


# `get_chart_info` returns `form_data` only when asked for it (`DEFAULT_GET_CHART_INFO_COLUMNS` in
# `superset/mcp_service/chart/schemas.py` leaves it out).
_CHART_FIELDS = ["id", "slice_name", "viz_type", "form_data", "changed_on"]
_DATASET_FIELDS = ["id", "table_name", "schema", "database_id", "is_virtual", "sql", "metrics"]

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}(:?\d{2})?)?)?")

_UNREADABLE = "mnemiq could not read the chart's settings: "
_NO_GRAIN = ("Superset's data tool returned this chart's rows without its time grain, as it does for a chart with no "
             "saved query context, so the numbers the chart shows by its time grain were not compared")
# The suffix of each `GROUPING()` marker column a grouping-sets query adds (`GROUPING_MARKER_SUFFIX` in
# `superset/common/grouping_sets.py`), one per grouping column, marking the level each row groups by.
_GROUPING_MARKER = "__superset_grouping"
# Where Postgres's `DATE_TRUNC` for each grain mnemiq reproduces puts a time (`_GRAINS`): which of its fields are zero
# or first. `DATE_TRUNC('week', ...)` gives a Monday.
_ON_GRAIN = {
    "PT1S": lambda t: t.microsecond == 0,
    "PT1M": lambda t: (t.second, t.microsecond) == (0, 0),
    "PT1H": lambda t: (t.minute, t.second, t.microsecond) == (0, 0, 0),
    "P1D": lambda t: t.time() == time(0),
    "P1W": lambda t: t.time() == time(0) and t.weekday() == 0,
    "P1M": lambda t: t.time() == time(0) and t.day == 1,
    "P3M": lambda t: t.time() == time(0) and t.day == 1 and t.month in (1, 4, 7, 10),
    "P1Y": lambda t: t.time() == time(0) and (t.month, t.day) == (1, 1),
}


# ---------------------------------------------------------------- reading Superset


def _failed(answer: dict) -> bool:
    """An MCP error model answered as the tool's result (`MCPBaseError`, `DatasetError`): it carries `error_type`."""
    return bool(answer.get("error_type") or answer.get("error"))


def _log_error(what: str, answer: dict) -> None:
    logger.warning("the value check: Superset answered %s with an error (%s): %s", what, answer.get("error_type"),
                   answer.get("error") or answer.get("message"))


def _read_chart(client, chart_id: int) -> tuple[str, dict] | NotCheckable:
    """The chart's type and `form_data`, the type as the chart row holds it (`get_chart_data` reads it the same way)."""
    try:
        answer = client.call("get_chart_info", {"identifier": chart_id, "select_columns": _CHART_FIELDS})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return NotCheckable(f"{_UNREADABLE}{failure}")
    if _failed(answer):
        _log_error("get_chart_info", answer)
        return NotCheckable(f"{_UNREADABLE}Superset answered with an error, which is in the log")
    form = answer.get("form_data")
    if answer.get("_response_truncated") or (isinstance(form, dict) and form.get("_truncated")):
        # The size guard (`truncate_oversized_response` in `superset/mcp_service/utils/response_size_utils.py`) clips
        # strings and summarizes dicts anywhere in the answer, `form_data` among them.
        return NotCheckable(f"{_UNREADABLE}Superset cut them to fit its response size")
    if not isinstance(form, dict):
        # `redact_chart_data_model_fields` (`superset/mcp_service/privacy.py`) withholds them from a caller without
        # the data-model privilege.
        return NotCheckable(f"{_UNREADABLE}Superset does not show them to this builder")
    viz_type = answer.get("viz_type") or form.get("viz_type") or ""
    return viz_type, {**form, "viz_type": viz_type}


def _dataset_of(form: dict, dataset_id: int) -> int | NotCheckable:
    """The dataset the chart reads now: the one its `form_data` names once saved in Explore, as
    `resolve_form_data_datasource` reads it, else the one mnemiq built it on (MCP's `generate_chart` saves no
    `datasource` in the chart's params, and `get_chart_info` returns no dataset id)."""
    combined = form.get("datasource")
    if isinstance(combined, str) and "__" in combined:
        head, _, kind = combined.partition("__")
        if kind != "table" or not head.isdecimal():
            return NotCheckable("the chart reads a datasource that is not a dataset mnemiq checks")
        return int(head)
    return dataset_id


def read_dataset_facts(client, dataset_id: int) -> DatasetFacts | NotCheckable:
    try:
        answer = client.call("get_dataset_info", {"identifier": dataset_id, "select_columns": _DATASET_FIELDS})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return NotCheckable(f"mnemiq could not read the chart's dataset: {failure}")
    if _failed(answer):
        _log_error("get_dataset_info", answer)
        return NotCheckable("mnemiq could not read the chart's dataset: Superset answered with an error, which is in "
                            "the log")
    if answer.get("_response_truncated"):
        return NotCheckable("mnemiq could not read the chart's dataset whole: Superset cut it to fit its response size")
    metrics = [m for m in answer.get("metrics") or [] if isinstance(m, dict) and isinstance(m.get("metric_name"), str)]
    sql = answer.get("sql")
    return DatasetFacts(
        dataset_id=dataset_id,
        # Fails closed: a dataset that does not say it is physical, or that carries SQL, is read as virtual.
        is_virtual=answer.get("is_virtual") is not False or bool(sql),
        table=answer.get("table_name"), sql=sql,
        saved_metric_names=frozenset(m["metric_name"] for m in metrics),
        database_id=answer.get("database_id"), schema=answer.get("schema"),
        saved_metric_sql=tuple((m["metric_name"], m.get("expression") or "") for m in metrics))


@dataclass(frozen=True)
class _Rows:
    """One side's rows: each group's metric values by its grouping values, in the order the side gave them."""

    values: dict[tuple, list]
    largest: int = 0  # the most rows any one of the side's results held
    cap: int | None = None  # the row cap put on the side's one read, when it reached it: it may hold only some groups


def _canonical(value):
    """A value as both sides can be compared on: a timestamp as a naive UTC `datetime` whichever way it was spelled (an
    ISO string from `get_chart_data`, a `datetime` from mnemiq's own result), a decimal as a float."""
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return tuple(_canonical(v) for v in value)
    return value


def _on_the_grain(value, grain: str) -> bool:
    """Whether a time Superset answered sits where Postgres's `DATE_TRUNC` for `grain` puts one, read as spelled: its
    wall clock, whatever offset it carries, since the database truncates in its own time zone. A null is a null
    group's key. Anything that is not a time is not on a grain, and no time is on a grain mnemiq does not know."""
    on_grain = _ON_GRAIN.get(grain)
    if on_grain is None:
        return False
    if value is None:
        return True
    if isinstance(value, str):
        if not _ISO_DATE.fullmatch(value):
            return False
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return False
    if not isinstance(value, date):
        return False
    moment = value if isinstance(value, datetime) else datetime(value.year, value.month, value.day)
    return on_grain(moment)


def _line_up(sets: list[tuple[list, list]], labels: list[str], dimensions: tuple[str, ...],
             grain: str | None) -> _Rows | NotCheckable:
    """Superset's rows by group, each result that holds every grouping column lending the metrics it holds: a mixed
    chart answers one result per axis. Every metric must be held by one. Read as the chart's groups only on evidence in
    the answer: no result carries grouping-sets markers, and under a time `grain` every x-axis value the lent rows hold
    sits on it."""
    values: dict[tuple, list] = {}
    held: set[int] = set()
    largest = 0
    for columns, _ in sets:
        if any(isinstance(name, str) and name.endswith(_GROUPING_MARKER) for name in columns):
            return NotCheckable("Superset's rows for this chart come from grouping sets, each row marked with the "
                                "level it groups by, so they cannot be lined up with mnemiq's")
    for columns, rows in sets:
        present = [i for i, label in enumerate(labels) if label in columns]
        if not present or not all(name in columns for name in dimensions):
            continue
        held.update(present)
        largest = max(largest, len(rows))
        seen: set[tuple] = set()
        for row in rows:
            if grain is not None and not _on_the_grain(row.get(dimensions[0]), grain):
                return NotCheckable(_NO_GRAIN)
            key = tuple(_canonical(row.get(name)) for name in dimensions)
            if key in seen:
                return NotCheckable("Superset's rows for this chart repeat a group, so they cannot be lined up with "
                                    "mnemiq's")
            seen.add(key)
            slot = values.setdefault(key, [None] * len(labels))
            for i in present:
                slot[i] = _canonical(row.get(labels[i]))
    if held != set(range(len(labels))):
        return NotCheckable("Superset's rows for this chart do not carry the columns its settings name, so they cannot "
                            "be lined up with mnemiq's")
    return _Rows(values, largest)


def _chart_rows(client, chart_id: int, expectation: Expectation) -> _Rows | ChartCheck:
    try:
        answer = client.call("get_chart_data", {"identifier": chart_id})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, f"mnemiq could not read the chart's data: {failure}")
    if answer.get("_response_truncated"):
        # `truncate_query_result` (`superset/mcp_service/utils/response_size_utils.py`) keeps the rows that fit.
        return _outcome(chart_id, ChartStatus.CUT, "Superset cut the chart's rows to fit its response size, so they "
                                                   "are not all there to compare")
    if _failed(answer):
        if answer.get("error_type") == "NoData":  # rc2 answers a query with no rows this way
            return _Rows({})
        _log_error("get_chart_data", answer)
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, "mnemiq could not read the chart's data: Superset answered "
                                                           "with an error, which is in the log")
    results = answer.get("query_results")
    if isinstance(results, list) and results:
        sets = [(list(r.get("columns") or []), list(r.get("data") or [])) for r in results if isinstance(r, dict)]
    else:
        sets = [([c.get("name") for c in answer.get("columns") or [] if isinstance(c, dict)],
                 list(answer.get("data") or []))]
    lined = _line_up(sets, [label for label, _ in expectation.metrics], expectation.dimensions,
                     expectation.time_grain)
    if isinstance(lined, NotCheckable):
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, lined.reason)
    return lined


# ---------------------------------------------------------------- mnemiq's own numbers


def _literal(value) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return exp.Literal.number(value).sql(dialect="postgres")
    return exp.Literal.string(str(value)).sql(dialect="postgres")


def _filter_sql(item: Filter) -> str:
    column = _quoted(item.column)
    if item.op in _NO_VALUE:
        return f"{column} {_NO_VALUE[item.op]}"
    if item.op in _LISTS:
        listed = f"{column} IN ({', '.join(_literal(v) for v in item.value)})"
        return f"NOT {listed}" if item.op == "NOT IN" else listed
    operator = _COMPARISONS.get(item.op) or _PATTERNS[item.op]
    return f"{column} {operator} {_literal(item.value)}"


def _statement(expectation: Expectation, dataset: DatasetFacts, *, grained: bool) -> str:
    """mnemiq's statement for the chart's query, through `measure_statement`: each metric aliased by position, the
    x-axis grouped by the grain when `grained`, as Postgres's grain expression writes it."""
    saved = dict(dataset.saved_metric_sql)
    measures = tuple(Measure(name=f"m{i}", label=label,
                             expression=saved[label] if label in dataset.saved_metric_names else expression,
                             kind=MeasureKind.NUMBER, dataset_id=dataset.dataset_id, description="", asked_by="")
                     for i, (label, expression) in enumerate(expectation.metrics))
    columns = [_quoted(name) for name in expectation.dimensions]
    if grained and expectation.time_grain:
        columns[0] = f"DATE_TRUNC('{_GRAINS[expectation.time_grain]}', {columns[0]}) AS {columns[0]}"
    return measure_statement(measures, _quoted(dataset.table), tuple(columns),
                             tuple(_filter_sql(item) for item in expectation.filters))


def _raise_terminal(adapter) -> None:
    """A refusal of the caller met while the statement was decided or run stays the same for every later call, so it
    ends the check, as it ends a question (`SupersetSqlAdapter.terminal_refusal`)."""
    if (refused := getattr(adapter, "terminal_refusal", None)) is not None:
        raise refused


def _own_rows(sql: str, expectation: Expectation, *, chart_id: int, adapter, snapshot: Snapshot, grants: GrantSet,
              timeout_s: float, guard_fanout: bool | None, before_call: Callable[[], None] | None = None,
              ) -> _Rows | NotCheckable:
    if before_call is not None:
        before_call()  # once for the decided statement: its catalogue reads, its probe and its run
    try:
        decided = run_decided(sql, adapter=adapter, snapshot=snapshot, grants=grants, timeout_s=timeout_s,
                              guard_fanout=guard_fanout)
    except ExecutionError as failure:
        _raise_terminal(adapter)
        logger.warning("the value check: mnemiq's statement for chart %s did not run: %s %s", chart_id,
                       failure.message, failure.source_detail or "")
        return NotCheckable("mnemiq's own statement for this chart did not run; why is in the log")
    _raise_terminal(adapter)
    if isinstance(decided, Refusal):
        logger.warning("the value check: mnemiq refused its statement for chart %s (%s, %r): %s", chart_id,
                       decided.code.value, decided.subject, decided.source_detail or "")
        return NotCheckable(f"mnemiq refused its own statement for this chart ({decided.code.value}): "
                            f"{decided.message}")
    table, width = decided.table, len(expectation.dimensions)
    if table.num_rows and table.num_columns != width + len(expectation.metrics):
        return NotCheckable("mnemiq's own statement for this chart returned other columns than it asked for")
    values: dict[tuple, list] = {}
    for row in zip(*(table.column(i).to_pylist() for i in range(table.num_columns))) if table.num_rows else ():
        key = tuple(_canonical(v) for v in row[:width])
        if key in values:
            # Two groups the database told apart read as one once made comparable (`2026-01-01` and
            # `2026-01-01 00:00:00` in a text column): which one the chart's row is cannot be told.
            return NotCheckable("mnemiq's own rows for this chart repeat a group, so they cannot be lined up with the "
                                "chart's")
        values[key] = [_canonical(v) for v in row[width:]]
    cap = decided.approved.row_cap
    return _Rows(values, table.num_rows, cap=cap if cap is not None and table.num_rows >= cap else None)


def _dataset_rows(client, expectation: Expectation, dataset: DatasetFacts) -> _Rows | NotCheckable:
    """A virtual dataset's saved metrics through `query_dataset`, which applies the dataset's own row rules."""
    request = {"dataset_id": dataset.dataset_id, "metrics": [label for label, _ in expectation.metrics],
               "columns": list(expectation.dimensions),
               "filters": [{"col": f.column, "op": f.op, "val": f.value} for f in expectation.filters],
               "row_limit": MAX_ROWS}
    try:
        answer = client.call("query_dataset", request)
    except (SupersetUnavailable, SupersetToolError) as failure:
        return NotCheckable(f"mnemiq could not read the dataset's numbers: {failure}")
    if _failed(answer):
        _log_error("query_dataset", answer)
        return NotCheckable("mnemiq could not read the dataset's numbers: Superset answered with an error, which is "
                            "in the log")
    if answer.get("_response_truncated"):
        return NotCheckable("Superset cut the dataset's numbers to fit its response size, so they are not all there "
                            "to compare")
    rows = list(answer.get("data") or [])
    values = {tuple(_canonical(row.get(name)) for name in expectation.dimensions):
              [_canonical(row.get(label)) for label, _ in expectation.metrics] for row in rows}
    if len(values) != len(rows):
        return NotCheckable("the dataset's rows repeat a group, so they cannot be lined up with the chart's")
    return _Rows(values, len(rows), cap=MAX_ROWS if len(rows) >= MAX_ROWS else None)


# ---------------------------------------------------------------- comparing


def _same(a, b) -> bool:
    """Two values agree: integers exactly, other numbers within `rel_tol=1e-12`, anything else by equality. Both sides
    are computed by the same database, so the tolerance absorbs no more than floats summed in another order; at
    `1e-9` a cent apart on a value above about 1e7 would agree, although `,.2f` shows it."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12)
    return a == b


def _shown(value):
    return value.isoformat() if isinstance(value, datetime) else value


def _values(rows: dict[tuple, list], order: list[tuple]) -> tuple:
    return tuple((*(_shown(v) for v in key), *(_shown(v) for v in rows[key])) for key in order)


def _outcome(chart_id: int, status: ChartStatus, reason: str, chart: tuple | None = None,
             mine: tuple | None = None) -> ChartCheck:
    return ChartCheck(chart_id, status, reason, chart, mine)


# What mnemiq's side of the comparison is, for the reasons: its own decided SQL over the physical table, or Superset's
# own query of a virtual dataset's saved metrics.
_OWN_SQL = "mnemiq's own SQL"
_QUERY_DATASET = "Superset's own query of the dataset's saved metrics (query_dataset)"


def _compare(chart_id: int, chart: _Rows, mine: _Rows, row_limit: int | None, *, source: str) -> ChartCheck:
    shown = list(chart.values)
    order = [key for key in shown if key in mine.values] + [key for key in mine.values if key not in chart.values]
    both = (_values(chart.values, shown), _values(mine.values, order))

    def outcome(status: ChartStatus, reason: str) -> ChartCheck:
        values = both if status in (ChartStatus.CHECKED, ChartStatus.MISMATCH, ChartStatus.EMPTY) else (None, None)
        return _outcome(chart_id, status, reason, *values)

    if any(key in mine.values and not all(map(_same, chart.values[key], mine.values[key])) for key in shown):
        return outcome(ChartStatus.MISMATCH, f"the chart shows a value {source} does not compute")
    if mine.cap is not None:
        return outcome(ChartStatus.NOT_CHECKED, f"the chart has more groups than {source} returns in one read "
                                                f"({mine.cap:,}), so not all of them were compared")
    if any(key not in mine.values for key in shown):
        return outcome(ChartStatus.MISMATCH, f"the chart shows a group {source} does not find")
    if len(mine.values) > len(shown):
        if row_limit is not None and chart.largest >= row_limit:
            return outcome(ChartStatus.NOT_CHECKED, f"the chart's row limit ({row_limit:,}) keeps only some of its "
                                                    "groups, so not all of them were compared")
        return outcome(ChartStatus.MISMATCH, f"{source} finds a group the chart does not show")
    if not shown:
        return outcome(ChartStatus.EMPTY, f"the chart shows no rows, and {source} finds none")
    return outcome(ChartStatus.CHECKED, f"the chart's numbers match {source}")


def _all_found(chart: _Rows, mine: _Rows) -> bool:
    """Positive evidence that the chart's rows are `mine`: there are some, and each is there with equal values."""
    return bool(chart.values) and all(key in mine.values and all(map(_same, values, mine.values[key]))
                                      for key, values in chart.values.items())


# ---------------------------------------------------------------- the format


def _describe(shown: Displayed) -> str:
    fmt = f"in the format {shown.d3format!r}" if shown.d3format else "in Superset's default number format"
    symbol = (shown.currency or {}).get("symbol")
    if not symbol:
        return f"{fmt}, with no currency"
    if not shows_currency(symbol):
        return f"{fmt}, with the currency symbol {symbol!r}, which Superset's renderer does not show"
    return f"{fmt}, with the currency {symbol}"


def _format_problem(viz_type: str, form: dict, kinds: dict[str, MeasureKind], saved: dict, *,
                    every_kind_known: bool = False) -> tuple[str | None, str | None]:
    """Of the metrics mnemiq knows the kind of: why one is shown in a format that does not fit it, and why one's format
    could not be judged, each the first found, or None. A format the renderer decides from what mnemiq cannot see
    (`CANNOT_TELL`), and a saved metric whose own record the caller did not pass, cannot be judged. With
    `every_kind_known`, a metric whose label `kinds` does not hold cannot be judged either: the caller gave the kind of
    every metric it made the chart with, so the chart shows one under a label mnemiq never sent."""
    unjudged = None
    for item in _metric_items(viz_type, form):
        label = metric_label(item)
        kind = kinds.get(label) if label else None
        if kind is None:
            if every_kind_known:
                unjudged = unjudged or (f"the format the chart shows {label!r} in could not be judged: no measure "
                                        "mnemiq made the chart with is labelled so" if label else
                                        "the format of a metric the chart shows could not be judged: mnemiq cannot "
                                        "read its label")
            continue
        record: SavedMetric | None = None
        if isinstance(item, str):
            record = saved.get(item)
            if record is None:
                unjudged = unjudged or (f"the format the chart shows {label!r} in could not be judged: mnemiq has no "
                                        "record of that saved metric's own format")
                continue
        shown = displayed(viz_type, form, item, record)
        if shown is CANNOT_TELL:
            unjudged = unjudged or (f"the format the chart shows {label!r} in could not be judged: Superset decides it "
                                    "from what mnemiq does not read")
        elif not fits(kind, shown):
            return f"the chart shows {label!r} {_describe(shown)}, which does not fit a {kind.value} measure", unjudged
    return None, unjudged


# ---------------------------------------------------------------- the check


def _numbers(client, adapter, *, chart_id: int, form: dict, dataset_id: int, snapshot: Snapshot, grants: GrantSet,
             guard_fanout: bool | None, timeout_s: float, before_call: Callable[[], None] | None = None) -> ChartCheck:
    """The chart's numbers against mnemiq's own: any status but `FORMAT_WRONG`."""
    def not_checked(reason: str) -> ChartCheck:
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, reason)

    dataset_id = _dataset_of(form, dataset_id)
    if isinstance(dataset_id, NotCheckable):
        return not_checked(dataset_id.reason)
    dataset = read_dataset_facts(client, dataset_id)
    if isinstance(dataset, NotCheckable):
        return not_checked(dataset.reason)
    expectation = expectation_from_form_data(form, dataset=dataset)
    if isinstance(expectation, NotCheckable):
        return not_checked(expectation.reason)
    sql = ""
    if dataset.is_virtual:
        written = [label for label, _ in expectation.metrics if label not in dataset.saved_metric_names]
        if written:
            return not_checked(f"the measure {written[0]!r} is written on the chart over a virtual dataset: mnemiq "
                               "never checks a virtual dataset with its own SQL, which a row rule attached to that "
                               "dataset alone would not narrow, and Superset's query_dataset takes saved metrics only")
        if expectation.time_grain:
            return not_checked("the chart groups a virtual dataset by a time grain, which Superset's query_dataset "
                               "cannot ask for")
    else:
        if getattr(adapter, "dialect", None) != "postgres":
            return not_checked("mnemiq's own statement for a chart is written for Postgres, Superset mode's one "
                               "dialect, and this adapter speaks another")
        if dataset.database_id != getattr(adapter, "database_id", dataset.database_id) \
                or (dataset.schema or None) != getattr(adapter, "schema", dataset.schema):
            return not_checked("the chart's dataset is on another database or schema than the one mnemiq reads")
        if not dataset.table:
            return not_checked("the chart's dataset names no table")
        saved = dict(dataset.saved_metric_sql)
        if empty := [label for label, _ in expectation.metrics if label in saved and not saved[label].strip()]:
            return not_checked(f"Superset gave no expression for the saved metric {empty[0]!r}")
        sql = _statement(expectation, dataset, grained=True)
        # Checked on the statement itself, so a saved metric's own expression is covered too: `execute_sql` renders a
        # statement as a template, and the adapter would refuse it later in words about the statement, not the chart.
        if (marker := next((m for m in JINJA_DELIMITERS if m in sql), None)) is not None:
            return not_checked(f"the chart's settings hold {marker}, which Superset reads as a template marker, so "
                               "mnemiq cannot compute the same and sends nothing")
    chart = _chart_rows(client, chart_id, expectation)
    if isinstance(chart, ChartCheck):
        return chart
    if dataset.is_virtual:
        mine, source = _dataset_rows(client, expectation, dataset), _QUERY_DATASET
    else:
        mine, source = _own_rows(sql, expectation, chart_id=chart_id, adapter=adapter, snapshot=snapshot, grants=grants,
                                 timeout_s=timeout_s, guard_fanout=guard_fanout, before_call=before_call), _OWN_SQL
    if isinstance(mine, NotCheckable):
        return not_checked(mine.reason)
    found = _compare(chart_id, chart, mine, _row_limit(form), source=source)
    if found.status is ChartStatus.MISMATCH and expectation.time_grain and source is _OWN_SQL:
        plain = _own_rows(_statement(expectation, dataset, grained=False), expectation, chart_id=chart_id,
                          adapter=adapter, snapshot=snapshot, grants=grants, timeout_s=timeout_s,
                          guard_fanout=guard_fanout, before_call=before_call)
        # The chart's rows sit on the grain (`_line_up`), yet rows rebuilt without it can too, under a row limit that
        # keeps only first-of-month days. Only on positive evidence: every one of the chart's rows found among mnemiq's
        # ungrouped ones, with equal values. An ungrouped read at its row cap, or an empty chart, proves nothing, and
        # the grouped finding stands.
        if not isinstance(plain, NotCheckable) and _all_found(chart, plain):
            return not_checked(_NO_GRAIN)
    return found


class _BeforeEachCall:
    """The adapter's own client, `before` run before each of its calls: how a caller renews a lease, or stops, before
    each call the check makes. The calls are the client's, carrying its caller's token."""

    def __init__(self, client, before: Callable[[], None]) -> None:
        self._client, self._before = client, before

    def call(self, tool: str, request: dict, **more):
        self._before()
        return self._client.call(tool, request, **more)


def check_chart(adapter, *, chart_id: int, dataset_id: int, kinds: dict[str, MeasureKind],
                saved: dict[str, SavedMetric], snapshot: Snapshot, grants: GrantSet, guard_fanout: bool | None,
                timeout_s: float = DEFAULT_TIMEOUT_S, before_call: Callable[[], None] | None = None,
                every_kind_known: bool = False) -> ChartCheck:
    """Check chart `chart_id` as the caller `adapter` acts for, the builder: Superset is read through `adapter.client`
    and no other, and `adapter` runs mnemiq's own statement under `grants` and `snapshot`. `dataset_id` is the dataset
    mnemiq built the chart on, which a chart saved since in Explore overrides with its own. `kinds` gives the kind of
    each metric mnemiq knows, by the label Superset names its values by; `saved` the dataset's saved metrics mnemiq
    read, by name.

    A wrong format is reported whatever the numbers, unless they are a mismatch or the chart is empty. A chart is
    checked only when the format of every metric mnemiq knows the kind of was judged: numbers that match beside a
    format mnemiq could not judge are not checked, their values kept. `every_kind_known` says `kinds` holds every
    metric the caller made the chart with (the builder's check), so a metric of any other label is one whose format
    was not judged: Superset saved it under other text than mnemiq sent, or someone added it since. A refusal of the caller (`SupersetRefusal`) is
    raised: it stays the same for every later call the request makes.

    `before_call`, when given, runs before each Superset call the check makes itself, and once before mnemiq's own
    decided statement (whose catalogue reads, probe and run the adapter makes); whatever it raises ends the check."""
    client = adapter.client if before_call is None else _BeforeEachCall(adapter.client, before_call)
    read = _read_chart(client, chart_id)
    if isinstance(read, NotCheckable):
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, read.reason)
    viz_type, form = read
    problem, unjudged = _format_problem(viz_type, form, kinds, saved, every_kind_known=every_kind_known)
    found = _numbers(client, adapter, chart_id=chart_id, form=form, dataset_id=dataset_id, snapshot=snapshot,
                     before_call=before_call,
                     grants=grants, guard_fanout=guard_fanout, timeout_s=timeout_s)
    if problem is None and unjudged is not None and found.status is ChartStatus.CHECKED:
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, f"{found.reason}, but {unjudged}", found.chart_values,
                        found.mnemiq_values)
    if problem is None or found.status in (ChartStatus.MISMATCH, ChartStatus.EMPTY):
        return found
    reason = problem if found.status is ChartStatus.CHECKED else f"{problem}. Its numbers were not compared: " \
                                                                  f"{found.reason}"
    return _outcome(chart_id, ChartStatus.FORMAT_WRONG, reason, found.chart_values, found.mnemiq_values)
