"""The value check: whether a saved chart shows the numbers mnemiq computes for it, in a format that fits each measure.

What to compare is read from the chart's current `form_data`, never from mnemiq's plan, so a chart changed by hand is
checked as it now is. Both sides are read as the builder: the chart's rows through MCP's `get_chart_data`, and mnemiq's
own through the request's adapter, its statement decided as every ask-path statement is. The values ride back in the
`ChartCheck` for the response and are kept nowhere, logged nowhere.

Whatever mnemiq cannot reproduce faithfully is `NOT_CHECKED`, with mnemiq's reason, and never `CHECKED`. Three facts of
Superset 7.0.0rc2 decide most of it:

- `get_chart_data` (`superset/mcp_service/chart/tool/get_chart_data.py`) runs a chart's saved query context when it has
  one, and otherwise rebuilds one from `form_data` (`build_query_dicts_from_form_data` in
  `superset/mcp_service/chart/chart_helpers.py`) with no post-processing. A chart mnemiq's extension creates
  (`create_chart`) is saved with one whenever the extension's `params.query_context` builds one for its family, built
  as the chart's plugin builds its query, without post-processing; one MCP's `generate_chart` saves has none. So a
  chart whose shown numbers come after the query (a rolling window, a time comparison, a contribution, a forecast,
  quartiles, bins) is not checked, whichever query its data tool runs.
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

A virtual dataset is never checked with mnemiq's own SQL, except one this build made (`check_chart`'s `made`) whose
SQL is still the SQL the build made it with and on which no row rule applies to the builder: a row rule attached to
the virtual dataset alone applies to queries of that dataset, not to SQL over its tables. Whether one applies is the
extension's `row_rules`, which asks Superset as the builder and answers only whether; when one does, or mnemiq cannot
tell, mnemiq's statement is not run. The build sent that SQL in Superset's spelling (`names.for_superset`), which is
the text Superset must still hold, and the dataset is read with the SQL the build decided, as Superset reads a virtual
one, `FROM (<its SQL>) AS "virtual_table"`, and on Oracle `FROM (<its SQL>) virtual_table` (`_virtual_source`). Any
other virtual dataset's saved metric is checked through `query_dataset`
(`superset/mcp_service/dataset/tool/query_dataset.py`), which takes saved metric names only, and a measure written on
the chart is not checked.

What the chart's query computes is read from its `form_data` by `chart_form.py`; this module reads Superset, runs
mnemiq's own statement and compares.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
import re
from dataclasses import dataclass, field
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
    _X_AXIS,
    _LISTS,
    _NO_VALUE,
    _PATTERNS,
    DatasetFacts,
    Expectation,
    NotCheckable,
    Ranking,
    _metric_items,
    _quoted,
    _row_limit,
    expectation_from_form_data,
    grain_expressions,
    ranking_of,
)
from mnemiq.superset.charts import Filter
from mnemiq.superset.client import SupersetToolError, SupersetUnavailable
from mnemiq.superset.extension import row_rule_applies
from mnemiq.superset.display_format import CANNOT_TELL, Displayed, displayed, fits, metric_label, shows_currency
from mnemiq.superset.measures import Measure, MeasureKind, measure_statement, run_decided
from mnemiq.superset.names import RefusedStatement, UnreadableStatement, for_superset
from mnemiq.superset.payload import SupersetAccessRefused, SupersetQueryFailed
from mnemiq.superset.saved_metrics import SavedMetric
from mnemiq.superset.settings import superset_dialect

__all__ = ["ChartCheck", "ChartRead", "ChartStatus", "DatasetFacts", "Expectation", "NotCheckable", "check_chart",
           "dataset_of", "expectation_from_form_data", "percent_problem", "read_chart", "read_dataset_facts"]

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


@dataclass(frozen=True)
class _Numbers(ChartCheck):
    """What `_numbers` found, with the metric labels the chart's rows hold values for (in their order) and those rows,
    for the checks that read the values themselves. Handed on only as a plain `ChartCheck`."""

    labels: tuple[str, ...] = ()
    rows: dict[tuple, list] = field(default_factory=dict)


def _plain(found: ChartCheck) -> ChartCheck:
    return ChartCheck(found.chart_id, found.status, found.reason, found.chart_values, found.mnemiq_values)


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
# Where Superset's expression for each grain mnemiq reproduces puts a time (`chart_form._GRAINS`), on PostgreSQL: which
# of its fields are zero or first. `DATE_TRUNC('week', ...)` gives a Monday.
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
# On Oracle (`chart_form._ORACLE_GRAINS`), where PostgreSQL's put one, but for a week: `TRUNC(..., 'WW')` gives the
# weekday 1 January of the time's year fell on, a whole number of weeks after that day, so the last week of a year of
# 365 days starts on 31 December.
_ORACLE_ON_GRAIN = _ON_GRAIN | {
    "P1W": lambda t: t.time() == time(0) and (t.date() - date(t.year, 1, 1)).days % 7 == 0,
}


# ---------------------------------------------------------------- reading Superset


def _failed(answer: dict) -> bool:
    """An MCP error model answered as the tool's result (`MCPBaseError`, `DatasetError`): it carries `error_type`."""
    return bool(answer.get("error_type") or answer.get("error"))


def _log_error(what: str, answer: dict) -> None:
    logger.warning("the value check: Superset answered %s with an error (%s): %s", what, answer.get("error_type"),
                   answer.get("error") or answer.get("message"))


@dataclass(frozen=True)
class ChartRead:
    """A chart as `get_chart_info` gives it to the caller: its title, its type as the chart row holds it
    (`get_chart_data` reads it the same way), its `form_data` with that type, and its `changed_on`. `form_data` is None
    when it cannot be read whole, and `unreadable` then says why, in mnemiq's words."""

    chart_id: int
    title: str | None
    viz_type: str
    form_data: dict | None
    changed_on: str | None
    unreadable: str | None = None
    # Superset holds no chart of this id that the caller may read: deleted, or in the trash (rc2's soft-delete filter
    # hides a trashed chart from every read), or never theirs to read.
    missing: bool = False


def read_chart(client, chart_id: int) -> ChartRead:
    """The chart, read as the caller: never raised over a failed or cut read, which is answered as unreadable."""
    try:
        answer = client.call("get_chart_info", {"identifier": chart_id, "select_columns": _CHART_FIELDS})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return ChartRead(chart_id, None, "", None, None, f"{_UNREADABLE}{failure}")
    if _failed(answer):
        _log_error("get_chart_info", answer)
        missing = str(answer.get("error_type") or "").replace("_", "").lower() == "notfound"
        return ChartRead(chart_id, None, "", None, None,
                         f"{_UNREADABLE}Superset answered with an error, which is in the log", missing=missing)
    title = answer.get("slice_name") if isinstance(answer.get("slice_name"), str) else None
    changed_on = answer.get("changed_on") if isinstance(answer.get("changed_on"), str) else None
    form = answer.get("form_data")
    viz_type = answer.get("viz_type") or (form.get("viz_type") if isinstance(form, dict) else None) or ""
    if answer.get("_response_truncated") or (isinstance(form, dict) and form.get("_truncated")):
        # The size guard (`truncate_oversized_response` in `superset/mcp_service/utils/response_size_utils.py`) clips
        # strings and summarizes dicts anywhere in the answer, `form_data` among them.
        return ChartRead(chart_id, title, viz_type, None, changed_on,
                         f"{_UNREADABLE}Superset cut them to fit its response size")
    if not isinstance(form, dict):
        # `redact_chart_data_model_fields` (`superset/mcp_service/privacy.py`) withholds them from a caller without
        # the data-model privilege.
        return ChartRead(chart_id, title, viz_type, None, changed_on,
                         f"{_UNREADABLE}Superset does not show them to this builder")
    return ChartRead(chart_id, title, viz_type, {**form, "viz_type": viz_type}, changed_on)


def _read_chart(client, chart_id: int) -> tuple[str, dict] | NotCheckable:
    """The chart's type and `form_data`, or why they cannot be read."""
    read = read_chart(client, chart_id)
    if read.form_data is None:
        return NotCheckable(read.unreadable or _UNREADABLE)
    return read.viz_type, read.form_data


def dataset_of(form: dict, dataset_id: int) -> int | NotCheckable:
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


def read_dttm_columns(client, dataset_id: int) -> dict[str, bool | None] | NotCheckable:
    """Each of the dataset's columns and whether Superset takes it for a time column (`TableColumnInfo.is_dttm`, which
    its time-series charts read to treat an x-axis as time): True, False, or None where Superset does not say. Or why
    mnemiq could not read them."""
    try:
        answer = client.call("get_dataset_info", {"identifier": dataset_id, "select_columns": ["columns"]})
    except (SupersetUnavailable, SupersetToolError) as failure:
        return NotCheckable(f"Superset did not answer for its dataset's columns: {failure}")
    if _failed(answer):
        _log_error("get_dataset_info", answer)
        return NotCheckable("Superset answered with an error, which is in the log")
    if answer.get("_response_truncated"):
        return NotCheckable("Superset cut its dataset's columns to fit its response size")
    columns = answer.get("columns")
    if not isinstance(columns, list):
        return NotCheckable("Superset answered no columns for its dataset")
    return {c["column_name"]: c.get("is_dttm") if isinstance(c.get("is_dttm"), bool) else None
            for c in columns if isinstance(c, dict) and isinstance(c.get("column_name"), str)}


@dataclass(frozen=True)
class _Rows:
    """One side's rows: each group's metric values by its grouping values, in the order the side gave them."""

    values: dict[tuple, list]
    largest: int = 0  # the most rows any one of the side's results held
    cap: int | None = None  # the row cap put on the side's one read, when it reached it: it may hold only some groups
    # When a read that reached its cap was made in a chart's ranking order (`_ranked_read`): that ranking, and the
    # ranking value of the last group it read, after which every group it did not read ranks.
    ranked: Ranking | None = None
    through: object = None
    # When a read that reached its cap was made in a chart's ranking order but its values cannot be ranked: why not.
    unranked: str | None = None


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
    """Whether a time Superset answered sits where Superset's expression for `grain` puts one on Superset mode's
    database (`_ON_GRAIN`, `_ORACLE_ON_GRAIN`), read as spelled: its wall clock, whatever offset it carries, since the
    database truncates in its own time zone. A null is a null group's key. Anything that is not a time is not on a
    grain, and no time is on a grain mnemiq does not know."""
    on_grain = (_ORACLE_ON_GRAIN if superset_dialect() == "oracle" else _ON_GRAIN).get(grain)
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
    """A filter's value as SQL in Superset mode's dialect, as Superset's query holds it: rc2 compiles a chart's query
    with its values in place (`compile_sqla_query` in `superset/models/core.py`), so a text is a text literal and a
    number a number on either database. Oracle before 23 has no TRUE or FALSE, and SQLAlchemy's Oracle dialect renders
    one as 1 or 0, as mnemiq does there. On Oracle a date or a time is its DATE or TIMESTAMP literal, which reads the
    same whatever the session's date format, and a time with a zone raises ValueError: mnemiq writes none."""
    oracle = superset_dialect() == "oracle"
    if isinstance(value, bool):
        if oracle:
            return "1" if value else "0"
        return "TRUE" if value else "FALSE"
    if oracle and isinstance(value, datetime):
        if value.tzinfo is not None:
            raise ValueError("mnemiq writes no time with a zone as an Oracle literal")
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if oracle and isinstance(value, date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, (int, float)):
        return exp.Literal.number(value).sql(dialect=superset_dialect())
    return exp.Literal.string(str(value)).sql(dialect=superset_dialect())


# Oracle has no ILIKE: SQLAlchemy's Oracle dialect, which compiles Superset's query, writes its `ilike` and `not_ilike`
# as `lower(<column>) LIKE lower(<pattern>)` and `lower(<column>) NOT LIKE lower(<pattern>)`, LOWER on both sides.
_ORACLE_ILIKE = {"ILIKE": "LIKE", "NOT ILIKE": "NOT LIKE"}
# The filters Superset writes as `IS` a true or false (`handle_boolean_filter` in `superset/db_engine_specs/base.py`),
# and the value SQLAlchemy's Oracle dialect renders for each there.
_TRUE_OR_FALSE = {"IS TRUE": 1, "IS FALSE": 0}


def _filter_sql(item: Filter) -> str:
    column = _quoted(item.column)
    if item.op in _NO_VALUE:
        return f"{column} {_NO_VALUE[item.op]}"
    if item.op in _LISTS:
        listed = f"{column} IN ({', '.join(_literal(v) for v in item.value)})"
        return f"NOT {listed}" if item.op == "NOT IN" else listed
    if item.op in _ORACLE_ILIKE and superset_dialect() == "oracle":
        return f"LOWER({column}) {_ORACLE_ILIKE[item.op]} LOWER({_literal(item.value)})"
    operator = _COMPARISONS.get(item.op) or _PATTERNS[item.op]
    return f"{column} {operator} {_literal(item.value)}"


def _unwritten(expectation: Expectation) -> str | None:
    """Why mnemiq does not write its statement for the chart in Superset mode's dialect, or None. On Oracle, a filter
    `IS TRUE` or `IS FALSE`, which Superset writes there as `IS 1` or `IS 0`: a test Oracle refuses, and mnemiq writes
    no `IS TRUE` there."""
    if superset_dialect() != "oracle":
        return None
    for item in expectation.filters:
        if item.op in _TRUE_OR_FALSE:
            return (f"the chart's filter on {item.column!r} is {item.op}, which Superset writes on Oracle as "
                    f"IS {_TRUE_OR_FALSE[item.op]}, a test Oracle refuses, so mnemiq does not write it")
    return None


def _virtual_source(sql: str) -> str:
    """A dataset this build made, read as Superset reads a virtual dataset: its SQL as a subquery aliased
    `virtual_table`, quoted after `AS` on PostgreSQL, as SQLAlchemy's PostgreSQL dialect writes it. Oracle refuses `AS`
    before a table's alias, and SQLAlchemy's Oracle dialect writes a lower-case name unquoted, which Oracle reads as
    VIRTUAL_TABLE: so there the alias follows bare."""
    alias = "virtual_table" if superset_dialect() == "oracle" else 'AS "virtual_table"'
    return f"({sql}) {alias}"


def _statement(expectation: Expectation, dataset: DatasetFacts, *, grained: bool, source: str | None = None,
               ranking: Ranking | None = None, limit: int | None = None) -> str:
    """mnemiq's statement for the chart's query, in Superset mode's dialect, through `measure_statement`: each metric
    aliased by position, the x-axis grouped by the grain when `grained`, as Superset's grain expression for its database
    writes it (`grain_expressions`), under the column's own name. It reads `source` when given (a made dataset's SQL as
    Superset reads a virtual one, `_virtual_source`), else the dataset's table. Under the chart's `ranking` its groups
    come in that order, NULL last as `_ranking_problem` ranks it, so the deciders' row cap, or `limit`, keeps the groups
    that decide the chart's cut: a LIMIT on PostgreSQL, Oracle's `FETCH FIRST n ROWS ONLY`."""
    saved = dict(dataset.saved_metric_sql)
    measures = tuple(Measure(name=f"m{i}", label=label,
                             expression=saved[label] if label in dataset.saved_metric_names else expression,
                             kind=MeasureKind.NUMBER, dataset_id=dataset.dataset_id, description="", asked_by="")
                     for i, (label, expression) in enumerate(expectation.metrics))
    columns: list[str | tuple[str, str]] = [_quoted(name) for name in expectation.dimensions]
    if grained and expectation.time_grain:
        axis = _quoted(expectation.dimensions[0])
        columns[0] = (grain_expressions()[expectation.time_grain].format(col=axis), axis)
    sql = measure_statement(measures, source or _quoted(dataset.table), tuple(columns),
                            tuple(_filter_sql(item) for item in expectation.filters))
    if ranking is not None:
        sql += f" ORDER BY m{ranking.metric_index} {'DESC' if ranking.descending else 'ASC'} NULLS LAST"
        if limit is not None:
            oracle = superset_dialect() == "oracle"
            sql += f" FETCH FIRST {int(limit)} ROWS ONLY" if oracle else f" LIMIT {int(limit)}"
    return sql


def _raise_terminal(adapter) -> None:
    """A refusal of the caller met while the statement was decided or run stays the same for every later call, so it
    ends the check, as it ends a question (`SupersetSqlAdapter.terminal_refusal`)."""
    if (refused := getattr(adapter, "terminal_refusal", None)) is not None:
        raise refused


def _rank_key(value, descending: bool) -> tuple:
    """Where `value` stands in a ranking: a NULL after every value, as `_ranking_problem` ranks one."""
    if value is None:
        return (1, 0)
    return (0, -value if descending else value)


def _rankable(value) -> bool:
    return value is None or (not isinstance(value, bool) and isinstance(value, (int, float, Decimal)))


def _first_read(ranking: Ranking | None) -> int | None:
    """How many groups a read in `ranking`'s order asks for first: twice the chart's cut, which sees past it unless many
    groups tie there, and keeps the answer far below Superset's MCP response size, which a full read's rows often pass.
    None when that is no fewer than one read's cap, or there is no ranking."""
    if ranking is None or ranking.limit < 1 or 2 * ranking.limit >= MAX_ROWS:
        return None
    return 2 * ranking.limit


def _ranked_read(rows: _Rows, ranking: Ranking | None, *, nulls_last: bool, source: str) -> _Rows:
    """`rows`, read in `ranking`'s order, marked as such when the read stopped at its cap, with the ranking value of the
    last group it read: every group it did not read ranks after that one. Only where the order is known: each value a
    number or NULL, and the NULLs last (`nulls_last`), or none read when the engine placed them; else why not."""
    if rows.cap is None or ranking is None:
        return rows
    values = [row[ranking.metric_index] if ranking.metric_index < len(row) else None for row in rows.values.values()]
    if not values:
        return rows
    if not all(map(_rankable, values)):
        return _Rows(rows.values, rows.largest, cap=rows.cap,
                     unranked="the chart's ranking measure holds values mnemiq cannot rank")
    if not nulls_last and None in values:
        return _Rows(rows.values, rows.largest, cap=rows.cap,
                     unranked=f"{source} places groups with no value where its database does, and the {rows.cap:,} "
                              "it read in the chart's ranking order hold some, so where its row limit cuts is not read")
    through = max(values, key=lambda value: _rank_key(value, ranking.descending))
    return _Rows(rows.values, rows.largest, cap=rows.cap, ranked=ranking, through=through)


def _own_rows(sql: str, expectation: Expectation, *, chart_id: int, adapter, snapshot: Snapshot, grants: GrantSet,
              timeout_s: float, guard_fanout: bool | None, before_call: Callable[[], None] | None = None,
              ranking: Ranking | None = None, limit: int | None = None) -> _Rows | NotCheckable:
    """mnemiq's rows from its own statement `sql`, which, under the chart's `ranking`, orders its groups by it, NULL
    last, and asks for `limit` of them when given (`_statement`): a read of that many stopped at it."""
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
    cap = limit if limit is not None else decided.approved.row_cap
    return _ranked_read(_Rows(values, table.num_rows, cap=cap if cap is not None and table.num_rows >= cap else None),
                        ranking, nulls_last=True, source=_OWN_SQL)


def _dataset_rows(client, expectation: Expectation, dataset: DatasetFacts, ranking: Ranking | None = None,
                  limit: int | None = None) -> _Rows | NotCheckable:
    """A virtual dataset's saved metrics through `query_dataset`, which applies the dataset's own row rules. Under the
    chart's `ranking` it is asked for its rows in that order (`order_by`, `order_desc`, which rc2's `build_query_dict`
    turns into the query's ORDER BY before its `row_limit`); it takes no null order, so the engine places the NULLs. It
    asks for `limit` rows when given, else one read's cap."""
    row_limit = limit if limit is not None else MAX_ROWS
    request = {"dataset_id": dataset.dataset_id, "metrics": [label for label, _ in expectation.metrics],
               "columns": list(expectation.dimensions),
               "filters": [{"col": f.column, "op": f.op, "val": f.value} for f in expectation.filters],
               "row_limit": row_limit}
    if ranking is not None and ranking.metric_index < len(expectation.metrics):
        request |= {"order_by": [expectation.metrics[ranking.metric_index][0]], "order_desc": ranking.descending}
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
    return _ranked_read(_Rows(values, len(rows), cap=row_limit if len(rows) >= row_limit else None),
                        ranking if "order_by" in request else None, nulls_last=False, source=_QUERY_DATASET)


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


def _ranking_problem(chart: _Rows, mine: _Rows, ranking: Ranking, label: str) -> str | None | NotCheckable:
    """Whether the chart's rows are the first `ranking.limit` of mnemiq's, ranked as the chart's query ranks them, a
    group whose ranking value is NULL counted after every group with a value, groups tied at the cut standing in for
    each other: None when they are, else why not. A ranking value that is not a number cannot be ranked."""
    index, limit = ranking.metric_index, ranking.limit
    values = {key: row[index] if index < len(row) else None for key, row in mine.values.items()}
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float, Decimal))) for v in values.values()):
        return NotCheckable("the chart's ranking measure holds values mnemiq cannot rank")

    def rank(value) -> tuple:
        return _rank_key(value, ranking.descending)

    ordered = sorted(values.values(), key=rank)
    nth = ordered[min(limit, len(ordered)) - 1]

    def place(value) -> int:
        """-1 before the cut, 0 tied with it (equal as `_same` compares, so float noise is a tie), 1 after it."""
        if _same(value, nth):
            return 0
        return -1 if rank(value) < rank(nth) else 1

    shown = set(chart.values)
    order = "largest" if ranking.descending else "smallest"
    if any(values[key] is None for key in shown) and any(values[key] is not None for key in values if key not in shown):
        return (f"the chart's row limit ({limit:,}) keeps groups with no value for {label!r} and leaves out groups that "
                f"have one, so they are not the {limit:,} {order}")
    if any(place(values[key]) < 0 for key in values if key not in shown) or any(place(values[key]) > 0
                                                                                for key in shown):
        return f"the chart's row limit ({limit:,}) keeps groups that are not the {limit:,} {order} by {label!r}"
    return None


def _ranking_label(labels: tuple[str, ...], ranking: Ranking) -> str:
    return labels[ranking.metric_index] if ranking.metric_index < len(labels) else f"number {ranking.metric_index + 1}"


def _unsettled(cap: int, source: str) -> str:
    return (f"the {cap:,} groups {source} read in the chart's ranking order do not settle where its row limit cuts: "
            "groups tied at the cut, or kept by its HAVING, run on past them, so not all of them were compared")


def _cut_inside(rows: _Rows, ranking: Ranking | None) -> bool:
    """Whether a read that stopped at its cap still decides `ranking`'s cut: it was made in that ranking's order
    (`_ranked_read`), holds at least `ranking.limit` groups, and the last group it read ranks strictly after the
    `ranking.limit`-th, a tie as `_ranking_problem` counts one, so every group it did not read ranks after the cut too.
    Groups tied at the cut that run on past the read do not."""
    if ranking is None or rows.ranked != ranking or ranking.limit < 1 or len(rows.values) < ranking.limit:
        return False
    index = ranking.metric_index
    values = sorted((row[index] if index < len(row) else None for row in rows.values.values()),
                    key=lambda value: _rank_key(value, ranking.descending))
    nth = values[ranking.limit - 1]
    return not _same(rows.through, nth) and _rank_key(rows.through, ranking.descending) > _rank_key(nth,
                                                                                                  ranking.descending)


def _compare(chart_id: int, chart: _Rows, mine: _Rows, row_limit: int | None, *, source: str,
             labels: tuple[str, ...] = (), ranking: Ranking | None = None,
             time_axis: Callable[[], bool | NotCheckable] | None = None) -> ChartCheck:
    shown = list(chart.values)
    order = [key for key in shown if key in mine.values] + [key for key in mine.values if key not in chart.values]
    both = (_values(chart.values, shown), _values(mine.values, order))

    def outcome(status: ChartStatus, reason: str) -> ChartCheck:
        values = both if status in (ChartStatus.CHECKED, ChartStatus.MISMATCH, ChartStatus.EMPTY) else (None, None)
        return _outcome(chart_id, status, reason, *values)

    if any(key in mine.values and not all(map(_same, chart.values[key], mine.values[key])) for key in shown):
        return outcome(ChartStatus.MISMATCH, f"the chart shows a value {source} does not compute")
    if mine.cap is not None and not _cut_inside(mine, ranking):
        if mine.unranked is not None:
            return outcome(ChartStatus.NOT_CHECKED, mine.unranked)
        if mine.ranked is not None:
            return outcome(ChartStatus.NOT_CHECKED, _unsettled(mine.cap, source))
        return outcome(ChartStatus.NOT_CHECKED, f"the chart has more groups than {source} returns in one read "
                                                f"({mine.cap:,}), so not all of them were compared")
    if any(key not in mine.values for key in shown):
        if mine.cap is not None and ranking is not None:
            # A read whose cut falls inside it: a group it does not hold ranks after the cut.
            return outcome(ChartStatus.MISMATCH, f"the chart's row limit ({ranking.limit:,}) keeps groups that are "
                                                 f"not the {ranking.limit:,} "
                                                 f"{'largest' if ranking.descending else 'smallest'} by "
                                                 f"{_ranking_label(labels, ranking)!r}")
        return outcome(ChartStatus.MISMATCH, f"the chart shows a group {source} does not find")
    checked = f"the chart's numbers match {source}"
    # A read that stopped at its cap holds only some groups, so as many groups on both sides proves nothing: a chart
    # showing more than its cut beside a short read of as many is read against the cut all the same.
    if len(mine.values) > len(shown) or mine.cap is not None:
        if row_limit is None or chart.largest < row_limit:
            return outcome(ChartStatus.MISMATCH, f"{source} finds a group the chart does not show")
        cut = (f"the chart's row limit ({row_limit:,}) keeps only some of its groups, so not all of them were "
               "compared")
        # A time axis's query ranks by its measure before its LIMIT and the axis sorts by time afterwards: the points
        # kept are the largest, not the latest, whatever a ranking would say of them. Read only on a cut.
        on_time = time_axis() if time_axis is not None else False
        if isinstance(on_time, NotCheckable):
            return outcome(ChartStatus.NOT_CHECKED,
                           f"{cut}: mnemiq could not read whether its x-axis is a time axis ({on_time.reason})")
        if on_time:
            return outcome(ChartStatus.NOT_CHECKED, f"{cut}: its x-axis is a time axis, where a row limit keeps the "
                                                    "points its query ranks first, not the latest")
        if ranking is None or ranking.limit != row_limit:
            return outcome(ChartStatus.NOT_CHECKED, cut)
        label = _ranking_label(labels, ranking)
        problem = _ranking_problem(chart, mine, ranking, label)
        if isinstance(problem, NotCheckable):
            return outcome(ChartStatus.NOT_CHECKED, f"{cut}: {problem.reason}")
        if problem is not None:
            return outcome(ChartStatus.MISMATCH, problem)
        if len(shown) > ranking.limit:
            # Groups tied at the cut stand in for each other (`_ranking_problem`), but do not add to it: rows beyond
            # the row limit are not the query the chart's settings describe, so no ranking is claimed for them.
            return outcome(ChartStatus.NOT_CHECKED, f"the chart shows {len(shown):,} groups, more than its row limit "
                                                    f"({ranking.limit:,}) keeps, so they are not the cut its settings "
                                                    "describe, and were not compared as one")
        checked = (f"{checked}, and its row limit keeps the {row_limit:,} "
                   f"{'largest' if ranking.descending else 'smallest'} groups by {label!r}")
    if not shown:
        return outcome(ChartStatus.EMPTY, f"the chart shows no rows, and {source} finds none")
    # A measure that is NULL on every row it shows, on both sides (the values matched above), is no number the chart
    # shows: the chart reads "No data" for it, which a check must not report as checked.
    # Over the groups the chart shows: mnemiq's other groups, those a ranked row limit left out, are on no screen.
    sides = (list(chart.values.values()), [mine.values[key] for key in shown if key in mine.values])
    width = max((len(values) for rows in sides for values in rows), default=0)
    empty = [i for i in range(width) if all(values[i] is None for rows in sides for values in rows if i < len(values))
             and any(i < len(values) for rows in sides for values in rows)]
    if empty:
        if len(empty) == width:
            return outcome(ChartStatus.EMPTY, f"the chart's measures are NULL on every row it shows, and {source} "
                                              "computes the same")
        first = labels[empty[0]] if empty[0] < len(labels) else f"number {empty[0] + 1}"
        return outcome(ChartStatus.EMPTY, f"the chart's measure {first!r} is NULL on every row it shows, and {source} "
                                          "computes the same")
    return outcome(ChartStatus.CHECKED, checked)


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


# A fraction of exactly 1 can come out a float's width above it; that is 100%, not past it.
_SCALE_NOISE = 1e-9


def percent_problem(kinds: dict[str, MeasureKind], labels: tuple[str, ...], rows: dict[tuple, list]) -> str | None:
    """Why a `percent` measure's number is shown wrongly: the first percent label holding a value above 1 in size. A
    percent is a fraction, which the percent format shows times 100, so a value above 1 shows above 100%: a 0 to 100
    number the format scaled again (a win rate of 55 shown as 5,500%), or a true rate above 100%."""
    for i, label in enumerate(labels):
        if kinds.get(label) is not MeasureKind.PERCENT:
            continue
        for values in rows.values():
            value = values[i] if i < len(values) else None
            if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) \
                    and math.isfinite(value) and abs(value) > 1 + _SCALE_NOISE:
                return (f"the chart shows {label!r} above 100% ({value}), as a measure on a 0 to 100 scale would "
                        "under a percent format")
    return None


# Why a dataset this build made is not checked with mnemiq's own SQL after all.
_SQL_CHANGED = ("the chart's dataset is one this build made, but its SQL is no longer the SQL the build made it with, "
                "so mnemiq does not check it with its own SQL")
_RULE_APPLIES = ("a row rule applies to you on the chart's dataset, and mnemiq's own SQL reads the tables beneath it, "
                 "which that rule does not narrow, so mnemiq does not compare")
_SQL_UNSHOWN = ("Superset did not show the SQL of the chart's dataset, which this build made, so mnemiq does not check "
                "it with its own SQL")
_DRY_RUN_REFUSED = ("Superset refused the SQL of the chart's dataset, which this build made, when mnemiq asked how it "
                    "would run it, so mnemiq does not check it with its own SQL; why is in the log")
_RULES_UNKNOWN = ("mnemiq could not tell whether a row rule applies to you on the chart's dataset, so it does not "
                  "check it with its own SQL")


def _time_axis(client, form: dict, expectation: Expectation,
               dataset: DatasetFacts) -> Callable[[], bool | NotCheckable] | None:
    """For a chart drawn on an x-axis (a time series or a mixed chart; a waterfall's query does not rank by its
    measure), whether that axis, the first grouping column, is one of the dataset's time columns, asked of Superset
    when called. None for any other chart."""
    if form.get("viz_type") not in _X_AXIS - {"waterfall"} or not expectation.dimensions:
        return None

    def on_time() -> bool | NotCheckable:
        # Only a column Superset says is no time column lets the cut be read as a ranking: one it does not list, or
        # whose flag it does not give, is not known to be one.
        columns = read_dttm_columns(client, dataset.dataset_id)
        if isinstance(columns, NotCheckable):
            return columns
        axis = expectation.dimensions[0]
        if axis not in columns:
            return NotCheckable(f"its x-axis {axis!r} is not among the dataset's columns")
        if columns[axis] is None:
            return NotCheckable(f"Superset does not say whether {axis!r} is a time column")
        return columns[axis]

    return on_time


def _numbers(client, adapter, *, chart_id: int, form: dict, dataset_id: int, snapshot: Snapshot, grants: GrantSet,
             guard_fanout: bool | None, timeout_s: float, made: Mapping[int, str],
             before_call: Callable[[], None] | None = None) -> _Numbers | ChartCheck:
    """The chart's numbers against mnemiq's own: any status but `FORMAT_WRONG`."""
    def not_checked(reason: str) -> ChartCheck:
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, reason)

    dataset_id = dataset_of(form, dataset_id)
    if isinstance(dataset_id, NotCheckable):
        return not_checked(dataset_id.reason)
    dataset = read_dataset_facts(client, dataset_id)
    if isinstance(dataset, NotCheckable):
        return not_checked(dataset.reason)
    expectation = expectation_from_form_data(form, dataset=dataset)
    if isinstance(expectation, NotCheckable):
        return not_checked(expectation.reason)
    # Read before mnemiq's side is: its read is made in this order, so a cut over more groups than one read holds is
    # decided by the groups it keeps.
    ranking = ranking_of(form, expectation, dataset=dataset)
    sql, source = "", None
    if dataset.is_virtual and dataset.dataset_id in made:
        if not isinstance(dataset.sql, str) or not dataset.sql:
            return not_checked(_SQL_UNSHOWN)
        # Superset holds the SQL as the build sent it, in Superset's spelling (`for_superset`), and the dataset is read
        # with the SQL the build decided, which names each table as the deciders read one.
        try:
            sent = for_superset(made[dataset.dataset_id])
        except (RefusedStatement, UnreadableStatement) as exc:
            return not_checked(f"the chart's dataset is one this build made with SQL mnemiq does not send to Superset "
                               f"({exc}), so mnemiq does not check it with its own SQL")
        if dataset.sql != sent:
            return not_checked(_SQL_CHANGED)
        applies = row_rule_applies(client, dataset.dataset_id)
        if applies is None:
            return not_checked(_RULES_UNKNOWN)
        if applies:
            return not_checked(_RULE_APPLIES)
        # On Oracle, Superset's own dry run of the SQL it holds says whether it would run it as mnemiq reads it.
        try:
            adapter.confirm_sent(sent, before_call=before_call)
        except RefusedStatement as exc:
            return not_checked(f"the chart's dataset is one this build made with SQL Superset would not run as mnemiq "
                               f"reads it ({exc}), so mnemiq does not check it with its own SQL")
        except (SupersetAccessRefused, SupersetQueryFailed) as exc:
            logger.warning("the value check: Superset refused the dry run of chart %s's dataset's SQL (%s): %s",
                           chart_id, type(exc).__name__, exc)
            return not_checked(_DRY_RUN_REFUSED)
        source = _virtual_source(made[dataset.dataset_id])
    own = source is not None or not dataset.is_virtual
    if not own:
        written = [label for label, _ in expectation.metrics if label not in dataset.saved_metric_names]
        if written:
            return not_checked(f"the measure {written[0]!r} is written on the chart over a virtual dataset: mnemiq "
                               "never checks a virtual dataset with its own SQL, which a row rule attached to that "
                               "dataset alone would not narrow, and Superset's query_dataset takes saved metrics only")
        if expectation.time_grain:
            return not_checked("the chart groups a virtual dataset by a time grain, which Superset's query_dataset "
                               "cannot ask for")
    else:
        if getattr(adapter, "dialect", None) != superset_dialect():
            return not_checked("mnemiq's own statement for a chart is written in the dialect of Superset mode's "
                               "database, Postgres or Oracle, and this adapter speaks another")
        if dataset.database_id != getattr(adapter, "database_id", dataset.database_id) \
                or (dataset.schema or None) != getattr(adapter, "schema", dataset.schema):
            return not_checked("the chart's dataset is on another database or schema than the one mnemiq reads")
        if source is None and not dataset.table:
            return not_checked("the chart's dataset names no table")
        saved = dict(dataset.saved_metric_sql)
        if empty := [label for label, _ in expectation.metrics if label in saved and not saved[label].strip()]:
            return not_checked(f"Superset gave no expression for the saved metric {empty[0]!r}")
        if (unwritten := _unwritten(expectation)) is not None:
            return not_checked(unwritten)
        sql = _statement(expectation, dataset, grained=True, source=source, ranking=ranking)
        # Checked on the statement itself, so a saved metric's own expression is covered too: `execute_sql` renders a
        # statement as a template, and the adapter would refuse it later in words about the statement, not the chart.
        if (marker := next((m for m in JINJA_DELIMITERS if m in sql), None)) is not None:
            return not_checked(f"the chart's settings hold {marker}, which Superset reads as a template marker, so "
                               "mnemiq cannot compute the same and sends nothing")
    chart = _chart_rows(client, chart_id, expectation)
    if isinstance(chart, ChartCheck):
        return chart
    compared = _OWN_SQL if own else _QUERY_DATASET

    def read(limit: int | None) -> _Rows | NotCheckable:
        if not own:
            return _dataset_rows(client, expectation, dataset, ranking, limit=limit)
        statement = sql if limit is None else _statement(expectation, dataset, grained=True, source=source,
                                                         ranking=ranking, limit=limit)
        return _own_rows(statement, expectation, chart_id=chart_id, adapter=adapter, snapshot=snapshot, grants=grants,
                         timeout_s=timeout_s, guard_fanout=guard_fanout, before_call=before_call, ranking=ranking,
                         limit=limit)

    # A ranked chart's groups are read in its ranking order, first only twice its cut (`_first_read`), and on up to one
    # read's cap only when those do not settle where the cut falls.
    first = _first_read(ranking)
    mine = read(first)
    if first is not None and isinstance(mine, _Rows):
        # What the HAVING leaves of it, which may hold every group it keeps (`_left_out`): only a read still stopped at
        # its cap whose cut it does not settle reads on.
        kept = _left_out(mine, expectation.not_null)
        if kept.cap is not None and not _cut_inside(kept, ranking):
            mine = read(None)
    if isinstance(mine, NotCheckable):
        return not_checked(mine.reason)
    mine = _left_out(mine, expectation.not_null)
    labels = tuple(label for label, _ in expectation.metrics)
    found = _compare(chart_id, chart, mine, _row_limit(form), source=compared, labels=labels, ranking=ranking,
                     time_axis=_time_axis(client, form, expectation, dataset))
    found = _Numbers(found.chart_id, found.status, found.reason, found.chart_values, found.mnemiq_values, labels,
                     chart.values)
    if found.status is ChartStatus.MISMATCH and expectation.time_grain and compared is _OWN_SQL:
        plain = _own_rows(_statement(expectation, dataset, grained=False, source=source), expectation, chart_id=chart_id,
                          adapter=adapter, snapshot=snapshot, grants=grants, timeout_s=timeout_s,
                          guard_fanout=guard_fanout, before_call=before_call)
        # The chart's rows sit on the grain (`_line_up`), yet rows rebuilt without it can too, under a row limit that
        # keeps only first-of-month days. Only on positive evidence: every one of the chart's rows found among mnemiq's
        # ungrouped ones, with equal values. An ungrouped read at its row cap, or an empty chart, proves nothing, and
        # the grouped finding stands.
        if not isinstance(plain, NotCheckable) and _all_found(chart, _left_out(plain, expectation.not_null)):
            return not_checked(_NO_GRAIN)
    return found


def _left_out(rows: _Rows, not_null: tuple[int, ...]) -> _Rows:
    """`rows` without the groups whose value at any of `not_null` is NULL: what the chart's HAVING
    `(<metric>) IS NOT NULL` leaves out of its query. A read at its cap stays one, and a read in a ranking's order stays
    in it (what is left of a ranking's first groups are the first of what is left), except one that read on into the
    groups with no ranking value which the HAVING leaves out: those come last, so it read every group the HAVING
    keeps."""
    if not not_null:
        return rows
    kept = {key: values for key, values in rows.values.items()
            if all(index < len(values) and values[index] is not None for index in not_null)}
    if rows.ranked is not None and rows.through is None and rows.ranked.metric_index in not_null:
        return _Rows(kept, len(kept))
    return _Rows(kept, len(kept), cap=rows.cap, ranked=rows.ranked, through=rows.through, unranked=rows.unranked)


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
                every_kind_known: bool = False, made: Mapping[int, str] | None = None) -> ChartCheck:
    """Check chart `chart_id` as the caller `adapter` acts for, the builder: Superset is read through `adapter.client`
    and no other, and `adapter` runs mnemiq's own statement under `grants` and `snapshot`. `dataset_id` is the dataset
    mnemiq built the chart on, which a chart saved since in Explore overrides with its own. `kinds` gives the kind of
    each metric mnemiq knows, by the label Superset names its values by; `saved` the dataset's saved metrics mnemiq
    read, by name.

    A percent measure is a fraction, shown by the format times 100: a checked chart whose percent value is above 1 is
    `FORMAT_WRONG` (`percent_problem`), its values kept.

    A wrong format is reported whatever the numbers, unless they are a mismatch or the chart is empty. A chart is
    checked only when the format of every metric mnemiq knows the kind of was judged: numbers that match beside a
    format mnemiq could not judge are not checked, their values kept. `every_kind_known` says `kinds` holds every
    metric the caller made the chart with (the builder's check), so a metric of any other label is one whose format
    was not judged: Superset saved it under other text than mnemiq sent, or someone added it since. A refusal of the
    caller (`SupersetRefusal`) is raised: it stays the same for every later call the request makes.

    `before_call`, when given, runs before each Superset call the check makes itself, and once before mnemiq's own
    decided statement (whose catalogue reads, probe and run the adapter makes); whatever it raises ends the check.

    `made` holds each virtual dataset the caller's build made, by id, with the SQL it decided for it, which it sent in
    Superset's spelling (`for_superset`). A virtual dataset is never checked with mnemiq's own SQL, except one in
    `made` whose SQL Superset holds as it was sent and on which no row rule applies
    to the caller (the extension's `row_rules`); when one applies, or mnemiq cannot tell, its statement is not run and
    the chart is not checked. On Oracle that SQL must also be SQL Superset's own dry run says it would run as mnemiq
    reads it (`SupersetSqlAdapter.confirm_sent`), or the chart is not checked, saying why."""
    client = adapter.client if before_call is None else _BeforeEachCall(adapter.client, before_call)
    read = _read_chart(client, chart_id)
    if isinstance(read, NotCheckable):
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, read.reason)
    viz_type, form = read
    problem, unjudged = _format_problem(viz_type, form, kinds, saved, every_kind_known=every_kind_known)
    found = _numbers(client, adapter, chart_id=chart_id, form=form, dataset_id=dataset_id, snapshot=snapshot,
                     before_call=before_call, made=made or {},
                     grants=grants, guard_fanout=guard_fanout, timeout_s=timeout_s)
    if problem is None and unjudged is not None and found.status is ChartStatus.CHECKED:
        return _outcome(chart_id, ChartStatus.NOT_CHECKED, f"{found.reason}, but {unjudged}", found.chart_values,
                        found.mnemiq_values)
    if problem is None and unjudged is None and found.status is ChartStatus.CHECKED \
            and isinstance(found, _Numbers) \
            and (scale := percent_problem(kinds, found.labels, found.rows)) is not None:
        return _outcome(chart_id, ChartStatus.FORMAT_WRONG, scale, found.chart_values, found.mnemiq_values)
    if problem is None or found.status in (ChartStatus.MISMATCH, ChartStatus.EMPTY):
        return _plain(found)
    reason = problem if found.status is ChartStatus.CHECKED else f"{problem}. Its numbers were not compared: " \
                                                                  f"{found.reason}"
    return _outcome(chart_id, ChartStatus.FORMAT_WRONG, reason, found.chart_values, found.mnemiq_values)
