"""The value check: a saved chart's numbers, as Superset gives them to the builder, against the same numbers mnemiq
computes with its own decided SQL, and the format the chart shows them in.

What is compared is read from the chart's current `form_data`, so a chart changed by hand is checked as it now is.
Whatever mnemiq cannot reproduce faithfully is "not checked", with mnemiq's own reason, and never reported checked.
Runs against `FakeSuperset`, a real MCP server over streamable HTTP, through the real client and adapter.
"""

from __future__ import annotations

import copy
import inspect
import logging
import re
from datetime import datetime, timedelta

import pytest

import mnemiq.superset.chart_check as chart_check
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.chart_check import (
    ChartStatus,
    DatasetFacts,
    Expectation,
    NotCheckable,
    check_chart,
    expectation_from_form_data,
)
from mnemiq.superset.charts import Filter
from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetPermissionDenied
from mnemiq.superset.measures import MeasureKind
from mnemiq.superset.saved_metrics import SavedMetric
from tests.superset_fake import FakeSuperset

TOKEN = "tok-builder"
CHART_ID = 41
DATASET_ID = 12
DATABASE_ID = 7
SCHEMA = "public"
GRANTS = GrantSet(frozenset({"policy"}))
SNAPSHOT = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id="policy.region", object_id="policy", name="region", data_type="VARCHAR"),
    Column(id="policy.channel", object_id="policy", name="channel", data_type="VARCHAR"),
    Column(id="policy.premium", object_id="policy", name="premium", data_type="INTEGER"),
    Column(id="policy.order_date", object_id="policy", name="order_date", data_type="TIMESTAMP"),
])
USD = {"symbol": "USD", "symbolPosition": "prefix"}
# What `execute_sql` answers the deciders' own statements: the catalogue reads (each sent with `force_refresh`) and the
# proof's probe, which returns no rows.
EMPTY = {"success": True, "columns": [], "rows": []}
_PROBE = re.compile(r"AS mnemiq_probe WHERE 1 = 0$")
# Text only Superset writes in these tests, so where it ends up can be read.
SAID = "superset-said-this-about-public.policy"
# An ad hoc SQL metric as MCP's `create_metric_object` (`superset/mcp_service/chart/chart_utils.py`) writes one.
PREMIUM = {"aggregate": None, "column": None, "datasourceWarning": False, "expressionType": "SQL",
           "hasCustomLabel": True, "label": "Total premium", "optionName": "metric_sql_1", "sqlExpression":
           "SUM(premium)"}
# The neutral time binding MCP's `_bind_dashboard_time_range_filter` adds on the dataset's main datetime column.
NO_TIME_FILTER = {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "order_date",
                  "operator": "TEMPORAL_RANGE", "comparator": "No filter"}
WEB_ONLY = {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "channel", "operator": "IN",
            "comparator": ["web"]}
TOTAL = {"kinds": {"Total premium": MeasureKind.CURRENCY}}


def _bar(**changes) -> dict:
    """A bar chart's form_data as rc2's MCP `generate_chart` saves it for mnemiq's bar chart spec."""
    form = {"viz_type": "echarts_timeseries_bar", "x_axis": "region", "groupby": [],
            "metrics": [copy.deepcopy(PREMIUM)],
            "adhoc_filters": [copy.deepcopy(WEB_ONLY), copy.deepcopy(NO_TIME_FILTER)], "row_limit": 100,
            "y_axis_format": ",.2f", "currency_format": dict(USD), "x_axis_sort_series_type": "name"}
    form.update(changes)
    return form


def _info(form_data, *, viz_type: str | None = None, **more) -> dict:
    """`get_chart_info`'s answer with `select_columns` naming the fields the check reads."""
    viz = viz_type or (form_data or {}).get("viz_type", "echarts_timeseries_bar")
    return {"id": CHART_ID, "slice_name": "Premium by region", "viz_type": viz, "form_data": form_data,
            "changed_on": "2026-11-13T10:00:00", **more}


def _dataset(*, virtual: bool = False, metrics=(), schema: str = SCHEMA, database_id: int = DATABASE_ID,
             dataset_id: int = DATASET_ID) -> dict:
    """`get_dataset_info`'s answer: each saved metric with the five fields rc2's `SqlMetricInfo` carries. A virtual
    dataset is named as the granted table its SQL reads, the case where SQL over that table would pass the deciders and
    read it without a row rule attached to the dataset alone."""
    return {"id": dataset_id, "table_name": "policy", "schema": schema,
            "database_id": database_id, "is_virtual": virtual, "sql": "SELECT * FROM policy" if virtual else None,
            "metrics": [{"metric_name": name, "verbose_name": None, "expression": expression, "description": None,
                         "d3format": None} for name, expression in metrics]}


def _chart_data(columns, *rows, **more) -> dict:
    """`get_chart_data`'s answer for a one-query chart: `ChartData`, rows keyed by column name."""
    return {"chart_id": CHART_ID, "chart_name": "Premium by region", "chart_type": "echarts_timeseries_bar",
            "columns": [{"name": name, "display_name": name, "data_type": "string", "sample_values": [],
                         "null_count": 0, "unique_count": 0} for name in columns],
            "data": [dict(zip(columns, row)) for row in rows], "query_results": None, "row_count": len(rows),
            "total_rows": len(rows), "summary": "", "insights": [], "data_quality": {},
            "recommended_visualizations": [], "data_freshness": None,
            "performance": {"query_duration_ms": 3, "cache_status": "fresh_query", "optimization_suggestions": []},
            **more}


# `get_chart_data`'s answer for a chart whose query returned no rows: rc2 answers a `ChartError`, not an empty table.
NO_DATA = {"error": f"No data available for chart {CHART_ID}", "message": f"No data available for chart {CHART_ID}",
           "error_type": "NoData", "timestamp": "2026-11-13T10:00:00Z", "details": None, "suggestions": [],
           "error_code": None}


def _result(*rows, columns=(("region", "object"), ("m0", "int64"))) -> dict:
    """A successful `execute_sql` payload holding `rows`, each a tuple in the order of `columns`."""
    names = [name for name, _ in columns]
    return {"success": True, "columns": [{"name": name, "type": dtype} for name, dtype in columns],
            "rows": [dict(zip(names, row)) for row in rows]}


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        yield served


def _serve(fake, *, form_data, chart, mnemiq=None, dataset=None, query=None, info=None) -> None:
    """Answer the four tools as rc2 does. `mnemiq` answers the check's own data statement: a payload, an exception, or
    a function of the statement returning either; the deciders' own statements are answered with no rows."""
    fake.reply("get_chart_info", lambda request, authorization: info if info is not None else _info(form_data))
    fake.reply("get_dataset_info", lambda request, authorization: dataset if dataset is not None else _dataset())
    fake.reply("get_chart_data", lambda request, authorization: chart(request) if callable(chart) else chart)

    def execute(request, authorization):
        if request.get("force_refresh") or _PROBE.search(request["sql"]):
            return EMPTY
        return mnemiq(request["sql"]) if callable(mnemiq) else mnemiq

    fake.reply("execute_sql", execute)
    if query is not None:
        fake.reply("query_dataset", lambda request, authorization: query)


def _adapter(fake) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken(TOKEN)), database_id=DATABASE_ID, schema=SCHEMA)


def _check(fake, *, kinds=None, saved=None, dataset_id: int = DATASET_ID):
    return check_chart(_adapter(fake), chart_id=CHART_ID, dataset_id=dataset_id,
                       kinds=TOTAL["kinds"] if kinds is None else kinds, saved=saved or {}, snapshot=SNAPSHOT,
                       grants=GRANTS, guard_fanout=False)


def _statements(fake) -> list[str]:
    """The data statements the check sent: not a catalogue read, a dry run or the proof's probe."""
    return [call.request["sql"] for call in fake.calls if call.tool == "execute_sql"
            and not call.request.get("force_refresh") and not call.request.get("dry_run")
            and not _PROBE.search(call.request["sql"])]


def _tools(fake) -> list[str]:
    return [call.tool for call in fake.calls]


# --- checked, and what is compared ------------------------------------------------------------------------------------


def test_matching_bar_chart_is_checked(fake):
    """The chart's rows and mnemiq's own agree: checked. Every call is made as the builder, the chart is read with its
    `form_data` and `changed_on`, and mnemiq's statement carries the chart's measure, grouping and SIMPLE filter, and
    nothing for the neutral time binding."""
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], ("east", 1200), ("west", 800)),
           mnemiq=_result(("east", 1200), ("west", 800)))
    check = _check(fake)
    assert check.status is ChartStatus.CHECKED, check.reason
    assert check.chart_id == CHART_ID
    assert check.chart_values == check.mnemiq_values == (("east", 1200), ("west", 800))
    info = next(call for call in fake.calls if call.tool == "get_chart_info")
    assert info.request["identifier"] == CHART_ID and {"form_data", "changed_on"} <= set(info.request["select_columns"])
    assert next(call for call in fake.calls if call.tool == "get_dataset_info").request["identifier"] == DATASET_ID
    assert next(call for call in fake.calls if call.tool == "get_chart_data").request["identifier"] == CHART_ID
    [sql] = _statements(fake)
    assert "SUM(premium)" in sql and '"region"' in sql and "GROUP BY" in sql and "'web'" in sql
    assert "order_date" not in sql
    assert {call.authorization for call in fake.calls} == {f"Bearer {TOKEN}"}


@pytest.mark.parametrize("chart_rows, mnemiq_rows", [
    ((("east", 1200), ("west", 800)), (("east", 1200), ("west", 801))),
    ((("east", 1200),), (("east", 1200), ("west", 800))),
    ((("east", 1200), ("west", 800), ("north", 5)), (("east", 1200), ("west", 800))),
], ids=["a value off", "a group missing", "a group extra"])
def test_one_group_off_is_a_mismatch(fake, caplog, chart_rows, mnemiq_rows):
    """Both sides' values are returned for the response, and neither is kept: nothing in the module holds them, the
    log carries none of them, and the next check answers with its own."""
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], *chart_rows),
           mnemiq=_result(*mnemiq_rows))
    with caplog.at_level(logging.DEBUG):
        check = _check(fake)
    assert check.status is ChartStatus.MISMATCH, check.reason
    assert set(check.chart_values) == set(chart_rows) and set(check.mnemiq_values) == set(mnemiq_rows)
    logged = " ".join(record.getMessage() for record in caplog.records if record.name.startswith("mnemiq"))
    for value in ("1200", "801", "north"):
        assert value not in logged
        assert not any(value in repr(held) for name, held in vars(chart_check).items()
                       if not name.startswith("__") and not callable(held) and not isinstance(held, type(re)))
    fake.reply("get_chart_data", lambda request, authorization: _chart_data(["region", "Total premium"], ("south", 3)))
    fake.reply("execute_sql", lambda request, authorization: EMPTY if request.get("force_refresh")
               or _PROBE.search(request["sql"]) else _result(("south", 3)))
    again = _check(fake)
    assert (again.status, again.chart_values, again.mnemiq_values) == (ChartStatus.CHECKED, (("south", 3),),
                                                                        (("south", 3),))


def test_empty_chart_is_empty(fake):
    """rc2 answers a chart whose query returns no rows with a `NoData` error; mnemiq's own statement returns none
    either."""
    _serve(fake, form_data=_bar(), chart=NO_DATA, mnemiq=_result())
    check = _check(fake)
    assert (check.status, check.chart_values, check.mnemiq_values) == (ChartStatus.EMPTY, (), ())


def test_empty_chart_with_rows_on_mnemiqs_side_is_a_mismatch(fake):
    _serve(fake, form_data=_bar(), chart=NO_DATA, mnemiq=_result(("east", 1200)))
    check = _check(fake)
    assert (check.status, check.chart_values, check.mnemiq_values) == (ChartStatus.MISMATCH, (), (("east", 1200),))


def test_truncated_chart_data_is_cut(fake):
    """Superset's size guard cut the chart's rows (`_response_truncated`): what arrived is not all of them, so nothing
    is compared and mnemiq runs nothing."""
    cut = _chart_data(["region", "Total premium"], ("east", 1200), _response_truncated=True,
                      _truncation_notes=["Result truncated: 1 of 2 rows returned (limit ~50,000 bytes)."])
    _serve(fake, form_data=_bar(), chart=cut, mnemiq=_result(("east", 1200), ("west", 800)))
    check = _check(fake)
    assert check.status is ChartStatus.CUT and check.reason
    assert _statements(fake) == []


@pytest.mark.parametrize("currency", [{"symbol": "$", "symbolPosition": "prefix"}, None],
                         ids=["a symbol rc2's renderer drops", "the currency cleared by hand"])
def test_wrong_currency_is_format_wrong_even_when_values_match(fake, currency):
    form = _bar() if currency is not None else {k: v for k, v in _bar().items() if k != "currency_format"}
    if currency is not None:
        form["currency_format"] = currency
    _serve(fake, form_data=form, chart=_chart_data(["region", "Total premium"], ("east", 1200)),
           mnemiq=_result(("east", 1200)))
    check = _check(fake)
    assert check.status is ChartStatus.FORMAT_WRONG
    assert "Total premium" in check.reason and "currency" in check.reason
    assert check.chart_values == check.mnemiq_values == (("east", 1200),)


def test_a_lower_case_currency_code_shows_as_rc2_shows_it(fake):
    """`Intl.NumberFormat` takes `usd` as a code, so rc2 shows a currency there and drops a percent's `%`: a percent
    measure shown beside it is in the wrong format, and the reason says the currency shows."""
    _serve(fake, form_data=_bar(y_axis_format=",.1%", currency_format={"symbol": "usd", "symbolPosition": "prefix"}),
           chart=_chart_data(["region", "Total premium"], ("east", 1200)), mnemiq=_result(("east", 1200)))
    check = _check(fake, kinds={"Total premium": MeasureKind.PERCENT})
    assert check.status is ChartStatus.FORMAT_WRONG, check.reason
    assert "with the currency usd" in check.reason and "does not show" not in check.reason


def test_a_metric_whose_kind_is_unknown_is_checked_for_its_values_only(fake):
    """A metric added by hand has no kind mnemiq gave it, so its format is not judged; its numbers still are."""
    _serve(fake, form_data=_bar(currency_format=None, y_axis_format=",d"),
           chart=_chart_data(["region", "Total premium"], ("east", 1200)), mnemiq=_result(("east", 1200)))
    assert _check(fake, kinds={}).status is ChartStatus.CHECKED


def test_format_wrong_is_reported_where_the_numbers_are_not_checked(fake):
    """A format that is wrong is wrong whatever the numbers: a chart whose numbers cannot be checked still says so,
    and says why its numbers were not checked."""
    sql_filter = {"clause": "WHERE", "expressionType": "SQL", "sqlExpression": "premium > 0"}
    _serve(fake, form_data=_bar(currency_format=None, adhoc_filters=[sql_filter]), chart=NO_DATA)
    check = _check(fake)
    assert check.status is ChartStatus.FORMAT_WRONG and "SQL filter" in check.reason
    assert "get_chart_data" not in _tools(fake) and _statements(fake) == []


# --- not checked ------------------------------------------------------------------------------------------------------

M_SAVED = "total_premium"
POST_PROCESSING = [
    ("rolling_type", _bar(rolling_type="cumsum"), "rolling"),
    ("time_compare", _bar(time_compare=["1 year ago"], comparison_type="values"), "time comparison"),
    ("contributionMode", _bar(contributionMode="row"), "contribution"),
    ("percent_metrics", {"viz_type": "table", "query_mode": "aggregate", "groupby": ["region"],
                         "metrics": [copy.deepcopy(PREMIUM)], "percent_metrics": [copy.deepcopy(PREMIUM)],
                         "adhoc_filters": [], "row_limit": 100}, "percentage"),
    ("stack Expand", _bar(viz_type="echarts_area", stack="Expand"), "100%"),
    ("forecast", _bar(forecastEnabled=True, forecastPeriods=10), "forecast"),
    ("rebase_percent_change", _bar(viz_type="echarts_timeseries_line", rebase_percent_change=True), "percent change"),
    ("big number with trendline", {"viz_type": "big_number", "metric": copy.deepcopy(PREMIUM),
                                   "granularity_sqla": "order_date", "time_grain_sqla": "P1M",
                                   "show_trend_line": True, "adhoc_filters": []}, "trendline"),
    ("box plot", {"viz_type": "box_plot", "metrics": [copy.deepcopy(PREMIUM)], "columns": ["region"],
                  "groupby": ["channel"], "whiskerOptions": "Tukey", "adhoc_filters": []}, "box plot"),
    ("histogram", {"viz_type": "histogram_v2", "column": "premium", "groupby": ["region"], "bins": 5,
                   "adhoc_filters": []}, "histogram"),
    ("handlebars", {"viz_type": "handlebars", "metrics": [copy.deepcopy(PREMIUM)], "groupby": ["region"],
                    "query_mode": "aggregate", "handlebarsTemplate": "<p>x</p>", "adhoc_filters": []}, "handlebars"),
    ("interactive pivot", {"viz_type": "ag-grid-pivot-table", "metrics": [copy.deepcopy(PREMIUM)],
                           "groupbyRows": ["region"], "groupbyColumns": [], "adhoc_filters": []},
     "interactive pivot"),
]


@pytest.mark.parametrize("form, said", [(form, said) for _, form, said in POST_PROCESSING],
                         ids=[name for name, _, _ in POST_PROCESSING])
def test_post_processing_is_not_checked(fake, form, said):
    """rc2's `get_chart_data` rebuilds a chart with no saved query context from its `form_data` without any
    post-processing, so a chart whose shown numbers come from post-processing (or from a plugin whose numbers the query
    does not hold) is not checked, says why, and nothing is asked for its data."""
    _serve(fake, form_data=form, chart=NO_DATA, mnemiq=_result())
    check = _check(fake, kinds={})
    assert check.status is ChartStatus.NOT_CHECKED
    assert said in check.reason
    assert (check.chart_values, check.mnemiq_values) == (None, None)
    assert "get_chart_data" not in _tools(fake) and "query_dataset" not in _tools(fake) and _statements(fake) == []


@pytest.mark.parametrize("changes", [
    {"adhoc_filters": [WEB_ONLY, {"clause": "WHERE", "expressionType": "SQL", "sqlExpression": "premium > 0"}]},
    {"adhoc_filters": [{"clause": "HAVING", "expressionType": "SQL", "sqlExpression": "SUM(premium) > 0"}]},
    {"where": "premium > 0"},
    {"having": "SUM(premium) > 0"},
], ids=["SQL WHERE", "SQL HAVING", "legacy where", "legacy having"])
def test_sql_filter_is_not_checked(fake, changes):
    _serve(fake, form_data=_bar(**changes), chart=NO_DATA)
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "SQL filter" in check.reason
    assert _statements(fake) == [] and "get_chart_data" not in _tools(fake)


@pytest.mark.parametrize("info", [
    _info(_bar()) | {"form_data": {"_truncated": True, "_message": "Dict with 31 keys truncated. Keys: viz_type..."},
                     "_response_truncated": True, "_truncation_notes": ["Field 'form_data' dict summarized (31 keys)"]},
    _info(_bar()) | {"_response_truncated": True, "_truncation_notes": ["Field 'description' truncated"]},
    _info(None),
], ids=["form_data stubbed", "the answer cut elsewhere", "form_data withheld"])
def test_stubbed_form_data_is_not_checked(fake, info):
    """`get_chart_info` replaces a `form_data` over its size guard with a `{"_truncated": True}` stub,
    and withholds it from a caller without the data-model privilege. Either reads as "could not read": nothing else is
    asked, and nothing is compared."""
    _serve(fake, form_data=None, chart=NO_DATA, info=info)
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "could not read" in check.reason
    assert _tools(fake) == ["get_chart_info"]


def test_virtual_dataset_chart_is_never_checked_by_raw_sql(fake):
    """A measure written on the chart over a virtual dataset is not checked, and no statement is sent: a row rule
    attached to the virtual dataset alone would not apply to mnemiq's SQL over its tables. A saved metric on it is
    checked through `query_dataset`, which applies the dataset's own rules."""
    rows = (("east", 1200), ("west", 800))
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], *rows), mnemiq=_result(*rows),
           dataset=_dataset(virtual=True, metrics=[(M_SAVED, "SUM(premium)")]))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "virtual dataset" in check.reason
    assert "execute_sql" not in _tools(fake) and "query_dataset" not in _tools(fake)

    fake.calls.clear()
    saved_form = _bar(metrics=[M_SAVED])
    _serve(fake, form_data=saved_form, chart=_chart_data(["region", M_SAVED], *rows),
           dataset=_dataset(virtual=True, metrics=[(M_SAVED, "SUM(premium)")]),
           query={"dataset_id": DATASET_ID, "dataset_name": "policy",
                  "columns": [{"name": "region"}, {"name": M_SAVED}],
                  "data": [{"region": r, M_SAVED: v} for r, v in rows], "row_count": 2, "total_rows": 2})
    record = SavedMetric(dataset_id=DATASET_ID, name=M_SAVED, expression="SUM(premium)", verbose_name=None,
                         d3format=None, currency=None, certified_by=None, warning_text=None)
    check = _check(fake, kinds={M_SAVED: MeasureKind.CURRENCY}, saved={M_SAVED: record})
    assert check.status is ChartStatus.CHECKED, check.reason
    assert check.chart_values == check.mnemiq_values == rows
    [query] = [call.request for call in fake.calls if call.tool == "query_dataset"]
    assert (query["dataset_id"], query["metrics"], query["columns"]) == (DATASET_ID, [M_SAVED], ["region"])
    assert query["filters"] == [{"col": "channel", "op": "IN", "val": ["web"]}]
    assert "execute_sql" not in _tools(fake)


@pytest.mark.parametrize("chart_value, mnemiq_value, dtype, status", [
    (0.30000000000000004, 0.3, "float64", ChartStatus.CHECKED),
    (1.0, 1.000001, "float64", ChartStatus.MISMATCH),
    (100_000_000.0, 100_000_000.00005, "float64", ChartStatus.CHECKED),
    (100_000_000.0, 100_000_000.0002, "float64", ChartStatus.MISMATCH),
    (123_456_789.01, 123_456_789.02, "float64", ChartStatus.MISMATCH),
    (1_000_000_000_000, 1_000_000_000_001, "int64", ChartStatus.MISMATCH),
    (12, 12.0, "float64", ChartStatus.CHECKED),
], ids=["float summation noise", "float beyond rel_tol", "just inside rel_tol (5e-13)",
        "just outside rel_tol (2e-12)", "a cent apart above 1e8", "integers exact", "int beside float"])
def test_float_tolerance(fake, chart_value, mnemiq_value, dtype, status):
    """Floats agree within `rel_tol=1e-12`, which absorbs the noise of summing floats in another order, and no more: a
    cent apart on a value above 1e7 shows in `,.2f`, so it is a mismatch. Two integers must be equal."""
    form = {"viz_type": "big_number_total", "metric": copy.deepcopy(PREMIUM), "adhoc_filters": [],
            "y_axis_format": ",.2f", "currency_format": dict(USD)}
    _serve(fake, form_data=form, chart=_chart_data(["Total premium"], (chart_value,)),
           mnemiq=_result((mnemiq_value,), columns=(("m0", dtype),)))
    assert _check(fake).status is status


def test_hand_changed_filter_is_checked_as_it_now_is(fake):
    """The filter mnemiq planned (`web`) was changed by hand to `store` and a second value added: mnemiq's statement
    carries the chart's filter as it now is, not the plan's."""
    changed = {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "channel", "operator": "IN",
               "comparator": ["store", "phone"]}
    _serve(fake, form_data=_bar(adhoc_filters=[changed, NO_TIME_FILTER]),
           chart=_chart_data(["region", "Total premium"], ("east", 40)), mnemiq=_result(("east", 40)))
    assert _check(fake).status is ChartStatus.CHECKED
    [sql] = _statements(fake)
    assert "'store'" in sql and "'phone'" in sql and "'web'" not in sql


def test_the_charts_current_dataset_is_read(fake):
    """A chart saved in Explore names its dataset in `form_data["datasource"]`, which wins over the one mnemiq built
    it on: the chart is checked on the dataset it now reads."""
    _serve(fake, form_data=_bar(datasource="13__table"), chart=_chart_data(["region", "Total premium"], ("east", 1)),
           mnemiq=_result(("east", 1)), dataset=_dataset(dataset_id=13))
    assert _check(fake).status is ChartStatus.CHECKED
    assert next(call for call in fake.calls if call.tool == "get_dataset_info").request["identifier"] == 13


# --- time grains, row limits and shapes -------------------------------------------------------------------------------


def _line(**changes) -> dict:
    return _bar(viz_type="echarts_timeseries_line", x_axis="order_date", granularity_sqla="order_date",
                time_grain_sqla="P1M", adhoc_filters=[copy.deepcopy(WEB_ONLY), copy.deepcopy(NO_TIME_FILTER)]) | changes


MONTHS = (("2026-01-01T00:00:00", 30), ("2026-02-01T00:00:00", 12))
DAYS = (("2026-01-03T00:00:00", 10), ("2026-01-20T00:00:00", 20), ("2026-02-07T00:00:00", 12))
TIME = (("order_date", "datetime64[ns]"), ("m0", "int64"))


def test_time_grain_is_applied_to_the_x_axis(fake):
    """mnemiq groups the x-axis by the chart's time grain, as Superset does for a chart that has a saved query
    context, and Superset's timestamps and mnemiq's agree however each side spells them."""
    _serve(fake, form_data=_line(), chart=_chart_data(["order_date", "Total premium"], *MONTHS),
           mnemiq=_result(*MONTHS, columns=TIME))
    check = _check(fake)
    assert check.status is ChartStatus.CHECKED, check.reason
    [sql] = _statements(fake)
    assert re.search(r"DATE_TRUNC\('month', \"order_date\"\)", sql, re.IGNORECASE)


def test_grain_superset_did_not_apply_is_not_checked(fake):
    """rc2's `get_chart_data` rebuilds a chart with no saved query context (every chart MCP made) from `form_data`, and
    that rebuild never applies `time_grain_sqla` to a string x-axis. Its rows are then the ungrouped ones, which match
    mnemiq's statement without the grain: the numbers the chart shows were not compared, and it says so rather than
    calling it a mismatch."""
    _serve(fake, form_data=_line(), chart=_chart_data(["order_date", "Total premium"], *DAYS),
           mnemiq=lambda sql: _result(*(MONTHS if "DATE_TRUNC" in sql.upper() else DAYS), columns=TIME))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "time grain" in check.reason
    grained, plain = _statements(fake)
    assert "DATE_TRUNC" in grained.upper() and "DATE_TRUNC" not in plain.upper()


def test_grain_superset_did_not_apply_is_not_checked_under_a_row_limit_too(fake):
    """Superset's ungrouped rows cut by the chart's row limit still agree with mnemiq's ungrouped ones as far as they
    go: the grain was not applied, which is not a mismatch."""
    _serve(fake, form_data=_line(row_limit=2), chart=_chart_data(["order_date", "Total premium"], *DAYS[:2]),
           mnemiq=lambda sql: _result(*(MONTHS if "DATE_TRUNC" in sql.upper() else DAYS), columns=TIME))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "time grain" in check.reason


def test_grain_mismatch_either_way_is_a_mismatch(fake):
    off = (("2026-01-01T00:00:00", 31), ("2026-02-01T00:00:00", 12))
    _serve(fake, form_data=_line(), chart=_chart_data(["order_date", "Total premium"], *off),
           mnemiq=lambda sql: _result(*(MONTHS if "DATE_TRUNC" in sql.upper() else DAYS), columns=TIME))
    check = _check(fake)
    assert check.status is ChartStatus.MISMATCH
    assert check.mnemiq_values == MONTHS


def test_rows_cut_by_the_charts_row_limit_are_not_checked(fake):
    """No order is fixed for a chart's rows, so which groups its row limit keeps is Superset's choice."""
    _serve(fake, form_data=_bar(row_limit=2), chart=_chart_data(["region", "Total premium"], ("east", 1), ("west", 2)),
           mnemiq=_result(("east", 1), ("west", 2), ("north", 3)))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "row limit" in check.reason


def test_chart_data_without_the_named_columns_is_not_checked(fake):
    """rc2's rebuild reads `groupby`, never a pivot table's `groupbyRows`, so a pivot with no saved query context comes
    back as one total: Superset's rows do not carry the columns the chart's settings name, and cannot be lined up."""
    pivot = {"viz_type": "pivot_table_v2", "metrics": [copy.deepcopy(PREMIUM)], "groupbyRows": ["region"],
             "groupbyColumns": ["channel"], "aggregateFunction": "Sum", "adhoc_filters": [], "row_limit": 100,
             "valueFormat": ",.2f", "currency_format": dict(USD)}
    _serve(fake, form_data=pivot, chart=_chart_data(["Total premium"], (2000,)),
           mnemiq=_result(("east", "web", 2000),
                          columns=(("region", "object"), ("channel", "object"), ("m0", "int64"))))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "columns" in check.reason


def test_mixed_chart_checks_both_queries(fake):
    """A mixed chart answers with two query results, one per axis; each one's numbers are compared."""
    orders = {"expressionType": "SQL", "sqlExpression": "COUNT(*)", "label": "Orders"}
    form = _line(viz_type="mixed_timeseries", metrics_b=[orders], groupby_b=[], y_axis_format_secondary=",d")
    first = (("2026-01-01T00:00:00", 30),)
    second = (("2026-01-01T00:00:00", 4),)
    chart = _chart_data(["order_date", "Total premium"], *first,
                        query_results=[{"query_index": 0, "columns": ["order_date", "Total premium"],
                                        "data": [{"order_date": d, "Total premium": v} for d, v in first],
                                        "row_count": 1, "total_rows": 1},
                                       {"query_index": 1, "columns": ["order_date", "Orders"],
                                        "data": [{"order_date": d, "Orders": v} for d, v in second],
                                        "row_count": 1, "total_rows": 1}])
    _serve(fake, form_data=form, chart=chart,
           mnemiq=_result(("2026-01-01T00:00:00", 30, 4),
                          columns=(("order_date", "datetime64[ns]"), ("m0", "int64"), ("m1", "int64"))))
    check = _check(fake, kinds={"Total premium": MeasureKind.CURRENCY, "Orders": MeasureKind.COUNT})
    assert check.status is ChartStatus.CHECKED, check.reason
    assert check.chart_values == check.mnemiq_values == (("2026-01-01T00:00:00", 30, 4),)


def test_a_dataset_mnemiq_does_not_read_through_this_adapter_is_not_checked(fake):
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], ("east", 1)),
           mnemiq=_result(("east", 1)), dataset=_dataset(schema="sales"))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "schema" in check.reason
    assert "execute_sql" not in _tools(fake)


@pytest.mark.parametrize("form", [
    _bar(metrics=[PREMIUM | {"sqlExpression": "SUM(premium) * {{ rate }}"}]),
    _bar(adhoc_filters=[WEB_ONLY | {"comparator": ["{% if x %}"]}]),
], ids=["metric", "filter value"])
def test_a_template_marker_is_never_sent(fake, form):
    _serve(fake, form_data=form, chart=NO_DATA)
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "template" in check.reason
    assert "execute_sql" not in _tools(fake)


def test_a_saved_metrics_own_template_marker_is_never_sent(fake):
    """The marker is looked for in the statement mnemiq would send, so a saved metric whose own expression holds one is
    caught too, before anything is asked for."""
    _serve(fake, form_data=_bar(metrics=[M_SAVED]), chart=NO_DATA,
           dataset=_dataset(metrics=[(M_SAVED, "SUM(premium) * {{ rate }}")]))
    check = _check(fake, kinds={})
    assert check.status is ChartStatus.NOT_CHECKED and "template" in check.reason
    assert "execute_sql" not in _tools(fake) and "get_chart_data" not in _tools(fake)


def test_a_label_colliding_with_a_dimension_is_not_checked(fake):
    """Superset keys a row by label, so a metric labelled as a grouping column cannot be read apart from it."""
    _serve(fake, form_data=_bar(metrics=[PREMIUM | {"label": "region"}]), chart=NO_DATA)
    check = _check(fake, kinds={})
    assert check.status is ChartStatus.NOT_CHECKED and "region" in check.reason


# --- what Superset says, and who it refuses ---------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["get_chart_info", "get_chart_data", "get_chart_data error", "execute_sql"])
def test_superset_text_never_enters_the_reason(fake, caplog, where):
    """A failure is reported in mnemiq's own words; what Superset said goes to the log only."""
    chart = _chart_data(["region", "Total premium"], ("east", 1))
    mnemiq = _result(("east", 1))
    info = None
    if where == "get_chart_info":
        info = {"error": SAID, "message": SAID, "error_type": "NotFound"}
    elif where == "get_chart_data":
        chart = lambda request: RuntimeError(SAID)  # noqa: E731
    elif where == "get_chart_data error":
        chart = {"error": SAID, "message": SAID, "error_type": "DatasetNotAccessible"}
    else:
        mnemiq = {"success": False, "error": SAID, "error_type": "GENERIC_DB_ENGINE_ERROR"}
    _serve(fake, form_data=_bar(), chart=chart, mnemiq=mnemiq, info=info)
    with caplog.at_level(logging.WARNING):
        check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and check.reason
    assert SAID not in check.reason
    assert SAID in caplog.text


def test_a_refusal_of_the_caller_ends_the_check(fake):
    """Superset refusing the builder's roles stays refused for every later call, so it is raised, not read as one
    chart not checked: the step that asked ends there."""
    _serve(fake, form_data=_bar(), chart=NO_DATA)
    fake.reply("get_chart_info", lambda request, authorization: RuntimeError(
        "Permission denied: can_read on Chart for user builder (tool get_chart_info)"))
    with pytest.raises(SupersetPermissionDenied):
        _check(fake)


# --- reading form_data ------------------------------------------------------------------------------------------------

PHYSICAL = DatasetFacts(dataset_id=DATASET_ID, is_virtual=False, table="policy", sql=None,
                        saved_metric_names=frozenset({M_SAVED}))


def test_expectation_reads_what_mcp_saves():
    """rc2's own `map_config_to_form_data` output for mnemiq's line chart spec, read as the chart's query reads it: the
    x-axis then the grouping, the grain, and the SIMPLE filters, the neutral time binding dropped."""
    form = {"_mcp_dashboard_time_filter_subject": "order_date",
            "adhoc_filters": [{"clause": "WHERE", "comparator": ["east"], "expressionType": "SIMPLE", "operator": "IN",
                               "subject": "region"},
                              {"clause": "WHERE", "comparator": None, "expressionType": "SIMPLE",
                               "operator": "IS NULL", "subject": "premium"},
                              {"clause": "WHERE", "comparator": "No filter", "expressionType": "SIMPLE",
                               "operator": "TEMPORAL_RANGE", "subject": "order_date"}],
            "currency_format": dict(USD), "granularity_sqla": "order_date", "groupby": ["region", "channel"],
            "metrics": [copy.deepcopy(PREMIUM)], "row_limit": 5, "time_grain_sqla": "P1M",
            "viz_type": "echarts_timeseries_line", "x_axis": "order_date", "y_axis_format": ",.2f"}
    assert expectation_from_form_data(form, dataset=PHYSICAL) == Expectation(
        metrics=(("Total premium", "SUM(premium)"),), dimensions=("order_date", "region", "channel"),
        time_grain="P1M", filters=(Filter("region", "IN", ["east"]), Filter("premium", "IS NULL", None)))


def test_expectation_reads_saved_and_simple_metrics():
    simple = {"expressionType": "SIMPLE", "aggregate": "COUNT_DISTINCT", "column": {"column_name": "channel"},
              "label": "Channels"}
    form = {"viz_type": "table", "query_mode": "aggregate", "groupby": ["region"], "metrics": [M_SAVED, simple],
            "adhoc_filters": []}
    assert expectation_from_form_data(form, dataset=PHYSICAL) == Expectation(
        metrics=((M_SAVED, M_SAVED), ("Channels", 'COUNT(DISTINCT "channel")')), dimensions=("region",),
        time_grain=None, filters=())


@pytest.mark.parametrize("form, said", [
    (_bar(metrics=["no_such_metric"]), "saved metric"),
    (_bar(x_axis={"expressionType": "SQL", "sqlExpression": "upper(region)", "label": "R"}), "custom SQL column"),
    (_bar(groupby=[{"expressionType": "SQL", "sqlExpression": "upper(channel)", "label": "C"}]), "custom SQL column"),
    (_bar(adhoc_filters=[{"clause": "WHERE", "expressionType": "SIMPLE", "subject": "order_date",
                          "operator": "TEMPORAL_RANGE", "comparator": "Last week"}]), "time range"),
    (_bar(time_range="Last year"), "time range"),
    (_bar(viz_type="world_map"), "world_map"),
    ({"viz_type": "table", "query_mode": "raw", "all_columns": ["region"], "adhoc_filters": []}, "rows"),
    # `getQueryMode` (`plugin-chart-table/src/buildQuery.ts`): no mode saved, and columns of rows chosen, is raw
    ({"viz_type": "table", "all_columns": ["region"], "metrics": [copy.deepcopy(PREMIUM)], "adhoc_filters": []},
     "rows"),
], ids=["unknown saved metric", "custom x-axis", "custom grouping", "relative time filter", "time range",
        "a type mnemiq does not check", "raw table", "a table of rows by its saved columns"])
def test_expectation_names_what_it_cannot_translate(form, said):
    found = expectation_from_form_data(form, dataset=PHYSICAL)
    assert isinstance(found, NotCheckable) and said in found.reason


def test_a_saved_metrics_format_without_its_record_is_not_judged(fake):
    """A saved metric the caller did not read has no known format of its own, so a big number showing it cannot be
    judged, and the chart is not reported checked: its numbers are kept, and the reason names the metric. Given its
    record, a wrong format is caught and a right one checked."""
    form = {"viz_type": "big_number_total", "metric": M_SAVED, "adhoc_filters": []}
    _serve(fake, form_data=form, chart=_chart_data([M_SAVED], (5,)), mnemiq=_result((5,), columns=(("m0", "int64"),)),
           dataset=_dataset(metrics=[(M_SAVED, "SUM(premium)")]))
    check = _check(fake, kinds={M_SAVED: MeasureKind.CURRENCY})
    assert check.status is ChartStatus.NOT_CHECKED, check.reason
    assert repr(M_SAVED) in check.reason and "could not be judged" in check.reason
    assert check.chart_values == check.mnemiq_values == ((5,),)
    record = SavedMetric(dataset_id=DATASET_ID, name=M_SAVED, expression="SUM(premium)", verbose_name=None,
                         d3format=",d", currency=None, certified_by=None, warning_text=None)
    check = _check(fake, kinds={M_SAVED: MeasureKind.CURRENCY}, saved={M_SAVED: record})
    assert check.status is ChartStatus.FORMAT_WRONG
    priced = SavedMetric(dataset_id=DATASET_ID, name=M_SAVED, expression="SUM(premium)", verbose_name=None,
                         d3format=",.2f", currency=dict(USD), certified_by=None, warning_text=None)
    assert _check(fake, kinds={M_SAVED: MeasureKind.CURRENCY}, saved={M_SAVED: priced}).status is ChartStatus.CHECKED


# --- what rc2's renderer shows that its data tool does not ------------------------------------------------------------


def _mixed(**changes) -> dict:
    orders = {"expressionType": "SQL", "sqlExpression": "COUNT(*)", "label": "Orders"}
    return _line(viz_type="mixed_timeseries", metrics_b=[orders], groupby_b=[], y_axis_format_secondary=",d") | changes


@pytest.mark.parametrize("changes, checkable", [
    ({"adhoc_filters_b": []}, False),
    ({"adhoc_filters_b": [copy.deepcopy(WEB_ONLY)]}, True),
    ({"adhoc_filters_b": [], "adhoc_filters": [copy.deepcopy(NO_TIME_FILTER)]}, True),
    ({"adhoc_filters_b": [WEB_ONLY | {"comparator": ["store"]}]}, False),
    ({"row_limit_b": 10}, False),
    ({"row_limit_b": 100}, True),
    ({"groupby": ["region"], "groupby_b": ["region"], "limit_b": 5}, False),
], ids=["B's filters emptied", "B's filters the primary's", "both effectively none", "B filtered otherwise",
        "B limited otherwise", "B limited alike", "B's series limited"])
def test_mixed_chart_reads_its_second_series_as_the_renderer_does(changes, checkable):
    """rc2's renderer takes each `_b` key the form holds in place of the primary's (`retainFormDataSuffix` in
    `plugin-chart-echarts/src/utils/formDataSuffix.ts`), an empty `adhoc_filters_b` included, while its data tool
    reads an empty one as absent and filters the second series with the primary's filters. Only a second series the
    renderer shows as Superset's rows hold it is checked."""
    found = expectation_from_form_data(_mixed(**changes), dataset=PHYSICAL)
    assert isinstance(found, Expectation) is checkable, found
    if not checkable:
        assert "second series" in found.reason


@pytest.mark.parametrize("changes, checkable, said", [
    ({"showValuesAs": "percent_row"}, False, "share"),
    ({"showValuesAs": "percent_total"}, False, "share"),
    ({"showValuesAs": "actual"}, True, ""),
    ({}, True, ""),
    ({"rowTotals": True}, False, "totals"),
    ({"colSubTotals": True}, False, "totals"),
    ({"rowTotals": True, "colTotals": True, "metrics": [{"expressionType": "SIMPLE", "aggregate": "SUM",
                                                         "column": {"column_name": "premium"}, "label": "P"}]},
     True, ""),
    ({"aggregateFunction": "Count"}, True, ""),
], ids=["% of row", "% of grand total", "actual values", "no setting", "SQL metric with totals",
        "SQL metric with subtotals", "additive metrics with totals", "an aggregateFunction rc2 never reads"])
def test_pivot_reads_its_values_and_totals_as_the_plugin_does(changes, checkable, said):
    """rc2's pivot shows a cell as a share of a total when `showValuesAs` asks (`plugin-chart-pivot-table`'s control
    panel), with no post-processing, and with a metric that is not additive asks for its totals in the same query as
    grouping sets (`buildQuery.ts`, `allMetricsAdditive`), whose total rows carry null keys. It never reads
    `aggregateFunction`."""
    pivot = {"viz_type": "pivot_table_v2", "metrics": [copy.deepcopy(PREMIUM)], "groupbyRows": ["region"],
             "groupbyColumns": ["channel"], "adhoc_filters": [], "row_limit": 100} | changes
    found = expectation_from_form_data(pivot, dataset=PHYSICAL)
    assert isinstance(found, Expectation) is checkable, found
    if not checkable:
        assert said in found.reason


@pytest.mark.parametrize("viz", ["pie", "funnel", "gauge_chart", "treemap_v2"])
def test_a_leftover_x_axis_among_the_grouping_is_grouped_by_the_grain(viz):
    """rc2's `buildQueryContext` runs `normalizeTimeColumn` for every chart, so a grouping column that is also the
    form's `x_axis` is shown by the time grain, which the data tool's rebuild never applies."""
    form = {"viz_type": viz, "metric": copy.deepcopy(PREMIUM), "groupby": ["order_date"], "x_axis": "order_date",
            "adhoc_filters": [], "row_limit": 100}
    found = expectation_from_form_data(form | {"time_grain_sqla": "P1M"}, dataset=PHYSICAL)
    assert isinstance(found, NotCheckable) and "time grain" in found.reason
    assert isinstance(expectation_from_form_data(form, dataset=PHYSICAL), Expectation)


def _days(count: int) -> tuple:
    start = datetime(2023, 1, 1)
    return tuple(((start + timedelta(days=i)).isoformat(), 1) for i in range(count))


@pytest.mark.parametrize("chart", [
    NO_DATA,
    _chart_data(["order_date", "Total premium"], ("2026-01-01T00:00:00", 31), ("2026-02-01T00:00:00", 12)),
], ids=["an empty chart", "a chart whose months are off"])
def test_the_grain_is_blamed_only_on_positive_evidence(fake, chart):
    """The ungrouped statement at its row cap proves nothing about the chart's rows: the chart is reported as the
    grouped comparison found it, never as "the grain was not applied"."""
    _serve(fake, form_data=_line(), chart=chart,
           mnemiq=lambda sql: _result(*(MONTHS if "DATE_TRUNC" in sql.upper() else _days(1000)), columns=TIME))
    check = _check(fake)
    assert check.status is ChartStatus.MISMATCH, check.reason
    assert "time grain" not in check.reason


def test_the_reason_names_what_the_chart_was_compared_with(fake):
    """A virtual dataset's saved metric is compared with Superset's own query of the dataset, not with mnemiq's SQL,
    and the reason says which."""
    rows = (("east", 1200),)
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], *rows), mnemiq=_result(*rows))
    assert "mnemiq's own SQL" in _check(fake).reason
    fake.calls.clear()
    _serve(fake, form_data=_bar(metrics=[M_SAVED]), chart=_chart_data(["region", M_SAVED], *rows),
           dataset=_dataset(virtual=True, metrics=[(M_SAVED, "SUM(premium)")]),
           query={"dataset_id": DATASET_ID, "dataset_name": "policy", "columns": [{"name": "region"},
                                                                                 {"name": M_SAVED}],
                  "data": [{"region": "east", M_SAVED: 1200}], "row_count": 1, "total_rows": 1})
    check = _check(fake, kinds={})
    assert check.status is ChartStatus.CHECKED
    assert "query_dataset" in check.reason and "mnemiq's own SQL" not in check.reason


def test_only_a_postgres_adapter_checks_with_its_own_sql(fake):
    """mnemiq's statement, its grain expression above all, is written for Postgres, Superset mode's one dialect."""
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], ("east", 1)),
           mnemiq=_result(("east", 1)))
    adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken(TOKEN)), database_id=DATABASE_ID,
                                 schema=SCHEMA, dialect="duckdb")
    check = check_chart(adapter, chart_id=CHART_ID, dataset_id=DATASET_ID, kinds={}, saved={},
                        snapshot=SNAPSHOT, grants=GRANTS, guard_fanout=False)
    assert check.status is ChartStatus.NOT_CHECKED and "Postgres" in check.reason
    assert "execute_sql" not in _tools(fake)


def test_mnemiqs_own_rows_that_repeat_a_group_are_not_checked(fake):
    """Two of mnemiq's groups that read as one after their timestamps are made comparable are never collapsed."""
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], ("2026-01-01", 5)),
           mnemiq=_result(("2026-01-01 00:00:00", 7), ("2026-01-01", 5)))
    check = _check(fake, kinds={})
    assert check.status is ChartStatus.NOT_CHECKED and "repeat" in check.reason


def test_the_cap_named_is_the_one_applied(fake, monkeypatch):
    """The reason quotes the row cap the decider put on mnemiq's statement."""
    monkeypatch.setattr(chart_check, "MAX_ROWS", 7)
    _serve(fake, form_data=_line(time_grain_sqla=None), chart=_chart_data(["order_date", "Total premium"],
                                                                          ("2023-01-01T00:00:00", 1)),
           mnemiq=_result(*_days(1000), columns=TIME))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "1,000" in check.reason and "(7)" not in check.reason


# --- numbers the renderer computes from the query's rows --------------------------------------------------------------

SUM_PREMIUM = {"expressionType": "SIMPLE", "aggregate": "SUM", "column": {"column_name": "premium"}, "label": "P"}
COUNT_PREMIUM = SUM_PREMIUM | {"aggregate": "COUNT"}
MIN_PREMIUM = SUM_PREMIUM | {"aggregate": "MIN"}


def _grouped(viz: str, **changes) -> dict:
    return {"viz_type": viz, "metric": copy.deepcopy(PREMIUM), "groupby": ["region"], "adhoc_filters": [],
            "row_limit": 100} | changes


TWO = ["region", "channel"]


@pytest.mark.parametrize("form, said", [
    (_grouped("treemap_v2", groupby=TWO), "sum"),
    (_grouped("treemap_v2", groupby=TWO, metric=MIN_PREMIUM), "sum"),
    (_grouped("treemap_v2", groupby=TWO, metric=M_SAVED), "sum"),
    (_grouped("treemap_v2", groupby=TWO, metric=SUM_PREMIUM), None),
    (_grouped("treemap_v2", groupby=TWO, metric=COUNT_PREMIUM), None),
    (_grouped("treemap_v2"), None),
    (_grouped("pie", threshold_for_other=5, metric=SUM_PREMIUM), "Other"),
    (_grouped("pie", threshold_for_other=0), None),
    (_grouped("pie", show_total=True), "sum"),
    (_grouped("pie", show_total=True, metric=MIN_PREMIUM), "sum"),
    (_grouped("pie", show_total=True, metric=SUM_PREMIUM), None),
    (_grouped("pie", show_total=True, metric=COUNT_PREMIUM), None),
    (_grouped("pie", show_total=False), None),
], ids=["treemap parents of a SQL metric", "treemap parents of a MIN", "treemap parents of a saved metric",
        "treemap parents of a SUM", "treemap parents of a COUNT", "a treemap of one level",
        "a pie's Other slice, even of a SUM", "a pie with no Other slice", "a pie's total of a SQL metric",
        "a pie's total of a MIN", "a pie's total of a SUM", "a pie's total of a COUNT", "a pie with no total"])
def test_a_number_the_renderer_sums_is_checked_only_when_the_sum_is_the_measures_own(form, said):
    """rc2's renderer sums some of what a chart shows from the query's rows, so the data tool's rows never hold it: a
    treemap's parent from its children (`treeBuilder` in `plugin-chart-echarts/src/utils/treeBuilder.ts`), a pie's
    Other slice from the slices under its threshold and its total from every slice (`Pie/transformProps.ts`). A sum of
    a SIMPLE SUM or COUNT is that measure over the groups summed; a sum of anything else is not, and the Other slice
    hides the slices it folds in."""
    found = expectation_from_form_data(form, dataset=PHYSICAL)
    if said is None:
        assert isinstance(found, Expectation), found
    else:
        assert isinstance(found, NotCheckable) and said in found.reason, found


def test_a_treemaps_parents_of_a_written_measure_are_not_checked(fake):
    """Nothing is asked for the chart's data or computed: a parent's number is not in either side's rows."""
    form = _grouped("treemap_v2", groupby=TWO, number_format=",.2f", currency_format=dict(USD))
    _serve(fake, form_data=form, chart=_chart_data(["region", "channel", "Total premium"], ("east", "web", 1)),
           mnemiq=_result(("east", "web", 1), columns=(("region", "object"), ("channel", "object"), ("m0", "int64"))))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "sum" in check.reason
    assert "get_chart_data" not in _tools(fake) and _statements(fake) == []


def _table(**changes) -> dict:
    return {"viz_type": "table", "query_mode": "aggregate", "groupby": ["region"],
            "metrics": [copy.deepcopy(PREMIUM)], "adhoc_filters": [copy.deepcopy(WEB_ONLY)], "row_limit": 100,
            "column_config": {"Total premium": {"d3NumberFormat": ",.2f", "currencyFormat": dict(USD)}}} | changes


ROWS = (("east", 1200), ("west", 800))
REBUILT = _chart_data(["region", "Total premium"], *ROWS)
# The table's saved query context answers its totals query too: one row, the metrics over no grouping.
SAVED_CONTEXT = _chart_data(["region", "Total premium"], *ROWS, query_results=[
    {"query_index": 0, "columns": ["region", "Total premium"],
     "data": [{"region": r, "Total premium": v} for r, v in ROWS], "row_count": 2, "total_rows": 2},
    {"query_index": 1, "columns": ["Total premium"], "data": [{"Total premium": 2000}], "row_count": 1,
     "total_rows": 1}])


@pytest.mark.parametrize("answer", [REBUILT, SAVED_CONTEXT], ids=["rebuilt from form_data", "its saved query context"])
def test_a_tables_totals_row_is_not_checked(fake, answer):
    """rc2's table asks for its totals row in a query of its own (`show_totals` in `plugin-chart-table/src/buildQuery.ts`,
    with `getTotalsMetrics`), which the data tool's rebuild leaves out and mnemiq does not reproduce: the table is not
    checked, however the data tool answers."""
    _serve(fake, form_data=_table(show_totals=True), chart=answer, mnemiq=_result(*ROWS))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "total" in check.reason
    assert "get_chart_data" not in _tools(fake) and _statements(fake) == []


def test_a_table_without_totals_is_checked(fake):
    _serve(fake, form_data=_table(show_totals=False), chart=REBUILT, mnemiq=_result(*ROWS))
    check = _check(fake)
    assert check.status is ChartStatus.CHECKED, check.reason


# --- a format mnemiq cannot judge, and the client the check reads through -----------------------------------------------


@pytest.mark.parametrize("form", [
    _bar(currency_format={"symbol": "AUTO", "symbolPosition": "prefix"}),
    _table(column_config={"Total premium": {"d3NumberFormat": ",.2f", "d3SmallNumberFormat": ",.4f",
                                            "currencyFormat": dict(USD)}}),
], ids=["an AUTO currency", "a table's small-number format"])
def test_a_format_mnemiq_cannot_judge_is_not_reported_checked(fake, form):
    """rc2 decides these formats from what mnemiq does not read (the data's own currency, each value's size), so the
    chart is not reported checked: its numbers are kept, and the reason names the metric. The same chart is checked for
    a metric mnemiq knows no kind of, and its numbers are a mismatch whatever its format."""
    _serve(fake, form_data=form, chart=_chart_data(["region", "Total premium"], *ROWS), mnemiq=_result(*ROWS))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED, check.reason
    assert "'Total premium'" in check.reason and "could not be judged" in check.reason
    assert check.chart_values == check.mnemiq_values == ROWS
    assert _check(fake, kinds={}).status is ChartStatus.CHECKED
    _serve(fake, form_data=form, chart=_chart_data(["region", "Total premium"], *ROWS), mnemiq=_result(*ROWS[:1]))
    assert _check(fake).status is ChartStatus.MISMATCH


def test_superset_is_read_only_through_the_adapters_own_client(fake):
    """The check reads Superset as the caller the adapter acts for, through `adapter.client`: no other client can be
    passed, so the chart and mnemiq's own statement are read by the same caller."""
    assert "client" not in inspect.signature(check_chart).parameters
    _serve(fake, form_data=_bar(), chart=_chart_data(["region", "Total premium"], *ROWS), mnemiq=_result(*ROWS))
    adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok-other")), database_id=DATABASE_ID,
                                 schema=SCHEMA)
    check = check_chart(adapter, chart_id=CHART_ID, dataset_id=DATASET_ID, kinds=TOTAL["kinds"], saved={},
                        snapshot=SNAPSHOT, grants=GRANTS, guard_fanout=False)
    assert check.status is ChartStatus.CHECKED, check.reason
    assert {call.authorization for call in fake.calls} == {"Bearer tok-other"}


# --- which metric a one-metric chart shows ------------------------------------------------------------------------------


@pytest.mark.parametrize("viz", ["pie", "gauge_chart", "treemap_v2", "funnel", "big_number_total"])
def test_a_one_metric_chart_is_read_by_the_metric_its_renderer_shows(viz):
    """Each one-metric plugin's `transformProps` shows `formData.metric` and nothing else (`Pie/transformProps.ts`,
    `Gauge/`, `Treemap/`, `Funnel/`, `BigNumber/BigNumberTotal/`). rc2's data tool reads `metric` first only for the
    big numbers (`resolve_metrics_and_groupby` in `superset/mcp_service/chart/chart_helpers.py`) and `metrics` first
    for the rest, but a `metrics` list is not what the chart shows: without a `metric` there is nothing to compare."""
    only_list = {"viz_type": viz, "metrics": [copy.deepcopy(PREMIUM)], "groupby": ["region"], "adhoc_filters": []}
    found = expectation_from_form_data(only_list, dataset=PHYSICAL)
    assert isinstance(found, NotCheckable) and "no metric" in found.reason, found
    shown = only_list | {"metric": copy.deepcopy(PREMIUM), "metrics": [M_SAVED]}
    found = expectation_from_form_data(shown, dataset=PHYSICAL)
    assert isinstance(found, Expectation) and found.metrics == (("Total premium", "SUM(premium)"),), found


def test_a_pies_metric_the_data_tool_does_not_return_is_not_checked(fake):
    """A pie holding a `metrics` list beside its `metric`: rc2's data tool rebuilds its query from the list, so its rows
    do not hold the metric the pie shows, and nothing is reported checked."""
    form = _grouped("pie", metrics=[M_SAVED], number_format=",.2f", currency_format=dict(USD))
    _serve(fake, form_data=form, chart=_chart_data(["region", M_SAVED], ("east", 1)), mnemiq=_result(("east", 1)),
           dataset=_dataset(metrics=[(M_SAVED, "SUM(premium)")]))
    check = _check(fake)
    assert check.status is ChartStatus.NOT_CHECKED and "columns" in check.reason, check.reason


def _waterfall(**changes) -> dict:
    """A waterfall's form_data as rc2's MCP `map_waterfall_config` saves it: `show_total` written, on by default."""
    return {"viz_type": "waterfall", "x_axis": "region", "groupby": [], "metric": copy.deepcopy(PREMIUM),
            "show_total": True, "adhoc_filters": [], "row_limit": 100, "y_axis_format": ",.2f"} | changes


@pytest.mark.parametrize("form, checkable", [
    (_waterfall(), False),
    (_waterfall(show_total=False), False),
    (_waterfall(groupby=["channel"]), False),
    (_waterfall(metric=MIN_PREMIUM), False),
    (_waterfall(metric=M_SAVED), False),
    (_waterfall(metric=SUM_PREMIUM), True),
    (_waterfall(metric=COUNT_PREMIUM, show_total=False), True),
    (_waterfall(metric=SUM_PREMIUM, groupby=["channel"]), True),
], ids=["total bar of a SQL metric", "running total of a SQL metric, no total bar", "per-period totals of a SQL metric",
        "totals of a MIN", "totals of a saved metric", "totals of a SUM", "running total of a COUNT",
        "per-period totals of a SUM"])
def test_a_waterfalls_totals_are_checked_only_when_the_sum_is_the_measures_own(form, checkable):
    """rc2's waterfall sums what it shows beyond each step (`Waterfall/transformProps.ts`): `transformer` adds a total
    bar per period and one at the end when `showTotal`, and every bar's tooltip shows `totalSum`, the running sum of
    the steps so far, whether or not a total bar shows. A sum of a SIMPLE SUM or COUNT is that measure's own."""
    found = expectation_from_form_data(form, dataset=PHYSICAL)
    assert isinstance(found, Expectation) is checkable, found
    if not checkable:
        assert "sum" in found.reason
