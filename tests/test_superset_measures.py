"""Measures: what makes an expression one, how its number is shown, and that its statement meets the deciders every
ask-path statement meets. Run against `FakeSuperset` where the deciders need an adapter."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import Column, Snapshot
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import DelegatedToken, SupersetMcp
from mnemiq.superset.measures import (
    Aggregation,
    Measure,
    MeasureKind,
    MeasureRefused,
    aggregates,
    check_measure,
    display_format,
    measure_statement,
    measure_value,
)
from mnemiq.sql.verdict import Approved
from tests.superset_fake import FakeSuperset

EMPTY = {"success": True, "columns": [], "rows": []}
_PROBE = re.compile(r"AS mnemiq_probe WHERE 1 = 0$")
SNAPSHOT = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id="policy.region", object_id="policy", name="region", data_type="VARCHAR"),
    Column(id="policy.premium", object_id="policy", name="premium", data_type="INTEGER"),
    Column(id="claim.amount", object_id="claim", name="amount", data_type="INTEGER"),
    # Profiled counts the fan-out guard reads: policy_id repeats in both fact tables.
    Column(id="fact_claim.policy_id", object_id="fact_claim", name="policy_id", row_count=30000, distinct_count=23322,
           null_count=0),
    Column(id="fact_claim.paid", object_id="fact_claim", name="paid", row_count=30000, distinct_count=29000,
           null_count=0),
    Column(id="fact_premium.policy_id", object_id="fact_premium", name="policy_id", row_count=125000,
           distinct_count=50000, null_count=0),
])
GRANTS = GrantSet(frozenset({"policy", "fact_claim", "fact_premium"}))


@pytest.fixture
def adapter():
    with FakeSuperset() as fake:
        fake.reply("execute_sql", lambda request, authorization: EMPTY)
        yield SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=7, schema="public")


def _measure(**changes) -> Measure:
    fields = dict(name="total_premium", label="Total premium", expression="SUM(premium)", kind=MeasureKind.CURRENCY,
                  dataset_id=1, description="premium written", asked_by="u1", currency=None)
    return Measure(**{**fields, **changes})


def _check(adapter, measure: Measure, *, source: str = "policy", dimension_names=(), grants=GRANTS, guard=False):
    return check_measure(measure, source=source, dimension_names=dimension_names, adapter=adapter,
                         snapshot=SNAPSHOT, grants=grants, guard_fanout=guard)


def test_sum_aggregates():
    assert aggregates("SUM(x)") is Aggregation.AGGREGATES
    assert aggregates("COUNT(DISTINCT x) * 2") is Aggregation.AGGREGATES


def test_window_alone_is_row_level():
    assert aggregates("SUM(x) OVER ()") is Aggregation.ROW_LEVEL


def test_nested_aggregate_in_window_aggregates():
    assert aggregates("SUM(SUM(x)) OVER ()") is Aggregation.AGGREGATES


@pytest.mark.parametrize("expression, expected", [
    ("SUM(x) FILTER (WHERE y > 0) OVER ()", Aggregation.ROW_LEVEL),
    ("COUNT(*) FILTER (WHERE y > 0) OVER (PARTITION BY z)", Aggregation.ROW_LEVEL),
    ("SUM(x) FILTER (WHERE y > 0)", Aggregation.AGGREGATES),
    ("SUM(SUM(x) FILTER (WHERE y > 0)) OVER ()", Aggregation.AGGREGATES),
    ("FIRST_VALUE(x) IGNORE NULLS OVER ()", Aggregation.ROW_LEVEL),
])
def test_a_filtered_aggregate_under_a_window_is_row_level(expression, expected):
    assert aggregates(expression) is expected


def test_bare_column_is_row_level():
    assert aggregates("x") is Aggregation.ROW_LEVEL


def test_unmodelled_function_cannot_tell(adapter):
    assert aggregates("my_agg(x)") is Aggregation.CANNOT_TELL
    refused = _check(adapter, _measure(expression="my_agg(premium)"))
    assert isinstance(refused, MeasureRefused) and "cannot tell whether the function aggregates" in refused.reason


@pytest.mark.parametrize("expression", ["SUM(", "", "SUM(x), 1", "SUM(x) FROM t", "SUM(x); SELECT 1"])
def test_parse_error_cannot_tell(expression):
    assert aggregates(expression) is Aggregation.CANNOT_TELL


@pytest.mark.parametrize("field", ["label", "expression", "description"])
@pytest.mark.parametrize("marker", ["{{", "{%", "{#"])
def test_jinja_is_refused(adapter, field, marker):
    refused = _check(adapter, _measure(**{field: f"SUM(premium) {marker}" if field == "expression" else f"a {marker}"}))
    assert isinstance(refused, MeasureRefused) and marker in refused.reason


def test_label_with_jinja_is_refused(adapter):
    assert isinstance(_check(adapter, _measure(label="{{ x }}")), MeasureRefused)


def test_expression_with_jinja_is_refused(adapter):
    assert isinstance(_check(adapter, _measure(expression="SUM({{ x }})")), MeasureRefused)


def test_description_with_jinja_is_refused(adapter):
    assert isinstance(_check(adapter, _measure(description="{% if %}")), MeasureRefused)


def test_label_colliding_with_a_dimension_is_refused(adapter):
    refused = _check(adapter, _measure(label="Region"), dimension_names=("region",))
    assert isinstance(refused, MeasureRefused) and "dimension" in refused.reason
    assert isinstance(_check(adapter, _measure(label="Region"), dimension_names=("state",)), Approved)


def test_display_formats():
    assert display_format(MeasureKind.CURRENCY, "USD") == (",.2f", {"symbol": "USD", "symbolPosition": "prefix"})
    assert display_format(MeasureKind.PERCENT, None) == (",.1%", None)
    assert display_format(MeasureKind.COUNT, None) == (",d", None)
    assert display_format(MeasureKind.NUMBER, None) == (",.2f", None)


def test_percent_never_carries_a_currency():
    assert display_format(MeasureKind.PERCENT, "USD") == (",.1%", None)
    assert display_format(MeasureKind.COUNT, "USD")[1] is None


def test_currency_must_be_iso(adapter):
    refused = _check(adapter, _measure(currency="$"))
    assert isinstance(refused, MeasureRefused) and "ISO 4217" in refused.reason
    assert isinstance(_check(adapter, _measure(currency="EUR")), Approved)


def test_the_setting_is_an_iso_code(monkeypatch):
    monkeypatch.setenv("MNEMIQ_SUPERSET_CURRENCY", "EUR")
    assert Settings().superset_currency == "EUR"
    monkeypatch.setenv("MNEMIQ_SUPERSET_CURRENCY", "$")
    with pytest.raises(ValueError):
        Settings()


def test_the_statement_names_its_dimensions_and_filters():
    sql = measure_statement(_measure(), "policy", ("region",), ("premium > 0",))
    assert sql == "SELECT region, SUM(premium) AS total_premium FROM policy WHERE (premium > 0) GROUP BY 1"


def test_check_measure_approves_a_granted_measure(adapter):
    assert isinstance(_check(adapter, _measure()), Approved)


def test_check_measure_runs_the_deciders(adapter):
    refused = _check(adapter, _measure(), grants=GrantSet(frozenset({"claim"})))
    assert isinstance(refused, MeasureRefused) and "unauthorized_table" in refused.reason
    fanout = _measure(expression="SUM(c.paid)")
    join = "fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id"
    refused = _check(adapter, fanout, source=join, guard=True)
    assert isinstance(refused, MeasureRefused) and "fan_out" in refused.reason
    assert isinstance(_check(adapter, fanout, source=join, guard=False), Approved)


def test_measure_value_is_a_float_or_none():
    with FakeSuperset() as fake:
        rows = [{"total_premium": 12}]
        fake.reply("execute_sql", lambda request, authorization: EMPTY if request.get("force_refresh") or
                   _PROBE.search(request["sql"]) else {"success": True, "columns": [
                       {"name": "total_premium", "type": "int64"}], "rows": rows})
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=7, schema="public")
        approved = _check(adapter, _measure())
        assert isinstance(approved, Approved)
        assert measure_value(approved, adapter, 30.0) == 12.0
        rows.clear()
        assert measure_value(approved, adapter, 30.0) is None


# The workflow that runs the Superset client on mcp 2.x. Read as text: no YAML parser is a dependency of the `dev` extra.
MCP_2X = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "mcp-2x.yml"


def test_the_mcp_2x_workflow_runs_this_module():
    """These tests drive the deciders through `SupersetMcp` over the wire to `FakeSuperset`, so they run where the
    client meets mcp 2.x too: the workflow's one pytest line names this module."""
    runs = [line for line in MCP_2X.read_text().splitlines() if re.search(r"\brun:.*\bpytest\b", line)]
    assert len(runs) == 1, f"expected one pytest line in {MCP_2X.name}, found {runs}"
    module = f"tests/{Path(__file__).name}"
    assert module in shlex.split(runs[0].split("run:", 1)[1]), f"{MCP_2X.name}'s pytest line does not run {module}"
