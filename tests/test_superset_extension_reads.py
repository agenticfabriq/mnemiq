"""What the extension's read tools (`agenticfabriq.mnemiq.reads`) answer, with Superset's lookups stood in for: each
request is read strictly, an object the caller may not read is left out (or answered as absent) without a name, and
the layout answer is refused over 40 KB; and how mnemiq's client reads `table_rules`'s answer
(`mnemiq.superset.extension.table_rules_apply`). What the tools do inside Superset is the live suite's:
`tests/superset_live/test_extension_tools.py`.
"""

import json
import sys
import types
from datetime import datetime
from types import SimpleNamespace

import pytest

from mnemiq.superset.client import SupersetPermissionDenied, SupersetToolError, SupersetUnavailable
from mnemiq.superset.extension import TABLE_RULES_BATCH, TOOL_PREFIX, table_rules_apply, tables_read
from tests import superset_stubs as stubs
from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import reads, writes  # noqa: E402


def refusal(tool, request):
    with pytest.raises(writes.Refused) as caught:
        reads.read(tool, request)
    return caught.value.answer


def test_every_read_tool_has_a_request_reader():
    assert set(reads.REQUESTS) == {"dataset_metrics", "editors_of", "get_layout", "row_rules", "table_rules",
                                   "tables_read", "session_binding"}


def test_dataset_metrics_takes_at_most_twenty_distinct_ids():
    assert reads.read("dataset_metrics", {"dataset_ids": [3, 3, 1]}) == {"dataset_ids": [3, 1]}
    assert reads.read("dataset_metrics", {"dataset_ids": list(range(1, 21))})["dataset_ids"] == list(range(1, 21))
    assert refusal("dataset_metrics", {"dataset_ids": list(range(1, 22))}) == {"refused": "request",
                                                                                "field": "dataset_ids"}
    for bad in (0, "3", [0], [True], ["3"], None):
        assert refusal("dataset_metrics", {"dataset_ids": bad}) == {"refused": "request", "field": "dataset_ids"}
    assert refusal("dataset_metrics", {}) == {"refused": "request", "field": "dataset_ids"}
    assert refusal("dataset_metrics", {"dataset_ids": [], "ids": []}) == {"refused": "request", "unknown": ["ids"]}


def test_dataset_metrics_answers_an_empty_request_without_touching_superset(monkeypatch):
    """mnemiq's client probes with exactly this request to tell a missing extension from a failing tool: it must
    answer, not fail, and it needs no Superset (none is importable here)."""
    def boom(*_args):
        raise AssertionError("touched Superset")

    monkeypatch.setattr(writes, "_dataset", boom)
    assert reads.dataset_metrics({"dataset_ids": []}) == {"datasets": []}


def test_editors_of_and_get_layout_requests():
    assert reads.read("editors_of", {}) == {"chart_ids": [], "dataset_ids": [], "dashboard_ids": []}
    assert reads.read("editors_of", {"chart_ids": [2, 2], "dataset_ids": [5]}) == {"chart_ids": [2], "dataset_ids": [5],
                                                                                   "dashboard_ids": []}
    assert reads.read("editors_of", {"dashboard_ids": [4, 4, 7]})["dashboard_ids"] == [4, 7]
    assert refusal("editors_of", {"dashboard_ids": list(range(1, 22))}) == {"refused": "request",
                                                                            "field": "dashboard_ids"}
    assert refusal("editors_of", {"chart_ids": [0]}) == {"refused": "request", "field": "chart_ids"}
    assert refusal("editors_of", {"chart_ids": list(range(1, 102))}) == {"refused": "request", "field": "chart_ids"}
    assert refusal("editors_of", {"dataset_ids": list(range(1, 22))}) == {"refused": "request",
                                                                          "field": "dataset_ids"}
    assert reads.read("get_layout", {"dashboard_id": 4}) == {"dashboard_id": 4}
    assert refusal("get_layout", {"dashboard_id": "4"}) == {"refused": "request", "field": "dashboard_id"}
    assert refusal("get_layout", {}) == {"refused": "request", "field": "dashboard_id"}
    assert refusal("get_layout", None) == {"refused": "request"}


def metric(name, *, currency=None, extra=None, d3format=None, warning=None):
    return SimpleNamespace(metric_name=name, expression=f"SUM({name})", d3format=d3format, currency=currency,
                           extra=extra, warning_text=warning)


def test_dataset_metrics_answers_each_reachable_dataset_and_leaves_out_the_rest(monkeypatch):
    certified = json.dumps({"certification": {"certified_by": "Finance", "details": "audited"}})
    held = {1: SimpleNamespace(id=1, metrics=[
        metric("revenue", currency={"symbol": "USD", "symbolPosition": "prefix"}, extra=certified, d3format=",.2f"),
        metric("plain", extra=None, warning="check me"), metric("odd", extra="not json"),
        metric("shaped", extra=json.dumps({"certification": None}))]),
            3: SimpleNamespace(id=3, metrics=[])}
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: held.get(dataset_id))
    answer = reads.dataset_metrics({"dataset_ids": [3, 2, 1]})
    assert [d["dataset_id"] for d in answer["datasets"]] == [3, 1]
    assert "2" not in json.dumps(answer["datasets"][0])
    revenue, plain, odd, shaped = answer["datasets"][1]["metrics"]
    assert revenue == {"metric_name": "revenue", "expression": "SUM(revenue)", "d3format": ",.2f",
                       "currency": {"symbol": "USD", "symbolPosition": "prefix"}, "certified_by": "Finance",
                       "certification_details": "audited", "warning_text": None}
    assert plain["warning_text"] == "check me" and plain["certified_by"] is None
    assert odd["certified_by"] is None and shaped["certified_by"] is None and shaped["certification_details"] is None
    assert set(plain) == set(revenue)


def test_a_currency_that_is_not_an_object_is_answered_as_none(monkeypatch):
    held = SimpleNamespace(id=1, metrics=[metric("a", currency='{"symbol": "USD"}')])
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: held)
    assert reads.dataset_metrics({"dataset_ids": [1]})["datasets"][0]["metrics"][0]["currency"] is None


def test_editors_of_names_editors_only_of_readable_objects(monkeypatch):
    def subject(label):
        return SimpleNamespace(label=label)

    charts = {2: SimpleNamespace(editors=[subject("zed"), subject("amy"), subject("amy")]),
              3: SimpleNamespace(editors=[subject("zed")])}
    datasets = {5: SimpleNamespace(editors=[subject("amy")])}
    monkeypatch.setattr(reads, "_chart", lambda chart_id: charts.get(chart_id))
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: datasets.get(dataset_id))
    # The caller edits chart 2 and dataset 5, and not chart 3: each entry says so, beside the editors' labels.
    monkeypatch.setattr(writes, "_security", lambda: SimpleNamespace(
        is_editor=lambda resource: resource is charts[2] or resource is datasets[5]))
    answer = reads.editors_of({"chart_ids": [9, 2, 3], "dataset_ids": [5, 6]})
    assert answer == {"charts": [{"chart_id": 2, "is_editor": True, "editors": ["amy", "zed"]},
                                 {"chart_id": 3, "is_editor": False, "editors": ["zed"]}],
                      "datasets": [{"dataset_id": 5, "is_editor": True, "editors": ["amy"]}]}
    assert reads.editors_of({"chart_ids": [], "dataset_ids": []}) == {"charts": [], "datasets": []}


def dashboard(position_json, changed_on=datetime(2026, 11, 14, 1, 2, 3, 456789)):
    return SimpleNamespace(id=4, position_json=position_json, changed_on=changed_on)


def test_get_layout_answers_the_raw_text_and_the_instant(monkeypatch):
    text = '{"ROOT_ID": {"type": "ROOT"}}'
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: dashboard(text))
    assert reads.get_layout({"dashboard_id": 4}) == {"dashboard_id": 4, "position_json": text,
                                                     "changed_on": "2026-11-14T01:02:03.456789"}


def test_get_layout_of_an_unreadable_or_missing_dashboard_is_not_found(monkeypatch):
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: None)
    assert reads.get_layout({"dashboard_id": 4}) == {"not_found": "dashboard"}


def test_get_layout_refuses_an_answer_over_forty_kilobytes(monkeypatch):
    just_fits = dashboard("x" * (reads.MAX_LAYOUT_BYTES - 200))
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: just_fits)
    assert "position_json" in reads.get_layout({"dashboard_id": 4})
    big = dashboard("x" * reads.MAX_LAYOUT_BYTES)
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: big)
    assert reads.get_layout({"dashboard_id": 4}) == {"too_large": True}
    multibyte = dashboard("é" * (reads.MAX_LAYOUT_BYTES // 2))  # two bytes each: the size is bytes, not characters
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: multibyte)
    assert reads.get_layout({"dashboard_id": 4}) == {"too_large": True}


def test_a_bad_request_is_answered_not_raised():
    assert reads.dataset_metrics({"dataset_ids": [0]}) == {"refused": "request", "field": "dataset_ids"}
    assert reads.get_layout("nope") == {"refused": "request"}


# ---------------------------------------------------------------- the size cap, one for all three tools

def size(answer):
    return len(json.dumps(answer).encode())


def test_dataset_metrics_marks_the_datasets_that_do_not_fit(monkeypatch):
    """Superset's size guard raises (with its own text) on an over-limit extension answer, so the tool keeps the
    envelope and answers the datasets that do not fit, in request order, as `{dataset_id, too_large}`."""
    fat = "x" * 15_000
    held = {i: SimpleNamespace(id=i, metrics=[metric(f"m{i}", warning=fat)]) for i in (1, 2, 3, 4)}
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: held.get(dataset_id))
    answer = reads.dataset_metrics({"dataset_ids": [1, 2, 3, 4]})
    assert set(answer) == {"datasets"}
    assert [d["dataset_id"] for d in answer["datasets"]] == [1, 2, 3, 4]
    assert ["metrics" in d for d in answer["datasets"]] == [True, True, False, False]
    assert answer["datasets"][2] == {"dataset_id": 3, "too_large": True}
    assert size(answer) <= reads.MAX_ANSWER_BYTES


def test_one_dataset_alone_over_the_cap_is_marked(monkeypatch):
    held = {1: SimpleNamespace(id=1, metrics=[metric("m", warning="x" * 50_000)]),
            2: SimpleNamespace(id=2, metrics=[metric("small")])}
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: held.get(dataset_id))
    answer = reads.dataset_metrics({"dataset_ids": [1]})
    assert answer == {"datasets": [{"dataset_id": 1, "too_large": True}]}
    # Order is kept and the rest are marked once one does not fit, so a later small one is not answered out of turn.
    both = reads.dataset_metrics({"dataset_ids": [1, 2]})
    assert both == {"datasets": [{"dataset_id": 1, "too_large": True}, {"dataset_id": 2, "too_large": True}]}


def test_editors_of_marks_the_entries_that_do_not_fit(monkeypatch):
    def subject(label):
        return SimpleNamespace(label=label)

    fat = "x" * 15_000
    charts = {i: SimpleNamespace(editors=[subject(fat + str(i))]) for i in (1, 2, 3)}
    datasets = {5: SimpleNamespace(editors=[subject("amy")])}
    monkeypatch.setattr(reads, "_chart", lambda chart_id: charts.get(chart_id))
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: datasets.get(dataset_id))
    monkeypatch.setattr(writes, "_security", lambda: SimpleNamespace(is_editor=lambda resource: True))
    answer = reads.editors_of({"chart_ids": [1, 2, 3], "dataset_ids": [5]})
    assert set(answer) == {"charts", "datasets"}
    assert [("editors" in c) for c in answer["charts"]] == [True, True, False]
    # An entry cut for size still says whether the caller edits it: only its labels are left out.
    assert answer["charts"][2] == {"chart_id": 3, "is_editor": True, "too_large": True}
    assert answer["datasets"] == [{"dataset_id": 5, "is_editor": True, "too_large": True}]
    assert size(answer) <= reads.MAX_ANSWER_BYTES


def test_the_three_tools_share_one_cap():
    assert reads.MAX_ANSWER_BYTES == reads.MAX_LAYOUT_BYTES == 40_000


def test_an_empty_currency_is_answered_as_none():
    assert reads.metric_answer(metric("a", currency={}))["currency"] is None


def test_the_marks_are_counted_in_the_cap_at_the_request_limits(monkeypatch):
    """The worst case: the request caps (20 datasets; 100 charts and 20 datasets) with real entries that just fill the
    cap, so the marks of the entries that do not fit are what tips the answer over unless they are counted too."""
    for fat in range(1_000, 2_200, 37):
        held = {i: SimpleNamespace(id=i, metrics=[metric("m", warning="x" * fat)]) for i in range(1, 21)}
        monkeypatch.setattr(writes, "_dataset", lambda dataset_id: held.get(dataset_id))
        answer = reads.dataset_metrics({"dataset_ids": list(range(1, 21))})
        assert size(answer) <= reads.MAX_ANSWER_BYTES, (fat, size(answer))
    subject = SimpleNamespace
    for fat in range(150, 420, 7):
        charts = {i: subject(editors=[subject(label="x" * fat)]) for i in range(1, 101)}
        datasets = {i: subject(editors=[subject(label="x" * fat)]) for i in range(1, 21)}
        monkeypatch.setattr(reads, "_chart", lambda chart_id: charts.get(chart_id))
        monkeypatch.setattr(writes, "_dataset", lambda dataset_id: datasets.get(dataset_id))
        monkeypatch.setattr(writes, "_security", lambda: SimpleNamespace(is_editor=lambda resource: False))
        answer = reads.editors_of({"chart_ids": list(range(1, 101)), "dataset_ids": list(range(1, 21))})
        assert size(answer) <= reads.MAX_ANSWER_BYTES, (fat, size(answer))
        assert len(answer["charts"]) == 100 and len(answer["datasets"]) == 20


def test_editors_of_answers_whether_the_caller_edits_each_readable_dashboard(monkeypatch):
    """Asked for dashboards, `editors_of` answers each one the caller may read with whether the caller edits it and
    its editors' labels, so a build refuses a non-editor before it makes anything. One the caller may not read, or
    that does not exist, is left out. Not asked for dashboards, the answer is as it was."""
    def subject(label):
        return SimpleNamespace(label=label)

    dashboards = {7: SimpleNamespace(editors=[subject("zed"), subject("amy")]),
                  8: SimpleNamespace(editors=[subject("amy")])}
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: dashboards.get(dashboard_id))
    monkeypatch.setattr(writes, "_security", lambda: SimpleNamespace(is_editor=lambda d: d is dashboards[7]))
    answer = reads.editors_of({"dashboard_ids": [9, 7, 8]})
    assert answer == {"charts": [], "datasets": [], "dashboards": [
        {"dashboard_id": 7, "is_editor": True, "editors": ["amy", "zed"]},
        {"dashboard_id": 8, "is_editor": False, "editors": ["amy"]}]}
    monkeypatch.setattr(reads, "_chart", lambda chart_id: None)
    assert reads.editors_of({"chart_ids": [1]}) == {"charts": [], "datasets": []}


def test_editors_of_marks_a_dashboard_that_does_not_fit(monkeypatch):
    fat = "x" * 45_000
    dashboards = {7: SimpleNamespace(editors=[SimpleNamespace(label=fat)])}
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: dashboards.get(dashboard_id))
    monkeypatch.setattr(writes, "_security", lambda: SimpleNamespace(is_editor=lambda d: True))
    answer = reads.editors_of({"dashboard_ids": [7]})
    assert answer["dashboards"] == [{"dashboard_id": 7, "is_editor": True, "too_large": True}]
    assert size(answer) <= reads.MAX_ANSWER_BYTES


def test_a_dashboard_cut_for_size_still_says_whether_the_caller_edits_it(monkeypatch):
    """Only the labels are cut: `is_editor`, one bool, stays in the mark, so a build reads an editor whose dashboard
    has too many editors to send as an editor, and a non-editor as one."""
    fat = "x" * 45_000
    dashboards = {7: SimpleNamespace(editors=[SimpleNamespace(label=fat)]),
                  8: SimpleNamespace(editors=[SimpleNamespace(label=fat)])}
    monkeypatch.setattr(writes, "_dashboard", lambda dashboard_id: dashboards.get(dashboard_id))
    monkeypatch.setattr(writes, "_security", lambda: SimpleNamespace(is_editor=lambda d: d is dashboards[7]))
    answer = reads.editors_of({"dashboard_ids": [7, 8]})
    assert answer["dashboards"] == [{"dashboard_id": 7, "is_editor": True, "too_large": True},
                                    {"dashboard_id": 8, "is_editor": False, "too_large": True}]
    assert size(answer) <= reads.MAX_ANSWER_BYTES


def test_row_rules_takes_at_most_twenty_distinct_ids():
    assert reads.read("row_rules", {"dataset_ids": [3, 3, 1]}) == {"dataset_ids": [3, 1]}
    assert refusal("row_rules", {"dataset_ids": list(range(1, 22))}) == {"refused": "request", "field": "dataset_ids"}
    for bad in (0, "3", [0], [True], ["3"], None):
        assert refusal("row_rules", {"dataset_ids": bad}) == {"refused": "request", "field": "dataset_ids"}
    assert refusal("row_rules", {}) == {"refused": "request", "field": "dataset_ids"}
    assert refusal("row_rules", {"dataset_ids": [], "ids": []}) == {"refused": "request", "unknown": ["ids"]}


def rules_on(monkeypatch, held, rules):
    """`held`: the datasets the caller may access, by id; `rules`: the rules that apply to the caller, by dataset id,
    as Superset's `get_rls_filters` returns them (each a clause the answer must never carry)."""
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: held.get(dataset_id))
    asked = []

    def get_rls_filters(table):
        asked.append(table.id)
        return rules.get(table.id, [])

    monkeypatch.setattr(writes, "_security", lambda: SimpleNamespace(get_rls_filters=get_rls_filters))
    return asked


def test_row_rules_answers_whether_any_rule_applies_to_the_caller_on_each_reachable_dataset(monkeypatch):
    held = {1: SimpleNamespace(id=1), 3: SimpleNamespace(id=3), 4: SimpleNamespace(id=4)}
    rule = SimpleNamespace(id=9, clause="region = 'west'", group_key=None)
    asked = rules_on(monkeypatch, held, {1: [rule], 4: [rule, rule]})
    answer = reads.row_rules({"dataset_ids": [3, 2, 1, 4]})
    assert answer == {"datasets": [{"dataset_id": 3, "applies": False}, {"dataset_id": 1, "applies": True},
                                   {"dataset_id": 4, "applies": True}]}
    assert asked == [3, 1, 4]
    assert "west" not in json.dumps(answer) and "9" not in json.dumps(answer)


def test_row_rules_leaves_out_a_dataset_out_of_reach_and_one_that_does_not_exist_alike(monkeypatch):
    rules_on(monkeypatch, {}, {})
    assert reads.row_rules({"dataset_ids": [5, 6]}) == {"datasets": []}


def test_row_rules_answers_an_empty_request_without_touching_superset(monkeypatch):
    def boom(*_args):
        raise AssertionError("touched Superset")

    monkeypatch.setattr(writes, "_dataset", boom)
    monkeypatch.setattr(writes, "_security", boom)
    assert reads.row_rules({"dataset_ids": []}) == {"datasets": []}


def test_a_bad_row_rules_request_is_answered_not_raised():
    assert reads.row_rules({"dataset_ids": "1"}) == {"refused": "request", "field": "dataset_ids"}


# ---------------------------------------------------------------- table_rules: Superset's own row-rule lookup, by table

TABLE_REQUEST = {"database_id": 7, "schema": "public", "tables": ["orders"]}


def test_table_rules_takes_a_database_a_schema_and_at_most_fifty_distinct_names():
    assert reads.read("table_rules", {**TABLE_REQUEST, "tables": ["orders", "orders", "Customers"]}) == {
        "database_id": 7, "schema": "public", "tables": ["orders", "Customers"]}
    names = [f"t{i}" for i in range(reads.MAX_TABLES)]
    assert reads.read("table_rules", {**TABLE_REQUEST, "tables": names})["tables"] == names
    assert reads.read("table_rules", {**TABLE_REQUEST, "tables": ["x" * 250]})["tables"] == ["x" * 250]
    assert reads.read("table_rules", {**TABLE_REQUEST, "tables": []})["tables"] == []
    assert refusal("table_rules", {**TABLE_REQUEST, "tables": [*names, "t50"]}) == {"refused": "request",
                                                                                    "field": "tables"}
    for bad in ("orders", [""], ["  "], [3], [None], [["orders"]], ["x" * 251], None):
        assert refusal("table_rules", {**TABLE_REQUEST, "tables": bad}) == {"refused": "request", "field": "tables"}
    for bad in ("", "  ", 3, None, "x" * 256):
        assert refusal("table_rules", {**TABLE_REQUEST, "schema": bad}) == {"refused": "request", "field": "schema"}
    for bad in (0, "7", True, None):
        assert refusal("table_rules", {**TABLE_REQUEST, "database_id": bad}) == {"refused": "request",
                                                                                 "field": "database_id"}
    for field in TABLE_REQUEST:
        request = {key: value for key, value in TABLE_REQUEST.items() if key != field}
        assert refusal("table_rules", request) == {"refused": "request", "field": field}
    assert refusal("table_rules", {**TABLE_REQUEST, "catalog": "live"}) == {"refused": "request",
                                                                            "unknown": ["catalog"]}


def superset_rules(monkeypatch, *, databases: dict, predicates: dict) -> list:
    """Superset's lookups `table_rules` makes, stood in: `DatabaseDAO.find_by_id` answering the databases the caller
    may see (`databases`, by id), and rc2's `get_predicates_for_table` answering the predicates it finds for the caller
    on a table (`predicates`, by table name; none for any other). Every lookup is recorded, in order, a database's
    with any option it was asked with (none: Superset's own filter of the databases the caller may see applies)."""
    asked: list = []

    def find_by_id(database_id, *args, **options):
        asked.append(("database", database_id, *args, *options.items()))
        return databases.get(database_id)

    def get_predicates_for_table(table, database, default_catalog):
        asked.append(("predicates", table, database, default_catalog))
        return list(predicates.get(table.table, []))

    modules = {"superset": {}, "superset.daos": {}, "superset.daos.database": {"DatabaseDAO": SimpleNamespace(
        find_by_id=find_by_id)}, "superset.sql": {}, "superset.sql.parse": {"Table": stubs.Table},
        "superset.utils": {}, "superset.utils.rls": {"get_predicates_for_table": get_predicates_for_table}}
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return asked


def database(catalog="live"):
    return SimpleNamespace(id=7, get_default_catalog=lambda: catalog)


def test_table_rules_asks_supersets_own_lookup_for_each_table_as_a_statement_reads_it(monkeypatch):
    """Each table is looked up as rc2's `apply_rls` looks up a table a statement reads with no catalog of its own:
    `get_predicates_for_table(Table(name, schema, <the database's default catalog>), database, <that catalog>)`, as
    the caller. A table applies when the lookup finds any predicate; the answer says only that, in the order asked."""
    found = database()
    asked = superset_rules(monkeypatch, databases={7: found},
                           predicates={"orders": ["(region = 'west')"], "Customers": ["(1 = 0)", "(2 = 2)"]})
    answer = reads.table_rules({"database_id": 7, "schema": "public", "tables": ["orders", "free", "Customers"]})
    assert answer == {"tables": [{"table": "orders", "applies": True}, {"table": "free", "applies": False},
                                 {"table": "Customers", "applies": True}]}
    assert asked == [("database", 7), *(("predicates", stubs.Table(name, "public", "live"), found, "live")
                                        for name in ("orders", "free", "Customers"))]
    assert "west" not in json.dumps(answer) and "1 = 0" not in json.dumps(answer)
    # A database without catalogs has none to qualify a table with, as rc2 has none.
    asked = superset_rules(monkeypatch, databases={7: database(None)}, predicates={})
    assert reads.table_rules(TABLE_REQUEST) == {"tables": [{"table": "orders", "applies": False}]}
    assert asked[1][1:] == (stubs.Table("orders", "public", None), asked[1][2], None)


def test_table_rules_of_a_database_the_caller_cannot_see_is_not_found(monkeypatch):
    """A database the caller may not see (Superset's own `DatabaseDAO` filter, as `get_database_info` reads one) and
    one that does not exist are answered alike, and no table is looked up."""
    asked = superset_rules(monkeypatch, databases={}, predicates={"orders": ["(1 = 0)"]})
    assert reads.table_rules(TABLE_REQUEST) == {"not_found": "database"}
    assert asked == [("database", 7)]


def test_table_rules_too_large_to_send_is_refused_before_superset_is_asked(monkeypatch):
    """The answer echoes each name asked, so fifty long names escaped in JSON could pass Superset's size guard: such a
    request is answered `{"too_large": true}`, read from the names alone, before any lookup. Fifty names of 250
    characters that JSON does not escape fit."""
    asked = superset_rules(monkeypatch, databases={7: database()}, predicates={})
    wide = [f"{i:02d}" + "é" * 248 for i in range(reads.MAX_TABLES)]
    assert reads.table_rules({**TABLE_REQUEST, "tables": wide}) == {"too_large": True}
    assert asked == []
    long = [f"{i:02d}" + "x" * 248 for i in range(reads.MAX_TABLES)]
    answer = reads.table_rules({**TABLE_REQUEST, "tables": long})
    assert [entry["table"] for entry in answer["tables"]] == long and size(answer) <= reads.MAX_ANSWER_BYTES


def test_a_bad_table_rules_request_is_answered_not_raised():
    assert reads.table_rules({**TABLE_REQUEST, "tables": "orders"}) == {"refused": "request", "field": "tables"}


# ---------------------------------------------------------------- tables_read: the tables Superset's access check reads

READ_REQUEST = {"database_id": 7, "schema": "public", "sql": 'SELECT o.id FROM orders o JOIN shop2.lines l ON 1 = 1'}


def test_tables_read_takes_a_database_a_schema_and_one_statements_text():
    assert reads.read("tables_read", READ_REQUEST) == READ_REQUEST
    longest = "SELECT 1 FROM t WHERE x = '" + "a" * (reads.MAX_SQL_BYTES - 28) + "'"
    assert len(longest.encode()) == reads.MAX_SQL_BYTES
    assert reads.read("tables_read", {**READ_REQUEST, "sql": longest})["sql"] == longest
    for bad in ("", "   ", 3, None, ["SELECT 1"], longest + "x", longest[:-2] + "é'"):
        assert refusal("tables_read", {**READ_REQUEST, "sql": bad}) == {"refused": "request", "field": "sql"}
    for bad in ("", "  ", 3, None, "x" * 256):
        assert refusal("tables_read", {**READ_REQUEST, "schema": bad}) == {"refused": "request", "field": "schema"}
    for bad in (0, "7", True, None):
        assert refusal("tables_read", {**READ_REQUEST, "database_id": bad}) == {"refused": "request",
                                                                                "field": "database_id"}
    for field in READ_REQUEST:
        request = {key: value for key, value in READ_REQUEST.items() if key != field}
        assert refusal("tables_read", request) == {"refused": "request", "field": field}
    assert refusal("tables_read", {**READ_REQUEST, "catalog": "live"}) == {"refused": "request",
                                                                           "unknown": ["catalog"]}


class ParseFailure(Exception):
    """One of the failures rc2's `process_jinja_sql` raises for a text it cannot read, stood in."""


def superset_reading(monkeypatch, *, databases: dict, tables=(), statements: int = 1, unparseable: bool = False,
                     fails: bool = False) -> list:
    """Superset's lookups `tables_read` makes, stood in: `DatabaseDAO.find_by_id` answering the databases the caller
    may see (`databases`, by id), and rc2's `process_jinja_sql` answering `tables` (rc2 `Table`s, as its access check
    reads them) for a script of `statements` statements, `unparseable` or not, or raising (`fails`). Every lookup is
    recorded, in order."""
    asked: list = []

    def find_by_id(database_id, *args, **options):
        asked.append(("database", database_id, *args, *options.items()))
        return databases.get(database_id)

    def process_jinja_sql(sql, database, template_params=None):
        asked.append(("parse", sql, database, template_params))
        if fails:
            raise ParseFailure("unreadable")
        return SimpleNamespace(tables=set(tables), script=SimpleNamespace(statements=[object()] * statements,
                                                                         has_unparseable_statement=unparseable))

    modules = {"superset": {}, "superset.daos": {}, "superset.daos.database": {"DatabaseDAO": SimpleNamespace(
        find_by_id=find_by_id)}, "superset.sql": {}, "superset.sql.parse": {"Table": stubs.Table,
                                                                            "process_jinja_sql": process_jinja_sql},
        "superset.exceptions": {"SupersetParseError": ParseFailure, "SupersetSecurityException": ParseFailure},
        "jinja2": {}, "jinja2.exceptions": {"TemplateError": ParseFailure}}
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return asked


def test_tables_read_answers_the_tables_supersets_own_access_check_reads(monkeypatch):
    """The tables rc2's access check reads in the text (`process_jinja_sql`, as `raise_for_access` reads a statement,
    with the empty template parameters `execute_sql` renders with), each qualified as the executor qualifies a table
    it applies rules to: the request's schema, and the database's default catalog, where the statement names none of
    its own. Sorted, each once."""
    found = database()
    asked = superset_reading(monkeypatch, databases={7: found},
                             tables=[stubs.Table("orders"), stubs.Table("lines", "shop2"),
                                     stubs.Table("items", "shop3", "other")])
    answer = reads.tables_read(READ_REQUEST)
    assert answer == {"tables": [{"catalog": "live", "schema": "public", "table": "orders"},
                                 {"catalog": "live", "schema": "shop2", "table": "lines"},
                                 {"catalog": "other", "schema": "shop3", "table": "items"}]}
    assert asked == [("database", 7), ("parse", READ_REQUEST["sql"], found, {})]
    superset_reading(monkeypatch, databases={7: database(None)}, tables=[stubs.Table("orders")])
    assert reads.tables_read(READ_REQUEST) == {"tables": [{"catalog": None, "schema": "public", "table": "orders"}]}


def test_tables_read_of_a_database_the_caller_cannot_see_is_not_found(monkeypatch):
    asked = superset_reading(monkeypatch, databases={}, tables=[stubs.Table("orders")])
    assert reads.tables_read(READ_REQUEST) == {"not_found": "database"}
    assert asked == [("database", 7)]


@pytest.mark.parametrize("reading", [{"fails": True}, {"statements": 2}, {"statements": 0}, {"unparseable": True}],
                         ids=["unparsed", "two statements", "none", "a command"])
def test_tables_read_of_a_text_superset_cannot_read_as_one_statement_says_so(monkeypatch, reading):
    """A text rc2 cannot parse, or reads as other than one statement, or as one it does not fully model (which its
    access check refuses under strict scoping), is answered `{"unreadable": true}`, and no table."""
    superset_reading(monkeypatch, databases={7: database()}, tables=[stubs.Table("orders")], **reading)
    assert reads.tables_read(READ_REQUEST) == {"unreadable": True}


@pytest.mark.parametrize("sql", [
    "SELECT * FROM {{ current_user_id() }}", "{% if 1 %}SELECT 1 FROM orders{% endif %}",
    "SELECT 1 FROM orders {# a note #}", "SELECT '{{ x }}' FROM orders", "SELECT 1 FROM orders /* {% x %} */",
    "SELECT 1 FROM orders -- {# x", "SELECT \"{{\" FROM orders",
], ids=["{{", "{%", "{#", "in a literal", "in a block comment", "in a line comment", "in a quoted name"])
def test_tables_read_of_a_text_holding_a_template_marker_is_unreadable_and_never_rendered(monkeypatch, sql):
    """A text holding a template marker, anywhere in it, is answered `{"unreadable": true}` before Superset renders or
    parses it: mnemiq never sends one (`adapter.JINJA_DELIMITERS`), and rendering one would answer what the caller's
    template renders to. Superset is not asked even for the database."""
    asked = superset_reading(monkeypatch, databases={7: database()}, tables=[stubs.Table("orders")])
    assert reads.tables_read({**READ_REQUEST, "sql": sql}) == {"unreadable": True}
    assert asked == []


def test_tables_read_refuses_the_markers_mnemiq_refuses():
    from mnemiq.superset.adapter import JINJA_DELIMITERS

    assert reads.TEMPLATE_MARKERS == JINJA_DELIMITERS


def test_tables_read_too_large_to_send_says_so(monkeypatch):
    superset_reading(monkeypatch, databases={7: database()},
                     tables=[stubs.Table(f"{i:04d}" + "x" * 240) for i in range(200)])
    assert reads.tables_read(READ_REQUEST) == {"too_large": True}


def test_a_bad_tables_read_request_is_answered_not_raised():
    assert reads.tables_read({**READ_REQUEST, "sql": ""}) == {"refused": "request", "field": "sql"}


class Reading:
    """mnemiq's Superset client, stood in for `tables_read`: each call answered by `answer` (a function of the
    request, or an answer or exception), every call recorded."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool, request, *, timeout_s=None):
        self.calls.append((tool, json.loads(json.dumps(request))))
        assert tool == TOOL_PREFIX + "tables_read", tool
        found = self.answer(request) if callable(self.answer) else self.answer
        if isinstance(found, Exception):
            raise found
        return json.loads(json.dumps(found))


def read_by(client, sql="SELECT 1 FROM orders"):
    return tables_read(client, database_id=7, schema="public", sql=sql)


def test_mnemiqs_client_reads_what_tables_read_answers(monkeypatch):
    """The join of the two ends: the extension's own `tables_read`, over Superset's lookups stood in, answering
    mnemiq's own request, read by mnemiq's client as a set of (catalog, schema, table)."""
    superset_reading(monkeypatch, databases={7: database()}, tables=[stubs.Table("orders"),
                                                                    stubs.Table("lines", "shop2")])
    client = Reading(reads.tables_read)
    assert read_by(client, READ_REQUEST["sql"]) == frozenset({("live", "public", "orders"),
                                                              ("live", "shop2", "lines")})
    assert client.calls == [(TOOL_PREFIX + "tables_read", READ_REQUEST)]
    assert read_by(Reading({"tables": []})) == frozenset()


def table(catalog="live", schema="public", name="orders"):
    return {"catalog": catalog, "schema": schema, "table": name}


@pytest.mark.parametrize("answer", [
    SupersetToolError("Internal error in extensions.agenticfabriq.mnemiq.tables_read: An unexpected error occurred."),
    {"not_found": "database"}, {"too_large": True}, {"unreadable": True}, {"refused": "request", "field": "sql"},
    ["not", "an", "object"], {"tables": "orders"}, {"tables": [table(), "orders"]},
    {"tables": [table(), {"catalog": None, "schema": "public"}]},
    {"tables": [{**table(), "clause": "x"}]}, {"tables": [table(name="")]}, {"tables": [table(name=3)]},
    {"tables": [table(schema=7)]}, {"tables": [table(catalog=False)]},
    {"tables": [table()], "_response_truncated": True}, {"tables": [table()], "more": True},
], ids=["not offered", "database not found", "too large", "unreadable", "refused", "not an object", "not a list",
        "a malformed entry", "an entry short of a part", "an entry saying more", "a blank name", "a number for a name",
        "a number for a schema", "a boolean for a catalog", "cut for size", "an envelope saying more"])
def test_a_tables_read_answer_mnemiq_cannot_read_whole_tells_it_nothing(answer):
    """Read strictly: exactly `{"tables": [...]}`, each entry exactly `{catalog, schema, table}`, the table a name,
    the catalog and schema a name or null. Anything else, or a call that fails (an extension older than the tool
    among them), is None: mnemiq cannot tell."""
    assert read_by(Reading(answer)) is None


def test_tables_read_raises_superset_not_answering_and_a_refusal_of_the_caller():
    """Superset not answering is not an answer mnemiq cannot read: it is raised, as the client raises it, and so is a
    refusal of the caller."""
    with pytest.raises(SupersetUnavailable):
        read_by(Reading(SupersetUnavailable("Superset did not answer")))
    with pytest.raises(SupersetPermissionDenied):
        read_by(Reading(SupersetPermissionDenied("no")))


def test_table_rules_raises_superset_not_answering_when_asked_to():
    """`table_rules_apply` reads a call Superset did not answer as unknown (the planner's reading), and raises it with
    `raise_unavailable` (the statement check's)."""
    client = Answering(SupersetUnavailable("Superset did not answer"))
    assert ask(client, ["orders"]) == {"orders": None}
    with pytest.raises(SupersetUnavailable):
        table_rules_apply(client, database_id=7, schema="public", tables=["orders"], raise_unavailable=True)


# ---------------------------------------------------------------- mnemiq's reading of table_rules

class Answering:
    """mnemiq's Superset client, stood in: each call of `table_rules` answered by `answer` (a function of the request,
    or an answer or exception for every call), every call recorded."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool, request, *, timeout_s=None):
        self.calls.append((tool, json.loads(json.dumps(request))))
        assert tool == TOOL_PREFIX + "table_rules", tool
        found = self.answer(request) if callable(self.answer) else self.answer
        if isinstance(found, Exception):
            raise found
        return json.loads(json.dumps(found))


def ask(client, tables, *, schema="public"):
    return table_rules_apply(client, database_id=7, schema=schema, tables=tables)


def test_mnemiqs_client_reads_what_the_tool_answers(monkeypatch):
    """The join of the two ends: the extension's own `table_rules`, over Superset's lookups stood in, answering mnemiq's
    own request, read by mnemiq's client."""
    superset_rules(monkeypatch, databases={7: database()}, predicates={"orders": ["(1 = 0)"]})
    client = Answering(reads.table_rules)
    assert ask(client, ["orders", "Customers", "orders"]) == {"orders": True, "Customers": False}
    assert client.calls == [(TOOL_PREFIX + "table_rules",
                             {"database_id": 7, "schema": "public", "tables": ["orders", "Customers"]})]


def test_the_client_asks_at_most_the_tools_limit_a_call_and_nothing_for_no_table():
    assert TABLE_RULES_BATCH == reads.MAX_TABLES == 50
    names = [f"t{i}" for i in range(120)]
    client = Answering(lambda request: {"tables": [{"table": name, "applies": name == "t77"}
                                                   for name in request["tables"]]})
    found = ask(client, names)
    assert found == {name: name == "t77" for name in names}
    assert [request["tables"] for _, request in client.calls] == [names[:50], names[50:100], names[100:]]
    client = Answering(AssertionError("asked"))
    assert ask(client, []) == {} and client.calls == []


def entries(*pairs):
    return {"tables": [{"table": name, "applies": applies} for name, applies in pairs]}


@pytest.mark.parametrize("answer", [
    SupersetToolError("Internal error in extensions.agenticfabriq.mnemiq.table_rules: An unexpected error occurred."),
    SupersetUnavailable("Superset did not answer"),
    {"not_found": "database"},
    {"too_large": True},
    {"refused": "request", "field": "tables"},
    ["not", "an", "object"],
    {"tables": []},
    entries(("orders", False)),
    entries(("orders", False), ("Customers", "no")),
    entries(("orders", False), ("customers", False)),
    entries(("Customers", False), ("orders", False)),
    entries(("orders", False), ("orders", False)),
    entries(("orders", False), ("Customers", False), ("extra", False)),
    {"tables": [{"table": "orders", "applies": False}, {"table": "Customers", "applies": False, "clause": "x"}]},
    {"tables": [{"table": "orders", "applies": False}, "malformed"]},
    {"tables": [{"table": "orders", "applies": False}, {"applies": False}]},
    {**entries(("orders", False), ("Customers", False)), "_response_truncated": True},
    {"tables": {"orders": False, "Customers": False}},
    entries(("orders", 0), ("Customers", 0)),
], ids=["not offered", "unavailable", "database not found", "too large", "refused", "not an object", "left out",
        "one left out", "not a boolean", "another table", "reordered", "twice", "one more", "an entry saying more",
        "a malformed entry", "an entry with no name", "cut for size", "not a list", "a number for a boolean"])
def test_an_answer_mnemiq_cannot_read_whole_tells_it_nothing(answer):
    """Read strictly, as `row_rule_applies` reads `row_rules`: exactly `{"tables": [...]}` with one entry for each name
    asked, in the order asked, each exactly `{"table": <that name>, "applies": <a boolean>}`. Anything else, or a call
    that fails, an extension older than the tool among them, leaves every name of the call unknown (None), never
    False."""
    assert ask(Answering(answer), ["orders", "Customers"]) == {"orders": None, "Customers": None}


def test_one_call_unread_leaves_only_its_own_names_unknown():
    names = [f"t{i}" for i in range(60)]
    client = Answering(lambda request: {"tables": []} if "t0" in request["tables"] else entries(
        *((name, True) for name in request["tables"])))
    found = ask(client, names)
    assert found == {**dict.fromkeys(names[:50]), **dict.fromkeys(names[50:], True)}


def test_a_refusal_of_the_caller_is_raised():
    with pytest.raises(SupersetPermissionDenied):
        ask(Answering(SupersetPermissionDenied("no")), ["orders"])
