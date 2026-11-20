"""What the extension's read tools (`agenticfabriq.mnemiq.reads`) answer, with Superset's lookups stood in for: each
request is read strictly, an object the caller may not read is left out (or answered as absent) without a name, and
the layout answer is refused over 40 KB. What the tools do inside Superset is the live suite's:
`tests/superset_live/test_extension_tools.py`.
"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import reads, writes  # noqa: E402


def refusal(tool, request):
    with pytest.raises(writes.Refused) as caught:
        reads.read(tool, request)
    return caught.value.answer


def test_every_read_tool_has_a_request_reader():
    assert set(reads.REQUESTS) == {"dataset_metrics", "editors_of", "get_layout", "row_rules"}


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

    charts = {2: SimpleNamespace(editors=[subject("zed"), subject("amy"), subject("amy")])}
    datasets = {5: SimpleNamespace(editors=[subject("amy")])}
    monkeypatch.setattr(reads, "_chart", lambda chart_id: charts.get(chart_id))
    monkeypatch.setattr(writes, "_dataset", lambda dataset_id: datasets.get(dataset_id))
    answer = reads.editors_of({"chart_ids": [9, 2], "dataset_ids": [5, 6]})
    assert answer == {"charts": [{"chart_id": 2, "editors": ["amy", "zed"]}],
                      "datasets": [{"dataset_id": 5, "editors": ["amy"]}]}
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
    answer = reads.editors_of({"chart_ids": [1, 2, 3], "dataset_ids": [5]})
    assert set(answer) == {"charts", "datasets"}
    assert [("editors" in c) for c in answer["charts"]] == [True, True, False]
    assert answer["charts"][2] == {"chart_id": 3, "too_large": True}
    assert answer["datasets"] == [{"dataset_id": 5, "too_large": True}]
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
