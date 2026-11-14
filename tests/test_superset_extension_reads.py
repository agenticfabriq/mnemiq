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
    assert set(reads.REQUESTS) == {"dataset_metrics", "editors_of", "get_layout"}


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
    assert reads.read("editors_of", {}) == {"chart_ids": [], "dataset_ids": []}
    assert reads.read("editors_of", {"chart_ids": [2, 2], "dataset_ids": [5]}) == {"chart_ids": [2], "dataset_ids": [5]}
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
