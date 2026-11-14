"""The saved-metric catalogue: read as the builder over Superset's MCP service, with the extension's certification
beside it, and reused when a planned measure says what a saved metric already says.

Runs against `FakeSuperset`, a real MCP server over streamable HTTP, through the real client.
"""

from __future__ import annotations

import pytest

from mnemiq.superset.client import DelegatedToken, SupersetMcp, SupersetResponseTooLarge
from mnemiq.superset.extension import TOOL_PREFIX, ExtensionMissing, call_tool
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.saved_metrics import UNKNOWN, SavedMetric, normalized, read_saved_metrics, reusable
from tests.superset_fake import FakeSuperset

DATASET_METRICS = TOOL_PREFIX + "dataset_metrics"
USD = {"symbol": "USD", "symbolPosition": "prefix"}
TOO_LARGE = ("Response too large: 61,234 bytes (limit: 50,000)\n\nThis response would overwhelm the LLM context window.\n"
             "Tool: list_metrics")


@pytest.fixture
def fake():
    with FakeSuperset() as served:
        yield served


def _client(fake) -> SupersetMcp:
    return SupersetMcp(fake.url, DelegatedToken("tok"))


def _metric(dataset_id: int, name: str, expression: str | None = None, **more) -> dict:
    return {"name": name, "verbose_name": more.get("verbose_name"), "description": None,
            "expression": expression or f"SUM({name})", "d3format": more.get("d3format"),
            "warning_text": more.get("warning_text"), "source": "builtin", "dataset_id": dataset_id,
            "dataset_name": f"t{dataset_id}", "view_id": None, "view_name": None, "compatible_dimensions": []}


def _serve_metrics(fake, by_dataset: dict[int, list[dict]]) -> None:
    """Answer `list_metrics` as rc2 does: sorted, paged, with the totals."""
    def answer(request, authorization):
        rows = by_dataset.get(request["dataset_id"], [])
        size, page = request["page_size"], request["page"]
        return {"metrics": rows[(page - 1) * size: page * size], "total_count": len(rows), "page": page,
                "page_size": size, "total_pages": max(1, -(-len(rows) // size)), "success": True}

    fake.reply("list_metrics", answer)


def _serve_extension(fake, by_dataset: dict[int, list[dict]]) -> None:
    def answer(request, authorization):
        return {"datasets": [{"dataset_id": i, "metrics": by_dataset[i]} for i in request["dataset_ids"]
                             if i in by_dataset]}

    fake.reply(DATASET_METRICS, answer)


def _extension_row(name: str, **more) -> dict:
    return {"metric_name": name, "expression": more.get("expression", f"SUM({name})"), "d3format": None,
            "currency": more.get("currency"), "certified_by": more.get("certified_by"),
            "certification_details": None, "warning_text": None}


def _measure(expression: str, dataset_id: int = 1, name: str = "revenue") -> Measure:
    return Measure(name=name, label="Revenue", expression=expression, kind=MeasureKind.NUMBER,
                   dataset_id=dataset_id, description="", asked_by="west")


def _saved(name: str, expression: str, *, dataset_id: int = 1, certified_by=None) -> SavedMetric:
    return SavedMetric(dataset_id=dataset_id, name=name, expression=expression, verbose_name=None, d3format=None,
                       currency=None, certified_by=certified_by, warning_text=None)


def test_list_metrics_paged_by_dataset(fake):
    many = [_metric(1, f"m{i:02d}") for i in range(30)]
    _serve_metrics(fake, {1: many, 2: [_metric(2, "total", "SUM(x)", d3format=",d", verbose_name="Total")]})
    _serve_extension(fake, {1: [_extension_row("m00", currency=USD, certified_by="finance")]
                            + [_extension_row(m["name"]) for m in many[1:]],
                            2: [_extension_row("total")]})
    saved = read_saved_metrics(_client(fake), dataset_ids=(1, 2))
    listed = [c.request for c in fake.calls if c.tool == "list_metrics"]
    assert listed == [{"dataset_id": 1, "page_size": 25, "page": 1}, {"dataset_id": 1, "page_size": 25, "page": 2},
                      {"dataset_id": 2, "page_size": 25, "page": 1}]
    assert len(saved) == 31
    first = next(s for s in saved if (s.dataset_id, s.name) == (1, "m00"))
    assert first.currency == USD and first.certified_by == "finance"
    plain = next(s for s in saved if (s.dataset_id, s.name) == (1, "m01"))
    assert plain.currency is None and plain.certified_by is None
    total = next(s for s in saved if s.dataset_id == 2)
    assert (total.verbose_name, total.d3format, total.expression) == ("Total", ",d", "SUM(x)")
    assert [c.request for c in fake.calls if c.tool == DATASET_METRICS] == [{"dataset_ids": [1, 2]}]


def test_the_extension_is_asked_twenty_datasets_at_a_time():
    """The extension's `dataset_metrics` takes at most 20 dataset ids a call and refuses more
    (`reads.MAX_DATASETS`); 45 datasets are asked in calls of 20, 20 and 5, in order, and every metric's
    certification is known."""
    from tests.superset_extension_path import use_extension_source

    use_extension_source()
    from agenticfabriq.mnemiq import reads

    from mnemiq.superset import extension

    ids = tuple(range(101, 146))
    with FakeSuperset() as fake:
        _serve_metrics(fake, {i: [_metric(i, f"m{i}")] for i in ids})

        def answer(request, authorization):
            if len(request["dataset_ids"]) > reads.MAX_DATASETS:
                return {"refused": "request", "field": "dataset_ids"}
            return {"datasets": [{"dataset_id": i, "metrics": [_extension_row(f"m{i}", certified_by="finance")]}
                                 for i in request["dataset_ids"]]}

        fake.reply(DATASET_METRICS, answer)
        saved = read_saved_metrics(_client(fake), dataset_ids=ids)
        asked = [c.request["dataset_ids"] for c in fake.calls if c.tool == DATASET_METRICS]
    assert asked == [list(ids[:20]), list(ids[20:40]), list(ids[40:])]
    assert len(saved) == 45 and all(s.certified_by == "finance" for s in saved)
    # mnemiq's batch is the extension's own limit.
    assert extension.DATASET_METRICS_BATCH == reads.MAX_DATASETS


def test_without_the_extension_certification_is_unknown():
    with FakeSuperset(extension=False) as fake:
        _serve_metrics(fake, {1: [_metric(1, "a"), _metric(1, "b")]})
        saved = read_saved_metrics(_client(fake), dataset_ids=(1,))
        extension_calls = [c.tool for c in fake.calls if c.tool.startswith("extensions.")]
    assert [s.name for s in saved] == ["a", "b"]
    assert all(s.certified_by is UNKNOWN and s.currency is UNKNOWN for s in saved)
    # The call and its one probe, and nothing retried after.
    assert extension_calls == [DATASET_METRICS, DATASET_METRICS]
    # With nothing known about certification, none is preferred: the order is by name.
    plain = [_saved("b", "SUM(amount)", certified_by=UNKNOWN), _saved("a", "SUM(amount)", certified_by=UNKNOWN)]
    assert reusable(_measure("sum(amount)"), plain).name == "a"


def test_a_missing_extension_is_told_from_a_failing_one():
    with FakeSuperset() as fake:
        # The tool exists and fails, but the probe with no datasets answers: the failure is the tool's own.
        def answer(request, authorization):
            if request["dataset_ids"]:
                raise RuntimeError(f"Internal error in {DATASET_METRICS}: An unexpected error occurred. Error ID: 1.")
            return {"datasets": []}

        fake.reply(DATASET_METRICS, answer)
        with pytest.raises(Exception) as caught:
            call_tool(_client(fake), "dataset_metrics", {"dataset_ids": [1]})
        assert not isinstance(caught.value, ExtensionMissing)
    with FakeSuperset(extension=False) as fake:
        with pytest.raises(ExtensionMissing) as caught:
            call_tool(_client(fake), "dataset_metrics", {"dataset_ids": [1]})
        assert "Error ID" not in str(caught.value)
        assert "Error ID" in caught.value.detail
    with FakeSuperset() as fake:
        # An unrelated failure is not read as a missing extension, and costs no probe.
        fake.reply(TOOL_PREFIX + "editors_of", lambda request, authorization: RuntimeError("boom"))
        with pytest.raises(Exception) as caught:
            call_tool(_client(fake), "editors_of", {})
        assert not isinstance(caught.value, ExtensionMissing)
        assert [c.tool for c in fake.calls] == [TOOL_PREFIX + "editors_of"]


def test_equivalent_saved_metric_is_reused():
    saved = [_saved("total_amount", "SUM(amount)"), _saved("other", "SUM(amount)", dataset_id=2, certified_by="x")]
    assert normalized("sum( amount )") == normalized("SUM(amount)")
    hit = reusable(_measure("sum( amount )"), saved)
    assert hit is not None and (hit.dataset_id, hit.name) == (1, "total_amount")
    assert reusable(_measure("sum( amount )", dataset_id=3), saved) is None
    assert reusable(_measure("avg(amount)"), saved) is None
    # What does not parse is never equal to anything, itself included.
    assert reusable(_measure("sum(("), [_saved("bad", "sum((")]) is None


def test_certified_metric_wins_a_tie():
    saved = [_saved("a_plain", "SUM(amount)"), _saved("b_certified", "SUM( amount )", certified_by="finance"),
             _saved("c_certified", "SUM(amount)", certified_by="finance")]
    assert reusable(_measure("SUM(amount)"), saved).name == "b_certified"


def test_over_limit_answer_is_a_stated_failure(fake):
    sizes: list[int] = []

    def too_big_once(request, authorization):
        sizes.append(request["page_size"])
        if request["page_size"] == 25:
            return RuntimeError(TOO_LARGE)
        return {"metrics": [_metric(1, "a")], "total_count": 1, "page": 1, "page_size": 5, "total_pages": 1,
                "success": True}

    fake.reply("list_metrics", too_big_once)
    _serve_extension(fake, {})
    assert [s.name for s in read_saved_metrics(_client(fake), dataset_ids=(1,))] == ["a"]
    assert sizes == [25, 5]

    sizes.clear()
    fake.reply("list_metrics", lambda request, authorization: (sizes.append(request["page_size"]),
                                                               RuntimeError(TOO_LARGE))[1])
    with pytest.raises(SupersetResponseTooLarge):
        read_saved_metrics(_client(fake), dataset_ids=(1,))
    assert sizes == [25, 5]


def test_a_metric_the_extension_does_not_answer_for_is_unknown_not_none(fake):
    _serve_metrics(fake, {1: [_metric(1, "a"), _metric(1, "b"), _metric(1, "c")], 2: [_metric(2, "z")]})
    # Dataset 2 is absent from the answer, and so is metric "b" of dataset 1; "c" is answered with null fields.
    _serve_extension(fake, {1: [_extension_row("a", currency=USD, certified_by="finance"), _extension_row("c")]})
    by_name = {(s.dataset_id, s.name): s for s in read_saved_metrics(_client(fake), dataset_ids=(1, 2))}
    for key in ((1, "b"), (2, "z")):
        assert by_name[key].currency is UNKNOWN and by_name[key].certified_by is UNKNOWN
    assert by_name[(1, "c")].currency is None and by_name[(1, "c")].certified_by is None
    assert by_name[(1, "a")].currency == USD and by_name[(1, "a")].certified_by == "finance"


def test_a_dataset_the_extension_marked_too_large_is_unknown_not_none(fake):
    """The extension answers a dataset whose metrics do not fit its size cap as `{dataset_id, too_large}` with no
    `metrics`: its saved metrics are unknown (the builder saw them in `list_metrics`), never "without currency"."""
    _serve_metrics(fake, {1: [_metric(1, "a")], 2: [_metric(2, "z")]})

    def answer(request, authorization):
        return {"datasets": [{"dataset_id": 1, "metrics": [_extension_row("a", currency=USD)]},
                             {"dataset_id": 2, "too_large": True}]}

    fake.reply(DATASET_METRICS, answer)
    by_name = {(s.dataset_id, s.name): s for s in read_saved_metrics(_client(fake), dataset_ids=(1, 2))}
    assert by_name[(2, "z")].currency is UNKNOWN and by_name[(2, "z")].certified_by is UNKNOWN
    assert by_name[(1, "a")].currency == USD
