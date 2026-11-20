"""The extension's read tools, one function per tool, as `writes` has the write tools: `tools.py` registers them on
Superset's MCP service, which runs each as the caller its token names (`mcp_auth_hook`, `superset/mcp_service/auth.py`).

They change nothing. Each keeps the write tools' rules that apply to a read:

- Its answer is short and structured; Superset's own text never reaches it.
- An object the caller may not read is left out of an answer about several, or answered as one that does not exist,
  by the same check Superset's own tools use (`can_access_datasource`, `can_access_chart`, `can_access_dashboard`),
  so an object out of reach reads as one that is not there.
- It takes no user id from the caller.

An editor's label (`Subject.label`) is returned only for an object the caller may read, and only in the answer
itself, which mnemiq passes to the person's panel; no label goes into a log line or an error.

Superset's MCP response size guard (`superset/mcp_service/middleware.py`, `ResponseSizeGuardMiddleware`) does not
truncate an extension tool's answer: over its limit (50,000 bytes by default) the call fails with Superset's own text.
So every read tool keeps its answer under `MAX_ANSWER_BYTES` itself: `row_rules`, whose at most 20 entries are a few
bytes each, by its size alone, and the rest by one helper (`fit`): `get_layout` refuses a
layout over it with `{"too_large": True}`, and the tools that answer a list keep their envelope and answer each entry
that does not fit as `{<id field>: N, "too_large": True}` (entries fill in request order, and once one does not fit
the rest are marked). A caller treats a marked entry as unanswered.

Superset is imported inside the tools, as in `writes`, so this module imports without it.
"""

from __future__ import annotations

import json
from typing import Any

from agenticfabriq.mnemiq import writes
from agenticfabriq.mnemiq.writes import _Field

MAX_DATASETS = 20
MAX_CHARTS = 100
MAX_DASHBOARDS = 20
# An answer larger than this, in bytes of its JSON text, is not sent: well under the guard's 50,000.
MAX_ANSWER_BYTES = 40_000
MAX_LAYOUT_BYTES = MAX_ANSWER_BYTES
TOO_LARGE = {"too_large": True}


def _distinct_ids(limit: int):
    def read(value: Any) -> list[int]:
        found = list(dict.fromkeys(writes._ids(value)))
        if len(found) > limit:
            raise ValueError("too many ids")
        return found

    return read


# Every read tool's request, field by field.
REQUESTS: dict[str, dict[str, _Field]] = {
    "dataset_metrics": {"dataset_ids": _Field(_distinct_ids(MAX_DATASETS))},
    "editors_of": {"chart_ids": _Field(_distinct_ids(MAX_CHARTS), False, []),
                   "dataset_ids": _Field(_distinct_ids(MAX_DATASETS), False, []),
                   "dashboard_ids": _Field(_distinct_ids(MAX_DASHBOARDS), False, [])},
    "get_layout": {"dashboard_id": writes._DASHBOARD_ID},
    "row_rules": {"dataset_ids": _Field(_distinct_ids(MAX_DATASETS))},
}


def _size(answer: Any) -> int:
    return len(json.dumps(answer).encode())


def fit(entries: list, build, mark) -> dict:
    """`build(entries)`'s answer with as many of `entries`, in order, as keep the answer sent within
    `MAX_ANSWER_BYTES`; from the first that does not fit on, each entry is `mark(entry)` in its place. What is measured
    is what would be sent: the entries kept, this one, and the marks of every entry after it. So whenever an entry is
    kept the answer that ends the loop is no larger than the one just measured, and the final answer is within the
    cap (the marks alone, which a request's id limits, are far below it)."""
    marks = [mark(entry) for entry in entries]
    kept = 0
    for index, entry in enumerate(entries):
        if _size(build([*entries[:index + 1], *marks[index + 1:]])) > MAX_ANSWER_BYTES:
            break
        kept = index + 1
    return build([*entries[:kept], *marks[kept:]])


def read(tool: str, request: Any) -> dict:
    """`request` read for the read tool `tool`; raises `Refused` as `writes.read` does."""
    return writes.read(tool, request, REQUESTS)


def _certification(metric: Any) -> dict:
    """The certification a metric's `extra` holds (`CertificationMixin` in rc2: `extra.certification`, with
    `certified_by` and `details`); empty when there is none or `extra` is not a JSON object."""
    extra = writes._stored_object(getattr(metric, "extra", None)) or {}
    found = extra.get("certification")
    return found if isinstance(found, dict) else {}


def metric_answer(metric: Any) -> dict:
    """One metric as the tool answers it. `currency` is rc2's own object, `{symbol, symbolPosition}`, as `CurrencyType`
    reads it; an empty object (what rc2 reads from text it cannot parse) and anything else stored there are answered
    as none."""
    certification = _certification(metric)
    currency = getattr(metric, "currency", None)
    return {"metric_name": metric.metric_name, "expression": metric.expression, "d3format": metric.d3format,
            "currency": (dict(currency) or None) if isinstance(currency, dict) else None,
            "certified_by": certification.get("certified_by"), "certification_details": certification.get("details"),
            "warning_text": metric.warning_text}


@writes._answering("dataset_metrics", REQUESTS)
def dataset_metrics(found: dict) -> dict:
    """Every saved metric of each dataset the caller may access (`can_access_datasource`), in the order asked. A
    dataset the caller may not access, or that does not exist, is left out, so nothing tells the two apart. An empty
    request answers `{"datasets": []}` without touching Superset: mnemiq's client probes with it to tell a Superset
    without this extension from a tool that fails."""
    datasets = []
    for dataset_id in found["dataset_ids"]:
        dataset = writes._dataset(dataset_id)
        if dataset is not None:
            datasets.append({"dataset_id": dataset_id, "metrics": [metric_answer(m) for m in dataset.metrics]})
    return fit(datasets, lambda kept: {"datasets": kept}, lambda d: {"dataset_id": d["dataset_id"], **TOO_LARGE})


def _chart(chart_id: int) -> Any:
    """The chart, when the caller may read it; else None."""
    from superset.daos.chart import ChartDAO

    chart = ChartDAO.find_by_id(chart_id)
    if chart is None or not writes._security().can_access_chart(chart):
        return None
    return chart


def _labels(resource: Any) -> list[str]:
    return sorted({subject.label for subject in resource.editors})


_ID_FIELDS = {"charts": "chart_id", "datasets": "dataset_id", "dashboards": "dashboard_id"}


@writes._answering("editors_of", REQUESTS)
def editors_of(found: dict) -> dict:
    """The editors' labels of each chart and dataset the caller may read, for the person's panel (who to ask for
    access), and of each dashboard asked for, with whether the caller edits it (`is_editor`), so a build that would
    change it can refuse before it makes anything. An object the caller may not read, or that does not exist, is left
    out. Asked for no dashboard, the answer holds none: as it was before dashboards were asked for."""
    entries = [("charts", {"chart_id": chart_id, "editors": _labels(chart)})
               for chart_id in found["chart_ids"] if (chart := _chart(chart_id)) is not None]
    entries += [("datasets", {"dataset_id": dataset_id, "editors": _labels(dataset)})
                for dataset_id in found["dataset_ids"] if (dataset := writes._dataset(dataset_id)) is not None]
    entries += [("dashboards", {"dashboard_id": dashboard_id, "is_editor": bool(writes._security().is_editor(board)),
                                "editors": _labels(board)})
                for dashboard_id in found["dashboard_ids"] if (board := writes._dashboard(dashboard_id)) is not None]
    kinds = ("charts", "datasets", "dashboards") if found["dashboard_ids"] else ("charts", "datasets")

    def build(kept: list) -> dict:
        return {key: [entry for kind, entry in kept if kind == key] for key in kinds}

    def mark(pair: tuple) -> tuple:
        kind, entry = pair
        # A dashboard cut for size keeps whether the caller edits it, one bool: only its labels are left out.
        kept = {"is_editor": entry["is_editor"]} if kind == "dashboards" else {}
        return kind, {_ID_FIELDS[kind]: entry[_ID_FIELDS[kind]], **kept, **TOO_LARGE}

    return fit(entries, build, mark)


@writes._answering("get_layout", REQUESTS)
def get_layout(found: dict) -> dict:
    """The dashboard's stored `position_json` as the text it is stored in, and its `changed_on` as ISO 8601 text (what
    MCP's `get_dashboard_info` answers, and `set_layout` takes back as `expect_changed_on`), for a dashboard the caller
    may read; else `{"not_found": "dashboard"}`. An answer over `MAX_ANSWER_BYTES` is `{"too_large": True}`."""
    dashboard = writes._dashboard(found["dashboard_id"])
    if dashboard is None:
        return writes._not_found("dashboard")
    answer = {"dashboard_id": found["dashboard_id"], "position_json": dashboard.position_json,
              "changed_on": dashboard.changed_on.isoformat() if dashboard.changed_on is not None else None}
    if _size(answer) > MAX_ANSWER_BYTES:
        return dict(TOO_LARGE)
    return answer


@writes._answering("row_rules", REQUESTS)
def row_rules(found: dict) -> dict:
    """Whether any row rule applies to the caller on each dataset the caller may access (`can_access_datasource`), in
    the order asked: whether Superset's `security_manager.get_rls_filters` finds a rule attached to the dataset that
    applies to the caller. The guest rules of an embedded dashboard are not counted; a guest is never the caller of
    an MCP tool. Only whether: never a rule's clause, name or roles, which an Alpha may not list. A dataset the caller may not access, or that does not exist, is left out, so nothing tells the two apart.
    At most `MAX_DATASETS` short entries, so the answer is far under `MAX_ANSWER_BYTES`."""
    datasets = []
    for dataset_id in found["dataset_ids"]:
        dataset = writes._dataset(dataset_id)
        if dataset is not None:
            datasets.append({"dataset_id": dataset_id,
                             "applies": bool(writes._security().get_rls_filters(dataset))})
    return {"datasets": datasets}
