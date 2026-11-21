"""What the extension's write tools (`agenticfabriq.mnemiq.writes`) refuse before they touch Superset: each request
is read strictly, field by field, and a marker, a metric and an expected `changed_on` are held to their shapes.

`writes` imports Superset only inside its tools, so this runs in `unit` with neither Superset nor Flask installed.
What the tools do inside Superset (the locks, the commands, the answers) is the live suite's:
`tests/superset_live/test_extension_tools.py`.
"""

import ast
import contextlib
import copy
import hashlib
import itertools
import json
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY

import pytest

from tests import superset_stubs as stubs

from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import params, writes  # noqa: E402
from agenticfabriq.mnemiq.params import format_marker  # noqa: E402

BUILD = "0f8fad5b-d9cb-469f-a165-70867728950e"
WHEN = "2026-11-13T23:57:34.123456"


def refusal(tool, request):
    with pytest.raises(writes.Refused) as caught:
        writes.read(tool, request)
    return caught.value.answer


def test_every_write_tool_has_a_request_reader():
    assert set(writes.REQUESTS) == {"create_dashboard", "create_dataset", "create_chart", "patch_chart",
                                    "add_saved_metric", "delete_dataset", "set_layout"}


@pytest.mark.parametrize("bad", [0, -3, True, False, "7", 7.0, None, [7]])
def test_ids_are_positive_integers(bad):
    good = {"chart_id": 7, "set": {}, "unset": [], "expect_changed_on": WHEN}
    assert writes.read("patch_chart", good)["chart_id"] == 7
    assert refusal("patch_chart", {**good, "chart_id": bad}) == {"refused": "request", "field": "chart_id"}


def test_missing_and_unknown_fields_are_refused():
    good = {"dataset_id": 4, "dashboard_id": 12}
    assert writes.read("delete_dataset", good) == good
    assert refusal("delete_dataset", {"dataset_id": 4}) == {"refused": "request", "field": "dashboard_id"}
    # A misspelt field is refused rather than ignored, so a caller's typo never reads as the field being absent.
    assert refusal("delete_dataset", {**good, "dashbaord_id": 12}) == {"refused": "request",
                                                                       "unknown": ["dashbaord_id"]}
    assert refusal("delete_dataset", ["dataset_id", 4]) == {"refused": "request"}
    assert refusal("delete_dataset", None) == {"refused": "request"}


def test_optional_lists_default_to_empty():
    found = writes.read("patch_chart", {"chart_id": 7, "set": {"row_limit": 10}, "expect_changed_on": WHEN})
    assert found["unset"] == []
    found = writes.read("set_layout", {"dashboard_id": 3, "positions": {}, "expect_changed_on": WHEN})
    assert found["add_chart_ids"] == [] and found["remove_chart_ids"] == []


@pytest.mark.parametrize("bad", ["row_limit", [1], [None], {"row_limit": 1}])
def test_unset_is_a_list_of_names(bad):
    assert refusal("patch_chart", {"chart_id": 7, "set": {}, "unset": bad, "expect_changed_on": WHEN}) == {
        "refused": "request", "field": "unset"}


@pytest.mark.parametrize("field", ["add_chart_ids", "remove_chart_ids"])
@pytest.mark.parametrize("bad", [[0], [True], ["5"], 5, [[5]]])
def test_layout_chart_ids_are_lists_of_ids(field, bad):
    request = {"dashboard_id": 3, "positions": {}, "expect_changed_on": WHEN, field: bad}
    assert refusal("set_layout", request) == {"refused": "request", "field": field}


def test_expect_changed_on_is_the_instant_superset_serialises():
    """MCP's `get_chart_info`, `get_dataset_info` and `get_dashboard_info` serialise `changed_on`, a naive datetime,
    as ISO 8601 text; that text is what a caller sends back."""
    read = writes.read("patch_chart", {"chart_id": 7, "set": {}, "expect_changed_on": WHEN})
    assert read["expect_changed_on"] == datetime(2026, 11, 13, 23, 57, 34, 123456)
    read = writes.read("patch_chart", {"chart_id": 7, "set": {}, "expect_changed_on": "2026-11-13T23:57:34"})
    assert read["expect_changed_on"] == datetime(2026, 11, 13, 23, 57, 34)
    # An instant with a zone cannot be compared with Superset's naive one, and is refused rather than read as stale.
    for bad in ("2026-11-13T23:57:34+00:00", "yesterday", "", 1763107054, None):
        assert refusal("patch_chart", {"chart_id": 7, "set": {}, "expect_changed_on": bad}) == {
            "refused": "request", "field": "expect_changed_on"}


def test_a_claim_marker_is_exactly_the_dashboards():
    marker = format_marker(BUILD, "revenue_by_month", 12)
    assert writes.claim_marker(marker, 12).dashboard_id == 12
    # Compared as an integer: dashboard 12 is not dashboard 123, nor the other way round.
    assert writes.claim_marker(format_marker(BUILD, "revenue_by_month", 123), 12) is None
    assert writes.claim_marker(marker, 123) is None
    # The dashboard shell's own marker says `new`, and only the shell takes it.
    shell = format_marker(BUILD, "dashboard", None)
    assert writes.claim_marker(shell, None).dashboard_id is None
    assert writes.claim_marker(shell, 12) is None
    assert writes.claim_marker(marker, None) is None
    # Exactly one marker line: no prose, no surrounding space, never a non-string.
    for bad in (f"Revenue\n{marker}", f" {marker}", f"{marker}\n", marker.upper(), "", None, 12):
        assert writes.claim_marker(bad, 12) is None


def test_a_description_keeps_its_prose_and_ends_with_the_marker():
    marker = format_marker(BUILD, "revenue_by_month", 12)
    assert writes.with_marker(None, marker) == marker
    assert writes.with_marker("", marker) == marker
    assert writes.with_marker("Revenue by month.\n\n", marker) == f"Revenue by month.\n{marker}"
    assert writes.with_marker("Revenue\nby month.", marker) == f"Revenue\nby month.\n{marker}"
    # A description that already ends with the marker keeps one marker line, not two.
    assert writes.with_marker(f"Revenue by month.\n{marker}\n", marker) == f"Revenue by month.\n{marker}"
    assert writes.with_marker(marker, marker) == marker
    # A marker pasted mid-text is prose (a marker is read from the last line only), and stays.
    assert writes.with_marker(f"{marker}\nnotes", marker) == f"{marker}\nnotes\n{marker}"


METRIC = {"metric_name": "avg_premium", "expression": "AVG(premium)", "verbose_name": "Average premium",
          "d3format": ",.2f", "currency": {"symbol": "USD", "symbolPosition": "prefix"}, "description": "per policy"}


def test_a_new_metric_is_read_strictly():
    request = {"dataset_id": 4, "metric": METRIC, "expect_changed_on": WHEN}
    assert writes.read("add_saved_metric", request)["metric"] == METRIC
    assert writes.read("add_saved_metric", {**request, "metric": {"metric_name": "n", "expression": "COUNT(*)"}})[
        "metric"] == {"metric_name": "n", "expression": "COUNT(*)"}
    bad_metrics = [
        {**METRIC, "id": 3},  # an existing metric's id: the tool adds, never updates
        {**METRIC, "extra": "{}"},  # certification is not the builder's to set
        {**METRIC, "metric_name": ""}, {**METRIC, "metric_name": "x" * 256}, {**METRIC, "expression": " "},
        {k: v for k, v in METRIC.items() if k != "expression"},
        {**METRIC, "currency": "USD"}, {**METRIC, "currency": {"symbol": "$", "symbolPosition": "prefix"}},
        {**METRIC, "currency": {"symbol": "usd", "symbolPosition": "prefix"}},
        {**METRIC, "currency": {"symbol": "USD", "symbolPosition": "before"}},
        {**METRIC, "currency": {"symbol": "USD"}},
        {**METRIC, "d3format": 3}, {**METRIC, "verbose_name": "y" * 1025},
        "AVG(premium)",
    ]
    for bad in bad_metrics:
        assert refusal("add_saved_metric", {**request, "metric": bad}) == {"refused": "request", "field": "metric"}
    assert writes.read("add_saved_metric", {**request, "metric": {**METRIC, "currency": None}})["metric"][
        "currency"] is None


@pytest.mark.parametrize("field", ["metric_name", "expression", "verbose_name", "description"])
@pytest.mark.parametrize("opening", ["{{", "{%", "{#"])
def test_a_metric_holding_a_template_is_refused(field, opening):
    metric = {**METRIC, field: f"AVG(premium) {opening} x"}
    assert refusal("add_saved_metric", {"dataset_id": 4, "metric": metric, "expect_changed_on": WHEN}) == {
        "refused": "template"}


def test_create_chart_reads_its_own_types_only():
    request = {"dataset_id": 4, "viz_type": "funnel", "slice_name": "Policies by status",
               "params": {"viz_type": "funnel", "groupby": ["status"], "metric": "count"}, "dashboard_id": 12,
               "marker": format_marker(BUILD, "funnel", 12)}
    assert writes.read("create_chart", request)["viz_type"] == "funnel"
    assert writes.read("create_chart", {**request, "viz_type": "sankey_v2"})["viz_type"] == "sankey_v2"
    for bad in ("echarts_timeseries_line", "pie", "", None):
        assert refusal("create_chart", {**request, "viz_type": bad}) == {"refused": "request", "field": "viz_type"}
    for bad in ("", " ", "x" * 251, None):
        assert refusal("create_chart", {**request, "slice_name": bad}) == {"refused": "request",
                                                                           "field": "slice_name"}
    for bad in ([], "{}", {1: "x"}, {"x": {1, 2}}):
        assert refusal("create_chart", {**request, "params": bad}) == {"refused": "request", "field": "params"}


def test_a_dashboard_title_is_a_short_name():
    request = {"title": "Policies", "marker": format_marker(BUILD, "dashboard", None)}
    assert writes.read("create_dashboard", request)["title"] == "Policies"
    for bad in ("", "   ", "x" * 501, None, 5):
        assert refusal("create_dashboard", {**request, "title": bad}) == {"refused": "request", "field": "title"}


# ---------------------------------------------------------------- the registration

TOOLS_SOURCE = Path(writes.__file__).with_name("tools.py")
# The class permission each write tool is gated by; its method is `write`.
WRITE_CLASSES = {"create_dashboard": "Dashboard", "create_dataset": "Dataset", "create_chart": "Chart",
                 "patch_chart": "Chart", "add_saved_metric": "Dataset", "delete_dataset": "Dataset",
                 "set_layout": "Dashboard"}
# The class permission each read tool is gated by; its method is `read`.
READ_CLASSES = {"dataset_metrics": "Dataset", "editors_of": "Chart", "get_layout": "Dashboard", "row_rules": "Dataset"}
ALL_CLASSES = {**WRITE_CLASSES, **READ_CLASSES}


def registration_problems(source: str) -> list[str]:
    """What is wrong with the `@tool` registrations in `source`, read with `ast`: each must be named one of mnemiq's
    tool names, with that tool's stock class permission and the method `write` (a write tool) or `read` (a read
    tool), take one `request`, and call `writes.<its name>` or `reads.<its name>` (by kind) and nothing else of
    either; and every tool must be there once."""
    problems, seen = [], []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and getattr(decorator.func, "id", None) == "tool"):
                continue
            keywords = {k.arg: k.value.value for k in decorator.keywords if isinstance(k.value, ast.Constant)}
            name = keywords.get("name")
            seen.append(name)
            if name not in ALL_CLASSES:
                problems.append(f"{name!r} is not one of mnemiq's tools")
                continue
            module, method = ("reads", "read") if name in READ_CLASSES else ("writes", "write")
            if keywords.get("class_permission_name") != ALL_CLASSES[name]:
                problems.append(f"{name} is not gated by {ALL_CLASSES[name]}")
            if keywords.get("method_permission_name") != method:
                problems.append(f"{name} is not gated as a {method}")
            if [arg.arg for arg in node.args.args] != ["request"]:
                problems.append(f"{name} does not take one request")
            called = [f"{call.func.value.id}.{call.func.attr}" for call in ast.walk(node)
                      if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                      and getattr(call.func.value, "id", None) in ("writes", "reads")]
            if called != [f"{module}.{name}"]:
                problems.append(f"{name} calls {called}")
    if sorted(seen) != sorted(ALL_CLASSES):
        problems.append(f"registered {sorted(filter(None, seen))}, not each tool once")
    return problems


def test_the_tools_are_mnemiqs_names_with_stock_permissions():
    from mnemiq.superset.extension import EXTENSION_TOOLS

    assert set(EXTENSION_TOOLS) == set(ALL_CLASSES) and len(EXTENSION_TOOLS) == 11
    assert set(writes.REQUESTS) == set(WRITE_CLASSES)
    assert registration_problems(TOOLS_SOURCE.read_text()) == []


@pytest.mark.parametrize(("old", "new", "flagged"), [
    ('name="patch_chart"', 'name="patch_charts"', "'patch_charts' is not one of mnemiq's tools"),
    ('name="get_layout", class_permission_name="Dashboard", method_permission_name="read"',
     'name="get_layout", class_permission_name="Dashboard", method_permission_name="write"',
     "get_layout is not gated as a read"),
    ('name="editors_of", class_permission_name="Chart"', 'name="editors_of", class_permission_name="Dataset"',
     "editors_of is not gated by Chart"),
    ("return reads.dataset_metrics(request)", "return writes.dataset_metrics(request)",
     "dataset_metrics calls ['writes.dataset_metrics']"),
    ('name="set_layout", class_permission_name="Dashboard"', 'name="set_layout", class_permission_name="mnemiq"',
     "set_layout is not gated by Dashboard"),
    ('name="delete_dataset", class_permission_name="Dataset", method_permission_name="write"',
     'name="delete_dataset", class_permission_name="Dataset", method_permission_name="read"',
     "delete_dataset is not gated as a write"),
    ("return writes.create_dataset(request)", "return writes.create_chart(request)",
     "create_dataset calls ['writes.create_chart']"),
])
def test_the_registration_check_flags_a_planted_defect(old, new, flagged):
    source = TOOLS_SOURCE.read_text()
    assert source.count(old) == 1
    assert flagged in registration_problems(source.replace(old, new))


README = TOOLS_SOURCE.parents[4] / "README.md"


def registered(source: str) -> dict[str, tuple[str, str]]:
    """Each `@tool` registration in `source`, read with `ast`: its name, with its class permission and method."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "tool":
            keywords = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
            found[keywords["name"]] = (keywords["class_permission_name"], keywords["method_permission_name"])
    return found


def readme_problems(readme: str, tools_source: str, extension_sources: list[str]) -> list[str]:
    """What the extension's README does not say that its code does: each registered tool has a row in the table of
    its kind (`## The write tools` for the method `write`, `## The read tools` for `read`) naming the class
    permission it is gated by, and each `MNEMIQ_*` setting the extension reads has a row in `## Configuration`."""
    sections: dict[str, list[list[str]]] = {}
    heading = ""
    for line in readme.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
        elif line.startswith("| `"):
            sections.setdefault(heading, []).append([cell.strip() for cell in line.strip("|").split("|")])
    problems = []
    for name, (class_name, method) in sorted(registered(tools_source).items()):
        rows = {row[0].strip("`"): row[1] for row in sections.get(f"The {method} tools", [])}
        if rows.get(name) != class_name:
            problems.append(f"{name} has no row gated by {class_name} under 'The {method} tools'")
    settings = {row[0].strip("`") for row in sections.get("Configuration", [])}
    read = sorted({key for source in extension_sources for key in re.findall(r'"(MNEMIQ_[A-Z_]+)"', source)})
    problems += [f"{key} is not in the configuration table" for key in read if key not in settings]
    return problems


def _extension_sources() -> list[str]:
    return [path.read_text() for path in sorted(TOOLS_SOURCE.parent.glob("*.py"))]


def test_the_readme_documents_every_tool_and_setting():
    assert readme_problems(README.read_text(), TOOLS_SOURCE.read_text(), _extension_sources()) == []
    assert len(registered(TOOLS_SOURCE.read_text())) == 11


@pytest.mark.parametrize(("old", "new", "flagged"), [
    ("| `get_layout` | Dashboard |", "| `get_layout` | Chart |",
     "get_layout has no row gated by Dashboard under 'The read tools'"),
    ("| `set_layout` | Dashboard |", "| `set_layouts` | Dashboard |",
     "set_layout has no row gated by Dashboard under 'The write tools'"),
    ("| `MNEMIQ_LOCK_TIMEOUT_S` |", "| `MNEMIQ_LOCK_TIMEOUT` |", "MNEMIQ_LOCK_TIMEOUT_S is not in the configuration table"),
])
def test_the_readme_check_flags_a_planted_defect(old, new, flagged):
    readme = README.read_text()
    assert readme.count(old) == 1, old
    assert flagged in readme_problems(readme.replace(old, new), TOOLS_SOURCE.read_text(), _extension_sources())


# ---------------------------------------------------------------- the lock

WRITES_SOURCE = Path(writes.__file__)
# Each tool that writes an existing row: the model whose row it locks first, and the request field naming the row.
# Creating a dashboard, a dataset or a chart writes a new row and does not lock a row itself: it locks its marker
# (`_find_or_create`), and on MySQL its second lock is a row lock taken under that (`_take_create_lock`).
LOCKED = {"patch_chart": ("Slice", "chart_id"), "add_saved_metric": ("SqlaTable", "dataset_id"),
          "delete_dataset": ("SqlaTable", "dataset_id"), "set_layout": ("Dashboard", "dashboard_id")}
UNLOCKED = {"create_dashboard", "create_dataset", "create_chart"}


def _calls(node: ast.AST) -> list[ast.Call]:
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))


def _callee(call: ast.Call) -> str | None:
    return getattr(call.func, "attr", None) or getattr(call.func, "id", None)


def lock_problems(source: str) -> list[str]:
    """What is wrong with how `writes` locks, read with `ast`:
    - `lock_row` is the one locking read: `populate_existing`, `enable_eagerloads(False)` and `with_for_update`, and
      nothing else in the module locks a row (no `with_for_update` elsewhere, no `lock_entity_for_update`);
    - `_locked` calls `lock_row(model, entity_id)` before it calls `write`;
    - each locked tool answers `_locked(<its model>, found[<its id field>], lambda ...)` or a refusal constant of its
      request checks, and an unlocked one never locks;
    - every function that compares `found["expect_changed_on"]` is called only inside the write a `_locked` call
      runs, so no `changed_on` is compared before the lock."""
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    problems: list[str] = []
    lock = functions.get("lock_row")
    if lock is None:
        return ["no lock_row"]
    named = [_callee(call) for call in _calls(lock)]
    for needed in ("populate_existing", "enable_eagerloads", "with_for_update"):
        if needed not in named:
            problems.append(f"lock_row does not call {needed}")
    eager = [call for call in _calls(lock) if _callee(call) == "enable_eagerloads"]
    if not eager or any([ast.dump(a) for a in call.args] != [ast.dump(ast.Constant(False))] for call in eager):
        problems.append("lock_row leaves eager loads on")
    for name, function in functions.items():
        for call in _calls(function):
            if _callee(call) == "lock_entity_for_update":
                problems.append(f"{name} calls lock_entity_for_update")
            if _callee(call) == "with_for_update" and name != "lock_row":
                problems.append(f"{name} locks a row outside lock_row")
    locked = functions.get("_locked")
    order = [_callee(call) for call in _calls(locked)] if locked else []
    if "lock_row" not in order or "write" not in order or order.index("write") < order.index("lock_row"):
        problems.append("_locked does not lock before it writes")
    for tool, (model, field) in LOCKED.items():
        answers = [node.value for node in ast.walk(functions[tool]) if isinstance(node, ast.Return)]
        good = [value for value in answers if isinstance(value, ast.Call) and _callee(value) == "_locked"
                and len(value.args) == 3 and getattr(value.args[0], "id", None) == model
                and isinstance(value.args[1], ast.Subscript) and getattr(value.args[1].value, "id", None) == "found"
                and getattr(value.args[1].slice, "value", None) == field and isinstance(value.args[2], ast.Lambda)]
        # Besides the lock, a tool may answer only a refusal constant its request checks chose (a bad marker), which
        # reads no row.
        others = [value for value in answers if value not in good and not isinstance(value, ast.Name)]
        if len(good) != 1 or others:
            problems.append(f"{tool} does not answer through _locked({model}, found[{field!r}], ...)")
        # Before its lock a tool may only check its request (the marker's shape); any other call could read the row
        # it is about to lock, and a decision made on that read is made without the lock.
        in_lock = {id(call) for value in good for call in _calls(value)}
        before = sorted({_callee(call) for statement in functions[tool].body for call in _calls(statement)
                         if id(call) not in in_lock} - {"claim_marker"})
        if before:
            problems.append(f"{tool} reads before its lock: {', '.join(before)}")
    for tool in UNLOCKED:
        if {"_locked", "lock_row"} & {_callee(call) for call in _calls(functions[tool])}:
            problems.append(f"{tool} takes a lock the table does not give it")
    inside_lock = {id(call) for function in functions.values() for outer in _calls(function)
                   if _callee(outer) == "_locked" for arg in outer.args if isinstance(arg, ast.Lambda)
                   for call in _calls(arg)}
    comparing = {name for name, function in functions.items() for node in ast.walk(function)
                 if isinstance(node, ast.Subscript) and getattr(node.slice, "value", None) == "expect_changed_on"}
    for function in functions.values():
        for call in _calls(function):
            if _callee(call) in comparing and id(call) not in inside_lock:
                problems.append(f"{call.func.id} compares changed_on outside a lock")
    return problems


def test_every_locked_write_locks_its_row_first():
    assert lock_problems(WRITES_SOURCE.read_text()) == []


@pytest.mark.parametrize(("old", "new", "flagged"), [
    (".enable_eagerloads(False)", "", "lock_row does not call enable_eagerloads"),
    (".enable_eagerloads(False)", ".enable_eagerloads(True)", "lock_row leaves eager loads on"),
    ("entity = lock_row(model, entity_id)",
     "from superset.versioning.api_helpers import lock_entity_for_update\n"
     "            entity = lock_entity_for_update(model, entity_id)", "_locked calls lock_entity_for_update"),
    ("entity = lock_row(model, entity_id)\n", "entity = None\n", "_locked does not lock before it writes"),
    ('return _locked(Slice, found["chart_id"], lambda chart: _patch(found, chart))',
     'return _patch(found, _session().get(Slice, found["chart_id"]))', "_patch compares changed_on outside a lock"),
    ('return _locked(Dashboard, found["dashboard_id"]', 'return _locked(Slice, found["dashboard_id"]',
     "set_layout does not answer through _locked(Dashboard, found['dashboard_id'], ...)"),
    # A patch that decides on an unlocked read of the chart, before its lock.
    ('    return _locked(Slice, found["chart_id"], lambda chart: _patch(found, chart))',
     '    if not _readable_chart(found["chart_id"]):\n        return _not_found("chart")\n'
     '    return _locked(Slice, found["chart_id"], lambda chart: _patch(found, chart))',
     "patch_chart reads before its lock: _not_found, _readable_chart"),
    ('    return _locked(SqlaTable, found["dataset_id"], lambda dataset: _delete(found, dataset))',
     '    if _dataset(found["dataset_id"]) is None:\n        return _not_found("dataset")\n'
     '    return _locked(SqlaTable, found["dataset_id"], lambda dataset: _delete(found, dataset))',
     "delete_dataset reads before its lock: _dataset, _not_found"),
])
def test_the_lock_check_flags_a_planted_defect(old, new, flagged):
    source = WRITES_SOURCE.read_text()
    assert source.count(old) == 1, old
    assert flagged in lock_problems(source.replace(old, new))


# ---------------------------------------------------------------- inside Superset, stood in

@pytest.fixture
def superset(monkeypatch):
    return stubs.install(monkeypatch)


def _funnel_context() -> dict:
    return params.query_context("funnel", {"viz_type": "funnel", "groupby": ["status"], "metric": "count"}, 7)


@pytest.mark.parametrize(("raised", "answer"), [
    # rc2 raises this while building a query object (`superset/exceptions.py`): not a command's exception.
    (stubs.QueryObjectValidationError("Columns missing in dataset: ['no_such_column']"), {"refused": "query_failed"}),
    (stubs.SupersetException("Superset's own words"), {"refused": "query_failed"}),
    (stubs.CommandException("Error: relation does not exist"), {"refused": "query_failed"}),
    (stubs.MarshmallowValidationError("Unknown field."), {"refused": "query_failed"}),
    (stubs.SupersetSecurityException("You need access to the following tables: secret"), {"not_found": "dataset"}),
])
def test_a_query_superset_refuses_is_answered_without_its_text(superset, raised, answer):
    """`create_chart` runs the chart's query before saving it. Whatever Superset raises is answered by its kind and
    never carries Superset's words, which the MCP service would otherwise pass on as the tool's error."""
    superset.chart_data = raised
    assert writes._check_query(_funnel_context()) == answer


def test_a_query_that_runs_is_checked_on_two_rows(superset):
    context = _funnel_context()
    assert writes._check_query(context) is None
    (ran,) = superset.ran
    assert [query["row_limit"] for query in ran[1]["queries"]] == [2]
    assert context == _funnel_context()  # the context saved keeps its own row limit
    superset.chart_data = {"queries": [{"error": "Superset's words", "status": "failed"}]}
    assert writes._check_query(context) == {"refused": "query_failed"}


MARKED = format_marker(BUILD, "data", 12)


def _dataset(description: str | None) -> SimpleNamespace:
    return SimpleNamespace(id=4, uuid="6b1d3c1e-0f63-4f5c-9a51-0c7f0b9f2a11", description=description,
                           editors=[SimpleNamespace(label="analyst Live")])


@pytest.fixture
def deleting(monkeypatch, superset):
    """`delete_dataset`'s work under the lock, with Superset stood in: the caller may read the dataset and edits it
    unless a test says not; `charts` is what each count of the dataset's live charts answers, in turn; the delete is
    soft unless a test says not."""
    state = SimpleNamespace(editor=True, charts=[[], []], soft=True, superset=superset)
    security = SimpleNamespace(can_access_datasource=lambda dataset: True, is_editor=lambda dataset: state.editor)
    monkeypatch.setattr(writes, "_security", lambda: security)
    monkeypatch.setattr(writes, "_live_chart_ids", lambda dataset_id: state.charts.pop(0))
    monkeypatch.setattr(writes, "_soft_deleted", lambda dataset_id: state.soft)
    return state


def delete(dataset: SimpleNamespace, dashboard_id: int = 12) -> dict:
    return writes._delete({"dataset_id": dataset.id, "dashboard_id": dashboard_id}, dataset)


def test_delete_moves_an_unused_dataset_to_the_trash(deleting):
    assert delete(_dataset(MARKED)) == {"ok": True, "soft_deleted": True}
    assert deleting.superset.ran == [("DeleteDatasetCommand", [4])]
    deleting.charts, deleting.soft = [[], []], False
    assert delete(_dataset(MARKED)) == {"ok": True, "soft_deleted": False}


def test_delete_refuses_while_a_chart_uses_the_dataset(deleting):
    deleting.charts = [[31, 32]]
    assert delete(_dataset(MARKED)) == {"in_use": True, "chart_ids": [31, 32]}
    assert deleting.superset.ran == []


def test_a_chart_made_while_the_dataset_is_deleted_brings_it_back(deleting):
    """No lock keeps a chart from being made on a dataset meanwhile, so the charts are counted again once the delete
    has committed: one found then restores the dataset, and the answer says it is in use."""
    deleting.charts = [[], [31]]
    dataset = _dataset(MARKED)
    assert delete(dataset) == {"in_use": True, "chart_ids": [31], "restored": True}
    assert deleting.superset.ran == [("DeleteDatasetCommand", [4]), ("RestoreDatasetCommand", dataset.uuid)]
    # Gone rather than in the trash, it cannot be restored, and the answer says so.
    deleting.superset.ran.clear()
    deleting.charts, deleting.soft = [[], [31]], False
    assert delete(dataset) == {"in_use": True, "chart_ids": [31], "restored": False}
    assert deleting.superset.ran == [("DeleteDatasetCommand", [4])]
    # A restore Superset refuses is answered as not restored, never with its words.
    deleting.superset.ran.clear()
    deleting.charts, deleting.soft = [[], [31]], True
    deleting.superset.restore = stubs.DatasetRestoreFailedError("Superset's words")
    assert delete(dataset) == {"in_use": True, "chart_ids": [31], "restored": False}


def test_delete_asks_for_editorship_before_the_marker(deleting):
    """Not being an editor comes first, as in every tool: no retry changes it, and its answer names whom to ask."""
    deleting.editor = False
    for description in (None, format_marker(BUILD, "data", 123), f"Made by hand.\n{MARKED}\nmore", MARKED):
        assert delete(_dataset(description)) == {"not_editor": True, "editors": ["analyst Live"]}
    assert deleting.superset.ran == []


def test_a_command_failure_is_answered_by_its_kind(deleting):
    """A command's failure, any of Superset's exceptions, is answered without Superset's words."""
    for raised in (stubs.DeleteFailedError("Datasets could not be deleted."), stubs.SupersetException("words")):
        deleting.charts = [[]]
        deleting.superset.delete = raised
        assert delete(_dataset(MARKED)) == {"failed": True}
    deleting.charts = [[]]
    deleting.superset.delete = stubs.ForbiddenError()
    assert delete(_dataset(MARKED)) == {"not_editor": True, "editors": ["analyst Live"]}


# ---------------------------------------------------------------- waiting on a lock

class Slice:
    """A model stood in: `lock_row` reads its `id` column and its name."""

    id = "slices.id"


class _Query:
    def __init__(self, log: list) -> None:
        self.log = log

    def __getattr__(self, name):
        def step(*args):
            self.log.append((name, *args))
            return "the row" if name == "one_or_none" else self

        return step


class _Session:
    """A session stood in, on a database of `dialect`: it records what `lock_row` executes and queries."""

    def __init__(self, dialect: str) -> None:
        self.log: list = []
        self.rollbacks = 0
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect))

    def get_bind(self):
        return self.bind

    def execute(self, statement, parameters=None) -> None:
        self.log.append(("execute", statement, parameters))

    def query(self, model):
        self.log.append(("query", model))
        return _Query(self.log)

    def rollback(self) -> None:
        self.rollbacks += 1


LOCK_TIMEOUT = "SELECT set_config('lock_timeout', :timeout, true)"


def test_the_row_lock_waits_at_most_the_lock_timeout(superset, monkeypatch):
    """On Postgres, `lock_row` first sets the transaction's `lock_timeout` (`SET LOCAL`, as `set_config(..., true)`),
    30 seconds unless Superset's config sets `MNEMIQ_LOCK_TIMEOUT_S`, and only then takes the row's lock. So a tool
    stuck behind a long transaction stops waiting well inside mnemiq's deadline for the call (45 seconds by default)
    with SQLSTATE 55P03, which rc2 counts as contention, and answers `{busy}` instead of writing after mnemiq gave up."""
    session = _Session("postgresql")
    monkeypatch.setattr(writes, "_session", lambda: session)
    assert writes.lock_row(Slice, 4) == "the row"
    assert session.log[0] == ("execute", LOCK_TIMEOUT, {"timeout": "30000ms"})
    assert [entry[0] for entry in session.log[1:]] == ["query", "populate_existing", "enable_eagerloads", "filter",
                                                      "with_for_update", "one_or_none"]
    for configured, waits in ((3, "3000ms"), (2.5, "2500ms"), (0, "30000ms"), (-1, "30000ms"), ("10", "30000ms"),
                              (True, "30000ms"), (float("nan"), "30000ms")):
        session.log.clear()
        superset.config["MNEMIQ_LOCK_TIMEOUT_S"] = configured
        writes.lock_row(Slice, 4)
        assert session.log[0] == ("execute", LOCK_TIMEOUT, {"timeout": waits}), configured
    # SQLite takes no row locks and MySQL has no transaction-local lock timeout: the lock is taken as it was.
    for dialect in ("sqlite", "mysql"):
        session = _Session(dialect)
        monkeypatch.setattr(writes, "_session", lambda: session)
        writes.lock_row(Slice, 4)
        assert session.log[0] == ("query", Slice)


def test_a_lock_timeout_at_or_past_the_call_deadline_is_not_used(superset, monkeypatch, caplog):
    """`MNEMIQ_LOCK_TIMEOUT_S` is used only when it is a number of seconds above 0 and below mnemiq's deadline for a
    call (45 seconds by default). At or past the deadline a write could land after mnemiq gave up on it; past
    Postgres' largest `lock_timeout` (2147483647 ms) the `set_config` itself fails; and a value too large to be a
    number of milliseconds (1e306 seconds) fails before it is sent. Each falls back to the default 30 seconds, with
    a warning naming the setting. The controls: values just inside the bounds are used as they are, without one."""
    session = _Session("postgresql")
    monkeypatch.setattr(writes, "_session", lambda: session)
    for configured in (45, 45.0, 60, 2_147_484, 10**400, 1e306, float("inf")):
        session.log.clear()
        caplog.clear()
        superset.config["MNEMIQ_LOCK_TIMEOUT_S"] = configured
        with caplog.at_level("WARNING", logger=writes.__name__):
            writes.lock_row(Slice, 4)
        assert session.log[0] == ("execute", LOCK_TIMEOUT, {"timeout": "30000ms"}), configured
        assert ["MNEMIQ_LOCK_TIMEOUT_S" in record.getMessage() for record in caplog.records] == [True], configured
    for configured, waits in ((44.9, "44900ms"), (0.001, "1ms"), (3, "3000ms")):
        session.log.clear()
        caplog.clear()
        superset.config["MNEMIQ_LOCK_TIMEOUT_S"] = configured
        with caplog.at_level("WARNING", logger=writes.__name__):
            writes.lock_row(Slice, 4)
        assert session.log[0] == ("execute", LOCK_TIMEOUT, {"timeout": waits}), configured
        assert caplog.records == [], configured


def test_a_contended_lock_answers_busy_and_ends_the_transaction(superset, monkeypatch):
    """A lock wait past the timeout (55P03), a deadlock (40P01) or a serialization failure (40001) on the lock answers
    `{busy}`: the same request may be sent again. Nothing is written and the transaction ends. Any other database
    error is not contention: it is raised, and the transaction still ends."""
    session = _Session("postgresql")
    monkeypatch.setattr(writes, "_session", lambda: session)
    wrote: list = []
    for sqlstate in stubs.PG_LOCK_CONTENTION:
        monkeypatch.setattr(writes, "lock_row", lambda model, entity_id, code=sqlstate: (_ for _ in ()).throw(
            stubs.database_error(code)))
        assert writes._locked(Slice, 4, wrote.append) == {"busy": True}, sqlstate
    assert wrote == [] and session.rollbacks == 3
    monkeypatch.setattr(writes, "lock_row", lambda model, entity_id: (_ for _ in ()).throw(
        stubs.database_error("42P01")))
    with pytest.raises(stubs.DBAPIError):
        writes._locked(Slice, 4, wrote.append)
    assert wrote == [] and session.rollbacks == 4


def test_a_command_that_loses_a_lock_race_answers_busy(deleting):
    """A command whose own write waits past the lock timeout fails as its command's error raised from the database
    error (`superset/utils/decorators.py`, `on_error`: `raise reraise() from ex`), and the tool answers `{busy}`. The
    control: the same command failure from any other database error answers `{failed}`."""
    for sqlstate, answer in (("55P03", {"busy": True}), ("40P01", {"busy": True}), ("23505", {"failed": True})):
        failure = stubs.DeleteFailedError()
        failure.__cause__ = stubs.database_error(sqlstate)
        deleting.charts, deleting.superset.delete = [[]], failure
        assert delete(_dataset(MARKED)) == answer, sqlstate


# ---------------------------------------------------------------- the metadata database, stood in

def _holds(predicate: tuple, row: SimpleNamespace) -> bool:
    """Whether `row` passes a filter a stood-in column made (`stubs.Column`), a `LIKE` decided by SQLite's own."""
    kind, column, *operands = predicate
    value = getattr(row, column)
    if kind == "is":
        return value is operands[0]
    pattern, escape = operands
    with contextlib.closing(sqlite3.connect(":memory:")) as database:
        if escape is None:
            return database.execute("SELECT ? LIKE ?", (value, pattern)).fetchone()[0] == 1
        return database.execute("SELECT ? LIKE ? ESCAPE ?", (value, pattern, escape)).fetchone()[0] == 1


class _Lookup:
    """A query of one model, stood in: its filters applied to the model's rows, in the order it asks for."""

    def __init__(self, metadata: "Metadata", model: type) -> None:
        self.metadata, self.model, self.predicates, self.ordered = metadata, model, [], []

    def populate_existing(self):
        return self

    def filter(self, *predicates):
        self.predicates.extend(predicates)
        return self

    def order_by(self, *columns):
        self.ordered.extend(column.name for column in columns)
        return self

    def __iter__(self):
        self.metadata.log.append(("lookup", self.model.__name__))
        self.metadata.at("lookup")
        self.metadata.statement()
        self.metadata.occurs("lookup")
        rows = [row for row in self.metadata.superset.rows[self.model.__name__]
                if all(_holds(predicate, row) for predicate in self.predicates)]
        return iter(sorted(rows, key=lambda row: [getattr(row, name) for name in self.ordered]))

    def all(self) -> list:
        return list(self)


# What `Metadata.holds_answer` is when no test set it: the hold check then answers whether the connection holds the
# lock. A sentinel, so that a test may set None, MySQL's answer for a lock nobody holds.
_UNSET = object()


class _Connection:
    """A connection of its own, as `engine.connect()` hands one out, on which a create takes its marker's lock. What
    runs on it is logged as `("on", n, ...)`: its transaction (`begin`) as `"begin"`, then `"commit"`, or `"rollback"`
    when its block raises; each statement with its parameters. Its lock statements take a real lock of the
    metadata's (`Metadata.locks`), keyed by the marker (Postgres) or the lock's name (MySQL), so requests in two
    threads wait for each other as two sessions of the database do: Postgres's `pg_advisory_lock` for the
    `lock_timeout` set in its transaction, then failing with SQLSTATE 55P03; MySQL's `GET_LOCK` for its seconds, then
    answering 0 (or the metadata's `got_lock`, when that is not 1, without taking the lock). The lock outlives the
    transaction it was taken in, as a session's lock does. The hold check answers whether this connection holds the
    marker's lock (`HOLDS`, Postgres: true or false; `MYSQL_HOLDS`, MySQL: 1 or 0), or the metadata's `holds_answer`
    when a test set it (None among them, MySQL's NULL for a lock nobody holds; raised, when it is an exception, the
    connection kept). An unlock answers the metadata's `released` (raised, when it is an exception), or else the
    database's own answer: Postgres true when this database session holds the lock and false when it does not; MySQL 1
    when it does, 0 when another session does, NULL when none does. It lets the lock go only when that is 1.

    As SQLAlchemy 2.0's `Connection` does: a statement on a connection whose database session ended (`lost`) fails,
    and the failure invalidates the connection (`("invalidate", n)`, `invalidated`); the next statement on an
    invalidated connection quietly reconnects it (`("reconnect", n)`), to a fresh database session that holds nothing.
    Invalidating the connection lets go whatever it holds, as closing a database session does, and invalidating it
    again does nothing; closing it while it still holds a lock leaves the lock held, as a pool's reset does, and
    records it in the metadata's `leaked`."""

    def __init__(self, metadata: "Metadata", number: int) -> None:
        self.metadata, self.number = metadata, number
        self.timeout_s: float | None = None
        self.holding: str | None = None
        self.lost = False  # its database session ended: its next statement fails, and it holds no lock
        self.invalidated = False  # SQLAlchemy closed its database connection; the next statement reconnects
        metadata.log.append(("connect", number))
        metadata.opened.append(self)

    def _log(self, *entry) -> None:
        self.metadata.log.append(("on", self.number, *entry))

    @contextlib.contextmanager
    def begin(self):
        self._log("begin")
        try:
            yield
        except BaseException:
            self._log("rollback")
            raise
        finally:
            self.timeout_s = None  # `set_config(..., true)` holds until the transaction ends
        self._log("commit")

    def execute(self, statement, parameters=None) -> SimpleNamespace:
        if self.invalidated:  # SQLAlchemy's `_revalidate_connection`: a fresh pooled connection, a new session
            self.metadata.log.append(("reconnect", self.number))
            self.invalidated = self.lost = False
        self._log(statement, parameters)
        metadata = self.metadata
        if self.lost:  # SQLAlchemy finds the connection lost, invalidates it, and raises
            self.invalidate()
            raise stubs.database_error("08006", invalidated=True)
        if metadata.lock_error is not None and statement in (PG_LOCK, GET_LOCK):
            raise metadata.lock_error
        if metadata.contended is not None and metadata.contended in statement:
            raise stubs.database_error("55P03")
        answer: Any = 1
        if statement == LOCK_TIMEOUT:
            self.timeout_s = int(parameters["timeout"].removesuffix("ms")) / 1000
        elif statement == PG_LOCK:
            if not self._take(parameters["marker"], self.timeout_s) or metadata.late_grant:
                raise stubs.database_error("55P03")
            answer = None  # `pg_advisory_lock` returns void
        elif statement == GET_LOCK:
            answer = metadata.got_lock
            if answer == 1:
                answer = int(self._take(parameters["name"], parameters["seconds"]))
        elif statement in (PG_UNLOCK, RELEASE_LOCK):
            answer = self._release(statement, parameters["marker" if statement == PG_UNLOCK else "name"])
        elif statement in (HOLDS, MYSQL_HOLDS):
            if isinstance(metadata.holds_answer, BaseException):
                raise metadata.holds_answer
            if metadata.holds_answer is not _UNSET:
                answer = metadata.holds_answer
            elif statement == HOLDS:
                answer = self.holding == parameters["marker"]
            else:
                answer = int(self.holding == parameters["name"])
            if metadata.after_hold_check is not None:
                meanwhile, metadata.after_hold_check = metadata.after_hold_check, None
                meanwhile()
        return SimpleNamespace(scalar=lambda: answer)

    def _take(self, key: str, seconds: float | None) -> bool:
        lock = self.metadata.locks.setdefault(key, threading.Lock())
        if not lock.acquire(blocking=False):
            self.metadata.waiting.append(self.number)
            try:
                if not lock.acquire(timeout=30 if seconds is None else seconds):
                    return False
            finally:
                self.metadata.waiting.remove(self.number)
        self.holding = key
        return True

    def _release(self, statement: str, key: str) -> Any:
        answer = self.metadata.released
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:  # the database's own answer: Postgres a boolean, MySQL a number or NULL
            if statement == PG_UNLOCK:
                answer = self.holding == key
            elif self.holding == key:
                answer = 1
            else:
                lock = self.metadata.locks.get(key)
                answer = 0 if lock is not None and lock.locked() else None
        if answer == 1 and self.holding is not None:
            self._let_go()
        return answer

    def _let_go(self) -> None:
        self.metadata.locks[self.holding].release()
        self.holding = None

    def invalidate(self) -> None:
        if self.invalidated:  # SQLAlchemy's `invalidate` does nothing on a connection already invalidated
            return
        self.invalidated = True
        self.metadata.log.append(("invalidate", self.number))
        if self.holding is not None:
            self._let_go()

    def close(self) -> None:
        self.metadata.log.append(("close", self.number))
        if self.holding is not None:
            self.metadata.leaked.append(self.holding)


class Metadata:
    """Superset's metadata database on `dialect`, with its session (`db.session`), stood in. Its rows are the
    stand-ins' `rows`, and what it does goes to the stand-ins' `ran`, in order with the commands run there:
    `("isolation",)` when the session's connection's isolation level is read, `("execute", statement, parameters)`
    for each statement the session executes, `("lookup", model)` when a query is read, `("commit",)` and
    `("rollback",)`; and, on a connection of its own (`engine.connect()`, `_Connection`), `("connect", n)`, `("on", n,
    ...)`, `("invalidate", n)` and `("close", n)`. A test may set `isolation` (what the session's connection reports),
    `contended` (a text whose statement on a connection of its own waits past the lock timeout, SQLSTATE 55P03),
    `got_lock` (what `GET_LOCK` answers without taking the lock, when not 1), `lock_error` (what the lock statement
    raises instead), `late_grant` (Postgres grants the lock just as its wait times out, and the statement still fails
    with 55P03, the lock held), `released` (what an unlock answers, or raises), `holds_answer` (what the hold check
    answers instead of whether the connection holds the lock, when a test set it: None too; or raises, the connection
    kept), `pool_exhausted`
    (`engine.connect()` times out),
    `aborted` (the session's transaction is aborted, as Postgres aborts one a statement failed in: every statement fails
    with SQLSTATE 25P02 until a rollback), `gone` (the session's connection is gone: every statement fails, a rollback
    or not; SQLAlchemy marks the error as an invalidated connection), `breaks` (what of `aborted` or `gone` happens,
    once, when the session reaches a moment: `"isolation"` or `"lookup"`), `before_create_lock` (run once, just
    before the session takes a create's second lock: Postgres's transaction lock, or MySQL's row lock) and
    `after_hold_check` (run once, on the lock's connection, once the hold check has its answer and before it is
    returned). The session's transaction locks (`CREATE_LOCK`) are real locks too, keyed by the marker, held until the
    session commits or rolls back; so are the row locks `lock_row` takes (`Metadata.lock_row`, logged as
    `("lock_row", model, id)`), keyed by `(model, id)`. The session's transaction state, whether it is `aborted` and
    the transaction and row locks it holds (`held_xact`), is each thread's own, as Superset's scoped session is one
    session per thread: two sends in two threads each roll back only their own transaction. `waiting` holds the
    connections waiting for a lock, `waiting_xact` the threads whose session waits for a transaction or row lock, and
    `leaked` each lock a connection still held when it was closed. `drop_lock_connections()` ends the database session
    of every connection of its own: their locks are let go, and every later statement on them fails. `fail_at` maps a
    session statement, `"lookup"` or `"lock_row"`, to `(n, error)`: its n-th occurrence on the session raises
    `error`."""

    def __init__(self, superset: SimpleNamespace, dialect: str) -> None:
        self.superset = superset
        self._transaction = threading.local()  # each thread's session's own transaction state
        self.isolation, self.contended, self.got_lock, self.pool_exhausted = "READ COMMITTED", None, 1, False
        self.lock_error: BaseException | None = None
        self.late_grant = False
        self.released: Any = None
        self.holds_answer: Any = _UNSET
        self.aborted = self.gone = False
        self.breaks: dict[str, str] = {}
        self.before_create_lock: Callable[[], None] | None = None
        self.after_hold_check: Callable[[], None] | None = None
        self.locks: dict[str, threading.Lock] = {}
        self.xact_locks: dict[str, threading.Lock] = {}
        self.waiting_xact: list[str] = []
        self.opened: list[_Connection] = []
        self.fail_at: dict[str, tuple[int, BaseException]] = {}
        self.occurrences: dict[str, int] = {}
        self.waiting: list[int] = []
        self.leaked: list[str] = []
        self._numbers = itertools.count(1)
        self.engine = SimpleNamespace(dialect=SimpleNamespace(name=dialect), connect=self._connect)

    @property
    def aborted(self) -> bool:
        """Whether this thread's session's transaction is aborted."""
        return getattr(self._transaction, "aborted", False)

    @aborted.setter
    def aborted(self, value: bool) -> None:
        self._transaction.aborted = value

    @property
    def held_xact(self) -> list[str]:
        """The transaction locks this thread's session holds, by marker."""
        if not hasattr(self._transaction, "held_xact"):
            self._transaction.held_xact = []
        return self._transaction.held_xact

    def at(self, moment: str) -> None:
        """The session reaches `moment`: what `breaks` names for it happens now, once."""
        if (broken := self.breaks.pop(moment, None)) is not None:
            setattr(self, broken, True)

    def drop_lock_connections(self) -> None:
        for connection in self.opened:
            if connection.holding is not None:
                connection._let_go()
            connection.lost = True

    def occurs(self, key: str) -> None:
        """One more `key` (a statement, or `"lookup"`) on the session: the error `fail_at` names for this occurrence is
        raised."""
        self.occurrences[key] = self.occurrences.get(key, 0) + 1
        if key in self.fail_at and self.fail_at[key][0] == self.occurrences[key]:
            raise self.fail_at[key][1]

    def statement(self) -> None:
        """A statement on the session, which fails while its connection is gone or its transaction is aborted."""
        if self.gone:
            raise stubs.database_error("08006", invalidated=True)
        if self.aborted:
            raise stubs.database_error("25P02")

    @property
    def log(self) -> list:
        return self.superset.ran

    def get_bind(self) -> SimpleNamespace:
        return self.engine

    def connection(self) -> SimpleNamespace:
        return SimpleNamespace(get_isolation_level=self._isolation)

    def _isolation(self) -> str:
        self.log.append(("isolation",))
        self.at("isolation")
        self.statement()
        return self.isolation

    def _connect(self) -> _Connection:
        if self.pool_exhausted:
            raise stubs.PoolTimeoutError("QueuePool limit of size 5 overflow 10 reached")
        return _Connection(self, next(self._numbers))

    def execute(self, statement, parameters=None) -> None:
        if statement == CREATE_LOCK:
            self._before_second_lock()
        self.log.append(("execute", statement, parameters))
        self.statement()
        self.occurs(statement)
        if self.contended is not None and self.contended in statement:
            raise stubs.database_error("55P03")
        if statement == CREATE_LOCK:
            self._take_xact(parameters["marker"])

    def lock_row(self, model: type, entity_id: int | None) -> Any:
        """`writes.lock_row` on this database, for the rows a create's second lock locks on MySQL: a dashboard (the
        stand-ins' `dashboards`, by id) or the request's own user (`g.user`, by its id). The row, locked until the
        session's transaction ends, or None when there is no such row, an id of None among them (`id = NULL` matches no
        row), and no lock taken."""
        self._before_second_lock()
        self.log.append(("lock_row", model.__name__, entity_id))
        self.statement()
        self.occurs("lock_row")
        if model.__name__ == "Dashboard":
            row = self.superset.dashboards.get(entity_id)
        else:
            assert model.__name__ == "User", f"the fake locks no {model.__name__} row"
            user = getattr(self.superset.g, "user", None)
            row = user if user is not None and entity_id is not None and user.id == entity_id else None
        if row is not None:
            self._take_xact((model.__name__, entity_id))
        return row

    def _before_second_lock(self) -> None:
        if self.before_create_lock is not None:
            meanwhile, self.before_create_lock = self.before_create_lock, None
            meanwhile()

    def _take_xact(self, key: Any) -> None:
        """Takes the lock `key` for this thread's session's transaction, held until it commits or rolls back: at once
        when the transaction holds it already, as a transaction locking its own row again does; else waiting up to 5
        seconds for another thread's session to let it go, then failing with SQLSTATE 55P03."""
        if key in self.held_xact:
            return
        lock = self.xact_locks.setdefault(key, threading.Lock())
        if not lock.acquire(blocking=False):
            self.waiting_xact.append(threading.current_thread().name)
            try:
                if not lock.acquire(timeout=5):
                    raise stubs.database_error("55P03")
            finally:
                self.waiting_xact.remove(threading.current_thread().name)
        self.held_xact.append(key)

    def _end_transaction(self) -> None:
        while self.held_xact:
            self.xact_locks[self.held_xact.pop()].release()

    def query(self, model: type) -> _Lookup:
        return _Lookup(self, model)

    def commit(self) -> None:
        self.log.append(("commit",))
        self._end_transaction()

    def rollback(self) -> None:
        self.log.append(("rollback",))
        self.aborted = False
        self._end_transaction()


def _all_let_go(metadata: Metadata) -> bool:
    """Whether every lock of `metadata` is free, whichever thread's session or connection took it: the markers' locks,
    the create locks and the row locks."""
    return not any(lock.locked() for lock in (*metadata.locks.values(), *metadata.xact_locks.values()))


def commands(superset: SimpleNamespace, *names: str) -> list:
    """The commands run, each with its arguments: those `names` names, or every one."""
    return [entry for entry in superset.ran if entry[0].endswith("Command") and (not names or entry[0] in names)]


# ---------------------------------------------------------------- set_layout's access check

def _one_chart_layout(chart_id: int) -> dict:
    return {"ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
            "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [f"CHART-{chart_id}"]},
            f"CHART-{chart_id}": {"type": "CHART", "id": f"CHART-{chart_id}", "children": [],
                                  "meta": {"chartId": chart_id, "width": 4, "height": 50}}}


def test_set_layout_adds_only_charts_the_caller_may_read(superset, monkeypatch):
    """`DashboardDAO.set_dash_metadata` makes every chart a layout names a member with no access check of its own, so
    `set_layout` asks `can_access_chart` of each chart it adds: one the caller may not read, or that does not exist,
    is refused by id, and no command runs. The control: a chart the caller may read is added, in one command."""
    superset.charts = {11: SimpleNamespace(id=11), 12: SimpleNamespace(id=12)}
    security = SimpleNamespace(can_access_dashboard=lambda dashboard: True, is_editor=lambda resource: True,
                               can_access_chart=lambda chart: chart.id == 11)
    monkeypatch.setattr(writes, "_security", lambda: security)
    monkeypatch.setattr(writes, "_member_chart_ids", lambda dashboard_id: [])
    dashboard = SimpleNamespace(id=3, changed_on=datetime.fromisoformat(WHEN), position_json=None, json_metadata=None,
                                editors=[])

    def layout(chart_id: int) -> dict:
        found = writes.read("set_layout", {"dashboard_id": 3, "positions": _one_chart_layout(chart_id),
                                           "add_chart_ids": [chart_id], "expect_changed_on": WHEN})
        return writes._layout(found, dashboard)

    assert layout(12) == {"refused": "chart_access", "chart_ids": [12]}
    assert layout(13) == {"refused": "chart_access", "chart_ids": [13]}
    assert superset.ran == []
    assert layout(11) == {"ok": True}
    assert [name for name, _ in superset.ran] == ["UpdateDashboardCommand"]
    assert writes._readable_chart(11) is True
    assert writes._readable_chart(12) is False and writes._readable_chart(13) is False


def test_set_layout_does_not_expect_a_trashed_member_the_layout_no_longer_places(superset, monkeypatch):
    """rc2 keeps a chart in the trash a member of its dashboard. `set_layout` reads which members are trashed, so a
    layout that leaves out one the dashboard no longer places (a Remove took it off) is the whole set; the control:
    the same layout, the member not read as trashed, is refused as dropping it."""
    superset.charts = {11: SimpleNamespace(id=11), 12: SimpleNamespace(id=12)}
    security = SimpleNamespace(can_access_dashboard=lambda dashboard: True, is_editor=lambda resource: True,
                               can_access_chart=lambda chart: True)
    monkeypatch.setattr(writes, "_security", lambda: security)
    monkeypatch.setattr(writes, "_member_chart_ids", lambda dashboard_id: [11, 9])
    dashboard = SimpleNamespace(id=3, changed_on=datetime.fromisoformat(WHEN),
                                position_json=json.dumps(_one_chart_layout(11)), json_metadata=None, editors=[])
    positions = _one_chart_layout(11)
    positions["GRID_ID"]["children"].append("CHART-12")
    positions["CHART-12"] = {"type": "CHART", "id": "CHART-12", "children": [],
                             "meta": {"chartId": 12, "width": 4, "height": 50}}

    def layout() -> dict:
        found = writes.read("set_layout", {"dashboard_id": 3, "positions": positions, "add_chart_ids": [12],
                                           "expect_changed_on": WHEN})
        return writes._layout(found, dashboard)

    monkeypatch.setattr(writes, "_trashed_chart_ids", lambda chart_ids: [])
    assert layout() == {"refused": "chart_set", "missing": [9], "unexpected": []}
    asked: list = []
    monkeypatch.setattr(writes, "_trashed_chart_ids", lambda chart_ids: asked.append(chart_ids) or [9])
    assert layout() == {"ok": True}
    assert asked == [[11, 9]]


# ---------------------------------------------------------------- who becomes an editor

USER, ROLE, GROUP = stubs.SubjectType.USER, stubs.SubjectType.ROLE, stubs.SubjectType.GROUP
DATASET_PERM = "[warehouse].[sales](id:4)"
SCHEMA_PERM = "[warehouse].[public]"
DATABASE_PERM = "[warehouse].(id:1)"


def _person(user_id: int, name: str, grants=(), member_of=()) -> SimpleNamespace:
    """A user, its own subject (id 100 + its id) and the role and group subjects it is a member of; `grants` are the
    `(permission, view)` pairs its roles hold."""
    user = SimpleNamespace(id=user_id, username=name, grants=set(grants))
    user.subject = SimpleNamespace(id=100 + user_id, type=USER, label=f"{name} Live", user=user)
    user.subject_ids = {user.subject.id, *member_of}
    return user


class Security:
    """rc2's security manager, stood in for the checks a create makes, each read as `flask.g.user`, as rc2 reads them
    (`superset/security/manager.py`): `can_access` is a grant of the user's roles; `can_access_database` is
    all-datasource, all-database or database access, and `can_access_schema` that or schema access; `is_editor` is
    any of the user's subjects (its own, its roles', its groups') among the object's editors; and
    `can_access_datasource` is `raise_for_access`'s datasource leg, which counts editorship; `raise_for_access`, for
    a table or a dataset's SQL on a database, admits the user whose grants reach the database. `judged` records each
    user the checks were asked as, and `log` each `raise_for_access` asked."""

    def __init__(self, g, raises_for: str | None = None, log: list | None = None) -> None:
        self.g, self.raises_for, self.judged = g, raises_for, []
        self.log = [] if log is None else log
        self.unparseable, self.fails = False, None

    def raise_for_access(self, database=None, table=None, sql=None, catalog=None, schema=None) -> None:
        """rc2's `raise_for_access` as `CreateDatasetCommand.validate` asks it, for a table or for a dataset's SQL:
        `SupersetSecurityException` unless the user's grants reach the database, `SupersetParseError` for SQL when
        `unparseable` is set, and `fails` (a database error reading the metadata) when set. Recorded in `log` with the
        user and every argument."""
        self.log.append(("raise_for_access", self.g.user.username, {
            "database": database, "table": table, "sql": sql, "catalog": catalog, "schema": schema}))
        if self.fails is not None:
            raise self.fails
        if sql is not None and self.unparseable:
            raise stubs.SupersetParseError("the SQL does not parse")
        if not self.can_access_database(database):
            raise stubs.SupersetSecurityException("the table may not be read")

    def can_access(self, permission: str, view: str) -> bool:
        user = self.g.user
        self.judged.append(user.username)
        if user.username == self.raises_for:
            raise RuntimeError("the metadata database went away")
        return (permission, view) in user.grants

    def can_access_database(self, database) -> bool:
        return (self.can_access("all_datasource_access", "all_datasource_access")
                or self.can_access("all_database_access", "all_database_access")
                or self.can_access("database_access", database.perm))

    def can_access_schema(self, dataset) -> bool:
        return self.can_access_database(dataset.database) or self.can_access("schema_access", dataset.schema_perm)

    def is_editor(self, resource) -> bool:
        return bool(self.g.user.subject_ids & {subject.id for subject in resource.editors})

    def can_access_datasource(self, dataset) -> bool:
        return (self.can_access_schema(dataset) or self.can_access("datasource_access", dataset.perm)
                or self.is_editor(dataset))

    def can_access_dashboard(self, dashboard) -> bool:
        return True

    def can_access_chart(self, chart) -> bool:
        return True


@pytest.fixture
def claiming(monkeypatch, superset):
    """A dashboard (3) a builder edits with six co-editors, and a dataset (4) on table `sales`:
    - `builder`, the caller, who may read every dataset and made the dataset;
    - `reader`, granted `datasource_access` on the dataset, and `database_reader`, granted its database: both may read
      it, and SQL Lab's strict check admits both to its table;
    - `schema_reader`, granted its schema: it may read the dataset, but SQL Lab's strict check skips schema access, so
      only an editor of a dataset over the table would reach the table there;
    - `outsider`, granted nothing, and `member`, granted nothing but a member of the role `R2`, which already edits the
      dataset: neither may read it by a grant (`member` reads it only as an editor);
    - the role `R` and the group `G`, editors of the dashboard.
    The dataset's editors: its creator `creator` (not on the dashboard), the builder and `R2`."""
    people = {name: _person(user_id, name, grants, member_of) for user_id, name, grants, member_of in (
        (1, "builder", {("all_datasource_access", "all_datasource_access")}, ()),
        (2, "reader", {("datasource_access", DATASET_PERM)}, ()),
        (3, "outsider", set(), ()),
        (4, "creator", set(), ()),
        (5, "schema_reader", {("schema_access", SCHEMA_PERM)}, ()),
        (6, "member", set(), (302,)),
        (7, "database_reader", {("database_access", DATABASE_PERM)}, ()))}
    role, group, role_2 = (SimpleNamespace(id=301, type=ROLE, label="R", user=None),
                           SimpleNamespace(id=401, type=GROUP, label="G", user=None),
                           SimpleNamespace(id=302, type=ROLE, label="R2", user=None))
    builder = people["builder"]
    superset.g.user = builder
    superset.user_subjects = {person.id: person.subject for person in people.values()}
    dashboard = SimpleNamespace(id=3, editors=[people[name].subject for name in (
        "builder", "reader", "outsider", "schema_reader", "member", "database_reader")] + [role, group])
    dataset = SimpleNamespace(id=4, perm=DATASET_PERM, schema_perm=SCHEMA_PERM, description=None,
                              database=SimpleNamespace(perm=DATABASE_PERM),
                              editors=[people["creator"].subject, builder.subject, role_2])
    superset.dashboards, superset.datasets = {3: dashboard}, {4: dataset}
    security = superset.security = Security(superset.g, log=superset.ran)
    monkeypatch.setattr(writes, "_security", lambda: security)
    superset.session = Metadata(superset, "sqlite")
    locked = {"SqlaTable": dataset}
    monkeypatch.setattr(writes, "lock_row", lambda model, entity_id: locked[model.__name__])
    return SimpleNamespace(superset=superset, security=security, people=people, dashboard=dashboard,
                           dataset=dataset, locked=locked, ids={name: p.subject.id for name, p in people.items()})


def _sent_editors(superset, command: str) -> list[int]:
    """The editors the one `command` run sent: an update command takes `(id, properties)`, a create `properties`."""
    (properties,) = [arguments if isinstance(arguments, dict) else arguments[-1]
                     for _, arguments in commands(superset, command)]
    return properties["editors"]


def _found(claiming) -> dict:
    """`create_dataset` sent for the `claiming` dataset (4), which already carries the request's marker, as an earlier
    send of it left it: found by its marker, and its editors set again."""
    claiming.dataset.deleted_at = None
    claiming.dataset.description = DATA
    claiming.superset.rows["SqlaTable"].append(claiming.dataset)
    return make_dataset()


def test_a_found_dataset_keeps_its_editors_and_adds_only_those_who_could_already_read_it(claiming):
    """rc2 admits a dataset's editors to its data (`raise_for_access`) and, in SQL Lab, to its table, and an editor
    subject that is a role or a group makes every member an editor. So `create_dataset` keeps the dataset's editors,
    adds the caller, and adds only those of the dashboard's users whose grants already reach both the dataset and, in
    SQL Lab's strict check, its table, judged as each of them: `reader` (datasource access) and `database_reader`
    (database access). Not `schema_reader`, whom only editorship would admit to the table in SQL Lab; not `outsider`;
    not `member` (who reads it only as an editor, through `R2`); never the role `R` or the group `G`. The request's
    user is the builder again afterwards."""
    ids = claiming.ids
    assert _found(claiming) == {"dataset_id": 4}
    assert commands(claiming.superset, "CreateDatasetCommand") == []
    assert _sent_editors(claiming.superset, "UpdateDatasetCommand") == [
        ids["creator"], ids["builder"], 302, ids["reader"], ids["database_reader"]]
    assert {"reader", "schema_reader", "outsider", "member", "database_reader"} <= set(claiming.security.judged)
    assert claiming.superset.g.user is claiming.people["builder"]


def test_the_request_user_comes_back_when_a_check_fails(claiming):
    """A check that raises while the create judges a co-editor leaves the request's user the builder, which rc2's
    `override_user` would not (it puts the user back only when its block returns), and saves nothing."""
    claiming.security.raises_for = "outsider"
    with pytest.raises(RuntimeError):
        _found(claiming)
    assert claiming.superset.g.user is claiming.people["builder"]
    assert commands(claiming.superset) == []


def test_a_dashboard_user_subject_without_a_user_is_skipped(claiming):
    """A user subject whose `user` is None (its account is gone) is neither judged nor made an editor."""
    gone = SimpleNamespace(id=199, type=USER, label="Gone", user=None)
    claiming.dashboard.editors.append(gone)
    _found(claiming)
    assert 199 not in _sent_editors(claiming.superset, "UpdateDatasetCommand")


def test_as_user_removes_the_user_it_set_when_the_request_had_none(superset):
    """Where `g` had no user before, `_as_user` leaves it with none again, rather than the user it ran as."""
    assert not hasattr(superset.g, "user")
    with writes._as_user(SimpleNamespace(id=5)):
        assert superset.g.user.id == 5
    assert not hasattr(superset.g, "user")


@pytest.mark.parametrize("features, promiscuous", [(set(), False), ({"ENABLE_VIEWERS"}, False), (set(), True)])
def test_a_chart_gets_the_dashboards_editors(claiming, features, promiscuous):
    """A chart's editors may not read its data for being its editors unless `ENABLE_VIEWERS` is on and
    `VIEWER_PROMISCUOUS_MODE` is set (`has_promiscuous_chart_access` in rc2's `raise_for_access`), so short of both a
    created chart gets the dashboard's editors, every one, roles and groups included."""
    claiming.superset.features, claiming.superset.config["VIEWER_PROMISCUOUS_MODE"] = features, promiscuous
    everyone = [subject.id for subject in claiming.dashboard.editors]
    made = writes.create_chart({"dataset_id": 4, "viz_type": "funnel", "slice_name": "stages",
                                "params": {"groupby": ["status"], "metric": "count"}, "dashboard_id": 3,
                                "marker": format_marker(BUILD, "stages", 3)})
    assert set(made) == {"chart_id"}, made
    assert _sent_editors(claiming.superset, "CreateChartCommand") == everyone


def test_a_chart_whose_editors_read_its_data_gets_the_dataset_rule(claiming):
    """With `ENABLE_VIEWERS` on and `VIEWER_PROMISCUOUS_MODE` set, a chart's editors may read its dataset's data, so a
    created chart gets as editors the caller and the dashboard's users whose grants already reach the chart's dataset
    and its table (`reader`, `database_reader`; not `schema_reader`)."""
    claiming.superset.features, claiming.superset.config["VIEWER_PROMISCUOUS_MODE"] = {"ENABLE_VIEWERS"}, True
    ids = claiming.ids
    made = writes.create_chart({"dataset_id": 4, "viz_type": "funnel", "slice_name": "stages",
                                "params": {"groupby": ["status"], "metric": "count"}, "dashboard_id": 3,
                                "marker": format_marker(BUILD, "stages", 3)})
    assert set(made) == {"chart_id"}, made
    assert _sent_editors(claiming.superset, "CreateChartCommand") == [ids["builder"], ids["reader"],
                                                                       ids["database_reader"]]
    assert claiming.superset.g.user is claiming.people["builder"]


# ---------------------------------------------------------------- a create carrying a build's marker

SHELL = format_marker(BUILD, "dashboard", None)
STEP = format_marker(BUILD, "chart_1", 12)
PG_LOCK = "SELECT pg_advisory_lock(hashtextextended(:marker, 0))"
PG_UNLOCK = "SELECT pg_advisory_unlock(hashtextextended(:marker, 0))"
CREATE_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended('mnemiq-create:' || :marker, 0))"
HOLDS = writes._PG_HOLDS
GET_LOCK = "SELECT GET_LOCK(:name, :seconds)"
RELEASE_LOCK = "SELECT RELEASE_LOCK(:name)"
# MySQL's hold check, written out rather than read from `writes`, so a change of the statement is seen here: 1 when
# this connection holds the named lock, 0 when another does, NULL when nobody does.
MYSQL_HOLDS = "SELECT IS_USED_LOCK(:name) = CONNECTION_ID()"
# The request's own user in `creating`, whose row a dashboard create locks on MySQL.
CALLER_ID = 1


def _lock_name(marker: str) -> str:
    """MySQL's name for `marker`'s lock: `mnemiq:` and the marker's SHA-256, cut to 63 characters (MySQL allows 64)."""
    return "mnemiq:" + hashlib.sha256(marker.encode()).hexdigest()[:56]


def taken(dialect: str, marker: str, n: int = 1, seconds: float = 30) -> list:
    """What taking `marker`'s lock runs, with the lock timeout `seconds`: connection `n` of its own, and on it a short
    transaction, committed once the lock is held (Postgres's with its wait bounded first)."""
    if dialect == "postgresql":
        asked = [("on", n, LOCK_TIMEOUT, {"timeout": f"{round(seconds * 1000)}ms"}),
                 ("on", n, PG_LOCK, {"marker": marker})]
    else:
        asked = [("on", n, GET_LOCK, {"name": _lock_name(marker), "seconds": max(1, int(seconds))})]
    return [("connect", n), ("on", n, "begin"), *asked, ("on", n, "commit")]


def let_go(dialect: str, marker: str, n: int = 1) -> list:
    """What letting `marker`'s lock go runs: the release on connection `n`, then the connection closed."""
    if dialect == "postgresql":
        return [("on", n, PG_UNLOCK, {"marker": marker}), ("close", n)]
    return [("on", n, RELEASE_LOCK, {"name": _lock_name(marker)}), ("close", n)]


# The bound on the session's own lock waits, as `lock_row` sets it on Postgres.
BOUND = ("execute", LOCK_TIMEOUT, {"timeout": "30000ms"})


def looked(dialect: str, model: str) -> list:
    """What the look runs on the session: the isolation read, on Postgres the bound on the session's lock waits, and
    the lookup."""
    return [("isolation",), *([BOUND] if dialect == "postgresql" else []), ("lookup", model)]


@pytest.fixture
def creating(monkeypatch, superset):
    """Creates with Superset stood in, on a Postgres metadata database (`metadata`; `on(dialect)` makes it another):
    dashboard 12, edited by `analyst Live`, and dataset 7 are there; the caller, user `CALLER_ID`, reads every object
    but those in `unreadable` and edits every one but those in `not_edited`. A row lock is the metadata database's
    (`Metadata.lock_row`). Afterwards no connection was put back in the pool still holding a marker's lock."""
    state = SimpleNamespace(superset=superset, unreadable=[], not_edited=[],
                            dashboard=SimpleNamespace(id=12, editors=[SimpleNamespace(id=101, label="analyst Live")]))
    superset.g.user = SimpleNamespace(id=CALLER_ID)
    monkeypatch.setattr(writes, "lock_row", lambda model, entity_id: superset.session.lock_row(model, entity_id))

    def readable(resource) -> bool:
        return not any(resource is hidden for hidden in state.unreadable)

    security = SimpleNamespace(can_access_dashboard=readable, can_access_chart=readable,
                               can_access_datasource=readable, user_model=stubs.model("User"),
                               is_editor=lambda resource: not any(resource is kept for kept in state.not_edited))
    monkeypatch.setattr(writes, "_security", lambda: security)
    superset.dashboards, superset.datasets = {12: state.dashboard}, {7: SimpleNamespace(id=7)}
    databases: list[Metadata] = []

    def on(dialect: str) -> Metadata:
        state.metadata = superset.session = Metadata(superset, dialect)
        databases.append(state.metadata)
        return state.metadata

    state.on = on
    on("postgresql")
    yield state
    assert [leaked for database in databases for leaked in database.leaked] == []


def make_dashboard(marker: str = SHELL) -> dict:
    return writes.create_dashboard({"title": "Policies", "marker": marker})


def make_funnel(marker: str = STEP) -> dict:
    return writes.create_chart({"dataset_id": 7, "viz_type": "funnel", "slice_name": "Stages",
                                "params": {"groupby": ["status"], "metric": "count"}, "dashboard_id": 12,
                                "marker": marker})


def _row(row_id: int, description: str, deleted_at: datetime | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=row_id, description=description, deleted_at=deleted_at, editors=[])


def _kinds(superset: SimpleNamespace) -> list[str]:
    return [entry[0] for entry in superset.ran]


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_create_dashboard_twice_with_one_marker_makes_one(creating, dialect):
    """A create whose answer was lost is sent again with the same marker: the second finds the dashboard the first
    made, by its marker, and answers it; the command ran once. The guarded look just before the command would also
    find it: the second send's trail in `test_the_marker_lock_is_held_from_the_look_through_the_creates_commit` is what
    shows the first look answers it."""
    creating.on(dialect)
    first = make_dashboard()
    assert set(first) == {"dashboard_id"}, first
    assert make_dashboard() == first
    assert commands(creating.superset) == [("CreateDashboardCommand", {
        "dashboard_title": "Policies", "published": False, "description": SHELL})]


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_create_chart_twice_with_one_marker_makes_one(creating, dialect):
    creating.on(dialect)
    first = make_funnel()
    assert set(first) == {"chart_id"}, first
    assert make_funnel() == first
    ((_, properties),) = commands(creating.superset)
    assert properties["description"] == STEP and properties["editors"] == [101]


def test_the_sql_filter_reads_an_underscore_literally():
    """Every step key may hold `_`, which `LIKE` reads as any one character: the filter escapes it, and `%` and the
    escape character itself, so a marker matches only itself. Run by SQLite's own `LIKE`."""
    step, twin = format_marker(BUILD, "chart_1", 7), format_marker(BUILD, "chartx1", 7)

    def selected(suffix: str, descriptions: list[str]) -> list[str]:
        with contextlib.closing(sqlite3.connect(":memory:")) as database:
            database.execute("CREATE TABLE slices (description TEXT)")
            database.executemany("INSERT INTO slices VALUES (?)", [(text,) for text in descriptions])
            return [found for (found,) in database.execute(
                "SELECT description FROM slices WHERE description LIKE ? ESCAPE '\\' ORDER BY rowid",
                (writes._like_suffix(suffix),))]

    assert selected(step, [f"Revenue\n{twin}", f"Revenue\n{step}"]) == [f"Revenue\n{step}"]
    assert selected("a%b_c\\d", ["x a%b_c\\d", "x aZbYc\\d", "x a%b_c\\\\d"]) == ["x a%b_c\\d"]


@pytest.mark.parametrize("description", [
    pytest.param("prose" + SHELL, id="text typed before it on its line"),
    pytest.param(SHELL + " and text typed after it", id="text typed after it on its line"),
    pytest.param(f"{SHELL}\nA line typed under it.", id="a line typed under it")])
@pytest.mark.parametrize("filters", ["both filters", "Python's alone"])
def test_a_marker_not_alone_on_the_last_line_is_not_found(creating, monkeypatch, description, filters):
    """A person typed next to the marker: before it on its line, after it on its line, or a line under it. The lookup
    keeps only a description whose last line is the marker alone, so the row is not found and the retry makes a new
    dashboard. Each filter holds on its own: the database's passes only the first shape (its description still ends
    with the marker), and Python's turns away all three when the database's passes every row."""
    typed = _row(3, description)
    creating.superset.rows["Dashboard"].append(typed)
    if filters == "Python's alone":
        monkeypatch.setattr(writes, "_like_suffix", lambda marker: "%")
        assert _holds(("like", "description", "%", "\\"), typed)
    else:
        assert _holds(("like", "description", writes._like_suffix(SHELL), "\\"), typed) is description.endswith(SHELL)
    made = make_dashboard()
    assert set(made) == {"dashboard_id"} and made["dashboard_id"] != 3, made
    assert len(commands(creating.superset, "CreateDashboardCommand")) == 1


def test_an_object_in_the_trash_is_not_found(creating):
    creating.superset.rows["Dashboard"].append(_row(3, SHELL, deleted_at=datetime(2026, 11, 15)))
    made = make_dashboard()
    assert set(made) == {"dashboard_id"} and made["dashboard_id"] != 3, made
    assert len(commands(creating.superset, "CreateDashboardCommand")) == 1


def test_two_rows_with_one_marker_answer_candidates(creating):
    """More than one object carries the marker: the build cannot tell which is its own, so both are named, by id,
    and nothing is made."""
    creating.superset.rows["Dashboard"] += [_row(5, SHELL), _row(3, f"Copied by hand.\n{SHELL}")]
    assert make_dashboard() == {"candidates": [3, 5]}
    assert commands(creating.superset) == []


def test_candidates_name_only_objects_the_caller_may_read(creating):
    """An object the caller may not read is answered as one that does not exist, among candidates too: a copy of the
    marker someone else holds is left out of the list, and the create still stops rather than making one more."""
    theirs = _row(5, f"Copied by hand.\n{SHELL}")
    creating.superset.rows["Dashboard"] += [_row(3, SHELL), theirs]
    creating.unreadable.append(theirs)
    assert make_dashboard() == {"candidates": [3]}
    assert commands(creating.superset) == []


def test_candidates_none_of_which_the_caller_may_read_still_stop_the_create(creating):
    """Two marked objects, neither readable: the list is empty, and still nothing is made, as one unreadable marked
    object answers as not found rather than being made again."""
    creating.superset.rows["Dashboard"] += [_row(3, SHELL), _row(5, f"Copied by hand.\n{SHELL}")]
    creating.unreadable.extend(creating.superset.rows["Dashboard"])
    assert make_dashboard() == {"candidates": []}
    assert commands(creating.superset) == []


def test_a_found_object_is_answered_only_to_an_editor_who_may_read_it(creating):
    """The object a retry finds is answered as any other: unreadable, as one that does not exist; not edited by the
    caller, with its editors, whom to ask."""
    made = make_dashboard()
    (row,) = creating.superset.rows["Dashboard"]
    row.editors = [SimpleNamespace(id=102, label="owner Live")]
    creating.not_edited.append(row)
    assert make_dashboard() == {"not_editor": True, "editors": ["owner Live"]}
    creating.unreadable.append(row)
    assert make_dashboard() == {"not_found": "dashboard"}
    creating.unreadable.clear()
    creating.not_edited.clear()
    assert make_dashboard() == made
    assert len(commands(creating.superset)) == 1


def test_a_contended_marker_lock_answers_busy(creating):
    """Another request holding the marker's lock past the lock timeout (55P03 on Postgres): busy, with nothing of the
    create's own work run (no look, none of rc2's checks, no create), since the lock is taken before all of it. The
    lock's transaction is rolled back and its connection invalidated (closed, not put back in the pool), since the lock
    may have been granted as the wait timed out; the request's transaction ends."""
    creating.metadata.contended = "pg_advisory_lock"
    for n, (make, marker) in enumerate(((make_dashboard, SHELL), (make_funnel, STEP), (make_from_config, STEP)), 1):
        creating.superset.ran.clear()
        assert make() == {"busy": True}
        assert creating.superset.ran == [
            ("connect", n), ("on", n, "begin"), ("on", n, LOCK_TIMEOUT, {"timeout": "30000ms"}),
            ("on", n, PG_LOCK, {"marker": marker}), ("on", n, "rollback"), ("invalidate", n), ("close", n),
            ("rollback",)]


@pytest.mark.parametrize("dialect", ["mysql", "mariadb"])
def test_mysql_takes_and_releases_a_named_lock(creating, dialect):
    """A MySQL named lock belongs to a connection, not a transaction, so it is taken and released on a connection of
    its own, never the session's, held from before the look until after the create's commit, and released also when
    the create raises, the create's own exception going on once the settle has found nothing. Its name is built in
    Python from the marker: 63 characters, inside MySQL's 64. Just before its command the create locks the caller's own
    user row and asks its lock's connection whether it still holds the named lock (`guarded`), MariaDB's dialect as
    MySQL's."""
    creating.on(dialect)
    assert len(_lock_name(SHELL)) == 63
    made = make_dashboard()
    assert set(made) == {"dashboard_id"}, made
    assert creating.superset.ran == [
        *taken(dialect, SHELL), ("isolation",), ("lookup", "Dashboard"), *guarded("dashboard", dialect),
        ("CreateDashboardCommand", ANY), ("commit",), ("rollback",), *let_go(dialect, SHELL)]
    creating.superset.ran.clear()
    creating.superset.rows["Dashboard"].clear()
    creating.superset.create = RuntimeError("the metadata database went away")
    with pytest.raises(RuntimeError):
        make_dashboard()
    assert creating.superset.ran == [
        *taken(dialect, SHELL, 2), ("isolation",), ("lookup", "Dashboard"), *guarded("dashboard", dialect, 2),
        ("CreateDashboardCommand", ANY), ("rollback",), ("rollback",), *guarded("dashboard", dialect, 2),
        ("rollback",), *let_go(dialect, SHELL, 2)]


@pytest.mark.parametrize(("configured", "seconds"), [(2.5, 2), (0.5, 1), (44.9, 44)])
def test_mysql_waits_whole_seconds_inside_the_lock_timeout(creating, configured, seconds):
    """`GET_LOCK` waits `MNEMIQ_LOCK_TIMEOUT_S` in whole seconds, rounded down so the wait stays inside the setting's
    bound, and at least one."""
    creating.on("mysql")
    creating.superset.config["MNEMIQ_LOCK_TIMEOUT_S"] = configured
    make_dashboard()
    (asked,) = [entry for entry in creating.superset.ran if entry[:3] == ("on", 1, GET_LOCK)]
    assert asked[3]["seconds"] == seconds


@pytest.mark.parametrize("got", [0, None])
def test_a_mysql_lock_timeout_answers_busy(creating, got):
    """`GET_LOCK` answers 0 when its wait times out and NULL on an error: busy, with no look, none of rc2's checks
    and no create; the lock's transaction is rolled back and its connection invalidated, as on Postgres."""
    creating.on("mysql").got_lock = got
    for n, make in enumerate((make_dashboard, make_funnel), 1):
        creating.superset.ran.clear()
        assert make() == {"busy": True}
        assert creating.superset.ran == [("connect", n), ("on", n, "begin"), ("on", n, GET_LOCK, ANY),
                                         ("on", n, "rollback"), ("invalidate", n), ("close", n), ("rollback",)]


@pytest.mark.parametrize("dialect", ["postgresql", "mysql", "mariadb"])
@pytest.mark.parametrize("level", ["REPEATABLE READ", "SERIALIZABLE", "READ UNCOMMITTED"])
def test_a_create_under_another_isolation_level_is_refused(creating, dialect, level):
    """The look must see an object the moment another create commits it, which READ COMMITTED gives; under a
    snapshot fixed earlier in the transaction it would not, and under READ UNCOMMITTED it would see a row another
    create has written but may still roll back. The level is read under the marker's lock, before the look: another
    level is refused with no look, none of rc2's checks and no create, and the lock is let go. A chart the marker names
    already is refused as every create is, never answered from a row the look could misread."""
    creating.on(dialect).isolation = level
    for n, (make, marker) in enumerate(((make_dashboard, SHELL), (make_funnel, STEP)), 1):
        creating.superset.ran.clear()
        assert make() == {"refused": "isolation"}
        assert creating.superset.ran == [*taken(dialect, marker, n), ("isolation",), ("rollback",),
                                         *let_go(dialect, marker, n)]
    creating.superset.rows["Slice"].append(_row(9, STEP))
    creating.superset.ran.clear()
    assert make_funnel() == {"refused": "isolation"}
    assert "lookup" not in _kinds(creating.superset) and commands(creating.superset) == []
    # The control: READ COMMITTED, however SQLAlchemy spells it, answers the chart and creates.
    creating.metadata.isolation = "read committed"
    assert make_funnel() == {"chart_id": 9}
    assert set(make_dashboard()) == {"dashboard_id"}


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_an_exhausted_pool_answers_busy(creating, dialect):
    """No connection comes free in the pool for the marker's lock within the pool's timeout: busy, nothing looked up,
    checked or made, and the request's transaction ends."""
    creating.on(dialect).pool_exhausted = True
    for make in (make_dashboard, make_funnel, make_from_config):
        creating.superset.ran.clear()
        assert make() == {"busy": True}
        assert creating.superset.ran == [("rollback",)]


def test_a_retry_by_a_caller_no_longer_an_editor_is_refused_first(creating):
    """A caller who stopped editing the dashboard between the lost answer and the retry is told so, naming its
    editors, before anything else: no lock, no lookup, no query check."""
    make_funnel()
    creating.not_edited.append(creating.dashboard)
    creating.superset.ran.clear()
    assert make_funnel() == {"not_editor": True, "editors": ["analyst Live"]}
    assert creating.superset.ran == []


def test_sqlite_takes_no_marker_lock(creating):
    """SQLite, Superset's single-person development default: no lock and no isolation check, and a retry still finds
    the object its marker names, a chart's before its query check too."""
    creating.on("sqlite")
    first = make_dashboard()
    assert make_dashboard() == first
    chart = make_funnel()
    assert {"isolation", "execute", "connect"} & set(_kinds(creating.superset)) == set()
    assert len(commands(creating.superset)) == 2
    creating.superset.ran.clear()
    assert make_funnel() == chart
    assert creating.superset.ran == [("lookup", "Slice"), ("rollback",)]


def test_another_metadata_database_is_refused(creating):
    """A metadata database the marker's lock is not written for is refused, rather than created on without a lock; a
    chart its marker names already too: no connection is taken, nothing looked up, checked or made."""
    creating.on("mssql")
    assert make_dashboard() == {"refused": "metadata_database"}
    creating.superset.rows["Slice"].append(_row(9, STEP))
    assert make_funnel() == {"refused": "metadata_database"}
    assert set(_kinds(creating.superset)) == {"rollback"} and commands(creating.superset) == []


def test_the_lookup_reads_each_row_as_the_database_holds_it(superset, monkeypatch):
    """A row the session already holds (an earlier read in the same session loaded it) is read again as the database
    holds it now, under the lock, not as the session first loaded it: the lookup asks for `populate_existing`, as
    `lock_row` does, before it reads a row."""
    session = _Session("postgresql")

    class Rows(_Query):
        def __iter__(self):
            self.log.append(("read",))
            return iter([])

    session.query = lambda model: (session.log.append(("query", model)), Rows(session.log))[1]
    monkeypatch.setattr(writes, "_session", lambda: session)
    slices = stubs.model("Slice")
    assert writes._marked(slices, STEP) == []
    kinds = [entry[0] for entry in session.log]
    assert kinds[:2] == ["query", "populate_existing"] and kinds[-1] == "read"


# ---------------------------------------------------------------- create_chart from MCP's own chart config

CONFIG = {"chart_type": "big_number", "metric": {"name": "premium", "aggregate": "SUM"}}


def make_from_config(marker: str = STEP, **more) -> dict:
    return writes.create_chart({"dataset_id": 7, "slice_name": "Premium", "dashboard_id": 12, "marker": marker,
                                "config": CONFIG, **more})


def test_create_chart_from_a_config_saves_the_mapped_params(creating):
    """The params saved are what MCP's mapping makes of the config, without its warnings key; the query context is
    built from them; the marker is the description; the editors are a created chart's."""
    made = make_from_config()
    assert set(made) == {"chart_id"}, made
    ((_, properties),) = commands(creating.superset)
    saved = {"viz_type": "big_number_total", "metric": "count"}
    assert json.loads(properties["params"]) == saved
    assert json.loads(properties["query_context"]) == params.query_context("big_number_total", saved, 7)
    assert properties["viz_type"] == "big_number_total" and properties["slice_name"] == "Premium"
    assert (properties["datasource_id"], properties["datasource_type"]) == (7, "table")
    assert properties["description"] == STEP and properties["editors"] == [101]
    assert ("validate", {"dataset_id": 7, "config": CONFIG, "chart_name": "Premium", "save_chart": True}) in (
        creating.superset.ran)
    assert ("map", CONFIG, 7) in creating.superset.ran
    assert make_from_config() == made
    assert len(commands(creating.superset)) == 1


def test_owed_format_params_are_saved_with_the_chart(creating):
    """`set` is merged into the mapped params before the query context is built, so the format keys are saved with
    the chart and are in the query context's source params."""
    made = make_from_config(set={"number_format": ",.2f"})
    assert set(made) == {"chart_id"}, made
    ((_, properties),) = commands(creating.superset)
    saved = {"viz_type": "big_number_total", "metric": "count", "number_format": ",.2f"}
    assert json.loads(properties["params"]) == saved
    assert json.loads(properties["query_context"]) == params.query_context("big_number_total", saved, 7)
    assert json.loads(properties["query_context"])["form_data"]["number_format"] == ",.2f"


@pytest.mark.parametrize("extra", [
    {},  # neither shape
    {"viz_type": "funnel", "params": {"groupby": ["status"]}, "config": CONFIG},  # both
    {"viz_type": "funnel", "params": {}, "config": CONFIG, "set": {}},
    {"viz_type": "funnel", "config": CONFIG},  # half of one and the other
    {"params": {"groupby": ["status"]}, "config": CONFIG},
    {"viz_type": "funnel"},  # half of one
    {"params": {"groupby": ["status"]}},
    {"set": {"number_format": ","}},  # a set without a config
    {"viz_type": "funnel", "params": {"groupby": ["status"]}, "set": {"number_format": ","}},
])
def test_create_chart_takes_one_shape_only(creating, extra):
    request = {"dataset_id": 7, "slice_name": "Premium", "dashboard_id": 12, "marker": STEP, **extra}
    assert writes.create_chart(request) == {"refused": "request"}
    assert commands(creating.superset) == [] and _kinds(creating.superset) == []


def test_each_shape_alone_is_read(creating):
    assert set(make_funnel()) == {"chart_id"}
    assert set(make_from_config(format_marker(BUILD, "chart_2", 12))) == {"chart_id"}
    assert set(make_from_config(format_marker(BUILD, "chart_3", 12), set={})) == {"chart_id"}


def test_a_config_is_checked_for_the_dataset_and_dashboard_before_it_is_validated(creating):
    creating.not_edited.append(creating.dashboard)
    assert make_from_config() == {"not_editor": True, "editors": ["analyst Live"]}
    assert _kinds(creating.superset) == []


@pytest.mark.parametrize("key", ["datasource", "datasource_id", "datasource_type", "viz_type", "time_range",
                                 "extra_form_data", "bins", "adhoc_filters", "color_scheme"])
def test_set_carries_only_the_format_keys(creating, key):
    """`set` carries the format keys `owed_format_params` makes, and nothing that could change which rows a chart reads
    or what it is bound to. Refused before anything is validated, mapped, run, locked or created."""
    assert make_from_config(set={key: "funnel" if key == "viz_type" else 8}) == {"refused": "set"}
    assert _kinds(creating.superset) == [] and commands(creating.superset) == []
    assert creating.superset.ran == []


@pytest.mark.parametrize("key, value", [("column_config", {"sales": {"d3NumberFormat": ",.2f"}}),
                                        ("number_format", ",.2f"), ("currency_format", {"symbol": "USD"})])
def test_each_format_key_is_merged_into_the_saved_params(creating, key, value):
    made = make_from_config(set={key: value})
    assert set(made) == {"chart_id"}, made
    ((_, properties),) = commands(creating.superset)
    assert json.loads(properties["params"])[key] == value


@pytest.mark.parametrize(("raised", "answer"), [
    pytest.param(stubs.SupersetSecurityException("You need access to the following tables: secret"),
                 {"not_found": "dataset"}, id="access refused"),
    pytest.param(stubs.SupersetDisallowedSQLFunctionException("pg_sleep is not allowed"),
                 {"refused": "query_failed"}, id="a function Superset disallows")])
def test_a_compile_check_that_raises_a_superset_error_is_answered_not_raised(creating, caplog, raised, answer):
    """MCP's `_compile_chart` answers a query that fails as a failed compile, but it raises Superset's own errors
    (`SupersetErrorException`), such as the one for a dataset's SQL using a function Superset disallows in a chart's
    query. The tool answers them as `_probe` answers the same errors: a refused access as a dataset that does not
    exist, any other as a query that failed, logged with its error. Never raised out of the tool, which would reach
    mnemiq as the tool failing, with Superset's words; nothing is made."""
    creating.superset.compiled = raised
    with caplog.at_level("WARNING", logger=writes.__name__):
        assert make_from_config() == answer
    assert commands(creating.superset) == [] and creating.superset.rows["Slice"] == []
    logged = [record for record in caplog.records if record.getMessage().startswith("mnemiq extension: create_chart")]
    expected = [] if answer == {"not_found": "dataset"} else [("WARNING", raised)]
    assert [(record.levelname, record.exc_info and record.exc_info[1]) for record in logged] == expected


# ---------------------------------------------------------------- the marker's lock, held through the create

def _failing_query(superset: SimpleNamespace, path: str) -> None:
    """The chart's query check fails from now on, as it does when the source database is slow or gone: rc2's
    `_compile_chart` answers a connection error or a timeout as a failure, and `_check_query` reads the database's
    error as one."""
    if path == "config":
        superset.compiled = SimpleNamespace(success=False)
    else:
        superset.chart_data = stubs.SQLAlchemyError("connection to the source database timed out")


MAKERS = {"config": make_from_config, "funnel": make_funnel}
PATHS = [pytest.param("config", id="config: validation, mapping, _compile_chart"),
         pytest.param("funnel", id="funnel: _check_query")]
# Each create: how it is sent, the model it looks in, its marker, and the key of its answer.
CREATES = {"dashboard": (make_dashboard, "Dashboard", SHELL, "dashboard_id"),
           "funnel chart": (make_funnel, "Slice", STEP, "chart_id"),
           "config chart": (make_from_config, "Slice", STEP, "chart_id"),
           "dataset": (lambda: make_dataset(), "SqlaTable", format_marker(BUILD, "data_1", 3), "dataset_id")}
# The row each create locks as its second lock on MySQL: a chart's or a dataset's dashboard, the caller's own user row
# for a dashboard.
SECOND_ROWS = {"dashboard": ("User", CALLER_ID), "funnel chart": ("Dashboard", 12), "config chart": ("Dashboard", 12),
               "dataset": ("Dashboard", 3)}


def guarded(create: str, dialect: str, n: int = 1) -> list:
    """What `create`'s guarded look runs, just before its command and again in a settle: in the session's transaction,
    its second lock (on Postgres the session's lock waits bounded again, as rc2's checks may have ended the transaction
    the look bounded, then the transaction's lock on the marker's second key; on MySQL the row lock, `SECOND_ROWS`)
    and a look again; then, finding nothing, the hold check on the marker lock's own connection `n`, in a short
    transaction of its own. SQLite runs none of it."""
    if dialect == "sqlite":
        return []
    _, model, marker, _ = CREATES[create]
    if dialect == "postgresql":
        second, holds = [BOUND, ("execute", CREATE_LOCK, {"marker": marker})], (HOLDS, {"marker": marker})
    else:
        second, holds = [("lock_row", *SECOND_ROWS[create])], (MYSQL_HOLDS, {"name": _lock_name(marker)})
    return [*second, ("lookup", model), ("on", n, "begin"), ("on", n, *holds), ("on", n, "commit")]


def work(create: str, dialect: str, n: int = 1) -> list:
    """What `create` runs on a miss, under the marker's lock (held on connection `n`): rc2's checks; the guarded look
    just before the command (`guarded`); the command and its commit; and a dataset's editors step (whose row lock's
    transaction ends with a rollback)."""
    last = guarded(create, dialect, n)
    return {"dashboard": [*last, ("CreateDashboardCommand", ANY), ("commit",)],
            "funnel chart": [("chart_data", ANY), *last, ("CreateChartCommand", ANY), ("commit",)],
            "config chart": [("validate", ANY), ("map", CONFIG, 7), ("compile", ANY, 7), *last,
                             ("CreateChartCommand", ANY), ("commit",)],
            "dataset": [*last, ("CreateDatasetCommand", ANY), ("raise_for_access", "builder", ANY), ("commit",),
                        ("UpdateDatasetCommand", ANY), ("commit",), ("rollback",)]}[create]


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("create", list(CREATES))
def test_the_marker_lock_is_held_from_the_look_through_the_creates_commit(request, dialect, create):
    """Once the caller's own checks pass, a create takes its marker's lock, on a connection of its own, in a short
    transaction committed once the lock is held, and holds it through: the isolation check, on Postgres the bound on
    the session's own lock waits, the look, on a miss rc2's own checks (a chart's validation, mapping and compile
    check, or a funnel's query check), the guarded look in the command's own transaction (`guarded`: its second lock
    is a transaction's lock on Postgres and a row lock on MySQL), the create command and its commit, and a dataset's
    editors step. The request's transaction then ends, and only then is
    the lock released and its connection closed: the hold check's short transaction on that connection did not end
    the lock, whose release still answers that it let it go (no invalidation). Nothing of the work runs before the
    lock. Sent again, the create takes the lock, finds the object and answers it (a dataset after setting its editors
    again, still under the lock), running none of rc2's checks."""
    stage = request.getfixturevalue("lock_rows" if create == "dataset" else "creating")
    superset = stage.superset
    metadata = superset.session = Metadata(superset, dialect)
    make, model, marker, key = CREATES[create]
    made = make()
    assert set(made) == {key}, made
    assert superset.ran == [*taken(dialect, marker), *looked(dialect, model), *work(create, dialect, 1), ("rollback",),
                            *let_go(dialect, marker)]
    superset.ran.clear()
    assert make() == made
    again = [("UpdateDatasetCommand", ANY), ("commit",), ("rollback",)] if create == "dataset" else []
    assert superset.ran == [*taken(dialect, marker, 2), *looked(dialect, model), *again, ("rollback",),
                            *let_go(dialect, marker, 2)]
    assert metadata.leaked == [] and metadata.waiting == []


def _slow_first_check(monkeypatch, superset: SimpleNamespace, path: str, checks: list, in_check: threading.Event,
                      go: threading.Event) -> None:
    """The chart's query check, each run recorded in `checks`: the first holds until `go` is set (a slow source
    database), having set `in_check`; every later one fails at once, as a retry's check against that database may time
    out sooner than the first's."""

    def check(run: Callable[[], Any], failure: Any) -> Any:
        checks.append(threading.current_thread().name)
        if len(checks) > 1:
            return failure
        in_check.set()
        go.wait(10)
        return run()

    if path == "config":
        module = sys.modules["superset.mcp_service.chart.compile"]
        compiling = module._compile_chart
        monkeypatch.setattr(module, "_compile_chart", lambda form_data, dataset_id: check(
            lambda: compiling(form_data, dataset_id), SimpleNamespace(success=False)))
    else:
        command = sys.modules["superset.commands.chart.data.get_data_command"].ChartDataCommand
        running = command.run
        monkeypatch.setattr(command, "run", lambda self: check(
            lambda: running(self), {"queries": [{"error": "the source database timed out", "status": "failed"}]}))


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("first_finishes", [pytest.param(True, id="the first finishes inside the wait"),
                                            pytest.param(False, id="the wait passes the lock timeout")])
def test_a_second_request_waits_for_the_first_and_answers_what_it_made(creating, monkeypatch, dialect, path,
                                                                       first_finishes):
    """A first request holds the marker's lock through its query check, which a slow source database holds up; its
    answer is lost, and the same create is sent again, whose own query check would fail at once. The second request
    waits for the lock. When the first finishes inside the wait, the second finds the chart the first made and answers
    it; when the wait passes the lock timeout, it answers busy, and the same create sent once the first has finished
    answers the first's chart. In neither case does the second run a query check or answer a refusal, which would stop
    the build though the first makes the chart."""
    metadata = creating.on(dialect)
    creating.superset.config["MNEMIQ_LOCK_TIMEOUT_S"] = 10 if first_finishes else 0.2
    checks: list[str] = []
    in_check, go = threading.Event(), threading.Event()
    _slow_first_check(monkeypatch, creating.superset, path, checks, in_check, go)
    make = MAKERS[path]
    answers: dict[str, dict] = {}
    errors: list[BaseException] = []

    def send(name: str) -> None:
        try:
            answers[name] = make()
        except BaseException as exc:  # noqa: BLE001 (reported below)
            errors.append(exc)

    first = threading.Thread(target=send, args=("first",), name="first", daemon=True)
    try:
        first.start()
        assert in_check.wait(5), ("the first request never reached its query check", errors)
        if first_finishes:
            second = threading.Thread(target=send, args=("second",), name="second", daemon=True)
            second.start()
            deadline = time.monotonic() + 5
            while not metadata.waiting:
                assert time.monotonic() < deadline, ("the second request never waited for the lock", answers, errors)
                time.sleep(0.01)
            go.set()
            second.join(10)
        else:
            send("second")
    finally:
        go.set()
        first.join(10)
    assert not errors, errors
    assert set(answers["first"]) == {"chart_id"}, answers
    if first_finishes:
        assert answers["second"] == answers["first"], answers
    else:
        assert answers["second"] == {"busy": True}, answers
        assert make() == answers["first"]
    assert checks == ["first"]
    assert len(commands(creating.superset, "CreateChartCommand")) == 1


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("breaks", ["gone", "aborted"])
@pytest.mark.parametrize("moment", ["isolation", "lookup"])
@pytest.mark.parametrize("create", ["dashboard", "funnel chart", "config chart"])
def test_a_look_that_cannot_run_answers_busy(creating, dialect, breaks, moment, create):
    """Another send of the create has committed its object, and this one's look cannot run: the metadata session's
    connection is gone, or its transaction aborted, when it reads its isolation or looks. Whether the object is there
    is not known, so the create answers busy (the same request may be sent again), never a refusal and never an
    error out of the tool; it runs none of rc2's checks, makes nothing, rolls the session back (which ends an aborted
    transaction) and lets the lock go. Sent again with the database back, it answers the object the other send
    made."""
    metadata = creating.on(dialect)
    make, model, marker, key = CREATES[create]
    creating.superset.rows[model].append(_row(9, marker))
    metadata.breaks[moment] = breaks
    assert make() == {"busy": True}
    assert not {"validate", "map", "compile", "chart_data"} & set(_kinds(creating.superset))
    assert commands(creating.superset) == []
    assert creating.superset.ran[-3:] == [("rollback",), *let_go(dialect, marker)]
    assert metadata.aborted is False
    metadata.gone = False
    assert make() == {key: 9}
    assert commands(creating.superset) == []


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("release", [pytest.param(stubs.database_error("08006"), id="the release raises"),
                                     pytest.param(0, id="the release answers that nothing was released")])
def test_a_failed_unlock_invalidates_the_connection(creating, dialect, release):
    """A session's lock outlives a pooled connection's reset, so a connection whose release failed (it raised, or
    answered that it released nothing) is closed rather than put back in the pool: invalidated, which ends the
    database session and with it the lock. The create's own answer stands, and when the create itself raised, its
    exception is the one that goes on, not the release's. Either way the lock is free afterwards."""
    metadata = creating.on(dialect)
    metadata.released = release
    made = make_dashboard()
    assert set(made) == {"dashboard_id"}, made
    assert creating.superset.ran[-4:] == [("rollback",), let_go(dialect, SHELL)[0], ("invalidate", 1), ("close", 1)]
    creating.superset.ran.clear()
    creating.superset.rows["Dashboard"].clear()
    creating.superset.create = RuntimeError("the metadata database went away")
    with pytest.raises(RuntimeError):
        make_dashboard()
    assert creating.superset.ran[-3:] == [let_go(dialect, SHELL, 2)[0], ("invalidate", 2), ("close", 2)]
    assert metadata.leaked == []
    assert all(lock.acquire(blocking=False) for lock in metadata.locks.values())


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize(("error", "answer"), [
    pytest.param(stubs.database_error("08006", invalidated=True), {"busy": True}, id="the connection lost"),
    pytest.param(stubs.database_error("42883"), {"failed": True}, id="another database error")])
def test_a_failure_taking_the_lock_closes_its_connection(creating, dialect, error, answer):
    """The lock statement fails other than by waiting past the lock timeout: whether the lock was taken on the
    connection is not known, so it is closed rather than put back in the pool (invalidated). The error is answered,
    never raised out of the tool: busy for a lost connection, failed for another error. Nothing is looked up, checked
    or made."""
    metadata = creating.on(dialect)
    metadata.lock_error = error
    for make in (make_dashboard, make_funnel):
        creating.superset.ran.clear()
        assert make() == answer
        assert creating.superset.ran[-4:] == [("on", ANY, "rollback"), ("invalidate", ANY), ("close", ANY),
                                              ("rollback",)]
        assert not {"isolation", "lookup", "validate", "chart_data"} & set(_kinds(creating.superset))
        assert commands(creating.superset) == []


# The command each create runs.
COMMANDS = {"dashboard": "CreateDashboardCommand", "funnel chart": "CreateChartCommand",
            "config chart": "CreateChartCommand", "dataset": "CreateDatasetCommand"}


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("create", list(CREATES))
def test_a_create_whose_marker_lock_was_lost_answers_what_a_second_send_made(request, create, dialect):
    """The marker's lock is held by a connection of its own, which sits idle while the create works (a chart's query
    check). If that connection's database session ends, its lock goes with it while the create's own session works
    on, and a second send of the create takes the lock, finds nothing and makes the object. So just before its command
    each create takes a second lock in the command's own transaction (on Postgres a transaction's lock on another key
    of the marker, on MySQL the row it is made under) and looks again under it: the first send, reaching its command,
    finds the object the second made and answers it. Superset holds one object. The first learns its lock's
    connection was lost only when it lets the lock go, and closes that connection."""
    stage = request.getfixturevalue("lock_rows" if create == "dataset" else "creating")
    superset = stage.superset
    metadata = superset.session = Metadata(superset, dialect)
    make, model, _, key = CREATES[create]
    second: dict = {}

    def meanwhile() -> None:
        metadata.drop_lock_connections()
        second.update(make())

    metadata.before_create_lock = meanwhile
    first = make()
    assert set(second) == {key}, second
    assert first == second
    assert len(superset.rows[model]) == 1
    assert len(commands(superset, COMMANDS[create])) == 1
    assert ("invalidate", 1) in superset.ran and metadata.leaked == []


def _lose_the_marker_lock_before_the_command(request, create: str,
                                             dialect: str = "postgresql") -> tuple[SimpleNamespace, Metadata]:
    """`create`'s stage on `dialect`, its marker lock's connection to end just before its command's own transaction
    takes its second lock, with no other send of the create meanwhile."""
    stage = request.getfixturevalue("lock_rows" if create == "dataset" else "creating")
    superset = stage.superset
    metadata = superset.session = Metadata(superset, dialect)
    metadata.before_create_lock = metadata.drop_lock_connections
    return superset, metadata


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("create", list(CREATES))
def test_a_create_whose_lock_was_lost_before_its_command_makes_nothing_and_answers_busy(request, create, dialect):
    """The marker lock's connection ends while the create works (an idle timeout during a chart's query check), and
    no other send of the create comes: the guarded look finds nothing, and the hold check on the lock's own connection
    fails. The create makes nothing and answers busy: it no longer holds its marker's lock without a break since its
    look, so another send may have taken the lock and refused, and a create made now could stand behind that refusal.
    Nothing takes the lost lock again. The session is rolled back, which releases the second lock, and the lost
    connection is closed when the lock is let go."""
    superset, metadata = _lose_the_marker_lock_before_the_command(request, create, dialect)
    make, model, _, _ = CREATES[create]
    assert make() == {"busy": True}
    assert superset.rows[model] == []
    assert commands(superset, COMMANDS[create]) == []
    assert ("invalidate", 1) in superset.ran and metadata.held_xact == [] and metadata.leaked == []
    assert _all_let_go(metadata)
    lock = PG_LOCK if dialect == "postgresql" else GET_LOCK
    assert [entry for entry in superset.ran if entry[0] == "connect" or entry[2:3] == (lock,)] == [
        ("connect", 1), ("on", 1, lock, ANY)]


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("create", list(CREATES))
def test_a_lock_lost_with_its_connection_is_not_released_on_a_fresh_one(request, caplog, create, dialect):
    """The hold check's statement fails on the lock's lost connection, and SQLAlchemy invalidates that connection;
    its next statement would reconnect it quietly, to a fresh pooled connection whose database session holds nothing,
    where an unlock answers that it released nothing. So the release runs nothing on a connection already
    invalidated: the lock went with its database session, and the loss is logged once, as the lost-lock warning. No
    unlock statement, no reconnection (no second pooled connection taken), no "not released" warning; the connection is
    closed, and every lock is free."""
    superset, metadata = _lose_the_marker_lock_before_the_command(request, create, dialect)
    make, _, marker, _ = CREATES[create]
    with caplog.at_level("INFO", logger=writes.__name__):
        assert make() == {"busy": True}
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert warnings == [LOST], warnings
    unlock = PG_UNLOCK if dialect == "postgresql" else RELEASE_LOCK
    assert not [entry for entry in superset.ran if entry[2:3] == (unlock,) or entry[0] == "reconnect"], superset.ran
    holds = (HOLDS, {"marker": marker}) if dialect == "postgresql" else (MYSQL_HOLDS, {"name": _lock_name(marker)})
    assert ("on", 1, *holds) in superset.ran, superset.ran
    at = superset.ran.index(("on", 1, *holds))
    assert superset.ran[at + 1:] == [("invalidate", 1), ("on", 1, "rollback"), ("rollback",), ("rollback",),
                                     ("close", 1)]
    assert metadata.leaked == [] and _all_let_go(metadata)


@pytest.mark.parametrize(("dialect", "answer"), [("postgresql", False), ("mysql", 0)])
def test_a_hold_check_on_another_session_answers_busy(creating, dialect, answer):
    """The hold check answers that the lock is not held while its connection is still there, as it does when it
    reaches another database session than the one that took the lock (a pooler in transaction mode; on MySQL, 0: a
    session holds the named lock, but not this one): busy, and nothing is made."""
    creating.on(dialect).holds_answer = answer
    assert make_from_config() == {"busy": True}
    assert creating.superset.rows["Slice"] == [] and commands(creating.superset) == []
    assert creating.metadata.held_xact == [] and creating.metadata.leaked == []


def test_mysql_a_free_named_lock_is_not_held(creating):
    """MySQL's hold check answers NULL when nobody holds the named lock: the create's lock was let go or lost. Not
    held: busy, nothing made, and the second lock let go."""
    creating.on("mysql").holds_answer = None
    assert make_from_config() == {"busy": True}
    assert creating.superset.rows["Slice"] == [] and commands(creating.superset) == []
    assert ("on", 1, MYSQL_HOLDS, {"name": _lock_name(STEP)}) in creating.superset.ran
    assert creating.metadata.held_xact == [] and _all_let_go(creating.metadata)


ROW_GONE = "mnemiq extension: a create's row to lock is gone; answered busy"


@pytest.mark.parametrize("create", ["funnel chart", "config chart", "dataset"])
def test_mysql_a_create_whose_dashboard_row_is_gone_answers_busy(request, caplog, create):
    """On MySQL a chart's or a dataset's second lock is its dashboard's row. A dashboard deleted after the create's own
    checks read it leaves no row to lock: the create answers busy, says why, makes nothing and runs no look or hold
    check without the lock; the session is rolled back. The retry's own checks then find no dashboard."""
    stage = request.getfixturevalue("lock_rows" if create == "dataset" else "creating")
    superset = stage.superset
    metadata = superset.session = Metadata(superset, "mysql")
    make, model, marker, _ = CREATES[create]
    dashboard_id = SECOND_ROWS[create][1]
    metadata.before_create_lock = lambda: superset.dashboards.pop(dashboard_id)
    with caplog.at_level("INFO", logger=writes.__name__):
        assert make() == {"busy": True}
    assert [record.getMessage() for record in caplog.records if record.getMessage() == ROW_GONE] == [ROW_GONE]
    assert superset.rows[model] == [] and commands(superset, COMMANDS[create]) == []
    ran = superset.ran
    at = ran.index(("lock_row", "Dashboard", dashboard_id))
    assert ran[at + 1:] == [("rollback",), ("rollback",), *let_go("mysql", marker)]
    assert metadata.held_xact == [] and _all_let_go(metadata)
    assert make() == {"not_found": "dashboard"}


@pytest.mark.parametrize("gone", ["the user deleted meanwhile", "no user id"])
def test_mysql_a_dashboard_create_whose_user_row_is_gone_answers_busy(creating, gone):
    """On MySQL a dashboard's second lock is the caller's own user row. No such row to lock (the user deleted after the
    create began, or no user id, which matches no row): busy, and nothing made."""
    metadata, superset = creating.on("mysql"), creating.superset
    if gone == "no user id":
        superset.g.user = None
    else:
        metadata.before_create_lock = lambda: setattr(superset.g, "user", None)
    assert make_dashboard() == {"busy": True}
    assert superset.rows["Dashboard"] == [] and commands(superset) == []
    assert ("lock_row", "User", None if gone == "no user id" else CALLER_ID) in superset.ran
    assert metadata.held_xact == [] and _all_let_go(metadata)


def test_mysql_a_row_lock_wait_past_the_databases_timeout_answers_busy(creating):
    """MySQL bounds a row lock's wait by its own `innodb_lock_wait_timeout` and then fails the statement (error 1205),
    as it does when an edit of the dashboard, or another create on it, holds the row longer: the create answers busy,
    makes nothing and lets its marker's lock go. Sent again once the row is free, it makes the chart."""
    metadata, superset = creating.on("mysql"), creating.superset
    metadata.fail_at["lock_row"] = (1, stubs.DBAPIError(
        "(MySQLdb.OperationalError) (1205, 'Lock wait timeout exceeded; try restarting transaction')"))
    assert make_funnel() == {"busy": True}
    assert superset.rows["Slice"] == [] and commands(superset) == []
    assert superset.ran[-4:] == [("rollback",), ("rollback",), *let_go("mysql", STEP)]
    assert set(make_funnel()) == {"chart_id"}
    assert len(superset.rows["Slice"]) == 1 and _all_let_go(metadata)


LOST = "mnemiq extension: a marker's lock was lost while its create worked; answered busy, nothing made"


@pytest.mark.parametrize("cause", [pytest.param("lost", id="the lock's connection ended"),
                                   pytest.param(False, id="the hold check answered false"),
                                   pytest.param(0, id="the hold check answered 0"),
                                   pytest.param(None, id="the hold check answered NULL")])
def test_a_lost_lock_is_logged_as_lost(request, caplog, cause):
    """A create that answers busy because its marker's lock was lost says so, once, as a warning: an idle timeout
    shorter than a chart's query check makes every send of that chart lose its lock this way, and the log is how an
    operator learns why the build never finishes it. The database's error rides on the record when the hold check
    raised one on its lost connection (a database error with the connection kept is logged apart:
    `test_a_hold_check_that_cannot_answer_is_logged_apart_from_a_lost_lock`). An answer neither true nor 1 counts as
    not held, NULL among them."""
    if cause == "lost":
        _lose_the_marker_lock_before_the_command(request, "config chart")
    else:
        stage = request.getfixturevalue("creating")
        stage.metadata.holds_answer = cause
    with caplog.at_level("WARNING", logger=writes.__name__):
        assert make_from_config() == {"busy": True}
    lost = [record for record in caplog.records if record.getMessage() == LOST]
    assert [record.levelname for record in lost] == ["WARNING"], caplog.records
    assert (lost[0].exc_info is not None) is (cause == "lost")


UNCONFIRMED = "mnemiq extension: a create could not confirm its marker's lock is held; answered busy, nothing made"


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("error", [pytest.param(stubs.database_error("57014"), id="the statement canceled"),
                                   pytest.param(stubs.database_error("42883"), id="another database error")])
def test_a_hold_check_that_cannot_answer_is_logged_apart_from_a_lost_lock(creating, caplog, dialect, error):
    """The hold check's statement fails while its connection is kept (a statement the database canceled, for one, by
    its `statement_timeout`): whether the lock is held is not known, and it may well be. The create answers busy and
    makes nothing, as for a lost lock, but says it could not confirm the lock, once, as a warning carrying the error,
    never that the lock was lost. The release then runs on the same connection and lets the lock go."""
    metadata = creating.on(dialect)
    metadata.holds_answer = error
    with caplog.at_level("WARNING", logger=writes.__name__):
        assert make_from_config() == {"busy": True}
    assert [(record.getMessage(), record.exc_info and record.exc_info[1]) for record in caplog.records] == [
        (UNCONFIRMED, error)]
    assert creating.superset.rows["Slice"] == [] and commands(creating.superset) == []
    assert creating.superset.ran[-3:] == [("rollback",), *let_go(dialect, STEP)]
    assert ("invalidate", 1) not in creating.superset.ran
    assert metadata.leaked == [] and _all_let_go(metadata)


@pytest.mark.parametrize("answer", [True, 1])
def test_a_hold_check_answering_true_or_1_lets_the_create_make_its_object(creating, answer):
    """The hold check counts the lock as held on Postgres's `true`, or a 1, and on nothing else (`false`, `0`, `NULL`:
    `test_a_lost_lock_is_logged_as_lost`)."""
    creating.metadata.holds_answer = answer
    assert set(make_from_config()) == {"chart_id"}
    assert len(commands(creating.superset, "CreateChartCommand")) == 1


def test_a_second_lock_not_taken_answers_busy(creating, monkeypatch):
    """A guarded look whose second lock is not taken (`_take_create_lock` answers False) answers busy, rolls the
    session back and makes nothing: neither the look nor the hold check runs without the second lock."""
    monkeypatch.setattr(writes, "_take_create_lock", lambda session, held, row: False)
    assert make_dashboard() == {"busy": True}
    ran = creating.superset.ran
    assert ran[-5:] == [BOUND, ("rollback",), ("rollback",), *let_go("postgresql", SHELL)]
    assert ran.count(("lookup", "Dashboard")) == 1 and ("on", 1, HOLDS, {"marker": SHELL}) not in ran
    assert commands(creating.superset) == []


@pytest.mark.parametrize(("held_on", "bind", "second"), [
    pytest.param("mysql", "postgresql", ("lock_row", "Dashboard", 12), id="a MySQL marker lock: the row lock"),
    pytest.param("postgresql", "mysql", ("execute", CREATE_LOCK, {"marker": STEP}),
                 id="a Postgres marker lock: the transaction lock"),
    pytest.param("oracle", "mysql", None, id="any other marker lock: none")])
def test_a_creates_second_lock_follows_its_marker_locks_dialect(creating, caplog, held_on, bind, second):
    """Which second lock a guarded look takes is read from the marker's lock as it was taken (`_HeldLock.dialect`), the
    one source its hold check reads too, never from the session's bind again: a Postgres transaction lock for a
    Postgres marker lock, the row's lock for a MySQL one. For any other dialect it takes none and answers busy, says
    why, and runs no look and no hold check without it (fail closed). The marker lock's connection here holds nothing,
    so every case answers busy; what differs is the lock taken."""
    metadata = creating.on(bind)
    held = writes._HeldLock(metadata.engine.connect(), held_on, STEP)
    models = sys.modules["superset.models.slice"].Slice, sys.modules["superset.models.dashboard"].Dashboard
    with caplog.at_level("WARNING", logger=writes.__name__):
        assert writes._guarded_look(models[0], STEP, kind="chart", held=held, row=(models[1], 12)) == {"busy": True}
    ran = creating.superset.ran
    assert [entry for entry in ran if entry[0] == "lock_row" or entry[1:2] == (CREATE_LOCK,)] == (
        [second] if second else [])
    if second is None:
        assert "lookup" not in _kinds(creating.superset) and ("on", 1, "begin") not in ran
        assert [record.getMessage() for record in caplog.records] == [
            "mnemiq extension: a create takes no second lock on a oracle metadata database; answered busy"]
    assert metadata.held_xact == []


def test_a_contended_create_lock_answers_busy(creating):
    """Another transaction holding a create's own transaction lock past the lock timeout (55P03): busy, nothing made,
    and the transaction the error aborted ends before the marker's lock is let go."""
    creating.metadata.contended = "pg_advisory_xact_lock"
    assert make_dashboard() == {"busy": True}
    assert creating.superset.ran[-5:] == [("execute", CREATE_LOCK, {"marker": SHELL}), ("rollback",), ("rollback",),
                                          *let_go("postgresql", SHELL)]
    assert commands(creating.superset) == []


class _Unreachable(dict):
    """Objects by id, every read of which fails with `error`, as a DAO's read does when the metadata database fails."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def get(self, key, default=None):
        raise self.error


@pytest.mark.parametrize("lost", [pytest.param(True, id="the connection lost"),
                                  pytest.param(False, id="another database error")])
@pytest.mark.parametrize("where", ["taking the lock", "the checks before the lock", "the chart's last bound"])
def test_a_database_error_outside_a_command_is_answered_not_raised(creating, monkeypatch, where, lost):
    """A database error outside any command (taking the marker's lock, or the caller's checks before it) reaches the
    write tool's one boundary for database errors: the session is rolled back and the error answered, busy for a lost
    connection (the same request may be sent again), failed for another; never raised out of the tool, which mnemiq
    would count as the tool failing. At the chart's last bound on its lock waits, inside the guarded look, any
    database error answers busy, as a look's does (`test_a_guarded_look_that_cannot_run_answers_busy`). A retry once
    the database answers again, with the chart an earlier send committed, answers that chart."""
    metadata, superset = creating.metadata, creating.superset
    error = stubs.database_error("08006", invalidated=True) if lost else stubs.database_error("42883")
    if where == "taking the lock":
        metadata.lock_error = error
    elif where == "the checks before the lock":
        superset.dashboards = _Unreachable(error)
    else:
        module = sys.modules["superset.mcp_service.chart.compile"]
        compiling = module._compile_chart
        monkeypatch.setattr(module, "_compile_chart", lambda form_data, dataset_id: (
            setattr(metadata, "gone" if lost else "aborted", True), compiling(form_data, dataset_id))[1])
    assert make_from_config() == ({"busy": True} if lost or where == "the chart's last bound" else {"failed": True})
    assert commands(superset) == []
    if where == "the chart's last bound":
        assert superset.ran[-5:] == [BOUND, ("rollback",), ("rollback",), *let_go("postgresql", STEP)]
    else:
        assert superset.ran[-1] == ("rollback",)
    metadata.lock_error, metadata.gone = None, False
    superset.dashboards = {12: creating.dashboard}
    superset.rows["Slice"].append(_row(9, STEP))
    assert make_from_config() == {"chart_id": 9}
    assert commands(superset) == []


# Where a guarded look may fail: the n-th occurrence on the session of its bound, its create lock, or its lookup (the
# first look's bound and lookup come first).
GUARDED_LOOK = {"the bound": (LOCK_TIMEOUT, 2), "the create lock": (CREATE_LOCK, 1), "the lookup": ("lookup", 2)}


@pytest.mark.parametrize("moment", list(GUARDED_LOOK))
@pytest.mark.parametrize("create", list(CREATES))
def test_a_guarded_look_that_cannot_run_answers_busy(request, caplog, create, moment):
    """The guarded look, like the first look, answers busy for any database error, a failure neither contention nor a
    lost connection among them (SQLSTATE 42883 here): whether another send made the object is not known, and a
    refusal would stop the build though it may have. It says so in a warning carrying the error, in words true of
    both its runs, before the command and in a settle. The rollback ends the transaction, releasing the create lock
    with nothing made, and the marker's lock is let go. Sent again with the database answering, the create makes the
    object."""
    stage = request.getfixturevalue("lock_rows" if create == "dataset" else "creating")
    superset = stage.superset
    metadata = superset.session = Metadata(superset, "postgresql")
    make, model, marker, key = CREATES[create]
    statement, n = GUARDED_LOOK[moment]
    error = stubs.database_error("42883")
    metadata.fail_at[statement] = (n, error)
    with caplog.at_level("WARNING", logger=writes.__name__):
        assert make() == {"busy": True}
    assert [(record.levelname, record.getMessage()) for record in caplog.records
            if record.exc_info and record.exc_info[1] is error] == [
        ("WARNING", "mnemiq extension: a create could not look for its object under its second lock; answered busy")]
    assert commands(superset, COMMANDS[create]) == [] and metadata.held_xact == []
    assert superset.ran[-4:] == [("rollback",), ("rollback",), *let_go("postgresql", marker)]
    made = make()
    assert set(made) == {key}, made
    assert len(superset.rows[model]) == 1 and metadata.leaked == []


@pytest.mark.parametrize("where", ["the marker lock's take", "the create lock", "a command"])
def test_a_statement_the_database_canceled_answers_busy(creating, where):
    """A metadata database whose `statement_timeout` is shorter than the lock timeout cancels a long wait with SQLSTATE
    57014 rather than 55P03: at the marker lock's take, at the create's own transaction lock, or in a command's own
    statements. Busy, as for a wait past the lock timeout, since the same create sent again is safe by its marker;
    nothing is made. The control: another database error from a command answers failed."""
    canceled = stubs.database_error("57014")
    metadata, superset = creating.metadata, creating.superset
    if where == "the marker lock's take":
        metadata.lock_error = canceled
    elif where == "the create lock":
        metadata.fail_at[CREATE_LOCK] = (1, canceled)
    else:
        superset.create = stubs.CommandException("the statement was canceled")
        superset.create.__cause__ = canceled
    assert make_dashboard() == {"busy": True}
    assert superset.rows["Dashboard"] == [] and metadata.leaked == []
    if where == "a command":
        superset.create.__cause__ = stubs.database_error("42883")
        assert make_dashboard() == {"failed": True}


def test_only_a_write_tool_answers_a_database_error(superset):
    """The boundary for database errors is the write tools' (`_answering`): the same failure in a read tool's body is
    raised as it was, and a non-database error is raised by both."""
    from agenticfabriq.mnemiq import reads

    superset.session = Metadata(superset, "postgresql")

    def failing(found: dict) -> dict:
        raise stubs.database_error("08006", invalidated=True)

    def broken(found: dict) -> dict:
        raise RuntimeError("not a database error")

    assert writes._answering("delete_dataset")(failing)({"dataset_id": 4, "dashboard_id": 12}) == {"busy": True}
    assert superset.ran == [("rollback",)]
    with pytest.raises(stubs.DBAPIError):
        writes._answering("get_layout", reads.REQUESTS)(failing)({"dashboard_id": 12})
    with pytest.raises(RuntimeError):
        writes._answering("delete_dataset")(broken)({"dataset_id": 4, "dashboard_id": 12})


def test_a_lock_granted_as_its_wait_times_out_is_not_put_back_in_the_pool(creating):
    """Postgres can grant a waited-for lock just as the wait's lock timeout fires, and the statement still fails with
    SQLSTATE 55P03: the session's lock is then held, and the failed transaction's rollback does not end a session's
    lock. The create answers busy as for any wait past the timeout, and its connection is invalidated (closed, not put
    back in the pool), which ends the database session and so the lock: no pooled connection holds it afterwards."""
    metadata = creating.metadata
    metadata.late_grant = True
    assert make_dashboard() == {"busy": True}
    assert creating.superset.ran == [
        ("connect", 1), ("on", 1, "begin"), ("on", 1, LOCK_TIMEOUT, {"timeout": "30000ms"}),
        ("on", 1, PG_LOCK, {"marker": SHELL}), ("on", 1, "rollback"), ("invalidate", 1), ("close", 1), ("rollback",)]
    assert metadata.leaked == []
    assert metadata.locks[SHELL].acquire(blocking=False)


@pytest.mark.parametrize("create", ["dashboard", "funnel chart", "config chart"])
def test_a_creates_own_lock_waits_are_bounded_on_postgres(creating, monkeypatch, create):
    """Under the marker's lock, a create's own statements may wait for a row another transaction holds (its command's
    writes), and a wait past the lock timeout fails with 55P03, which answers busy, rather than waiting unbounded
    while the marker's lock is held. The bound is set on the session as `lock_row` sets it, before the look, and, for a
    chart, again just before its command: rc2's checks may end the transaction the first bound was set in (Superset's
    metastore cache commits the session when it stores a chart's data), and a bound holds only to the end of its
    transaction. So no commit or rollback comes between the last bound and the command."""
    make, model, marker, key = CREATES[create]
    superset = creating.superset
    if create == "config chart":
        module = sys.modules["superset.mcp_service.chart.compile"]
        compiling = module._compile_chart
        monkeypatch.setattr(module, "_compile_chart", lambda form_data, dataset_id: (
            superset.session.commit(), compiling(form_data, dataset_id))[1])
    elif create == "funnel chart":
        command = sys.modules["superset.commands.chart.data.get_data_command"].ChartDataCommand
        running = command.run
        monkeypatch.setattr(command, "run", lambda self: (superset.session.commit(), running(self))[1])
    failure = stubs.CommandException("the create waited past the lock timeout")
    failure.__cause__ = stubs.database_error("55P03")
    superset.create = failure
    assert make() == {"busy": True}
    ran = superset.ran
    isolation = ran.index(("isolation",))
    assert ran[isolation + 1:isolation + 3] == [BOUND, ("lookup", model)]
    (command_at,) = [at for at, entry in enumerate(ran) if entry[0].endswith("Command")]
    last_bound = max(at for at, entry in enumerate(ran[:command_at]) if entry == BOUND)
    assert not {"commit", "rollback"} & {entry[0] for entry in ran[last_bound + 1:command_at]}, ran


class _Clock:
    """A monotonic clock stood in: the tool's start reads 1000 seconds, every later reading `after` seconds more."""

    def __init__(self, after: float) -> None:
        self.after, self.readings = after, 0

    def __call__(self) -> float:
        self.readings += 1
        return 1000.0 if self.readings == 1 else 1000.0 + self.after


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
@pytest.mark.parametrize("create", ["dashboard", "config chart", "dataset"])
def test_a_lock_granted_after_the_call_deadline_answers_busy(request, monkeypatch, dialect, create):
    """The marker's lock came free only after mnemiq's deadline for the call (`CALL_DEADLINE_S` from the tool's start,
    on a monotonic clock): mnemiq has stopped waiting for this answer, and its own retry decides. The create lets the
    lock go and answers busy, without a look, a check or a command. The control: granted just inside the deadline,
    the create looks and makes its object."""
    stage = request.getfixturevalue("lock_rows" if create == "dataset" else "creating")
    superset = stage.superset
    metadata = superset.session = Metadata(superset, dialect)
    make, model, marker, key = CREATES[create]
    monkeypatch.setattr(writes, "_clock", _Clock(writes.CALL_DEADLINE_S + 0.001))
    assert make() == {"busy": True}
    assert superset.ran == [*taken(dialect, marker), *let_go(dialect, marker), ("rollback",)]
    assert commands(superset) == []
    superset.ran.clear()
    monkeypatch.setattr(writes, "_clock", _Clock(writes.CALL_DEADLINE_S - 0.001))
    made = make()
    assert set(made) == {key}, made
    expected = [*taken(dialect, marker, 2), *looked(dialect, model)]
    assert superset.ran[:len(expected)] == expected
    assert metadata.leaked == []


@pytest.mark.parametrize("tool", ["delete_dataset", "create_dashboard"])
@pytest.mark.parametrize(("cause", "answer"), [
    pytest.param(stubs.PoolTimeoutError("QueuePool limit of size 5 overflow 10 reached"), {"busy": True},
                 id="no pooled connection came free"),
    pytest.param(stubs.database_error("08006", invalidated=True), {"busy": True},
                 id="the connection lost mid-statement"),
    pytest.param(stubs.database_error("08006"), {"failed": True}, id="a database error, the connection kept"),
    pytest.param(stubs.database_error("23505"), {"failed": True}, id="a unique violation")])
def test_a_command_whose_metadata_connection_was_lost_answers_busy(request, tool, cause, answer):
    """A command that failed because the metadata database's connection could not be had (no pooled connection came
    free in time) or was lost mid-statement (SQLAlchemy invalidated it) may or may not have written: busy, and the
    same request may be sent again (a create finds what an earlier send made by its marker). Any other database
    error, the connection kept, is answered as a failure."""
    failure = stubs.CommandException("Superset's words")
    failure.__cause__ = cause
    if tool == "delete_dataset":
        stage = request.getfixturevalue("deleting")
        stage.charts, stage.superset.delete = [[]], failure
        assert delete(_dataset(MARKED)) == answer
    else:
        stage = request.getfixturevalue("creating")
        stage.superset.create = failure
        assert make_dashboard() == answer


@pytest.mark.parametrize("path", PATHS)
def test_a_retry_of_a_chart_already_made_runs_no_query_check_again(creating, path):
    """The first create committed and its answer was lost, most likely because Superset or the source database was
    slow; now the chart's query check would fail. The retry looks for the step's chart under the marker's lock before
    any of rc2's work and answers it: no validation, mapping or query check runs again. The control: a step whose chart
    was never made, sent now, is refused by the failing check."""
    make = MAKERS[path]
    made = make()
    assert set(made) == {"chart_id"}, made
    _failing_query(creating.superset, path)
    creating.superset.ran.clear()
    assert make() == made
    assert creating.superset.ran == [*taken("postgresql", STEP, 2), *looked("postgresql", "Slice"), ("rollback",),
                                     *let_go("postgresql", STEP, 2)]
    assert make(format_marker(BUILD, "chart_2", 12)) == {"refused": "query_failed"}
    assert commands(creating.superset) == []


@pytest.mark.parametrize("found", ["not edited", "unreadable", "two carrying the marker"])
def test_a_found_chart_is_answered_by_the_found_rules_before_any_query_check(creating, found):
    """The look answers what it finds by the rules every object found by its marker is answered by: a chart the caller
    does not edit, with its editors; one it may not read, as one that does not exist; more than one, as the
    candidates the caller may read. None of it runs rc2's validation, mapping or query check, which would now fail."""
    rows = creating.superset.rows["Slice"]
    rows.append(_row(9, STEP))
    if found == "not edited":
        rows[0].editors = [SimpleNamespace(id=102, label="owner Live")]
        creating.not_edited.append(rows[0])
        expected = {"not_editor": True, "editors": ["owner Live"]}
    elif found == "unreadable":
        creating.unreadable.append(rows[0])
        expected = {"not_found": "chart"}
    else:
        rows.append(_row(11, f"Copied by hand.\n{STEP}"))
        creating.unreadable.append(rows[1])
        expected = {"candidates": [9]}
    _failing_query(creating.superset, "config")
    assert make_from_config() == expected
    assert not {"validate", "map", "compile"} & set(_kinds(creating.superset))
    assert commands(creating.superset) == []


# Each of a chart's own checks that may refuse it, under the marker's lock: the create it belongs to, what the
# check's run is recorded as, and the refusal it answers.
CHECKS = {
    "validation": (make_from_config, "validate", {"refused": "config"}),
    "mapping": (make_from_config, "map", {"refused": "config"}),
    "compile": (make_from_config, "compile", {"refused": "query_failed"}),
    "query check": (make_funnel, "chart_data", {"refused": "query_failed"}),
}


def _refusing(superset: SimpleNamespace, check: str) -> None:
    """`check` refuses the chart: rc2's validation answers the config invalid; its mapping leaves no chart type;
    `_compile_chart` answers a failure, as it does for a source database that timed out; a funnel's query answers a
    failed query."""
    if check == "validation":
        superset.config_valid = False
    elif check == "mapping":
        superset.mapped = {"metric": "count"}
    elif check == "compile":
        superset.compiled = SimpleNamespace(success=False)
    else:
        superset.chart_data = {"queries": [{"error": "the source database timed out", "status": "failed"}]}


def _after(superset: SimpleNamespace, ran: str) -> list:
    """What ran after the check `ran` names (its first run)."""
    return superset.ran[_kinds(superset).index(ran) + 1:]


def settled(create: str, n: int = 1, dialect: str = "postgresql") -> list:
    """What `create` runs on `dialect` once a check of its own refused it, the marker's lock held on connection `n`:
    the settle (the session rolled back, then the guarded look, `guarded`, which finds nothing and the lock held), the
    request's transaction ended, and the lock let go."""
    marker = CREATES[create][2]
    return [("rollback",), *guarded(create, dialect, n), ("rollback",), *let_go(dialect, marker, n)]


@pytest.mark.parametrize("check", CHECKS)
def test_a_chart_its_own_check_refuses_is_answered_the_refusal_under_the_lock(creating, check):
    """rc2's validation, mapping or query check refuses the chart under the marker's lock. The refusal is answered
    once the settle finds no chart with the lock still held: after nothing more of the chart's path (a config the
    validation refuses is not mapped, nor its query run), the session is rolled back and the guarded look runs (the
    second lock, the look, the hold check); then the request's transaction ends and the lock is let go. Nothing is
    made."""
    make, ran, refusal = CHECKS[check]
    _refusing(creating.superset, check)
    assert make() == refusal
    assert creating.superset.ran[:8] == [*taken("postgresql", STEP), *looked("postgresql", "Slice")]
    assert _after(creating.superset, ran) == settled("funnel chart" if make is make_funnel else "config chart")
    assert commands(creating.superset) == []


@pytest.mark.parametrize("check", ["compile", "query check"])
def test_sqlite_runs_no_settle(creating, check):
    """A SQLite metadata database takes no marker lock, so a create there has no guarded look and nothing to settle a
    refusal under: a chart its own check refuses is answered the refusal with nothing run after the check but the
    request's transaction ending, once."""
    creating.on("sqlite")
    make, ran, refusal = CHECKS[check]
    _refusing(creating.superset, check)
    assert make() == refusal
    assert creating.superset.ran[0] == ("lookup", "Slice")
    assert _after(creating.superset, ran) == [("rollback",)]


def _when_checked(monkeypatch, check: str, meanwhile: Callable[[], None]) -> None:
    """`meanwhile()` runs each time `check` (one of `CHECKS`) runs, just before the check itself: rc2's validation, its
    mapping, `_compile_chart`, or a funnel's query."""
    if check == "validation":
        pipeline = sys.modules["superset.mcp_service.chart.validation"].ValidationPipeline
        validating = pipeline.validate_request_with_warnings
        monkeypatch.setattr(pipeline, "validate_request_with_warnings",
                            staticmethod(lambda request: (meanwhile(), validating(request))[1]))
    elif check == "query check":
        command = sys.modules["superset.commands.chart.data.get_data_command"].ChartDataCommand
        running = command.run
        monkeypatch.setattr(command, "run", lambda self: (meanwhile(), running(self))[1])
    else:
        module, name = {"mapping": ("superset.mcp_service.chart.chart_utils", "map_config_to_form_data"),
                        "compile": ("superset.mcp_service.chart.compile", "_compile_chart")}[check]
        checking = getattr(sys.modules[module], name)
        monkeypatch.setattr(sys.modules[module], name,
                            lambda *args, **kwargs: (meanwhile(), checking(*args, **kwargs))[1])


def _sender(make: Callable[[], dict], answers: dict, errors: list) -> Callable[[str], None]:
    """A send of `make`, by name: its answer goes to `answers[name]`, anything it raises to `errors`."""

    def send(name: str) -> None:
        try:
            answers[name] = make()
        except BaseException as exc:  # noqa: BLE001 (reported by the test)
            errors.append(exc)

    return send


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_a_retrys_refusal_answers_the_chart_a_first_send_made_meanwhile(creating, dialect):
    """A first send of a chart passes its guarded look: it holds the second lock (on MySQL the dashboard's row), has
    found nothing and has found its marker's lock held. Then that lock's connection ends, and a retry of the same
    create, sent meanwhile, takes the marker's lock and finds nothing yet; the retry's own query check refuses the
    chart, as a check against a slow source may. Before it answers that refusal the retry settles it: it waits for the
    second lock, which the first holds until its command commits, then looks again and finds the first's chart. Both
    sends answer the one chart, and the build never records a refusal for a chart Superset holds. Afterwards every
    lock is free, whichever send took it."""
    metadata, superset = creating.on(dialect), creating.superset
    answers: dict[str, dict] = {}
    errors: list[BaseException] = []
    send = _sender(make_from_config, answers, errors)
    retry = threading.Thread(target=send, args=("retry",), name="retry", daemon=True)
    waited: list[list[str]] = []  # who waited for the second lock while the first held it

    def meanwhile() -> None:
        metadata.drop_lock_connections()
        superset.compiled = SimpleNamespace(success=False)
        retry.start()
        deadline = time.monotonic() + 5
        while not metadata.waiting_xact and retry.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        waited.append(list(metadata.waiting_xact))

    metadata.after_hold_check = meanwhile
    try:
        send("first")
    finally:
        if retry.ident is not None:
            retry.join(10)
    assert not errors, errors
    assert waited == [["retry"]], ("the retry never waited for the second lock", waited, answers)
    assert set(answers.get("first", {})) == {"chart_id"}, answers
    assert answers.get("retry") == answers["first"], answers
    assert _kinds(superset).count("compile") == 2
    assert len(superset.rows["Slice"]) == 1 and len(commands(superset, "CreateChartCommand")) == 1
    assert metadata.waiting_xact == [] and _all_let_go(metadata)


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_a_first_send_whose_lock_was_lost_while_a_retry_refused_makes_nothing(creating, monkeypatch, dialect):
    """The interleaving the settle and the hold check close together. A first send's marker lock's connection ends
    while its query check runs. A retry takes the marker's lock, finds nothing, and its own check refuses the chart;
    it settles the refusal, finding nothing with its lock still held, and answers the refusal, which the build
    records. Then the first send's check passes: at its guarded look it finds nothing and finds its lock lost, so it
    answers busy and makes nothing. No chart stands behind the refusal the build recorded, and afterwards every lock
    is free, whichever send took it."""
    metadata, superset = creating.on(dialect), creating.superset
    answers: dict[str, dict] = {}
    errors: list[BaseException] = []
    send = _sender(make_from_config, answers, errors)
    retry = threading.Thread(target=send, args=("retry",), name="retry", daemon=True)

    def meanwhile() -> None:
        if retry.ident is not None:
            return  # the retry's own check, which refuses
        metadata.drop_lock_connections()
        superset.compiled = SimpleNamespace(success=False)
        retry.start()
        retry.join(10)
        superset.compiled = SimpleNamespace(success=True)

    _when_checked(monkeypatch, "compile", meanwhile)
    send("first")
    assert not errors, errors
    assert answers == {"retry": {"refused": "query_failed"}, "first": {"busy": True}}, answers
    assert superset.rows["Slice"] == [] and commands(superset) == []
    ran = superset.ran
    retry_ran = ran[ran.index(("connect", 2)):ran.index(("close", 2)) + 1]
    assert retry_ran == [*taken(dialect, STEP, 2), *looked(dialect, "Slice"), ("validate", ANY),
                         ("map", CONFIG, 7), ("compile", ANY, 7), *settled("config chart", 2, dialect)]
    assert metadata.held_xact == [] and metadata.waiting_xact == [] and _all_let_go(metadata)


@pytest.mark.parametrize("check", CHECKS)
def test_a_refusal_after_its_own_lock_was_lost_answers_busy(creating, monkeypatch, check):
    """A chart's own check refuses it after the marker lock's connection ended while the check ran, with no other send
    of the create. Another send may have taken the lock meanwhile and be making the chart, so the refusal is not
    answered: the settle finds nothing and finds the lock lost, and the create answers busy, having made nothing. The
    same request may be sent again."""
    make, _, _ = CHECKS[check]
    _refusing(creating.superset, check)
    _when_checked(monkeypatch, check, creating.metadata.drop_lock_connections)
    assert make() == {"busy": True}
    assert creating.superset.rows["Slice"] == [] and commands(creating.superset) == []
    assert creating.metadata.held_xact == [] and creating.metadata.leaked == []


# What the settle logs when what a create raised is replaced by what the settle found, for each kind of object.
SETTLED = "mnemiq extension: a {} create's error was settled: answered what its settle found"


@pytest.mark.parametrize("made_meanwhile", [pytest.param(True, id="another send made the chart meanwhile"),
                                            pytest.param(False, id="nothing made, the lock held")])
def test_a_database_error_in_a_creates_checks_is_settled_before_it_is_answered(creating, monkeypatch, caplog,
                                                                              made_meanwhile):
    """A database error raised out of a chart's own checks (here the config's mapping, SQLSTATE XX000, which aborts the
    session's transaction) is settled as a refusal is: the session is rolled back, which ends the aborted transaction,
    and the guarded look runs. When another send committed the step's chart meanwhile (this send's lock lost), that
    chart is answered, and the error it replaces is logged as a warning, with the error on the record. Else, nothing
    found and the lock held, the error goes on to the write tool's boundary for database errors and is answered
    `{"failed": true}`, as before, and logged there."""
    metadata, superset = creating.metadata, creating.superset
    error = superset.mapped = stubs.database_error("XX000")

    def aborting() -> None:
        metadata.aborted = True
        if made_meanwhile:
            metadata.drop_lock_connections()
            superset.rows["Slice"].append(_row(9, STEP))

    _when_checked(monkeypatch, "mapping", aborting)
    with caplog.at_level("WARNING", logger=writes.__name__):
        if made_meanwhile:
            assert make_from_config() == {"chart_id": 9}
        else:
            assert make_from_config() == {"failed": True}
            assert _after(superset, "map") == [*settled("config chart"), ("rollback",)]
    carrying = [(record.levelname, record.getMessage()) for record in caplog.records
                if record.exc_info and record.exc_info[1] is error]
    if made_meanwhile:
        assert carrying == [("WARNING", SETTLED.format("chart"))], caplog.records
    else:
        assert carrying == [("WARNING", "mnemiq extension: create_chart failed")], caplog.records
    assert commands(superset) == [] and metadata.leaked == []


def test_a_commands_own_refusal_is_answered_after_the_settle(lock_rows):
    """A dataset whose command refuses it (a dataset not made for this marker already holds the table), its marker's
    lock held throughout, answers `{"exists": true}`, but only once the settle has run after the command's rollback:
    the second lock, the look and the hold check, as for any answer but the object or busy."""
    superset = lock_rows.superset
    superset.session = Metadata(superset, "postgresql")
    superset.create = stubs.DatasetInvalidError({"DatasetExistsValidationError"})
    assert make_dataset() == {"exists": True}
    assert _after(superset, "CreateDatasetCommand") == [("raise_for_access", "builder", ANY), ("rollback",),
                                                        *settled("dataset")]
    assert commands(superset, "UpdateDatasetCommand") == []


def test_a_dataset_the_settle_finds_gets_its_editors_set(lock_rows):
    """The settle runs before a dataset's editors step. A dataset create passes its guarded look, and then its marker
    lock's connection ends; its command refuses the dataset, and once that command has rolled back (letting the second
    lock go), another send of the create, holding the lost lock, commits the dataset over the same table. The first
    send's settle finds that dataset and answers it, with its editors set as for any dataset found by its marker."""
    superset, dataset, ids = lock_rows.superset, lock_rows.dataset, lock_rows.ids
    metadata = superset.session = Metadata(superset, "postgresql")
    superset.create = stubs.DatasetInvalidError({"DatasetExistsValidationError"})
    dataset.deleted_at, dataset.description = None, DATA

    def lost_after_the_hold_check() -> None:
        metadata.drop_lock_connections()
        metadata.before_create_lock = lambda: superset.rows["SqlaTable"].append(dataset)

    metadata.after_hold_check = lost_after_the_hold_check
    assert make_dataset() == {"dataset_id": 4}
    assert _sent_editors(superset, "UpdateDatasetCommand") == [
        ids["creator"], ids["builder"], 302, ids["reader"], ids["database_reader"]]
    assert _kinds(superset).index("CreateDatasetCommand") < _kinds(superset).index("UpdateDatasetCommand")


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_a_raised_twin_refusal_answers_the_dataset_a_settle_finds(lock_rows, caplog, dialect):
    """rc2's dataset command refuses a table a dataset in the trash holds by raising
    (`DatasetSoftDeletedTwinExistsError`), which the tool answers by the caller's access once the marker's lock is let
    go (`_twin_answer`). The settle covers whatever a create raises, this refusal among it. A dataset create passes its
    guarded look, and then its marker lock's connection ends; its command raises the refusal, and once that command
    has rolled back (letting the second lock go), another send of the create, holding the lost lock, commits the
    step's dataset. The first send's settle finds that dataset and answers it, with its editors set; the refusal it
    replaces is logged as a warning carrying it, and the caller's access to the table is never asked."""
    superset, dataset, ids = lock_rows.superset, lock_rows.dataset, lock_rows.ids
    metadata = superset.session = Metadata(superset, dialect)
    superset.twin = True
    dataset.deleted_at, dataset.description = None, DATA

    def lost_after_the_hold_check() -> None:
        metadata.drop_lock_connections()
        metadata.before_create_lock = lambda: superset.rows["SqlaTable"].append(dataset)

    metadata.after_hold_check = lost_after_the_hold_check
    with caplog.at_level("WARNING", logger=writes.__name__):
        assert make_dataset() == {"dataset_id": 4}
    assert _asked_for_access(superset) == []
    assert _sent_editors(superset, "UpdateDatasetCommand") == [
        ids["creator"], ids["builder"], 302, ids["reader"], ids["database_reader"]]
    assert [(record.levelname, record.getMessage()) for record in caplog.records if record.exc_info
            and type(record.exc_info[1]).__name__ == "DatasetSoftDeletedTwinExistsError"] == [
        ("WARNING", SETTLED.format("dataset"))]


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_a_raised_twin_refusal_the_settle_keeps_is_answered_by_the_callers_access(lock_rows, dialect):
    """The control: the same refusal, the marker's lock held throughout and no dataset made meanwhile. The settle runs
    after the command's rollback (the second lock, the look, the hold check) and finds nothing, so the refusal goes on:
    once the marker's lock is let go it is answered by the caller's access to the table, `{"exists": true}` for a
    caller whose grants reach it, as without the settle. Nothing more is written."""
    superset = lock_rows.superset
    superset.session = Metadata(superset, dialect)
    superset.twin = True
    assert make_dataset() == {"exists": True}
    assert _after(superset, "CreateDatasetCommand") == [("rollback",), *settled("dataset", dialect=dialect),
                                                        ("raise_for_access", "builder", ANY)]
    assert commands(superset, "UpdateDatasetCommand") == []


@pytest.mark.parametrize("kind", ["config chart", "funnel chart", "physical dataset"])
@pytest.mark.parametrize("planted", ["no marker", "another build's marker"])
def test_an_unmarked_object_of_the_same_name_is_never_answered(request, kind, planted):
    """Objects of the request's own name (a chart on the same dataset) whose description does not end in the
    request's marker, none or another build's, are never answered: the create runs, answers the new object's id, and
    leaves the planted ones as they were."""
    chart = "chart" in kind
    stage = request.getfixturevalue("creating" if chart else "lock_rows")
    superset = stage.superset
    other = format_marker("1b4e28ba-2fa1-41d2-883f-0016d3cca427", "chart_1" if chart else "data_1", 12 if chart else 3)
    description = "Prose of its own." if planted == "no marker" else f"Prose of its own.\n{other}"
    named = {"slice_name": "Premium" if kind == "config chart" else "Stages", "datasource_id": 7} if chart \
        else {"table_name": "sales"}
    twins = [SimpleNamespace(**vars(_row(41 + n, description)), **named) for n in range(2)]
    before = [dict(vars(twin)) for twin in twins]
    superset.rows["Slice" if chart else "SqlaTable"].extend(twins)
    answer = {"config chart": make_from_config, "funnel chart": make_funnel, "physical dataset": make_dataset}[kind]()
    key = "chart_id" if chart else "dataset_id"
    assert set(answer) == {key} and answer[key] not in {41, 42}, answer
    assert len(commands(superset, "CreateChartCommand" if chart else "CreateDatasetCommand")) == 1
    assert [dict(vars(twin)) for twin in twins] == before


# ---------------------------------------------------------------- create_dataset

DATA = format_marker(BUILD, "data_1", 3)


@pytest.fixture
def making(claiming):
    """`claiming`'s dashboard (3) and people, and datasets the create commands add: a row the editors step locks is
    the one made, found by id; any other row lock (a create's second lock on MySQL) is the metadata database's
    (`Metadata.lock_row`)."""
    superset = claiming.superset
    superset.dataset_database = SimpleNamespace(perm=DATABASE_PERM, get_default_catalog=lambda: "warehouse")
    rows = superset.rows["SqlaTable"]

    def stubs_lock(model: type, entity_id: int) -> Any:
        if model.__name__ != "SqlaTable":
            return superset.session.lock_row(model, entity_id)
        return next((row for row in rows if row.id == entity_id), None)

    claiming.monkeypatch_lock = stubs_lock
    return claiming


@pytest.fixture
def lock_rows(monkeypatch, making):
    monkeypatch.setattr(writes, "lock_row", making.monkeypatch_lock)
    return making


def make_dataset(marker: str = DATA, **more) -> dict:
    request = {"database_id": 1, "schema": "public", "table_name": "sales", "dashboard_id": 3, "marker": marker}
    return writes.create_dataset({**request, **more})


def test_create_dataset_physical_carries_the_marker_and_data_editors(lock_rows):
    """The table's dataset is made with the marker as its description, in the create's own command; then, under the
    dataset's row lock, its editors are the caller and the dashboard's users whose grants already reach it."""
    superset, ids = lock_rows.superset, lock_rows.ids
    answer = make_dataset()
    assert set(answer) == {"dataset_id"}, answer
    assert commands(superset, "CreateDatasetCommand") == [("CreateDatasetCommand", {
        "database": 1, "schema": "public", "table_name": "sales", "description": DATA})]
    ((_, (dataset_id, properties)),) = commands(superset, "UpdateDatasetCommand")
    assert dataset_id == answer["dataset_id"] and properties == {"editors": [ids["builder"], ids["database_reader"]]}
    assert _kinds(superset).index("CreateDatasetCommand") < _kinds(superset).index("UpdateDatasetCommand")
    assert superset.g.user is lock_rows.people["builder"]


def test_create_dataset_virtual_carries_its_prose_then_the_marker(lock_rows):
    answer = make_dataset(table_name="active_sales", sql="SELECT 1 AS n", description="Active sales")
    assert set(answer) == {"dataset_id"}, answer
    ((_, properties),) = commands(lock_rows.superset, "CreateDatasetCommand")
    assert properties == {"database": 1, "schema": "public", "table_name": "active_sales", "sql": "SELECT 1 AS n",
                          "description": f"Active sales\n{DATA}"}


def test_create_dataset_twice_with_one_marker_makes_one(lock_rows):
    first = make_dataset()
    assert make_dataset() == first
    assert len(commands(lock_rows.superset, "CreateDatasetCommand")) == 1


def test_a_physical_dataset_this_step_made_is_found_before_the_exists_check(lock_rows):
    """The first create's answer was lost; the table now has its dataset, so rc2 would refuse the same create as
    already existing. The retry finds the dataset by its marker first and never sends the create."""
    from tests.superset_stubs import DatasetInvalidError

    superset = lock_rows.superset
    first = make_dataset()
    superset.create = DatasetInvalidError({"DatasetExistsValidationError"})
    assert make_dataset() == first
    assert len(commands(superset, "CreateDatasetCommand")) == 1


def test_a_found_dataset_gets_its_editors_again(lock_rows, monkeypatch):
    """A call cut between the create and the editors leaves a marked dataset; the retry finds it and sets them."""
    from tests import superset_stubs as stubs_module

    superset = lock_rows.superset
    real = lock_rows.monkeypatch_lock

    def contended(model, entity_id):
        raise stubs_module.database_error("55P03")

    monkeypatch.setattr(writes, "lock_row", contended)
    assert make_dataset() == {"busy": True}
    assert commands(superset, "UpdateDatasetCommand") == []
    (row,) = superset.rows["SqlaTable"]
    monkeypatch.setattr(writes, "lock_row", real)
    assert make_dataset() == {"dataset_id": row.id}
    assert len(commands(superset, "CreateDatasetCommand")) == 1
    assert _sent_editors(superset, "UpdateDatasetCommand") == [lock_rows.ids["builder"],
                                                              lock_rows.ids["database_reader"]]


def test_an_existing_table_dataset_answers_exists(lock_rows):
    lock_rows.superset.create = stubs.DatasetInvalidError({"DatasetExistsValidationError"})
    assert make_dataset() == {"exists": True}
    assert commands(lock_rows.superset, "UpdateDatasetCommand") == []


SALES_SQL = "SELECT * FROM public.sales"
# A dataset over a table, and one over SQL.
SOURCES = [pytest.param({}, id="a table"), pytest.param({"table_name": "active_sales", "sql": SALES_SQL}, id="sql")]


def _asked_for_access(superset: SimpleNamespace) -> list:
    return [entry for entry in superset.ran if entry[0] == "raise_for_access"]


def _access_rc2_asks(superset: SimpleNamespace, source: dict) -> dict:
    """What rc2's `CreateDatasetCommand.validate` asks `raise_for_access` for `make_dataset(**source)`: the database
    read from its id, and the database's default catalog, as `validate` resolves them; the table, or the SQL with its
    catalog and schema."""
    database = superset.dataset_database
    if "sql" in source:
        return {"database": database, "table": None, "sql": SALES_SQL, "catalog": "warehouse", "schema": "public"}
    return {"database": database, "table": stubs.Table("sales", "public", "warehouse"), "sql": None, "catalog": None,
            "schema": None}


@pytest.mark.parametrize("source", SOURCES)
def test_a_table_whose_dataset_is_in_the_trash_is_no_dataset_to_a_caller_who_may_not_read_it(lock_rows, source):
    """A caller who may not read the table (or what a virtual dataset's SQL reads) is answered
    `{"not_found": "dataset"}` when a dataset in the trash holds the table, as when none does (the control): the
    answer never says one is there. Its access is asked as the caller, with what the command resolved (the database
    read from its id, its default catalog), once the marker's lock is released. Nothing more is written."""
    superset = lock_rows.superset
    superset.session = Metadata(superset, "mysql")
    superset.g.user = lock_rows.people["outsider"]
    assert make_dataset(**source) == {"not_found": "dataset"}
    superset.ran.clear()
    superset.twin = True
    assert make_dataset(**source) == {"not_found": "dataset"}
    ((_, user, asked),) = _asked_for_access(superset)
    assert user == "outsider" and asked == _access_rc2_asks(superset, source)
    kinds = _kinds(superset)
    assert kinds.index("CreateDatasetCommand") < kinds.index("close") < kinds.index("raise_for_access")
    assert commands(superset, "UpdateDatasetCommand") == []


@pytest.mark.parametrize("source", SOURCES)
def test_a_table_whose_dataset_is_in_the_trash_exists_to_a_caller_who_may_read_it(lock_rows, source):
    """The same table, for a caller whose grants reach it: `{"exists": true}`, its access asked as that caller.
    Nothing more is written."""
    superset = lock_rows.superset
    superset.twin = True
    assert make_dataset(**source) == {"exists": True}
    ((_, user, asked),) = _asked_for_access(superset)
    assert user == "builder" and asked == _access_rc2_asks(superset, source)
    assert commands(superset, "UpdateDatasetCommand") == []


@pytest.mark.parametrize(("error", "answer"), [
    pytest.param(stubs.SQLAlchemyError("the metadata database went away"), {"failed": True}, id="a database error"),
    pytest.param(stubs.database_error("55P03"), {"busy": True}, id="a lock wait past the timeout")])
def test_a_database_error_checking_access_over_a_table_whose_dataset_is_in_the_trash_is_answered(lock_rows, error,
                                                                                                answer):
    """The access check reads Superset's metadata database, which may fail: answered as any database failure of the
    create is (`{"failed": true}`, or `{"busy": true}` for lock contention), never raised out of the tool."""
    lock_rows.security.fails = error
    lock_rows.superset.twin = True
    assert make_dataset() == answer
    assert len(_asked_for_access(lock_rows.superset)) == 1
    assert commands(lock_rows.superset, "UpdateDatasetCommand") == []


def test_sql_rc2_cannot_parse_over_a_table_whose_dataset_is_in_the_trash_is_refused_as_invalid(lock_rows):
    """SQL whose access cannot be asked, because it does not parse: refused as invalid, as the same SQL is with no
    dataset in the trash (the control), so this answer does not say one is there either."""
    lock_rows.security.unparseable = True
    assert make_dataset(table_name="active_sales", sql=SALES_SQL) == {"refused": "invalid"}
    lock_rows.superset.twin = True
    assert make_dataset(table_name="active_sales", sql=SALES_SQL) == {"refused": "invalid"}
    assert commands(lock_rows.superset, "UpdateDatasetCommand") == []


@pytest.mark.parametrize("classnames", [
    pytest.param({"DatasetDataAccessIsNotAllowed"}, id="a table the caller may not read"),
    pytest.param({"DatasetDataAccessIsNotAllowed", "DatasetExistsValidationError"},
                 id="a table the caller may not read that has a dataset")])
def test_a_table_the_caller_may_not_read_is_answered_as_no_dataset(lock_rows, classnames):
    """A caller who may not read the table (or what a virtual dataset's SQL reads), refused with
    `DatasetDataAccessIsNotAllowed`, is answered `{"not_found": "dataset"}`, with a dataset over the table or none:
    an object the caller may not read is answered as one that does not exist. Nothing more is written."""
    from tests import superset_stubs as stubs_module

    lock_rows.superset.create = stubs_module.DatasetInvalidError(classnames)
    assert make_dataset() == {"not_found": "dataset"}
    assert commands(lock_rows.superset, "UpdateDatasetCommand") == []


def test_another_invalid_dataset_is_refused_as_invalid(lock_rows):
    from tests import superset_stubs as stubs_module

    lock_rows.superset.create = stubs_module.DatasetInvalidError({"TableNotFoundValidationError"})
    assert make_dataset() == {"refused": "invalid"}


def test_create_dataset_checks_the_marker_and_the_dashboard_before_anything(lock_rows):
    superset = lock_rows.superset
    assert make_dataset(marker=format_marker(BUILD, "data_1", 4)) == {"refused": "marker"}
    assert make_dataset(dashboard_id=99, marker=format_marker(BUILD, "data_1", 99)) == {"not_found": "dashboard"}
    assert [entry for entry in superset.ran if entry[0] in ("lookup", "execute")] == []
    assert commands(superset) == []


def test_the_extension_offers_exactly_the_eleven_tools():
    from mnemiq.superset.extension import EXTENSION_TOOLS

    assert EXTENSION_TOOLS == (
        "create_dashboard", "create_dataset", "create_chart", "patch_chart", "add_saved_metric", "delete_dataset",
        "set_layout", "dataset_metrics", "editors_of", "get_layout", "row_rules")
    assert registered(TOOLS_SOURCE.read_text()).keys() == set(EXTENSION_TOOLS)


# ---------------------------------------------------------------- a histogram, checked by the extension's own query

HISTOGRAM_FORM = {"viz_type": "histogram_v2", "column": "amount", "groupby": ["region"], "bins": 10,
                  "normalize": False, "cumulative": False, "row_limit": 10000,
                  "adhoc_filters": [{"clause": "WHERE", "expressionType": "SIMPLE", "subject": "region",
                                     "operator": "==", "comparator": "West"}]}


def test_a_histogram_from_a_config_is_saved(creating):
    """rc2's mapping makes the form data and `set` is merged over it; the saved params are those; the query context is
    none (the histogram is not checked by the value check); the marker and the editors are a created chart's."""
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    made = make_from_config(set={"number_format": ",.2f"})
    assert set(made) == {"chart_id"}, made
    ((_, properties),) = commands(creating.superset)
    assert json.loads(properties["params"]) == {**HISTOGRAM_FORM, "number_format": ",.2f"}
    assert properties["query_context"] is None
    assert properties["viz_type"] == "histogram_v2" and properties["description"] == STEP
    assert properties["editors"] == [101]


def test_a_histogram_without_a_column_is_refused_as_config(creating):
    """Refused under the marker's lock, after the look, as a params error is; nothing is made."""
    creating.superset.mapped = {key: value for key, value in HISTOGRAM_FORM.items() if key != "column"}
    assert make_from_config() == {"refused": "config"}
    ran = creating.superset.ran
    assert ran[:8] == [*taken("postgresql", STEP), *looked("postgresql", "Slice")]
    assert ran[-3:] == [("rollback",), *let_go("postgresql", STEP)]
    assert commands(creating.superset) == [] and "chart_data" not in _kinds(creating.superset)


def test_a_histograms_check_selects_its_column_its_groups_under_its_filters(creating):
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    creating.superset.chart_data = {"queries": [{"data": [{"region": "West", "0 - 1": 2}], "status": "success"}]}
    assert set(make_from_config()) == {"chart_id"}
    ((_, context),) = [entry for entry in creating.superset.ran if entry[0] == "chart_data"]
    (query,) = context["queries"]
    assert query["columns"] == ["region", "amount"] and "metrics" not in query
    assert query["filters"] == [{"col": "region", "op": "==", "val": "West"}, {"col": "region", "op": "IS NOT NULL"},
                                {"col": "amount", "op": "IS NOT NULL"}]
    assert query["row_limit"] == writes._CHART_ROW_LIMIT_PROBE
    assert context["result_type"] == "full"
    assert query["post_processing"] == [{"operation": "histogram", "options": {
        "column": "amount", "groupby": ["region"], "bins": 10, "cumulative": False, "normalize": False}}]


def test_a_mapping_a_plugin_refuses_is_refused_as_config(creating):
    """rc2's `map_config_to_form_data` raises ValueError when a plugin's `post_map_validate` refuses (a histogram's
    non-numeric column; the mapping runs before the viz type is known, so any family): refused under the marker's lock, after the look; nothing is made."""
    creating.superset.mapped = ValueError("the column is not numeric")
    assert make_from_config() == {"refused": "config"}
    ran = creating.superset.ran
    assert ran[:8] == [*taken("postgresql", STEP), *looked("postgresql", "Slice")]
    assert ran[-3:] == [("rollback",), *let_go("postgresql", STEP)]
    assert commands(creating.superset) == [] and "chart_data" not in _kinds(creating.superset)


@pytest.mark.parametrize("bins", ["inf", "Infinity", 1e999])
def test_a_histogram_with_bins_that_are_not_finite_is_refused_as_config(creating, bins):
    creating.superset.mapped = {**copy.deepcopy(HISTOGRAM_FORM), "bins": bins}
    assert make_from_config() == {"refused": "config"}
    assert commands(creating.superset) == [] and "chart_data" not in _kinds(creating.superset)


def _rows_when_groups_are_not_filtered(rows):
    """A stub answer for a probe: the filtered probe (a group-not-null filter) finds no rows; the one without it
    finds `rows`."""
    def answer(context):
        (query,) = context["queries"]
        filtered = {"col": "region", "op": "IS NOT NULL"} in query["filters"]
        return {"queries": [{"data": [] if filtered else rows, "status": "success"}]}
    return answer


def test_a_histogram_with_observations_and_no_usable_group_is_refused(creating):
    """Every group value null: the group-not-null probe is empty, but the chart keeps those rows and its operator
    cannot unstack them, so a second probe, without the group filter, finds them and the chart is refused."""
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    creating.superset.chart_data = _rows_when_groups_are_not_filtered([{"region": None, "amount": 1}])
    assert make_from_config() == {"refused": "query_failed"}
    assert len([entry for entry in creating.superset.ran if entry[0] == "chart_data"]) == 2
    assert commands(creating.superset) == []


def test_a_histogram_of_nothing_is_saved(creating):
    """No rows at all (an empty table or a filter that keeps none): an empty chart, which passes."""
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    creating.superset.chart_data = _rows_when_groups_are_not_filtered([])
    assert set(make_from_config()) == {"chart_id"}


def test_a_histogram_with_usable_groups_runs_one_probe(creating):
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    creating.superset.chart_data = {"queries": [{"data": [{"region": "West", "0 - 1": 2}], "status": "success"}]}
    assert set(make_from_config()) == {"chart_id"}
    assert len([entry for entry in creating.superset.ran if entry[0] == "chart_data"]) == 1


@pytest.mark.parametrize("bins", [None, "", 10**400])
def test_a_histogram_with_bins_the_operator_cannot_take_is_refused(creating, bins):
    """Null and a blank string are 0 bins in the frontend, which the operator refuses on the probe; an integer past
    any float is refused as config."""
    creating.superset.mapped = {**copy.deepcopy(HISTOGRAM_FORM), "bins": bins}
    creating.superset.chart_data = lambda context: (_ for _ in ()).throw(
        ValueError("`bins` must be an integer")) if context["queries"][0]["post_processing"][0]["options"]["bins"] == 0 \
        else {"queries": [{"data": [], "status": "success"}]}
    answer = make_from_config()
    assert answer == ({"refused": "config"} if bins == 10**400 else {"refused": "query_failed"})
    assert commands(creating.superset) == []


def test_a_histogram_whose_post_processing_raises_is_refused_and_nothing_is_made(creating):
    """A populated non-numeric column: the histogram operator raises ValueError on the probe's rows."""
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    creating.superset.chart_data = ValueError("Column 'amount' contains non-numeric values")
    assert make_from_config() == {"refused": "query_failed"}
    assert commands(creating.superset) == []


def test_a_histogram_whose_check_fails_is_refused_and_nothing_is_made(creating):
    """A column the dataset lacks, or a query the database fails: refused as `query_failed`, the look under the lock
    having run first; nothing is made."""
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    creating.superset.chart_data = {"queries": [{"error": "no such column", "status": "failed"}]}
    assert make_from_config() == {"refused": "query_failed"}
    ran = creating.superset.ran
    assert ran[:8] == [*taken("postgresql", STEP), *looked("postgresql", "Slice")]
    assert _after(creating.superset, "chart_data") == settled("config chart")
    assert commands(creating.superset) == []


def test_only_a_histogram_bypasses_rc2s_compile_check(creating):
    """A line chart is still checked by rc2's `_compile_chart`; a histogram never is."""
    creating.superset.mapped = {"viz_type": "echarts_timeseries_line", "x_axis": "day", "metrics": ["count"]}
    assert set(make_from_config()) == {"chart_id"}
    assert "compile" in _kinds(creating.superset) and "chart_data" not in _kinds(creating.superset)
    creating.superset.ran.clear()
    creating.superset.mapped = copy.deepcopy(HISTOGRAM_FORM)
    assert set(make_from_config(format_marker(BUILD, "chart_2", 12))) == {"chart_id"}
    assert "compile" not in _kinds(creating.superset) and "chart_data" in _kinds(creating.superset)


# ---------------------------------------------------------------- create_chart's leave_out_no_value, config route only

PREMIUM_SQL = {"expressionType": "SQL", "sqlExpression": "SUM(premium)", "label": "Total premium",
               "hasCustomLabel": True, "optionName": "metric_sql_1"}
WEST = {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "region", "operator": "==", "comparator": "west"}
NO_TIME = {"clause": "WHERE", "expressionType": "SIMPLE", "subject": "issued", "operator": "TEMPORAL_RANGE",
           "comparator": "No filter"}
BAR_FORM = {"viz_type": "echarts_timeseries_bar", "x_axis": "channel", "groupby": [], "metrics": [PREMIUM_SQL],
            "adhoc_filters": [WEST, NO_TIME], "row_limit": 10}


def _leave_out(condition: str) -> dict:
    return {"clause": "HAVING", "expressionType": "SQL", "sqlExpression": condition, "subject": None,
            "operator": None, "comparator": None, "isExtra": False, "isNew": False,
            "filterOptionName": "mnemiq_leave_out_no_value"}


def test_the_extensions_condition_is_the_text_mnemiqs_check_recognises():
    from mnemiq.superset.charts import not_null_condition

    for sql in ("SUM(premium)", "SUM(CASE WHEN status = 'R&D' THEN premium END)"):
        assert writes.not_null_condition(sql) == not_null_condition(sql)


@pytest.mark.parametrize("flag", ["yes", 1, None, [True]])
def test_create_chart_refuses_a_leave_out_no_value_that_is_not_a_boolean(flag):
    with pytest.raises(writes.Refused) as caught:
        writes.read("create_chart", {"dataset_id": 7, "slice_name": "Premium", "dashboard_id": 12, "marker": STEP,
                                     "config": CONFIG, "leave_out_no_value": flag})
    assert caught.value.answer == {"refused": "request", "field": "leave_out_no_value"}


def test_the_condition_is_built_from_the_metric_as_mcp_saved_it_and_saved_last(creating):
    """The HAVING is derived from the ranked metric in the mapped form, after MCP's own sanitising (here the mapped
    metric holds '&' where the plan's text might have held an entity), so the saved filter, the query's extras and
    the chart's own metric agree."""
    mapped = copy.deepcopy(BAR_FORM)
    mapped["metrics"][0]["sqlExpression"] = "SUM(CASE WHEN dept = 'R&D' THEN premium END)"
    creating.superset.mapped = mapped
    made = make_from_config(set={"number_format": ",.2f"}, leave_out_no_value=True)
    assert set(made) == {"chart_id"}, made
    ((_, properties),) = commands(creating.superset)
    saved = json.loads(properties["params"])
    condition = "(SUM(CASE WHEN dept = 'R&D' THEN premium END)) IS NOT NULL"
    assert saved["adhoc_filters"] == [WEST, NO_TIME, _leave_out(condition)] and saved["number_format"] == ",.2f"
    context = json.loads(properties["query_context"])
    assert context == params.query_context("echarts_timeseries_bar", saved, 7)
    assert context["queries"][0]["extras"]["having"] == f"({condition})"  # `sanitizeClause` wraps each clause


def test_a_saved_metrics_condition_is_its_expression_on_the_dataset(creating):
    creating.superset.mapped = {**copy.deepcopy(BAR_FORM), "metrics": ["revenue"]}
    creating.superset.datasets[7].metrics = [SimpleNamespace(metric_name="revenue", expression="SUM(net_amount)")]
    assert set(make_from_config(leave_out_no_value=True)) == {"chart_id"}
    ((_, properties),) = commands(creating.superset)
    assert json.loads(properties["params"])["adhoc_filters"][-1] == _leave_out("(SUM(net_amount)) IS NOT NULL")


@pytest.mark.parametrize("mapped", [
    {**BAR_FORM, "metrics": [{"expressionType": "SIMPLE", "aggregate": "SUM", "column": {"column_name": "premium"},
                              "label": "Premium"}]},
    {**BAR_FORM, "metrics": ["no_such_metric"]},
    {**BAR_FORM, "metrics": []},
    {"viz_type": "big_number_total", "metric": "count"},
], ids=["a SIMPLE metric", "a saved metric the dataset does not hold", "no metric", "a type ranked by no metric"])
def test_a_chart_whose_ranked_metric_cannot_be_read_is_refused_and_not_saved(creating, mapped):
    creating.superset.mapped = copy.deepcopy(mapped)
    creating.superset.datasets[7].metrics = []
    assert make_from_config(leave_out_no_value=True) == {"refused": "leave_out_no_value"}
    assert commands(creating.superset) == []


def test_without_leave_out_no_value_the_saved_chart_is_as_before(creating):
    creating.superset.mapped = copy.deepcopy(BAR_FORM)
    made = make_from_config()
    assert set(made) == {"chart_id"}, made
    ((_, properties),) = commands(creating.superset)
    assert json.loads(properties["params"]) == BAR_FORM


def test_a_funnel_or_sankey_asked_to_leave_out_no_value_is_refused(creating):
    found = {"dataset_id": 7, "viz_type": "funnel", "slice_name": "Stages",
             "params": {"groupby": ["status"], "metric": "count"}, "dashboard_id": 12, "marker": STEP,
             "leave_out_no_value": True}
    assert writes.create_chart(found) == {"refused": "request"}
    assert commands(creating.superset) == []
