"""What the extension's write tools (`agenticfabriq.mnemiq.writes`) refuse before they touch Superset: each request
is read strictly, field by field, and a marker, a metric and an expected `changed_on` are held to their shapes.

`writes` imports Superset only inside its tools, so this runs in `unit` with neither Superset nor Flask installed.
What the tools do inside Superset (the locks, the commands, the answers) is the live suite's:
`tests/superset_live/test_extension_tools.py`.
"""

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
    assert set(writes.REQUESTS) == {"create_dashboard", "claim_chart", "claim_dataset", "create_chart", "patch_chart",
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
    # The same claim made again leaves one marker line, not two.
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
WRITE_CLASSES = {"create_dashboard": "Dashboard", "claim_chart": "Chart", "claim_dataset": "Dataset",
                 "create_chart": "Chart", "patch_chart": "Chart", "add_saved_metric": "Dataset",
                 "delete_dataset": "Dataset", "set_layout": "Dashboard"}
# The class permission each read tool is gated by; its method is `read`.
READ_CLASSES = {"dataset_metrics": "Dataset", "editors_of": "Chart", "get_layout": "Dashboard"}
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
    ("return writes.claim_dataset(request)", "return writes.claim_chart(request)",
     "claim_dataset calls ['writes.claim_chart']"),
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
# Creating a dashboard or a chart writes a new row and locks none.
LOCKED = {"patch_chart": ("Slice", "chart_id"), "claim_chart": ("Slice", "chart_id"),
          "claim_dataset": ("SqlaTable", "dataset_id"), "add_saved_metric": ("SqlaTable", "dataset_id"),
          "delete_dataset": ("SqlaTable", "dataset_id"), "set_layout": ("Dashboard", "dashboard_id")}
UNLOCKED = {"create_dashboard", "create_chart"}


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
    # A claim that compares markers on an unlocked read of the chart, before its lock.
    ('    return _locked(Slice, found["chart_id"], lambda chart: _claim(\n',
     '    from superset.daos.chart import ChartDAO\n'
     '    held = params.parse_marker(ChartDAO.find_by_id(found["chart_id"]).description)\n'
     '    if held is not None and held != claim_marker(found["marker"], found["dashboard_id"]):\n'
     '        return CLAIMED_BY_ANOTHER\n'
     '    return _locked(Slice, found["chart_id"], lambda chart: _claim(\n',
     "claim_chart reads before its lock: find_by_id, parse_marker"),
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


# ---------------------------------------------------------------- who becomes an editor

USER, ROLE, GROUP = stubs.SubjectType.USER, stubs.SubjectType.ROLE, stubs.SubjectType.GROUP
DATASET_PERM = "[warehouse].[sales](id:4)"
SCHEMA_PERM = "[warehouse].[public]"


def _person(user_id: int, name: str, grants=(), member_of=()) -> SimpleNamespace:
    """A user, its own subject (id 100 + its id) and the role and group subjects it is a member of; `grants` are the
    `(permission, view)` pairs its roles hold."""
    user = SimpleNamespace(id=user_id, username=name, grants=set(grants))
    user.subject = SimpleNamespace(id=100 + user_id, type=USER, label=f"{name} Live", user=user)
    user.subject_ids = {user.subject.id, *member_of}
    return user


class Security:
    """rc2's security manager, stood in for the checks a claim makes, each read as `flask.g.user`, as rc2 reads them
    (`superset/security/manager.py`): `can_access` is a grant of the user's roles; `can_access_schema` is
    all-datasource, database or schema access; `is_editor` is any of the user's subjects (its own, its roles', its
    groups') among the object's editors; and `can_access_datasource` is `raise_for_access`'s datasource leg, which
    counts editorship. `judged` records each user the checks were asked as."""

    def __init__(self, g, raises_for: str | None = None) -> None:
        self.g, self.raises_for, self.judged = g, raises_for, []

    def can_access(self, permission: str, view: str) -> bool:
        user = self.g.user
        self.judged.append(user.username)
        if user.username == self.raises_for:
            raise RuntimeError("the metadata database went away")
        return (permission, view) in user.grants

    def can_access_schema(self, dataset) -> bool:
        return (self.can_access("all_datasource_access", "all_datasource_access")
                or self.can_access("database_access", dataset.database.perm)
                or self.can_access("schema_access", dataset.schema_perm))

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
    """A dashboard (3) a builder edits with five co-editors, and a dataset (4) on table `sales`:
    - `builder`, the caller, who may read every dataset and made the dataset;
    - `reader`, granted `datasource_access` on the dataset, and `schema_reader`, granted its schema: both may read it;
    - `outsider`, granted nothing, and `member`, granted nothing but a member of the role `R2`, which already edits the
      dataset: neither may read it by a grant (`member` reads it only as an editor);
    - the role `R` and the group `G`, editors of the dashboard.
    The dataset's editors before the claim: its creator `creator` (not on the dashboard), the builder and `R2`."""
    people = {name: _person(user_id, name, grants, member_of) for user_id, name, grants, member_of in (
        (1, "builder", {("all_datasource_access", "all_datasource_access")}, ()),
        (2, "reader", {("datasource_access", DATASET_PERM)}, ()),
        (3, "outsider", set(), ()),
        (4, "creator", set(), ()),
        (5, "schema_reader", {("schema_access", SCHEMA_PERM)}, ()),
        (6, "member", set(), (302,)))}
    role, group, role_2 = (SimpleNamespace(id=301, type=ROLE, label="R", user=None),
                           SimpleNamespace(id=401, type=GROUP, label="G", user=None),
                           SimpleNamespace(id=302, type=ROLE, label="R2", user=None))
    builder = people["builder"]
    superset.g.user = builder
    superset.user_subjects = {person.id: person.subject for person in people.values()}
    dashboard = SimpleNamespace(id=3, editors=[people[name].subject for name in (
        "builder", "reader", "outsider", "schema_reader", "member")] + [role, group])
    dataset = SimpleNamespace(id=4, perm=DATASET_PERM, schema_perm=SCHEMA_PERM, description=None,
                              database=SimpleNamespace(perm="[warehouse].(id:1)"),
                              editors=[people["creator"].subject, builder.subject, role_2])
    superset.dashboards, superset.datasets = {3: dashboard}, {4: dataset}
    security = Security(superset.g)
    monkeypatch.setattr(writes, "_security", lambda: security)
    monkeypatch.setattr(writes, "_session", lambda: SimpleNamespace(rollback=lambda: None))
    locked = {"SqlaTable": dataset}
    monkeypatch.setattr(writes, "lock_row", lambda model, entity_id: locked[model.__name__])
    return SimpleNamespace(superset=superset, security=security, people=people, dashboard=dashboard,
                           dataset=dataset, locked=locked, ids={name: p.subject.id for name, p in people.items()})


def _sent_editors(superset, command: str) -> list[int]:
    """The editors the one `command` run sent: an update command takes `(id, properties)`, a create `properties`."""
    (properties,) = [arguments if isinstance(arguments, dict) else arguments[-1] for name, arguments in superset.ran
                     if name == command]
    return properties["editors"]


def test_a_claimed_dataset_gets_only_editors_who_could_already_read_it(claiming):
    """rc2 admits a dataset's editors to its data (`raise_for_access`) and, in SQL Lab, to its table, and an editor
    subject that is a role or a group makes every member an editor. So `claim_dataset` keeps the dataset's editors,
    adds the caller, and adds only those of the dashboard's users who may already read the dataset by a grant, judged
    as each of them: `reader` (datasource access) and `schema_reader` (schema access). Not `outsider`, not `member`
    (who reads it only as an editor, through `R2`), never the role `R` or the group `G`. The request's user is the
    builder again afterwards."""
    ids = claiming.ids
    answer = writes.claim_dataset({"dataset_id": 4, "dashboard_id": 3, "marker": format_marker(BUILD, "data", 3)})
    assert answer == {"ok": True}
    assert _sent_editors(claiming.superset, "UpdateDatasetCommand") == [
        ids["creator"], ids["builder"], 302, ids["reader"], ids["schema_reader"]]
    assert {"reader", "schema_reader", "outsider", "member"} <= set(claiming.security.judged)
    assert claiming.superset.g.user is claiming.people["builder"]
    (properties,) = [arguments[1] for name, arguments in claiming.superset.ran if name == "UpdateDatasetCommand"]
    assert properties["description"] == format_marker(BUILD, "data", 3)


def test_the_request_user_comes_back_when_a_check_fails(claiming):
    """A check that raises while the claim judges a co-editor leaves the request's user the builder, which rc2's
    `override_user` would not (it puts the user back only when its block returns), and saves nothing."""
    claiming.security.raises_for = "outsider"
    with pytest.raises(RuntimeError):
        writes.claim_dataset({"dataset_id": 4, "dashboard_id": 3, "marker": format_marker(BUILD, "data", 3)})
    assert claiming.superset.g.user is claiming.people["builder"]
    assert claiming.superset.ran == []


def _chart(claiming, editors) -> SimpleNamespace:
    form_data = {"viz_type": "funnel", "groupby": ["status"], "metric": "count", "datasource": "4__table"}
    return SimpleNamespace(id=9, description=None, datasource_type="table", datasource_id=4, viz_type="funnel",
                           params=json.dumps(form_data), datasource=claiming.dataset, editors=editors)


@pytest.mark.parametrize("features, promiscuous", [(set(), False), ({"ENABLE_VIEWERS"}, False), (set(), True)])
def test_a_chart_gets_the_dashboards_editors(claiming, features, promiscuous):
    """A chart's editors may not read its data for being its editors unless `ENABLE_VIEWERS` is on and
    `VIEWER_PROMISCUOUS_MODE` is set (`has_promiscuous_chart_access` in rc2's `raise_for_access`), so short of both a
    claimed or created chart gets the dashboard's editors, every one, roles and groups included, as before."""
    claiming.superset.features, claiming.superset.config["VIEWER_PROMISCUOUS_MODE"] = features, promiscuous
    claiming.locked["Slice"] = _chart(claiming, [claiming.people["builder"].subject,
                                                 claiming.people["creator"].subject])
    everyone = [subject.id for subject in claiming.dashboard.editors]
    assert writes.claim_chart({"chart_id": 9, "dashboard_id": 3,
                               "marker": format_marker(BUILD, "kpi", 3)}) == {"ok": True}
    assert _sent_editors(claiming.superset, "UpdateChartCommand") == everyone
    made = writes.create_chart({"dataset_id": 4, "viz_type": "funnel", "slice_name": "stages",
                                "params": {"groupby": ["status"], "metric": "count"}, "dashboard_id": 3,
                                "marker": format_marker(BUILD, "stages", 3)})
    assert set(made) == {"chart_id"}, made
    assert _sent_editors(claiming.superset, "CreateChartCommand") == everyone


def test_a_chart_whose_editors_read_its_data_gets_the_dataset_rule(claiming):
    """With `ENABLE_VIEWERS` on and `VIEWER_PROMISCUOUS_MODE` set, a chart's editors may read its dataset's data, so a
    claimed chart keeps its editors and adds the caller and the dashboard's users who may already read the chart's
    dataset by a grant; a created chart gets the caller and those users."""
    claiming.superset.features, claiming.superset.config["VIEWER_PROMISCUOUS_MODE"] = {"ENABLE_VIEWERS"}, True
    ids = claiming.ids
    claiming.locked["Slice"] = _chart(claiming, [claiming.people["creator"].subject,
                                                 claiming.people["builder"].subject])
    assert writes.claim_chart({"chart_id": 9, "dashboard_id": 3,
                               "marker": format_marker(BUILD, "kpi", 3)}) == {"ok": True}
    assert _sent_editors(claiming.superset, "UpdateChartCommand") == [
        ids["creator"], ids["builder"], ids["reader"], ids["schema_reader"]]
    made = writes.create_chart({"dataset_id": 4, "viz_type": "funnel", "slice_name": "stages",
                                "params": {"groupby": ["status"], "metric": "count"}, "dashboard_id": 3,
                                "marker": format_marker(BUILD, "stages", 3)})
    assert set(made) == {"chart_id"}, made
    assert _sent_editors(claiming.superset, "CreateChartCommand") == [ids["builder"], ids["reader"],
                                                                       ids["schema_reader"]]
    assert claiming.superset.g.user is claiming.people["builder"]
