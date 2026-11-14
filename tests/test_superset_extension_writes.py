"""What the extension's write tools (`agenticfabriq.mnemiq.writes`) refuse before they touch Superset: each request
is read strictly, field by field, and a marker, a metric and an expected `changed_on` are held to their shapes.

`writes` imports Superset only inside its tools, so this runs in `unit` with neither Superset nor Flask installed.
What the tools do inside Superset (the locks, the commands, the answers) is the live suite's:
`tests/superset_live/test_extension_tools.py`.
"""

import ast
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
READS = {"dataset_metrics", "editors_of", "get_layout"}  # the read tools, registered by a later change


def registration_problems(source: str) -> list[str]:
    """What is wrong with the `@tool` registrations in `source`, read with `ast`: each must be named one of mnemiq's
    write tool names, with that tool's stock class permission and the method `write`, take one `request`, and call
    `writes.<its name>` and nothing else of `writes`; and every write tool must be there once."""
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
            if name not in WRITE_CLASSES:
                problems.append(f"{name!r} is not one of mnemiq's write tools")
                continue
            if keywords.get("class_permission_name") != WRITE_CLASSES[name]:
                problems.append(f"{name} is not gated by {WRITE_CLASSES[name]}")
            if keywords.get("method_permission_name") != "write":
                problems.append(f"{name} is not gated as a write")
            if [arg.arg for arg in node.args.args] != ["request"]:
                problems.append(f"{name} does not take one request")
            called = [call.func.attr for call in ast.walk(node) if isinstance(call, ast.Call)
                      and isinstance(call.func, ast.Attribute) and getattr(call.func.value, "id", None) == "writes"]
            if called != [name]:
                problems.append(f"{name} calls writes.{called}")
    if sorted(seen) != sorted(WRITE_CLASSES):
        problems.append(f"registered {sorted(filter(None, seen))}, not each write tool once")
    return problems


def test_the_write_tools_are_mnemiqs_names_with_stock_permissions():
    from mnemiq.superset.extension import EXTENSION_TOOLS

    assert set(EXTENSION_TOOLS) == set(WRITE_CLASSES) | READS
    assert set(writes.REQUESTS) == set(WRITE_CLASSES)
    assert registration_problems(TOOLS_SOURCE.read_text()) == []


@pytest.mark.parametrize(("old", "new", "flagged"), [
    ('name="patch_chart"', 'name="patch_charts"', "'patch_charts' is not one of mnemiq's write tools"),
    ('name="set_layout", class_permission_name="Dashboard"', 'name="set_layout", class_permission_name="mnemiq"',
     "set_layout is not gated by Dashboard"),
    ('name="delete_dataset", class_permission_name="Dataset", method_permission_name="write"',
     'name="delete_dataset", class_permission_name="Dataset", method_permission_name="read"',
     "delete_dataset is not gated as a write"),
    ("return writes.claim_dataset(request)", "return writes.claim_chart(request)",
     "claim_dataset calls writes.['claim_chart']"),
])
def test_the_registration_check_flags_a_planted_defect(old, new, flagged):
    source = TOOLS_SOURCE.read_text()
    assert source.count(old) == 1
    assert flagged in registration_problems(source.replace(old, new))


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
