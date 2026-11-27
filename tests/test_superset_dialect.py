"""Superset mode's SQL dialect: a value set once at boot (`use_superset_dialect`) and read at each use
(`superset_dialect`), never a constant a module names or binds as it is imported, since a module imported before boot
would keep the value it found then."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.settings import superset_dialect, use_superset_dialect

SRC = Path(__file__).resolve().parents[1] / "src" / "mnemiq"
# The modules Superset mode's dialect reaches: its own package, and the runtime that boots it.
MODULES = sorted((SRC / "superset").rglob("*.py")) + [SRC / "runtime.py"]
# Where the value lives, set and read.
HOME = SRC / "superset" / "settings.py"
# The places that name PostgreSQL's dialect and keep it whatever Superset mode's is, each by the line that does: the
# dialect of a view's definition as the PostgreSQL catalogue reads it, which `pg_get_viewdef` writes in PostgreSQL's
# SQL. (Enrichment through Superset reads PostgreSQL only, and its check of the backend names SQLAlchemy's
# `postgresql`, which is no dialect sqlglot reads.)
KEPT = {("superset/pg_catalog.py", 'VIEW_DIALECT = "postgres"')}
# The modules that write or read SQL in Superset mode's dialect, each of which must be seen reading it at call time.
READERS = {"superset/adapter.py", "superset/chart_check.py", "superset/chart_form.py", "superset/grants.py",
           "superset/measures.py", "superset/names.py", "superset/plan_rows.py", "superset/planner.py",
           "superset/saved_metrics.py", "runtime.py"}

_CONSTANT = re.compile(r"\bSUPERSET_DIALECT\b")


def _relative(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _calls(tree: ast.AST) -> tuple[list[int], list[int]]:
    """The lines where `superset_dialect()` is called, split by when the call runs: as the module is imported (in a
    module's or a class's own statements, a default argument value, a decorator), or each time a function runs."""
    at_import: list[int] = []
    at_call: list[int] = []

    def visit(node: ast.AST, in_body: bool) -> None:
        if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id == "superset_dialect"
                or isinstance(node.func, ast.Attribute) and node.func.attr == "superset_dialect"):
            (at_call if in_body else at_import).append(node.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for part in (*node.decorator_list, node.args, *([node.returns] if node.returns else [])):
                visit(part, in_body)
            for statement in node.body:
                visit(statement, True)
        elif isinstance(node, ast.Lambda):
            visit(node.args, in_body)
            visit(node.body, True)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, in_body)

    visit(tree, False)
    return at_import, at_call


def test_the_dialect_defaults_to_postgres():
    """In a process that has only imported mnemiq, as before boot, and in this one, where every test but those under
    `superset_oracle` sees it."""
    fresh = subprocess.run(
        [sys.executable, "-c", "from mnemiq.superset.settings import superset_dialect; print(superset_dialect())"],
        capture_output=True, text=True, check=True)
    assert fresh.stdout.strip() == "postgres"
    assert superset_dialect() == "postgres"


@pytest.mark.parametrize("dialect", ["postgresql", "Postgres", "ORACLE", "oracle ", "duckdb", "mysql", "", None])
def test_use_superset_dialect_refuses_others(dialect):
    with pytest.raises(ValueError, match="postgres or oracle"):
        use_superset_dialect(dialect)
    assert superset_dialect() == "postgres"


def test_use_superset_dialect_sets_what_every_use_reads():
    """Read at each use: an adapter built after the dialect is set speaks it, though its module was imported before."""
    assert SupersetSqlAdapter(object(), database_id=7, schema="shop").dialect == "postgres"
    use_superset_dialect("oracle")
    assert superset_dialect() == "oracle"
    assert SupersetSqlAdapter(object(), database_id=7, schema="SHOP").dialect == "oracle"


@pytest.mark.parametrize("first, second", [("oracle", "postgres"), ("postgres", "oracle")])
def test_the_dialect_is_set_once_in_a_process(first, second):
    """Set once: every module reads the one value, so a second runtime booted in another dialect would change the SQL
    of the first one under it. Another value is refused with a plain error and changes nothing; the same one again is
    no change. Before any is set the dialect reads as PostgreSQL's, which is not a setting: either may be set."""
    assert superset_dialect() == "postgres"
    use_superset_dialect(first)
    use_superset_dialect(first)
    with pytest.raises(RuntimeError) as refused:
        use_superset_dialect(second)
    assert str(refused.value) == (
        f"Superset mode's SQL dialect is already {first} in this process, set as a server booted for a snapshot read "
        f"from that database: one process answers in one dialect, so a snapshot read from {second} is served from a "
        "process of its own")
    assert superset_dialect() == first


def test_an_unknown_dialect_is_refused_after_one_is_set():
    use_superset_dialect("oracle")
    with pytest.raises(ValueError, match="postgres or oracle"):
        use_superset_dialect("mysql")
    assert superset_dialect() == "oracle"


def test_a_tests_dialect_is_put_back_after_it(superset_oracle):
    """The conftest's guard: this test's process is in Oracle's dialect, and the next test's is not."""
    assert superset_dialect() == "oracle"


def test_the_test_before_left_no_dialect():
    """Runs after `test_a_tests_dialect_is_put_back_after_it`, in file order: nothing was left set, so Oracle's may be
    set here, which a dialect left set would refuse."""
    assert superset_dialect() == "postgres"
    use_superset_dialect("oracle")
    assert superset_dialect() == "oracle"


def test_no_module_names_the_dialect_but_settings():
    """No module names the dialect itself: neither the constant that once held it, nor sqlglot's name for PostgreSQL's
    dialect, except where the value lives and where PostgreSQL's SQL is read whatever the dialect (`KEPT`). sqlglot
    knows PostgreSQL's dialect by `postgres` alone, so every way to ask it for that dialect is a string `postgres`."""
    assert len(MODULES) > 20 and HOME in MODULES
    named, kept = [], set()
    for path in MODULES:
        text = path.read_text()
        named += [f"{_relative(path)}: {line.strip()}" for line in text.splitlines() if _CONSTANT.search(line)]
        if path == HOME:
            continue
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and node.value == "postgres":
                where = (_relative(path), lines[node.lineno - 1].strip())
                if where in KEPT:
                    kept.add(where)
                else:
                    named.append(f"{where[0]}: {where[1]}")
    assert named == []
    # Each kept place is still there, so the scan above found the literals it was looking for.
    assert kept == KEPT


def test_no_module_reads_the_dialect_as_it_is_imported():
    # The scan finds a call wherever one runs at import: a module's statement, a default value, a class's statement.
    planted = "X = superset_dialect()\ndef f(d=superset_dialect()):\n    return superset_dialect()\n" \
              "class A:\n    y = settings.superset_dialect()\n    def g(self):\n        return (lambda: superset_dialect())()\n"
    assert _calls(ast.parse(planted)) == ([1, 2, 5], [3, 7])
    at_import, reading = [], set()
    for path in MODULES:
        early, late = _calls(ast.parse(path.read_text()))
        at_import += [f"{_relative(path)}:{line}" for line in early]
        if late:
            reading.add(_relative(path))
    assert at_import == []
    assert READERS <= reading
