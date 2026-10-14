"""Codex's review of #133 (M13): the developer scripts resolve grants once per question too.

`scripts/ask.py` and `scripts/answer.py` let `retrieve` resolve the grants and then resolved them
again for the planner -- and both use FileAuthzProvider, which re-reads its policy on every call,
so a policy edit between the two scoped the cards under one version and the decision under
another, the failure M13 closed for `Runtime.ask`. Each now resolves once and hands that one name
to both. They also build the adapter that runs model SQL with the server-side bound (M152), as the
serving runtime and the eval door do.

The scripts are not importable modules, so this reads their source: a structural check, which is
what it can be for a `main()` that needs a store, a model server and a database to run.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SCRIPTS = ["scripts/ask.py", "scripts/answer.py"]
CONSUMERS = {"scripts/ask.py": "plan_query", "scripts/answer.py": "answer"}


def _calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", None)) == name]


@pytest.mark.parametrize("path", SCRIPTS)
def test_grants_are_resolved_once_and_handed_to_retrieval_and_the_planner(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    assert len(_calls(tree, "grants_for")) == 1, "one resolution per question"
    (retrieve,) = _calls(tree, "retrieve")
    passed = {k.arg: k.value for k in retrieve.keywords}
    assert isinstance(passed.get("grants"), ast.Name), "retrieve must be handed the resolved grants"
    name = passed["grants"].id
    (consumer,) = _calls(tree, CONSUMERS[path])
    assert any(isinstance(a, ast.Name) and a.id == name for a in consumer.args), \
        "the planner must be handed the same grants retrieval was"


@pytest.mark.parametrize("path", SCRIPTS)
def test_the_adapter_that_runs_model_sql_is_bounded_on_the_server(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    (adapter,) = _calls(tree, "DuckDBPostgresAdapter")
    assert "statement_timeout_s" in {k.arg for k in adapter.keywords}
