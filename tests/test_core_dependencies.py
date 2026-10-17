"""Every third-party module the package imports when a module loads is one its install declares.

`mnemiq.llm`'s client, embedder and window check import `httpx` at module level, and `httpx` was declared only in
the `dev` extra. `openai` brought it in until openai 3 dropped it; from then a plain install of mnemiq could not
import its own model client. Nothing in the suite could see it, because the dev venv installs `httpx` for the test
client, so this reads the source and pyproject.toml instead of importing anything.

An import inside a function, a `try`, or an `if TYPE_CHECKING:` block is lazy or guarded by design, and the extras
that serve one are their own business. A module that only an extra serves is named in EXTRA_MODULES.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "mnemiq"
# A module that only an extra's install loads, and the extra that must declare what it imports at load time.
EXTRA_MODULES = {"server/app.py": "server"}


def _name(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", re.split(r"[\[<>=!~ ;]", requirement, maxsplit=1)[0]).lower()


def _load_time_imports(path: Path) -> set[str]:
    """Top-level names of third-party modules imported by a statement directly in the module body."""
    tree = ast.parse(path.read_text())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return {r for r in roots if r not in sys.stdlib_module_names and r not in ("mnemiq", "__future__")}


def test_every_load_time_import_is_a_declared_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    core = {_name(r) for r in project["dependencies"]}
    extras = {extra: {_name(r) for r in reqs} for extra, reqs in project["optional-dependencies"].items()}
    distributions = packages_distributions()

    undeclared = []
    for path in sorted(PACKAGE.rglob("*.py")):
        module = path.relative_to(PACKAGE).as_posix()
        declared = core | extras.get(EXTRA_MODULES.get(module, ""), set())
        for root in sorted(_load_time_imports(path)):
            if not {_name(d) for d in distributions.get(root, [root])} & declared:
                undeclared.append(f"{module} imports {root}")
    assert not undeclared, undeclared


def test_the_rule_sees_a_load_time_import(tmp_path):
    """The reader finds an import in the module body and leaves lazy and guarded ones alone."""
    source = tmp_path / "m.py"
    source.write_text("import httpx\nfrom sqlglot import exp\nimport os\n"
                      "try:\n    import oracledb\nexcept ImportError:\n    oracledb = None\n"
                      "def f():\n    import rdflib\n")
    assert _load_time_imports(source) == {"httpx", "sqlglot"}
