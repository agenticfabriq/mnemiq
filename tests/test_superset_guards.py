"""Static guards over the engine's own source for Superset mode.

The ask path holds only the asking person's delegated token, and one factory, `superset_executor`, turns it into the
adapter a request's statements run through. A second place that builds a Superset client or adapter, or wraps a raw
credential as a delegated token, would be a second way for a request to reach Superset, and nothing at runtime would
notice it carrying the wrong token. So every module under `src/mnemiq` is parsed, and each reference to one of these
classes outside an import or a type annotation must be one the allow-list names.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import mnemiq

SOURCE = Path(mnemiq.__file__).parent

# The classes whose every use is accounted for: the Superset adapter and client, and the two kinds of credential.
WATCHED = frozenset({"SupersetSqlAdapter", "SupersetMcp", "DelegatedToken", "ServiceKey"})

DEFINES = "defines"
USES = "uses"

# Each module, by its path under `src/mnemiq`, with the references to watched classes it may hold. Any module not
# named here may hold none, so a module added later is covered without editing this.
ALLOWED: dict[str, frozenset[tuple[str, str]]] = {
    # The client defines the delegated token and the client, and builds neither.
    "superset/client.py": frozenset({("DelegatedToken", DEFINES), ("SupersetMcp", DEFINES)}),
    # The adapter defines itself, and builds nothing watched.
    "superset/adapter.py": frozenset({("SupersetSqlAdapter", DEFINES)}),
    # The one constructor on the ask path: a client carrying the request's delegated token, and the adapter over it.
    "superset/executor.py": frozenset({("SupersetSqlAdapter", USES), ("SupersetMcp", USES), ("DelegatedToken", USES)}),
    # Enrichment's own adapter, built offline from the service key, which only this module defines.
    "superset/service.py": frozenset({("SupersetSqlAdapter", USES), ("SupersetMcp", USES), ("ServiceKey", DEFINES),
                                      ("ServiceKey", USES)}),
    # The door that verified a Bearer, wrapping it as the request's delegated token.
    "auth/doors.py": frozenset({("DelegatedToken", USES)}),
}


@dataclass(frozen=True)
class Reference:
    """A reference to the watched class `name`: its definition, or any other use of it."""

    name: str
    kind: str
    line: int


def references(source: str) -> list[Reference]:
    """Every reference to a watched class in `source`, outside import statements and type annotations.

    A reference is a name or an attribute that is the class, however it is used: called, assigned, passed as an
    argument, subclassed, compared with. A name an import binds to a watched class under another name (`import ...
    as`) is followed, so the class is found by that name too. A class or function definition under a watched name is a
    definition."""
    tree = ast.parse(source)
    names = {name: name for name in WATCHED}
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported = alias.name.rsplit(".", 1)[-1]
                if imported in WATCHED and alias.asname:
                    names[alias.asname] = imported
        for annotation in _annotations(node):
            exempt.update(id(inner) for inner in ast.walk(annotation))
    found: list[Reference] = []
    for node in ast.walk(tree):
        if id(node) in exempt:
            continue
        if isinstance(node, ast.Name) and node.id in names:
            found.append(Reference(names[node.id], USES, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in WATCHED:
            found.append(Reference(node.attr, USES, node.lineno))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found.append(Reference(names[node.name], DEFINES, node.lineno))
    return found


def _annotations(node: ast.AST):
    """The type annotations `node` holds itself: a function's parameter and return annotations, and an annotated
    assignment's annotation (never its value)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = node.args
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg,
                         arguments.kwarg):
            if argument is not None and argument.annotation is not None:
                yield argument.annotation
        if node.returns is not None:
            yield node.returns
    elif isinstance(node, ast.AnnAssign):
        yield node.annotation


def violations(module: str, source: str) -> list[str]:
    """The references in `source`, the module at `module` under `src/mnemiq`, that the allow-list does not name."""
    allowed = ALLOWED.get(module, frozenset())
    return [f"{module} line {ref.line}: {ref.kind} {ref.name}" for ref in references(source)
            if (ref.name, ref.kind) not in allowed]


def _modules() -> dict[str, str]:
    return {path.relative_to(SOURCE).as_posix(): path.read_text() for path in sorted(SOURCE.rglob("*.py"))}


def test_one_constructor_on_the_ask_path():
    """Every module under `src/mnemiq` refers to the Superset adapter, the Superset client, a delegated token and the
    service key only where the allow-list says: outside the client and the adapter, which define them, only
    `superset_executor` builds an adapter and its client from a delegated token, only the door that verified a Bearer
    makes a delegated token, and only enrichment's own module uses the service key.

    Outside its reach, since no reading of the source can see them: a class reached by `getattr` on a string or
    through a module's `__dict__`, by `type(obj)(...)` or `obj.__class__`, or by copying an object already built."""
    modules = _modules()
    assert "superset/executor.py" in modules and "superset/client.py" in modules  # the tree read is this one
    found = [problem for module, source in modules.items() for problem in violations(module, source)]
    assert found == []


def test_the_checker_finds_the_constructor_it_allows():
    """A positive control on the real tree: the only module that uses the adapter class in code is the executor (and
    enrichment's own module, once it exists), so the scan reads code, not only imports and annotations."""
    using = {module for module, source in _modules().items()
             if ("SupersetSqlAdapter", USES) in {(ref.name, ref.kind) for ref in references(source)}}
    assert "superset/executor.py" in using
    assert using <= {"superset/executor.py", "superset/service.py"}


@pytest.mark.parametrize("module, source, reaches", [
    ("server/app.py", "from mnemiq.superset.adapter import SupersetSqlAdapter\n"
                      "adapter = SupersetSqlAdapter(client, database_id=1, schema='public')\n", "SupersetSqlAdapter"),
    ("server/app.py", "from mnemiq.superset.client import SupersetMcp\nmake = SupersetMcp\n", "SupersetMcp"),
    ("runtime.py", "import functools\nfrom mnemiq.superset.adapter import SupersetSqlAdapter\n"
                   "make = functools.partial(SupersetSqlAdapter, database_id=1)\n", "SupersetSqlAdapter"),
    ("runtime.py", "from mnemiq.superset.client import SupersetMcp as M\nclient = M(url, token)\n", "SupersetMcp"),
    ("runtime.py", "from mnemiq.superset import client\nmade = client.SupersetMcp(url, token)\n", "SupersetMcp"),
    ("server/app.py", "from mnemiq.superset.client import DelegatedToken as Token\ntoken = Token(raw)\n",
     "DelegatedToken"),
    ("server/app.py", "from mnemiq.superset.adapter import SupersetSqlAdapter\n"
                      "class Mine(SupersetSqlAdapter):\n    pass\n", "SupersetSqlAdapter"),
    ("server/app.py", "from mnemiq.superset.client import SupersetMcp\n"
                      "def build(make=SupersetMcp) -> None:\n    pass\n", "SupersetMcp"),
    ("server/app.py", "from dataclasses import dataclass, field\nfrom mnemiq.superset.client import SupersetMcp\n"
                      "@dataclass\nclass Held:\n    client: SupersetMcp = field(default_factory=SupersetMcp)\n",
     "SupersetMcp"),
    ("server/app.py", "class ServiceKey:\n    pass\n", "ServiceKey"),
    # The modules allowed to define a class may not build it, nor the executor define one.
    ("superset/client.py", "class SupersetMcp:\n    pass\n\nmade = SupersetMcp(url, token)\n", "SupersetMcp"),
    ("superset/executor.py", "class DelegatedToken:\n    pass\n", "DelegatedToken"),
], ids=["call", "alias-assignment", "partial", "aliased-import-call", "attribute-call", "aliased-token", "subclass",
        "default-argument", "field-default", "definition-elsewhere", "client-builds-itself", "executor-defines"])
def test_the_checker_flags_each_planted_reference(module, source, reaches):
    found = violations(module, source)
    assert found and all(problem.endswith(reaches) for problem in found), found


def test_the_checker_passes_imports_and_annotations():
    source = ("from __future__ import annotations\n"
              "from mnemiq.superset.adapter import SupersetSqlAdapter\n"
              "from mnemiq.superset.client import DelegatedToken, SupersetMcp as Client\n"
              "import mnemiq.superset.client\n"
              "def run(adapter: SupersetSqlAdapter, *more: Client, token: DelegatedToken | None = None,\n"
              "        **rest: mnemiq.superset.client.SupersetMcp) -> SupersetSqlAdapter:\n"
              "    held: Client\n"
              "    return adapter\n")
    assert references(source) == []
    assert violations("server/app.py", source) == []
