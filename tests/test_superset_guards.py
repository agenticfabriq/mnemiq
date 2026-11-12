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


# --- the service module, imported by the CLI alone -------------------------------------------------------------------

SERVICE = "mnemiq.superset.service"
SUPERSET = "mnemiq.superset"
# The modules that may reach the service module, by path under `src/mnemiq`. The module itself is not counted.
SERVICE_IMPORTERS = frozenset({"cli.py"})
SERVICE_MODULE = "superset/service.py"
# The calls that import a module named by a string.
_IMPORT_CALLS = frozenset({"import_module", "__import__"})


def _package(module: str) -> str:
    """The package a relative import in the module at `module` under `src/mnemiq` resolves against: the module's
    parent, or for a package's `__init__.py` the package itself."""
    parts = ["mnemiq", *module.removesuffix(".py").split("/")]
    return ".".join(parts[:-1])


def _absolute(name: str | None, level: int, package: str) -> str | None:
    """`name` imported `level` dots up from `package`, as Python resolves it; None past the top of `package`."""
    if level == 0:
        return name
    parts = package.split(".")
    if level > len(parts):
        return None
    base = parts[:len(parts) - (level - 1)]
    return ".".join(base + ([name] if name else []))


def _from_string(text: str, package: str) -> str | None:
    """A module name as `import_module` reads it: relative to `package` when it starts with dots."""
    level = len(text) - len(text.lstrip("."))
    return _absolute(text[level:] or None, level, package) if level else text


def _called(func: ast.expr) -> str | None:
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """The argument a call passes at `position` or as `keyword`, or None."""
    if len(call.args) > position:
        return call.args[position]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def _names_the_service(call: ast.Call, package: str) -> bool:
    """Whether a call to `import_module` or `__import__` names the service module: by its name, relative to the package
    the call gives or else to the module's own, or as `service` in a `fromlist` beside the Superset package.

    `__import__` takes a relative name without its dots, and how many levels up it reaches as its fifth argument,
    `level`; its name is read from the module's own package at that level, as `globals()` gives it."""
    strings = [node.value for node in ast.walk(call) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    given = next((kw.value.value for kw in call.keywords if kw.arg == "package" and isinstance(kw.value, ast.Constant)),
                 None)
    if given is None and _called(call.func) == "import_module" and len(call.args) > 1:
        given = getattr(call.args[1], "value", None)
    anchor = given if isinstance(given, str) else package
    resolved = {_from_string(text, anchor) for text in strings}
    if _called(call.func) == "__import__":
        name, level = _argument(call, 0, "name"), _argument(call, 4, "level")
        if (isinstance(level, ast.Constant) and type(level.value) is int and level.value > 0
                and isinstance(name, ast.Constant) and isinstance(name.value, str)):
            resolved.add(_absolute(name.value or None, level.value, package))
    return SERVICE in resolved or (SUPERSET in resolved and "service" in strings)


def service_reaches(module: str, source: str) -> list[int]:
    """The lines on which `source`, the module at `module` under `src/mnemiq`, reaches `mnemiq.superset.service`.

    An import of it in any form: `import mnemiq.superset.service`; `from mnemiq.superset.service import ...` and the
    relative `from .service import ...`, resolved against the module's package; `from mnemiq.superset import service`
    and `from . import service` in `superset/`, where the module imported from is the package and `service` a name in
    it; a star import from the package, which binds the module once anything has loaded it; and `import_module` or
    `__import__` given a string that names it. And a reference that needs no import: an attribute chain ending in
    `superset.service` (`mnemiq.superset.service.enrichment_adapter(...)`, or the package under a name an import gave
    it), which reaches the module once something else has loaded it, and the module's full name as a string, as a key
    into `sys.modules` would spell it."""
    package = _package(module)
    tree = ast.parse(source)
    # The names an import binds to the Superset package itself.
    packages = {"superset"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            packages |= {alias.asname for alias in node.names if alias.name == SUPERSET and alias.asname}
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node.module, node.level, package)
            packages |= {alias.asname or alias.name for alias in node.names if f"{base}.{alias.name}" == SUPERSET}
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == SERVICE or alias.name.startswith(f"{SERVICE}.") for alias in node.names):
                lines.add(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node.module, node.level, package) or ""
            if (base == SERVICE or base.startswith(f"{SERVICE}.")
                    or any(f"{base}.{alias.name}" == SERVICE or (base == SUPERSET and alias.name == "*")
                           for alias in node.names)):
                lines.add(node.lineno)
        elif isinstance(node, ast.Call) and _called(node.func) in _IMPORT_CALLS:
            if _names_the_service(node, package):
                lines.add(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "service":
            inner = node.value
            if (isinstance(inner, ast.Attribute) and inner.attr == "superset") or (
                    isinstance(inner, ast.Name) and inner.id in packages):
                lines.add(node.lineno)
        elif isinstance(node, ast.Constant) and node.value == SERVICE:
            lines.add(node.lineno)
    return sorted(lines)


def test_service_module_is_imported_only_by_the_cli():
    """`mnemiq.superset.service` holds the service key and the adapter built from it, so only the CLI's enrichment may
    reach it: a module on the ask path that imported it could put the service key where only a delegated token
    belongs. Every module under `src/mnemiq` is parsed, and only `cli.py` may reach it in any form `service_reaches`
    reads; the module itself does not count.

    Outside its reach, since no reading of the source can see them: a module name built at run time (joined, formatted
    or read from configuration) and handed to an importer, `getattr` on the package with the name as a string,
    `sys.modules` keyed by a name built at run time, `exec`, `eval` or `runpy` over text that imports it, loading the
    file by its path, an object the CLI made from the module handed on to other code, and code outside `src/mnemiq`."""
    modules = _modules()
    assert SERVICE_MODULE in modules  # the tree read is this one
    found = [f"{module} line {line}" for module, source in modules.items()
             if module not in SERVICE_IMPORTERS and module != SERVICE_MODULE
             for line in service_reaches(module, source)]
    assert found == []
    assert service_reaches("cli.py", modules["cli.py"])  # the known-good: the CLI's own import is read, and allowed


@pytest.mark.parametrize("module, source", [
    ("server/app.py", "import mnemiq.superset.service\n"),
    ("server/app.py", "import mnemiq.superset.service as enrichment\n"),
    ("server/app.py", "from mnemiq.superset.service import enrichment_adapter\n"),
    ("superset/executor.py", "from .service import ServiceKey\n"),
    ("server/app.py", "from ..superset.service import service_credential_from_env\n"),
    ("runtime.py", "from mnemiq.superset import service\n"),
    ("runtime.py", "from .superset import service as enrichment\n"),
    ("superset/executor.py", "from . import client, service\n"),
    ("superset/__init__.py", "from . import service\n"),
    ("runtime.py", "from mnemiq.superset import *\n"),
    ("server/app.py", "import importlib\nmodule = importlib.import_module('mnemiq.superset.service')\n"),
    ("server/app.py", "from importlib import import_module\nmodule = import_module('.service', 'mnemiq.superset')\n"),
    ("superset/executor.py", "from importlib import import_module\nmodule = import_module('.service', __package__)\n"),
    ("server/app.py", "module = __import__('mnemiq.superset.service')\n"),
    ("server/app.py", "package = __import__('mnemiq.superset', fromlist=['service'])\n"),
    ("superset/executor.py", "module = __import__('service', globals(), None, ['x'], 1)\n"),
    ("superset/executor.py", "module = __import__('service', globals(), level=1)\n"),
    ("superset/executor.py", "package = __import__('', globals(), None, ['service'], 1)\n"),
    ("server/app.py", "module = __import__('superset.service', globals(), None, ['x'], 2)\n"),
    ("server/app.py", "import mnemiq\nadapter = mnemiq.superset.service.enrichment_adapter(settings, key)\n"),
    ("server/app.py", "from mnemiq import superset\nkey = superset.service.service_credential_from_env()\n"),
    ("server/app.py", "import mnemiq.superset as ss\nkey = ss.service.service_credential_from_env()\n"),
    ("server/app.py", "import sys\nmodule = sys.modules['mnemiq.superset.service']\n"),
], ids=["import", "import-as", "from-import", "relative-from", "relative-two-levels", "from-package",
        "relative-from-package", "from-dot-among-others", "package-init", "star", "import-module", "import-module-relative",
        "import-module-own-package", "dunder-import", "dunder-import-fromlist", "dunder-import-relative-positional",
        "dunder-import-relative-keyword", "dunder-import-relative-fromlist", "dunder-import-two-levels", "attribute-chain", "package-attribute",
        "aliased-package-attribute", "sys-modules"])
def test_the_service_check_flags_each_planted_import(module, source):
    assert service_reaches(module, source), source


@pytest.mark.parametrize("module, source", [
    ("cli.py", "def enrich(settings):\n"
               "    from mnemiq.superset.service import enrichment_adapter, service_credential_from_env\n"
               "    return enrichment_adapter(settings, service_credential_from_env())\n"),
    ("superset/executor.py", "from . import client\n"),
    ("superset/executor.py", "from .client import DelegatedToken\nfrom .adapter import SupersetSqlAdapter\n"),
    ("server/app.py", "from mnemiq.superset import executor\nadapter = executor.superset_executor(s, t)\n"),
    ("server/app.py", "status = self.service.status\nname = 'mnemiq.superset.services'\n"),
], ids=["the-cli", "client-from-dot", "relative-siblings", "executor-from-package", "other-service"])
def test_the_service_check_passes_what_is_not_the_service_module(module, source):
    assert (service_reaches(module, source) != []) == (module in SERVICE_IMPORTERS)


# The variable naming the service key file, as a string holds it: only the service module may name it.
_SERVICE_KEY_VARIABLE = "SUPERSET_SERVICE_KEY"
# The service module's constant holding the variable's name.
_SERVICE_KEY_CONSTANT = "SERVICE_KEY_ENV"


def names_the_key_variable(source: str) -> list[int]:
    """The lines on which `source` names the service key file's variable: a string holding the variable's name,
    f-strings' text included, or the service module's constant for it, by its name, as an attribute, in an import
    under any name, or as a string naming it, as `getattr` takes it."""
    lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _SERVICE_KEY_VARIABLE in node.value or _SERVICE_KEY_CONSTANT in node.value:
                lines.add(node.lineno)
        elif ((isinstance(node, ast.Name) and node.id == _SERVICE_KEY_CONSTANT)
              or (isinstance(node, ast.Attribute) and node.attr == _SERVICE_KEY_CONSTANT)
              or (isinstance(node, (ast.Import, ast.ImportFrom))
                  and any(alias.name.rsplit(".", 1)[-1] == _SERVICE_KEY_CONSTANT for alias in node.names))):
            lines.add(node.lineno)
    return sorted(lines)


def test_only_the_service_module_names_the_key_variable():
    """The service key file is read where it is named, in the service module, and nowhere else: not a setting, not the
    server, not the CLI. Nothing else may name the variable, nor the service module's constant holding its name, so no
    module reads it through that constant either. The CLI needs neither: the refusals that name the variable are the
    service module's own.

    Outside its reach: a name assembled from pieces at run time, whether the variable's or the constant's (a
    computed key into the module's `vars()`)."""
    modules = _modules()
    found = [f"{module} line {line}" for module, source in modules.items() if module != SERVICE_MODULE
             for line in names_the_key_variable(source)]
    assert found == []
    assert names_the_key_variable(modules[SERVICE_MODULE])  # the known-good: the module that reads it is seen


@pytest.mark.parametrize("source", [
    "import os\npath = os.environ['MNEMIQ_SUPERSET_SERVICE_KEY_FILE']\n",
    "import os\npath = os.getenv(f'MNEMIQ_SUPERSET_SERVICE_KEY_FILE')\n",
    "service_key_file: str = Field(validation_alias='MNEMIQ_SUPERSET_SERVICE_KEY_FILE')\n",
    "import os\nfrom mnemiq.superset.service import SERVICE_KEY_ENV\npath = os.environ[SERVICE_KEY_ENV]\n",
    "from mnemiq.superset.service import SERVICE_KEY_ENV as NAME\n",
    "import os\nimport mnemiq.superset.service as service\npath = os.environ[service.SERVICE_KEY_ENV]\n",
    "path = os.environ[getattr(service, 'SERVICE_KEY_ENV')]\n",
    "import os\nfrom mnemiq.superset.service import *\npath = os.environ[SERVICE_KEY_ENV]\n",
], ids=["environ", "f-string", "settings-alias", "the-constant", "the-constant-renamed", "the-constant-as-attribute",
        "the-constant-by-getattr", "the-constant-by-star-import"])
def test_the_key_variable_check_flags_each_planted_read(source):
    assert names_the_key_variable(source)
