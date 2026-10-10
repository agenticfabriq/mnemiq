"""**M107.** Under MNEMIQ_LOCAL_ONLY a DuckDB extension is loaded only if it is already installed.

`INSTALL vss; INSTALL fts` in the store bootstrap -- and an adapter's `INSTALL postgres` -- fetched
from extensions.duckdb.org on a fresh machine, after `assert_local_only` had printed that the
deployment was verified local; DuckDB also auto-installs a known extension on demand. A fresh
`extension_directory` stands in for that machine here: nothing is installed in it, so anything
that would have been fetched shows up as a file written there.
"""

from pathlib import Path

import duckdb
import pytest

from mnemiq.duckdb_extensions import ExtensionNotInstalled, load_extension

# The real `connect`, taken before any fixture redirects it to an empty extension directory -- what
# this machine actually has installed is read through it.
_REAL_CONNECT = duckdb.connect


def _files(directory: Path) -> list[str]:
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())


@pytest.fixture
def fresh_extensions(tmp_path, monkeypatch):
    """Every `duckdb.connect` opens with an empty extension directory -- a machine that has never
    installed anything -- so the doors under test meet it the way a fresh air-gapped host would."""
    directory = tmp_path / "extensions"
    directory.mkdir()
    real_connect = duckdb.connect

    def connect(database=":memory:", *args, config=None, **kwargs):
        return real_connect(database, *args,
                            config={**(config or {}), "extension_directory": str(directory)},
                            **kwargs)

    monkeypatch.setattr(duckdb, "connect", connect)
    return directory


def test_an_extension_that_is_not_installed_is_refused_not_downloaded(fresh_extensions):
    con = duckdb.connect()
    with pytest.raises(ExtensionNotInstalled) as refused:
        load_extension(con, "vss", local_only=True)
    message = str(refused.value)
    assert "'vss' is not installed" in message and "Pre-seed it" in message
    # The directory DuckDB actually reads: versioned and per-platform, not the extension root.
    platform = con.execute("PRAGMA platform").fetchone()[0]
    assert f"into {fresh_extensions}/v{duckdb.__version__}/{platform}/ on this one" in message
    assert _files(fresh_extensions) == [], "nothing was fetched"


def test_duckdbs_own_auto_install_is_switched_off(fresh_extensions):
    con = duckdb.connect()
    load_extension(con, "json", local_only=True)  # built in: installed without a download
    assert con.execute("SELECT current_setting('autoinstall_known_extensions')").fetchone()[0] is False


def test_an_adapter_asks_by_alias_and_is_refused_by_it(fresh_extensions):
    # `postgres` is DuckDB's `postgres_scanner`; asked for by its alias, still not installed --
    # and the file to pre-seed carries the extension's own name, not the alias.
    with pytest.raises(ExtensionNotInstalled, match="'postgres' is not installed") as refused:
        load_extension(duckdb.connect(), "postgres", local_only=True)
    assert "postgres_scanner.duckdb_extension" in str(refused.value)


def _installed_here(name: str) -> bool:
    row = duckdb.connect().execute(
        "SELECT installed FROM duckdb_extensions() WHERE list_contains(aliases, ?)", [name]
    ).fetchone()
    return bool(row and row[0])


@pytest.mark.skipif(not _installed_here("sqlite"),
                    reason="needs sqlite_scanner installed in this machine's extension directory")
def test_an_installed_extension_asked_for_by_alias_loads_without_a_download():
    con = duckdb.connect()
    load_extension(con, "sqlite", local_only=True)
    loaded = con.execute(
        "SELECT loaded FROM duckdb_extensions() WHERE extension_name = 'sqlite_scanner'"
    ).fetchone()[0]
    assert loaded is True


# The doors, through the constructors a deployment uses.

def test_the_store_opened_local_only_refuses_instead_of_downloading(fresh_extensions, tmp_path):
    from mnemiq.store.bootstrap import init_store

    with pytest.raises(ExtensionNotInstalled, match="'vss'"):
        init_store(str(tmp_path / "s.duckdb"), local_only=True)
    assert _files(fresh_extensions) == []


def test_the_runtime_opens_its_store_local_only_when_the_setting_says_so(fresh_extensions, tmp_path):
    from mnemiq.config import Settings
    from mnemiq.runtime import build_runtime

    with pytest.raises(ExtensionNotInstalled):
        build_runtime(Settings(local_only=True, store_path=str(tmp_path / "s.duckdb")))
    assert _files(fresh_extensions) == []


def test_a_sqlite_source_attached_local_only_refuses_instead_of_downloading(fresh_extensions,
                                                                             tmp_path):
    import sqlite3

    from mnemiq.adapters.resolve import adapter_for
    from mnemiq.config import Settings, SourceSpec

    db = tmp_path / "src.sqlite"
    sqlite3.connect(db).execute("CREATE TABLE t (x INTEGER)").connection.commit()
    spec = SourceSpec(id="s", kind="sqlite", target=str(db), catalog="src", schema="main")
    with pytest.raises(ExtensionNotInstalled, match="'sqlite'"):
        adapter_for(spec, Settings(local_only=True))
    assert _files(fresh_extensions) == []


def test_a_federation_attached_local_only_refuses_instead_of_downloading(fresh_extensions,
                                                                         tmp_path):
    import sqlite3

    from mnemiq.adapters.federated import FederatedAdapter
    from mnemiq.config import SourceSpec

    specs = []
    for name in ("a", "b"):
        db = tmp_path / f"{name}.sqlite"
        sqlite3.connect(db).execute("CREATE TABLE t (x INTEGER)").connection.commit()
        specs.append(SourceSpec(id=name, kind="sqlite", target=str(db), catalog=name,
                                schema="main"))
    with pytest.raises(ExtensionNotInstalled, match="'sqlite'"):
        FederatedAdapter(specs, local_only=True)
    assert _files(fresh_extensions) == []


def test_the_adapter_factories_and_the_postgres_subclass_forward_local_only(fresh_extensions,
                                                                          tmp_path):
    # `mnemiq eval` and the scripts build `DuckDBPostgresAdapter` and `DuckDBAdapter.postgres`
    # directly, not through the resolver -- the review gate found eval downloading through them.
    from mnemiq.adapters.duckdb import DuckDBAdapter, DuckDBPostgresAdapter

    for build in (lambda: DuckDBPostgresAdapter("postgresql://h/db", local_only=True),
                  lambda: DuckDBAdapter.postgres("postgresql://h/db", local_only=True),
                  lambda: DuckDBAdapter.sqlite(str(tmp_path / "x.sqlite"), local_only=True)):
        with pytest.raises(ExtensionNotInstalled):
            build()
    assert _files(fresh_extensions) == []


def test_every_door_passes_local_only():
    """The ratchet: a store or DuckDB adapter opened without `local_only=` installs by download.

    Read from the syntax tree, so a call is a call: prose naming `DuckDBAdapter.postgres(dsn)` in a
    docstring is not one, and a call split across lines is still one."""
    import ast
    import re

    root = Path(__file__).resolve().parents[1]
    sources = [*(root / "src" / "mnemiq").rglob("*.py"), *(root / "scripts").glob("*.py")]

    def door(call: ast.Call) -> str | None:
        func = call.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            name = f"{func.value.id}.{func.attr}"
        else:
            return None
        head = name.split(".")[0]
        if head in ("init_store", "FederatedAdapter", "connect_duckdb") or re.fullmatch(
                r"DuckDB\w*Adapter", head):
            return name
        return None

    missing, raw_connects = [], []
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (name := door(node)) is not None:
                passed = [k.value for k in node.keywords if k.arg == "local_only"]
                # From the caller, never a constant: `local_only=False` written at a door is the
                # setting dropped with the keyword still present.
                if not passed or isinstance(passed[0], ast.Constant):
                    missing.append(f"{path.relative_to(root)}:{node.lineno} {name}")
            elif (isinstance(func, ast.Attribute) and func.attr == "connect"
                  and isinstance(func.value, ast.Name) and func.value.id == "duckdb"
                  and path.name != "duckdb_extensions.py"):
                raw_connects.append(f"{path.relative_to(root)}:{node.lineno}")
    assert missing == [], f"pass the caller's local_only= at: {missing}"
    # A connection opened around `connect_duckdb` keeps DuckDB's auto-install on (Codex on M107).
    assert raw_connects == [], f"open it with duckdb_extensions.connect_duckdb: {raw_connects}"
    # ...and no extension installed anywhere but the one function that knows the rule.
    raw = [str(path.relative_to(root)) for path in sources
           if path.name != "duckdb_extensions.py"
           # `{` as well as a word: the form removed from the adapters was `f"INSTALL {extension}"`.
           and re.search(r"INSTALL [\w{]", path.read_text())]
    assert raw == [], f"load the extension through duckdb_extensions.load_extension: {raw}"


def test_the_verified_line_says_extensions_are_checked_when_the_store_opens(capsys):
    """It printed "verified" and the next step downloaded; now it says what it has not checked yet,
    and where that check happens."""
    from mnemiq.config import Settings

    Settings(local_only=True, llm_base_url=None, embed_base_url=None).assert_local_only()
    line = capsys.readouterr().err
    assert "MNEMIQ_LOCAL_ONLY verified" in line
    assert "DuckDB extensions are checked when the store opens, and never downloaded" in line


# Codex's review of M107: auto-install was switched off only inside `load_extension`, so a
# connection that loads nothing -- a native DuckDB source -- kept it on, and model-written SQL that
# needs a known extension (`read_csv` on a URL needs `httpfs`) downloaded it. Off at connect now.

def _an_implicit_extension_trigger(con) -> None:
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM read_csv('https://example.invalid/data.csv')").fetchall()


def test_a_native_duckdb_source_local_only_cannot_auto_install(fresh_extensions):
    from mnemiq.adapters.duckdb import DuckDBAdapter

    adapter = DuckDBAdapter.duckdb(":memory:", read_only=False, local_only=True)
    con = adapter._con
    assert con.execute("SELECT current_setting('autoinstall_known_extensions')").fetchone()[0] is False
    _an_implicit_extension_trigger(con)
    assert _files(fresh_extensions) == [], "nothing was fetched"


def _default_install_path(name: str) -> Path | None:
    row = _REAL_CONNECT().execute(
        "SELECT install_path FROM duckdb_extensions() WHERE extension_name = ? AND installed",
        [name]).fetchone()
    return Path(row[0]) if row and row[0] and row[0] != "(BUILT-IN)" else None


_SEEDS = ("vss", "fts", "sqlite_scanner")


@pytest.fixture
def pre_seeded(fresh_extensions):
    """A local-only machine prepared the way the refusal says: the extension files copied into
    `<directory>/v<version>/<platform>/`. Skipped where this machine has none to copy."""
    import shutil

    paths = {name: _default_install_path(name) for name in _SEEDS}
    if not all(paths.values()):
        pytest.skip("needs vss, fts and sqlite_scanner installed in this machine's extension directory")
    for path in paths.values():
        target = fresh_extensions / path.parent.parent.name / path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target / path.name)
    return fresh_extensions


def test_a_pre_seeded_store_opens_local_only_and_still_cannot_auto_install(pre_seeded, tmp_path):
    from mnemiq.store.bootstrap import init_store

    before = _files(pre_seeded)
    con = init_store(str(tmp_path / "s.duckdb"), local_only=True)  # not stranded: it opens
    assert con.execute("SELECT current_setting('autoinstall_known_extensions')").fetchone()[0] is False
    _an_implicit_extension_trigger(con)
    assert _files(pre_seeded) == before, "nothing beyond what was pre-seeded"


def test_a_pre_seeded_federation_attaches_local_only_and_still_cannot_auto_install(pre_seeded,
                                                                                    tmp_path):
    import sqlite3

    from mnemiq.adapters.federated import FederatedAdapter
    from mnemiq.config import SourceSpec

    specs = []
    for name in ("a", "b"):
        db = tmp_path / f"{name}.sqlite"
        sqlite3.connect(db).execute("CREATE TABLE t (x INTEGER)").connection.commit()
        specs.append(SourceSpec(id=name, kind="sqlite", target=str(db), catalog=name,
                                schema="main"))
    before = _files(pre_seeded)
    adapter = FederatedAdapter(specs, local_only=True)
    _an_implicit_extension_trigger(adapter._con)
    assert _files(pre_seeded) == before


def test_connect_duckdb_opens_local_only_connections_with_auto_install_off():
    from mnemiq.duckdb_extensions import connect_duckdb

    def autoinstall(con):
        return con.execute("SELECT current_setting('autoinstall_known_extensions')").fetchone()[0]

    assert autoinstall(connect_duckdb(local_only=True)) is False
    assert autoinstall(connect_duckdb(local_only=False)) is True
