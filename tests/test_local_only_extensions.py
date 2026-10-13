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
        tree = ast.parse(path.read_text())
        # The names `duckdb` goes by here (`import duckdb as ddb`), and any `connect` imported from
        # it (`from duckdb import connect`) -- a raw connection by another spelling (review gate).
        modules, bare = {"duckdb"}, set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.asname or a.name for a in node.names if a.name == "duckdb"}
            elif isinstance(node, ast.ImportFrom) and node.module == "duckdb":
                bare |= {a.asname or a.name for a in node.names if a.name == "connect"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (name := door(node)) is not None:
                passed = [k.value for k in node.keywords if k.arg == "local_only"]
                # From the caller, never a constant: `local_only=False` written at a door is the
                # setting dropped with the keyword still present. `True` only tightens, and the
                # boot check, which exists only under local-only, writes it.
                if not passed or (isinstance(passed[0], ast.Constant)
                                  and passed[0].value is not True):
                    missing.append(f"{path.relative_to(root)}:{node.lineno} {name}")
            elif path.name != "duckdb_extensions.py" and (
                    (isinstance(func, ast.Attribute) and func.attr == "connect"
                     and isinstance(func.value, ast.Name) and func.value.id in modules)
                    or (isinstance(func, ast.Name) and func.id in bare)):
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


@pytest.mark.skipif(not _installed_here("sqlite"),
                    reason="needs sqlite_scanner installed in this machine's extension directory")
def test_a_local_only_connection_auto_loads_nothing(tmp_path):
    """An installed extension is not auto-loaded under local-only: where `httpfs` is installed, an
    auto-load is a URL fetched from model-written SQL (measured). `sqlite_scanner` stands in for it
    here, with no network: `sqlite_scan` auto-loads it unless auto-load is off."""
    import sqlite3

    from mnemiq.duckdb_extensions import connect_duckdb

    db = tmp_path / "x.sqlite"
    sqlite3.connect(db).execute("CREATE TABLE t (x INTEGER)").connection.commit()
    scan = f"SELECT count(*) FROM sqlite_scan('{db}', 't')"
    assert connect_duckdb(local_only=False).execute(scan).fetchone() == (0,), "control: it loads"
    con = connect_duckdb(local_only=True)
    assert con.execute("SELECT current_setting('autoload_known_extensions')").fetchone()[0] is False
    with pytest.raises(duckdb.Error):
        con.execute(scan).fetchall()


# **M107, the owner's call (2026-10-12): the operator pre-seeds.** Under local-only every command
# loads the extensions the deployment needs at startup -- before the verified line -- and
# `mnemiq extensions fetch` is how a connected machine prepares them.

def _local_only(tmp_path, kind: str | None = None):
    from mnemiq.config import Settings

    fields = {"local_only": True, "llm_base_url": None, "embed_base_url": None}
    if kind:
        manifest = tmp_path / "sources.json"
        manifest.write_text(f'[{{"id": "s", "kind": "{kind}", "target": "t", "catalog": "s", '
                            f'"schema": "main"}}]')
        fields["sources_path"] = str(manifest)
    return Settings(**fields)


def test_local_only_refuses_at_startup_naming_every_extension_it_would_need(fresh_extensions,
                                                                            tmp_path, capsys):
    from mnemiq.duckdb_extensions import ExtensionsNotReady

    with pytest.raises(ExtensionsNotReady) as refused:
        _local_only(tmp_path, "postgres").assert_local_only()
    message = str(refused.value)
    for missing in ("vss: not installed", "fts: not installed", "postgres_scanner: not installed"):
        assert missing in message, message
    assert "sqlite_scanner" not in message, "a deployment with no SQLite source needs no scanner"
    platform = duckdb.connect().execute("PRAGMA platform").fetchone()[0]
    assert f"mnemiq extensions fetch --platform {platform} --out <dir>" in message
    assert "verified" not in capsys.readouterr().err, "refused before the verified line"
    assert _files(fresh_extensions) == [], "nothing was fetched"


def test_a_file_that_is_not_an_extension_is_refused_at_startup(fresh_extensions, tmp_path):
    from mnemiq.duckdb_extensions import ExtensionsNotReady

    platform = duckdb.connect().execute("PRAGMA platform").fetchone()[0]
    seeded = fresh_extensions / f"v{duckdb.__version__}" / platform
    seeded.mkdir(parents=True)
    (seeded / "vss.duckdb_extension").write_bytes(b"not an extension")  # reads as installed
    with pytest.raises(ExtensionsNotReady, match="vss: installed but does not load"):
        _local_only(tmp_path).assert_local_only()


def test_a_pre_seeded_deployment_starts_and_says_what_it_loaded(pre_seeded, tmp_path, capsys):
    _local_only(tmp_path, "sqlite").assert_local_only()
    line = capsys.readouterr().err
    assert "MNEMIQ_LOCAL_ONLY verified" in line
    assert "DuckDB extensions loaded, none downloaded: fts, sqlite_scanner, vss" in line
    assert _files(pre_seeded) == sorted(_files(pre_seeded)), "nothing beyond what was pre-seeded"



@pytest.mark.parametrize("manifest", [
    None,                      # no file: OSError
    "not json",                # ValueError
    '{"id": "s"}',             # an object, not a list: TypeError
    '["s"]',                   # a list of strings: TypeError
    '[{"kind": "sqlite"}]',    # a source with no id: KeyError
])
def test_a_manifest_that_cannot_be_read_still_has_the_store_extensions_checked(
        fresh_extensions, tmp_path, manifest):
    """Its own error still comes from `source_specs`, where the command meets it; the boot check
    reads no kinds from it, and still checks the store's two -- it does not escape as a traceback."""
    from mnemiq.config import Settings
    from mnemiq.duckdb_extensions import ExtensionsNotReady

    path = tmp_path / "sources.json"
    if manifest is not None:
        path.write_text(manifest)
    settings = Settings(local_only=True, llm_base_url=None, embed_base_url=None,
                        sources_path=str(path))
    with pytest.raises((OSError, ValueError, TypeError, KeyError)):
        settings.source_specs()
    with pytest.raises(ExtensionsNotReady) as refused:
        settings.assert_local_only()
    assert "vss: not installed" in str(refused.value) and "fts: not installed" in str(refused.value)

def _gzip_server(bodies: dict[str, bytes], asked: list[str]):
    """An `urlopen` that serves `bodies` by file name, gzipped, and records every URL asked. The
    repository refuses Python's default User-Agent with 403 (measured), so a request without its
    own is refused here too."""
    import gzip
    import io

    def urlopen(request, timeout=None):
        agent = request.get_header("User-agent") or ""
        assert agent and not agent.startswith("Python-urllib"), "the repository answers 403"
        asked.append(request.full_url)
        name = request.full_url.rsplit("/", 1)[1].removesuffix(".gz")
        return io.BytesIO(gzip.compress(bodies[name]))

    return urlopen


def test_fetch_writes_the_layout_duckdb_reads_from_where_duckdb_installs(tmp_path):
    from mnemiq.duckdb_extensions import fetch_extensions

    here = duckdb.connect().execute("PRAGMA platform").fetchone()[0]
    other = next(p for p in ("windows_amd64", "linux_arm64") if p != here)  # nothing to load here
    files = ["vss", "fts", "postgres_scanner", "sqlite_scanner"]
    bodies = {f"{f}.duckdb_extension": f"bytes of {f}".encode() for f in files}
    asked: list[str] = []
    written = fetch_extensions(tmp_path, platform=other, urlopen=_gzip_server(bodies, asked))
    version = f"v{duckdb.__version__}"
    assert asked == [f"https://extensions.duckdb.org/{version}/{other}/{f}.duckdb_extension.gz"
                     for f in files]
    assert written == [tmp_path / version / other / f"{f}.duckdb_extension" for f in files]
    assert [p.read_bytes() for p in written] == [bodies[p.name] for p in written]


def test_fetch_for_this_platform_loads_what_it_wrote(tmp_path):
    from mnemiq.duckdb_extensions import fetch_extensions

    asked: list[str] = []
    bodies = {"fts.duckdb_extension": b"x" * 600, "vss.duckdb_extension": b"y" * 600}
    with pytest.raises(RuntimeError, match="does not load, so removed") as refused:
        fetch_extensions(tmp_path, ["fts", "vss"], urlopen=_gzip_server(bodies, asked))
    assert "fts.duckdb_extension" in str(refused.value) and "vss.duckdb_extension" in str(refused.value)
    assert len(asked) == 2
    assert _files(tmp_path) == [], "none left to be copied across"


def test_fetch_from_a_real_extension_loads(tmp_path):
    from mnemiq.duckdb_extensions import fetch_extensions

    path = _default_install_path("fts")
    if path is None:
        pytest.skip("needs fts installed in this machine's extension directory")
    written = fetch_extensions(tmp_path, ["fts"],
                               urlopen=_gzip_server({path.name: path.read_bytes()}, []))
    assert written[0].read_bytes() == path.read_bytes()


def test_fetch_refuses_under_local_only(monkeypatch, tmp_path):
    import mnemiq.duckdb_extensions as extensions
    from mnemiq.cli import main

    def fetched(*args, **kwargs):
        raise AssertionError("fetched under MNEMIQ_LOCAL_ONLY")

    monkeypatch.setattr(extensions, "fetch_extensions", fetched)
    monkeypatch.setenv("MNEMIQ_LOCAL_ONLY", "1")
    assert main(["extensions", "fetch", "--out", str(tmp_path)]) == 2
