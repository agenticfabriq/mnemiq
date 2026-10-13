"""**M107.** Loading a DuckDB extension without reaching the network when nothing may.

`INSTALL x` fetches `x` from extensions.duckdb.org when it is not already in the extension
directory, and DuckDB also installs a known extension on demand by itself
(`autoinstall_known_extensions`, on by default). So under MNEMIQ_LOCAL_ONLY the store's
`INSTALL vss; INSTALL fts` -- and an adapter's `INSTALL postgres` -- reached the internet on a
fresh machine, after `assert_local_only` had printed that the deployment was verified local.

Under local-only an extension is LOADED only if it is already installed (or built in), and
refused otherwise, naming what to pre-seed; DuckDB's own auto-install is switched off on the
connection. Everywhere else it is `INSTALL` then `LOAD`, as before.

How a local-only machine gets its extensions is the owner's call, 2026-10-12: the operator
pre-seeds them, not mnemiq bundling them -- the four come to about 89 MB per platform and DuckDB
version (measured, osx_arm64, DuckDB 1.5.4). `fetch_extensions` (`mnemiq extensions fetch`) is
the one command that does it on a connected machine, and `check_extensions` is the boot check
that loads each one this deployment needs before anything runs.
"""

from __future__ import annotations

import gzip
import urllib.request
import zlib
from collections.abc import Callable, Iterable
from pathlib import Path

import duckdb

# What every deployment loads: the store's vector and full-text indexes.
STORE_EXTENSIONS = ("vss", "fts")
# The scanner each source kind attaches through, by the name an adapter asks for -- DuckDB files
# `postgres` as `postgres_scanner`. `adapters.resolve` and `adapters.federated` read it from here.
SOURCE_EXTENSIONS = {"postgres": "postgres", "sqlite": "sqlite"}
# Everything mnemiq can load: what `mnemiq extensions fetch` downloads when given no names.
ALL_EXTENSIONS = (*STORE_EXTENSIONS, *SOURCE_EXTENSIONS.values())
# Where DuckDB's own `INSTALL` downloads from. It answers Python's default User-Agent,
# `Python-urllib/3.x`, with 403 and other agents with 200 (measured 2026-10-12), so the fetch names
# itself.
_REPOSITORY = "https://extensions.duckdb.org"
_USER_AGENT = f"mnemiq-extensions-fetch (DuckDB {duckdb.__version__})"


def connect_duckdb(database: str = ":memory:", *, local_only: bool) -> duckdb.DuckDBPyConnection:
    """Every DuckDB connection mnemiq opens. **Codex's review of M107**: auto-install was switched
    off only inside `load_extension`, so a connection that loads no extension -- a native DuckDB
    source -- kept it on, and model-written SQL needing a known extension (`read_csv` on a URL needs
    `httpfs`) installed it by download. Under local-only it is off in the connection's config, so
    it holds before any ATTACH or query. `local_only` has no default: a caller decides.

    Auto-LOAD is off too (review gate): where `httpfs` was already installed, the same `read_csv`
    loaded it and fetched the URL -- measured, the request reached the server -- which is network
    egress from model-written SQL, and a query can carry data out in a URL. What stays usable is
    what is built in (`core_functions`, `icu`, `json`, `parquet`) and what is loaded by name
    through `load_extension`. The cost: an auto-loadable extension that never touches the network
    is no longer loaded on demand either -- measured with `sqlite_scan`, which auto-loads
    `sqlite_scanner` on an ordinary connection and is refused on a local-only one -- so a
    local-only query needing such an extension fails until it is loaded by name. Telling the two
    kinds apart per extension is not something DuckDB offers, and the network ones are the reason
    for local-only."""
    config = ({"autoinstall_known_extensions": False, "autoload_known_extensions": False}
              if local_only else {})
    return duckdb.connect(database, config=config)


class ExtensionNotInstalled(RuntimeError):
    """MNEMIQ_LOCAL_ONLY is set and loading this extension would download it."""


class ExtensionsNotReady(RuntimeError):
    """MNEMIQ_LOCAL_ONLY is set and an extension this deployment loads is missing or will not load."""


def _extension_row(con: duckdb.DuckDBPyConnection, name: str) -> tuple[str, bool] | None:
    # By name or alias: an adapter asks for `postgres`, which DuckDB lists as `postgres_scanner`.
    return con.execute(
        "SELECT extension_name, installed FROM duckdb_extensions() "
        "WHERE extension_name = ? OR list_contains(aliases, ?)",
        [name, name],
    ).fetchone()


def _platform(con: duckdb.DuckDBPyConnection) -> str:
    return con.execute("PRAGMA platform").fetchone()[0]


def _preseed(con: duckdb.DuckDBPyConnection, names: Iterable[str]) -> str:
    """How to pre-seed `names` here: where DuckDB looks -- a versioned, per-platform directory under
    the extension directory -- and the command that fetches them on a connected machine."""
    root = con.execute("SELECT current_setting('extension_directory')").fetchone()[0]
    platform, version = _platform(con), f"v{duckdb.__version__}"
    return (f"On a connected machine, run `mnemiq extensions fetch --platform {platform} --out "
            f"<dir> {' '.join(names)}` (the same mnemiq release, so DuckDB {duckdb.__version__}), "
            f"then copy <dir>/{version}/{platform}/ into "
            f"{root or '~/.duckdb/extensions'}/{version}/{platform}/ on this one.")


def load_extension(con: duckdb.DuckDBPyConnection, name: str, *, local_only: bool) -> None:
    if not local_only:
        con.execute(f"INSTALL {name}; LOAD {name};")
        return
    con.execute("SET autoinstall_known_extensions = false")
    row = _extension_row(con, name)
    if not (row and row[1]):
        # The file carries the extension's own name -- `postgres_scanner`, not `postgres`.
        file = f"{row[0] if row else name}.duckdb_extension"
        raise ExtensionNotInstalled(
            f"MNEMIQ_LOCAL_ONLY is set and the DuckDB extension {name!r} is not installed "
            f"({file}); installing it would download it from extensions.duckdb.org. Pre-seed it: "
            + _preseed(con, [name])
        )
    con.execute(f"LOAD {name}")


def check_extensions(names: Iterable[str]) -> list[str]:
    """The boot check under MNEMIQ_LOCAL_ONLY: load each of `names` on a throwaway local-only
    connection, so a deployment missing one refuses to start rather than mid-build (M107). Returns
    the extensions loaded, by DuckDB's names; raises `ExtensionsNotReady` naming every one that is
    missing or will not load.

    Loaded, not only looked up: a file of the right name that is not an extension, or is for
    another platform or DuckDB version, reads as installed and fails only at LOAD -- measured, a
    stray file reads `installed` and LOAD refuses it. Loading all four took 142 ms."""
    names = list(dict.fromkeys(names))
    con = connect_duckdb(local_only=True)
    try:
        loaded, problems = [], []
        for name in names:
            row = _extension_row(con, name)
            label = row[0] if row else name
            try:
                load_extension(con, name, local_only=True)
            except ExtensionNotInstalled:
                problems.append(f"{label}: not installed")
            except duckdb.Error as exc:
                problems.append(f"{label}: installed but does not load -- {str(exc).splitlines()[0]}")
            else:
                loaded.append(label)
        if problems:
            raise ExtensionsNotReady(
                "MNEMIQ_LOCAL_ONLY is set and DuckDB extensions this deployment loads are not "
                "ready; mnemiq never downloads one under local-only:\n  " + "\n  ".join(problems)
                + "\n\n" + _preseed(con, names)
            )
        return loaded
    finally:
        con.close()


def fetch_extensions(out: Path, names: Iterable[str] = ALL_EXTENSIONS, *,
                     platform: str | None = None,
                     urlopen: Callable = urllib.request.urlopen) -> list[Path]:
    """`mnemiq extensions fetch`: download `names` for `platform` -- this machine's by default --
    and this mnemiq's DuckDB, into `out/v<version>/<platform>/`, the layout DuckDB reads, to copy
    to a local-only machine. From the repository DuckDB's own `INSTALL` uses. The files are
    signed and DuckDB checks the signature at LOAD, so one altered in transit is refused where it
    is used; where `platform` is this machine's, each is loaded from `out` here as well, so a bad
    download is found before it is carried across."""
    con = duckdb.connect(config={"autoinstall_known_extensions": False,
                                 "autoload_known_extensions": False})
    try:
        here, version = _platform(con), f"v{duckdb.__version__}"
        files = []
        for name in dict.fromkeys(names):
            row = _extension_row(con, name)
            if row is None:
                raise ValueError(f"DuckDB {duckdb.__version__} knows no extension {name!r}")
            files.append(row[0])
    finally:
        con.close()
    platform = platform or here
    target = Path(out) / version / platform
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for file in files:
        url = f"{_REPOSITORY}/{version}/{platform}/{file}.duckdb_extension.gz"
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urlopen(request, timeout=60) as response:
                data = gzip.decompress(response.read())
        except (OSError, EOFError, zlib.error) as exc:  # the fetch, or a truncated/bad gzip
            raise RuntimeError(f"could not fetch {url}: {exc}") from exc
        path = target / f"{file}.duckdb_extension"
        part = path.with_name(path.name + ".part")
        part.write_bytes(data)
        part.replace(path)
        written.append(path)
    if platform == here:
        check = duckdb.connect(config={"autoinstall_known_extensions": False,
                                       "autoload_known_extensions": False,
                                       "extension_directory": str(out)})
        try:
            for file in files:
                check.execute(f"LOAD {file}")
        finally:
            check.close()
    return written
