"""**M107.** Loading a DuckDB extension without reaching the network when nothing may.

`INSTALL x` fetches `x` from extensions.duckdb.org when it is not already in the extension
directory, and DuckDB also installs a known extension on demand by itself
(`autoinstall_known_extensions`, on by default). So under MNEMIQ_LOCAL_ONLY the store's
`INSTALL vss; INSTALL fts` -- and an adapter's `INSTALL postgres` -- reached the internet on a
fresh machine, after `assert_local_only` had printed that the deployment was verified local.

Under local-only an extension is LOADED only if it is already installed (or built in), and
refused otherwise, naming what to pre-seed; DuckDB's own auto-install is switched off on the
connection. Everywhere else it is `INSTALL` then `LOAD`, as before. How a local-only machine gets
its extensions -- bundled, or pre-seeded by the operator -- is a deployment decision this does not
make; it only makes sure nothing is fetched behind the operator's back.
"""

from __future__ import annotations

import duckdb


class ExtensionNotInstalled(RuntimeError):
    """MNEMIQ_LOCAL_ONLY is set and loading this extension would download it."""


def load_extension(con: duckdb.DuckDBPyConnection, name: str, *, local_only: bool) -> None:
    if not local_only:
        con.execute(f"INSTALL {name}; LOAD {name};")
        return
    con.execute("SET autoinstall_known_extensions = false")
    # By name or alias: an adapter asks for `postgres`, which DuckDB lists as `postgres_scanner`.
    row = con.execute(
        "SELECT extension_name, installed FROM duckdb_extensions() "
        "WHERE extension_name = ? OR list_contains(aliases, ?)",
        [name, name],
    ).fetchone()
    if not (row and row[1]):
        # Where DuckDB looks: a versioned, per-platform directory under the extension directory,
        # holding the file under the extension's own name -- `postgres_scanner`, not `postgres`.
        root = con.execute("SELECT current_setting('extension_directory')").fetchone()[0]
        platform = con.execute("PRAGMA platform").fetchone()[0]
        version = f"v{duckdb.__version__}"
        file = f"{row[0] if row else name}.duckdb_extension"
        raise ExtensionNotInstalled(
            f"MNEMIQ_LOCAL_ONLY is set and the DuckDB extension {name!r} is not installed; "
            f"installing it would download it from extensions.duckdb.org. Pre-seed it: on a "
            f"connected {platform} machine with DuckDB {duckdb.__version__}, run `INSTALL {name}`, "
            f"then copy ~/.duckdb/extensions/{version}/{platform}/{file} from it into "
            f"{root or '~/.duckdb/extensions'}/{version}/{platform}/ on this one."
        )
    con.execute(f"LOAD {name}")
