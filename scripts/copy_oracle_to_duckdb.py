"""Copy an Oracle source's tables into one DuckDB file, to enrich there and serve the store on Oracle.

    python scripts/copy_oracle_to_duckdb.py --out copy.duckdb [--sources sources.json] [--source-id ID]
        [--max-gb 10] [--replace] [--batch-rows 100000]

Reads one Oracle source from a manifest (`--sources`, default `MNEMIQ_SOURCES_PATH`; `--source-id`, default the
manifest's only Oracle source): its target, its schema (the owner) and its `"tables"` (every table the owner holds when
it lists none). Credentials are mnemiq's own (`MNEMIQ_ORACLE_USER`, `MNEMIQ_ORACLE_PASSWORD`, and for a wallet
`MNEMIQ_ORACLE_CONFIG_DIR` and `MNEMIQ_ORACLE_WALLET_PASSWORD`), and none is ever printed.

What it does, in order:

- **Disk first.** From the dictionary, `NUM_ROWS x AVG_ROW_LEN` per table, plus the LOB segments when the connected
  user owns the tables (no other user can see their size), and the tables with no statistics named. The estimate is
  printed, and past `--max-gb` (decimal GB) the copy refuses, before any row is read.
- **Refuses names DuckDB cannot keep apart.** DuckDB matches table and column names without regard to case, so two
  tables, or two columns of one table, whose names differ only in case are refused before any row is read.
- **Copies read-only.** Each table is read in its own read-only transaction (`SET TRANSACTION READ ONLY` before it,
  `ROLLBACK` after), `SELECT` only, streamed with python-oracledb's `fetch_df_batches` into the file's `main` schema
  under the exact names Oracle stores. One transaction per table, not one for the whole copy: a read-consistent view
  held for hours makes Oracle's undo reach back to the copy's start, and a long copy fails with ORA-01555.
- **Resumes.** A table lands as `<TABLE>__partial` and is renamed only once its last row is in, so a stopped copy never
  leaves a table a rerun would take for done. A table already in the file is skipped unless `--replace`.
- **Comments and keys.** `ALL_TAB_COMMENTS` and `ALL_COL_COMMENTS` become DuckDB `COMMENT ON` statements, and the keys
  `ALL_CONSTRAINTS` declares between copied tables of the owner become rows of `mnemiq.declared_keys`. The DuckDB
  adapter reads both, so a store enriched on the copy keeps them. Nothing in DuckDB enforces a key, so an orphan row
  cannot stop the copy.
- **Writes the copy's source entry** to `<out>.source.json` (`kind: duckdb`, the Oracle source's `id` and `catalog`,
  `schema: main`, the tables) and prints the steps that follow.

Types: a `NUMBER(p, s)` arrives exact, as BIGINT when `s` is 0 and `p` at most 18 and as DECIMAL(p, s) otherwise; a
`NUMBER` declared with no precision arrives as DOUBLE, the only type its values all fit, so digits past about 15
significant ones are lost; `DATE` lands as TIMESTAMP; CLOB and BLOB values as VARCHAR and BLOB (each under 1 GB: the
DataFrame fetch reads a LOB's value whole, whatever `oracledb.defaults.fetch_lobs` says, measured on 4.0.2).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase

import duckdb
import pyarrow as pa

from mnemiq.adapters.duckdb import KEYS_SCHEMA, KEYS_TABLE
from mnemiq.config import Settings, SourceSpec

# A table being copied lands under this suffix and is renamed when its last row is in.
STAGING_SUFFIX = "__partial"
REFUSED = 2

_TABLES = "SELECT table_name, num_rows, avg_row_len FROM all_tables WHERE owner = :owner ORDER BY table_name"
_COLUMNS = ("SELECT table_name, column_name FROM all_tab_columns WHERE owner = :owner "
            "ORDER BY table_name, column_id")
_TABLE_COMMENTS = "SELECT table_name, comments FROM all_tab_comments WHERE owner = :owner AND comments IS NOT NULL"
_COLUMN_COMMENTS = ("SELECT table_name, column_name, comments FROM all_col_comments "
                    "WHERE owner = :owner AND comments IS NOT NULL")
# The statement `OracleAdapter.foreign_keys` reads, kept to keys whose target is the owner's too: a key into another
# schema names a table this copy does not hold, or worse, one of the owner's own that happens to share its name.
_KEYS = ("SELECT c.table_name, cc.column_name, rc.table_name, rcc.column_name, c.constraint_name "
         "FROM all_constraints c "
         "JOIN all_cons_columns cc ON cc.owner = c.owner AND cc.constraint_name = c.constraint_name "
         "JOIN all_constraints rc ON rc.owner = c.r_owner AND rc.constraint_name = c.r_constraint_name "
         "JOIN all_cons_columns rcc ON rcc.owner = rc.owner AND rcc.constraint_name = rc.constraint_name "
         "AND rcc.position = cc.position "
         "WHERE c.owner = :owner AND c.constraint_type = 'R' AND c.r_owner = c.owner "
         "ORDER BY c.table_name, c.constraint_name, cc.position")
_USER = "SELECT user FROM dual"
# LOB segments are sized only in USER_SEGMENTS (DBA_SEGMENTS needs a grant), so only the owner can see them.
_LOBS = ("SELECT l.table_name, SUM(s.bytes) FROM user_lobs l JOIN user_segments s ON s.segment_name = l.segment_name "
         "GROUP BY l.table_name")


@dataclass(frozen=True)
class Estimate:
    """What the copy needs on disk, before compression: `total_bytes` is every table's `NUM_ROWS x AVG_ROW_LEN`
    (`per_table`, 0 for a table in `unanalyzed`, which has no statistics) plus `lob_bytes` when they could be seen."""

    total_bytes: int
    per_table: dict[str, int]
    unanalyzed: list[str]
    lob_bytes: int | None


def select_tables(owner_tables: list[str], patterns: tuple[str, ...]) -> list[str]:
    """The owner's tables a manifest's `"tables"` names, in the owner's order: every table when it names none, else
    each matching an exact name or shell pattern without regard to case, as `mnemiq.adapters.scoped` matches them."""
    if not patterns:
        return list(owner_tables)
    folded = [p.casefold() for p in patterns]
    return [t for t in owner_tables if any(fnmatchcase(t.casefold(), p) for p in folded)]


def estimate(rows: list[tuple[str, int | None, int | None]], lob_bytes: int | None) -> Estimate:
    """The copy's size from `(table, num_rows, avg_row_len)`: a table missing either is unanalyzed and adds 0."""
    per_table: dict[str, int] = {}
    unanalyzed: list[str] = []
    for table, num_rows, avg_row_len in rows:
        if num_rows is None or avg_row_len is None:
            unanalyzed.append(table)
            per_table[table] = 0
        else:
            per_table[table] = int(num_rows) * int(avg_row_len)
    return Estimate(total_bytes=sum(per_table.values()) + (lob_bytes or 0), per_table=per_table,
                    unanalyzed=unanalyzed, lob_bytes=lob_bytes)


def _quoted(name: str) -> str:
    """A quoted identifier, an embedded double quote doubled: Oracle and DuckDB both read it so."""
    return '"' + name.replace('"', '""') + '"'


def _literal(text: str) -> str:
    """A SQL string literal, an embedded single quote doubled."""
    return "'" + text.replace("'", "''") + "'"


def comment_statements(tables: dict[str, str], columns: dict[tuple[str, str], str]) -> list[str]:
    """DuckDB `COMMENT ON` statements for table and column comments, every name quoted and every text a literal: a
    comment holding a quote, or a name that needs one (`ORDER`, mixed case), reaches DuckDB as Oracle holds it."""
    statements = [f"COMMENT ON TABLE {_quoted(t)} IS {_literal(text)}" for t, text in tables.items()]
    statements += [f"COMMENT ON COLUMN {_quoted(t)}.{_quoted(c)} IS {_literal(text)}"
                   for (t, c), text in columns.items()]
    return statements


KeyRow = tuple[str, str, str, str, str]


def declared_key_rows(rows: list[KeyRow], copied: set[str]) -> list[KeyRow]:
    """The declared-key rows whose child and parent tables are both in the copy: a key to a table it lacks joins to
    nothing a store built on it can name."""
    return [row for row in rows if row[0] in copied and row[2] in copied]


def source_entry(spec: SourceSpec, out_path: str, tables: list[str]) -> dict:
    """The copy's manifest entry: the Oracle source's `id` and `catalog`, so the store built on the copy names the
    source it will serve, and the DuckDB file's `main` schema."""
    return {"id": spec.id, "kind": "duckdb", "target": out_path, "catalog": spec.catalog, "schema": "main",
            "tables": list(tables)}


def _rows(conn, sql: str, **binds) -> list[tuple]:
    with conn.cursor() as cur:
        if binds:
            cur.execute(sql, binds)
        else:
            cur.execute(sql)
        return cur.fetchall()


@contextlib.contextmanager
def _read_only(conn):
    """A read-only transaction around a block, ended by a rollback however the block ends. The rollback before it
    starts one cleanly: `SET TRANSACTION` must be a transaction's first statement."""
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
    try:
        yield
    finally:
        conn.rollback()


def _for_duckdb(table: pa.Table) -> pa.Table:
    """A fetched batch with the types DuckDB should store: an integer decimal that fits 64 bits as int64 (decimals are
    fetched so `NUMBER(p, s)` stays exact, and `NUMBER(10)` is an integer, not a DECIMAL(10,0)), and a timestamp in
    seconds or milliseconds (an Oracle `DATE`, `TIMESTAMP(3)`) in microseconds, DuckDB's own TIMESTAMP. Both widen
    without loss."""
    fields = []
    changed = False
    for field in table.schema:
        kind = field.type
        if pa.types.is_decimal(kind) and kind.scale == 0 and kind.precision <= 18:
            kind = pa.int64()
        elif pa.types.is_timestamp(kind) and kind.unit in ("s", "ms"):
            kind = pa.timestamp("us", tz=kind.tz)
        changed = changed or kind != field.type
        fields.append(field.with_type(kind))
    return table.cast(pa.schema(fields)) if changed else table


def _append(duck, table: str, data: pa.Table, *, create: bool) -> None:
    duck.register("_mnemiq_batch", data)
    try:
        if create:
            duck.execute(f"CREATE TABLE main.{_quoted(table)} AS SELECT * FROM _mnemiq_batch")
        else:
            duck.execute(f"INSERT INTO main.{_quoted(table)} SELECT * FROM _mnemiq_batch")
    finally:
        duck.unregister("_mnemiq_batch")


def copy_table(conn, duck, owner: str, table: str, batch_rows: int) -> int:
    """Copy `owner.table` into `duck`'s `main` under the same name, in one read-only transaction; the rows copied.

    The rows land in `<table>__partial`, dropped first if a stopped copy left one, and the table is renamed into place
    only after its last batch, in one DuckDB transaction with dropping the table it replaces. So the file never holds
    `<table>` with fewer rows than Oracle gave: a copy stopped mid-table leaves nothing a resume would skip as done.
    A table with no rows is created from `SELECT * ... WHERE 1 = 0`, so it keeps its columns."""
    staging = table + STAGING_SUFFIX
    duck.execute(f"DROP TABLE IF EXISTS main.{_quoted(staging)}")
    select = f"SELECT * FROM {_quoted(owner)}.{_quoted(table)}"
    rows = 0
    created = False
    try:
        with _read_only(conn):
            for batch in conn.fetch_df_batches(select, size=batch_rows, fetch_decimals=True):
                data = _for_duckdb(pa.table(batch))
                _append(duck, staging, data, create=not created)
                created = True
                rows += data.num_rows
            if not created:
                empty = conn.fetch_df_all(f"{select} WHERE 1 = 0", fetch_decimals=True)
                _append(duck, staging, _for_duckdb(pa.table(empty)), create=True)
    except BaseException:
        # Best effort: the next run drops a leftover before it starts the table, so a failure here hides nothing.
        with contextlib.suppress(Exception):
            duck.execute(f"DROP TABLE IF EXISTS main.{_quoted(staging)}")
        raise
    duck.execute("BEGIN TRANSACTION")
    try:
        duck.execute(f"DROP TABLE IF EXISTS main.{_quoted(table)}")
        duck.execute(f"ALTER TABLE main.{_quoted(staging)} RENAME TO {_quoted(table)}")
        duck.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(Exception):
            duck.execute("ROLLBACK")
        raise
    return rows


def connect(spec: SourceSpec, settings: Settings):
    """A python-oracledb connection to the source, with credentials passed as `OracleAdapter` passes them: the
    directory and wallet location only when a config directory is set, the wallet password only when given."""
    import oracledb

    extra: dict[str, str] = {}
    if settings.oracle_config_dir:
        extra["config_dir"] = settings.oracle_config_dir
        extra["wallet_location"] = settings.oracle_config_dir
    if settings.oracle_wallet_password:
        extra["wallet_password"] = settings.oracle_wallet_password
    return oracledb.connect(user=settings.oracle_user, password=settings.oracle_password, dsn=spec.target, **extra)


class _Refused(Exception):
    """A reason not to start: printed, and the script exits 2."""


def _choose_source(specs: list[SourceSpec], source_id: str | None) -> SourceSpec:
    if source_id is not None:
        found = [s for s in specs if s.id == source_id]
        if not found:
            raise _Refused(f"the manifest has no source {source_id!r}")
        if found[0].kind != "oracle":
            raise _Refused(f"source {source_id!r} is {found[0].kind!r}; this script copies an Oracle source")
        return found[0]
    oracle = [s for s in specs if s.kind == "oracle"]
    if len(oracle) != 1:
        raise _Refused(f"the manifest has {len(oracle)} Oracle sources; name the one to copy with --source-id")
    return oracle[0]


def _case_clashes(tables: list[str], columns: list[tuple[str, str]]) -> list[str]:
    """Names DuckDB would take for one: tables, or columns of one table, equal without regard to case, and a table
    whose name is another's staging name."""
    clashes: list[str] = []
    by_fold: dict[str, list[str]] = {}
    for t in tables:
        by_fold.setdefault(t.casefold(), []).append(t)
    clashes += [f"tables {', '.join(map(repr, names))}" for names in by_fold.values() if len(names) > 1]
    clashes += [f"tables {t!r} and {by_fold[(t + STAGING_SUFFIX).casefold()][0]!r} (the copy stages {t!r} under "
                f"{t + STAGING_SUFFIX!r})" for t in tables if (t + STAGING_SUFFIX).casefold() in by_fold]
    selected = set(tables)
    per_table: dict[str, dict[str, list[str]]] = {}
    for t, c in columns:
        if t in selected:
            per_table.setdefault(t, {}).setdefault(c.casefold(), []).append(c)
    for t, folds in per_table.items():
        clashes += [f"columns {', '.join(map(repr, names))} of table {t!r}" for names in folds.values()
                    if len(names) > 1]
    return clashes


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def _print_estimate(est: Estimate, tables: list[str], lob_note: str | None) -> None:
    print(f"estimate: {_gb(est.total_bytes)} for {len(tables)} table(s), from NUM_ROWS x AVG_ROW_LEN"
          + (f" plus {_gb(est.lob_bytes)} of LOB segments" if est.lob_bytes else "")
          + "; DuckDB compresses by column, so the file is usually smaller")
    if lob_note:
        print(lob_note)
    if est.unanalyzed:
        print(f"{len(est.unanalyzed)} table(s) have no statistics and add nothing to the estimate: "
              + ", ".join(est.unanalyzed))


def _duck_tables(duck) -> list[str]:
    return [r[0] for r in duck.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = current_database() AND table_schema = 'main' ORDER BY table_name").fetchall()]


def _duck_columns(duck, table: str) -> set[str]:
    return {r[0] for r in duck.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?", [table]).fetchall()}


def _write_keys(duck, rows: list[KeyRow]) -> None:
    duck.execute(f"CREATE SCHEMA IF NOT EXISTS {KEYS_SCHEMA}")
    duck.execute(f"CREATE OR REPLACE TABLE {KEYS_SCHEMA}.{KEYS_TABLE} (from_table VARCHAR, from_column VARCHAR, "
                 "to_table VARCHAR, to_column VARCHAR, constraint_name VARCHAR)")
    if rows:
        duck.executemany(f"INSERT INTO {KEYS_SCHEMA}.{KEYS_TABLE} VALUES (?, ?, ?, ?, ?)", rows)


def _copy(conn, spec: SourceSpec, args, out: str) -> int:
    owner = spec.schema.upper()
    with _read_only(conn):
        stats = _rows(conn, _TABLES, owner=owner)
        owner_tables = [r[0] for r in stats]
        tables = select_tables(owner_tables, spec.tables)
        missing = [p for p in spec.tables if not select_tables(owner_tables, (p,))]
        if missing:
            raise _Refused(f"source {spec.id!r} lists table(s) owner {owner!r} does not hold: {', '.join(missing)} "
                           "(matched without regard to case; * and ? are patterns)")
        if not tables:
            raise _Refused(f"owner {owner!r} holds no table to copy")
        clashes = _case_clashes(tables, _rows(conn, _COLUMNS, owner=owner))
        if clashes:
            raise _Refused("DuckDB matches names without regard to case, so it cannot keep these apart: "
                           + "; ".join(clashes))
        lob_bytes: int | None = None
        lob_note = None
        if _rows(conn, _USER)[0][0] == owner:
            wanted = set(tables)
            lob_bytes = sum(int(b or 0) for t, b in _rows(conn, _LOBS) if t in wanted)
        else:
            lob_note = (f"LOB segments: only the owner {owner!r} can see their size, so CLOB and BLOB values stored "
                        "outside the row are not in the estimate")
        selected = set(tables)
        est = estimate([r for r in stats if r[0] in selected], lob_bytes)
        _print_estimate(est, tables, lob_note)
        if est.total_bytes > args.max_gb * 1e9:
            raise _Refused(f"the estimate, {_gb(est.total_bytes)}, is past --max-gb {args.max_gb:g}; raise it to copy")
        table_comments = {t: c for t, c in _rows(conn, _TABLE_COMMENTS, owner=owner) if t in selected}
        column_comments = {(t, c): text for t, c, text in _rows(conn, _COLUMN_COMMENTS, owner=owner)
                           if t in selected}
        keys = _rows(conn, _KEYS, owner=owner)

    duck = duckdb.connect(out)
    try:
        present = {t.casefold() for t in _duck_tables(duck)}
        started = time.monotonic()
        for table in tables:
            if table.casefold() in present and not args.replace:
                print(f"{table}: already in {out}, skipped (--replace copies it again)")
                continue
            t0 = time.monotonic()
            rows = copy_table(conn, duck, owner, table, args.batch_rows)
            print(f"{table}: {rows} rows in {time.monotonic() - t0:.1f} s", flush=True)
        in_file = set(_duck_tables(duck))
        landed = [t for t in tables if t in in_file]
        columns = {t: _duck_columns(duck, t) for t in landed}
        statements = comment_statements(
            {t: c for t, c in table_comments.items() if t in in_file},
            {(t, c): text for (t, c), text in column_comments.items() if t in in_file and c in columns[t]})
        for statement in statements:
            duck.execute(statement)
        key_rows = declared_key_rows([tuple(r) for r in keys], in_file & set(owner_tables))
        _write_keys(duck, key_rows)
        duck.execute("CHECKPOINT")
    finally:
        duck.close()
    entry_path = out + ".source.json"
    with open(entry_path, "w") as fh:
        json.dump([source_entry(spec, out, tables)], fh, indent=1)
        fh.write("\n")
    print(f"copied {len(tables)} table(s) in {time.monotonic() - started:.1f} s, with {len(statements)} comment(s) and "
          f"{len(key_rows)} declared key column pair(s)")
    print(f"""
Next:
  1. Done: the tables are in {out}, schema main, named as Oracle stores them.
  2. In a separate folder, point a source at the copy: {entry_path} holds its entry, with the Oracle source's id
     and catalog. Use the same definitions and access policy files as on Oracle, set MNEMIQ_SOURCES_PATH to that
     file, and run `mnemiq enrich`, then `mnemiq build`.
  3. Copy the store.duckdb it writes into your Oracle deployment. The Oracle source entry must keep the same id and
     catalog, and list the same tables under "tables", so the store's tables name the source it serves.""")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sources", help="the manifest (default: MNEMIQ_SOURCES_PATH)")
    p.add_argument("--source-id", help="the Oracle source to copy (default: the manifest's only Oracle source)")
    p.add_argument("--out", required=True, help="the DuckDB file to copy into (made if absent)")
    p.add_argument("--max-gb", type=float, default=10.0, help="refuse when the estimate is past this, in decimal GB")
    p.add_argument("--replace", action="store_true", help="copy again a table the file already holds")
    p.add_argument("--batch-rows", type=int, default=100_000, help="rows per fetched batch")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = Settings()
        path = args.sources or settings.sources_path
        if not path:
            raise _Refused("no manifest: pass --sources or set MNEMIQ_SOURCES_PATH")
        spec = _choose_source(settings.model_copy(update={"sources_path": path}).source_specs(), args.source_id)
        missing = [name for name, value in (("MNEMIQ_ORACLE_USER", settings.oracle_user),
                                            ("MNEMIQ_ORACLE_PASSWORD", settings.oracle_password)) if not value]
        if missing:
            raise _Refused(f"set {' and '.join(missing)}: the copy reads Oracle as that user")
        if settings.oracle_wallet_password and not settings.oracle_config_dir:
            raise _Refused("MNEMIQ_ORACLE_WALLET_PASSWORD is set but MNEMIQ_ORACLE_CONFIG_DIR is not, so there is no "
                           "directory to find a wallet in")
        out = os.path.abspath(args.out)
        conn = connect(spec, settings)
        try:
            return _copy(conn, spec, args, out)
        finally:
            conn.close()
    except _Refused as refusal:
        print(f"copy_oracle_to_duckdb: {refusal}", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
