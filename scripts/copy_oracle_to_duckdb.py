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
- **Resumes.** Each table is copied in one DuckDB transaction, committed after its last row with its row in
  `mnemiq.copied_tables`, so a copy stopped mid-table, even by the process dying, leaves no trace of that table. A rerun
  skips a table recorded complete unless `--replace`. The file records the one source it copies (`mnemiq.copy_source`:
  target, owner and read user, never a password), and a file recording another, or holding tables with no record of
  their source, is refused before Oracle is reached.
- **Comments and keys.** `ALL_TAB_COMMENTS` and `ALL_COL_COMMENTS` become DuckDB `COMMENT ON` statements, and the keys
  `ALL_CONSTRAINTS` declares between copied tables of the owner become rows of `mnemiq.declared_keys`. The DuckDB
  adapter reads both, so a store enriched on the copy keeps them. Nothing in DuckDB enforces a key, so an orphan row
  cannot stop the copy.
- **Writes the copy's source entry** to `<out>.source.json` (`kind: duckdb`, the Oracle source's `id` and `catalog`,
  `schema: main`, the tables) and prints the steps that follow.

A `NUMBER` whose declaration bounds its values lands exactly (`number_type`): with no scale and at most 18 digits as
BIGINT, otherwise as DECIMAL(p, s), up to DuckDB's widest, 38 digits. A `NUMBER` with no precision or scale declared,
a `FLOAT`, or one past 38 digits keeps python-oracledb's own mapping, DOUBLE, so digits past about 15 significant ones
are lost. Each column's type is read from `ALL_TAB_COLUMNS` and requested of the driver by name; its decimal mapping
for the whole table is not used, as it fails the fetch on a `FLOAT` or a negative scale (measured on 26.1). Each batch
is checked with Arrow's own validation, so a malformed one stops the copy rather than land. `DATE` and `TIMESTAMP` land
as TIMESTAMP; CLOB and BLOB values as VARCHAR and BLOB (each under 1 GB: the DataFrame fetch reads a LOB's value whole,
whatever `oracledb.defaults.fetch_lobs` says, measured).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase

import duckdb
import pyarrow as pa

from mnemiq.adapters.duckdb import COPY_STATE_TABLE, KEYS_SCHEMA, KEYS_TABLE
from mnemiq.config import Settings, SourceSpec
from mnemiq.duckdb_extensions import connect_duckdb

REFUSED = 2
# What the copy records in the file, in mnemiq's own schema beside the declared keys (outside `main`, so no reader of the
# source's tables sees them): the one source the file is a copy of, and each table copied whole from it.
SOURCE_TABLE = "copy_source"
COMPLETED_TABLE = "copied_tables"

_TABLES = "SELECT table_name, num_rows, avg_row_len FROM all_tables WHERE owner = :owner ORDER BY table_name"
_COLUMNS = ("SELECT table_name, column_name FROM all_tab_columns WHERE owner = :owner "
            "ORDER BY table_name, column_id")
_DECLARED = ("SELECT column_name, data_type, data_precision, data_scale FROM all_tab_columns "
             "WHERE owner = :owner AND table_name = :table_name")
_TABLE_COMMENTS = "SELECT table_name, comments FROM all_tab_comments WHERE owner = :owner AND comments IS NOT NULL"
_COLUMN_COMMENTS = ("SELECT table_name, column_name, comments FROM all_col_comments "
                    "WHERE owner = :owner AND comments IS NOT NULL")
# The statement `OracleAdapter.foreign_keys` reads, kept to keys whose target is the owner's too: a key into another
# schema names a table this copy does not hold, or worse, one of the owner's own that happens to share its name.
_KEYS = ("SELECT c.table_name, cc.column_name, rc.table_name, rcc.column_name, c.constraint_name, cc.position "
         "FROM all_constraints c "
         "JOIN all_cons_columns cc ON cc.owner = c.owner AND cc.constraint_name = c.constraint_name "
         "JOIN all_constraints rc ON rc.owner = c.r_owner AND rc.constraint_name = c.r_constraint_name "
         "JOIN all_cons_columns rcc ON rcc.owner = rc.owner AND rcc.constraint_name = rc.constraint_name "
         "AND rcc.position = cc.position "
         "WHERE c.owner = :owner AND c.constraint_type = 'R' AND c.r_owner = c.owner "
         "ORDER BY c.table_name, c.constraint_name, cc.position")
_USER = "SELECT user FROM dual"
# The database's own names, which every user can read: which database a target reached, whatever the alias says.
_DATABASE = "SELECT SYS_CONTEXT('USERENV', 'DB_UNIQUE_NAME'), SYS_CONTEXT('USERENV', 'CON_NAME') FROM dual"
# LOB segments are sized only in USER_SEGMENTS (DBA_SEGMENTS needs a grant), so only the owner can see them.
_LOBS = ("SELECT l.table_name, SUM(s.bytes) FROM user_lobs l JOIN user_segments s ON s.segment_name = l.segment_name "
         "GROUP BY l.table_name")


@dataclass(frozen=True)
class CopySource:
    """The source a file is a copy of: the Oracle target, the schema owner whose tables it holds, the user that read
    them, whose grants (and any row policy) decide which rows it saw, and the database the target reached, by its own
    names (`DB_UNIQUE_NAME`, `CON_NAME`): with a config directory one TNS alias can resolve to another database, so the
    target string alone does not say which one it is. The two names are None until the copy has connected. Never a
    password."""

    target: str
    owner: str
    read_user: str
    db_unique_name: str | None = None
    con_name: str | None = None


@dataclass(frozen=True)
class _FileState:
    """What a copy's file already records, read before anything is read or written: its source (`sources` rows of it),
    the tables recorded complete, and every table in its `main`."""

    source: CopySource | None
    sources: int
    completed: frozenset[str]
    tables: frozenset[str]


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


def stale_comment_clears(duck, tables: list[str], current_tables: dict[str, str],
                         current_columns: dict[tuple[str, str], str]) -> list[str]:
    """`COMMENT ON ... IS NULL` for every comment the file holds on one of `tables` that the source no longer has. A
    resumed copy skips the tables it finished, and the dictionary reads return no row for a comment that is gone, so
    without these the file would keep a comment the source dropped (Codex's third review)."""
    wanted = set(tables)
    held_tables = duck.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = current_database() AND schema_name = 'main' "
        "AND comment IS NOT NULL AND comment <> ''").fetchall()
    held_columns = duck.execute(
        "SELECT table_name, column_name FROM duckdb_columns() WHERE database_name = current_database() "
        "AND schema_name = 'main' AND comment IS NOT NULL AND comment <> ''").fetchall()
    clears = [f"COMMENT ON TABLE {_quoted(t)} IS NULL" for (t,) in held_tables
              if t in wanted and t not in current_tables]
    clears += [f"COMMENT ON COLUMN {_quoted(t)}.{_quoted(c)} IS NULL" for t, c in held_columns
               if t in wanted and (t, c) not in current_columns]
    return clears


# (from_table, from_column, to_table, to_column, constraint_name, position): `position` keeps a composite key's pairs
# in the order the key declares them, which the DuckDB adapter reads them by.
KeyRow = tuple[str, str, str, str, str, int]


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


def number_type(data_type: str, precision: int | None, scale: int | None) -> pa.DataType | None:
    """The Arrow type that holds, exactly, every value an Oracle column of this declaration can; None where no fixed
    type does and the driver's own mapping stands (a NUMBER with nothing declared, a FLOAT, past 38 digits).

    A scale of 0 or less allows only integers, of up to `precision - scale` digits: int64 to 18, else a decimal of no
    scale. A positive scale gets a decimal of that scale, as wide as the precision, or as the scale where that is wider
    (NUMBER(3, 5) holds 0.00123). `NUMBER(*, s)`, listed with no precision, has 38. Never an integer type for a column
    that can hold a fraction: the driver truncates one into it without a word (measured), where a decimal too narrow
    for a value fails the fetch (DPY-4042)."""
    if data_type != "NUMBER" or scale is None:
        return None
    digits = 38 if precision is None else precision
    if scale <= 0:
        width = digits - scale
        if width <= 18:
            return pa.int64()
        return pa.decimal128(width, 0) if width <= 38 else None
    width = max(digits, scale)
    return pa.decimal128(width, scale) if width <= 38 else None


def _requested(fetched: pa.Schema, declared: dict[str, tuple]) -> pa.Schema:
    """The fetch's own schema with each column `number_type` gives a type for set to it: the driver matches a requested
    schema by position (measured: a reversed one fails), so the order is the fetch's, never the dictionary's."""
    fields = []
    for field in fetched:
        kind = number_type(*declared[field.name]) if field.name in declared else None
        fields.append(field if kind is None else field.with_type(kind))
    return pa.schema(fields)


def _whole(batch, table: str, number: int) -> pa.Table:
    """The driver's batch as an Arrow table, refused unless Arrow's own validation passes. `pa.table` imports a
    DataFrame through the C data interface without checking it, and python-oracledb before 26.0.1 handed over a batch
    with one column a value short (its decimal fetch, measured): stored, it would be a table whose rows no longer line
    up. The pin's floor is past that release; the check stays, so a driver that does it again stops the copy."""
    data = pa.table(batch)
    try:
        data.validate()
    except pa.ArrowInvalid as exc:
        raise RuntimeError(f"python-oracledb returned a malformed batch for {table!r} (batch {number}): {exc}; "
                           "the copy stops rather than store it") from exc
    return data


def _for_duckdb(table: pa.Table) -> pa.Table:
    """A fetched batch with a timestamp in seconds or milliseconds (an Oracle `DATE`, `TIMESTAMP(3)`) widened to
    microseconds, without loss, so it lands as DuckDB's own TIMESTAMP rather than TIMESTAMP_S or TIMESTAMP_MS."""
    fields = []
    changed = False
    for field in table.schema:
        kind = field.type
        if pa.types.is_timestamp(kind) and kind.unit in ("s", "ms"):
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

    The DuckDB side is one transaction too: dropping the table it replaces, creating it, every batch, committed only
    after the last. So no other name is ever used (a staging name could be a real table of the schema), and a copy that
    stops mid-table, by an error or by the process dying, leaves no trace of the table and keeps the one it was
    replacing: DuckDB 1.5.4 discards an uncommitted transaction on restart, measured with 231 MB already written to the
    file when the process was killed. The table's row in `mnemiq.copied_tables` is written in the same transaction,
    so a resume skips a table only when it was copied whole. The columns and their order come from `SELECT * ... WHERE
    1 = 0`, so a table with no rows keeps its columns, with the types its rows would have had."""
    select = f"SELECT * FROM {_quoted(owner)}.{_quoted(table)}"
    rows = 0
    created = False
    duck.execute("BEGIN TRANSACTION")
    try:
        duck.execute(f"DROP TABLE IF EXISTS main.{_quoted(table)}")
        with _read_only(conn):
            declared = {name: (kind, precision, scale)
                        for name, kind, precision, scale in _rows(conn, _DECLARED, owner=owner, table_name=table)}
            # `fetch_decimals` named, whatever `oracledb.defaults` says: on, it fails the whole fetch on a FLOAT or a
            # negative scale (DPY-4042, measured), so exact types are requested column by column instead.
            fetched = _whole(conn.fetch_df_all(f"{select} WHERE 1 = 0", fetch_decimals=False), table, 0).schema
            requested = _requested(fetched, declared)
            for number, batch in enumerate(conn.fetch_df_batches(select, size=batch_rows, fetch_decimals=False,
                                                                 requested_schema=requested), 1):
                data = _for_duckdb(_whole(batch, table, number))
                _append(duck, table, data, create=not created)
                created = True
                rows += data.num_rows
            if not created:
                _append(duck, table, _for_duckdb(requested.empty_table()), create=True)
        # Recorded in the same transaction, so a table is recorded complete exactly when it is in the file whole.
        duck.execute(f"CREATE SCHEMA IF NOT EXISTS {KEYS_SCHEMA}")
        duck.execute(f"CREATE TABLE IF NOT EXISTS {KEYS_SCHEMA}.{COMPLETED_TABLE} "
                     "(table_name VARCHAR, row_count BIGINT, completed_at TIMESTAMPTZ)")
        duck.execute(f"DELETE FROM {KEYS_SCHEMA}.{COMPLETED_TABLE} WHERE table_name = ?", [table])
        duck.execute(f"INSERT INTO {KEYS_SCHEMA}.{COMPLETED_TABLE} VALUES (?, ?, now())", [table, rows])
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
    """Names DuckDB would take for one: tables, or columns of one table, equal without regard to case."""
    clashes: list[str] = []
    by_fold: dict[str, list[str]] = {}
    for t in tables:
        by_fold.setdefault(t.casefold(), []).append(t)
    clashes += [f"tables {', '.join(map(repr, names))}" for names in by_fold.values() if len(names) > 1]
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
    sys.stdout.flush()  # before any refusal on stderr, so a log of both reads in the order they happened


def _open_writer(out: str, local_only: bool):
    """The one write connection to `out`, opened before anything is checked and held until the copy ends: DuckDB's file
    lock then keeps every other process out, so what the checks read cannot change under them. Two runs that both
    found no file used to both go on. Refused (exit 2) when another process holds the file."""
    try:
        # mnemiq's own door: under MNEMIQ_LOCAL_ONLY, DuckDB installs and loads no extension by download here either.
        return connect_duckdb(out, local_only=local_only)
    except duckdb.IOException as exc:
        if "lock" not in str(exc).lower():
            raise
        raise _Refused(f"{out} is held by another process (another copy, or something reading it): wait for it to "
                       "end, or copy into a new --out") from exc


def _discard(out: str) -> None:
    """Remove the file this run created, before its write connection closes: while the connection holds the lock no
    other process can have opened it, and DuckDB leaves nothing behind for an unlinked file (measured)."""
    for path in (out, out + ".wal"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def _file_state(duck) -> _FileState:
    """What the file records, read through the copy's own write connection."""
    held = {(schema, name) for schema, name in duck.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_catalog = current_database()").fetchall()}
    sources = []
    if (KEYS_SCHEMA, SOURCE_TABLE) in held:
        columns = {c for (c,) in duck.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_catalog = current_database() "
            "AND table_schema = ? AND table_name = ?", [KEYS_SCHEMA, SOURCE_TABLE]).fetchall()}
        if "db_unique_name" not in columns:
            raise _Refused(f"{KEYS_SCHEMA}.{SOURCE_TABLE} records its source without the database's names (an "
                           "earlier version of this script), so which database it copied cannot be checked; copy "
                           "into a new --out")
        sources = duck.execute(f"SELECT target, owner, read_user, db_unique_name, con_name "
                               f"FROM {KEYS_SCHEMA}.{SOURCE_TABLE}").fetchall()
    completed = ({r[0] for r in duck.execute(f"SELECT table_name FROM {KEYS_SCHEMA}.{COMPLETED_TABLE}").fetchall()}
                 if (KEYS_SCHEMA, COMPLETED_TABLE) in held else set())
    return _FileState(CopySource(*sources[0]) if len(sources) == 1 else None, len(sources), frozenset(completed),
                      frozenset(name for schema, name in held if schema == "main"))


def _check_file(state: _FileState, source: CopySource, out: str) -> None:
    """Refuse a file this copy cannot add to: one recording another source, which a resume would skip identically named
    tables of and whose records it would overwrite, or one holding tables with no record of where they came from."""
    if state.sources > 1:
        raise _Refused(f"{out} records {state.sources} sources; copy into a new --out")
    if state.source is None:
        if state.tables:
            raise _Refused(f"{out} holds {len(state.tables)} table(s) but no record of the source they were copied "
                           "from, so they cannot be told apart from this source's; copy into a new --out")
        return
    differ = [name for name in ("target", "owner", "read_user")
              if getattr(state.source, name) != getattr(source, name)]
    if differ:
        raise _Refused(f"{out} is a copy of another source: its {', '.join(differ)} differ(s) from this run's (it "
                       f"records owner {state.source.owner!r}, read as {state.source.read_user!r}); a resume would "
                       "skip tables that only share a name, so copy into a new --out")
    if source.db_unique_name is None:
        return  # not connected yet: the database's names are checked once it is
    database = [name for name in ("db_unique_name", "con_name") if getattr(state.source, name) != getattr(source, name)]
    if database:
        raise _Refused(f"{out} is a copy of another database: the target {source.target!r} now reaches "
                       f"{source.db_unique_name}/{source.con_name}, and the file records "
                       f"{state.source.db_unique_name}/{state.source.con_name}; copy into a new --out")


def _set_state(duck, state: str) -> None:
    """The copy's progress in the file, in its own transaction: `copying` before its first write, `complete` only after
    its tables, comments and keys are all published. `mnemiq enrich` refuses a file that records anything else."""
    duck.execute("BEGIN TRANSACTION")
    duck.execute(f"CREATE SCHEMA IF NOT EXISTS {KEYS_SCHEMA}")
    duck.execute(f"CREATE OR REPLACE TABLE {KEYS_SCHEMA}.{COPY_STATE_TABLE} (state VARCHAR, updated_at TIMESTAMPTZ)")
    duck.execute(f"INSERT INTO {KEYS_SCHEMA}.{COPY_STATE_TABLE} VALUES (?, now())", [state])
    duck.execute("COMMIT")


def _record_source(duck, source: CopySource) -> None:
    duck.execute("BEGIN TRANSACTION")
    duck.execute(f"CREATE SCHEMA IF NOT EXISTS {KEYS_SCHEMA}")
    duck.execute(f"CREATE TABLE IF NOT EXISTS {KEYS_SCHEMA}.{SOURCE_TABLE} "
                 "(target VARCHAR, owner VARCHAR, read_user VARCHAR, db_unique_name VARCHAR, con_name VARCHAR)")
    duck.execute(f"INSERT INTO {KEYS_SCHEMA}.{SOURCE_TABLE} VALUES (?, ?, ?, ?, ?)",
                 [source.target, source.owner, source.read_user, source.db_unique_name, source.con_name])
    duck.execute("COMMIT")


def _completed(duck) -> set[str]:
    found = duck.execute("SELECT count(*) FROM information_schema.tables WHERE table_catalog = current_database() "
                         "AND table_schema = ? AND table_name = ?", [KEYS_SCHEMA, COMPLETED_TABLE]).fetchone()[0]
    return {r[0] for r in duck.execute(f"SELECT table_name FROM {KEYS_SCHEMA}.{COMPLETED_TABLE}").fetchall()} \
        if found else set()


def _duck_tables(duck) -> list[str]:
    return [r[0] for r in duck.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = current_database() AND table_schema = 'main' ORDER BY table_name").fetchall()]


def _duck_columns(duck, table: str) -> set[str]:
    return {r[0] for r in duck.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?", [table]).fetchall()}


def _write_keys(duck, rows: list[KeyRow]) -> None:
    """Replace the file's declared keys with `rows`, all or nothing: half of a composite key would read as a valid
    one-column join, and DuckDB's `executemany` keeps the rows before a failing one, so the replace and every insert
    are one transaction, and a failure leaves the previous key set as it was."""
    duck.execute("BEGIN TRANSACTION")
    try:
        duck.execute(f"CREATE SCHEMA IF NOT EXISTS {KEYS_SCHEMA}")
        duck.execute(f"CREATE OR REPLACE TABLE {KEYS_SCHEMA}.{KEYS_TABLE} (from_table VARCHAR, from_column VARCHAR, "
                     "to_table VARCHAR, to_column VARCHAR, constraint_name VARCHAR, position INTEGER)")
        if rows:
            duck.executemany(f"INSERT INTO {KEYS_SCHEMA}.{KEYS_TABLE} VALUES (?, ?, ?, ?, ?, ?)", rows)
        duck.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(Exception):
            duck.execute("ROLLBACK")
        raise


@dataclass(frozen=True)
class _Plan:
    """What the dictionary says the copy will do, read before anything is written."""

    owner: str
    owner_tables: list[str]
    tables: list[str]
    table_comments: dict[str, str]
    column_comments: dict[tuple[str, str], str]
    keys: list[tuple]


def _plan(conn, spec: SourceSpec, args, state: _FileState) -> _Plan:
    """Every dictionary read, the estimate and every refusal that rests on them, in one read-only transaction."""
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
        # The tables this file already holds whole count too: a table selected now whose name equals one of theirs
        # without regard to case would collide with it in DuckDB.
        copied_before = sorted(state.completed & state.tables)
        clashes = _case_clashes(sorted(set(tables) | set(copied_before)), _rows(conn, _COLUMNS, owner=owner))
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
    return _Plan(owner, owner_tables, tables, table_comments, column_comments, keys)


def _write(conn, duck, spec: SourceSpec, args, out: str, plan: _Plan, *, source: CopySource,
           state: _FileState) -> int:
    owner, owner_tables, tables = plan.owner, plan.owner_tables, plan.tables
    table_comments, column_comments, keys = plan.table_comments, plan.column_comments, plan.keys
    _set_state(duck, "copying")
    if state.source is None:
        _record_source(duck, source)
    # Skipped only when recorded complete and still there: a table present without the record is copied again.
    done = _completed(duck) & set(_duck_tables(duck))
    started = time.monotonic()
    skipped = 0
    for table in tables:
        if table in done and not args.replace:
            print(f"{table}: already in {out}, skipped (--replace copies it again)", flush=True)
            skipped += 1
            continue
        t0 = time.monotonic()
        rows = copy_table(conn, duck, owner, table, args.batch_rows)
        print(f"{table}: {rows} rows in {time.monotonic() - t0:.1f} s", flush=True)
    in_file = _completed(duck) & set(_duck_tables(duck))
    landed = [t for t in tables if t in in_file]
    columns = {t: _duck_columns(duck, t) for t in landed}
    current_tables = {t: c for t, c in table_comments.items() if t in in_file}
    current_columns = {(t, c): text for (t, c), text in column_comments.items() if t in in_file and c in columns[t]}
    for clear in stale_comment_clears(duck, landed, current_tables, current_columns):
        duck.execute(clear)
    statements = comment_statements(current_tables, current_columns)
    for statement in statements:
        duck.execute(statement)
    key_rows = declared_key_rows([tuple(r) for r in keys], in_file & set(owner_tables))
    _write_keys(duck, key_rows)
    _set_state(duck, "complete")
    duck.execute("CHECKPOINT")
    entry_path = out + ".source.json"
    with open(entry_path, "w") as fh:
        json.dump([source_entry(spec, out, tables)], fh, indent=1)
        fh.write("\n")
    print(f"{len(tables)} table(s) in the file: {len(tables) - skipped} copied, {skipped} already there, in "
          f"{time.monotonic() - started:.1f} s; {len(statements)} comment(s) and {len(key_rows)} declared key column "
          "pair(s) written")
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


def _run(duck, created: bool, spec: SourceSpec, args, out: str, settings: Settings, source: CopySource) -> int:
    """The copy through the held write connection: checks, plan, writes. A run that stops before its first write
    removes a file it created, so a refused copy leaves nothing behind."""
    conn = None
    try:
        state = _file_state(duck)
        _check_file(state, source, out)
        conn = connect(spec, settings)
        # Right after connecting, before the dictionary: which database the target reached.
        with _read_only(conn):
            db_unique_name, con_name = _rows(conn, _DATABASE)[0]
        source = dataclasses.replace(source, db_unique_name=db_unique_name, con_name=con_name)
        _check_file(state, source, out)
        plan = _plan(conn, spec, args, state)
    except BaseException:
        if conn is not None:
            conn.close()
        if created:
            _discard(out)
        raise
    try:
        return _write(conn, duck, spec, args, out, plan, source=source, state=state)
    finally:
        conn.close()


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
        source = CopySource(target=spec.target, owner=spec.schema.upper(), read_user=settings.oracle_user.upper())
        created = not os.path.exists(out)
        duck = _open_writer(out, settings.local_only)
        try:
            return _run(duck, created, spec, args, out, settings, source)
        finally:
            duck.close()
    except _Refused as refusal:
        print(f"copy_oracle_to_duckdb: {refusal}", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
