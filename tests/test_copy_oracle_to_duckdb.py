"""`scripts/copy_oracle_to_duckdb.py`: Oracle tables into one DuckDB file to enrich, read-only, disk estimate first.

Enriching on a DuckDB copy and serving the store on Oracle is the route a large schema can afford, and nothing made the
copy: a design partner would have written one by hand, and lost the names' case, the comments and the declared keys on
the way. These tests drive the script against a fake connection, whose `fetch_df_batches` yields Arrow tables and whose
cursor answers the dictionary queries from canned rows, so no Oracle is needed. What they pin: nothing is read past
`--max-gb` or past a name DuckDB cannot keep apart; a table a stopped copy left behind is never one a resume skips as
done; each table is read in its own read-only transaction; comments and keys arrive where the DuckDB adapter reads them.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import duckdb
import pyarrow as pa
import pytest

import scripts.copy_oracle_to_duckdb as copy
from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.config import SourceSpec
from scripts.copy_oracle_to_duckdb import (
    Estimate,
    comment_statements,
    copy_table,
    declared_key_rows,
    estimate,
    main,
    select_tables,
    source_entry,
)

_ORDERS = pa.schema([("ID", pa.int64()), ("Status", pa.large_string())])
_CUSTOMER = pa.schema([("ID", pa.int64()), ("NAME", pa.large_string())])


def _batch(schema: pa.Schema, ids: range) -> pa.Table:
    return pa.table({schema.names[0]: list(ids), schema.names[1]: [f"v{i}" for i in ids]}, schema=schema)


class _Cursor:
    def __init__(self, fake: "_Oracle") -> None:
        self._fake = fake
        self._rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, parameters=None, **binds):
        self._fake.log.append(sql)
        low = " ".join(sql.lower().split())
        if low.startswith("set transaction"):
            self._rows = []
        elif "from dual" in low:
            self._rows = [(self._fake.user,)]
        elif "from all_tables" in low:
            self._rows = list(self._fake.stats)
        elif "from all_tab_columns" in low:
            self._rows = [(t, c) for t, schema in self._fake.schemas.items() for c in schema.names]
        elif "from all_tab_comments" in low:
            self._rows = list(self._fake.table_comments.items())
        elif "from all_col_comments" in low:
            self._rows = [(t, c, text) for (t, c), text in self._fake.column_comments.items()]
        elif "from all_constraints" in low:
            self._rows = list(self._fake.keys)
        elif "from user_lobs" in low:
            self._rows = list(self._fake.lobs)
        else:
            raise AssertionError(f"the fake does not answer {sql!r}")

    def fetchall(self):
        return self._rows


class _Oracle:
    """A connection the script reads through: dictionary rows by query, table rows as Arrow batches."""

    def __init__(self, batches: dict[str, list[pa.Table]], schemas: dict[str, pa.Schema] | None = None, *,
                 stats=None, table_comments=None, column_comments=None, keys=(), lobs=(), user="READER",
                 fail_on: tuple[str, int] | None = None) -> None:
        self.batches = batches
        self.schemas = schemas or {t: b[0].schema for t, b in batches.items()}
        self.stats = stats if stats is not None else [(t, 10, 100) for t in self.schemas]
        self.table_comments = table_comments or {}
        self.column_comments = column_comments or {}
        self.keys = list(keys)
        self.lobs = list(lobs)
        self.user = user
        self.fail_on = fail_on
        self.on_fail = lambda: None  # what a test looks at the moment the copy stops, before anything cleans up
        self.log: list = []
        self.fetched: list[str] = []
        self.fetch_kwargs: list[dict] = []

    def cursor(self):
        return _Cursor(self)

    def rollback(self):
        self.log.append("ROLLBACK")

    def close(self):
        self.log.append("CLOSE")

    @staticmethod
    def _table(statement: str) -> str:
        return statement.split(".")[-1].split()[0].strip('"')

    def fetch_df_batches(self, statement, parameters=None, size=None, **kwargs):
        table = self._table(statement)
        self.fetched.append(table)
        self.fetch_kwargs.append(kwargs)
        for i, batch in enumerate(self.batches[table]):
            if self.fail_on == (table, i):
                self.on_fail()
                raise RuntimeError("the connection was lost")
            self.log.append(("batch", table, i))
            yield batch

    def fetch_df_all(self, statement, parameters=None, arraysize=None, **kwargs):
        assert "WHERE 1 = 0" in statement
        self.fetch_kwargs.append(kwargs)
        return self.schemas[self._table(statement)].empty_table()


_SOURCE = {"id": "plant", "kind": "oracle", "target": "db.example:1521/APP", "catalog": "plant", "schema": "APP",
           "tables": ["ORDERS", "CUSTOMER"]}


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A manifest naming one Oracle source, its credentials in the environment, and a seam for the fake."""
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([_SOURCE]))
    monkeypatch.setenv("MNEMIQ_SOURCES_PATH", str(manifest))
    monkeypatch.setenv("MNEMIQ_ORACLE_USER", "READER")
    monkeypatch.setenv("MNEMIQ_ORACLE_PASSWORD", "not-a-real-password")
    for name in ("MNEMIQ_ORACLE_CONFIG_DIR", "MNEMIQ_ORACLE_WALLET_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    def run(fake: _Oracle, *flags: str) -> int:
        def connected(spec, settings):
            run.connects.append((spec.target, spec.schema, settings.oracle_user))
            return fake

        monkeypatch.setattr(copy, "connect", connected)
        return main(["--out", str(tmp_path / "copy.duckdb"), *flags])

    run.out = str(tmp_path / "copy.duckdb")
    run.manifest = manifest
    run.connects = []
    return run


def _shop(**kwargs) -> _Oracle:
    return _Oracle({"ORDERS": [_batch(_ORDERS, range(3)), _batch(_ORDERS, range(3, 5))],
                    "CUSTOMER": [_batch(_CUSTOMER, range(2))]}, **kwargs)


def _tables(path: str, read_only: bool = True) -> dict[str, int]:
    con = duckdb.connect(path, read_only=read_only)
    try:
        names = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()]
        return {n: con.execute(f'SELECT count(*) FROM main."{n}"').fetchone()[0] for n in names}
    finally:
        con.close()


def test_select_tables_matches_patterns_without_case():
    owner = ["ORDERS", "ORDER_LINES", "CUSTOMER"]
    assert select_tables(owner, ("order*",)) == ["ORDERS", "ORDER_LINES"]
    assert select_tables(owner, ()) == owner
    assert select_tables(owner, ("customer", "orders")) == ["ORDERS", "CUSTOMER"], "the owner's order is kept"


def test_estimate_sums_rows_times_length_and_names_unanalyzed():
    est = estimate([("A", 10, 100), ("B", None, 50)], None)
    assert est == Estimate(total_bytes=1000, per_table={"A": 1000, "B": 0}, unanalyzed=["B"], lob_bytes=None)
    assert estimate([("A", 10, 100), ("C", 5, None)], 500).total_bytes == 1500, "LOB segments add when seen"


def test_the_copy_refuses_past_max_gb_before_reading_a_row(deployment, capsys):
    fake = _shop(stats=[("ORDERS", 11_000_000, 1000), ("CUSTOMER", 0, 10)])
    assert deployment(fake, "--max-gb", "10") == 2
    assert fake.fetched == []
    out = capsys.readouterr()
    assert "11.00 GB" in out.out and "--max-gb" in out.err
    assert not os.path.exists(deployment.out), "no file is made for a copy that does not start"


def test_the_estimate_is_printed_before_any_row_is_read(deployment, capsys):
    fake = _shop(stats=[("ORDERS", 5, 100), ("CUSTOMER", None, None)])
    printed: list[str] = []
    real = fake.fetch_df_batches

    def watched(*args, **kwargs):
        printed.append(capsys.readouterr().out)
        return real(*args, **kwargs)

    fake.fetch_df_batches = watched
    assert deployment(fake) == 0
    assert printed, "a table was read"
    assert "0.00 GB" in printed[0] and "CUSTOMER" in printed[0], "the estimate and the unanalyzed table, first"


def test_comment_statements_escape_quotes_and_quote_names():
    statements = comment_statements({"ORDER": "the customer's order"}, {("ORDER", "Status"): "it's 'open'"})
    assert statements == [
        "COMMENT ON TABLE \"ORDER\" IS 'the customer''s order'",
        "COMMENT ON COLUMN \"ORDER\".\"Status\" IS 'it''s ''open'''",
    ]
    con = duckdb.connect(":memory:")
    con.execute('CREATE TABLE "ORDER" ("Status" VARCHAR)')
    for statement in statements:
        con.execute(statement)
    assert con.execute("SELECT comment FROM duckdb_tables() WHERE table_name = 'ORDER'").fetchone()[0] == \
        "the customer's order"
    assert con.execute("SELECT comment FROM duckdb_columns() WHERE column_name = 'Status'").fetchone()[0] == \
        "it's 'open'"


def test_a_name_holding_a_double_quote_is_quoted_whole():
    assert comment_statements({'A"B': "x"}, {}) == ["COMMENT ON TABLE \"A\"\"B\" IS 'x'"]


def test_a_table_is_copied_with_oracles_names_and_all_rows(tmp_path):
    fake = _shop()
    duck = duckdb.connect(str(tmp_path / "c.duckdb"))
    assert copy_table(fake, duck, "APP", "ORDERS", 3) == 5
    assert duck.execute('SELECT count(*) FROM main."ORDERS"').fetchone()[0] == 5
    assert [r[0] for r in duck.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'ORDERS' "
        "ORDER BY ordinal_position").fetchall()] == ["ID", "Status"]
    assert duck.execute("SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1, 2").fetchall() == [
        ("main", "ORDERS"), ("mnemiq", "copied_tables")], "the table, and the record that it is complete"
    assert duck.execute("SELECT table_name, row_count FROM mnemiq.copied_tables").fetchall() == [("ORDERS", 5)]


def test_the_table_is_read_by_its_qualified_quoted_name(tmp_path):
    seen: list[str] = []
    fake = _shop()
    real = fake.fetch_df_batches
    fake.fetch_df_batches = lambda statement, **kw: (seen.append(statement), real(statement, **kw))[1]
    copy_table(fake, duckdb.connect(str(tmp_path / "c.duckdb")), "APP", "ORDERS", 3)
    assert seen == ['SELECT * FROM "APP"."ORDERS"']


def test_an_empty_table_is_copied_with_its_columns(tmp_path):
    fake = _Oracle({"ORDERS": []}, {"ORDERS": _ORDERS})
    duck = duckdb.connect(str(tmp_path / "c.duckdb"))
    assert copy_table(fake, duck, "APP", "ORDERS", 3) == 0
    assert duck.execute('SELECT count(*) FROM main."ORDERS"').fetchone()[0] == 0
    assert [r[0] for r in duck.execute('DESCRIBE main."ORDERS"').fetchall()] == ["ID", "Status"]


def test_numbers_and_dates_land_as_the_duckdb_types_they_hold(tmp_path):
    # The driver's own mapping, decimals off: NUMBER(p, 0) with p up to 18 as int64, any other NUMBER as double. An
    # Oracle DATE (seconds) and a TIMESTAMP(3) (milliseconds) land as DuckDB's plain TIMESTAMP, widened without loss.
    schema = pa.schema([("N10", pa.int64()), ("N", pa.float64()), ("D", pa.timestamp("s")),
                        ("T3", pa.timestamp("ms")), ("T6", pa.timestamp("us"))])
    when = dt.datetime(2026, 1, 2, 3, 4, 5, 123000)
    batch = pa.table({"N10": [7], "N": [0.5], "D": [when.replace(microsecond=0)], "T3": [when], "T6": [when]},
                     schema=schema)
    duck = duckdb.connect(str(tmp_path / "c.duckdb"))
    fake = _Oracle({"T": [batch]})
    copy_table(fake, duck, "APP", "T", 10)
    types = dict(duck.execute("SELECT column_name, data_type FROM information_schema.columns "
                              "WHERE table_name = 'T'").fetchall())
    assert types == {"N10": "BIGINT", "N": "DOUBLE", "D": "TIMESTAMP", "T3": "TIMESTAMP", "T6": "TIMESTAMP"}
    assert duck.execute('SELECT "D", "T3" FROM "T"').fetchone() == (when.replace(microsecond=0), when)


def test_numbers_are_fetched_without_decimals(tmp_path):
    # python-oracledb 4.0.2's decimal fetch returned a decimal128 column one value short in some later batches of a
    # large table (measured, batches of 50,000 and 100,000; whole when one batch held every row). So the copy
    # asks for the driver's default mapping, by name, whatever `oracledb.defaults.fetch_decimals` is set to.
    fake = _Oracle({"ORDERS": [_batch(_ORDERS, range(2))], "EMPTY": []}, {"ORDERS": _ORDERS, "EMPTY": _ORDERS})
    duck = duckdb.connect(str(tmp_path / "c.duckdb"))
    copy_table(fake, duck, "APP", "ORDERS", 10)
    copy_table(fake, duck, "APP", "EMPTY", 10)
    assert fake.fetch_kwargs and all(kw.get("fetch_decimals") is False for kw in fake.fetch_kwargs)


class _Malformed:
    """What `pa.table` made of the driver's short batch: a table Arrow's own validation refuses."""

    num_rows = 3

    def validate(self, *, full=False):
        raise pa.ArrowInvalid("Column 1 named Status expected length 3 but got length 2")


def test_a_batch_arrow_finds_malformed_stops_the_copy_with_nothing_landed(tmp_path, monkeypatch):
    fake = _shop()
    real = copy.pa.table
    seen: list[int] = []

    def table(obj):
        seen.append(1)
        return _Malformed() if len(seen) == 2 else real(obj)

    monkeypatch.setattr(copy.pa, "table", table)
    duck = duckdb.connect(str(tmp_path / "c.duckdb"))
    with pytest.raises(RuntimeError, match="malformed batch") as raised:
        copy_table(fake, duck, "APP", "ORDERS", 3)
    assert "ORDERS" in str(raised.value) and "batch 2" in str(raised.value)
    assert duck.execute("SELECT count(*) FROM information_schema.tables").fetchone()[0] == 0
    assert fake.log[-1] == "ROLLBACK"


def test_a_stopped_copy_leaves_no_table_a_resume_would_skip(deployment):
    stopped = _shop(fail_on=("ORDERS", 1))
    at_the_stop: dict[str, int] | None = None

    def look():
        nonlocal at_the_stop
        at_the_stop = _tables(deployment.out, read_only=False)

    # The file as another connection sees it: read when the copy stops, before any handler can tidy up. (A second
    # connection in the same process, so not read-only: DuckDB refuses one file opened with two configurations.)
    stopped.on_fail = look
    with pytest.raises(RuntimeError, match="connection was lost"):
        deployment(stopped)
    assert at_the_stop is not None and "ORDERS" not in at_the_stop, "mid-table, no other connection sees the table"
    assert "ORDERS" not in _tables(deployment.out)
    assert not [t for t in _tables(deployment.out) if t.casefold().startswith("orders")], "no partial table left"
    assert stopped.log[-2:] == ["ROLLBACK", "CLOSE"], "the stopped table's transaction ended"
    resumed = _shop()
    assert deployment(resumed) == 0
    assert _tables(deployment.out) == {"ORDERS": 5, "CUSTOMER": 2}
    assert "ORDERS" in resumed.fetched


_KILLED_MID_TABLE = """
import os, signal, sys
import duckdb, pyarrow as pa
from scripts.copy_oracle_to_duckdb import copy_table

class Cursor:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def execute(self, sql, *args, **kwargs): pass

class Killed:
    def cursor(self): return Cursor()
    def rollback(self): pass
    def fetch_df_batches(self, statement, parameters=None, size=None, **kwargs):
        yield pa.table({"ID": list(range(3)), "Status": ["a", "b", "c"]})
        os.kill(os.getpid(), signal.SIGKILL)  # the process dies mid-table: no handler, no rollback, nothing tidied

copy_table(Killed(), duckdb.connect(sys.argv[1]), "APP", "ORDERS", 3)
"""


def test_a_process_killed_mid_table_leaves_no_table_at_all(tmp_path):
    import signal
    import subprocess
    import sys
    from pathlib import Path

    out = str(tmp_path / "c.duckdb")
    root = Path(__file__).resolve().parents[1]
    killed = subprocess.run([sys.executable, "-c", _KILLED_MID_TABLE, out], env={**os.environ, "PYTHONPATH": str(root)},
                            capture_output=True, text=True)
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert _tables(out) == {}, "the table and its first batch were never committed"
    duck = duckdb.connect(out)
    assert copy_table(_shop(), duck, "APP", "ORDERS", 3) == 5, "a rerun copies it whole"
    duck.close()
    assert _tables(out) == {"ORDERS": 5}


def test_a_real_table_named_like_another_with_partial_appended_survives_copying_it(deployment):
    # `X__partial` was the copy's staging name for X; an Oracle table of that name, copied earlier, is a table like any.
    both = {"ORDERS__partial": [_batch(_ORDERS, range(4))], "ORDERS": [_batch(_ORDERS, range(5))]}
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["ORDERS__partial"]}]))
    assert deployment(_Oracle(both)) == 0
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["ORDERS"]}]))
    assert deployment(_Oracle(both)) == 0
    assert _tables(deployment.out) == {"ORDERS__partial": 4, "ORDERS": 5}


def test_a_stopped_replace_keeps_the_table_it_was_replacing(deployment):
    assert deployment(_shop()) == 0
    with pytest.raises(RuntimeError, match="connection was lost"):
        deployment(_shop(fail_on=("ORDERS", 1)), "--replace")
    assert _tables(deployment.out)["ORDERS"] == 5


def test_each_table_is_copied_in_its_own_read_only_transaction(deployment):
    fake = _shop()
    assert deployment(fake) == 0
    log = fake.log
    for table in ("ORDERS", "CUSTOMER"):
        batches = [i for i, entry in enumerate(log) if isinstance(entry, tuple) and entry[1] == table]
        first, last = batches[0], batches[-1]
        # Adjacent, not merely earlier and later: one transaction around both tables would put the other table's
        # batches next to these, with no ROLLBACK and SET between them.
        assert log[first - 2:first] == ["ROLLBACK", "SET TRANSACTION READ ONLY"], f"{table} starts its own"
        assert log[last + 1] == "ROLLBACK", f"{table}'s transaction ends right after its last batch"
        between = log[first:last + 1]
        assert all(isinstance(e, tuple) and e[1] == table for e in between), "no other read inside it"
    assert not [e for e in log if isinstance(e, str) and e.split()[0].upper() not in
                ("SET", "ROLLBACK", "SELECT", "CLOSE")], "nothing but SELECT reaches Oracle"


def test_a_copied_table_is_skipped_unless_replace(deployment, capsys):
    assert deployment(_shop()) == 0
    assert "2 table(s) in the file: 2 copied, 0 already there" in capsys.readouterr().out
    again = _shop()
    assert deployment(again) == 0
    assert again.fetched == [], "both tables were in the file"
    out = capsys.readouterr().out
    assert "already in" in out
    assert "2 table(s) in the file: 0 copied, 2 already there" in out, "the summary counts what this run copied"
    replaced = _shop()
    assert deployment(replaced, "--replace") == 0
    assert replaced.fetched == ["ORDERS", "CUSTOMER"]
    assert _tables(deployment.out) == {"ORDERS": 5, "CUSTOMER": 2}, "replaced, not appended"


def _records(path: str) -> tuple[list[tuple], list[tuple]]:
    con = duckdb.connect(path, read_only=True)
    try:
        return (con.execute("SELECT * FROM mnemiq.copy_source").fetchall(),
                con.execute("SELECT table_name, row_count, completed_at IS NOT NULL FROM mnemiq.copied_tables "
                            "ORDER BY table_name").fetchall())
    finally:
        con.close()


def test_the_source_and_each_completed_table_are_recorded_in_the_file(deployment):
    assert deployment(_shop()) == 0
    assert _records(deployment.out) == ([("db.example:1521/APP", "APP", "READER")],
                                        [("CUSTOMER", 2, True), ("ORDERS", 5, True)])
    assert deployment(_shop(), "--replace") == 0
    assert _records(deployment.out)[1] == [("CUSTOMER", 2, True), ("ORDERS", 5, True)], "replaced, not doubled"
    with open(deployment.out, "rb") as fh:
        assert b"not-a-real-password" not in fh.read(), "no password is recorded"
    assert sorted(DuckDBAdapter.duckdb(deployment.out).introspect()) == ["CUSTOMER", "ORDERS"], "records are not tables"


@pytest.mark.parametrize("changed", ["target", "schema", "user"])
def test_a_file_holding_another_sources_copy_is_refused_before_any_read(deployment, monkeypatch, capsys, changed):
    assert deployment(_shop()) == 0
    entry = open(deployment.out + ".source.json").read()
    if changed == "user":
        monkeypatch.setenv("MNEMIQ_ORACLE_USER", "OTHER_READER")
    else:
        deployment.manifest.write_text(json.dumps([{**_SOURCE, changed: "OTHER" if changed == "schema" else
                                                    "db.example:1521/OTHER"}]))
    deployment.connects.clear()
    other = _shop()
    assert deployment(other, "--replace") == 2, "not even --replace mixes two sources in one file"
    assert deployment.connects == [] and other.log == [], "refused before Oracle is reached"
    assert "another source" in capsys.readouterr().err
    assert _tables(deployment.out) == {"ORDERS": 5, "CUSTOMER": 2}
    assert open(deployment.out + ".source.json").read() == entry, "nothing in the file or beside it changed"


def test_a_file_holding_tables_but_no_recorded_source_is_refused(deployment, capsys):
    con = duckdb.connect(deployment.out)
    con.execute('CREATE TABLE main."ORDERS" AS SELECT 1 AS "ID"')
    con.close()
    fake = _shop()
    assert deployment(fake) == 2
    assert deployment.connects == [] and fake.log == []
    assert "no record of the source" in capsys.readouterr().err


def test_a_table_is_skipped_only_when_recorded_complete(deployment):
    assert deployment(_shop()) == 0
    con = duckdb.connect(deployment.out)
    # ORDERS present but not recorded complete (as a table made outside the copy would be): copied again, whole.
    con.execute("DELETE FROM mnemiq.copied_tables WHERE table_name = 'ORDERS'")
    con.execute('DELETE FROM main."ORDERS" WHERE "ID" > 0')
    con.close()
    again = _shop()
    assert deployment(again) == 0
    assert again.fetched == ["ORDERS"], "CUSTOMER, recorded complete, is skipped"
    assert _tables(deployment.out) == {"ORDERS": 5, "CUSTOMER": 2}


def test_a_selected_name_equal_to_a_copied_one_without_case_is_refused(deployment, capsys):
    # Patterns match without regard to case, so the two can only meet across runs: the schema changed in between
    # (`ORDERS` dropped, `Orders` created), and DuckDB would take the new table for the copied one.
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["orders"]}]))
    assert deployment(_Oracle({"ORDERS": [_batch(_ORDERS, range(5))]})) == 0
    later = _Oracle({"Orders": [_batch(_ORDERS, range(2))]})
    assert deployment(later) == 2
    assert later.fetched == [] and "'Orders'" in capsys.readouterr().err
    assert _tables(deployment.out) == {"ORDERS": 5}


def test_keys_with_both_ends_copied_are_written_to_the_keys_table(deployment):
    keys = [("ORDERS", "CUSTOMER_ID", "CUSTOMER", "ID", "FK_ORD_CUST", 1),
            ("ORDERS", "STORE_ID", "STORE", "ID", "FK_ORD_STORE", 1)]
    assert deployment(_shop(keys=keys)) == 0
    con = duckdb.connect(deployment.out, read_only=True)
    assert con.execute("SELECT * FROM mnemiq.declared_keys").fetchall() == [keys[0]]
    con.close()
    # The DuckDB adapter reads what the copy wrote: the join between the two, not either end alone.
    assert DuckDBAdapter.duckdb(deployment.out).foreign_keys() == [keys[0][:5]]


def test_a_composite_keys_declared_order_survives_the_copy(deployment):
    from mnemiq.catalog import introspect
    from mnemiq.enrichment.joins import relationships_from_foreign_keys

    store = pa.schema([("ZONE", pa.large_string()), ("AREA", pa.large_string())])
    lines = pa.schema([("ZONE", pa.large_string()), ("AREA", pa.large_string())])
    fake = _Oracle({"LINES": [lines.empty_table()], "STORE": [store.empty_table()]},
                   keys=[("LINES", "ZONE", "STORE", "ZONE", "FK_STORE", 1),
                         ("LINES", "AREA", "STORE", "AREA", "FK_STORE", 2)])  # as Oracle orders them, by position
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["LINES", "STORE"]}]))
    assert deployment(fake) == 0
    adapter = DuckDBAdapter.duckdb(deployment.out)
    oracle = type("OracleOrder", (), {"foreign_keys": lambda self: [k[:5] for k in fake.keys]})()
    catalog = introspect(adapter)
    assert [r.id for r in relationships_from_foreign_keys(adapter, catalog)] == \
        [r.id for r in relationships_from_foreign_keys(oracle, catalog)] == ["LINES.ZONE+AREA->STORE"]


def test_a_key_write_that_fails_part_way_leaves_the_previous_key_set_whole(tmp_path):
    # Half of a composite key reads as a valid one-column join. DuckDB's executemany keeps the rows before a failing
    # one (measured), so without one transaction a failure on the second pair published the first alone.
    duck = duckdb.connect(str(tmp_path / "k.duckdb"))
    before = [("LINES", "ZONE", "STORE", "ZONE", "FK_STORE", 1), ("LINES", "AREA", "STORE", "AREA", "FK_STORE", 2)]
    copy._write_keys(duck, before)
    after = [("LINES", "ZONE", "SITE", "ZONE", "FK_SITE", 1), ("LINES", "AREA", "SITE", "AREA", "FK_SITE", "second")]
    with pytest.raises(duckdb.ConversionException):
        copy._write_keys(duck, after)
    assert duck.execute("SELECT * FROM mnemiq.declared_keys ORDER BY position").fetchall() == before


def test_keys_are_read_with_their_position(deployment):
    fake = _shop()
    deployment(fake)
    statement = next(e for e in fake.log if isinstance(e, str) and "all_constraints" in e.lower())
    assert "cc.column_name, rc.table_name, rcc.column_name, c.constraint_name, cc.position" in statement


def test_declared_key_rows_keeps_keys_whose_both_ends_are_copied():
    rows = [("A", "X", "B", "Y", "K1"), ("A", "Z", "C", "W", "K2"), ("C", "W", "A", "X", "K3")]
    assert declared_key_rows(rows, {"A", "B"}) == [rows[0]]


def test_keys_are_read_only_within_the_owner(deployment):
    fake = _shop()
    deployment(fake)
    statement = next(e for e in fake.log if isinstance(e, str) and "all_constraints" in e.lower())
    assert "r_owner = c.owner" in " ".join(statement.lower().split()), "a key into another schema is left out"


def test_comments_reach_the_file_where_the_duckdb_adapter_reads_them(deployment):
    fake = _shop(table_comments={"ORDERS": "One row per order", "OTHER": "not copied"},
                 column_comments={("ORDERS", "Status"): "O open, C closed", ("ORDERS", "GONE"): "no such column"})
    assert deployment(fake) == 0
    assert DuckDBAdapter.duckdb(deployment.out).comments(["ORDERS", "CUSTOMER"]) == (
        {"ORDERS": "One row per order"}, {("ORDERS", "Status"): "O open, C closed"})


def test_names_that_differ_only_in_case_are_refused_before_any_row(deployment, capsys):
    tables = _Oracle({"Orders": [_batch(_ORDERS, range(1))], "ORDERS": [_batch(_ORDERS, range(1))]})
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["orders"]}]))
    assert deployment(tables) == 2
    assert tables.fetched == []
    err = capsys.readouterr().err
    assert "'Orders'" in err and "'ORDERS'" in err
    columns = _Oracle({"ORDERS": [pa.schema([("STATUS", pa.int64()), ("Status", pa.int64())]).empty_table()]})
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["ORDERS"]}]))
    assert deployment(columns) == 2
    assert columns.fetched == []
    err = capsys.readouterr().err
    assert "'STATUS'" in err and "'Status'" in err


def test_a_table_and_one_named_like_its_old_staging_copy_are_both_copied(deployment):
    fake = _Oracle({"ORDERS": [_batch(_ORDERS, range(1))], "ORDERS__PARTIAL": [_batch(_ORDERS, range(2))]})
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["ORDERS*"]}]))
    assert deployment(fake) == 0
    assert _tables(deployment.out) == {"ORDERS": 1, "ORDERS__PARTIAL": 2}


def test_a_pattern_matching_no_table_is_refused(deployment, capsys):
    deployment.manifest.write_text(json.dumps([{**_SOURCE, "tables": ["ORDERS", "ORDRES"]}]))
    fake = _shop()
    assert deployment(fake) == 2
    assert fake.fetched == [] and "ORDRES" in capsys.readouterr().err


def test_the_copys_source_entry_matches_the_oracle_source(deployment, capsys):
    spec = SourceSpec(id="plant", kind="oracle", target="db.example:1521/APP", catalog="plant", schema="APP",
                      tables=("ORDERS", "CUSTOMER"), profile="statistics", exact_tables=("ORDERS",))
    assert source_entry(spec, "/data/copy.duckdb", ["ORDERS", "CUSTOMER"]) == {
        "id": "plant", "kind": "duckdb", "target": "/data/copy.duckdb", "catalog": "plant", "schema": "main",
        "tables": ["ORDERS", "CUSTOMER"]}
    assert deployment(_shop()) == 0
    with open(deployment.out + ".source.json") as fh:
        written = json.load(fh)
    assert written == [{"id": "plant", "kind": "duckdb", "target": deployment.out, "catalog": "plant",
                        "schema": "main", "tables": ["ORDERS", "CUSTOMER"]}]
    out = capsys.readouterr().out
    assert "mnemiq enrich" in out and "mnemiq build" in out and deployment.out + ".source.json" in out


def test_the_written_entry_is_a_manifest_mnemiq_reads(deployment, monkeypatch):
    from mnemiq.config import Settings

    assert deployment(_shop()) == 0
    monkeypatch.setenv("MNEMIQ_SOURCES_PATH", deployment.out + ".source.json")
    (spec,) = Settings().source_specs()
    assert (spec.kind, spec.target, spec.schema, spec.tables) == ("duckdb", deployment.out, "main",
                                                                  ("ORDERS", "CUSTOMER"))


def test_the_only_oracle_source_is_chosen_and_another_is_refused(deployment, tmp_path):
    other = {"id": "local", "kind": "duckdb", "target": str(tmp_path / "x.duckdb"), "catalog": "local",
             "schema": "main"}
    deployment.manifest.write_text(json.dumps([other, _SOURCE]))
    assert deployment(_shop()) == 0
    assert deployment(_shop(), "--source-id", "local") == 2
    deployment.manifest.write_text(json.dumps([_SOURCE, {**_SOURCE, "id": "plant2"}]))
    assert deployment(_shop()) == 2, "two Oracle sources and no --source-id"


def test_missing_credentials_are_refused_before_connecting(deployment, monkeypatch, capsys):
    monkeypatch.delenv("MNEMIQ_ORACLE_PASSWORD")
    fake = _shop()
    assert deployment(fake) == 2
    assert fake.log == [] and "MNEMIQ_ORACLE_PASSWORD" in capsys.readouterr().err


def test_the_file_is_opened_through_mnemiqs_door_with_the_deployments_local_only(deployment, monkeypatch):
    seen: list[bool] = []
    real = copy.connect_duckdb

    def door(database, *, local_only):
        seen.append(local_only)
        return real(database, local_only=local_only)

    monkeypatch.setattr(copy, "connect_duckdb", door)
    monkeypatch.setenv("MNEMIQ_LOCAL_ONLY", "1")
    assert deployment(_shop()) == 0
    assert seen == [True]


def test_the_password_is_never_printed(deployment, capsys):
    deployment(_shop())
    out = capsys.readouterr()
    assert "not-a-real-password" not in out.out + out.err
