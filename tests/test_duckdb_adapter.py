import inspect
import sqlite3


def _make_sqlite(tmp_path):
    p = str(tmp_path / "t.sqlite")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE person (id INTEGER, name TEXT, born DATE)")
    con.execute("INSERT INTO person VALUES (1, 'A', '1990-05-01'), (2, 'B', '1985-12-31')")
    con.commit()
    con.close()
    return p


def test_sqlite_factory_introspects_and_runs_duckdb_functions(tmp_path):
    from mnemiq.adapters.duckdb import DuckDBAdapter

    a = DuckDBAdapter.sqlite(_make_sqlite(tmp_path))
    assert a.dialect == "duckdb"
    assert "person" in a.introspect()
    cols = {c[1] for c in a.list_columns()}
    assert {"id", "name", "born"} <= cols
    # DuckDB's sqlite scanner does not expose declared FKs -- documented limitation
    assert a.foreign_keys() == []
    # a DuckDB-native function over a sqlite-backed column: the whole point of the change
    assert a.execute("SELECT YEAR(born) FROM person WHERE id = 1")[0][0] == 1990
    tbl = a.execute_arrow("SELECT count(*) AS n FROM person")
    assert tbl.column("n")[0].as_py() == 2


def test_duckdb_postgres_adapter_is_a_thin_compat_subclass():
    from mnemiq.adapters.duckdb import DuckDBAdapter
    from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter

    assert issubclass(DuckDBPostgresAdapter, DuckDBAdapter)
    assert DuckDBPostgresAdapter.dialect == "duckdb"
    params = list(inspect.signature(DuckDBPostgresAdapter.__init__).parameters)
    assert params == ["self", "dsn", "schema", "read_only"]  # + read_only for the write path
