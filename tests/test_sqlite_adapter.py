import sqlite3

import pyarrow as pa
import pytest

from mnemiq.adapters.sqlite import SQLiteAdapter


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE "order" (id INTEGER, customer_id INTEGER, amount REAL);
        INSERT INTO customer VALUES (1,'Ada','NYC'),(2,'Bo',NULL);
        INSERT INTO "order" VALUES (10,1,4.5),(11,1,NULL),(12,2,7.0);
        """
    )
    con.commit()
    con.close()
    return str(path)


def test_introspect_lists_user_tables(db):
    assert sorted(SQLiteAdapter(db).introspect()) == ["customer", "order"]


def test_list_columns_gives_table_column_type(db):
    cols = SQLiteAdapter(db).list_columns()
    assert ("customer", "name", "TEXT") in cols
    assert ("order", "amount", "REAL") in cols


def test_execute_returns_tuples(db):
    rows = SQLiteAdapter(db).execute("SELECT count(*) FROM customer")
    assert rows == [(2,)]


def test_execute_arrow_returns_a_typed_table(db):
    table = SQLiteAdapter(db).execute_arrow('SELECT amount FROM "order" ORDER BY amount')
    assert isinstance(table, pa.Table)
    assert table.schema.names == ["amount"]
    assert table.num_rows == 3  # NULLs included


def test_execute_arrow_on_empty_result_keeps_the_columns(db):
    table = SQLiteAdapter(db).execute_arrow("SELECT id, name FROM customer WHERE id = -1")
    assert table.schema.names == ["id", "name"]
    assert table.num_rows == 0


def test_duplicate_output_column_names_are_preserved(db):
    # BIRD candidates can alias two columns the same; a dict-keyed table would collapse them
    table = SQLiteAdapter(db).execute_arrow('SELECT id AS n, customer_id AS n FROM "order"')
    assert table.schema.names == ["n", "n"]
    assert table.num_columns == 2


def test_the_database_is_read_only(db):
    with pytest.raises(Exception):
        SQLiteAdapter(db).execute("CREATE TABLE evil (x INT)")


def test_foreign_keys_reads_declared_constraints(tmp_path):
    path = tmp_path / "fk.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE child (
            id INTEGER PRIMARY KEY,
            pid INTEGER REFERENCES parent(id),
            note TEXT
        );
        """
    )
    con.commit()
    con.close()

    fks = SQLiteAdapter(str(path)).foreign_keys()
    assert ("child", "pid", "parent", "id") in fks


def test_foreign_keys_empty_when_none_declared(db):
    # the `db` fixture (customer/order) declares no FKs
    assert SQLiteAdapter(db).foreign_keys() == []


def test_a_runaway_query_is_interrupted(db):
    # a recursive CTE that never stops; the interrupt must end it
    runaway = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT count(*) FROM r"
    with pytest.raises(Exception) as exc:
        SQLiteAdapter(db).execute_arrow(runaway, timeout_s=0.5)
    assert "interrupt" in str(exc.value).lower() or "timed out" in str(exc.value).lower()
