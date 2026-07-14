import duckdb

from mnemiq.catalog import ColumnInfo, TableInfo
from mnemiq.enrichment.joins import infer_relationships


class _DuckAdapter:
    """Minimal SourceAdapter over an in-memory DuckDB -- controlled data, no Postgres."""

    def __init__(self, con):
        self._con = con

    def execute(self, sql):
        return self._con.execute(sql).fetchall()


def _parent_child(child_rows: str):
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" INTEGER, "name" TEXT)')
    con.execute("INSERT INTO parent VALUES (1,'a'),(2,'b'),(3,'c')")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" INTEGER)')
    con.execute(f"INSERT INTO child VALUES {child_rows}")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "INTEGER"), ColumnInfo("name", "TEXT")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "INTEGER")]),
    ]
    return _DuckAdapter(con), catalog


def test_infers_child_to_parent_fk():
    adapter, catalog = _parent_child("(10,1),(11,1),(12,2)")
    rels = infer_relationships(adapter, catalog)

    rel = next(r for r in rels if r.from_ == "child" and r.to == "parent")
    assert rel.cardinality == "many_to_one"
    assert rel.join_keys[0].left == "parent_id"
    assert rel.join_keys[0].right == "parent_id"


def test_no_relationship_when_inclusion_fails():
    adapter, catalog = _parent_child("(10,99)")  # 99 is not a parent
    rels = infer_relationships(adapter, catalog)
    assert not any(r.from_ == "child" and r.to == "parent" for r in rels)


def test_nullable_fk_still_infers():
    # a NULL foreign key is "no parent", not a broken reference
    adapter, catalog = _parent_child("(10,1),(11,NULL)")
    rels = infer_relationships(adapter, catalog)
    assert any(r.from_ == "child" and r.to == "parent" for r in rels)


def test_parent_must_be_a_key():
    # duplicate "keys" in the parent -> not a key -> no relationship
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" INTEGER)')
    con.execute("INSERT INTO parent VALUES (1),(1)")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" INTEGER)')
    con.execute("INSERT INTO child VALUES (10,1)")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "INTEGER")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "INTEGER")]),
    ]
    assert infer_relationships(_DuckAdapter(con), catalog) == []


def test_direction_comes_from_naming_not_uniqueness():
    # both sides are unique here; only the column name says which one is the parent
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" INTEGER)')
    con.execute("INSERT INTO parent VALUES (1),(2)")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" INTEGER)')
    con.execute("INSERT INTO child VALUES (10,1)")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "INTEGER")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "INTEGER")]),
    ]
    rels = infer_relationships(_DuckAdapter(con), catalog)
    assert [(r.from_, r.to) for r in rels] == [("child", "parent")]  # never parent -> child


def test_cross_type_join_keys_compare_as_text():
    # ACME mixes DDL-typed and CSV-inferred (all-text) tables; INTEGER vs TEXT must still match
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" INTEGER)')
    con.execute("INSERT INTO parent VALUES (1),(2)")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" TEXT)')
    con.execute("INSERT INTO child VALUES (10,'1'),(11,'2')")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "INTEGER")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "TEXT")]),
    ]
    rels = infer_relationships(_DuckAdapter(con), catalog)
    assert any(r.from_ == "child" and r.to == "parent" for r in rels)


# --- Plan 11: declared foreign keys ---------------------------------------------------
import sqlite3  # noqa: E402

from mnemiq.adapters.sqlite import SQLiteAdapter  # noqa: E402
from mnemiq.catalog import introspect  # noqa: E402
from mnemiq.enrichment.joins import build_relationships, relationships_from_foreign_keys  # noqa: E402


def _fk_src(tmp_path):
    # parent<-child by DECLARED fk (differently-named key);
    # widget.gadget_identifier names table `gadget` by CONVENTION but declares no fk
    path = tmp_path / "j.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE child (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES parent(id));
        CREATE TABLE gadget (gadget_identifier INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE widget (id INTEGER PRIMARY KEY, gadget_identifier INTEGER);
        INSERT INTO parent VALUES (1,'a'),(2,'b');
        INSERT INTO child VALUES (10,1),(11,2);
        INSERT INTO gadget VALUES (100,'x'),(101,'y');
        INSERT INTO widget VALUES (1,100),(2,101);
        """
    )
    con.commit()
    con.close()
    return SQLiteAdapter(str(path))


def test_from_foreign_keys_builds_relationship_with_differing_keys(tmp_path):
    adapter = _fk_src(tmp_path)
    rels = relationships_from_foreign_keys(adapter, introspect(adapter))

    child_rel = next(r for r in rels if r.from_ == "child")
    assert child_rel.to == "parent"
    assert child_rel.cardinality == "many_to_one"
    assert (child_rel.join_keys[0].left, child_rel.join_keys[0].right) == ("pid", "id")


def test_build_relationships_is_declared_first_with_inference_fallback(tmp_path):
    adapter = _fk_src(tmp_path)
    rels = build_relationships(adapter, introspect(adapter))
    by_from = {r.from_: r for r in rels}

    assert by_from["child"].to == "parent"  # declared
    assert by_from["widget"].to == "gadget"  # inferred fallback (no declared fk)


def test_declared_fk_table_is_not_also_inferred(tmp_path):
    adapter = _fk_src(tmp_path)
    rels = build_relationships(adapter, introspect(adapter))
    assert sum(1 for r in rels if r.from_ == "child") == 1
