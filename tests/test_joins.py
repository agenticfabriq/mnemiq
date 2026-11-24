import duckdb
import pytest

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


def _numeric_keys(parent_type: str, child_type: str, child_rows: str):
    con = duckdb.connect()
    con.execute(f'CREATE TABLE parent ("parent_id" {parent_type})')
    con.execute("INSERT INTO parent VALUES (1),(2)")
    con.execute(f'CREATE TABLE child ("child_id" INTEGER, "parent_id" {child_type})')
    con.execute(f"INSERT INTO child VALUES {child_rows}")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", parent_type)]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", child_type)]),
    ]
    return infer_relationships(_DuckAdapter(con), catalog)


@pytest.mark.parametrize(("parent_type", "child_type"), [
    ("DECIMAL(19,0)", "DECIMAL(19,2)"), ("BIGINT", "DECIMAL(19,2)"), ("DECIMAL(12,2)", "DOUBLE"),
    ("INTEGER", "DOUBLE"), ("BIGINT", "DOUBLE")])
def test_numeric_join_keys_of_different_types_compare_as_numbers(parent_type, child_type):
    # A DuckDB copy of Oracle lands each NUMBER as BIGINT, DECIMAL(p, s) or DOUBLE by its declaration, where as text
    # one number is 1, 1.00 or 1.0; Oracle, and the join the relationship describes, compare them as numbers.
    rels = _numeric_keys(parent_type, child_type, "(10, 1), (11, 2)")
    assert [(r.from_, r.to) for r in rels] == [("child", "parent")]


def test_a_double_key_does_not_vouch_for_exact_keys_it_cannot_tell_apart():
    # Codex's second review of the exact-number copy: compared as a DOUBLE, two DECIMAL(38, 0) keys past 2^53 are one
    # number, so one child row joins both parents, though each parent key is unique in its own type.
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" DECIMAL(38,0))')
    con.execute("INSERT INTO parent VALUES (9007199254740992), (9007199254740993)")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" DOUBLE)')
    con.execute("INSERT INTO child VALUES (10, 9007199254740992)")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "DECIMAL(38,0)")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "DOUBLE")]),
    ]
    assert con.execute('SELECT count(*) FROM child c JOIN parent p ON p."parent_id" = c."parent_id"').fetchone() == (2,)
    assert infer_relationships(_DuckAdapter(con), catalog) == []


def test_a_text_parent_key_still_matches_a_numeric_child():
    # The mirror of INTEGER against TEXT: the parent from an all-text CSV, the child typed. Either side text is text:
    # natively, DuckDB would cast the parent's 'X1' to a number and raise, and the relationship would go unchecked.
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" VARCHAR)')
    con.execute("INSERT INTO parent VALUES ('1'), ('2'), ('X1')")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" DECIMAL(19,0))')
    con.execute("INSERT INTO child VALUES (10, 1), (11, 2)")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "VARCHAR")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "DECIMAL(19,0)")]),
    ]
    assert [(r.from_, r.to) for r in infer_relationships(_DuckAdapter(con), catalog)] == [("child", "parent")]


def test_exact_keys_a_source_compares_in_floating_point_do_not_vouch_for_a_join_either():
    # Codex's third review: DuckDB compares HUGEINT with UHUGEINT as DOUBLE (its EXPLAIN), so two exact types can merge
    # keys too. Which comparisons a source widens is the source's to say, so the check asks it, under the predicate.
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" HUGEINT)')
    con.execute("INSERT INTO parent VALUES (18014398509481984), (18014398509481985)")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" UHUGEINT)')
    con.execute("INSERT INTO child VALUES (10, 18014398509481986)")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "HUGEINT")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "UHUGEINT")]),
    ]
    assert con.execute('SELECT count(*) FROM child c JOIN parent p ON p."parent_id" = c."parent_id"').fetchone() == (2,)
    assert infer_relationships(_DuckAdapter(con), catalog) == []


def test_child_rows_sharing_a_key_are_not_a_fan_out():
    # Many child rows to one parent is the many-to-one itself: the check counts each child value once.
    assert [(r.from_, r.to) for r in _numeric_keys("DECIMAL(12,2)", "DOUBLE", "(10, 1), (11, 1), (12, 1), (13, 2)")] == [
        ("child", "parent")]


def test_numeric_join_keys_still_refuse_an_orphan():
    assert _numeric_keys("DECIMAL(19,0)", "DECIMAL(19,2)", "(10, 1), (11, 2.5)") == []


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


def test_independent_fks_to_same_parent_are_separate_relationships(tmp_path):
    # superhero has eye/hair/skin colour_id -> colour, THREE independent FKs, not a composite.
    path = tmp_path / "fk2.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE colour (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE hero (
            id INTEGER PRIMARY KEY,
            eye_id INTEGER REFERENCES colour(id),
            hair_id INTEGER REFERENCES colour(id)
        );
        INSERT INTO colour VALUES (1,'blue'),(2,'red');
        INSERT INTO hero VALUES (1,1,2);
        """
    )
    con.commit()
    con.close()
    adapter = SQLiteAdapter(str(path))
    rels = [r for r in relationships_from_foreign_keys(adapter, introspect(adapter)) if r.from_ == "hero"]

    assert len(rels) == 2  # not one merged "composite"
    left_cols = sorted(r.join_keys[0].left for r in rels)
    assert left_cols == ["eye_id", "hair_id"]
    assert all(len(r.join_keys) == 1 for r in rels)


def test_a_true_composite_fk_is_one_relationship_with_two_keys(tmp_path):
    path = tmp_path / "fk3.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE parent (a INTEGER, b INTEGER, PRIMARY KEY (a, b));
        CREATE TABLE child (
            id INTEGER PRIMARY KEY, pa INTEGER, pb INTEGER,
            FOREIGN KEY (pa, pb) REFERENCES parent(a, b)
        );
        """
    )
    con.commit()
    con.close()
    adapter = SQLiteAdapter(str(path))
    rels = [r for r in relationships_from_foreign_keys(adapter, introspect(adapter)) if r.from_ == "child"]

    assert len(rels) == 1
    assert {(k.left, k.right) for k in rels[0].join_keys} == {("pa", "a"), ("pb", "b")}


def test_keys_of_one_name_on_two_tables_are_two_relationships():
    """A key's name is unique on its table, not in its schema (Postgres), so `fk_customer` on `invoices` and on `orders`
    are two keys: two relationships, each with its own table's column, never one merged across both tables."""
    from mnemiq.catalog import TableInfo

    class _Keys:
        def foreign_keys(self):
            return [("invoices", "cust", "customers", "id", "fk_customer"),
                    ("orders", "customer_id", "customers", "id", "fk_customer")]

    catalog = [TableInfo(name) for name in ("customers", "invoices", "orders")]
    rels = relationships_from_foreign_keys(_Keys(), catalog)

    assert [(r.id, r.from_, r.to, [(k.left, k.right) for k in r.join_keys]) for r in rels] == [
        ("invoices.cust->customers", "invoices", "customers", [("cust", "id")]),
        ("orders.customer_id->customers", "orders", "customers", [("customer_id", "id")]),
    ]


class _KeyRecorder:
    """Records every statement and answers as a source whose parent keys are unique and whose child values all exist:
    `count(*), count(DISTINCT), count()` gives 10, 10, 10 and the orphan count gives 0. `fail_on` names a column whose
    orphan query raises, as Oracle raised ORA-00906 on `CAST(... AS VARCHAR)` (measured on 23.26, 2026-11-19)."""

    def __init__(self, dialect: str | None, fks=(), fail_on: str | None = None, catalog=(), fan_out: int = 0):
        if dialect is not None:
            self.dialect = dialect
        self._fan_out = fan_out
        self.sql: list[str] = []
        self._fks = list(fks)
        self._fail_on = fail_on
        self._catalog = list(catalog)

    def foreign_keys(self):
        return self._fks

    def list_columns(self):
        return [(t.name, c.name, c.data_type) for t in self._catalog for c in t.columns]

    def view_definitions(self):
        return []

    def execute(self, sql):
        self.sql.append(sql)
        if "HAVING count(*) > 1" in sql:  # child values that meet two parents
            return [(self._fan_out,)]
        if "NOT EXISTS" in sql:
            if self._fail_on and f'"{self._fail_on}"' in sql:
                raise RuntimeError("ORA-00942: table or view does not exist")
            return [(0,)]
        # every count a profile asks for is 10: each column unique, none null
        return [tuple([10] + [10, 10] * sql.count("count(DISTINCT"))]


def _invoices() -> list[TableInfo]:
    return [
        TableInfo("customer", [ColumnInfo("customer_id", "NUMBER")]),
        TableInfo("invoice", [ColumnInfo("invoice_id", "NUMBER"), ColumnInfo("customer_id", "NUMBER")]),
        TableInfo("invoice_line", [ColumnInfo("line_id", "NUMBER"), ColumnInfo("invoice_id", "NUMBER")]),
    ]


def test_orphan_query_compares_natively_on_oracle():
    """Oracle refuses `CAST(x AS VARCHAR)` (ORA-00906), and sqlglot's transpile gives `CAST(x AS CLOB)`, which Oracle
    cannot compare with `=`. Text is the wrong ground anyway: `TO_CHAR` of a DATE follows the session's format, which
    can drop the time and make two different values one. The check compares as the join will, natively."""
    recorder = _KeyRecorder("oracle")
    rels = infer_relationships(recorder, _invoices())
    orphan = next(sql for sql in recorder.sql if "NOT EXISTS" in sql and '"customer_id"' in sql)
    assert "CAST" not in orphan and "TO_CHAR" not in orphan
    assert 'WHERE p."customer_id" = c."customer_id")' in orphan
    assert ("invoice", "customer") in {(r.from_, r.to) for r in rels}


def test_orphan_query_on_duckdb_compares_as_text_unless_both_keys_are_numbers():
    def orphan_check(child_key_type: str) -> str:
        catalog = _invoices()
        catalog[1] = TableInfo("invoice", [ColumnInfo("invoice_id", "NUMBER"), ColumnInfo("customer_id", child_key_type)])
        recorder = _KeyRecorder(None)
        infer_relationships(recorder, catalog)
        return next(sql for sql in recorder.sql if "NOT EXISTS" in sql and '"customer_id"' in sql)

    head = 'SELECT count(*) FROM "invoice" c WHERE c."customer_id" IS NOT NULL AND NOT EXISTS (  SELECT 1 FROM "customer" p '
    assert orphan_check("NUMBER") == head + '  WHERE p."customer_id" = c."customer_id")'
    assert orphan_check("VARCHAR") == head + '  WHERE CAST(p."customer_id" AS VARCHAR) = CAST(c."customer_id" AS VARCHAR))'


def test_every_candidate_is_checked_for_a_fan_out_under_the_join_predicate():
    """The parent is counted unique in its own type and collation, which is not always how the join compares (a DOUBLE
    against DECIMAL(38, 0) past 2^53, DuckDB's HUGEINT against UHUGEINT, a NOCASE collation), so the source is asked,
    for every candidate, whether a child value meets two parents."""
    fanned = _KeyRecorder("oracle", fan_out=1)
    assert ("invoice", "customer") not in {(r.from_, r.to) for r in infer_relationships(fanned, _invoices())}
    clean = _KeyRecorder("oracle")
    assert ("invoice", "customer") in {(r.from_, r.to) for r in infer_relationships(clean, _invoices())}
    checks = [sql for sql in clean.sql if "HAVING count(*) > 1" in sql and '"customer_id"' in sql]
    assert len(checks) == 1 and 'ON p."customer_id" = d.v' in checks[0]


def test_keys_one_collation_makes_equal_do_not_vouch_for_a_join():
    # Codex's fourth review: under NOCASE 'a' and 'A' are two distinct parent values, and a child's 'a' meets both.
    con = duckdb.connect()
    con.execute('CREATE TABLE parent ("parent_id" VARCHAR COLLATE NOCASE)')
    con.execute("INSERT INTO parent VALUES ('a'), ('A')")
    con.execute('CREATE TABLE child ("child_id" INTEGER, "parent_id" VARCHAR COLLATE NOCASE)')
    con.execute("INSERT INTO child VALUES (10, 'a')")
    catalog = [
        TableInfo("parent", [ColumnInfo("parent_id", "VARCHAR")]),
        TableInfo("child", [ColumnInfo("child_id", "INTEGER"), ColumnInfo("parent_id", "VARCHAR")]),
    ]
    assert con.execute('SELECT count(*) FROM child c JOIN parent p ON p."parent_id" = c."parent_id"').fetchone() == (2,)
    assert infer_relationships(_DuckAdapter(con), catalog) == []


def test_a_candidate_that_cannot_be_checked_costs_only_itself(caplog):
    """One candidate's failed check used to raise out of `infer_relationships`, and `enrich_structural` then kept NO
    relationship, the declared ones included, logging nothing. Now that candidate is skipped and said."""
    recorder = _KeyRecorder("oracle", fks=[("invoice_line", "invoice_id", "invoice", "invoice_id", "fk_line")],
                            fail_on="customer_id")
    with caplog.at_level("WARNING"):
        rels = build_relationships(recorder, _invoices())
    assert ("invoice_line", "invoice") in {(r.from_, r.to) for r in rels}
    assert ("invoice", "customer") not in {(r.from_, r.to) for r in rels}
    assert "customer_id" in caplog.text and "ORA-00942" in caplog.text


def test_a_skipped_candidate_is_recorded_on_the_snapshot():
    """A warning alone left an incomplete inference looking like a complete one: the snapshot's job says `partial` and
    names what was not checked."""
    from mnemiq.enrichment.pipeline import enrich_structural

    recorder = _KeyRecorder("oracle", fks=[("invoice_line", "invoice_id", "invoice", "invoice_id", "fk_line")],
                            fail_on="customer_id", catalog=_invoices())
    snap = enrich_structural(recorder, "src")
    job = next(j for j in snap.jobs if j.id == "infer:relationships")
    assert job.status == "partial"
    assert "invoice.customer_id -> customer" in job.detail and "ORA-00942" in job.detail
    assert ("invoice_line", "invoice") in {(r.from_, r.to) for r in snap.relationships}


def test_a_complete_inference_records_done_with_no_detail():
    from mnemiq.enrichment.pipeline import enrich_structural

    snap = enrich_structural(_KeyRecorder("oracle", catalog=_invoices()), "src")
    job = next(j for j in snap.jobs if j.id == "infer:relationships")
    assert (job.status, job.detail) == ("done", None)
