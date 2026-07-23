import os

import pytest

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.catalog import introspect
from mnemiq.enrichment.profiling import profile_column, profile_table

pytestmark = pytest.mark.integration

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"


def _adapter():
    return DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))


def _table(adapter, name):
    return next(t for t in introspect(adapter) if t.name == name)


def test_profile_column_counts():
    s = profile_column(_adapter(), "fireclaim", "fireplace")
    assert s.row_count == 820
    assert s.distinct_count == 2  # yes / no
    assert s.null_count == 0
    assert {v for v, _ in s.top_k} == {"yes", "no"}


def test_profile_column_all_null():
    # Party.csv populates only the identifier; every other column is empty
    s = profile_column(_adapter(), "party", "party_type_code")
    assert s.row_count == 30
    assert s.null_count == 30
    assert s.distinct_count == 0
    assert s.top_k == []  # NULL is an absence, never a code


def test_profile_table_covers_every_column_and_finds_codes():
    adapter = _adapter()
    fireclaim = _table(adapter, "fireclaim")
    stats = profile_table(adapter, fireclaim)

    assert {s.column for s in stats} == {c.name for c in fireclaim.columns}
    assert all(s.row_count == 820 for s in stats)

    code = next(s for s in stats if s.column == "fireplace")
    assert code.top_k, "low-cardinality column should have observed values"
    assert all(isinstance(v, tuple) and len(v) == 2 for v in code.top_k)


def test_names_are_never_harvested():
    # person.first_name holds 3 distinct values in the sample -- it *looks* like a coded
    # vocabulary. Harvesting it would copy real people into the snapshot and into a prompt.
    adapter = _adapter()
    stats = {s.column: s for s in profile_table(adapter, _table(adapter, "person"))}

    for column in ("first_name", "last_name", "middle_name", "birth_date"):
        assert stats[column].top_k == [], f"{column} is personal data, not a vocabulary"

    # the counts are still computed -- we suppress the values, not the profiling
    assert stats["first_name"].row_count == 29


def test_profile_table_skips_keys_and_nulls():
    adapter = _adapter()
    stats = {s.column: s for s in profile_table(adapter, _table(adapter, "policy_amount"))}

    # a foreign key is low-cardinality in a small sample, but it is not a vocabulary
    assert stats["policy_identifier"].distinct_count == 2
    assert stats["policy_identifier"].top_k == []

    # a real coded column still gets harvested, and NULLs are not among its codes
    assert stats["amount_type_code"].top_k == [("Year", 6)]


def test_declared_fk_column_is_not_harvested_as_a_vocabulary(tmp_path):
    import sqlite3

    from mnemiq.adapters.sqlite import SQLiteAdapter
    from mnemiq.catalog import introspect
    from mnemiq.enrichment.profiling import profile_table

    path = tmp_path / "p.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE region (code TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE sale (id INTEGER PRIMARY KEY, rc TEXT);
        INSERT INTO region VALUES ('N','North'),('S','South');
        INSERT INTO sale VALUES (1,'N'),(2,'S'),(3,'N');
        """
    )
    con.commit()
    con.close()
    adapter = SQLiteAdapter(str(path))
    sale = next(t for t in introspect(adapter) if t.name == "sale")

    # rc has 2 distinct values in 3 rows -> looks like a vocabulary by name alone
    without = {s.column: s for s in profile_table(adapter, sale)}
    assert without["rc"].top_k  # harvested when we don't know it's a key

    with_key = {s.column: s for s in profile_table(adapter, sale, key_columns={"rc"})}
    assert with_key["rc"].top_k == []  # a declared FK column is a key, not a vocabulary


def test_coded_column_harvests_all_distinct_up_to_the_cap(tmp_path):
    # A coded column with 12 distinct values (> the old top-10 slice) must surface ALL of them:
    # it already qualifies as a vocabulary (<= code_max_distinct), so a rare code must not be
    # dropped -- else a dictionary can never ground it and the model never sees it.
    import sqlite3

    from mnemiq.adapters.sqlite import SQLiteAdapter
    from mnemiq.catalog import introspect
    from mnemiq.enrichment.profiling import profile_table

    codes = list("abcdefghijkl")  # 12 distinct
    rows = [(i + 1, c) for i, c in enumerate(codes)] + [(13, "a"), (14, "b")]  # distinct 12 < rows 14
    path = tmp_path / "t.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, kind TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", rows)
    con.commit()
    con.close()

    adapter = SQLiteAdapter(str(path))
    t = next(x for x in introspect(adapter) if x.name == "t")
    stats = {s.column: s for s in profile_table(adapter, t)}
    assert {v for v, _ in stats["kind"].top_k} == set(codes)  # all 12, not just the top 10
