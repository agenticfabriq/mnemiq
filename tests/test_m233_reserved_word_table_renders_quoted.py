"""A row-filtered table whose name is a reserved word renders quoted, so the governed query
parses and runs on the engine (M233).

The rewrite wraps a filtered table in `(SELECT <cols> FROM <table> WHERE <filter>) AS <alias>`.
When the table and its alias are a word the executing engine reserves (`null`, `true`, `false`),
rendering them bare made the engine refuse the whole query (`syntax error at or near "null"`),
so the table could not be read under a filter at all -- an availability loss that fails closed.
The identifiers must be quoted where the executing engine reserves the word, and left bare
otherwise (quoting a bare alias changes its folded identity on Oracle and Postgres).
"""
from __future__ import annotations

import duckdb
import pytest
import sqlglot

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask


def _governed_sql(table: str, filt: str, visible: dict, dialect: str = "duckdb") -> str:
    quoted = sqlglot.exp.to_identifier(table, quoted=True).sql(dialect=dialect)
    ast = sqlglot.parse_one(f"SELECT id FROM {quoted}", read=dialect)
    out, _ = apply_row_and_mask(
        ast, AccessPolicy(row_filters={table: filt}), visible, dialect=dialect
    )
    return out.sql(dialect=dialect)


@pytest.mark.parametrize("table", ["null", "true", "false"])
def test_a_reserved_word_table_runs_on_duckdb_under_a_filter(table: str) -> None:
    visible = {table: {"id", "flag"}}
    governed = _governed_sql(table, "flag", visible, "duckdb")

    con = duckdb.connect(":memory:")
    q = sqlglot.exp.to_identifier(table, quoted=True).sql(dialect="duckdb")
    con.execute(f"CREATE TABLE {q}(id INTEGER, flag BOOLEAN)")
    con.execute(f"INSERT INTO {q} VALUES (1, true), (2, false)")
    # Before the fix the engine refuses the whole governed query with a syntax error at the
    # bare reserved word, so no row can be read; after it, the filter keeps only row 1.
    rows = con.execute(governed).fetchall()
    assert rows == [(1,)], governed


@pytest.mark.parametrize("table", ["join", "inner", "like"])
def test_a_duckdb_keyword_not_in_the_reserved_category_still_quotes(table: str) -> None:
    # `join`/`inner`/`like` are keywords DuckDB rejects as a bare table name but are NOT in
    # duckdb_keywords()'s 'reserved' category, so the narrow reserved() set missed them
    # (Codex's review). The bare-table set is built from the engine's parser, so they quote.
    visible = {table: {"id", "flag"}}
    governed = _governed_sql(table, "flag", visible, "duckdb")
    con = duckdb.connect(":memory:")
    q = sqlglot.exp.to_identifier(table, quoted=True).sql(dialect="duckdb")
    con.execute(f"CREATE TABLE {q}(id INTEGER, flag BOOLEAN)")
    con.execute(f"INSERT INTO {q} VALUES (1, true), (2, false)")
    assert con.execute(governed).fetchall() == [(1,)], governed


def test_a_plain_table_name_is_left_bare() -> None:
    # Non-reserved names must NOT be quoted: a quoted bare alias folds differently on Oracle and
    # Postgres, so over-quoting would change identity. Here it must stay `claim`, not `"claim"`.
    governed = _governed_sql("claim", "amount > 0", {"claim": {"id", "amount"}}, "duckdb")
    assert '"claim"' not in governed, governed
    assert "claim" in governed, governed


def test_a_dotted_reserved_name_quotes_only_the_reserved_part() -> None:
    # A qualified key is spelled part by part: a federated `cat.null` must quote the reserved
    # `null` but leave the non-reserved catalog `cat` bare -- `cat."null"`, not `"cat.null"`.
    from mnemiq.sql.rls import _derived_table

    out = _derived_table("cat.null", "null", {"id"}, set(), sqlglot.parse_one("flag"),
                         engine="duckdb").sql(dialect="duckdb")
    assert 'cat."null"' in out, out
    assert '"cat"' not in out, out


def test_a_reserved_word_table_runs_on_native_sqlite_under_a_filter() -> None:
    import sqlite3
    import tempfile

    # SQLite's native adapter (eval runners) and a sqlite-decided path: `null` is reserved as a
    # bare table on SQLite, so the governed query must render quoted there too.
    governed = _governed_sql("null", "flag", {"null": {"id", "flag"}}, "sqlite")
    path = f"{tempfile.mkdtemp()}/x.db"
    w = sqlite3.connect(path)
    w.execute('CREATE TABLE "null"(id INTEGER, flag INTEGER)')
    w.execute('INSERT INTO "null" VALUES (1, 1), (2, 0)')
    w.commit()
    w.close()
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    assert ro.execute(governed).fetchall() == [(1,)], governed


def test_reserved_rendering_per_engine() -> None:
    from mnemiq.sql.rls import _derived_table

    flag = sqlglot.parse_one("flag")
    # Postgres reserves null/true/false -> quoted.
    pg = _derived_table("true", "true", {"id"}, set(), flag, engine="postgres").sql(dialect="postgres")
    assert '"true"' in pg, pg
    # Oracle does not reserve a non-reserved name -> stays bare, as a trailing alias with no AS.
    ora = _derived_table("claim", "claim", {"id"}, set(), flag, engine="oracle").sql(dialect="oracle")
    assert '"claim"' not in ora and "claim" in ora, ora
