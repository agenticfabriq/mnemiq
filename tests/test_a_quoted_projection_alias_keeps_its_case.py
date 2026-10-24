"""A derived table's quoted alias is the name the engine resolves it to, its case kept where the
engine keeps it: `SELECT (SELECT secret FROM (SELECT 1 AS "SECRET") d) FROM claim` was approved
with claim's `id` alone granted, the alias folded to `secret` and the reference read as it, where
Postgres distinguishes `"SECRET"` from `secret` and binds `secret` outward, to the hidden
claim.secret (measured on Postgres 16: the hidden values returned). Oracle folds the other way,
`secret` to `SECRET`, past a quoted `"secret"`. Codex's fifth review of #178, outside that PR's
diff and on main; M209 in the internal register.
"""

from __future__ import annotations

import duckdb

from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"id"}}


def _decide(sql, dialect):
    return decide(sql, VISIBLE, dialect=dialect, target=dialect)


def _refused_unknown(sql, dialect):
    verdict = _decide(sql, dialect)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (sql, verdict)


def test_a_quoted_alias_is_not_the_folded_name_where_quoting_keeps_case():
    # Postgres: `secret` names no output of d, which exposes `SECRET`, so it binds outward to
    # claim.secret (a correlated subquery, a CTE the same, a USING join's WHERE), hidden, refused.
    for sql in ('SELECT (SELECT secret FROM (SELECT 1 AS "SECRET") d) FROM claim',
                'WITH d AS (SELECT 1 AS "SECRET" FROM claim) SELECT (SELECT secret FROM d LIMIT 1) FROM claim',
                'SELECT d.id FROM (SELECT 1 AS "SECRET", id FROM claim) d JOIN claim c USING (id) WHERE secret = 41'):
        _refused_unknown(sql, "postgres")
    # Oracle folds an unquoted name up, past a quoted lowercase alias
    _refused_unknown('SELECT (SELECT secret FROM (SELECT 1 AS "secret" FROM claim) d) FROM claim', "oracle")
    # the alias referenced as written is the alias (refused as unknown before this change)
    assert isinstance(_decide('SELECT (SELECT "SECRET" FROM (SELECT 1 AS "SECRET") d) FROM claim', "postgres"), Approved)
    assert isinstance(_decide('SELECT (SELECT "secret" FROM (SELECT 1 AS "secret" FROM claim) d) FROM claim', "oracle"), Approved)
    # unquoted, the alias folds to the reference on both
    assert isinstance(_decide('SELECT (SELECT secret FROM (SELECT 1 AS secret) d) FROM claim', "postgres"), Approved)
    assert isinstance(_decide('SELECT (SELECT SECRET FROM (SELECT 1 AS "SECRET" FROM claim) d) FROM claim', "oracle"), Approved)
    # and a quoted reference to an unquoted alias names nothing on Postgres, as before
    _refused_unknown('SELECT (SELECT "SECRET" FROM (SELECT 1 AS secret) d) FROM claim', "postgres")


def test_every_reader_of_a_derived_output_keeps_the_quoting():
    # The review gate on the first commit: the generated-name rule and the member check read a
    # derived output's names too, and folded they refused honest queries on Postgres. `(SELECT
    # 1 AS "SECRET", 2 AS secret, secret_1 FROM claim) d` carries `secret` once there, so
    # `d.secret_1` is the column (DuckDB folds, renames the duplicate `secret_1` and the column
    # away, and is refused as before); a source named `secret` beside a `d` exposing `SECRET` is
    # read one way on Postgres (measured: the query runs), two on DuckDB.
    granted = {"claim": {"id", "secret", "secret_1"}}
    honest = 'SELECT d.secret_1 FROM (SELECT 1 AS "SECRET", 2 AS secret, secret_1 FROM claim) d'
    assert isinstance(decide(honest, granted, dialect="postgres", target="postgres"), Approved)
    assert isinstance(decide(honest, granted, dialect="duckdb", target="duckdb"), Refusal)
    one_way = 'SELECT (SELECT secret.x FROM (SELECT 1 AS "SECRET" FROM claim) d LIMIT 1) FROM (SELECT 1 AS x) secret'
    assert isinstance(_decide(one_way, "postgres"), Approved)
    assert isinstance(_decide(one_way, "duckdb"), Refusal)


def test_a_projection_without_an_alias_exposes_no_name():
    # Codex's first review of #180: `(SELECT 'secret') d` is `secret` to sqlglot's
    # `alias_or_name`, and the walk stopped at d; Postgres names the column `?column?`, DuckDB
    # and SQLite `'secret'`, and each binds a bare `secret` outward to claim.secret (measured
    # on all three: the hidden value returned). MySQL alone names the literal `secret`; the
    # refusal there is the fail-closed cost.
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, secret INT); INSERT INTO claim VALUES (1, 41)")
    assert duck.execute("SELECT (SELECT secret FROM (SELECT 'secret') d) FROM claim").fetchall() == [(41,)]
    for sql in ("SELECT (SELECT secret FROM (SELECT 'secret') d) FROM claim",
                "SELECT (SELECT secret FROM (SELECT upper('secret')) d) FROM claim",
                "SELECT (SELECT secret FROM (SELECT id, 'secret' FROM claim) d LIMIT 1) FROM claim",
                "WITH d AS (SELECT 'secret') SELECT (SELECT secret FROM d) FROM claim"):
        for dialect in ("postgres", "duckdb", "sqlite", "mysql"):
            _refused_unknown(sql, dialect)
    # aliased, the literal is the alias, as before
    assert isinstance(_decide("SELECT (SELECT secret FROM (SELECT 'secret' AS secret) d) FROM claim", "postgres"), Approved)
    # an unaliased projection keeps its position among the output's names, so a column list
    # names it and an honest `_N` column beside it is no generated name (the gate's round);
    # unlisted, the engine's own name may twin a column's (DuckDB names `(secret)` `secret`), so
    # a `_N` reference beside one is the engine's own and refused (the gate's second round)
    granted = {"claim": {"id", "secret_1"}}
    for dialect in ("postgres", "duckdb"):
        assert isinstance(decide("WITH w(a, secret_1) AS (SELECT 1, secret_1 FROM claim) SELECT w.secret_1 FROM w",
                                 granted, dialect=dialect, target=dialect), Approved), dialect
        assert isinstance(decide("SELECT d.secret_1 FROM (SELECT 1, secret_1 FROM claim) d",
                                 granted, dialect=dialect, target=dialect), Refusal), dialect
    assert duck.execute("SELECT d.secret_1 FROM (SELECT 'SAFE' AS secret, (secret) FROM claim) d").fetchall() == [(41,)]
    for sql in ("SELECT d.secret_1 FROM (SELECT 'SAFE' AS secret, (secret) FROM claim) d",
                "SELECT d.secret_1 FROM (SELECT 'SAFE' AS secret, ((c.secret)) FROM claim c) d"):
        for dialect in ("duckdb", "postgres"):
            assert isinstance(_decide(sql, dialect), Refusal), (sql, dialect)


def test_duckdb_folds_a_quoted_alias_and_reads_it():
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, secret INT); INSERT INTO claim VALUES (1, 41)")
    sql = 'SELECT (SELECT secret FROM (SELECT 1 AS "SECRET") d) FROM claim'
    assert duck.execute(sql).fetchall() == [(1,)]
    assert isinstance(_decide(sql, "duckdb"), Approved)
