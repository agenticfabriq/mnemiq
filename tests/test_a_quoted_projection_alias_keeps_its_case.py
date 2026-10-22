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


def test_duckdb_folds_a_quoted_alias_and_reads_it():
    duck = duckdb.connect()
    duck.execute("CREATE TABLE claim (id INT, secret INT); INSERT INTO claim VALUES (1, 41)")
    sql = 'SELECT (SELECT secret FROM (SELECT 1 AS "SECRET") d) FROM claim'
    assert duck.execute(sql).fetchall() == [(1,)]
    assert isinstance(_decide(sql, "duckdb"), Approved)
