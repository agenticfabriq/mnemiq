"""M166 and M170: a column name is compared as the engine resolves it, not as written.

`column_tables` keyed a scope's base tables by their written alias, so (M166) `SELECT C.secret
FROM claim c` matched no alias, read as a derived table's qualifier, and was approved for a
`secret` the caller was never granted, on every engine. The qualifier and the alias now resolve
the same way before they meet. `check_access` tested `column.name not in visible[table]` on the
text as typed, so (M170) `SELECT SECRET FROM claim` was refused `unknown_column` for a granted
`secret`. On DuckDB and SQLite, which keep a name's case without telling names apart by it, both
sides now fold. On Postgres and Oracle, where quoting keeps case, both sides stay as typed: a
stored `"Amount"` is another column than `amount`, and that boundary (table keys and row filters
are as typed too) is one decision, in the register. Measured on main `dd781f3` through `decide`.
"""
from __future__ import annotations

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

GRANTED = {"claim": {"id", "amount", "secret"}}
HIDDEN = {"claim": {"id", "amount"}}
DENIED = AccessPolicy(denied={("claim", "secret")}, policy_schema=GRANTED)


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INT, amount INT, secret VARCHAR)")
    con.execute("INSERT INTO claim VALUES (1, 5, 'ssn-123'), (2, 6, 'ssn-456')")
    return con


def _decide(sql, visible, engine, policy=None):
    return decide(sql, visible, dialect=engine, target=engine, policy=policy)


# M170 on DuckDB: a granted column spelled in another case, a quoted alias met by its bare
# spelling, and a star's EXCLUDE naming the column in upper case.
SPELLINGS = [
    ("SELECT SECRET FROM claim", "SELECT secret FROM claim"),
    ("SELECT Secret FROM claim", "SELECT secret FROM claim"),
    ("SELECT c.SECRET FROM claim c", "SELECT c.secret FROM claim c"),
    ('SELECT amount AS "K" FROM claim ORDER BY k', "SELECT amount AS k FROM claim ORDER BY k"),
    ("SELECT id FROM (SELECT * EXCLUDE (SECRET) FROM claim) d", "SELECT id FROM claim"),
]


@pytest.mark.parametrize("sql, same", SPELLINGS)
def test_on_duckdb_another_case_of_a_granted_column_is_approved_and_reads_the_same(con, sql, same):
    verdict = _decide(sql, GRANTED, "duckdb")
    assert isinstance(verdict, Approved), (sql, verdict)
    assert con.execute(verdict.target_sql).fetchall() == con.execute(same).fetchall()


def test_on_duckdb_a_stored_name_in_another_case_is_met_too(con):
    # DuckDB keeps `Secret` as created and reads `secret` as it; the grant map carries the
    # catalog's spelling.
    verdict = _decide("SELECT secret FROM claim", {"claim": {"id", "amount", "Secret"}}, "duckdb")
    assert isinstance(verdict, Approved), verdict
    assert con.execute(verdict.target_sql).fetchall() == con.execute("SELECT secret FROM claim").fetchall()


def test_a_hidden_column_in_another_case_is_still_unknown():
    for engine in ("duckdb", "oracle", "postgres"):
        verdict = _decide("SELECT SECRET FROM claim", HIDDEN, engine)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
        assert verdict.subject == "SECRET", verdict  # the refusal keeps the spelling written


def test_a_quoted_alias_keeps_its_case_where_the_engine_does():
    # Postgres reads `"K"` as `K` and a bare `k` as `k`: two names, and the engine would refuse
    # the ORDER BY. Refused here first, with the spelling written.
    verdict = _decide('SELECT amount AS "K" FROM claim ORDER BY k', GRANTED, "postgres")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.subject == "k"


@pytest.mark.parametrize("engine, stored, bare, quoted", [
    ("postgres", {"id", "Amount"}, "SELECT amount FROM claim", 'SELECT "Amount" FROM claim'),
    ("postgres", {"id", "Amount"}, "SELECT c.amount FROM claim c", 'SELECT c."Amount" FROM claim c'),
    ("oracle", {"ID", "secret"}, "SELECT SECRET FROM claim", 'SELECT "secret" FROM claim'),
    ("oracle", {"ID", "secret"}, "SELECT c.SECRET FROM claim c", 'SELECT c."secret" FROM claim c'),
])
def test_a_case_kept_name_is_not_met_by_its_fold(engine, stored, bare, quoted):
    # A column created with quotes keeps its case, and the engine reads the bare fold as ANOTHER
    # column, one the caller is not granted here (the review gate's case): the stored side is
    # never folded where quoting keeps case.
    refused = _decide(bare, {"claim": stored}, engine)
    assert isinstance(refused, Refusal) and refused.code is RefusalCode.UNKNOWN_COLUMN, (
        engine, bare, refused)
    assert isinstance(_decide(quoted, {"claim": stored}, engine), Approved), (engine, quoted)


# M166: a qualifier in another case than the alias it names. Every engine reads `C.secret` as
# `c.secret`; matched as written, `C` named no base table and the column went unchecked.
QUALIFIERS = [
    "SELECT C.secret FROM claim c",
    "SELECT Claim.secret FROM claim",
    "SELECT id FROM claim c WHERE C.secret = 'ssn-123'",
    "SELECT id FROM claim c WHERE EXISTS (SELECT 1 FROM claim d WHERE D.secret = C.secret)",
]


@pytest.mark.parametrize("engine", ["duckdb", "oracle", "postgres"])
@pytest.mark.parametrize("sql", QUALIFIERS)
def test_a_hidden_column_behind_a_qualifier_in_another_case_is_refused(sql, engine):
    verdict = _decide(sql, HIDDEN, engine)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, (
        engine, sql, verdict)
    assert verdict.subject == "secret", verdict


@pytest.mark.parametrize("sql", QUALIFIERS)
def test_a_denied_column_behind_a_qualifier_in_another_case_is_refused(sql):
    verdict = _decide(sql, GRANTED, "duckdb", policy=DENIED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNAUTHORIZED_COLUMN, (
        sql, verdict)


@pytest.mark.parametrize("sql", QUALIFIERS)
def test_a_granted_column_behind_a_qualifier_in_another_case_is_approved_and_runs(con, sql):
    verdict = _decide(sql, GRANTED, "duckdb")
    assert isinstance(verdict, Approved), (sql, verdict)
    assert con.execute(verdict.target_sql).fetchall() == con.execute(sql).fetchall()


def test_on_oracle_a_quoted_qualifier_meets_the_alias_it_folds_to():
    # `"C"` is `C` on Oracle, as the unquoted alias `c` is; matched as written it named nothing.
    verdict = _decide('SELECT "C".secret FROM claim c', HIDDEN, "oracle")
    assert isinstance(verdict, Refusal) and verdict.subject == "secret", verdict


def test_a_write_reads_a_hidden_column_behind_a_qualifier_in_another_case_as_that_column():
    grants = GrantSet(frozenset({"claim", "other"}), writable=frozenset({"claim"}))
    other = {"other": {"id", "amount"}}

    def write(sql, visible):
        return decide_write(sql, {**visible, **other}, grants, policy=AccessPolicy(),
                            dialect="duckdb", target="duckdb", writes_enabled=True)

    for sql in ("UPDATE claim c SET amount = 1 WHERE C.secret = 'x'",
                "INSERT INTO claim (id, amount) VALUES (1, 2) ON CONFLICT (id) DO UPDATE "
                "SET amount = length(CLAIM.secret)",
                # The SELECT gives the statement a scope, and the target is in none of them, so
                # only the write-target map can say what `CLAIM` names; matched as written it
                # said nothing, and the column went unchecked.
                "INSERT INTO claim (id, amount) SELECT id, amount FROM other ON CONFLICT (id) "
                "DO UPDATE SET amount = length(CLAIM.secret)",
                "INSERT INTO claim AS t (id, amount) SELECT id, amount FROM other ON CONFLICT (id) "
                "DO UPDATE SET amount = length(T.secret)"):
        refused = write(sql, HIDDEN)
        assert isinstance(refused, Refusal) and refused.subject == "secret", (sql, refused)
        assert isinstance(write(sql, GRANTED), ApprovedWrite), sql
