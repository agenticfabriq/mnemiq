"""M223: a table reference is decided only where the engine reads it as the name it spells.

Oracle folds an unquoted name to upper case and keeps a quoted one as written, so a schema can hold
both `SALES` and a quoted `"sales"`, and the snapshot keys both as stored. The grant check compared
the reference as written, so a grant on the quoted `sales` approved `SELECT ID FROM sales`, which
Oracle runs against SALES. Postgres keeps a quoted name too and folds the other way. Each later
check looks a table up by the same written spelling, so the step that makes the grant check right
is to decide a reference only where the engine reads it as written, and to say how to write it
where it does not. DuckDB and SQLite fold quoted names too, so a reference there always reads as
written.
"""
from __future__ import annotations

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

COLUMNS = {"ID", "NOTE"}


def _oracle(sql, granted):
    visible = {granted: set(COLUMNS)}
    policy = AccessPolicy(policy_schema={"sales": set(COLUMNS), "SALES": set(COLUMNS)})
    return decide(sql, visible, dialect="oracle", target="oracle", policy=policy)


@pytest.mark.parametrize("sql", ["SELECT ID FROM sales", "SELECT ID FROM Sales",
                                 "SELECT s.ID FROM sales s",
                                 "SELECT ID FROM SALES WHERE ID IN (SELECT ID FROM sales)",
                                 "WITH x AS (SELECT ID FROM sales) SELECT ID FROM x"])
def test_on_oracle_a_grant_on_a_quoted_name_does_not_approve_the_unquoted_spelling(sql):
    verdict = _oracle(sql, "sales")
    assert isinstance(verdict, Refusal), verdict
    assert verdict.code in (RefusalCode.UNKNOWN_TABLE, RefusalCode.UNAUTHORIZED_TABLE), verdict


def test_the_refusal_says_to_quote_the_granted_name_and_names_nothing_else():
    verdict = _oracle("SELECT ID FROM sales", "sales")
    assert verdict.code == RefusalCode.UNKNOWN_TABLE and verdict.repairable, verdict
    assert '"sales"' in verdict.message, verdict.message
    assert "SALES" not in verdict.message  # nothing but the name written and the one held


@pytest.mark.parametrize("sql", ['SELECT ID FROM "sales"', 'SELECT s.ID FROM "sales" s'])
def test_on_oracle_the_quoted_spelling_of_the_granted_name_is_approved(sql):
    verdict = _oracle(sql, "sales")
    assert isinstance(verdict, Approved), verdict
    assert '"sales"' in verdict.target_sql


@pytest.mark.parametrize("sql", ["SELECT ID FROM SALES", 'SELECT ID FROM "SALES"'])
def test_on_oracle_the_unquoted_name_is_approved_against_its_stored_spelling(sql):
    assert isinstance(_oracle(sql, "SALES"), Approved)


def test_a_name_oracle_folds_onto_the_granted_one_is_refused_saying_how_it_is_listed():
    # `sales` is read as SALES, which the caller holds: written as listed, it is approved.
    verdict = _oracle("SELECT ID FROM sales", "SALES")
    assert verdict.code == RefusalCode.UNKNOWN_TABLE and verdict.repairable, verdict
    assert "'SALES'" in verdict.message, verdict.message


def test_on_oracle_a_long_s_that_folds_onto_another_table_is_refused():
    # Oracle upper-cases `ſ` to `S`, so an unquoted `ſALES` reads SALES (measured on Oracle Free).
    visible = {"ſALES": set(COLUMNS)}
    policy = AccessPolicy(policy_schema={"ſALES": set(COLUMNS), "SALES": set(COLUMNS)})
    unquoted = decide("SELECT ID FROM ſALES", visible, dialect="oracle", target="oracle", policy=policy)
    quoted = decide('SELECT ID FROM "ſALES"', visible, dialect="oracle", target="oracle", policy=policy)
    assert isinstance(unquoted, Refusal) and isinstance(quoted, Approved), (unquoted, quoted)



@pytest.mark.parametrize("name", ["ᾳ", "ᾀ", "ⓐBC", "ⅰX"])
def test_on_oracle_a_name_whose_unquoted_fold_is_unmeasured_is_decided_only_quoted(name):
    # Oracle folds some Greek letters with a subscript iota to one letter where the shared rule
    # keeps them, and the sweep never covered a non-letter with an uppercase, so an unquoted
    # spelling may read another table: refused, quoted approved.
    visible = {name: set(COLUMNS)}
    unquoted = decide(f"SELECT ID FROM {name}", visible, dialect="oracle", target="oracle")
    quoted = decide(f'SELECT ID FROM "{name}"', visible, dialect="oracle", target="oracle")
    assert isinstance(unquoted, Refusal) and isinstance(quoted, Approved), (unquoted, quoted)



@pytest.mark.parametrize("sql", ['WITH "ᾀ" AS (SELECT 1 AS ID FROM DUAL) SELECT ID FROM ᾀ',
                                 'WITH ᾀ AS (SELECT 1 AS ID FROM DUAL) SELECT ID FROM "ᾀ"',
                                 "WITH ᾀ AS (SELECT 1 AS ID FROM DUAL) SELECT ID FROM ᾀ"])
def test_on_oracle_a_cte_or_reference_named_with_an_unmeasured_fold_is_refused(sql):
    # Which one a reference binds to cannot be told: the shared rules keep `ᾀ` either way, and
    # Oracle folds the unquoted `ᾀ` to a letter of its own, so the reference can read a table of
    # that name past the grant (Codex's review of the M223 branch). The third, unquoted on both
    # sides, binds whatever the fold is; refused all the same, quoting is the repair.
    verdict = decide(sql, {"DUAL": {"DUMMY"}}, dialect="oracle", target="oracle")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_TABLE, verdict
    assert "quoted" in verdict.message, verdict.message


def test_on_oracle_the_quoted_cte_and_reference_are_approved():
    verdict = decide('WITH "ᾀ" AS (SELECT 1 AS ID FROM DUAL) SELECT ID FROM "ᾀ"', {"DUAL": {"DUMMY"}},
                     dialect="oracle", target="oracle")
    assert isinstance(verdict, Approved), verdict

def test_on_oracle_a_kept_letter_the_sweep_measured_reads_as_written():
    # Oracle stores an unquoted `straße` as STRAßE: `ß` is kept, measured.
    verdict = decide("SELECT ID FROM STRAßE", {"STRAßE": set(COLUMNS)}, dialect="oracle",
                     target="oracle")
    assert isinstance(verdict, Approved), verdict

def test_on_postgres_a_grant_on_a_quoted_mixed_case_name_does_not_approve_its_unquoted_spelling():
    visible = {"Claim": {"id"}}
    unquoted = decide("SELECT id FROM Claim", visible, dialect="postgres", target="postgres")
    quoted = decide('SELECT id FROM "Claim"', visible, dialect="postgres", target="postgres")
    assert isinstance(unquoted, Refusal) and isinstance(quoted, Approved), (unquoted, quoted)


@pytest.mark.parametrize("engine", ["duckdb", "sqlite"])
@pytest.mark.parametrize("sql", ["SELECT id FROM Claim", 'SELECT id FROM "Claim"'])
def test_where_quoting_folds_too_a_reference_spelled_as_stored_is_approved(engine, sql):
    assert isinstance(decide(sql, {"Claim": {"id"}}, dialect=engine, target=engine), Approved)


@pytest.mark.parametrize("engine", ["duckdb", "sqlite", "postgres"])
def test_another_case_of_a_granted_name_is_refused_saying_how_it_is_listed(engine):
    # One object to the engine, and every later check looks the table up as written, so the
    # spelling is repaired rather than approved under a key the policy maps do not hold.
    stored = "claim" if engine == "postgres" else "Claim"
    verdict = decide("SELECT id FROM CLAIM", {stored: {"id"}}, dialect=engine, target=engine)
    assert verdict.code == RefusalCode.UNKNOWN_TABLE and verdict.repairable, verdict
    assert repr(stored) in verdict.message, verdict.message


def test_a_name_no_grant_reads_is_still_unauthorized():
    verdict = _oracle("SELECT ID FROM other", "sales")
    assert verdict.code == RefusalCode.UNAUTHORIZED_TABLE, verdict


def _write(sql, writable):
    visible = {writable: set(COLUMNS)}
    grants = GrantSet(frozenset({writable}), writable=frozenset({writable}))
    return decide_write(sql, visible, grants, policy=AccessPolicy(), dialect="oracle",
                        writes_enabled=True)


@pytest.mark.parametrize("sql", ["INSERT INTO sales (ID, NOTE) VALUES (1, 'x')",
                                 # the target alone misread: what the grant check walks reads
                                 # only the quoted source here
                                 'INSERT INTO sales SELECT ID, NOTE FROM "sales"',
                                 "UPDATE sales SET NOTE = 'x' WHERE ID = 1",
                                 "DELETE FROM sales WHERE ID = 1"])
def test_on_oracle_a_write_grant_on_a_quoted_name_does_not_approve_the_unquoted_target(sql):
    verdict = _write(sql, "sales")
    assert isinstance(verdict, Refusal), verdict


@pytest.mark.parametrize("sql", ["""INSERT INTO "sales" (ID, NOTE) VALUES (1, 'x')""",
                                 """UPDATE "sales" SET NOTE = 'x' WHERE ID = 1""",
                                 'DELETE FROM "sales" WHERE ID = 1'])
def test_on_oracle_the_quoted_target_is_approved(sql):
    assert isinstance(_write(sql, "sales"), ApprovedWrite), _write(sql, "sales")
