"""M150: a governed CTE is decided only for an engine whose binding of a CTE name the rule models.

The scope rule models DuckDB: a WITH's earlier siblings are in scope, the CTE itself only under
RECURSIVE, a later sibling never. Postgres binds a later sibling under RECURSIVE and SQLite always,
so there the row-policy rewrite would wrap a CTE's literal rows as if they were the table --
measured on Postgres 16, `[40]` for `[1, 40]`. Production always decides as the adapter's dialect
(duckdb, or oracle, which refuses the shapes where the bindings differ), so this was latent; the
owner's decision (2026-10-14) was to make the precondition a check rather than leave it to stay so.
"""

from __future__ import annotations

import os

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode

SCHEMA = {"claim": {"amount"}, "scratch": {"amount"}}
FILTERED = AccessPolicy(row_filters={"claim": "amount > 10"}, policy_schema=SCHEMA)
FORWARD = ("WITH RECURSIVE foo AS (SELECT amount FROM claim), claim(amount) AS "
           "(SELECT 1 UNION ALL SELECT 40) SELECT amount FROM foo")


def _read(sql, target, policy=FILTERED):
    return decide(sql, SCHEMA, dialect="duckdb", target=target, policy=policy)


@pytest.mark.parametrize("target", ["postgres", "sqlite"])
def test_a_governed_cte_decided_for_an_unmodelled_engine_is_refused(target):
    verdict = _read(FORWARD, target)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.POLICY_TARGET_UNSUPPORTED
    assert not verdict.repairable  # a deployment setting: the model cannot write its way past it


@pytest.mark.parametrize("target", ["duckdb", "oracle"])
def test_the_engines_the_rule_holds_for_are_decided_as_before(target):
    verdict = _read(FORWARD, target)
    assert getattr(verdict, "code", None) is not RefusalCode.POLICY_TARGET_UNSUPPORTED


def test_a_statement_naming_no_cte_binds_the_same_everywhere_and_is_decided():
    verdict = _read("SELECT amount FROM claim", "postgres")
    assert not isinstance(verdict, Refusal), verdict
    assert "amount > 10" in verdict.target_sql  # the control: the policy applied


def test_with_no_policy_a_cte_is_decided_on_any_engine():
    """The eval runners decide as sqlite with no policy, and are unaffected."""
    verdict = _read(FORWARD, "sqlite", policy=AccessPolicy(policy_schema=SCHEMA))
    assert getattr(verdict, "code", None) is not RefusalCode.POLICY_TARGET_UNSUPPORTED


def test_the_write_decider_refuses_it_too():
    grants = GrantSet(frozenset(SCHEMA), writable=frozenset({"scratch"}),
                      row_filters={"claim": "amount > 10"})
    verdict = decide_write("INSERT INTO scratch (amount) WITH c AS (SELECT amount FROM claim) "
                           "SELECT amount FROM c", SCHEMA, grants, policy=FILTERED,
                           dialect="duckdb", target="postgres", writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.POLICY_TARGET_UNSUPPORTED


ORACLE_DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not ORACLE_DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_oracle_rejects_a_forward_reference_and_a_rewritten_self_reference():
    """Why Oracle is on POLICY_TARGETS, for the two shapes where its binding of a CTE name is known
    to differ from the rule's: decided as oracle under a row filter, both reach Oracle -- the
    decider does not stop them -- and Oracle rejects each (23ai, which takes SELECT without FROM
    dual). Other shapes are not enumerated here."""
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"),
                           dsn=ORACLE_DSN)
    cur = con.cursor()
    try:  # a run killed before its own cleanup left the table behind
        cur.execute("DROP TABLE m150_claim PURGE")
    except oracledb.DatabaseError:
        pass
    schema = {"m150_claim": {"amount"}}
    policy = AccessPolicy(row_filters={"m150_claim": "amount > 10"}, policy_schema=schema)
    cur.execute("CREATE TABLE m150_claim (amount NUMBER)")
    try:
        cur.execute("INSERT INTO m150_claim VALUES (5)")
        cur.execute("INSERT INTO m150_claim VALUES (20)")
        cases = {
            "ORA-32031": "WITH foo AS (SELECT amount FROM m150_claim), m150_claim (amount) AS "
                         "(SELECT 1 UNION ALL SELECT 40) SELECT amount FROM foo",
            "ORA-32042": "WITH m150_claim (amount) AS (SELECT 1 UNION ALL SELECT amount + 1 "
                         "FROM m150_claim WHERE amount < 3) SELECT amount FROM m150_claim",
        }
        for error, sql in cases.items():
            verdict = decide(sql, schema, dialect="oracle", target="oracle", policy=policy)
            assert not isinstance(verdict, Refusal), (sql, verdict)
            with pytest.raises(oracledb.DatabaseError, match=error):
                cur.execute(verdict.target_sql)
        # The control: with no CTE named like the table, the policy applies and rows come back.
        verdict = decide("WITH foo AS (SELECT amount FROM m150_claim) SELECT amount FROM foo",
                         schema, dialect="oracle", target="oracle", policy=policy)
        cur.execute(verdict.target_sql)
        assert [r[0] for r in cur.fetchall()] == [20]
    finally:
        cur.execute("DROP TABLE m150_claim PURGE")
        con.close()
