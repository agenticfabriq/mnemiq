"""M163: on Oracle, ROWNUM is a pseudo-column, not a column the table lacks.

`SELECT id FROM (SELECT id FROM claim ORDER BY amount DESC) WHERE ROWNUM <= 5`, Oracle's classic
top-N, was refused `unknown_column`. The repair cost a round, and it most likely wrote FETCH FIRST,
which M162 (#156) now keeps. So ROWNUM is admitted on an Oracle target: unqualified and unquoted,
as Oracle reads it, and on a read only: a write bounded by ROWNUM changes an arbitrary set of
rows. Through a row policy's rewrite it numbers only the rows the caller may see.

ROWID is not admitted beside it, against the register's first recommendation. A physical row
address is not data an answer reports, and Oracle cannot read it through the view a row policy
puts over a table (ORA-01446, measured below). Admitted, it would fail every governed statement
naming it, and since #159 that failure would be reported as the operator's.
"""

from __future__ import annotations

import os

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"id", "amount"}}
TOP_N = "SELECT id FROM (SELECT id FROM claim ORDER BY amount DESC) WHERE ROWNUM <= 3"
FILTERED = AccessPolicy(row_filters={"claim": "id <= 10"}, policy_schema={"claim": {"id", "amount"}})


def _decide(sql, engine="oracle", policy=None):
    return decide(sql, VISIBLE, dialect=engine, target=engine, policy=policy)


@pytest.mark.parametrize("sql", [TOP_N, "SELECT ROWNUM AS r, id FROM claim",
                                 "SELECT id FROM claim WHERE rownum <= 5"])
@pytest.mark.parametrize("policy", [None, FILTERED])
def test_rownum_is_admitted_on_oracle(sql, policy):
    verdict = _decide(sql, policy=policy)
    assert isinstance(verdict, Approved), verdict
    assert "ROWNUM" in verdict.target_sql.upper()
    assert '"ROWNUM"' not in verdict.target_sql.upper(), "quoted, Oracle reads a column"
    assert "ROWNUM" not in [c.upper() for c in verdict.columns], "not a column the answer read"


@pytest.mark.parametrize("engine,sql", [
    ("oracle", 'SELECT "ROWNUM" FROM claim'),       # quoted: an identifier, not the pseudo-column
    ("oracle", "SELECT claim.rownum FROM claim"),   # Oracle has no table-qualified ROWNUM
    ("duckdb", "SELECT id FROM claim WHERE ROWNUM <= 5"),  # not DuckDB's
])
def test_rownum_is_a_column_like_any_other_elsewhere(engine, sql):
    verdict = _decide(sql, engine)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


@pytest.mark.parametrize("sql", ["DELETE FROM claim WHERE ROWNUM <= 5",
                                 "UPDATE claim SET amount = 1 WHERE rownum < 2"])
def test_a_write_bounded_by_rownum_stays_refused(sql):
    """It would change an arbitrary set of rows: the top-N reading is a read's alone."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write

    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))
    verdict = decide_write(sql, VISIBLE, grants, policy=AccessPolicy(), dialect="oracle",
                           target="oracle", writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


@pytest.mark.parametrize("sql", ["SELECT id, ROWID FROM claim", "SELECT c.ROWID FROM claim c"])
def test_rowid_is_refused_with_its_own_reason(sql):
    verdict = _decide(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.repairable and "physical address" in verdict.message


DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_the_top_n_is_the_caller_s_and_rowid_cannot_cross_the_policy():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    data = "(SELECT level AS id, level * 10 AS amount FROM dual CONNECT BY level <= 20) claim"
    try:
        def ran(policy):
            verdict = _decide(TOP_N, policy=policy)
            assert isinstance(verdict, Approved), verdict
            return cur.execute(verdict.target_sql.replace("FROM claim", f"FROM {data}")).fetchall()

        assert ran(None) == [(20,), (19,), (18,)]
        assert ran(FILTERED) == [(10,), (9,), (8,)], "numbered over the caller's rows only"
        fenced = _decide("SELECT id FROM claim", policy=FILTERED).target_sql
        with pytest.raises(oracledb.DatabaseError, match="ORA-01446"):
            cur.execute(fenced.replace("SELECT id FROM", "SELECT id, ROWID FROM", 1)
                        .replace("FROM claim", f"FROM {data}")).fetchall()
    finally:
        con.close()
