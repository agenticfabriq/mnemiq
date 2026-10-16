"""M163: on Oracle, ROWNUM is refused with what to write instead, not as a column the table lacks.

`SELECT id FROM (SELECT id FROM claim ORDER BY amount DESC) WHERE ROWNUM <= 5` was refused
`unknown_column` with the generic sentence. The register's first answer was to admit ROWNUM as the
pseudo-column it is. The ui lane measured why not, on Oracle 23: admitted, the same pseudo-column
runs and answers wrong in the shapes models write most. `WHERE ROWNUM <= 5 ORDER BY amount DESC`
sorts five arbitrary rows, `ROWNUM > 5` is always empty, and a selected ROWNUM numbers rows in
storage order, not the query's order. The owner's decision (2026-10-16): keep it refused, and say
to write `ORDER BY ... NULLS LAST FETCH FIRST n ROWS ONLY`, the bound #156 keeps and the row cap
reads, with NULLS LAST so M161's lint does not cost a second repair.

ROWID is refused with its own reason: a physical row address is not data an answer reports, and
Oracle cannot read it through the view a row policy puts over a table (ORA-01446, measured below).
"""

from __future__ import annotations

import os

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"id", "amount"}}
FILTERED = AccessPolicy(row_filters={"claim": "id <= 10"}, policy_schema={"claim": {"id", "amount"}})
ADVISED = "SELECT id FROM claim ORDER BY amount DESC NULLS LAST FETCH FIRST 5 ROWS ONLY"


def _decide(sql, engine="oracle", policy=None):
    return decide(sql, VISIBLE, dialect=engine, target=engine, policy=policy)


@pytest.mark.parametrize("sql", [
    "SELECT id FROM (SELECT id FROM claim ORDER BY amount DESC) WHERE ROWNUM <= 5",
    "SELECT id FROM claim WHERE ROWNUM <= 5 ORDER BY amount DESC",   # five arbitrary rows, sorted
    "SELECT id FROM claim WHERE ROWNUM > 5",                          # always empty
    "SELECT id, ROWNUM AS r FROM claim ORDER BY amount DESC",         # numbers in storage order
    "SELECT claim.rownum FROM claim",
])
@pytest.mark.parametrize("policy", [None, FILTERED])
def test_rownum_is_refused_with_the_bound_to_write_instead(sql, policy):
    verdict = _decide(sql, policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.repairable
    assert "write ORDER BY ... NULLS LAST FETCH FIRST n ROWS ONLY" in verdict.message


@pytest.mark.parametrize("policy", [None, FILTERED])
def test_the_advised_bound_is_approved_as_written(policy):
    verdict = _decide(ADVISED, policy=policy)
    assert isinstance(verdict, Approved), verdict
    assert verdict.target_sql.endswith("FETCH FIRST 5 ROWS ONLY") and verdict.row_cap is None
    # Why the advice says NULLS LAST: without it, M161's lint costs a second repair.
    bare = _decide(ADVISED.replace(" NULLS LAST", ""), policy=policy)
    assert isinstance(bare, Refusal) and bare.code is RefusalCode.LOGIC_LINT, bare


def test_a_write_bounded_by_rownum_is_refused_too():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write

    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))
    verdict = decide_write("DELETE FROM claim WHERE ROWNUM <= 5", VISIBLE, grants,
                           policy=AccessPolicy(), dialect="oracle", target="oracle",
                           writes_enabled=True)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict


def test_elsewhere_rownum_is_an_unknown_column_like_any_other():
    verdict = _decide("SELECT id FROM claim WHERE ROWNUM <= 5", "duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN
    assert "FETCH FIRST" not in verdict.message


@pytest.mark.parametrize("sql", ["SELECT id, ROWID FROM claim", "SELECT c.ROWID FROM claim c"])
def test_rowid_is_refused_with_its_own_reason(sql):
    verdict = _decide(sql)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.repairable and "physical address" in verdict.message


DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_rownum_answers_wrong_and_the_advised_bound_right():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    # Amounts out of id order, so the first rows stored are not the top rows by amount.
    data = "(SELECT level AS id, MOD(level * 7, 20) AS amount FROM dual CONNECT BY level <= 20) claim"
    top_5 = [(17,), (14,), (11,), (8,), (5,)]

    def run(sql):
        return cur.execute(sql.replace("FROM claim", f"FROM {data}")).fetchall()

    try:
        # The premise: what admitting ROWNUM let run.
        assert run("SELECT id FROM claim WHERE ROWNUM <= 5 ORDER BY amount DESC") != top_5
        assert run("SELECT id FROM claim WHERE ROWNUM > 5") == []
        assert run(_decide(ADVISED).target_sql) == top_5
        caller_s = run(_decide(ADVISED, policy=FILTERED).target_sql)
        assert len(caller_s) == 5 and all(i <= 10 for (i,) in caller_s), caller_s
        fenced = _decide("SELECT id FROM claim", policy=FILTERED).target_sql
        with pytest.raises(oracledb.DatabaseError, match="ORA-01446"):
            run(fenced.replace("SELECT id FROM", "SELECT id, ROWID FROM", 1))
    finally:
        con.close()
