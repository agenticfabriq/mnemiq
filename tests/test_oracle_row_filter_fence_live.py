"""M160, live: the rewrite mnemiq ships keeps Oracle from evaluating the caller's predicate on a
row the filter removes.

The caller's predicate fails only where `amount = 20`, which only rows the filter removes have.
Run fenced, as `decide` renders it, it must succeed and return the caller's rows; run with the fence
stripped -- the control -- it must fail, or this shape no longer reaches the reorder on this
instance and the fenced half proves nothing. Whether Oracle reorders depends on its plan: with a
primary key and fresh statistics none of these shapes did (measured), so these tables have neither.
The shapes are the eight of twenty that reordered without the fence (register M160).

Integration: skips without `MNEMIQ_ORACLE_TEST_DSN`, like the adapter suite, whose module
docstring has the container recipe.
"""

from __future__ import annotations

import os

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved

DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")
USER = os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser")
PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw")

if not DSN:
    pytest.skip("set MNEMIQ_ORACLE_TEST_DSN to run the Oracle fence tests",
                allow_module_level=True)

oracledb = pytest.importorskip("oracledb")

FENCE = " AND ROWNUM > 0"
SCHEMA = {"m160_claim": {"id", "region", "amount"},
          "m160_payment": {"id", "claim_id", "paid"},
          "m160_entitlement": {"id", "who"}}
VISIBLE = {"m160_claim": SCHEMA["m160_claim"], "m160_payment": SCHEMA["m160_payment"]}
ENTITLED = "id IN (SELECT id FROM m160_entitlement WHERE who = 'r')"
POLICIES = {
    "subquery": {"m160_claim": ENTITLED},
    "two tables": {"m160_claim": ENTITLED,
                   "m160_payment": "claim_id IN (SELECT id FROM m160_entitlement WHERE who = 'r')"},
}
FAILS_ON_REMOVED = "CASE WHEN amount = 20 THEN TO_NUMBER(region) ELSE 1 END > 0"
CALLERS = {
    "where": ("SELECT id FROM m160_claim WHERE {p}", [(i,) for i in range(1, 6)]),
    "join on": ("SELECT c.id FROM m160_claim c JOIN m160_payment p ON p.claim_id = c.id AND {p}",
                [(i,) for i in range(1, 6)]),
    "in subquery": ("SELECT id FROM m160_payment WHERE claim_id IN "
                    "(SELECT id FROM m160_claim WHERE {p})", [(i,) for i in range(1, 6)]),
    "cte": ("WITH x AS (SELECT id, amount, region FROM m160_claim WHERE {p}) SELECT id FROM x",
            [(i,) for i in range(1, 6)]),
}


@pytest.fixture(scope="module")
def cursor():
    con = oracledb.connect(user=USER, password=PASSWORD, dsn=DSN)
    c = con.cursor()
    names = ("m160_claim", "m160_payment", "m160_entitlement")

    def drop():
        for name in names:
            try:
                c.execute(f"DROP TABLE {name} PURGE")
            except oracledb.DatabaseError:
                pass  # not there

    drop()
    c.execute("CREATE TABLE m160_claim (id NUMBER, region VARCHAR2(20), amount NUMBER)")
    c.execute("CREATE TABLE m160_payment (id NUMBER, claim_id NUMBER, paid NUMBER)")
    c.execute("CREATE TABLE m160_entitlement (id NUMBER, who VARCHAR2(20))")
    # Ids 1..5 are the caller's; every other row is removed by the filter and has a region that
    # is not a number.
    c.executemany("INSERT INTO m160_claim VALUES (:1, :2, :3)",
                  [(i, "west" if i <= 5 else f"r{i}", 10 if i <= 5 else 20) for i in range(1, 3001)])
    c.executemany("INSERT INTO m160_payment VALUES (:1, :2, 1)", [(i, i) for i in range(1, 3001)])
    c.executemany("INSERT INTO m160_entitlement VALUES (:1, 'r')", [(i,) for i in range(1, 6)])
    con.commit()
    yield c
    drop()
    con.close()


def _fenced_sql(policy_name, caller):
    template, _ = CALLERS[caller]
    verdict = decide(template.format(p=FAILS_ON_REMOVED), VISIBLE, dialect="oracle",
                     target="oracle",
                     policy=AccessPolicy(row_filters=POLICIES[policy_name], policy_schema=SCHEMA))
    assert isinstance(verdict, Approved), verdict
    return verdict.target_sql  # the fence's shape is the unit suite's; this runs it


@pytest.mark.integration
@pytest.mark.parametrize("policy_name", sorted(POLICIES))
@pytest.mark.parametrize("caller", sorted(CALLERS))
def test_the_fenced_rewrite_never_evaluates_the_caller_s_predicate_on_a_removed_row(
        cursor, policy_name, caller):
    sql = _fenced_sql(policy_name, caller)
    with pytest.raises(oracledb.DatabaseError):  # the control: this shape reorders unfenced
        cursor.execute(sql.replace(FENCE, "")).fetchall()
    assert sorted(cursor.execute(sql).fetchall()) == CALLERS[caller][1]
