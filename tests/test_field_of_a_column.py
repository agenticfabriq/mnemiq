"""M165: a field of a nested column is read as that column, so every column rule applies to it.

sqlglot reads `c.secret.status` as a column `status` qualified by `secret` (and `c` as its db),
so every guard saw a qualifier no scope defines -- read as a CTE or subquery and skipped -- and
no guard saw `secret`. Measured on main `ec0ad7d`: with `secret` a DuckDB STRUCT or JSON column,
or an Oracle JSON one, the field reads below were approved and returned the value whether
`secret` was left out of the grant, denied or masked, while `SELECT c.secret` was refused.
"""

from __future__ import annotations

import os

import duckdb
import pytest
import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.scope import fields_as_members
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

GRANTED = {"claim": {"id", "amount", "secret"}}
HIDDEN = {"claim": {"id", "amount"}}
DENIED = AccessPolicy(denied={("claim", "secret")}, policy_schema=GRANTED)
MASKED = AccessPolicy(masked={("claim", "secret")}, policy_schema=GRANTED)
FILTERED = AccessPolicy(row_filters={"claim": "id <= 10"}, policy_schema=GRANTED)

FIELD_READS = [
    "SELECT c.secret.status FROM claim c",
    "SELECT secret.status FROM claim",
    "SELECT claim.secret.status FROM claim",
    "SELECT c.secret.detail.code FROM claim c",
    "SELECT id FROM claim c WHERE c.secret.status LIKE 'ssn%'",
]
WITHHELD = [
    (HIDDEN, None, RefusalCode.UNKNOWN_COLUMN),
    (GRANTED, DENIED, RefusalCode.UNAUTHORIZED_COLUMN),
    (GRANTED, MASKED, RefusalCode.MASKED_COLUMN_IN_PREDICATE),
]


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INT, amount INT, "
                "secret STRUCT(status VARCHAR, detail STRUCT(code VARCHAR)))")
    con.execute("INSERT INTO claim VALUES (1, 5, {'status': 'ssn-123', 'detail': {'code': 'k-9'}}), "
                "(20, 6, {'status': 'ssn-456', 'detail': {'code': 'k-8'}})")
    return con


def test_the_premise_duckdb_reads_each_as_a_field_of_secret(con):
    for sql in FIELD_READS:
        assert con.execute(sql).fetchall(), sql


# Oracle reads a field only after a table alias (measured below), so there `secret.status` is
# not one and is left as written.
@pytest.mark.parametrize("engine, sql", [
    *(("duckdb", sql) for sql in FIELD_READS),
    *(("oracle", sql) for sql in FIELD_READS if sql != "SELECT secret.status FROM claim"),
])
@pytest.mark.parametrize("visible, policy, code", WITHHELD, ids=["hidden", "denied", "masked"])
def test_a_field_of_a_withheld_column_is_refused_as_that_column(sql, visible, policy, code, engine):
    verdict = decide(sql, visible, dialect=engine, target=engine, policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is code, verdict
    assert verdict.subject == "secret", verdict


def test_a_masked_column_s_fields_are_refused_saying_so():
    verdict = decide("SELECT c.secret.status FROM claim c", GRANTED, dialect="duckdb",
                     target="duckdb", policy=MASKED)
    assert isinstance(verdict, Refusal) and "its fields cannot be read" in verdict.message


@pytest.mark.parametrize("sql", FIELD_READS)
@pytest.mark.parametrize("policy", [None, FILTERED])
def test_a_field_of_a_granted_column_is_approved_and_reads_the_same(con, sql, policy):
    verdict = decide(sql, GRANTED, dialect="duckdb", target="duckdb", policy=policy)
    assert isinstance(verdict, Approved), verdict
    rows = con.execute(verdict.target_sql).fetchall()
    if policy is None:
        assert sorted(rows) == sorted(con.execute(sql).fetchall())
        # The rewrite renders as written: the statement run is the one asked, capped.
        assert verdict.target_sql == f"{sqlglot.transpile(sql, read='duckdb')[0]} LIMIT 1000"
    else:
        assert len(rows) == 1  # id <= 10


def test_a_field_of_a_subquery_s_column_is_that_column_s_rule_inside():
    sql = "SELECT t.x.status FROM (SELECT secret AS x FROM claim) t"
    denied = decide(sql, GRANTED, dialect="duckdb", target="duckdb", policy=DENIED)
    assert isinstance(denied, Refusal) and denied.code is RefusalCode.UNAUTHORIZED_COLUMN
    masked = decide(sql, GRANTED, dialect="duckdb", target="duckdb", policy=MASKED)
    # The mask applies inside the subquery, so the field read sees NULL, not the value.
    assert isinstance(masked, Approved) and "NULL AS secret" in masked.target_sql, masked


@pytest.mark.parametrize("engine, sql", [
    ("oracle", "SELECT C.ID FROM claim c"),
    ("oracle", "SELECT c.id FROM claim C"),
    ("duckdb", "SELECT C.id FROM claim c"),
    ("postgres", "SELECT C.id FROM claim c"),
    ("duckdb", "SELECT CLAIM.id FROM claim"),
])
def test_a_qualifier_is_matched_to_its_source_as_the_engine_folds_it(engine, sql):
    tree = sqlglot.parse_one(sql, read=engine)
    fields_as_members(tree, engine)
    assert tree.find(exp.Dot) is None, tree  # a column of the source, not a field read
    assert isinstance(decide(sql, GRANTED, dialect=engine, target=engine), Approved)


@pytest.mark.parametrize("engine", ["oracle", "duckdb"])
def test_a_field_through_a_qualifier_in_another_case_meets_the_deny(engine):
    verdict = decide("SELECT C.secret.status FROM claim c", GRANTED, dialect=engine, target=engine,
                     policy=DENIED)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNAUTHORIZED_COLUMN


@pytest.mark.parametrize("sql", ["SELECT secret.status FROM claim",
                                 "SELECT claim_seq.NEXTVAL FROM claim",
                                 "SELECT app.claim_seq.NEXTVAL FROM claim"])
def test_on_oracle_a_reference_not_led_by_a_source_is_left_as_written(sql):
    tree = sqlglot.parse_one(sql, read="oracle")
    fields_as_members(tree, "oracle")
    assert tree.find(exp.Dot) is None, tree
    assert isinstance(decide(sql, HIDDEN, dialect="oracle", target="oracle"), Approved)


def test_a_qualifier_naming_nothing_here_is_an_unknown_column_not_skipped():
    # Before: skipped as if it named a subquery, approved, and left for the source to reject.
    verdict = decide("SELECT clm.amount FROM claim c", GRANTED, dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN
    assert verdict.subject == "clm"


@pytest.mark.parametrize("sql, dialect", [
    ("SELECT o.id FROM claim o WHERE EXISTS (SELECT 1 FROM claim i WHERE i.id = o.id)", "duckdb"),
    ("SELECT s.claim.id FROM s.claim", "duckdb"),
    ("SELECT t.x FROM (SELECT secret AS x FROM claim) t", "duckdb"),
    ("WITH w AS (SELECT id FROM claim) SELECT w.id FROM w", "oracle"),
    ("UPDATE scratch SET amount = 0 WHERE EXISTS (SELECT 1 FROM claim WHERE claim.id = scratch.id)",
     "duckdb"),
    ("INSERT INTO claim (id, amount) VALUES (1, 2) ON CONFLICT (id) DO UPDATE "
     "SET amount = excluded.amount", "duckdb"),
    ("MERGE INTO claim t USING src s ON (t.id = s.id) WHEN MATCHED THEN UPDATE "
     "SET t.amount = s.amount", "oracle"),
])
def test_a_reference_qualified_by_a_source_is_left_as_it_is(sql, dialect):
    tree = sqlglot.parse_one(sql, read=dialect)
    before = tree.sql(dialect=dialect)
    fields_as_members(tree, dialect)
    assert tree.find(exp.Dot) is None and tree.sql(dialect=dialect) == before


def test_a_write_reads_a_field_as_its_column_too():
    grants = GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))

    def write(sql, visible):
        return decide_write(sql, visible, grants, policy=AccessPolicy(), dialect="duckdb",
                            target="duckdb", writes_enabled=True)

    for sql in ("UPDATE claim c SET amount = 1 WHERE c.secret.status = 'x'",
                # the target's field read from inside a subquery, whose scope lacks the target
                "UPDATE claim c SET amount = 1 WHERE EXISTS "
                "(SELECT 1 FROM claim d WHERE d.id = c.id AND c.secret.status = 'x')"):
        refused = write(sql, HIDDEN)
        assert isinstance(refused, Refusal) and refused.subject == "secret", (sql, refused)
        assert isinstance(write(sql, GRANTED), ApprovedWrite), sql
    upsert = ("INSERT INTO claim (id, amount) VALUES (1, 2) ON CONFLICT (id) DO UPDATE "
              "SET amount = excluded.amount")
    assert isinstance(write(upsert, HIDDEN), ApprovedWrite)


DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_a_json_column_s_field_is_read_and_now_checked_as_the_column():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    table = "m165_field_probe"

    def run(sql):
        return cur.execute(sql.replace("FROM claim", f"FROM {table}")).fetchall()

    try:
        try:
            cur.execute(f"DROP TABLE {table} PURGE")
        except oracledb.DatabaseError:
            pass
        # USERS, because a JSON column cannot live in a tablespace without automatic segment
        # space management, and SYSTEM is one (ORA-43853).
        cur.execute(f"CREATE TABLE {table} (id NUMBER, secret JSON) TABLESPACE users")
        cur.execute(f"""INSERT INTO {table} VALUES (1, JSON('{{"status": "ssn-123"}}'))""")
        cur.execute(f"""INSERT INTO {table} VALUES (20, JSON('{{"status": "ssn-456"}}'))""")
        sql = "SELECT c.secret.status FROM claim c"
        assert sorted(run(sql)) == [("ssn-123",), ("ssn-456",)]  # the premise
        assert run("SELECT C.secret.status FROM claim c") == run(sql)  # folded like any alias
        # Why a reference not led by an alias is left as written on Oracle: it is no field there.
        for not_a_field in ("SELECT secret.status FROM claim", "SELECT claim.secret.status FROM claim"):
            with pytest.raises(oracledb.DatabaseError, match="ORA-00904"):
                run(not_a_field)
        hidden = decide(sql, {"claim": {"id"}}, dialect="oracle", target="oracle")
        assert isinstance(hidden, Refusal) and hidden.subject == "secret", hidden
        granted = {"claim": {"id", "secret"}}
        filtered = AccessPolicy(row_filters={"claim": "id <= 10"}, policy_schema=granted)
        assert sorted(run(decide(sql, granted, dialect="oracle", target="oracle").target_sql)) \
            == [("ssn-123",), ("ssn-456",)]
        assert run(decide(sql, granted, dialect="oracle", target="oracle",
                          policy=filtered).target_sql) == [("ssn-123",)]
    finally:
        try:
            cur.execute(f"DROP TABLE {table} PURGE")
        finally:
            con.close()
