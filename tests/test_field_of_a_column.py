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
    con.execute("CREATE TABLE other (id INT, status VARCHAR, detail STRUCT(code VARCHAR))")
    con.execute("INSERT INTO other VALUES (1, 'other-status', {'code': 'other-code'})")
    con.execute("CREATE TABLE plain (id INT)")
    con.execute("INSERT INTO plain VALUES (1)")
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


# Codex's review of #162: a source named like the column. `c.secret.status` was left a column of
# the source `secret`, while both engines read it as a field of `c.secret` (measured below), so a
# withheld `secret` was read past every rule. A name does not settle it the other way either:
# DuckDB reads `s.claim.id` as schema `s`, table `claim`, with a source aliased `s` beside it, and
# Oracle calls that ambiguous (ORA-00918). So a reference led by a source, with a source named
# further along it too, is refused as ambiguous, whatever is granted.
OTHER = {"other": {"id", "status"}}
DEEPER = "SELECT c.secret.detail.code FROM claim c CROSS JOIN other secret"  # named second of three
COLLIDING = [
    "SELECT c.secret.status FROM claim c CROSS JOIN (SELECT 1 AS status) secret",
    "SELECT c.secret.status FROM claim c CROSS JOIN other secret",
    "SELECT claim.secret.status FROM claim CROSS JOIN (SELECT 1 AS status) secret",
    "SELECT c.secret.status FROM claim c CROSS JOIN (SELECT 1 AS status) SECRET",
    "WITH secret AS (SELECT 1 AS status) SELECT c.secret.status FROM claim c CROSS JOIN secret",
    "SELECT (SELECT c.secret.status FROM (SELECT 1 AS status) secret) FROM claim c",
    "SELECT id FROM claim c CROSS JOIN (SELECT 1 AS status) secret WHERE c.secret.status LIKE 'ssn%'",
]


def test_the_premise_duckdb_reads_a_colliding_reference_as_the_field(con):
    for sql in COLLIDING[:-1]:
        assert sorted(con.execute(sql).fetchall()) == [("ssn-123",), ("ssn-456",)], sql
    assert con.execute(COLLIDING[-1]).fetchall() == [(1,), (20,)]  # the WHERE read the field
    assert sorted(con.execute(DEEPER).fetchall()) == [("k-8",), ("k-9",)]


@pytest.mark.parametrize("engine, sql", [
    *(("duckdb", sql) for sql in [*COLLIDING, DEEPER]),
    ("oracle", "SELECT c.secret.status FROM claim c CROSS JOIN other secret"),
    ("oracle", "WITH secret AS (SELECT id FROM other) SELECT c.secret.status FROM claim c "
               "CROSS JOIN secret"),
])
@pytest.mark.parametrize("visible, policy", [(HIDDEN, None), (GRANTED, DENIED), (GRANTED, MASKED),
                                             (GRANTED, None)],
                         ids=["hidden", "denied", "masked", "granted"])
def test_a_reference_two_sources_could_lead_is_refused_as_ambiguous(engine, sql, visible, policy):
    verdict = decide(sql, {**visible, **OTHER}, dialect=engine, target=engine, policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.repairable and "two ways" in verdict.message, verdict
    assert verdict.subject == "secret", verdict


@pytest.mark.parametrize("sql", ["SELECT s.claim.id FROM s.claim CROSS JOIN other s",
                                 "SELECT s.claim.id FROM s.claim CROSS JOIN other S"])
def test_a_schema_named_like_a_source_alias_is_ambiguous_too(sql):
    refused = fields_as_members(sqlglot.parse_one(sql, read="duckdb"), "duckdb")
    assert isinstance(refused, Refusal) and refused.subject == "claim", refused
    assert fields_as_members(sqlglot.parse_one("SELECT s.claim.id FROM s.claim"), "duckdb") is None


@pytest.mark.parametrize("visible", [HIDDEN, GRANTED], ids=["hidden", "granted"])
def test_two_parts_read_the_source_s_column(con, visible):
    # `secret.status` with a source `secret` is that source's column: DuckDB and Oracle return its
    # 1 (measured), so two parts are left as they are -- with `claim.secret` in the same scope too,
    # since there the source of that name binds first.
    sql = "SELECT secret.status FROM claim CROSS JOIN (SELECT 1 AS status) secret"
    verdict = decide(sql, {**visible, **OTHER}, dialect="duckdb", target="duckdb", policy=None)
    assert isinstance(verdict, Approved) and con.execute(verdict.target_sql).fetchall() == [(1,), (1,)]


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
    # A source named like the column, as on the read side: ambiguous, refused.
    for sql in ("UPDATE claim c SET amount = 1 FROM other secret WHERE c.secret.status = 'x'",
                "UPDATE claim c SET amount = 1 WHERE EXISTS "
                "(SELECT 1 FROM other secret WHERE c.secret.status = 'x')"):
        ambiguous = write(sql, {**GRANTED, **OTHER})
        assert isinstance(ambiguous, Refusal) and "two ways" in ambiguous.message, (sql, ambiguous)


# M167: DuckDB binds a qualifier scope by scope, a source of that name and then a column of that
# name among the scope's sources, before it looks outward. So a qualifier naming an outer source
# reads a nearer column of that name where one exists (measured below), and Oracle reads the outer
# source (measured live below). The source set also held every CTE a scope could see, not only
# those it selects.
SCOPES = {**GRANTED, "other": {"id", "status", "detail"}, "plain": {"id"}}
NEARER = [
    "SELECT (SELECT secret.status FROM claim WHERE id = 1) FROM other secret",
    "SELECT (SELECT secret.detail.code FROM claim WHERE id = 1) FROM other secret",
    "SELECT (SELECT (SELECT secret.status FROM plain) FROM claim WHERE id = 1) FROM other secret",
    "WITH secret AS (SELECT id, status FROM other) "
    "SELECT (SELECT secret.status FROM claim WHERE id = 1) FROM secret",
    "SELECT id FROM other secret WHERE EXISTS (SELECT 1 FROM claim WHERE secret.status LIKE 'ssn%')",
]
UNSELECTED = "WITH secret AS (SELECT status FROM other) SELECT secret.status FROM claim"


def test_the_premise_duckdb_reads_the_nearer_column(con):
    for sql, read in zip(NEARER[:-1], ["ssn-123", "k-9", "ssn-123", "ssn-123"]):
        assert con.execute(sql).fetchall() == [(read,)], sql
    assert con.execute(NEARER[-1]).fetchall() == [(1,)]  # the EXISTS read claim's field
    assert sorted(con.execute(UNSELECTED).fetchall()) == [("ssn-123",), ("ssn-456",)]


@pytest.mark.parametrize("sql", NEARER)
@pytest.mark.parametrize("policy", [DENIED, MASKED, None], ids=["denied", "masked", "granted"])
def test_a_qualifier_a_nearer_column_shares_is_refused_as_ambiguous(sql, policy):
    verdict = decide(sql, SCOPES, dialect="duckdb", target="duckdb", policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
    assert verdict.repairable and "two ways" in verdict.message and verdict.subject == "secret"


@pytest.mark.parametrize("visible, policy, code", WITHHELD, ids=["hidden", "denied", "masked"])
def test_a_cte_no_from_selects_is_no_source(visible, policy, code):
    verdict = decide(UNSELECTED, {**visible, **OTHER}, dialect="duckdb", target="duckdb",
                     policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is code and verdict.subject == "secret"


def test_a_nearer_column_is_known_from_a_subquery_s_projection_and_from_the_policy(con):
    # A star reads through: DuckDB returns the field through `SELECT *` (the gate's case).
    assert con.execute("SELECT (SELECT secret.status FROM (SELECT * FROM claim WHERE id = 1) d) "
                       "FROM other secret").fetchall() == [("ssn-123",)]
    for derived in ("SELECT (SELECT secret.status FROM (SELECT secret FROM claim) d) FROM other secret",
                    "SELECT (SELECT secret.status FROM (SELECT * FROM claim) d) FROM other secret",
                    "SELECT (SELECT secret.status FROM (SELECT c.* FROM claim c) d) FROM other secret",
                    "WITH w AS (SELECT * FROM claim) SELECT (SELECT secret.status FROM w) "
                    "FROM other secret",
                    "SELECT (SELECT secret.status FROM (SELECT id FROM plain UNION ALL "
                    "SELECT * FROM claim) d) FROM other secret"):
        verdict = decide(derived, SCOPES, dialect="duckdb", target="duckdb")  # no policy
        assert isinstance(verdict, Refusal) and "two ways" in verdict.message, (derived, verdict)
    # `secret` left out of the grant, and known from the policy: its schema, or what it denies.
    told_by_policy = {**SCOPES, "claim": {"id", "amount"}}
    for policy in (DENIED, AccessPolicy(denied={("claim", "secret")}),
                   AccessPolicy(row_filters={"claim": "id <= 10"}, policy_schema=GRANTED)):
        verdict = decide(NEARER[0], told_by_policy, dialect="duckdb", target="duckdb",
                         policy=policy)
        assert isinstance(verdict, Refusal) and "two ways" in verdict.message, (policy, verdict)


# M169 (Codex's review of #164): a star's projection is what it exposes. Measured on DuckDB:
# `t.*` is t's columns, EXCLUDE drops a column (folded, `EXCLUDE (SECRET)` too), RENAME renames
# one, REPLACE keeps every name, and COLUMNS(...) picks by pattern.
STAR_EXPOSES_NO_SECRET = [
    "SELECT p.* FROM plain p CROSS JOIN claim c",
    "SELECT * EXCLUDE (secret) FROM claim",
    "SELECT * RENAME (secret AS s2) FROM claim",
    "SELECT c.* EXCLUDE (secret) FROM plain p CROSS JOIN claim c",
    "SELECT * EXCLUDE (secret) FROM claim c, claim k",             # unqualified: every source
    "SELECT * EXCLUDE (c.secret, k.secret) FROM claim c, claim k",
    "SELECT COUNT(*) FROM claim",                                  # a star in a call is no star
    # A RENAME of a column the picked source lacks is ignored (measured; Codex's review of #165).
    # Qualified, `p.amount`, the column check refuses the name itself, so only the bare one is here.
    "SELECT p.* RENAME (amount AS secret, secret AS s2) FROM plain p CROSS JOIN claim c",
    "SELECT p.* RENAME (amount AS secret, id AS k) FROM plain p CROSS JOIN claim c",  # one applies
]
STAR_EXPOSES_SECRET = [
    "SELECT c.* FROM plain p CROSS JOIN claim c",
    "SELECT * EXCLUDE (id) FROM claim",
    "SELECT * REPLACE (id + 1 AS id) FROM claim",
    "SELECT * RENAME (id AS secret) FROM plain",  # DuckDB then fails on `.status`: refused first
    # Every RENAME applies to the original names at once: `secret` (the old `id`) stays, in
    # either order (measured; Codex's review of #165).
    "SELECT * RENAME (id AS secret, secret AS s2) FROM claim",
    "SELECT * RENAME (secret AS s2, id AS secret) FROM claim",
    "SELECT zz.* FROM claim c",                   # names no source: read as every source's
    "SELECT * EXCLUDE (c.secret) FROM claim c, claim k",       # k.secret stays (the gate's case)
    "SELECT * RENAME (c.secret AS s2) FROM claim c, claim k",
]


def _through(projection):
    return f"SELECT (SELECT secret.status FROM ({projection}) d LIMIT 1) FROM other secret"


@pytest.mark.parametrize("projection", STAR_EXPOSES_NO_SECRET)
def test_a_star_exposing_no_nearer_column_leaves_the_outer_source_read(con, projection):
    sql = _through(projection)
    assert con.execute(sql).fetchall() == [("other-status",)]  # the premise
    # No policy: a RENAME of `secret` names the column, which a deny would refuse on its own.
    verdict = decide(sql, SCOPES, dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Approved), (projection, verdict)
    assert con.execute(verdict.target_sql).fetchall() == [("other-status",)]


@pytest.mark.parametrize("projection", STAR_EXPOSES_SECRET)
def test_a_star_exposing_the_nearer_column_is_still_ambiguous(projection):
    # No policy, so the only refusal left is the ambiguity's: a deny would refuse `c.secret` in
    # an EXCLUDE list on its own, and a leak would read as Approved.
    verdict = decide(_through(projection), SCOPES, dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Refusal) and "two ways" in verdict.message, (projection, verdict)


def test_columns_by_pattern_are_read_as_every_column():
    tree = sqlglot.parse_one(_through("SELECT COLUMNS('sec.*') FROM claim"), read="duckdb")
    refused = fields_as_members(tree, "duckdb", SCOPES)
    assert isinstance(refused, Refusal) and "two ways" in refused.message, refused


def test_an_exclude_is_folded_as_duckdb_folds_it():
    # DuckDB reads `EXCLUDE (SECRET)` as excluding `secret` (measured). The column check later
    # refuses the upper-case spelling on its own, so this is asked of the rewrite alone.
    tree = sqlglot.parse_one(_through("SELECT * EXCLUDE (SECRET) FROM claim"), read="duckdb")
    assert fields_as_members(tree, "duckdb", SCOPES) is None


def test_a_scope_whose_sources_share_a_name_is_refused(con):
    # DuckDB runs it, and sqlglot cannot say which source a qualifier there names.
    sql = "SELECT 1 FROM claim c, (SELECT 1) c"
    assert con.execute(sql).fetchall()
    verdict = decide(sql, SCOPES, dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNMODELLED_SOURCE, verdict
    assert verdict.repairable


def test_a_statement_with_no_scopes_counts_no_unselected_cte_as_a_source():
    # Such a write is refused elsewhere (unscoped_cte); the fallback reads it right regardless.
    tree = sqlglot.parse_one("WITH secret AS (SELECT status FROM other) "
                             "UPDATE claim SET amount = 1 WHERE secret.status = 'x'", read="duckdb")
    assert fields_as_members(tree, "duckdb") is None and tree.find(exp.Dot) is not None, tree


def test_an_outer_qualifier_no_nearer_column_shares_reads_the_outer_source(con):
    outer = decide("SELECT (SELECT secret.status FROM plain) FROM other secret", SCOPES,
                   dialect="duckdb", target="duckdb", policy=DENIED)
    assert isinstance(outer, Approved) and con.execute(outer.target_sql).fetchall() == [
        ("other-status",)]
    correlated = "SELECT o.id FROM claim o WHERE EXISTS (SELECT 1 FROM claim i WHERE i.id = o.id)"
    assert isinstance(decide(correlated, SCOPES, dialect="duckdb", target="duckdb"), Approved)


def test_on_oracle_the_outer_source_is_read_so_it_is_approved():
    verdict = decide(NEARER[0], SCOPES, dialect="oracle", target="oracle", policy=DENIED)
    assert isinstance(verdict, Approved), verdict


def test_a_write_s_subquery_binds_the_nearer_column_too():
    grants = GrantSet(frozenset({"claim", "other"}), writable=frozenset({"other"}))
    verdict = decide_write("UPDATE other secret SET id = 2 WHERE EXISTS "
                           "(SELECT 1 FROM claim WHERE secret.status = 'x')", SCOPES, grants,
                           policy=DENIED, dialect="duckdb", target="duckdb", writes_enabled=True)
    assert isinstance(verdict, Refusal) and "two ways" in verdict.message, verdict


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


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_a_colliding_reference_reads_the_field_and_is_refused():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    table, other = "m165_collide_claim", "m165_collide_other"

    def run(sql):
        sql = sql.replace("FROM claim", f"FROM {table}").replace("JOIN other", f"JOIN {other}")
        return sorted(cur.execute(sql).fetchall())

    try:
        for name in (table, other):
            try:
                cur.execute(f"DROP TABLE {name} PURGE")
            except oracledb.DatabaseError:
                pass
        cur.execute(f"CREATE TABLE {table} (id NUMBER, secret JSON) TABLESPACE users")
        cur.execute(f"""INSERT INTO {table} VALUES (1, JSON('{{"status": "ssn-123"}}'))""")
        cur.execute(f"CREATE TABLE {other} (id NUMBER, status VARCHAR2(20))")
        cur.execute(f"INSERT INTO {other} VALUES (1, 'other-status')")
        sql = "SELECT c.secret.status FROM claim c CROSS JOIN other secret"
        assert run(sql) == [("ssn-123",)]  # the premise: the field, not `secret`'s own column
        verdict = decide(sql, {"claim": {"id"}, **OTHER}, dialect="oracle", target="oracle")
        assert isinstance(verdict, Refusal) and "two ways" in verdict.message, verdict
    finally:
        try:
            for name in (table, other):
                cur.execute(f"DROP TABLE {name} PURGE")
        finally:
            con.close()


@pytest.mark.integration
@pytest.mark.skipif(not DSN, reason="set MNEMIQ_ORACLE_TEST_DSN to run against Oracle")
def test_on_oracle_a_qualifier_reads_the_outer_source_before_a_nearer_column():
    oracledb = pytest.importorskip("oracledb")
    con = oracledb.connect(user=os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser"),
                           password=os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw"), dsn=DSN)
    cur = con.cursor()
    table, other = "m167_bind_claim", "m167_bind_other"

    def run(sql):
        sql = sql.replace("FROM claim", f"FROM {table}").replace("FROM other", f"FROM {other}")
        return cur.execute(sql).fetchall()

    try:
        for name in (table, other):
            try:
                cur.execute(f"DROP TABLE {name} PURGE")
            except oracledb.DatabaseError:
                pass
        cur.execute(f"CREATE TABLE {table} (id NUMBER, secret JSON) TABLESPACE users")
        cur.execute(f"""INSERT INTO {table} VALUES (1, JSON('{{"status": "ssn-123"}}'))""")
        cur.execute(f"CREATE TABLE {other} (id NUMBER, status VARCHAR2(20))")
        cur.execute(f"INSERT INTO {other} VALUES (1, 'other-status')")
        assert run(NEARER[0]) == [("other-status",)]  # the premise: the outer source's column
        verdict = decide(NEARER[0], SCOPES, dialect="oracle", target="oracle", policy=DENIED)
        assert run(verdict.target_sql) == [("other-status",)]
    finally:
        try:
            for name in (table, other):
                cur.execute(f"DROP TABLE {name} PURGE")
        finally:
            con.close()
