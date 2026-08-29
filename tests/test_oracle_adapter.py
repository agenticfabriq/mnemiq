"""The Oracle adapter, against a real Oracle instance.

Oracle is a v1 target because the first enterprise deployment runs on it. These tests are marked
`integration` and skip without `MNEMIQ_ORACLE_TEST_DSN`, the same posture the other live suites
take -- an adapter cannot be meaningfully tested against a mock, and every defect these caught was
one a mock would have agreed with:

  - `execute()` raised DPY-1003 on an INSERT, where sqlite3 answers `[]` and DuckDB a row count.
    `Runtime.write` runs the approved mutation through `execute()`, so an Oracle write crashed
    inside the adapter.
  - `oracledb` does not autocommit, so a write executed, returned cleanly, and was rolled back on
    close -- data loss reported to the caller as success.
  - Oracle has no session-level read-only switch at all. `ALTER SESSION SET READ ONLY = TRUE` is
    ORA-02248 and `ALTER SESSION ENABLE READ ONLY` is ORA-00922. The mechanism that works,
    `SET TRANSACTION READ ONLY`, is scoped to the transaction and LAPSES at the first commit.

Run against the container the spike used:

    docker run -d --name mnemiq-oracle -p 1522:1521 -e ORACLE_PASSWORD=svc \\
      -e APP_USER=appuser -e APP_USER_PASSWORD=apppw gvenzl/oracle-free:23-slim-faststart
    MNEMIQ_ORACLE_TEST_DSN=127.0.0.1:1522/FREEPDB1 MNEMIQ_ORACLE_TEST_USER=appuser \\
      MNEMIQ_ORACLE_TEST_PASSWORD=apppw uv run pytest tests/test_oracle_adapter.py -m integration
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.integration

DSN = os.environ.get("MNEMIQ_ORACLE_TEST_DSN")
USER = os.environ.get("MNEMIQ_ORACLE_TEST_USER", "appuser")
PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_PASSWORD", "apppw")

pytest.importorskip("oracledb", reason="the Oracle adapter needs mnemiq[oracle]")
if not DSN:
    pytest.skip("set MNEMIQ_ORACLE_TEST_DSN to run the Oracle adapter tests",
                allow_module_level=True)

from mnemiq.adapters.oracle import OracleAdapter  # noqa: E402


def _adapter(read_only: bool = True) -> OracleAdapter:
    return OracleAdapter(dsn=DSN, user=USER, password=PASSWORD, read_only=read_only)


@pytest.fixture(scope="module", autouse=True)
def schema():
    """Two tables with a real FK and a view, torn down after. Built through a WRITABLE adapter,
    which is itself a check that `read_only=False` is usable."""
    import oracledb

    w = _adapter(read_only=False)
    for stmt in ("DROP VIEW t_claim_v", "DROP TABLE t_claim CASCADE CONSTRAINTS",
                 "DROP TABLE t_region CASCADE CONSTRAINTS"):
        try:
            w.execute(stmt)
        except oracledb.DatabaseError:
            pass  # first run: nothing to drop
    w.execute("CREATE TABLE t_region (id NUMBER PRIMARY KEY, name VARCHAR2(20))")
    w.execute("CREATE TABLE t_claim (id NUMBER PRIMARY KEY, "
              "region_id NUMBER REFERENCES t_region(id), amount NUMBER)")
    w.execute("INSERT INTO t_region VALUES (1, 'west')")
    w.execute("INSERT INTO t_claim VALUES (1, 1, 100)")
    w.execute("CREATE VIEW t_claim_v AS SELECT id, amount FROM t_claim")
    # A read-only transaction pins a snapshot, and Oracle refuses to read a table whose DDL is
    # newer than it: ORA-01466, measured to clear after ~1s. The read-only adapter under test
    # would otherwise fail on every read for reasons that have nothing to do with what is being
    # tested. This is the fixture paying a documented cost of the safeguard, not hiding it --
    # `test_a_read_only_adapter_cannot_read_a_table_created_this_instant` pins the behaviour.
    time.sleep(2)
    yield
    for stmt in ("DROP VIEW t_claim_v", "DROP TABLE t_claim CASCADE CONSTRAINTS",
                 "DROP TABLE t_region CASCADE CONSTRAINTS"):
        try:
            w.execute(stmt)
        except oracledb.DatabaseError:
            pass


# -- the Protocol ----------------------------------------------------------------------------

def test_the_dialect_is_what_the_engine_transpiles_to():
    """The engine writes its plan in duckdb and transpiles to `adapter.dialect` before running,
    so this string is what makes `LIMIT 10` become `FETCH FIRST 10 ROWS ONLY`."""
    assert _adapter().dialect == "oracle"


def test_introspect_reports_the_schemas_tables():
    tables = _adapter().introspect()
    assert "T_CLAIM" in tables and "T_REGION" in tables


def test_list_columns_reports_a_type_for_every_column():
    cols = [c for c in _adapter().list_columns() if c[0] == "T_CLAIM"]
    assert {c[1] for c in cols} == {"ID", "REGION_ID", "AMOUNT"}
    assert all(c[2] and c[2] != "unknown" for c in cols)


def test_list_columns_does_not_report_views_as_tables():
    """`catalog.introspect()` builds its entire table set from `list_columns()`, NOT from
    `introspect()` -- so a view leaking in here is recorded as a table with
    `binding_type="table"`, profiled, and carried through the snapshot. `ALL_TAB_COLUMNS`
    describes views too, so this needs the join to `all_tables` rather than an owner filter.

    Measured before the fix: `T_CLAIM_V` appeared here while `introspect()` correctly omitted it.
    """
    a = _adapter()
    assert "T_CLAIM_V" in a.view_definitions()[0][0] or any(
        v[0] == "T_CLAIM_V" for v in a.view_definitions()
    ), "the fixture view must exist, or this test proves nothing"
    assert {c[0] for c in a.list_columns()} <= set(a.introspect()), (
        "list_columns() must not report an object introspect() does not call a table"
    )


def test_foreign_keys_pairs_child_and_parent_columns():
    fks = [f for f in _adapter().foreign_keys() if f[0] == "T_CLAIM"]
    assert len(fks) == 1
    from_table, from_col, to_table, to_col, cid = fks[0]
    assert (from_table, from_col, to_table, to_col) == ("T_CLAIM", "REGION_ID", "T_REGION", "ID")
    assert cid, "the constraint id groups a composite FK's columns; it must not be empty"


def test_view_definitions_reads_the_long_body():
    """`ALL_VIEWS.TEXT` is a LONG column, which cannot be filtered or read after another LONG in
    the same fetch. This is the wrinkle that looks trivial and is not."""
    views = {v[0]: v for v in _adapter().view_definitions()}
    assert "T_CLAIM_V" in views
    name, body, dialect = views["T_CLAIM_V"]
    assert "t_claim" in body.lower() and dialect == "oracle"


def test_view_definitions_raises_rather_than_reporting_no_views():
    """M52: `enrich_structural` wraps this call so a source that will not answer records
    `discover:views` as failed. The DuckDB adapter used to swallow every exception and return
    `[]`, so the pipeline's handler could never fire and an unreadable inventory was reported as
    a complete empty one. A bare except here would reintroduce that."""
    import oracledb

    a = _adapter()
    a._schema = "NO_SUCH_SCHEMA"
    assert a.view_definitions() == [], "an unknown owner legitimately has no views"

    a._con.close()  # a dead connection is 'could not ask', not 'has none'
    # `oracledb.Error`, not `DatabaseError`: a closed connection raises InterfaceError (DPY-1001),
    # which is NOT a DatabaseError subclass. The first version of this test asserted the narrower
    # class and failed on the very case it exists to pin -- the contract is that the failure
    # ESCAPES, whatever the driver calls it.
    with pytest.raises(oracledb.Error):
        a.view_definitions()


def test_execute_arrow_returns_named_columns():
    t = _adapter().execute_arrow("SELECT id, amount FROM t_claim ORDER BY id")
    assert t.column_names == ["ID", "AMOUNT"] and t.num_rows == 1


# -- read_only, which is M3's control on this adapter ------------------------------------------

def test_read_only_refuses_a_write():
    import oracledb

    with pytest.raises(oracledb.DatabaseError):
        _adapter().execute("INSERT INTO t_claim VALUES (2, 1, 200)")


def test_read_only_still_reads():
    """The control: a safeguard that also blocked reads would be a broken connection, not a
    read plane."""
    assert _adapter().execute("SELECT count(*) FROM t_claim") == [(1,)]


def test_the_read_only_transaction_does_not_lapse_across_statements():
    """The reason it is re-established PER STATEMENT rather than once at construction.

    `SET TRANSACTION READ ONLY` is scoped to the transaction: measured, a connection set
    read-only at construction is writable again after the first commit. Set-once is correct until
    the first transaction boundary and then silently not, which is the worst shape a safeguard
    can have.
    """
    import oracledb

    a = _adapter()
    a.execute("SELECT count(*) FROM t_claim")
    a._con.commit()  # the boundary that would end a construction-time SET TRANSACTION
    with pytest.raises(oracledb.DatabaseError):
        a.execute("INSERT INTO t_claim VALUES (3, 1, 300)")


def test_a_writable_adapter_commits_so_the_row_survives_the_connection():
    """`oracledb` does not autocommit. Without the commit an approved write executes, returns
    cleanly, and is rolled back on close -- data loss reported as success. Verified from a
    SEPARATE connection, because the writing connection would see its own uncommitted row."""
    w = _adapter(read_only=False)
    assert w.execute("INSERT INTO t_claim VALUES (4, 1, 400)") == [(1,)], (
        "a DML statement must report its affected-row count -- `Runtime.write` reads it for "
        "`rows_affected` -- and must not raise DPY-1003 from fetchall()"
    )
    try:
        assert _adapter().execute("SELECT count(*) FROM t_claim WHERE id = 4") == [(1,)]
    finally:
        _adapter(read_only=False).execute("DELETE FROM t_claim WHERE id = 4")


def test_a_bounded_query_does_not_leak_its_timeout_onto_the_connection():
    """`call_timeout` is set on the CONNECTION, which outlives the call, so a leak silently bounds
    every later query on the same adapter.

    The first version acquired the cursor between setting the timeout and the `try/finally` that
    restores it, so a failure in `_cursor()` -- which issues its own `SET TRANSACTION READ ONLY`
    round trip -- left the tightened value in place forever. Setting it first also meant the
    safeguard's own setup was bounded by the timeout being installed for the query.
    """
    a = _adapter()
    a.execute_arrow("SELECT 1 FROM dual", timeout_s=5)
    assert a._con.call_timeout == 0, "the bounded query left its timeout on the connection"


def test_the_timeout_is_not_left_behind_when_the_cursor_cannot_be_acquired():
    """The path that originally had no `finally` covering it.

    `_cursor()` is called before the timeout is set, so a failure there must leave the connection
    untouched. The first version of this test closed the connection to force the failure -- and
    then could not read `call_timeout` at all, because that attribute raises DPY-1001 on a closed
    connection. Failing `_cursor()` while the connection stays alive is what actually tests it.
    """
    a = _adapter()
    before = a._con.call_timeout

    def boom():
        raise RuntimeError("cursor acquisition failed")

    a._cursor = boom
    with pytest.raises(RuntimeError):
        a.execute_arrow("SELECT 1 FROM dual", timeout_s=5)
    assert a._con.call_timeout == before, "a failed acquisition changed the connection's timeout"


# `test_a_read_only_adapter_cannot_read_a_table_created_this_instant` lived here and is REMOVED,
# not moved. The behaviour is real and measured -- a read-only transaction pins a snapshot and
# Oracle refuses to read a table whose DDL is newer, ORA-01466 -- but the window is SUB-SECOND, so
# the test passed 3/3 in isolation and failed inside the full suite, where the preceding fixtures
# had spent long enough for the window to close. A test whose verdict depends on how fast the
# suite ahead of it ran is worse than no test: it fails intermittently and teaches people to
# re-run rather than read. The measurement is recorded in `OracleAdapter._cursor`'s docstring,
# where it cannot flake.


# -- M57: can this connection be trusted to have VPD applied to it? ----------------------------

@pytest.fixture
def vpd():
    """A table with two rows and a VPD policy admitting one. Torn down after.

    `DBMS_RLS` and `CREATE ANY CONTEXT` must be granted to the test user; the module skips
    cleanly if the policy cannot be created, because a VPD-less instance cannot exercise this.
    """
    import oracledb

    w = _adapter(read_only=False)
    try:
        w.execute("BEGIN DBMS_RLS.DROP_POLICY('APPUSER','T_VPD','P'); END;")
    except oracledb.DatabaseError:
        pass
    try:
        w.execute("DROP TABLE t_vpd")
    except oracledb.DatabaseError:
        pass
    w.execute("CREATE TABLE t_vpd (id NUMBER, region VARCHAR2(10))")
    w.execute("INSERT INTO t_vpd VALUES (1,'west')")
    w.execute("INSERT INTO t_vpd VALUES (2,'east')")
    w.execute("CREATE OR REPLACE FUNCTION t_west_f(s VARCHAR2, o VARCHAR2) "
              "RETURN VARCHAR2 AS BEGIN RETURN 'region = ''west'''; END;")
    w.execute("CREATE OR REPLACE FUNCTION t_null_f(s VARCHAR2, o VARCHAR2) "
              "RETURN VARCHAR2 AS BEGIN RETURN NULL; END;")
    w.execute("CREATE OR REPLACE FUNCTION t_empty_f(s VARCHAR2, o VARCHAR2) "
              "RETURN VARCHAR2 AS BEGIN RETURN ''; END;")

    def attach(fn):
        try:
            w.execute("BEGIN DBMS_RLS.DROP_POLICY('APPUSER','T_VPD','P'); END;")
        except oracledb.DatabaseError:
            pass
        w.execute(f"BEGIN DBMS_RLS.ADD_POLICY(object_schema=>'APPUSER',object_name=>'T_VPD',"
                  f"policy_name=>'P',function_schema=>'APPUSER',policy_function=>'{fn}',"
                  f"statement_types=>'SELECT'); END;")

    try:
        attach("T_WEST_F")
    except oracledb.DatabaseError as exc:
        pytest.skip(f"this instance cannot create a VPD policy: {exc}")
    time.sleep(2)  # ORA-01466: a read-only transaction cannot read DDL newer than its snapshot
    yield attach
    try:
        w.execute("BEGIN DBMS_RLS.DROP_POLICY('APPUSER','T_VPD','P'); END;")
    except oracledb.DatabaseError:
        pass
    w.execute("DROP TABLE t_vpd")


def test_a_policy_actually_restricts_the_rows(vpd):
    """The control the rest of this section rests on: without it, a `bypassing` verdict could not
    be distinguished from a policy that never worked."""
    assert _adapter().execute("SELECT count(*) FROM t_vpd") == [(1,)], "2 rows exist; 1 is visible"


def test_an_ordinary_principal_reports_attached(vpd):
    verdict, _ = _adapter().assert_enforcing()
    assert verdict == "attached"


@pytest.mark.parametrize("inert_fn", ["T_NULL_F", "T_EMPTY_F"])
def test_attached_is_not_a_claim_of_enforcement(vpd, inert_fn):
    """The finding, and the reason this is not a port of the Postgres check.

    A VPD policy function returning NULL -- or `''`, which is why both are parametrised rather
    than one measured and the other asserted in prose -- yields no predicate, all rows, while
    `ALL_POLICIES` still reports it `enable = 'YES'`. So the catalog cannot distinguish an enforcing policy from
    an inert one, and `attached` must never be read as "enforcing". This test pins the gap rather
    than pretending the check closes it: the verdict is UNCHANGED while enforcement is gone.
    """
    vpd(inert_fn)
    time.sleep(2)
    assert _adapter().execute("SELECT count(*) FROM t_vpd") == [(2,)], \
        f"{inert_fn} restricts nothing: an inert policy function yields no predicate"
    verdict, reason = _adapter().assert_enforcing()
    assert verdict == "attached", "the catalog still reports the policy enabled"
    assert "not proof of enforcement" in reason, \
        "the verdict must say what it cannot establish, since the catalog cannot tell"


def test_no_policy_at_all_is_unverifiable_not_ok():
    """A third state, not an acceptance -- the same shape as the Postgres check's
    `rls_tables = 0`. An owner with no policies has nothing enforcing row security."""
    a = _adapter()
    a._schema = "NOBODY_OWNS_THIS"
    verdict, _ = a.assert_enforcing()
    assert verdict == "unverifiable"


# The bypass verdicts need a principal that actually bypasses, which needs an admin connection to
# create. Without one these skip -- and that skip is itself the finding: mutation showed that
# neutering the EXEMPT ACCESS POLICY refusal broke NOTHING in the suite as first written, because
# every test connected as a principal that could not bypass. The verdict that matters most was
# the one verified least.

ADMIN_USER = os.environ.get("MNEMIQ_ORACLE_TEST_ADMIN_USER")
ADMIN_PASSWORD = os.environ.get("MNEMIQ_ORACLE_TEST_ADMIN_PASSWORD")
needs_admin = pytest.mark.skipif(
    not (ADMIN_USER and ADMIN_PASSWORD),
    reason="set MNEMIQ_ORACLE_TEST_ADMIN_{USER,PASSWORD} to exercise the bypass verdicts",
)


@pytest.fixture
def exempt_principal(vpd):
    """A throwaway user holding EXEMPT ACCESS POLICY, granted SELECT on the fixture table.

    Depends on `vpd` so the table exists before the grant: without the grant the principal gets
    ORA-00942 and the test proves nothing about bypassing -- it would fail for the one reason
    that is not the subject.
    """
    import oracledb

    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    cur = admin.cursor()
    for stmt in ("DROP USER t_exempt CASCADE",):
        try:
            cur.execute(stmt)
        except oracledb.DatabaseError:
            pass
    cur.execute("CREATE USER t_exempt IDENTIFIED BY pw")
    cur.execute("GRANT CREATE SESSION TO t_exempt")
    cur.execute("GRANT EXEMPT ACCESS POLICY TO t_exempt")
    cur.execute("GRANT SELECT ON appuser.t_vpd TO t_exempt")
    admin.commit()
    yield "t_exempt", "pw"
    # DROP USER fails ORA-01940 while a session is open, so the test must close its adapter --
    # it does, in its own finally. Retried once regardless, since a leaked user would make the
    # next run's CREATE fail and read as a fixture bug rather than a leak.
    for _ in range(2):
        try:
            cur.execute("DROP USER t_exempt CASCADE")
            admin.commit()
            break
        except oracledb.DatabaseError:
            time.sleep(1)
    admin.close()


@needs_admin
def test_a_principal_with_exempt_access_policy_is_refused(vpd, exempt_principal):
    """The verdict the whole check exists for, and the one mutation proved untested.

    Measured: this principal reads 2 of 2 rows through a policy admitting 1, so the bypass is
    real and not merely a privilege on paper. `SESSION_PRIVS` is the test rather than a role
    name -- an owner and a DBA-role user both showed 0 here and both had VPD applied.
    """
    user, pw = exempt_principal
    a = OracleAdapter(dsn=DSN, user=user, password=pw, schema="APPUSER")
    try:
        assert a.execute("SELECT count(*) FROM appuser.t_vpd") == [(2,)], \
            "the fixture policy admits 1 row; seeing 2 is what makes this a bypass"
        verdict, reason = a.assert_enforcing()
        assert verdict == "bypassing"
        assert "EXEMPT ACCESS POLICY" in reason
    finally:
        a._con.close()  # DROP USER in teardown fails while this session is open


@needs_admin
def test_a_sysdba_connection_is_refused(vpd):
    """SYS holds EXEMPT ACCESS POLICY implicitly, so the ONE privilege test catches it.

    There is no separate ISDBA check: an earlier draft had one, mutation showed removing it broke
    nothing, and probing showed the branch was unreachable -- every `AS SYSDBA` connection becomes
    `SESSION_USER = SYS` and holds the privilege. This test exists because that reasoning is only
    as good as the case that exercises it.
    """
    import oracledb

    a = OracleAdapter.__new__(OracleAdapter)
    a._oracledb = oracledb
    a._con = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                              mode=oracledb.AUTH_MODE_SYSDBA)
    a._schema = "APPUSER"
    a._read_only = True
    try:
        assert a.execute("SELECT count(*) FROM appuser.t_vpd") == [(2,)]
        verdict, _ = a.assert_enforcing()
        assert verdict == "bypassing"
    finally:
        a._con.close()
