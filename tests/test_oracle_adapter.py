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
    # There is deliberately NO sleep here, and its absence is load-bearing. A read-only
    # transaction pins a snapshot and Oracle refuses to read an object whose definition changed
    # just BEFORE that transaction began (ORA-01466), so this fixture used to sleep 2s before
    # yielding -- which meant every read test
    # below ran against a schema old enough to dodge the race, and the adapter's behaviour inside
    # the window went untested by anything. `_with_cursor` retries it now, so the fixture hands
    # over tables created microseconds ago and every read test is also a test of that retry.
    # (The comment here previously pointed at a test that had been deleted for flakiness.)
    yield
    # Teardown opens a FRESH adapter rather than reusing the one that built the schema. A
    # module-scoped connection cannot survive this module: one test closes and reopens the PDB to
    # measure the open mode, and `CLOSE IMMEDIATE` kills every session including this fixture's.
    # Reusing `w` here raised DPY-1001 -- an InterfaceError, which the `DatabaseError` below does
    # not catch -- so the drops were skipped and the error surfaced as a teardown failure against a
    # test that had passed.
    t = _adapter(read_only=False)
    for stmt in ("DROP VIEW t_claim_v", "DROP TABLE t_claim CASCADE CONSTRAINTS",
                 "DROP TABLE t_region CASCADE CONSTRAINTS"):
        try:
            t.execute(stmt)
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
    """TWO independent layers refuse this, and each is tested separately below, because the outer
    one can hide a failure of the inner one. This asserts the outer: the adapter's own gate."""
    from mnemiq.adapters.oracle import ReadOnlyViolation

    with pytest.raises(ReadOnlyViolation):
        _adapter().execute("INSERT INTO t_claim VALUES (2, 1, 200)")


def test_oracles_own_read_only_transaction_also_refuses_a_write():
    """The INNER layer, reached by going around the gate on purpose.

    The adapter's gate exists because Oracle's read-only transaction does not stop DDL. It stops
    DML perfectly well, and that is a separate guarantee worth keeping tested -- otherwise the
    gate becomes the only thing anyone checks and a lapsed SET TRANSACTION would go unnoticed
    behind it.
    """
    import oracledb

    a = _adapter()
    cur = a._cursor()  # deliberately NOT execute(): we are testing the layer underneath the gate
    try:
        with pytest.raises(oracledb.DatabaseError, match="ORA-01456"):
            cur.execute("INSERT INTO t_claim VALUES (2, 1, 200)")
    finally:
        cur.close()


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

    # Through `_cursor()`, NOT `execute()`. The adapter's read-only gate would refuse this INSERT
    # before Oracle ever saw it, so routing the probe through `execute()` would leave this test
    # green even if the transaction HAD lapsed -- the outer layer masking the very failure this
    # test exists to catch. ORA-01456 is the inner layer answering, which is the assertion.
    cur = a._cursor()
    try:
        with pytest.raises(oracledb.DatabaseError, match="ORA-01456"):
            cur.execute("INSERT INTO t_claim VALUES (3, 1, 300)")
    finally:
        cur.close()


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
# Oracle refuses to read an object whose definition changed just before it began, ORA-01466 (NOT
# "DDL newer than the snapshot": that phrasing was retracted, see `_cursor`) -- and the window is
# SUB-SECOND, so
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
    # No sleep. There was a `time.sleep(2)` here for ORA-01466, and it was load-bearing: measured,
    # DBMS_RLS.ADD_POLICY followed by an immediate UNRETRIED read hit the race 12 times out of 12.
    # `_with_cursor` covers it now, so every VPD test below reads through a policy attached
    # microseconds earlier and is therefore also a live test of that retry. Three more sleeps of
    # the same kind were removed with this one; a review asked why one was going and three were
    # staying, and the honest answer was that all four should go.
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


def test_partial_coverage_is_not_reported_as_attached(vpd):
    """The fixture schema has several tables and a policy on one, which is the ordinary state of
    a schema mid-rollout -- and the honest verdict is `partial`, not `attached`.

    A schema-level policy COUNT reported `attached` here while the tables without a policy
    returned every row: measured, a policy on one table and none on the queried one gave
    `attached` for a query that was completely ungoverned. Coverage is per table now, and the
    reason string carries the numbers because "1 of 4 governed" is what an operator acts on.
    """
    verdict, reason = _adapter().assert_enforcing()
    assert verdict == "partial", "some tables are governed and some are not"
    assert "ungoverned" in reason


@pytest.mark.parametrize("inert_fn", ["T_NULL_F", "T_EMPTY_F"])
def test_attached_is_not_a_claim_of_enforcement(vpd, inert_fn):
    """The finding, and the reason this is not a port of the Postgres check.

    A VPD policy function returning NULL -- or `''`, which is why both are parametrised rather
    than one measured and the other asserted in prose -- yields no predicate, all rows, while
    `ALL_POLICIES` still reports it `enable = 'YES'`. So the catalog cannot distinguish an enforcing policy from
    an inert one, and `attached` must never be read as "enforcing". This test pins the gap rather
    than pretending the check closes it: the verdict is UNCHANGED while enforcement is gone.
    """
    enforcing_verdict, _ = _adapter().assert_enforcing()
    assert _adapter().execute("SELECT count(*) FROM t_vpd") == [(1,)], "the policy restricts"

    vpd(inert_fn)
    assert _adapter().execute("SELECT count(*) FROM t_vpd") == [(2,)], \
        f"{inert_fn} restricts nothing: an inert policy function yields no predicate"

    inert_verdict, reason = _adapter().assert_enforcing()
    assert inert_verdict == enforcing_verdict, (
        "the verdict is IDENTICAL while enforcement is gone -- that is the finding. The catalog "
        "reports the inert policy enabled, so enumerating it cannot tell the two apart"
    )
    assert "not proof of enforcement" in reason.lower(), \
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



def _drop_user(cur, username):
    """Drop a throwaway test user, or RAISE saying what is left behind.

    ONE implementation, because there were two and the second silently reacquired the flaw the
    first had just been hardened against: a teardown that gives up quietly leaves a standing
    privilege behind while reporting success. These users hold EXEMPT ACCESS POLICY or own
    governed tables, so a leak is not untidiness.

    `DROP USER` raises ORA-01940 while any session is open, so the sessions are killed on the
    second attempt rather than waited out -- a test that failed before closing its own connection
    would otherwise leak deterministically. ORA-01918, "user does not exist", is treated as
    SUCCESS: the postcondition is that no such user remains, and one that was never created meets
    it. That also makes this callable as a pre-test clean slate, not only as a teardown.

    **It warns rather than raises when another exception is already propagating**, and the two
    call sites reach that safely by different routes -- which is worth stating, because the
    obvious reading is wrong.

    From a plain `try/finally` inside a test body, a raise REPLACES the exception that brought us
    in: the real failure survives only as a chained `__context__`. `sys.exc_info()` is populated
    there, so the guard fires and the leak degrades to a warning. Verified.

    From a generator FIXTURE teardown, `sys.exc_info()` is `(None, None, None)` even when the
    test failed -- verified against this repo's pytest -- so the guard does NOT fire and this
    raises. That is correct, and not by luck: pytest reports a teardown error SEPARATELY from the
    test failure, so a double failure prints both `FAILED ... THE REAL TEST FAILURE` and
    `ERROR at teardown ... FAILED TO DROP`. Nothing is masked, and the leak keeps its own line.

    So the guard is load-bearing at one site and inert at the other, deliberately. A review found
    the inert half and predicted masking; the masking does not happen, but the reasoning in this
    docstring did not survive contact with either fact and has been rewritten to the measurements.
    """
    import sys
    import warnings

    import oracledb

    last = None
    for attempt in range(3):
        try:
            cur.execute(f"DROP USER {username} CASCADE")
            cur.connection.commit()
            return
        except oracledb.DatabaseError as exc:
            # ORA-01918, "user does not exist", is SUCCESS. This function's postcondition is that
            # no such user is left behind, and a user that was never created satisfies it. Raising
            # here made the helper usable only in teardown, so a test that also wanted a clean
            # slate BEFORE creating its user could not call it -- and one did, passing on the run
            # where a previous run had leaked and failing on the run where it had not.
            # By CODE, not by substring, for the reason `_is_ddl_race` gives in its own docstring:
            # a rendered message that merely quotes the number would satisfy a substring test, and
            # returning here on a wrapped error whose top-level failure is something else would
            # report "user gone" while it is still there -- the exact postcondition this function
            # exists to guarantee. The string is kept only as a fallback for an error re-raised
            # without the driver's error object.
            #
            # **Deliberately NARROWER than `_is_ddl_race`, which falls back unconditionally**, and
            # the asymmetry is about consequence rather than style. A false positive there costs a
            # retry. A false positive HERE declares a privileged account gone -- these users hold
            # EXEMPT ACCESS POLICY or own governed tables -- and reports a clean teardown while
            # the leak stands. A review asked for the two to be made identical; they were, and the
            # gate then blocked it for reintroducing exactly this. Same shape, different blast
            # radius, so they stay different on purpose.
            err = exc.args[0] if exc.args else None
            if getattr(err, "code", None) == 1918 or (
                err is None and "ORA-01918" in str(exc)
            ):
                return
            last = exc
            if attempt == 1:
                for sid, serial in cur.execute(
                    "SELECT sid, serial# FROM v$session WHERE username = :u",
                    u=username.upper(),
                ).fetchall():
                    try:
                        cur.execute(f"ALTER SYSTEM KILL SESSION '{sid},{serial}' IMMEDIATE")
                    except oracledb.DatabaseError:
                        pass
            time.sleep(1)
    message = (
        f"FAILED TO DROP the test user {username}, which this suite grants privileges to. "
        f"Remove it by hand: DROP USER {username} CASCADE. Underlying error: {last}"
    )
    if sys.exc_info()[0] is not None:  # something else is already failing; do not mask it
        warnings.warn(message, stacklevel=2)
        return
    raise AssertionError(message)


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
    try:
        _drop_user(cur, "t_exempt")
    finally:
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

    a = OracleAdapter.over(
        oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                         mode=oracledb.AUTH_MODE_SYSDBA),
        oracledb, schema="APPUSER",
    )
    try:
        assert a.execute("SELECT count(*) FROM appuser.t_vpd") == [(2,)]
        verdict, _ = a.assert_enforcing()
        assert verdict == "bypassing"
    finally:
        a._con.close()


def test_a_policy_that_does_not_apply_to_select_is_not_counted_as_governing(vpd):
    """`ALL_POLICIES.SEL` is the filter, and mutation showed it was the untested one.

    A VPD policy can be attached for UPDATE or DELETE only -- `sel = 'NO'` -- and it restricts no
    SELECT whatsoever. Counting it as governing would report a table as covered while every read
    returns every row: the same false positive as counting policies per schema, one column over.

    Oracle quirk worth knowing: `statement_types => 'INSERT'` and any combination containing
    INSERT is rejected with ORA-28104, while 'UPDATE', 'DELETE' and 'UPDATE,DELETE' are accepted.
    So this shape is reachable, which is why the filter is not dead code the way the removed
    `ISDBA` branch was.
    """
    import oracledb

    w = _adapter(read_only=False)
    try:
        w.execute("BEGIN DBMS_RLS.DROP_POLICY('APPUSER','T_VPD','P'); END;")
    except oracledb.DatabaseError:
        pass
    w.execute("BEGIN DBMS_RLS.ADD_POLICY(object_schema=>'APPUSER',object_name=>'T_VPD',"
              "policy_name=>'P',function_schema=>'APPUSER',policy_function=>'T_WEST_F',"
              "statement_types=>'UPDATE'); END;")

    a = _adapter()
    assert a.execute("SELECT count(*) FROM t_vpd") == [(2,)], \
        "an UPDATE-only policy restricts no SELECT -- both rows are visible"
    sel = a.execute("SELECT sel FROM all_policies WHERE object_name = 'T_VPD'")[0][0]
    assert sel == "NO", "the fixture must produce a non-SELECT policy, or this proves nothing"

    verdict, reason = a.assert_enforcing()
    # EXACTLY `unverifiable`, not `in (unverifiable, partial)`. The first version accepted both
    # and mutation proved it worthless: with the filter this policy counts 0 governed tables, and
    # WITHOUT it 1 -- but the schema has other ungoverned tables either way, so both land on
    # `partial` and an `in` assertion passes with the filter removed. The count is the only thing
    # that discriminates, so the count is what this asserts.
    assert verdict == "unverifiable", (
        f"an UPDATE-only policy must contribute NOTHING to SELECT coverage, leaving zero governed "
        f"tables; got {verdict} -- {reason}"
    )


@needs_admin
def test_full_coverage_reports_attached():
    """The terminal branch, `governed == tables`, which shipped untested when coverage went
    per-table: the test that used to reach `attached` was replaced by the `partial` one and
    nothing drove the schema into full coverage.

    It needs a schema where EVERY visible table is governed, which the shared fixture schema is
    not -- so this builds a throwaway owner with exactly one table and one SELECT policy. That is
    also the honest boundary: `attached` means what it says only when nothing is left out.

    It does NOT take the `vpd` fixture. An earlier version declared it and never used it: the
    coverage query filters `all_tables WHERE owner = :owner`, so APPUSER's state cannot reach
    this schema. That coupling bought nothing and inherited `vpd`'s ability to skip.
    """
    import oracledb

    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    cur = admin.cursor()
    try:
        cur.execute("DROP USER t_full CASCADE")
    except oracledb.DatabaseError:
        pass
    cur.execute("CREATE USER t_full IDENTIFIED BY pw QUOTA UNLIMITED ON users")
    cur.execute("GRANT CREATE SESSION, CREATE TABLE, CREATE PROCEDURE TO t_full")
    cur.execute("GRANT EXECUTE ON DBMS_RLS TO t_full")
    admin.commit()

    owner = OracleAdapter(dsn=DSN, user="t_full", password="pw", read_only=False)
    try:
        owner.execute("CREATE TABLE only_t (id NUMBER, region VARCHAR2(10))")
        owner.execute("INSERT INTO only_t VALUES (1,'west')")
        owner.execute("INSERT INTO only_t VALUES (2,'east')")
        owner.execute("CREATE OR REPLACE FUNCTION f_west(s VARCHAR2, o VARCHAR2) "
                      "RETURN VARCHAR2 AS BEGIN RETURN 'region = ''west'''; END;")
        owner.execute("BEGIN DBMS_RLS.ADD_POLICY(object_schema=>'T_FULL',object_name=>'ONLY_T',"
                      "policy_name=>'P',function_schema=>'T_FULL',policy_function=>'F_WEST',"
                      "statement_types=>'SELECT'); END;")

        reader = OracleAdapter(dsn=DSN, user="t_full", password="pw")
        assert reader.execute("SELECT count(*) FROM only_t") == [(1,)], \
            "the policy must actually restrict, or `attached` would be meaningless here"
        verdict, reason = reader.assert_enforcing()
        assert verdict == "attached", reason
        assert "all 1 visible tables" in reason
        assert "not proof of enforcement" in reason.lower(), \
            "even full coverage must disclaim: an inert policy function is reported enabled"
        reader._con.close()
    finally:
        owner._con.close()
        try:
            _drop_user(cur, "t_full")
        finally:
            admin.close()


def test_a_table_created_this_instant_is_readable_through_the_read_only_adapter():
    """The property the ORA-01466 retry exists for, asserted where it can be asserted honestly.

    An end-to-end enrich against a schema created moments earlier had EVERY table fail to profile,
    and the pipeline's fail-soft handler excluded each one and returned a snapshot reporting
    success with zero columns. "Provision the schema, then enrich" is what a migration pipeline
    does, so the window is reachable in production, not just in a test.

    This asserts the READ SUCCEEDS, never that the race fired. The inverse test -- that an
    unretried read fails inside the window -- was written once and deleted: the window is
    sub-second, so its verdict depended on how long the preceding tests took. The retry's own
    branches are covered deterministically in test_oracle_ddl_race.py with a synthetic ORA-01466.
    """
    import oracledb

    w = _adapter(read_only=False)
    try:
        w.execute("DROP TABLE t_fresh")
    except oracledb.DatabaseError:
        pass
    w.execute("CREATE TABLE t_fresh (id NUMBER)")
    w.execute("INSERT INTO t_fresh VALUES (7)")
    try:
        assert _adapter().execute("SELECT id FROM t_fresh") == [(7,)]
    finally:
        try:
            w.execute("DROP TABLE t_fresh")
        except oracledb.DatabaseError:
            pass


def test_the_real_drivers_ora_01466_is_recognised_by_the_retrys_predicate(vpd):
    """Ties the synthetic tests to the real driver: same error, same recogniser.

    `test_oracle_ddl_race.py` drives every retry branch with a fake exception, which proves the
    logic and NOT that the driver raises what the fake imitates. DBMS_RLS.ADD_POLICY followed by
    an immediate read provokes the real thing reliably -- 12/12 when measured -- so this re-attaches
    the policy and then reads through `_cursor()` directly, bypassing the retry, to inspect what
    comes back.

    It takes the `vpd` fixture rather than building its own policy. The first version of this test
    did the latter, and T_VPD did not exist at that point, so ADD_POLICY failed and the test
    reported a benign-looking SKIP on every run -- proving nothing while appearing to have had its
    chance. The fixture already skips cleanly on a VPD-less instance, which is the difference
    between "this instance cannot" and "the window did not open".
    """
    import oracledb

    from mnemiq.adapters.oracle import _is_ddl_race

    vpd("T_WEST_F")  # re-attach: the read below must land in the window this opens
    cur = _adapter()._cursor()  # deliberately NOT _with_cursor: we want the raw failure
    try:
        cur.execute("SELECT count(*) FROM t_vpd")
        cur.fetchall()
        pytest.skip("the DDL race did not fire on this run; nothing to inspect")
    except oracledb.DatabaseError as exc:
        assert _is_ddl_race(exc), f"the retry would not have recognised this: {exc}"
        assert exc.args[0].full_code == "ORA-01466"
        assert exc.args[0].code == 1466
    finally:
        cur.close()


# -- read_only against DDL, and the proof seam ---------------------------------------------------

@pytest.mark.parametrize("stmt", [
    "DROP TABLE t_region",
    "TRUNCATE TABLE t_region",
    "CREATE TABLE t_should_not_exist (id NUMBER)",
    "ALTER TABLE t_region ADD (extra NUMBER)",
    "GRANT SELECT ON t_region TO PUBLIC",
    "INSERT INTO t_region VALUES (9, 'north')",
    "  /* comment */ -- and a line comment\n  DROP TABLE t_region",
    "BEGIN EXECUTE IMMEDIATE 'DROP TABLE t_region'; END;",
])
def test_a_read_only_adapter_refuses_everything_that_is_not_a_read(stmt):
    """**Oracle's own read-only transaction does not stop DDL**, so this gate is not defence in
    depth -- it is the only thing enforcing the claim the constructor makes.

    Measured through this adapter with `read_only=True` before the gate existed: `INSERT` was
    refused by Oracle (ORA-01456) while `CREATE TABLE`, `TRUNCATE TABLE` and `DROP TABLE` all ran
    with no error and the table was gone afterwards. DDL performs an implicit COMMIT, which ends
    the read-only transaction, and then executes. A read plane whose backstop a DROP walks through
    is what **M3** is about.

    The comment-prefixed case is here because a prefix scan a comment can hide DDL behind is not a
    gate, and the PL/SQL block because `EXECUTE IMMEDIATE` is the obvious way around a check that
    only looks at the outermost verb.
    """
    from mnemiq.adapters.oracle import ReadOnlyViolation

    with pytest.raises(ReadOnlyViolation):
        _adapter().execute(stmt)
    assert "T_REGION" in _adapter().introspect(), "the table must still be there"


def test_a_read_only_adapter_still_reads():
    ro = _adapter()
    assert ro.execute("SELECT count(*) FROM t_region") == [(1,)]
    assert ro.execute("WITH c AS (SELECT 1 x FROM dual) SELECT x FROM c") == [(1,)]
    assert ro.execute_arrow("SELECT id FROM t_region").num_rows == 1


def test_validate_accepts_a_good_statement_and_rejects_by_the_sources_own_reason():
    """`EXPLAIN <sql>` is ORA-02000 on Oracle and `EXPLAIN PLAN FOR <sql>` writes to PLAN_TABLE,
    so it is ORA-01456 on the read-only adapter that does the proving. Both measured."""
    import oracledb

    ro = _adapter()
    ro.validate("SELECT id FROM t_region")  # must not raise

    for sql, code in [("SELECT nope FROM t_region", "ORA-00904"),
                      ("SELECT id FROM no_such_table_here", "ORA-00942"),
                      ("SELECT FROM WHERE", "ORA-00936")]:
        with pytest.raises(oracledb.DatabaseError) as exc:
            ro.validate(sql)
        assert code in str(exc.value)


def test_validate_does_not_execute_what_it_validates():
    w = _adapter(read_only=False)
    before = w.execute("SELECT count(*) FROM t_region")
    w.validate("INSERT INTO t_region VALUES (7, 'south')")
    w.validate("UPDATE t_region SET name = 'nowhere'")
    w.validate("DELETE FROM t_region")
    assert w.execute("SELECT count(*) FROM t_region") == before
    assert w.execute("SELECT name FROM t_region WHERE id = 1") == [("west",)]


@pytest.mark.parametrize("stmt", ["DROP TABLE t_region", "CREATE TABLE t_nope (id NUMBER)",
                                  "TRUNCATE TABLE t_region", "BEGIN NULL; END;"])
def test_validate_refuses_ddl_because_oracles_parser_executes_it(stmt):
    """**`cursor.parse()` EXECUTES DDL.** Measured: parsing a CREATE returned without error and the
    table existed afterwards, INSIDE a read-only transaction, because the DDL's implicit commit
    ends that transaction first. A validation seam that runs what it is asked to check is worse
    than no validation, so the allowlist is what stands between the two -- on the WRITABLE adapter
    too, which is why this runs there.
    """
    from mnemiq.adapters.oracle import ReadOnlyViolation

    with pytest.raises(ReadOnlyViolation):
        _adapter(read_only=False).validate(stmt)
    assert "T_REGION" in _adapter().introspect()
    assert "T_NOPE" not in _adapter().introspect()


def test_the_decider_proof_path_approves_a_valid_oracle_plan():
    """The end-to-end gap: the adapter's tests never traversed `prove`, and the deciders' tests
    use adapters that speak EXPLAIN, so an Oracle plan being refused before execution was invisible
    to both. This is the one assertion that would have failed."""
    from mnemiq.sql.prove import prove

    assert prove(_adapter(), "SELECT id FROM t_region") is None
    refused = prove(_adapter(), "SELECT id FROM ghost_table")
    assert refused is not None and "ORA-00942" in refused.message


def test_concurrent_reads_through_one_adapter_do_not_corrupt_each_other():
    """The deployed shape: ONE Runtime, one adapter, one connection, FastAPI's worker threadpool.

    This adapter mutates CONNECTION-wide state per statement -- `rollback()` then
    `SET TRANSACTION READ ONLY`, and `call_timeout` in `execute_arrow` -- so interleaving is not a
    slow path but a wrong one. Measured with the lock removed and everything else identical:
    **363 failures** across 8 threads, every one `ORA-01453: SET TRANSACTION must be first
    statement of transaction`, i.e. one thread's rollback landing inside another's setup. With the
    lock, 0.

    Only the passing side is asserted, because the control's failure count depends on scheduling.
    The control is recorded here rather than run: a test that needs a race to occur to pass is the
    flaky shape this file has already deleted one test for.
    """
    import threading

    a = _adapter()
    errors: list[str] = []
    counts: list[int] = []

    def go():
        for _ in range(15):
            try:
                counts.append(a.execute("SELECT count(*) FROM t_region")[0][0])
                a.execute_arrow("SELECT id FROM t_region", timeout_s=10.0)
            except Exception as exc:  # noqa: BLE001 - the point is that NOTHING escapes
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert set(counts) == {1}, "every read must see the same committed state"
    assert a._con.call_timeout == 0, "a bounded query must not leave the connection bounded"


def test_the_read_only_gate_does_not_stop_a_write_reached_through_plsql():
    """**A known, measured hole, pinned so it cannot be forgotten or silently "fixed" wrongly.**

    `_refuse_unless_read` stops direct DML and DDL and is the only thing that stops DDL. It cannot
    stop this: a function declared `PRAGMA AUTONOMOUS_TRANSACTION` runs in its OWN transaction, so
    the read-only one never applies to it. A function WITHOUT the pragma is stopped by Oracle
    itself (ORA-14551), so the pragma is the whole of the gap.

    The second half is why no statement check can close it: the call is wrapped in a VIEW, so the
    SQL this adapter sees is `SELECT n FROM v_sneaky` and contains no function name at all.
    Parsing for callables would not find it. Privilege is the control that works, which is what
    `assert_read_only` reports on and what `test_assert_read_only_says_the_gate_is_the_only_basis`
    covers.

    If this test ever starts FAILING, the hole has closed and that is good news -- but read
    `assert_read_only` before deleting it, because the likeliest cause is the test principal losing
    a privilege rather than the gate gaining a power.
    """
    import oracledb

    w = _adapter(read_only=False)
    for stmt in ("DROP VIEW v_sneaky", "DROP FUNCTION f_auto", "DROP TABLE se_probe"):
        try:
            w.execute(stmt)
        except oracledb.DatabaseError:
            pass
    w.execute("CREATE TABLE se_probe (id NUMBER)")
    w.execute("CREATE OR REPLACE FUNCTION f_auto RETURN NUMBER AS "
              "  PRAGMA AUTONOMOUS_TRANSACTION; "
              "BEGIN INSERT INTO se_probe VALUES (1); COMMIT; RETURN 1; END;")
    w.execute("CREATE VIEW v_sneaky AS SELECT f_auto AS n FROM dual")
    try:
        assert w.execute("SELECT count(*) FROM se_probe") == [(0,)]
        _adapter().execute("SELECT n FROM v_sneaky")  # the gate allows it: it is a SELECT
        assert w.execute("SELECT count(*) FROM se_probe") == [(1,)], (
            "the write did NOT happen -- if this is a real fix, update assert_read_only and this "
            "docstring; if the test principal merely lost a privilege, the hole is still open"
        )
    finally:
        for stmt in ("DROP VIEW v_sneaky", "DROP FUNCTION f_auto", "DROP TABLE se_probe"):
            try:
                w.execute(stmt)
            except oracledb.DatabaseError:
                pass


def test_assert_read_only_says_the_gate_is_the_only_basis_when_the_principal_can_write():
    """The reachable question at boot: not "is the gate on" but "could this connection write".

    The test user owns its schema, which is the common deployment shape and the one where the
    PL/SQL hole above is reachable.
    """
    verdict, detail = _adapter().assert_read_only()
    assert verdict == "gate_only"
    assert "CAN write" in detail and "AUTONOMOUS_TRANSACTION" in detail
    # It must name the narrowing AND refuse to call it sufficient. The earlier version of this
    # assertion required the string "SELECT and nothing else" and called it "the deployment fix" --
    # encoding, in a test, the premise that a SELECT-only principal cannot cause a write. Measured
    # false: one holding SELECT on a single view inserted a row.
    assert "Narrowing this principal to SELECT" in detail
    assert "does NOT make the connection unable to cause one" in detail
    assert "DATABASE's to enforce" in detail


def test_assert_read_only_does_not_answer_for_a_writable_adapter():
    verdict, _ = _adapter(read_only=False).assert_read_only()
    assert verdict == "writable"


@needs_admin
def test_assert_read_only_sees_dml_granted_through_a_role():
    """Role-granted DML is invisible to `user_tab_privs`, and it is the enterprise-standard shape.

    Measured on a principal owning nothing, holding INSERT/UPDATE/DELETE through a role:
    `user_tab_privs` returned **0 rows at all** -- not merely zero for `grantee = SESSION_USER` --
    while `role_tab_privs` joined to `session_roles` returned 3, and the principal could in fact
    insert. `session_privs` resolves roles, so the original single query had one half role-aware
    and the other not, and would have reported `constrained` for a connection that can write.

    A review found this. The half that was wrong is the half that reads object privileges, which
    is the half that matters for a read plane.
    """
    import oracledb

    owner = _adapter(read_only=False)
    for stmt in ("DROP TABLE role_target",):
        try:
            owner.execute(stmt)
        except oracledb.DatabaseError:
            pass
    owner.execute("CREATE TABLE role_target (id NUMBER)")

    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    cur = admin.cursor()
    try:
        cur.execute("DROP ROLE writer_role")
    except oracledb.DatabaseError:
        pass
    _drop_user(cur, "role_probe")
    probe = None
    try:
        cur.execute("CREATE USER role_probe IDENTIFIED BY pw")
        cur.execute("GRANT CREATE SESSION TO role_probe")
        cur.execute("CREATE ROLE writer_role")
        cur.execute(f"GRANT INSERT, UPDATE, DELETE ON {USER}.role_target TO writer_role")
        cur.execute("GRANT writer_role TO role_probe")

        probe = OracleAdapter(dsn=DSN, user="role_probe", password="pw", schema=USER)
        verdict, detail = probe.assert_read_only()
        assert verdict == "gate_only", (
            "a principal that can write through a role must not be reported constrained"
        )
        assert "0 direct, 3 through a role" in detail, (
            "the two routes are counted separately so an operator can see which one applies"
        )
    finally:
        if probe is not None:
            probe._con.close()
        try:
            cur.execute("DROP ROLE writer_role")
        except oracledb.DatabaseError:
            pass
        _drop_user(cur, "role_probe")
        cur.close()
        admin.close()
        try:
            owner.execute("DROP TABLE role_target")
        except oracledb.DatabaseError:
            pass


@needs_admin
def test_assert_read_only_tracks_every_route_by_which_a_principal_can_write():
    """Four routes and a CONTROL, because three earlier versions of this check passed against
    whichever principal happened to be in front of them.

    Each row here is a measured false verdict from a previous version:

      EXECUTE on a writing function -- reported `constrained`, and `SELECT owner.f_w FROM dual`
        inserted a row. The worst of them: EXECUTE IS the threat this method documents, a principal
        holding it owns nothing and holds no DML, and the query was blind to exactly that.
      granted through a ROLE -- `USER_TAB_PRIVS` shows no role-granted privilege, 0 rows even
        unfiltered, and the principal could insert.
      granted to PUBLIC -- invisible to every source the check read, and the principal could insert.
      SELECT only -- the control, which must NOT say `gate_only`. One version counted a PUBLIC
        INSERT held by a table in the RECYCLE BIN, so a principal with no write ability anywhere
        was told it had one; `BIN$%` is excluded for that reason. A check that can only ever return
        one verdict reports nothing.

    The control asserts `unverifiable`, not `constrained`, and the difference is the last finding
    in this sequence: **auditing the caller cannot establish read-onlyness at all.** A view
    resolves its references with the VIEW OWNER's rights, so a principal holding SELECT on one
    view -- no EXECUTE, no DML, owning nothing -- reaches a function inside it and writes. Measured
    in `test_a_definer_rights_view_writes_for_a_principal_with_only_select`. `constrained` was
    removed rather than fixed, because no principal that would legitimately be the read plane can
    reach it.
    """
    import oracledb

    owner = _adapter(read_only=False)
    users = ("p_exec", "p_role", "p_none")
    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    cur = admin.cursor()
    for stmt in ("DROP ROLE r_writer",):
        try:
            cur.execute(stmt)
        except oracledb.DatabaseError:
            pass
    for name in users:
        _drop_user(cur, name)
    made = []
    try:
        try:
            owner.execute("DROP TABLE t_route")
        except oracledb.DatabaseError:
            pass
        owner.execute("CREATE TABLE t_route (id NUMBER)")
        owner.execute("CREATE OR REPLACE FUNCTION f_route RETURN NUMBER AS "
                      "  PRAGMA AUTONOMOUS_TRANSACTION; "
                      "BEGIN INSERT INTO t_route VALUES (1); COMMIT; RETURN 1; END;")
        for name in users:
            cur.execute(f"CREATE USER {name} IDENTIFIED BY pw")
            cur.execute(f"GRANT CREATE SESSION TO {name}")
        cur.execute("CREATE ROLE r_writer")
        cur.execute("GRANT r_writer TO p_role")
        owner.execute("GRANT EXECUTE ON f_route TO p_exec")
        owner.execute(f"GRANT INSERT ON {USER}.t_route TO r_writer")
        owner.execute(f"GRANT SELECT ON {USER}.t_route TO p_none")

        def verdict(name):
            a = OracleAdapter(dsn=DSN, user=name, password="pw", schema=USER)
            made.append(a)
            return a.assert_read_only()[0]

        assert verdict("p_exec") == "gate_only", "EXECUTE alone is enough to write"
        assert verdict("p_role") == "gate_only", "role-granted DML must be seen"
        assert verdict("p_none") == "unverifiable", (
            "no write privilege found, and the schema's code is not visible to this principal, so "
            "the honest answer is that a write path cannot be ruled out"
        )

        owner.execute(f"GRANT INSERT ON {USER}.t_route TO PUBLIC")
        assert verdict("p_none") == "gate_only", "a grant to PUBLIC is usable by everyone"
        owner.execute(f"REVOKE INSERT ON {USER}.t_route FROM PUBLIC")
        assert verdict("p_none") == "unverifiable", "and it flips back when the grant goes"

        # The recycle-bin case, asserted rather than only described. A review pointed out that this
        # docstring called it the control while nothing here dropped a granted table -- a claim of
        # coverage the test did not have, which is the failure this file keeps finding elsewhere.
        owner.execute("CREATE TABLE t_dropped (id NUMBER)")
        owner.execute(f"GRANT INSERT ON {USER}.t_dropped TO PUBLIC")
        assert verdict("p_none") == "gate_only", "precondition: the live grant is counted"
        owner.execute("DROP TABLE t_dropped")  # NOT purged: it keeps its grants as BIN$...
        # The precondition, asserted rather than assumed. With `recyclebin=off` the DROP purges
        # immediately, the PUBLIC grant vanishes outright, and the assertion below then passes for
        # a reason that has nothing to do with the filter it is testing -- a test proving nothing
        # while reporting success, which is the same shape as the benign-looking SKIP this file
        # already records once. A review asked for this and was right to.
        orphaned = owner.execute(
            f"SELECT count(*) FROM all_tab_privs WHERE table_schema = '{USER.upper()}' "
            "AND privilege = 'INSERT' AND grantee = 'PUBLIC' AND table_name LIKE 'BIN$%'"
        )[0][0]
        if not orphaned:
            pytest.skip("recyclebin is off on this instance, so DROP purged the grant outright "
                        "and there is no BIN$ row for the filter to exclude")
        assert verdict("p_none") == "unverifiable", (
            "a grant held by a table in the recycle bin is not write ability -- counting it made a "
            "SELECT-only principal report gate_only, and a verdict that can only ever say "
            "gate_only is a warning nobody reads"
        )
    finally:
        for a in made:
            a._con.close()
        for name in users:
            _drop_user(cur, name)
        try:
            cur.execute("DROP ROLE r_writer")
        except oracledb.DatabaseError:
            pass
        cur.close()
        admin.close()
        for stmt in ("DROP FUNCTION f_route", "DROP TABLE t_route", "DROP TABLE t_dropped",
                     "PURGE RECYCLEBIN"):
            try:
                owner.execute(stmt)
            except oracledb.DatabaseError:
                pass


@needs_admin
def test_a_definer_rights_view_writes_for_a_principal_with_only_select():
    """The measurement that removed the `constrained` verdict.

    A view resolves its references with the VIEW OWNER's rights, so a function inside it runs as
    the owner and the caller needs no privilege on the function at all. This principal holds SELECT
    on one view and nothing else -- verified here, not assumed -- and reading that view inserts a
    row. Auditing the CALLER therefore cannot establish that a source is unwritable, which is why
    `assert_read_only` has no verdict that says so.

    The adapter's statement gate does not help: `SELECT n FROM v_def` is a read by every syntactic
    measure, and the SQL names no function.
    """
    import oracledb

    owner = _adapter(read_only=False)
    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    cur = admin.cursor()
    _drop_user(cur, "v_reader")
    probe = None
    try:
        for stmt in ("DROP VIEW v_def", "DROP FUNCTION f_def", "DROP TABLE v_target"):
            try:
                owner.execute(stmt)
            except oracledb.DatabaseError:
                pass
        owner.execute("CREATE TABLE v_target (id NUMBER)")
        owner.execute("CREATE OR REPLACE FUNCTION f_def RETURN NUMBER AS "
                      "  PRAGMA AUTONOMOUS_TRANSACTION; "
                      "BEGIN INSERT INTO v_target VALUES (1); COMMIT; RETURN 1; END;")
        owner.execute("CREATE VIEW v_def AS SELECT f_def AS n FROM dual")
        cur.execute("CREATE USER v_reader IDENTIFIED BY pw")
        cur.execute("GRANT CREATE SESSION TO v_reader")
        owner.execute(f"GRANT SELECT ON {USER}.v_def TO v_reader")

        probe = OracleAdapter(dsn=DSN, user="v_reader", password="pw", schema=USER)
        assert probe.execute(
            "SELECT count(*) FROM all_tab_privs WHERE grantee = 'V_READER' "
            "AND privilege != 'SELECT'"
        ) == [(0,)], "precondition: this principal holds SELECT and nothing else"
        assert owner.execute("SELECT count(*) FROM v_target") == [(0,)]

        assert probe.execute(f"SELECT n FROM {USER}.v_def") == [(1,)]  # the gate allows a SELECT

        assert owner.execute("SELECT count(*) FROM v_target") == [(1,)], (
            "a principal with only SELECT on a view caused a write; if this stops being true, "
            "read `assert_read_only` before relaxing anything -- the likeliest cause is the "
            "fixture losing a grant, not Oracle changing how views resolve references"
        )
        assert probe.assert_read_only()[0] == "unverifiable"
    finally:
        if probe is not None:
            probe._con.close()
        _drop_user(cur, "v_reader")
        cur.close()
        admin.close()
        for stmt in ("DROP VIEW v_def", "DROP FUNCTION f_def", "DROP TABLE v_target",
                     "PURGE RECYCLEBIN"):
            try:
                owner.execute(stmt)
            except oracledb.DatabaseError:
                pass


def test_a_tns_alias_resolves_through_the_config_dir(tmp_path):
    """The on-prem mechanism, and the same one Autonomous uses for mTLS.

    Qcell runs Oracle on-prem, where a DBA maintains `tnsnames.ora` and applications connect by
    ALIAS rather than by host and port. `config_dir` points the driver at that directory. The
    Autonomous case adds a wallet to the same directory and a password for its PEM; the alias
    machinery is identical, which is why this is not called `wallet_dir`.

    The third assertion is the control. Without it, a passing alias connection proves nothing --
    the driver could have fallen back to interpreting `mnemiq_local` as a host, or the test could
    be reaching the database by some path other than the file.
    """
    import oracledb

    host, _, service = DSN.partition("/")
    hostname, _, port = host.partition(":")
    (tmp_path / "tnsnames.ora").write_text(
        f"mnemiq_alias = (DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={hostname})"
        f"(PORT={port or 1521}))(CONNECT_DATA=(SERVICE_NAME={service})))\n"
    )

    aliased = OracleAdapter(dsn="mnemiq_alias", user=USER, password=PASSWORD,
                            config_dir=str(tmp_path))
    try:
        assert aliased.execute("SELECT 1 FROM dual") == [(1,)]
        assert aliased.introspect() == _adapter().introspect(), (
            "the alias must reach the same database as the Easy Connect path"
        )
    finally:
        aliased._con.close()

    with pytest.raises(oracledb.DatabaseError, match="DPY-4027"):
        OracleAdapter(dsn="mnemiq_alias", user=USER, password=PASSWORD)


@needs_admin
def test_a_reader_on_another_owners_schema_can_actually_read_it():
    """The least-privilege deployment this adapter's own advice recommends, and it did not work.

    Discovery filters the data dictionary by OWNER, but the SQL this adapter generates names tables
    UNQUALIFIED and Oracle resolves an unqualified name against the CONNECTING user. Measured with
    `user=READER, schema=APPUSER`: `introspect()` returned ORDERS, every read failed ORA-00942 on
    `"READER"."ORDERS"`, and `enrich_structural` produced 0 tables with outcome `unread`. So a
    principal holding SELECT on someone else's schema -- exactly what M66 tells operators to build
    -- could discover a schema it could not then query.

    `CURRENT_SCHEMA` changes name resolution only and grants nothing, so the reader still needs its
    SELECT; this asserts it reads, enriches, and passes the decider's proof seam.
    """
    import oracledb

    from mnemiq.enrichment.pipeline import enrich_structural, profile_outcome
    from mnemiq.sql.prove import prove

    owner = _adapter(read_only=False)
    admin = oracledb.connect(user=ADMIN_USER, password=ADMIN_PASSWORD, dsn=DSN,
                             mode=oracledb.AUTH_MODE_SYSDBA)
    cur = admin.cursor()
    _drop_user(cur, "x_reader")
    probe = None
    try:
        try:
            owner.execute("DROP TABLE x_orders")
        except oracledb.DatabaseError:
            pass
        owner.execute("CREATE TABLE x_orders (id NUMBER, status VARCHAR2(8))")
        owner.execute("INSERT INTO x_orders VALUES (1, 'open')")
        cur.execute("CREATE USER x_reader IDENTIFIED BY pw")
        cur.execute("GRANT CREATE SESSION TO x_reader")
        owner.execute(f"GRANT SELECT ON {USER}.x_orders TO x_reader")

        probe = OracleAdapter(dsn=DSN, user="x_reader", password="pw", schema=USER)
        assert "X_ORDERS" in probe.introspect(), "precondition: discovery sees the owner's table"
        assert probe.execute('SELECT count(*) FROM "X_ORDERS"') == [(1,)], (
            "an UNQUALIFIED read must resolve against the configured owner, not the connecting user"
        )
        snap = enrich_structural(probe, "cross_owner")
        assert profile_outcome(snap)[0] != "unread"
        assert any(c.object_id == "X_ORDERS" for c in snap.columns)
        assert prove(probe, 'SELECT count(*) FROM "X_ORDERS"') is None
    finally:
        if probe is not None:
            probe._con.close()
        _drop_user(cur, "x_reader")
        cur.close()
        admin.close()
        try:
            owner.execute("DROP TABLE x_orders")
        except oracledb.DatabaseError:
            pass


def _sysdba():
    """SYSDBA on the container, or None. The suite's own container documents ORACLE_PASSWORD=svc."""
    import oracledb
    try:
        return oracledb.connect(user="sys", password=os.environ.get("ORACLE_SYS_PASSWORD", "svc"),
                                dsn=DSN, mode=oracledb.AUTH_MODE_SYSDBA)
    except oracledb.DatabaseError:
        return None


def test_a_read_only_DATABASE_is_the_one_deployment_that_closes_the_plsql_hole():
    """M66's hole is closed by the database's open mode, and `constrained` reports it.

    M66 measured that a definer-rights view over an AUTONOMOUS_TRANSACTION function writes for a
    caller holding SELECT and nothing else, and concluded that no verdict above `unverifiable` was
    reachable. Both attempts behind that conclusion were PRIVILEGE queries, and the finding's own
    evidence is that privilege is the wrong instrument -- so the conclusion was broader than what
    was measured. Open mode is not a privilege question.

    This test flips the PDB and asserts BOTH directions, because a verdict that only ever runs
    against the passing case is not measured -- the mistake the privilege query in this same method
    made four times. It also re-measures the HOLE in both modes, so what is asserted is that the
    verdict tracks the actual write, not merely that an error code appeared.
    """
    import oracledb

    sysdba = _sysdba()
    if sysdba is None:
        pytest.skip("needs SYSDBA on the container to change the open mode")

    def pdb(sql):
        sysdba.cursor().execute(sql)

    w = _adapter(read_only=False)
    for stmt in ("DROP VIEW t_m66_v", "DROP FUNCTION f_m66", "DROP TABLE t_m66"):
        try:
            w.execute(stmt)
        except oracledb.DatabaseError:
            pass
    w.execute("CREATE TABLE t_m66 (n NUMBER)")
    w.execute("CREATE FUNCTION f_m66 RETURN NUMBER AS "
              "  PRAGMA AUTONOMOUS_TRANSACTION; "
              "BEGIN INSERT INTO t_m66 VALUES (1); COMMIT; RETURN 1; END;")
    w.execute("CREATE VIEW t_m66_v AS SELECT f_m66 AS n FROM dual")

    def wrote() -> bool:
        before = _adapter().execute("SELECT count(*) FROM t_m66")[0][0]
        try:
            _adapter().execute("SELECT n FROM t_m66_v")
        except oracledb.DatabaseError:
            pass
        return _adapter().execute("SELECT count(*) FROM t_m66")[0][0] > before

    try:
        # READ WRITE: the hole is open, and the verdict must NOT claim otherwise.
        assert wrote(), "the M66 shape must reproduce, or this test proves nothing"
        verdict, _ = _adapter().assert_read_only()
        assert verdict != "constrained", (
            f"a writable database reported {verdict!r} -- the probe must not read as an assurance")

        pdb("ALTER PLUGGABLE DATABASE CLOSE IMMEDIATE")
        pdb("ALTER PLUGGABLE DATABASE OPEN READ ONLY")

        # READ ONLY: the hole is shut by the database, and the verdict says so.
        assert not wrote(), "a read-only database must refuse the autonomous write"
        verdict, detail = _adapter().assert_read_only()
        assert verdict == "constrained", f"a read-only database reported {verdict!r}"
        assert "READ ONLY" in detail and "ORA-16000" in detail
        # ... and it is still a usable read plane, or the control is not one.
        assert _adapter().execute("SELECT count(*) FROM t_m66")[0][0] >= 1
    finally:
        try:
            pdb("ALTER PLUGGABLE DATABASE CLOSE IMMEDIATE")
            pdb("ALTER PLUGGABLE DATABASE OPEN READ WRITE")
        finally:
            sysdba.close()
        w2 = _adapter(read_only=False)
        for stmt in ("DROP VIEW t_m66_v", "DROP FUNCTION f_m66", "DROP TABLE t_m66"):
            try:
                w2.execute(stmt)
            except oracledb.DatabaseError:
                pass
