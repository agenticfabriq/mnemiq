"""A column named like a niladic SQL function is read through the function's node, which no
grant, deny or mask check sees (M194, the ui lane's replay of #169): DuckDB binds a bare
`current_date` to a same-named column of the query's own FROM before the function, and sqlglot
reads the name as `CurrentDate` with no Column node. Refused where any table the decider knows
has a column so named, fail-closed, on the read and the write path.
"""

from __future__ import annotations

import duckdb
import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

NAMES = ("current_date", "current_timestamp", "current_time", "localtime", "localtimestamp",
         "session_user", "current_catalog")
SCHEMA = {"kw": {"id", *NAMES}, "claim": {"id", "secret"}, "sink": {"i", "s"}}
HIDDEN = {"kw": {"id"}, "claim": {"id", "secret"}, "sink": {"i", "s"}}
WITH_SCHEMA = AccessPolicy(policy_schema=SCHEMA)
DENIED = AccessPolicy(denied={("kw", n) for n in NAMES}, policy_schema=SCHEMA)
MASKED = AccessPolicy(masked={("kw", n) for n in NAMES}, policy_schema=SCHEMA)


@pytest.fixture
def worlds():
    """Two engines differing only in kw's withheld values: an oracle answers differently on them."""
    made = []
    for secret in ("Z", "A"):
        con = duckdb.connect()
        columns = ", ".join(f'"{n}" VARCHAR' for n in NAMES)
        values = ", ".join(f"'{secret}'" for _ in NAMES)
        con.execute(f"CREATE TABLE kw (id INT, {columns}); INSERT INTO kw VALUES (1, {values}); "
                    "CREATE TABLE claim (id INT, secret VARCHAR); INSERT INTO claim VALUES (1, 'x'); "
                    "CREATE TABLE sink (i INT, s VARCHAR)")
        made.append(con)
    return made


def _decide(sql, visible, policy=None):
    return decide(sql, visible, dialect="duckdb", target="duckdb", policy=policy)


class _OkAdapter:
    def execute(self, sql):
        return []


def _decide_write(sql, visible, policy=None):
    grants = GrantSet(frozenset(visible), writable=frozenset({"sink"}))
    return decide_write(sql, visible, grants, adapter=_OkAdapter(), dialect="duckdb", target="duckdb",
                        writes_enabled=True, policy=policy)


_READS = [
    "SELECT current_date FROM kw",
    "SELECT current_date() FROM kw",             # the call form prints bare in the target SQL
    "SELECT today() FROM kw",
    "SELECT max(localtime) FROM kw",
    "SELECT id FROM kw WHERE current_timestamp LIKE 'Z%'",
    "SELECT id FROM kw ORDER BY current_time",
    "SELECT session_user FROM kw",
    "SELECT current_catalog FROM kw",
    "SELECT localtimestamp FROM kw",
    "SELECT d.v FROM (SELECT current_date AS v FROM kw) d",
    "WITH c AS (SELECT current_date AS v FROM kw) SELECT v FROM c",
    "SELECT (SELECT max(localtime) FROM kw) AS v FROM claim",
]


@pytest.mark.parametrize("sql", _READS)
def test_a_bare_niladic_name_is_refused_where_a_known_table_has_the_column(worlds, sql):
    if "()" in sql:
        # The call form reads the function as written; the transpile prints the bare name,
        # which the engine binds to the column: the premise is the target SQL's.
        assert "CURRENT_DATE FROM kw" in sqlglot.transpile(sql, read="duckdb", write="duckdb")[0], sql
    elif "ORDER BY" not in sql:
        first, second = (con.execute(sql).fetchall() for con in worlds)
        assert first != second, (sql, first, second)  # the statement is an oracle on the withheld value
    for visible, policy in ((HIDDEN, WITH_SCHEMA), (SCHEMA, DENIED), (SCHEMA, MASKED)):
        verdict = _decide(sql, visible, policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, (sql, verdict)
        assert "kw." in verdict.message and "now()" in verdict.message


@pytest.mark.parametrize("sql", [
    "INSERT INTO sink SELECT id, current_date FROM kw",
    "INSERT INTO sink SELECT id, current_date() FROM kw",
    "UPDATE sink SET s = (SELECT max(localtime) FROM kw) WHERE i = 1",
    "DELETE FROM sink WHERE s IN (SELECT id::VARCHAR FROM kw WHERE current_timestamp LIKE 'Z%')",
])
def test_the_write_path_refuses_the_same_names(sql):
    for visible, policy in ((HIDDEN, WITH_SCHEMA), (SCHEMA, DENIED)):
        verdict = _decide_write(sql, visible, policy)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, (sql, verdict)


def test_the_qualified_and_the_quoted_column_are_columns_and_checked_as_before(worlds):
    for sql in ("SELECT k.current_date FROM kw k", 'SELECT "current_date" FROM kw'):
        assert isinstance(_decide(sql, HIDDEN, WITH_SCHEMA), Refusal), sql
        verdict = _decide(sql, SCHEMA, DENIED)
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNAUTHORIZED_COLUMN, (sql, verdict)
        granted = _decide(sql, SCHEMA, WITH_SCHEMA)
        assert isinstance(granted, Approved), (sql, granted)


def test_where_no_known_table_has_the_column_the_name_is_the_function(worlds):
    schema = {"claim": {"id", "secret"}, "sink": {"i", "s"}}
    for sql in ("SELECT current_date FROM claim", "SELECT id FROM claim WHERE current_timestamp > '2000-01-01'"):
        verdict = _decide(sql, schema, AccessPolicy(policy_schema=schema))
        assert isinstance(verdict, Approved), (sql, verdict)
        assert [con.execute(verdict.target_sql).fetchall() for con in worlds][0] == [con.execute(verdict.target_sql).fetchall() for con in worlds][1]
    write = _decide_write("INSERT INTO sink SELECT id, current_date FROM claim", schema, AccessPolicy(policy_schema=schema))
    assert isinstance(write, ApprovedWrite), write


def test_a_table_the_statement_does_not_read_fires_nothing(worlds):
    # DuckDB binds the column only from the query's own FROM: with kw elsewhere in the schema,
    # `SELECT current_date FROM claim` reads the function, and a query with no FROM at all too
    # (the ui lane's replay of #175, where the broader form over every known table refused
    # both).
    for sql in ("SELECT current_date FROM claim", "SELECT current_date"):
        verdict = _decide(sql, HIDDEN, WITH_SCHEMA)
        assert isinstance(verdict, Approved), (sql, verdict)
        assert [con.execute(verdict.target_sql).fetchall() for con in worlds][0] == [con.execute(verdict.target_sql).fetchall() for con in worlds][1]


@pytest.mark.parametrize("dialect, call", [("duckdb", "now()"), ("postgres", "now()"), ("mysql", "now()"),
                                           ("sqlite", "datetime('now')"), ("oracle", "SYSTIMESTAMP")])
def test_the_repair_names_a_call_the_decider_approves_in_that_dialect(dialect, call):
    # The message is what the model retries with (the repair loop), so the advised call must
    # be approved by the decider as the retry, with no adapter's function inventory, in the
    # dialect the query is written in: measured, `get_current_timestamp()` on DuckDB and
    # every plain call on Postgres are unmodelled there, `today()`, `curdate()` and `SYSDATE`
    # parse to refused nodes, and `current_date()` reads the column on DuckDB (the ui lane).
    schema = {"kw": {"id", "current_date"}, "sink": {"i", "s"}}
    policy = AccessPolicy(policy_schema=schema)
    verdict = decide("SELECT current_date FROM kw", {"kw": {"id"}, "sink": {"i", "s"}}, dialect=dialect, target=dialect, policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
    assert "kw.current_date" in verdict.message and call in verdict.message, verdict.message
    assert "current_date()" not in verdict.message and "get_current_timestamp" not in verdict.message
    retry = decide(f"SELECT {call} FROM kw", {"kw": {"id"}, "sink": {"i", "s"}}, dialect=dialect, target=dialect, policy=policy)
    assert isinstance(retry, Approved), (dialect, call, retry)


def test_postgres_is_told_clock_timestamp_where_now_is_the_refused_name():
    # Postgres's `now()` parses to the CURRENT_TIMESTAMP node, so beside a column named
    # `current_timestamp` it would meet the same refusal; `clock_timestamp()` is a plain call
    # there, modelled only with an adapter's function inventory.
    schema = {"kw": {"id", "current_timestamp"}}
    verdict = decide("SELECT current_timestamp FROM kw", {"kw": {"id"}}, dialect="postgres", target="postgres",
                     policy=AccessPolicy(policy_schema=schema))
    assert isinstance(verdict, Refusal) and "clock_timestamp()" in verdict.message and "now()" not in verdict.message, verdict


def test_the_advice_is_in_the_dialect_the_model_writes_not_the_targets():
    # The retry is parsed as `dialect` and transpiled to `target`: told SYSTIMESTAMP for an
    # Oracle target while writing DuckDB, the model would write a column (the gate).
    verdict = decide("SELECT current_date FROM kw", HIDDEN, dialect="duckdb", target="oracle", policy=WITH_SCHEMA)
    assert isinstance(verdict, Refusal) and "now()" in verdict.message and "SYSTIMESTAMP" not in verdict.message, verdict.message


def test_current_user_reads_the_function_and_is_not_refused(worlds):
    # Measured by the ui lane on DuckDB and in the register on Oracle: `current_user` reads
    # the function beside a same-named column, so it is no longer among the names refused.
    schema = {"kw": {"id", "current_user"}}
    for sql in ("SELECT current_user FROM kw", "SELECT id FROM kw WHERE current_user LIKE 'Z%'"):
        verdict = decide(sql, {"kw": {"id"}}, dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=schema))
        assert isinstance(verdict, Approved), (sql, verdict)


def test_the_column_is_matched_as_the_engine_folds_it():
    schema = {"kw": {"id", "Current_Date"}, "sink": {"i", "s"}}
    verdict = _decide("SELECT current_date FROM kw", {"kw": {"id"}, "sink": {"i", "s"}}, AccessPolicy(policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
