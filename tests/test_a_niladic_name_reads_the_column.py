"""A column named like a niladic SQL function is read through the function's node, which no
grant, deny or mask check sees (M194, the ui lane's replay of #169): DuckDB binds a bare
`current_date` to a same-named column of the query's own FROM before the function, and sqlglot
reads the name as `CurrentDate` with no Column node. Refused where any table the decider knows
has a column so named, fail-closed, on the read and the write path.
"""

from __future__ import annotations

import re

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
    assert "clock_timestamp" not in verdict.message
    retry = decide(f"SELECT {call} FROM kw", {"kw": {"id"}, "sink": {"i", "s"}}, dialect=dialect, target=dialect, policy=policy)
    assert isinstance(retry, Approved), (dialect, call, retry)


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "mysql", "sqlite", "oracle"])
@pytest.mark.parametrize("taken", [{"current_date"}, {"current_timestamp"}, {"localtimestamp"},
                                   {"current_timestamp", "localtimestamp"}, {"current_date", "current_timestamp", "localtimestamp"}])
def test_the_advice_is_a_call_the_decider_approves_whatever_names_the_table_holds(dialect, taken):
    # Codex's first review of #175: Postgres's `now()` parses to the CURRENT_TIMESTAMP node,
    # so beside a column named `current_timestamp` it met the same refusal, and the
    # `clock_timestamp()` advised instead is unmodelled without an adapter's function
    # inventory, refused on the retry. The advice is verified as the decider verifies the
    # retry: a modelled call, not a niladic node whose name the table holds; where every
    # candidate's name is held, no example is given.
    from mnemiq.sql.guard import _NILADIC
    schema = {"kw": {"id", *taken}}
    policy = AccessPolicy(policy_schema=schema)
    name = sorted(taken)[0]
    verdict = decide(f"SELECT {name} FROM kw", {"kw": {"id"}}, dialect=dialect, target=dialect, policy=policy)
    if not isinstance(sqlglot.parse_one(f"SELECT {name}", read=dialect).selects[0], _NILADIC):
        # the dialect reads the spelling as a column (`localtimestamp` on SQLite): the grant check
        assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNKNOWN_COLUMN, verdict
        return
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
    advised = re.search(r"such as (.+?)\.$", verdict.message)
    if advised is None:
        assert "with its own name and parentheses" in verdict.message, verdict.message
        assert dialect == "postgres" and taken >= {"current_timestamp", "localtimestamp"}, (dialect, taken)
        return
    retry = decide(f"SELECT {advised.group(1)} FROM kw", {"kw": {"id"}}, dialect=dialect, target=dialect, policy=policy)
    assert isinstance(retry, Approved), (dialect, taken, advised.group(1), retry)


def test_a_niladic_node_is_named_by_its_class_not_its_printing(worlds):
    # The gate on the seventh commit: Oracle's `SYSDATE` parses to the CURRENT_TIMESTAMP node
    # with a time zone, printed `current_timestamp at time zone 'utc'`, a name no column
    # matched, and written as Oracle for a DuckDB target the bare CURRENT_TIMESTAMP printed
    # bound to the column there.
    schema = {"kw": {"id", "current_timestamp"}}
    verdict = decide("SELECT SYSDATE FROM kw", {"kw": {"id"}}, dialect="oracle", target="duckdb", policy=AccessPolicy(policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
    from mnemiq.sql.guard import _NILADIC, _niladic_name
    for spelling, dialect in (("SYSDATE", "oracle"), ("current_timestamp", "duckdb"), ("now()", "postgres"), ("curdate()", "mysql"), ("today()", "duckdb")):
        node = sqlglot.parse_one(f"SELECT {spelling}", read=dialect).selects[0]
        assert isinstance(node, _NILADIC) and _niladic_name(node) in {"current_timestamp", "current_date"}, (spelling, _niladic_name(node))


def test_an_unmodelled_spelling_is_never_advised(monkeypatch):
    # The guard itself, since no spelling in the tables is unmodelled any more: were one
    # listed first, as `clock_timestamp()` was, the advice passes over it.
    import mnemiq.sql.guard as guard
    monkeypatch.setitem(guard._CALL_ADVICE, "postgres", ("clock_timestamp()", "now()"))
    assert guard._call_advice("postgres", frozenset({"current_date"})) == "now()"
    assert guard._call_advice("postgres", frozenset({"current_timestamp"})) is None


def test_a_grant_and_a_policy_keyed_in_different_cases_are_one_table_to_the_rule(worlds):
    # The same review: a grant keyed `KW` beside a policy keyed `kw` read the grant's columns
    # alone, so the policy's `current_date` was unseen and the bare name approved, the
    # column's value returned on DuckDB, which folds both.
    policy = AccessPolicy(denied={("kw", "current_date")}, policy_schema={"kw": {"id", "current_date"}})
    verdict = decide("SELECT current_date FROM KW", {"KW": {"id"}}, dialect="duckdb", target="duckdb", policy=policy)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
    verdict = decide("SELECT current_date FROM kw", {"kw": {"id"}}, dialect="duckdb", target="duckdb",
                     policy=AccessPolicy(policy_schema={"KW": {"id", "current_date"}}))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict


def test_the_advice_is_in_the_dialect_the_model_writes_not_the_targets():
    # The retry is parsed as `dialect` and transpiled to `target`: told SYSTIMESTAMP for an
    # Oracle target while writing DuckDB, the model would write a column (the gate).
    verdict = decide("SELECT current_date FROM kw", HIDDEN, dialect="duckdb", target="oracle", policy=WITH_SCHEMA)
    assert isinstance(verdict, Refusal) and "now()" in verdict.message and "SYSTIMESTAMP" not in verdict.message, verdict.message


def test_current_user_prints_with_its_parentheses_and_reads_the_function():
    # Not among the names refused, for one reason: sqlglot prints `CURRENT_USER()` for DuckDB,
    # so the target SQL reads the function beside a same-named column (the ui lane's replay),
    # where the bare spelling reads the column (the gate). Measured here, on the target SQL.
    con = duckdb.connect()
    con.execute('CREATE TABLE kw (id INT, "current_user" VARCHAR); INSERT INTO kw VALUES (1, \'Z\')')
    assert con.execute("SELECT current_user FROM kw").fetchall() == [("Z",)]  # bare: the column
    schema = {"kw": {"id", "current_user"}}
    for sql in ("SELECT current_user FROM kw", "SELECT id FROM kw WHERE current_user LIKE 'Z%'"):
        verdict = decide(sql, {"kw": {"id"}}, dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=schema))
        assert isinstance(verdict, Approved), (sql, verdict)
        assert "CURRENT_USER()" in verdict.target_sql, verdict.target_sql
        assert "Z" not in str(con.execute(verdict.target_sql).fetchall())  # the function, not the column
    # The exemption rests on that printing, on every engine the deciders run as (the ui lane's
    # correction of its own replay): with the parentheses on these four, and bare on Postgres,
    # where the word is reserved and the function alone.
    for engine in ("mysql", "sqlite", "oracle"):
        verdict = decide("SELECT current_user FROM kw", {"kw": {"id"}}, dialect=engine, target=engine, policy=None)
        assert isinstance(verdict, Approved) and "CURRENT_USER()" in verdict.target_sql, (engine, verdict)
    verdict = decide("SELECT current_user FROM kw", {"kw": {"id"}}, dialect="postgres", target="postgres", policy=None)
    assert isinstance(verdict, Approved) and "CURRENT_USER" in verdict.target_sql, verdict


def test_the_tables_read_are_found_as_the_engine_binds_them():
    # A quoted CTE named like the table with the column, referenced unquoted in the same
    # spelling, which sqlglot matches: Postgres keeps the quoted name's case and folds the
    # reference, so the two differ and the base table is read, and the rule fires as the
    # engine would read the column; DuckDB folds both to one name and reads the CTE, whose
    # column list names no such column. The grant spells the table as written (its key is
    # read as written, a rule of its own).
    schema = {"Kw": {"id", "current_date"}, "sink": {"i", "s"}}
    sql = 'WITH "Kw" AS (SELECT id FROM sink) SELECT current_date FROM Kw'
    on_postgres = decide(sql, {"Kw": {"id"}, "sink": {"i", "s"}}, dialect="duckdb", target="postgres", policy=AccessPolicy(policy_schema=schema))
    assert isinstance(on_postgres, Refusal) and on_postgres.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, on_postgres
    on_duckdb = decide(sql, {"Kw": {"id"}, "sink": {"i", "s"}}, dialect="duckdb", target="duckdb", policy=AccessPolicy(policy_schema=schema))
    assert not (isinstance(on_duckdb, Refusal) and on_duckdb.code is RefusalCode.AMBIGUOUS_NILADIC_NAME), on_duckdb


def test_the_column_is_matched_as_the_engine_folds_it():
    schema = {"kw": {"id", "Current_Date"}, "sink": {"i", "s"}}
    verdict = _decide("SELECT current_date FROM kw", {"kw": {"id"}, "sink": {"i", "s"}}, AccessPolicy(policy_schema=schema))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.AMBIGUOUS_NILADIC_NAME, verdict
