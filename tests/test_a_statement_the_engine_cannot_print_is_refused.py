"""A statement sqlglot parses but cannot print for the engine that runs it is refused as unprintable, never raised
out of the decider.

`DATE()` with no argument, SQLite's "today", parses as DuckDB, and sqlglot's DuckDB printer then casts the missing
argument and raises. `niladic_names` reads the words off the statement as printed for the target, and on a failure
prints it for the dialect it was written in; when the two are one engine, that print raised again, out of `decide`,
and the server answered 500 where it should have refused with a reason. Found on a BIRD question written with
`STRFTIME('%Y', DATE())`."""

from __future__ import annotations

import pytest
import sqlglot

from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Refusal, RefusalCode

VISIBLE = {"games": {"id", "played_on", "season"}}


def test_sqlglot_still_cannot_print_a_bare_date_call_for_duckdb():
    # The tests below need a statement no engine's printer accepts. If sqlglot learns to print this one, they pass
    # without reaching the failure, so this one fails first and says to find another.
    with pytest.raises(sqlglot.errors.ParseError):
        sqlglot.parse_one("SELECT DATE() AS d", read="duckdb").sql(dialect="duckdb")


@pytest.mark.parametrize("sql", [
    "SELECT DATE() AS d FROM games",
    "SELECT STRFTIME('%Y', DATE()) - season AS years FROM games",
])
def test_a_statement_its_own_engine_cannot_print_is_refused_not_raised(sql):
    verdict = decide(sql, VISIBLE, dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNPRINTABLE_FOR_TARGET, verdict
    assert "duckdb" in verdict.message, verdict


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_written_in_another_dialect_it_is_still_refused_for_the_engine_that_runs_it(dialect):
    # The rule's fall-back reads the statement as printed for the dialect it was written in, which prints here, and
    # the decider's own printing for DuckDB then refuses it: the path this fix leaves as it was.
    verdict = decide("SELECT DATE() AS d FROM games", VISIBLE, dialect=dialect, target="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNPRINTABLE_FOR_TARGET, verdict
