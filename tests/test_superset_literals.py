"""A filter's value no row the builder can read holds, found by the ask path's own grounding (`ground_values`,
`confirm_by_source`, `source_holds`) and named with the values its column does hold.

The adapter here is a fake that answers each statement it is sent and records it: every probe and every read of a
column's values is one of its statements, so what the source was asked is exactly what it recorded.
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest
import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.contract.semantic import SourceBinding
from mnemiq.sql.values_check import PROBE_LIMIT
from mnemiq.agent.loop import PROBE_BUDGET_S
from mnemiq.superset.literals import Absent, Probes, absent_literals, filter_statement

SNAPSHOT = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id="t.status", object_id="t", name="status", data_type="VARCHAR"),
    Column(id="t.amount", object_id="t", name="amount", data_type="NUMERIC"),
    Column(id="t.owner", object_id="t", name="owner", data_type="VARCHAR"),
    Column(id="u.secret", object_id="u", name="secret", data_type="VARCHAR"),
])
GRANTS = GrantSet(frozenset({"t"}))
NONE = pa.table({"?column?": pa.array([], pa.int32())})
ONE = pa.table({"?column?": [1]})


class Source:
    """The request's adapter, faked: `answer(sql)` answers each statement (a table, or an exception it raises), and
    every statement is recorded in the order sent."""

    dialect = "postgres"

    def __init__(self, answer, *, clock: list[float] | None = None, takes_s: float = 0.0) -> None:
        self.answer = answer
        self.sent: list[str] = []
        self.timeouts: list[float | None] = []
        # A read takes `takes_s` on `clock` (a one-item list, the time now), the clock a `Probes` is given.
        self.clock, self.takes_s = clock, takes_s

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        self.sent.append(sql)
        self.timeouts.append(timeout_s)
        if self.clock is not None:
            self.clock[0] += self.takes_s
        result = self.answer(sql)
        if isinstance(result, Exception):
            raise result
        return result


def _probe(sql: str) -> bool:
    return sql.startswith("SELECT 1 FROM")


def _held(sql: str) -> bool:
    return sql.startswith("SELECT DISTINCT")


def _source(held: list[str], *, present: frozenset[str] = frozenset(), **timing) -> Source:
    """A source whose probes find only `present`, and whose column holds `held`."""
    def answer(sql: str):
        if _probe(sql):
            [literal] = [node for node in sqlglot.parse_one(sql, read="postgres").find_all(exp.Literal)
                         if node.is_string]
            return ONE if literal.this in present else NONE
        if _held(sql):
            return pa.table({"status": held})
        raise AssertionError(f"an unexpected statement: {sql}")
    return Source(answer, **timing)


def _absent(sql: str, source: Source, *, snapshot=SNAPSHOT, grants=GRANTS, probes=None) -> list[Absent]:
    return absent_literals(sql, adapter=source, snapshot=snapshot, grants=grants, timeout_s=5.0, probes=probes)


def test_an_equality_literal_no_row_holds_is_absent_and_names_the_held_values():
    source = _source(["LOST", "WON"])
    found = _absent("SELECT SUM(CASE WHEN status = 'Won' THEN amount END) AS won FROM \"t\"", source)
    assert found == [Absent("t.status", "Won", ("LOST", "WON"), False)]
    probes, held = [s for s in source.sent if _probe(s)], [s for s in source.sent if _held(s)]
    assert probes == ['SELECT 1 FROM "t" WHERE "status" = \'Won\' LIMIT 1']
    assert held == ['SELECT DISTINCT "status" FROM "t" WHERE "status" IS NOT NULL ORDER BY 1 LIMIT 21']


def test_a_literal_a_row_holds_is_not_absent_and_no_values_are_read():
    source = _source(["LOST", "WON"], present=frozenset({"WON"}))
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'WON'", source) == []
    assert not any(_held(s) for s in source.sent)


def test_not_in_literals_are_each_absent():
    source = _source(["LOST", "WON"])
    found = _absent("SELECT COUNT(*) FROM t WHERE status NOT IN ('Won', 'Lost')", source)
    assert found == [Absent("t.status", "Won", ("LOST", "WON"), False),
                     Absent("t.status", "Lost", ("LOST", "WON"), False)]
    assert sum(_held(s) for s in source.sent) == 1, "one column's values are read once"


def test_a_literal_on_an_expression_is_not_probed():
    source = _source(["LOST", "WON"])
    assert _absent("SELECT COUNT(*) FROM t WHERE LOWER(status) = 'won'", source) == []
    assert source.sent == []


def test_a_probe_the_source_does_not_answer_refuses_nothing():
    source = Source(lambda sql: RuntimeError("the source went away"))
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source) == []
    assert len(source.sent) == 1 and _probe(source.sent[0])


def test_at_most_20_held_values_are_named():
    many = [f"S{i:02d}" for i in range(21)]
    source = _source(many)
    [found] = _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source)
    assert found.held == tuple(many[:20]) and found.more is True
    [read] = [s for s in source.sent if _held(s)]
    assert read.endswith("LIMIT 21")
    [exactly] = _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", _source(many[:20]))
    assert exactly.held == tuple(many[:20]) and exactly.more is False


def test_a_long_held_value_is_cut_as_a_prompt_shows_one():
    document = "x" * 5000
    [found] = _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", _source([document, "WON"]))
    assert document not in found.held and found.held[0].startswith("x" * 200) and "never compare with it" in \
        found.held[0]
    assert found.held[1] == "WON"


def test_a_literal_with_a_quote_is_probed_safely():
    source = _source(["LOST"])
    [found] = _absent("SELECT COUNT(*) FROM t WHERE status = 'O''Brien'", source)
    assert found.literal == "O'Brien"
    [probe] = [s for s in source.sent if _probe(s)]
    parsed = sqlglot.parse_one(probe, read="postgres")
    assert [literal.this for literal in parsed.find_all(exp.Literal) if literal.is_string] == ["O'Brien"]


def test_every_read_goes_through_the_requests_adapter():
    """The probe and the read of the held values are both statements of the adapter it was given, and name the table
    as the source stores it (`Snapshot.source_bindings`), as the ask path's probe does."""
    bound = SNAPSHOT.model_copy(update={"source_bindings": [
        SourceBinding(id="b", source_id="only", object_id="t", source_object="t_stored", binding_type="table")]})
    source = _source(["LOST", "WON"])
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source, snapshot=bound) == [
        Absent("t.status", "Won", ("LOST", "WON"), False)]
    assert source.sent == ['SELECT 1 FROM "t_stored" WHERE "status" = \'Won\' LIMIT 1',
                           'SELECT DISTINCT "status" FROM "t_stored" WHERE "status" IS NOT NULL ORDER BY 1 LIMIT 21']


def test_a_probe_the_deciders_refuse_refuses_nothing_and_names_no_value():
    """A table the builder's grants hide is no table at all to the resolver: nothing is placed, so nothing is probed,
    no value is read, and nothing is refused."""
    source = _source(["anything"])
    assert _absent("SELECT COUNT(*) FROM u WHERE secret = 'x'", source) == []
    assert source.sent == []


def test_a_failed_held_values_read_still_refuses_and_logs_no_value(caplog):
    def answer(sql):
        if _probe(sql):
            return NONE
        return RuntimeError("SELECT DISTINCT status ... 'WON' 'LOST'")

    with caplog.at_level(logging.INFO):
        found = _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", Source(answer))
    assert found == [Absent("t.status", "Won", None, False)], "None: the source did not say"
    assert "RuntimeError" in caplog.text
    for value in ("Won", "WON", "LOST"):
        assert value not in caplog.text


def test_values_past_the_probe_limit_stay_unchecked_and_only_their_count_is_logged(caplog):
    literals = [f"v{i}" for i in range(PROBE_LIMIT + 2)]
    source = _source(["LOST"])
    with caplog.at_level(logging.INFO):
        found = _absent(f"SELECT COUNT(*) FROM t WHERE status IN ({', '.join(repr(v) for v in literals)})", source)
    assert [a.literal for a in found] == literals[:PROBE_LIMIT]
    assert sum(_probe(s) for s in source.sent) == PROBE_LIMIT
    assert "2 filter values" in caplog.text
    for value in literals:
        assert f"'{value}'" not in caplog.text and f" {value} " not in caplog.text


def test_a_statement_that_does_not_parse_refuses_nothing():
    source = _source(["LOST"])
    assert _absent("SELECT COUNT(* FROM t WHERE status = 'Won'", source) == []
    assert source.sent == []


# ---------------------------------------------------------------- a chart's filter, as a statement


@pytest.mark.parametrize(("op", "operand", "expected"), [
    ("=", "Won", "SELECT 1 FROM \"t\" WHERE \"status\" IN ('Won')"),
    ("!=", "Won", "SELECT 1 FROM \"t\" WHERE \"status\" IN ('Won')"),
    ("IN", ["Won", "Lost"], "SELECT 1 FROM \"t\" WHERE \"status\" IN ('Won', 'Lost')"),
    ("NOT IN", ["Won", 3, "Lost"], "SELECT 1 FROM \"t\" WHERE \"status\" IN ('Won', 'Lost')"),
    ("=", "O'Brien", "SELECT 1 FROM \"t\" WHERE \"status\" IN ('O''Brien')"),
])
def test_a_chart_filter_on_a_string_is_one_statement(op, operand, expected):
    assert filter_statement("t", "status", op, operand, "postgres") == expected


@pytest.mark.parametrize(("op", "operand"), [("=", 3), ("IN", [1, 2.5]), ("LIKE", "W%"), ("ILIKE", "w%"),
                                             ("NOT LIKE", "W%"), (">", "a"), ("IS NULL", None), ("IN", [])])
def test_a_chart_filter_with_no_string_to_check_is_no_statement(op, operand):
    assert filter_statement("t", "status", op, operand, "postgres") is None


def test_a_chart_filters_statement_goes_through_the_grounding():
    source = _source(["LOST", "WON"], present=frozenset({"WON"}))
    found = _absent(filter_statement("t", "status", "NOT IN", ["WON", "Lost"], "postgres"), source)
    assert found == [Absent("t.status", "Lost", ("LOST", "WON"), False)]


# ---------------------------------------------------------------- what the source said about the column


def test_a_column_that_holds_no_value_the_builder_can_read_is_said_to_hold_none():
    """The read ran and returned no row: the source said the column holds nothing the builder can read (`()`), which
    is not the same as a read that failed (`None`)."""
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", _source([])) == [
        Absent("t.status", "Won", (), False)]


# ---------------------------------------------------------------- what is never asked


POLICED = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id="t.status", object_id="t", name="status", data_type="VARCHAR"),
    Column(id="t.owner", object_id="t", name="owner", data_type="VARCHAR", pii_level="pii"),
])


@pytest.mark.parametrize("grants", [GrantSet(frozenset({"t"}), pii_mask=frozenset({"pii"})),
                                    GrantSet(frozenset({"t"}))], ids=["masked", "denied"])
def test_a_value_on_a_masked_or_denied_column_is_never_asked(grants):
    """Whether a value exists is what a mask withholds, and a denied column is not the builder's to ask about: the
    grounding is given the policy's masks and denials, as the decider gives it its masks, so neither is probed. A
    column beside it the builder reads raw still is."""
    source = _source([])
    assert _absent("SELECT COUNT(*) FROM t WHERE owner = 'Ann'", source, snapshot=POLICED, grants=grants) == []
    assert source.sent == []
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source, snapshot=POLICED, grants=grants) != []


def test_a_value_on_a_row_filtered_table_is_never_asked():
    """A probe runs as written, outside mnemiq's own row filter, so a table the grants filter by row is never asked
    about, as on the ask path (`row_filtered`)."""
    source = _source([])
    grants = GrantSet(frozenset({"t"}), row_filters={"t": "owner = 'me'"})
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source, grants=grants) == []
    assert source.sent == []


def test_a_request_superset_refused_for_good_asks_nothing_more():
    """Once the adapter holds a refusal every later call meets (`terminal_refusal`), nothing more is sent."""
    source = _source([])
    source.terminal_refusal = RuntimeError("Superset refused this caller")
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source) == []
    assert source.sent == []


# ---------------------------------------------------------------- one plan's probes: one budget, one memo


def test_the_budget_is_the_ask_paths():
    assert Probes().budget_s == PROBE_BUDGET_S


def test_after_the_budget_nothing_more_is_read_and_nothing_is_refused(caplog):
    """Every read takes 4 s on the plan's clock and the budget is 10 s. The first value's probe and its column's read
    take 8 s; the second's probe is given the 2 s left and takes 4; its column is then not read (the source did not
    say); the third value is never asked, refuses nothing, and is logged as a count."""
    clock = [0.0]
    probes = Probes(budget_s=10.0, clock=lambda: clock[0])
    status = _source(["LOST", "WON"], clock=clock, takes_s=4.0)
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", status, probes=probes) == [
        Absent("t.status", "Won", ("LOST", "WON"), False)]
    owner = _source(["ann"], clock=clock, takes_s=4.0)
    assert _absent("SELECT COUNT(*) FROM t WHERE owner = 'Bob'", owner, probes=probes) == [
        Absent("t.owner", "Bob", None, False)]
    assert status.timeouts == [5.0, 5.0] and owner.timeouts == [2.0]
    assert [_probe(s) for s in owner.sent] == [True], "the column's values were not read past the budget"
    late = _source(["x"], clock=clock, takes_s=4.0)
    with caplog.at_level(logging.INFO):
        assert _absent("SELECT COUNT(*) FROM t WHERE status IN ('Lost', 'Open')", late, probes=probes) == []
    assert late.sent == []
    assert "2 filter values" in caplog.text and "Lost" not in caplog.text and "Open" not in caplog.text


def test_the_same_value_is_probed_once_and_a_columns_values_read_once():
    """One plan's statements share what the source said: the same value in two statements is asked once, and the
    values of one column are read once, whichever value sent the read."""
    probes = Probes()
    source = _source(["LOST", "WON"])
    first = _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source, probes=probes)
    again = _absent("SELECT SUM(amount) FROM t WHERE status = 'Won'", source, probes=probes)
    other = _absent("SELECT 1 FROM \"t\" WHERE \"status\" IN ('Lost')", source, probes=probes)
    assert first == again == [Absent("t.status", "Won", ("LOST", "WON"), False)]
    assert other == [Absent("t.status", "Lost", ("LOST", "WON"), False)]
    assert sum(_probe(s) for s in source.sent) == 2 and sum(_held(s) for s in source.sent) == 1
