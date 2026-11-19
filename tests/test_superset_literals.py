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

from mnemiq.agent.loop import PROBE_BUDGET_S
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.contract.semantic import SourceBinding
from mnemiq.sql.values_check import PROBE_LIMIT
from mnemiq.superset.literals import Absent, Probes, absent_literals, filter_statement, holds_null, null_statement

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
    every statement is recorded in the order sent, with whether it asked to be run afresh (`execute_arrow_afresh`)
    rather than answered from Superset's result cache (`execute_arrow`). `cached(sql)`, when given, is what the cache
    answers a statement not run afresh with."""

    dialect = "postgres"

    def __init__(self, answer, *, clock: list[float] | None = None, takes_s: float = 0.0, cached=None) -> None:
        self.answer, self.cached = answer, cached
        self.sent: list[str] = []
        self.fresh: list[bool] = []
        self.timeouts: list[float | None] = []
        # A read takes `takes_s` on `clock` (a one-item list, the time now), the clock a `Probes` is given.
        self.clock, self.takes_s = clock, takes_s

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        return self._run(sql, timeout_s, fresh=False)

    def execute_arrow_afresh(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        return self._run(sql, timeout_s, fresh=True)

    def _run(self, sql: str, timeout_s: float | None, *, fresh: bool) -> pa.Table:
        self.sent.append(sql)
        self.fresh.append(fresh)
        self.timeouts.append(timeout_s)
        if self.clock is not None:
            self.clock[0] += self.takes_s
        result = self.answer(sql) if fresh or self.cached is None else self.cached(sql)
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


# ---------------------------------------------------------------- read afresh, never from Superset's result cache


def test_every_probe_and_read_of_held_values_is_run_afresh():
    """A cached answer may be older than the data: an empty one would refuse a value the column now holds, and cached
    values would tell the repair what the column used to hold. So every statement asks Superset to run it afresh."""
    source = _source(["LOST", "WON"], present=frozenset({"WON"}))
    assert _absent("SELECT COUNT(*) FROM t WHERE status IN ('Won', 'WON')", source) == [
        Absent("t.status", "Won", ("LOST", "WON"), False)]
    assert len(source.sent) == 3 and source.fresh == [True, True, True]


def test_a_cached_empty_answer_refuses_nothing_once_the_value_is_held():
    """The cache still says no row holds 'Won' (it was added since); run afresh, the source finds it."""
    source = _source(["WON", "Won"], present=frozenset({"Won"}), cached=lambda sql: NONE)
    assert _absent("SELECT COUNT(*) FROM t WHERE status = 'Won'", source) == []
    assert source.fresh == [True]


# ---------------------------------------------------------------- whether a column holds NULL


def _nulls(found) -> Source:
    """A source whose every statement finds a row (`found` True), none (False), or raises `found`."""
    return Source(lambda sql: found if isinstance(found, Exception) else ONE if found else NONE)


def _holds_null(column: str, source: Source, *, table: str = "t", snapshot=SNAPSHOT, grants=GRANTS,
                probes: Probes | None = None) -> bool | None:
    return holds_null(table, column, adapter=source, snapshot=snapshot, grants=grants, timeout_s=5.0,
                      probes=Probes() if probes is None else probes)


def test_null_statement_is_one_quoted_is_null_probe():
    assert null_statement("t", "Status", "postgres") == 'SELECT 1 FROM "t" WHERE "Status" IS NULL LIMIT 1'


def test_holds_null_reads_afresh_and_answers_from_the_rows():
    """One row back is a NULL the builder can read; none back is none. Each is one statement, run afresh, naming the
    table as the source stores it (`Snapshot.source_bindings`), as a probe of a value does."""
    bound = SNAPSHOT.model_copy(update={"source_bindings": [
        SourceBinding(id="b", source_id="only", object_id="t", source_object="t_stored", binding_type="table")]})
    found, none = _nulls(True), _nulls(False)
    assert _holds_null("status", found, snapshot=bound) is True
    assert _holds_null("status", none, snapshot=bound) is False
    for source in (found, none):
        assert source.sent == ['SELECT 1 FROM "t_stored" WHERE "status" IS NULL LIMIT 1']
        assert source.fresh == [True] and source.timeouts == [5.0]


def test_holds_null_asks_once_per_column_per_plan():
    """One plan's checks share what the source said: a column asked twice is asked once. Another column, and another
    plan's probes, are asked afresh."""
    probes = Probes()
    source = _nulls(True)
    assert _holds_null("status", source, probes=probes) is True
    assert _holds_null("status", source, probes=probes) is True
    assert len(source.sent) == 1 and probes.nulls == {("t", "status"): True}
    assert _holds_null("amount", source, probes=probes) is True
    assert _holds_null("status", source) is True
    assert len(source.sent) == 3


@pytest.mark.parametrize(("table", "column", "snapshot", "grants", "refused"), [
    ("t", "owner", POLICED, GrantSet(frozenset({"t"}), pii_mask=frozenset({"pii"})), False),
    ("t", "owner", POLICED, GrantSet(frozenset({"t"})), False),
    ("u", "secret", SNAPSHOT, GRANTS, False),
    ("t", "status", SNAPSHOT, GrantSet(frozenset({"t"}), row_filters={"t": "owner = 'me'"}), False),
    ("t", "status", SNAPSHOT, GrantSet(frozenset({"t"}), row_filters={"public.t": "owner = 'me'"}), False),
    ("t", "status", SNAPSHOT, GRANTS, True),
], ids=["masked", "denied", "ungranted table", "row-filtered table", "unknown row-filter target",
        "terminal refusal"])
def test_holds_null_asks_nothing_for_a_withheld_column_or_a_row_filtered_table_or_after_a_terminal_refusal(
        table, column, snapshot, grants, refused):
    """Whether a column holds NULL is what a mask withholds, a denied column and a table the grants hide are not the
    builder's to ask about, and a probe runs outside mnemiq's own row filter, so a table under one is never asked;
    nor is one while a row filter names a table the snapshot does not hold, which may be this one misspelled. Once
    Superset refused the request for good (`terminal_refusal`), nothing more is sent. Each is None, not asked, and not
    remembered."""
    source = _nulls(True)
    if refused:
        source.terminal_refusal = RuntimeError("Superset refused this caller")
    probes = Probes()
    assert _holds_null(column, source, table=table, snapshot=snapshot, grants=grants, probes=probes) is None
    assert source.sent == [] and probes.nulls == {}
    if column == "owner":  # a column beside it the builder reads raw still is asked
        assert _holds_null("status", source, snapshot=snapshot, grants=grants) is True


def test_holds_null_is_none_when_the_budget_is_spent_and_does_not_remember_it(caplog):
    """The probe shares the plan's budget: given 3 s of 10 left, it is sent with a 3 s limit, and the 4 s it takes are
    spent; past the budget the next column is not asked, not remembered, and logged by its name alone."""
    clock = [0.0]
    probes = Probes(budget_s=10.0, clock=lambda: clock[0], spent_s=7.0)
    source = Source(lambda sql: ONE, clock=clock, takes_s=4.0)
    assert _holds_null("status", source, probes=probes) is True
    assert source.timeouts == [3.0] and probes.spent_s == 11.0
    with caplog.at_level(logging.INFO):
        assert _holds_null("amount", source, probes=probes) is None
    assert len(source.sent) == 1 and probes.nulls == {("t", "status"): True}
    assert "t.amount" in caplog.text


def test_a_failed_null_probe_is_none_and_logs_no_value(caplog):
    """A probe Superset does not answer says nothing, and is remembered so: the column is not asked again in the plan.
    Its log line names the column and the failure's kind; a source's words can quote a value, so never them."""
    probes = Probes()
    source = _nulls(RuntimeError("invalid input syntax: 'Ann Smith'"))
    with caplog.at_level(logging.INFO, logger="mnemiq.superset.literals"):
        assert _holds_null("status", source, probes=probes) is None
    assert probes.nulls == {("t", "status"): None}
    assert _holds_null("status", source, probes=probes) is None and len(source.sent) == 1
    [line] = [r.getMessage() for r in caplog.records if r.name == "mnemiq.superset.literals"]
    assert "t.status" in line and "RuntimeError" in line and "Ann" not in line and "invalid" not in line
