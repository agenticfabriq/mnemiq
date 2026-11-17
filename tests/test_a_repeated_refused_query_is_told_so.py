"""A statement the decider refused, sent again unchanged and refused the same way, is fed back saying so, under
`repair_hints`.

Measured on a local model: refused as naming a column that does not exist, it sent the identical statement three times,
each told the same sentence, and the turn ended declined. Told the second time that it repeated itself, a writer has one
more thing to act on. Only the words change: the statement is still decided every time (a refusal resting on a source
call can change between attempts), and the deferral after the last attempt keeps today's reason.
"""

from __future__ import annotations

import json

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, DeferralReason, Snapshot
from mnemiq.generate.generator import FakeGenerator, SqlProposal
from mnemiq.generate.plan_query import Deferred, plan_query, same_query
from mnemiq.redact import scrub_credentials
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.verdict import Approved

_GRANTS = GrantSet(frozenset({"orders"}))
NOPE = "No table in this query has a column 'nope'. Use only the columns listed on the schema cards."
REPEAT = "You sent this same query before, and it is refused again for the same reason: "


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="src", created_at="2026-11-17T00:00:00Z",
                    columns=[Column(id="orders.id", object_id="orders", name="id"),
                             Column(id="orders.status", object_id="orders", name="status")])


def _packet() -> ContextPacket:
    return ContextPacket(question="which orders?", cards=[RetrievedCard(object_id="orders", card="TABLE orders",
                                                                        score=1.0)],
                         grant_fingerprint="fp", enrichment_version="v1")


def _plan(sqls, **kw):
    generator = FakeGenerator([json.dumps({"sql": s}) for s in sqls])
    return plan_query(_packet(), _snapshot(), _GRANTS, generator, dialect="duckdb", target="duckdb",
                      max_attempts=len(sqls), **kw), generator


def test_the_second_identical_query_is_told_it_repeats():
    out, gen = _plan(["SELECT nope FROM orders"] * 3, repair_hints=True)
    assert gen.calls[0] is None and gen.calls[1] == NOPE
    assert gen.calls[2] == REPEAT + NOPE + " Sent unchanged it cannot pass; change what that reason names."
    assert isinstance(out, Deferred) and out.code == DeferralReason.INVALID_QUERY
    assert out.reason == "Could not produce a valid query after 3 attempts. " + NOPE


def test_a_repeat_that_differs_only_in_spacing_case_or_a_semicolon_is_still_one():
    out, gen = _plan(["SELECT nope FROM orders", "select  nope\nfrom orders;", "SELECT id FROM orders"],
                     repair_hints=True)
    assert gen.calls[2].startswith(REPEAT)
    assert isinstance(out, Approved)


def test_whitespace_case_and_a_semicolon_are_the_same_query():
    assert same_query("select nope from orders;", "SELECT  nope\nFROM orders", "duckdb")


def test_a_changed_literal_is_another_query():
    assert not same_query("SELECT a FROM t WHERE b = 'x'", "SELECT a FROM t WHERE b = 'y'", "duckdb")


def test_text_that_does_not_parse_is_compared_as_text():
    assert same_query("SELEC nope FROM", "SELEC  nope\nFROM ;", "duckdb")
    assert not same_query("SELEC nope FROM", "SELEC other FROM", "duckdb")


def test_a_changed_query_gets_the_plain_feedback():
    out, gen = _plan(["SELECT nope FROM orders", "SELECT nada FROM orders", "SELECT id FROM orders"],
                     repair_hints=True)
    assert gen.calls[2] == "No table in this query has a column 'nada'. Use only the columns listed on the schema cards."
    assert isinstance(out, Approved)


def test_off_feeds_back_as_today():
    out, gen = _plan(["SELECT nope FROM orders"] * 3)
    assert gen.calls == [None, NOPE, NOPE]
    assert out.reason == "Could not produce a valid query after 3 attempts. " + NOPE


def test_a_repeat_keeps_the_source_words_marked_as_the_sources():
    """An EXPLAIN refusal carries the source's words. Repeated, they are fed back again inside the repeat sentence, and
    a model that quotes that feedback in its reason must still not carry them out (M94)."""
    leaky = "permission denied for table hr_prod.payroll_salary; postgresql://svc:pw@10.2.0.7/x"

    class _RefusingExplain:
        dialect = "duckdb"

        def execute(self, sql):
            raise RuntimeError(leaky)

    class _RepeatsThenQuotes:
        def __init__(self):
            self.calls = []

        def propose(self, packet, feedback=None, strategy=None):
            self.calls.append(feedback)
            if len(self.calls) < 3:
                return SqlProposal(sql="SELECT id FROM orders")
            return SqlProposal(sql=None, reason=f"the database said: {feedback}")

    generator = _RepeatsThenQuotes()
    out = plan_query(_packet(), _snapshot(), _GRANTS, generator, adapter=_RefusingExplain(), dialect="duckdb",
                     target="duckdb", max_attempts=3, repair_hints=True)
    assert generator.calls[2].startswith(REPEAT) and scrub_credentials(leaky) in generator.calls[2]
    assert isinstance(out, Deferred)
    for secret in ("payroll_salary", "hr_prod", "10.2.0.7", "postgresql://"):
        assert secret not in out.reason


def test_a_changed_literal_or_quoted_name_in_unparseable_text_is_another_query():
    """Codex's review: collapsing spacing over the raw text merged a literal's own spacing. Compared as tokens, only the
    spacing between them is set aside."""
    assert not same_query("SELECT 'a  b' FROM", "SELECT 'a b' FROM", "duckdb")
    assert not same_query('SELECT "a  b" FROM', 'SELECT "a b" FROM', "duckdb")
    assert same_query("SELECT 'a  b' FROM", "SELECT   'a  b'\nFROM ;", "duckdb")


def test_text_the_tokenizer_cannot_read_is_compared_exactly():
    assert same_query("SELECT 'abc FROM", "SELECT 'abc FROM", "duckdb")
    assert not same_query("SELECT 'abc FROM", "SELECT  'abc FROM", "duckdb")


def test_an_attempt_the_corrector_rewrote_is_never_called_a_repeat():
    """Codex's review: after the corrector runs, the refusal is about the corrected statement, not the writer's. Two
    identical proposals corrected two different ways and refused alike are not "sent unchanged", and the corrector
    path's feedback stays as it was."""
    from mnemiq.generate.correct import FakeCorrector

    class _Index:
        def has(self, table, column):
            return (table, column) == ("orders", "status")

        def contains(self, table, column, literal):
            return literal in {"open", "closed"}

        def nearest(self, table, column, literal, k=8):
            return ["closed", "open"]

    sql = "SELECT id FROM orders WHERE status = 'west'"
    corrector = FakeCorrector([sql + " AND id > 0", sql + " AND id > 1", sql + " AND id > 2"])
    out, gen = _plan([sql] * 3, repair_hints=True, corrector=corrector, values=_Index())
    assert len(corrector.calls) == 3 and isinstance(out, Deferred)
    assert not any((c or "").startswith(REPEAT) for c in gen.calls), gen.calls
