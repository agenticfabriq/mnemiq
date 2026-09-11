import logging

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, DeferralReason, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.verdict import Approved, RefusalCode


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[
            Column(id="claim.claim_identifier", object_id="claim", name="claim_identifier"),
            Column(id="claim.status", object_id="claim", name="status"),
            Column(id="person.last_name", object_id="person", name="last_name"),
        ],
    )


def _packet() -> ContextPacket:
    return ContextPacket(
        question="how many claims?",
        cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint="fp",
        enrichment_version="v1",
    )


_GRANTS = GrantSet(frozenset({"claim"}))


def _plan(replies, **kw):
    generator = FakeGenerator(replies)
    outcome = plan_query(_packet(), _snapshot(), _GRANTS, generator, target="duckdb", **kw)
    return outcome, generator


def test_a_good_query_is_approved_on_the_first_try():
    outcome, generator = _plan(['{"sql": "SELECT claim_identifier FROM claim"}'])
    assert isinstance(outcome, Approved)
    assert outcome.tables == ["claim"]
    assert generator.calls == [None]  # no feedback needed


def test_a_repairable_refusal_is_fed_back_and_repaired():
    outcome, generator = _plan(
        [
            '{"sql": "SELECT * FROM claim"}',  # rejected: star over a base table
            '{"sql": "SELECT claim_identifier FROM claim"}',  # repaired
        ]
    )
    assert isinstance(outcome, Approved)
    assert generator.calls[0] is None
    assert "SELECT *" in generator.calls[1]  # the refusal message came back as feedback


def test_an_unauthorized_table_is_never_retried():
    outcome, generator = _plan(
        [
            '{"sql": "SELECT last_name FROM person"}',
            '{"sql": "SELECT claim_identifier FROM claim"}',  # must never be reached
        ]
    )
    assert isinstance(outcome, Deferred)
    assert "person" in outcome.reason
    assert len(generator.calls) == 1  # a guard is not a puzzle to solve with another attempt


def test_the_loop_is_bounded():
    outcome, generator = _plan(['{"sql": "SELECT * FROM claim"}'] * 5, max_attempts=3)
    assert isinstance(outcome, Deferred)
    assert len(generator.calls) == 3
    # ...and a model that never writes valid SQL is exactly what INVALID_QUERY is for. The
    # exhaustion exit hands a SOURCE fault to `_not_our_sql` instead, and dropping the guard
    # that tells the two apart would report this one as ungovernable -- blaming the deployment
    # for three attempts at `SELECT *`.
    assert outcome.code is DeferralReason.INVALID_QUERY
    assert "3 attempts" in outcome.reason


def test_a_model_deferral_is_passed_through():
    outcome, _ = _plan(['{"sql": null, "reason": "no table records payouts"}'])
    assert isinstance(outcome, Deferred)
    assert "payouts" in outcome.reason


def test_an_empty_packet_defers_without_calling_the_model():
    empty = ContextPacket(
        question="anything?", cards=[], grant_fingerprint="fp", enrichment_version="v1"
    )
    generator = FakeGenerator(['{"sql": "SELECT 1"}'])
    outcome = plan_query(empty, _snapshot(), GrantSet(frozenset()), generator, target="duckdb")

    assert isinstance(outcome, Deferred)
    assert generator.calls == []  # no tables, no grants, no reason to spend a token


def test_logic_lint_is_corrected_once_and_approved():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column, Snapshot
    from mnemiq.generate.correct import FakeCorrector
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
    from mnemiq.sql.verdict import Approved

    snapshot = Snapshot(
        version="v1", source_id="s", created_at="2026-07-15T00:00:00Z",
        columns=[
            Column(id="claim.name", object_id="claim", name="name"),
            Column(id="claim.score", object_id="claim", name="score"),
        ],
    )
    packet = ContextPacket(
        question="q", cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1",
    )
    grants = GrantSet(frozenset({"claim"}))
    gen = FakeGenerator(['{"sql": "SELECT name FROM claim ORDER BY score LIMIT 1"}'])
    corrector = FakeCorrector(
        ["SELECT name FROM claim WHERE score IS NOT NULL ORDER BY score LIMIT 1"]
    )

    verdict = plan_query(
        packet, snapshot, grants, gen, adapter=None, target="duckdb", corrector=corrector
    )
    assert isinstance(verdict, Approved)
    assert corrector.calls and "IS NOT NULL" not in corrector.calls[0][0]  # got the flawed SQL
    # the corrected (null-guarded) form was approved; sqlglot renders it as "NOT score IS NULL"
    assert "IS NULL" in verdict.plan_sql and "NOT" in verdict.plan_sql


def test_value_grounding_is_corrected_once_and_approved():
    from mnemiq.contract import Column, Snapshot
    from mnemiq.generate.correct import FakeCorrector
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
    from mnemiq.sql.verdict import Approved

    class _FakeIndex:
        def __init__(self, data):
            self._data = data

        def has(self, t, c):
            return (t, c) in self._data

        def contains(self, t, c, v):
            return v in self._data.get((t, c), set())

        def nearest(self, t, c, v, k=8):
            return sorted(self._data.get((t, c), set()))[:k]

    snapshot = Snapshot(
        version="v1", source_id="s", created_at="2026-07-15T00:00:00Z",
        columns=[Column(id="gasstations.Country", object_id="gasstations", name="Country")],
    )
    packet = ContextPacket(
        question="q",
        cards=[RetrievedCard(object_id="gasstations", card="TABLE gasstations", score=1.0)],
        grant_fingerprint="fp", enrichment_version="v1",
    )
    grants = GrantSet(frozenset({"gasstations"}))
    idx = _FakeIndex({("gasstations", "Country"): {"Czech Republic", "Slovakia"}})
    gen = FakeGenerator(['{"sql": "SELECT Country FROM gasstations WHERE Country = \'CZE\'"}'])
    corrector = FakeCorrector(["SELECT Country FROM gasstations WHERE Country = 'Czech Republic'"])

    verdict = plan_query(
        packet, snapshot, grants, gen, adapter=None, target="duckdb",
        corrector=corrector, values=idx,
    )
    assert isinstance(verdict, Approved)
    assert "CZE" in corrector.calls[0][0]  # the corrector saw the flawed SQL
    assert "Czech Republic" in verdict.plan_sql  # the corrected literal was approved


def test_without_a_corrector_a_lint_falls_back_to_regenerate_then_defers():
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column, Snapshot
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import Deferred, plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    snapshot = Snapshot(
        version="v1", source_id="s", created_at="2026-07-15T00:00:00Z",
        columns=[Column(id="claim.score", object_id="claim", name="score"),
                 Column(id="claim.name", object_id="claim", name="name")],
    )
    packet = ContextPacket(question="q", cards=[RetrievedCard(object_id="claim", card="c", score=1.0)],
                           grant_fingerprint="fp", enrichment_version="v1")
    grants = GrantSet(frozenset({"claim"}))
    gen = FakeGenerator([
        '{"sql": "SELECT name FROM claim ORDER BY score LIMIT 1"}',
        '{"sql": "SELECT name FROM claim ORDER BY score LIMIT 1"}',
    ])
    out = plan_query(packet, snapshot, grants, gen, adapter=None, target="duckdb", max_attempts=2)
    assert isinstance(out, Deferred)  # lint fed back, never corrected, ran out of attempts



# --- M33: the corrector is the one mode difference nobody could observe ---------


def test_a_plan_that_needed_no_repair_says_so():
    """`thinking` differs from `instant` by the corrector and almost nothing else, so an
    answer that cannot say whether it fired cannot show the mode doing its work."""
    outcome, _ = _plan(['{"sql": "SELECT claim_identifier FROM claim"}'])
    assert isinstance(outcome, Approved) and outcome.corrected is False


def test_a_plan_the_corrector_carried_records_it():
    """An ORDER BY with a LIMIT and no NOT NULL guard is a lint refusal -- silently-wrong SQL
    that runs fine and answers wrong, which is exactly what the corrector exists for."""

    class _Corrector:
        def correct(self, sql, message):
            return "SELECT status FROM claim WHERE NOT status IS NULL ORDER BY status"

    outcome, _ = _plan(
        ['{"sql": "SELECT status FROM claim ORDER BY status"}'], corrector=_Corrector()
    )
    assert isinstance(outcome, Approved)
    assert outcome.corrected is True


def test_a_correction_that_still_refuses_is_not_reported_as_corrected():
    """Reporting it would overstate the work: a failed repair is not a repaired plan."""

    class _Useless:
        def correct(self, sql, message):
            return "SELECT status FROM claim ORDER BY status"  # the same lint violation

    outcome, _ = _plan(
        ['{"sql": "SELECT status FROM claim ORDER BY status"}'] * 3, corrector=_Useless()
    )
    assert isinstance(outcome, Deferred)


def test_decide_never_sets_corrected_because_the_decider_does_not_repair():
    from mnemiq.sql.decide import decide

    verdict = decide("SELECT claim_identifier FROM claim", {"claim": {"claim_identifier"}},
                     dialect="duckdb", target="duckdb")
    assert isinstance(verdict, Approved) and verdict.corrected is False


def test_an_explain_refusal_also_marks_the_feedback_as_the_sources(caplog):
    """The other way source words enter a prompt: `prove` refuses, and its detail is fed back.

    `plan_query`'s own repair loop is the shorter path -- no execution needed, so an EXPLAIN
    refusal reaches the model on attempt two of a single ask. The suppression has to key off
    the refusal that carried the words, not only off the agent's outer loop (M94).
    """
    leaky = 'permission denied for table hr_prod.payroll_salary; postgresql://svc:pw@10.2.0.7/x'

    class _RefusingExplain:
        dialect = "duckdb"

        def execute(self, sql):
            raise RuntimeError(leaky)

    class _Quoting:
        def __init__(self):
            self.calls = []

        def propose(self, packet, feedback=None, strategy=None):
            from mnemiq.generate.generator import SqlProposal
            self.calls.append(feedback)
            if feedback is None:
                return SqlProposal(sql="SELECT claim_identifier FROM claim")
            return SqlProposal(sql=None, reason=f"the database said: {feedback}")

    generator = _Quoting()
    with caplog.at_level(logging.WARNING, logger="mnemiq.sql.prove"):
        outcome = plan_query(_packet(), _snapshot(), _GRANTS, generator,
                             adapter=_RefusingExplain(), target="duckdb", max_attempts=2)

    assert isinstance(outcome, Deferred)
    for secret in ("payroll_salary", "hr_prod", "10.2.0.7", "postgresql://"):
        assert secret not in outcome.reason, f"the model carried {secret!r} out through prove"
    # Still fed to the model, and still recorded for the operator.
    assert any(leaky in (c or "") for c in generator.calls)
    assert leaky in caplog.text


def test_the_undefined_term_exit_does_not_carry_the_source_out_either():
    """`assumed_terms` is model-authored, and its exit runs BEFORE the stated-reason one.

    A model that echoes its feedback into the term list rather than into `reason` leaves
    through here, twelve lines earlier. Same rule, same provenance check (M94).
    """
    from mnemiq.generate.generator import SqlProposal
    from mnemiq.generate.plan_query import Feedback

    leaky = 'permission denied for table hr_prod.payroll_salary; postgresql://svc:pw@10.2.0.7/x'

    class _EchoesIntoTerms:
        def propose(self, packet, feedback=None, strategy=None):
            return SqlProposal(sql=None, reason="n/a", assumed_terms=[feedback or "x"])

    outcome = plan_query(_packet(), _snapshot(), _GRANTS, _EchoesIntoTerms(), target="duckdb",
                         feedback=Feedback(leaky, from_source=True), max_attempts=1,
                         guard_undefined_terms=True)

    assert isinstance(outcome, Deferred)
    for secret in ("payroll_salary", "hr_prod", "10.2.0.7", "postgresql://"):
        assert secret not in outcome.reason, f"the term list carried {secret!r} out"


def test_a_term_the_source_never_spoke_into_is_still_named():
    """The narrowing stays as narrow as the risk: naming the term is the point of this deferral."""
    from mnemiq.generate.generator import SqlProposal
    from mnemiq.generate.plan_query import Feedback

    class _Declares:
        def propose(self, packet, feedback=None, strategy=None):
            return SqlProposal(sql=None, reason="n/a", assumed_terms=["lifetime value"])

    outcome = plan_query(_packet(), _snapshot(), _GRANTS, _Declares(), target="duckdb",
                         feedback=Feedback("SELECT * is not allowed.", from_source=False),
                         max_attempts=1, guard_undefined_terms=True)

    assert isinstance(outcome, Deferred)
    assert "lifetime value" in outcome.reason


def test_the_unauthorized_table_exit_does_not_carry_the_source_out_either(caplog):
    """`subject` is read off the model's own SQL, so it is model-authored like the rest.

    Third exit of the same kind. The caller normally SHOULD be told which table it lacks --
    that is what makes the refusal actionable -- so this is suppressed only on a turn fed the
    source's words, and the name still reaches the operator (M94).
    """
    from mnemiq.generate.generator import SqlProposal
    from mnemiq.generate.plan_query import Feedback

    leaky = 'permission denied for table hr_prod.payroll_salary; postgresql://svc:pw@10.2.0.7/x'

    class _NamesItAsATable:
        def propose(self, packet, feedback=None, strategy=None):
            return SqlProposal(sql=f'SELECT claim_identifier FROM "{feedback}"')

    with caplog.at_level(logging.WARNING, logger="mnemiq.generate.plan_query"):
        outcome = plan_query(_packet(), _snapshot(), _GRANTS, _NamesItAsATable(), target="duckdb",
                             feedback=Feedback(leaky, from_source=True), max_attempts=1)

    assert isinstance(outcome, Deferred)
    for secret in ("payroll_salary", "hr_prod", "10.2.0.7", "postgresql://"):
        assert secret not in outcome.reason, f"the refused subject carried {secret!r} out"
    assert leaky in caplog.text, "the operator still has to be able to see what was refused"


def test_a_table_the_source_never_named_is_still_named_to_the_caller():
    """A caller told which table it lacks can ask for it; one told nothing cannot."""
    from mnemiq.generate.generator import SqlProposal
    from mnemiq.generate.plan_query import Feedback

    class _AsksForPerson:
        def propose(self, packet, feedback=None, strategy=None):
            return SqlProposal(sql="SELECT last_name FROM person")

    outcome = plan_query(_packet(), _snapshot(), _GRANTS, _AsksForPerson(), target="duckdb",
                         feedback=Feedback("SELECT * is not allowed.", from_source=False),
                         max_attempts=1)

    assert isinstance(outcome, Deferred)
    assert "person" in outcome.reason


# --------------------------------------------------------------------------------------------
# M98. `REPAIRABLE` classified ten codes and nothing read it: every refusal but
# UNAUTHORIZED_TABLE was retried, so an unrepairable one cost `max_attempts` model calls to reach
# the verdict the first one already had, and arrived as INVALID_QUERY -- "could not produce a
# valid query after 3 attempts", claiming an attempt that could not have worked.
# --------------------------------------------------------------------------------------------


def _pii_snapshot() -> Snapshot:
    """A column the identity's clearance denies, so `check_cls` refuses on it."""
    return Snapshot(
        version="v1", source_id="acme", created_at="2026-07-13T00:00:00Z",
        columns=[
            Column(id="claim.claim_identifier", object_id="claim", name="claim_identifier"),
            Column(id="claim.ssn", object_id="claim", name="ssn", pii_level="direct"),
        ],
    )


def test_a_denied_column_is_not_a_puzzle_to_solve_with_another_attempt():
    """The same argument the table branch makes, on the code beside it. Retrying spent three
    calls inviting the model to find another route to a column this identity may not read, and
    a route it FINDS answers a subtly different question than the one that was asked."""
    from mnemiq.contract import DeferralReason

    generator = FakeGenerator([
        '{"sql": "SELECT ssn FROM claim"}',
        '{"sql": "SELECT claim_identifier FROM claim"}',   # must never be reached
    ])
    outcome = plan_query(_packet(), _pii_snapshot(), _GRANTS, generator, target="duckdb")

    assert isinstance(outcome, Deferred)
    assert outcome.code is DeferralReason.AUTHORIZATION
    assert len(generator.calls) == 1
    assert "3 attempts" not in outcome.reason, "it did not make three attempts"


def test_a_source_the_engine_cannot_govern_is_not_retried_either():
    """The default arm of the mapping, and the reason the new code exists. A source that
    redefines a builtin's name cannot be decided at all, so no rewrite is a rewrite of anything
    -- and the refusal names what an OPERATOR must change, which three retries buried under a
    count of attempts that were never made."""
    import os
    import tempfile

    import duckdb

    from mnemiq.adapters.duckdb import DuckDBAdapter
    from mnemiq.contract import DeferralReason

    path = os.path.join(tempfile.mkdtemp(), "shadow.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(claim_identifier INTEGER, status VARCHAR)")
    con.execute("CREATE MACRO count_star() AS (SELECT 1)")
    con.close()

    generator = FakeGenerator([
        '{"sql": "SELECT count(*) AS n FROM claim"}',
        '{"sql": "SELECT claim_identifier FROM claim"}',   # must never be reached
    ])
    outcome = plan_query(_packet(), _snapshot(), _GRANTS, generator,
                         adapter=DuckDBAdapter.duckdb(path), dialect="duckdb", target="duckdb")

    assert isinstance(outcome, Deferred)
    assert outcome.code is DeferralReason.UNGOVERNABLE
    assert len(generator.calls) == 1
    assert "count_star" in outcome.reason, "the sentence an operator needs, not a retry count"


def test_a_repairable_refusal_is_still_retried():
    """The control, and the half that could quietly break. Reading `REPAIRABLE` where nothing
    read it before turns every membership decision into behaviour, and a code wrongly left out
    of the set now costs an answer rather than a retry."""
    outcome, generator = _plan([
        '{"sql": "SELECT * FROM claim"}',
        '{"sql": "SELECT claim_identifier FROM claim"}',
    ])
    assert isinstance(outcome, Approved) and len(generator.calls) == 2


def test_a_denied_columns_name_is_withheld_once_source_words_are_in_play():
    """`check_cls` reads the subject off `exp.Column` in the model's own SQL, so a model handed
    the source's words can name a "column" spelled out of them -- the same provenance as the
    table branch's subject, and the same suppression. The generic path would otherwise print
    whatever the model wrote.

    The seeded `Feedback` is how the agent hands back what the DATABASE said, which is the way
    source words enter this loop in production.
    """
    from mnemiq.contract import DeferralReason
    from mnemiq.generate.plan_query import Feedback

    generator = FakeGenerator(['{"sql": "SELECT ssn FROM claim"}'])
    outcome = plan_query(_packet(), _pii_snapshot(), _GRANTS, generator, target="duckdb",
                         feedback=Feedback("the source said: no column 'ssn'", from_source=True))

    assert isinstance(outcome, Deferred) and outcome.code is DeferralReason.AUTHORIZATION
    assert "ssn" not in outcome.reason, "the subject is model-authored and source words are live"

    # ...and it IS named when nothing source-derived reached the model, because a caller told
    # which column they lack can ask for it. Withholding it always would cost that for nothing.
    plain = FakeGenerator(['{"sql": "SELECT ssn FROM claim"}'])
    named = plan_query(_packet(), _pii_snapshot(), _GRANTS, plain, target="duckdb")
    assert "ssn" in named.reason


# The loop's contract, one row per refusal code, written out INDEPENDENTLY of `REPAIRABLE`.
# Reading the set here would make the test agree with the code by construction, which is exactly
# what it must not do: the set's membership only became behaviour when `plan_query` started
# reading it (M98), and until then a wrong entry cost nothing and so was never checked. Deleting
# `UNGOVERNED_VIEW` from the set has to fail something.
#
# True = the model is asked again. Otherwise the DeferralReason the caller is answered with at
# once, because "it deferred" is not the claim -- routing a code to the wrong reason sends the
# operator to the wrong place, and `policy_unavailable` over an invalid row filter would tell
# them the policy could not be read when it was read and one predicate in it is invalid.
_RETRIED = {
    # The model's own SQL, and the message says what to change.
    RefusalCode.PARSE_ERROR: True,
    RefusalCode.NOT_A_SINGLE_STATEMENT: True,
    RefusalCode.NOT_SELECT_ONLY: True,
    RefusalCode.SELECT_STAR: True,
    RefusalCode.UNKNOWN_TABLE: True,
    RefusalCode.UNKNOWN_COLUMN: True,
    RefusalCode.EXPLAIN_FAILED: True,
    RefusalCode.UNMODELLED_CALL: True,
    RefusalCode.LOGIC_LINT: True,
    RefusalCode.VALUE_GROUNDING: True,
    # "may only be selected, not used in a filter" / "Query the table directly." Both name a
    # different query, which is the test for belonging here.
    RefusalCode.MASKED_COLUMN_IN_PREDICATE: True,
    RefusalCode.UNGOVERNED_VIEW: True,
    # The weakest of the three: the model chose to use the view, and the messages name it
    # without saying to avoid it. Retried because the safe direction changes no behaviour.
    RefusalCode.UNRESOLVABLE_VIEW: True,
    # Grants. A guard that can be retried is a puzzle, not a guard, and a route the model FINDS
    # answers a subtly different question than the one that was asked.
    RefusalCode.UNAUTHORIZED_TABLE: DeferralReason.AUTHORIZATION,
    RefusalCode.UNAUTHORIZED_COLUMN: DeferralReason.AUTHORIZATION,
    # Properties of the source or the policy. No rewrite is a rewrite of anything.
    RefusalCode.UNRESOLVABLE_CALLS: DeferralReason.UNGOVERNABLE,
    RefusalCode.VIEW_INVENTORY_UNAVAILABLE: DeferralReason.UNGOVERNABLE,
    RefusalCode.INVALID_ROW_FILTER: DeferralReason.UNGOVERNABLE,
    # Write-path codes. `decide_write` is called from the runtime, not through this loop, so
    # none of these can arrive here -- listed so a future caller that routes writes through
    # `plan_query` finds a decision already made rather than a default.
    RefusalCode.NOT_A_WRITE: DeferralReason.UNGOVERNABLE,
    RefusalCode.UNBOUNDED_WRITE: DeferralReason.UNGOVERNABLE,
    RefusalCode.UNAUTHORIZED_WRITE: DeferralReason.UNGOVERNABLE,
    RefusalCode.WRITES_DISABLED: DeferralReason.UNGOVERNABLE,
    RefusalCode.AMBIGUOUS_WRITE_TARGET: DeferralReason.UNGOVERNABLE,
    RefusalCode.UNSCOPED_CTE: DeferralReason.UNGOVERNABLE,
}


def _attempts_for(code, monkeypatch):
    """Drive the loop with a refusal of `code`, then an approval. Returns the generator calls.

    The refusal is INJECTED rather than provoked: what is under test is the loop's response to a
    code, not the twenty different fixtures it takes to make `decide` emit each one -- several of
    which (an invalid row filter, a self-referential view) need a snapshot shaped for that one
    case and would say nothing about the branch that reads them.
    """
    from mnemiq.sql.verdict import Refusal

    calls = {"n": 0}

    def fake_decide(sql, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return Refusal(code=code, message="refused for a test", subject="thing")
        return Approved(plan_sql=sql, target_sql=sql)

    monkeypatch.setattr("mnemiq.generate.plan_query.decide", fake_decide)
    generator = FakeGenerator(['{"sql": "SELECT claim_identifier FROM claim"}'] * 3)
    outcome = plan_query(_packet(), _snapshot(), _GRANTS, generator, target="duckdb")
    return outcome, generator.calls


@pytest.mark.parametrize("code, expected", sorted(_RETRIED.items(), key=lambda kv: kv[0].value))
def test_the_loop_retries_exactly_what_the_classification_says(code, expected, monkeypatch):
    outcome, calls = _attempts_for(code, monkeypatch)

    if expected is True:
        assert isinstance(outcome, Approved), f"{code.value} was not retried"
        assert len(calls) == 2
    else:
        assert isinstance(outcome, Deferred), f"{code.value} was retried"
        assert len(calls) == 1
        assert "attempts" not in outcome.reason, "it made one attempt, not three"
        assert outcome.code is expected, f"{code.value} deferred as {outcome.code}"


def test_every_refusal_code_has_a_row_above():
    """A new code otherwise inherits `not repairable` from the set's default and is answered at
    once, which is the safe direction and still a decision somebody should make on purpose."""
    assert set(_RETRIED) == set(RefusalCode), set(_RETRIED) ^ set(RefusalCode)


def test_the_denied_column_is_written_where_the_caller_cannot_see_it(caplog):
    """When the name is withheld from the caller, the deployment's log is the only place it is
    written. The table branch pins the same rule, which is what made its absence here a gap
    rather than a preference: an operator asked "why did this defer" has nothing otherwise.

    The server log is the right destination and deliberately so -- it already sits inside the
    boundary that holds the connection settings.
    """
    from mnemiq.generate.plan_query import Feedback

    generator = FakeGenerator(['{"sql": "SELECT ssn FROM claim"}'])
    with caplog.at_level(logging.WARNING):
        outcome = plan_query(
            _packet(), _pii_snapshot(), _GRANTS, generator, target="duckdb",
            feedback=Feedback("the source said: no column 'ssn'", from_source=True))

    assert "ssn" not in outcome.reason
    assert "ssn" in caplog.text, "the operator still has to be able to see what was refused"
    assert "unauthorized_column" in caplog.text


def test_a_source_fault_that_survives_every_attempt_is_still_a_source_fault():
    """The exhaustion exit is the one M98 was about, and retrying an override-marked refusal put
    it back: a catalogue call that fails on all three attempts used to leave as INVALID_QUERY --
    "could not produce a valid query after 3 attempts" over a query that was never the problem,
    with the sentence naming what an operator must fix demoted to a trailing clause.

    Both endings are pinned, because the retry only earns its cost if recovery is real: the
    source that answers on the second attempt gets an answer, and the one that never answers
    gets the same code it would have got without the attempts.
    """
    from mnemiq.contract import DeferralReason
    from mnemiq.sql.verdict import Refusal

    def blips(n_failures):
        calls = {"n": 0}

        def fake_decide(sql, *a, **kw):
            calls["n"] += 1
            if calls["n"] <= n_failures:
                return Refusal(code=RefusalCode.UNRESOLVABLE_CALLS,
                               message="This source could not say. Try again.",
                               repairable_override=True)
            return Approved(plan_sql=sql, target_sql=sql)
        return fake_decide

    import pytest as _pytest
    for failures, check in ((1, "recovers"), (9, "exhausts")):
        mp = _pytest.MonkeyPatch()
        mp.setattr("mnemiq.generate.plan_query.decide", blips(failures))
        generator = FakeGenerator(['{"sql": "SELECT claim_identifier FROM claim"}'] * 5)
        outcome = plan_query(_packet(), _snapshot(), _GRANTS, generator, target="duckdb")
        mp.undo()

        if check == "recovers":
            assert isinstance(outcome, Approved), "the retry has to be able to succeed"
            assert len(generator.calls) == 2
        else:
            assert isinstance(outcome, Deferred)
            assert outcome.code is DeferralReason.UNGOVERNABLE, outcome.code
            assert "attempts" not in outcome.reason
            assert len(generator.calls) == 3, "it did use the whole budget"
