"""M35 — the engine invents a definition for a term nobody certified.

Beacon's `fs_payments_v1` refusal band: the certified arm refuses 26 of 30, and all four failures
are one item -- `fs-refuse-ltv`, "What is the lifetime value of our average customer?". The grader
is one rule (`if answerable is False: PASS if deferred else FAIL`) and never inspects the reason,
so FAIL means exactly one thing: it did not defer.

Verified from the corpus rather than the register's summary: `policy_records.json` carries five
certified definitions -- payment_date, payment_identity, revenue, total_payment, payment_revisions
-- and NO lifetime-value definition. So the model has `revenue` and `total_payment` to hand and
invents a derivation from them.

The four refusal items that DO refuse every time each name a COLUMN that does not exist. The guard
that holds is "no such column"; there is no guard for "no such definition". That is the finding.

The rule here: a term the model says it had to assume, with no certified definition behind it, is a
deferral naming the term -- not a confident answer. Grounded-or-bare, the pattern this codebase
already runs for code meanings: the model supplies candidates, the ENGINE keeps the decision.
Deliberately not a term list, because an unlisted term passes and that is the shape `views.py`
documents as unfixable.
"""


import pytest

from mnemiq.contract.seams import DeferralReason
from mnemiq.contract.semantic import Definition
from mnemiq.generate.generator import SqlProposal, _parse
from mnemiq.generate.undefined_terms import ungrounded_terms


def _defs():
    """The fs_payments certified set, as `policy_records.json` actually carries it.

    The REAL `Definition`, not a fake. The first version of this file used a stand-in with an
    `object_id` attribute, which the model does not have -- so the code read a field that never
    existed, the id branch was dead in production, and every test here passed anyway. A fixture
    shaped to the code cannot falsify the code.
    """
    return [
        Definition(id="fspay:policy:revenue", term="revenue", domain="fspay",
                   definition="recognised revenue"),
        Definition(id="fspay:policy:total_payment", term="total payment", domain="fspay",
                   definition="sum of settled payments"),
        Definition(id="fspay:policy:payment_date", term="payment date", domain="fspay",
                   definition="the settlement date"),
    ]


# -- the parser carries the model's own declaration ------------------------------------------------


def test_the_proposal_carries_the_terms_the_model_had_to_assume():
    """No new model call: the generator already reads a structured JSON channel, so the declaration
    rides in the reply that produces the SQL. A separate call would cost a round trip per question
    to ask the same model about the question it just read."""
    proposal = _parse('{"sql": "SELECT 1", "reason": "", "assumed_terms": ["lifetime value"]}')
    assert proposal.sql == "SELECT 1"
    assert proposal.assumed_terms == ["lifetime value"]


def test_a_reply_without_the_field_assumes_nothing():
    """Every existing prompt and every test double omits it. Absent must mean "declared nothing",
    not "unknown" -- a missing field that deferred every question would break the engine rather
    than guard it."""
    assert _parse('{"sql": "SELECT 1", "reason": ""}').assumed_terms == []


# -- the engine keeps the decision -----------------------------------------------------------------


def test_a_term_with_no_certified_definition_is_ungrounded():
    """`lifetime value` against the real fs_payments definition set."""
    assert ungrounded_terms(["lifetime value"], _defs()) == ["lifetime value"]


def test_a_term_that_is_defined_is_grounded():
    """The guard must not fire on the terms a glossary exists for, or it refuses the questions the
    product is for."""
    assert ungrounded_terms(["revenue"], _defs()) == []


@pytest.mark.parametrize("spelling", ["Revenue", "  revenue  ", "REVENUE"])
def test_matching_ignores_case_and_padding(spelling):
    """The model writes prose. A guard that fires on `Revenue` because the record says `revenue`
    would defer a defined term, which is the opposite failure and the more damaging one."""
    assert ungrounded_terms([spelling], _defs()) == []


def test_a_namespaced_id_is_not_a_spelling_a_model_would_use():
    """The tail of an id is a spelling; the whole namespaced id is not, and matching on the whole
    would leave those definitions unmatched.

    Asserted with a definition whose TERM cannot also satisfy it. The earlier version used
    `total payment`, which the fixture's `id="fspay:policy:total_payment"` yields on its own via
    the tail -- so it passed through either path and could not have caught the `term` lookup
    regressing, while its docstring claimed to be testing exactly that.
    """
    from mnemiq.contract.semantic import Definition

    defs = [Definition(id="fspay:policy:x9", term="settled volume", domain="fspay",
                       definition="volume of settled payments")]
    assert ungrounded_terms(["settled volume"], defs) == [], "the term must match"
    assert ungrounded_terms(["x9"], defs) == [], "the id's tail is a spelling too"
    assert ungrounded_terms(["fspay:policy:x9"], defs) == ["fspay:policy:x9"], (
        "a namespaced id is not something a model writes"
    )


def test_the_guard_names_every_ungrounded_term_not_just_the_first():
    """The refusal is a repair instruction: a caller told about one missing definition fixes one
    and comes back. Naming all of them is the difference between one round trip and three."""
    assert ungrounded_terms(["lifetime value", "revenue", "churn rate"], _defs()) == [
        "lifetime value",
        "churn rate",
    ]


def test_declaring_nothing_grounds_nothing_and_refuses_nothing():
    """The common case by far, and the one that must stay free."""
    assert ungrounded_terms([], _defs()) == []


def test_no_definitions_at_all_still_refuses_a_declared_term():
    """An un-enriched or ablated deployment has no certified set. The register measured the ablated
    arm refusing 0 of 4 -- with no definitions, EVERY assumed term is ungrounded, which is the
    honest answer rather than a reason to skip the check."""
    assert ungrounded_terms(["lifetime value"], []) == ["lifetime value"]


def test_the_deferral_carries_its_own_code():
    """A separate code on M6's OWN test -- what should the caller do next?

    UNANSWERABLE says "rephrase, these tables cannot answer that". This says "the tables can answer
    it and nobody has said what the term MEANS", where the fix is external and specific -- closer
    in shape to AUTHORIZATION's "request a grant" than to trying different words. Rephrasing cannot
    help, so collapsing them would tell a caller to do the one thing that will not work.

    An earlier draft of this test asserted UNANSWERABLE on M6 grounds and was wrong for M6's own
    reason. It is also what makes the guard measurable: beacon's grader reads `deferred` as a
    boolean, but the reason is persisted -- and today every deferral in that corpus carries the
    identical string `unanswerable`, so a pass cannot be told from luck.
    """
    assert DeferralReason.UNDEFINED_TERM == "undefined_term"
    assert DeferralReason.UNDEFINED_TERM != DeferralReason.UNANSWERABLE


def test_a_proposal_is_still_a_proposal_when_it_declares_terms():
    """Non-vacuity: the field must not turn a good answer into a deferral by itself. The SQL is
    still there; only an UNGROUNDED term stops it."""
    proposal = SqlProposal(sql="SELECT 1", reason="", assumed_terms=["revenue"])
    assert proposal.sql == "SELECT 1"
    assert ungrounded_terms(proposal.assumed_terms, _defs()) == []


def test_the_planner_defers_on_an_ungrounded_term_before_deciding_the_sql():
    """End to end through `plan_query`, because the unit pieces passing is not the claim.

    Checked BEFORE the SQL is decided, and that ordering is the finding: the SQL is exactly what
    makes an invented derivation look answerable. It parses, it runs, it returned four rows in
    beacon's failing arm. A guard placed after the decider would be asking a valid query whether it
    means anything.
    """
    from mnemiq.authz.grants import GrantSet
    from mnemiq.generate.plan_query import Deferred, plan_query
    from mnemiq.semantic.retrieval import ContextPacket

    class _Gen:
        def propose(self, packet, feedback=None):
            # A confident, VALID query -- which is what the failing runs actually produced.
            return SqlProposal(
                sql="SELECT 1",
                reason="",
                assumed_terms=["lifetime value"],
            )

    from mnemiq.contract import Column, Snapshot

    snapshot = Snapshot(
        version="v1",
        source_id="fs_payments",
        created_at="t",
        columns=[Column(id="payment.amount", object_id="payment", name="amount")],
    )


    packet = ContextPacket(
        question="What is the lifetime value of our average customer?",
        cards=[object()],  # non-empty: the NO_TABLES guard is not what is under test
        grant_fingerprint="gf",
        enrichment_version="v1",
        definitions=list(_defs()),
    )

    out = plan_query(packet, snapshot, GrantSet(frozenset({"payment"})), _Gen())
    assert isinstance(out, Deferred), f"a plausible derivation still answered: {out}"
    assert out.code == DeferralReason.UNDEFINED_TERM
    assert "lifetime value" in out.reason


def test_a_definition_is_matched_by_its_id_when_that_is_how_a_corpus_spells_the_term():
    """The branch that was dead in production. Some corpora carry `loss_ratio` as the id and the
    term a model writes is "loss ratio" -- so a bare id, underscores normalised, is a spelling.

    Asserted against the REAL `Definition`. The earlier fake carried `object_id`, the code read
    `object_id`, and both were wrong together: a test and its subject sharing a mistake cannot
    detect it.
    """
    from mnemiq.contract.semantic import Definition

    defs = [Definition(id="loss_ratio", term="", domain="ins", definition="claims over premium")]
    assert ungrounded_terms(["loss ratio"], defs) == []
    assert ungrounded_terms(["lifetime value"], defs) == ["lifetime value"]


def test_deep_mode_does_not_outvote_the_guard():
    """Deep mode ran five candidates, dropped every non-Approved outcome, and let the survivors
    vote -- so two candidates declaring "lifetime value" deferred, three answered from a guessed
    meaning, and the majority won. The guard was in force on instant and thinking and inert on the
    mode a caller reaches for precisely when the question is hard.

    The asymmetry is why one declaration settles it rather than getting a vote: a candidate
    DECLARING the term is evidence the question names an undefined one; the others not declaring it
    is not evidence against -- they simply did not say.

    RUNS THE AGENT. The first version of this test installed a fake over `plan_query`, never called
    it, and asserted on the SOURCE TEXT of `_answer_consistent` -- so the behaviour change shipped
    with no test that executed it, and the file's own `_defs` docstring two screens up says exactly
    why that does not count. It also encoded a contract the code contradicts, raising on the third
    candidate when the code deliberately runs all of them, so the first maintainer to make it real
    would have got a failure on correct code and "fixed" the code to match the test.
    """
    import pyarrow as pa

    from mnemiq.agent.budget import Budget
    from mnemiq.agent.loop import Agent
    from mnemiq.agent.synthesize import FakeSynthesizer
    from mnemiq.authz.grants import GrantSet
    from mnemiq.cache.store import L1Cache, TwoTierCache
    from mnemiq.contract import Column, IdentityContext, Snapshot
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    class _MixedGenerator:
        """Two candidates declare the undefined term; the rest answer confidently."""

        def __init__(self):
            self.calls = 0

        def propose(self, packet, feedback=None, strategy=None):
            #  because deep mode wraps each candidate in StrategyGenerator, which passes
            # it through -- a real Agent run finds that; a hand-rolled fake over plan_query did not.
            self.calls += 1
            if self.calls <= 2:
                return SqlProposal(sql="SELECT 1", reason="", assumed_terms=["lifetime value"])
            return SqlProposal(sql="SELECT 1", reason="", assumed_terms=[])

    class _Adapter:
        dialect = "duckdb"

        def execute(self, sql):
            # The decider's proof step calls `execute`, not `execute_arrow`. Without it every
            # candidate came back `Deferred(invalid_query, "'_Adapter' object has no attribute
            # 'execute'")`, `executed` stayed empty, and the branch this test exists for -- the
            # early return taken while a MAJORITY is available to outvote the guard -- never ran.
            # The test passed and documented a run that did not happen.
            return []

        def execute_arrow(self, sql, timeout_s=30):
            return pa.table({"n": [1]})

    generator = _MixedGenerator()
    agent = Agent(
        generator=generator,
        synthesizer=FakeSynthesizer(),
        adapter=_Adapter(),
        cache=TwoTierCache(L1Cache()),
        budget=Budget(wall_clock_s=5.0, max_attempts=1),
        candidates=5,
    )
    packet = ContextPacket(
        question="What is the lifetime value of our average customer?",
        cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint="f",
        enrichment_version="v1",
        definitions=_defs(),
    )
    snapshot = Snapshot(version="v1", source_id="fs_payments", created_at="t",
                        columns=[Column(id="claim.n", object_id="claim", name="n")])
    answer = agent.answer(packet, snapshot, GrantSet(frozenset({"claim"})),
                          IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"]))

    assert answer.deferred is True, "three silent candidates outvoted two that declared the term"
    assert answer.reason_code == DeferralReason.UNDEFINED_TERM
    assert "lifetime value" in answer.answer
    assert generator.calls == 5, "every candidate still runs; the guard decides after, not by short-circuit"
    assert answer.candidates_executed == 3, (
        "three candidates produced a table and were available to outvote the guard -- which is the "
        "branch this test exists for, and which never ran while the adapter lacked `execute`"
    )
