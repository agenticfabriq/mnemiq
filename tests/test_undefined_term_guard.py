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


@pytest.mark.parametrize("spelling", ["total payments", "revenues", "Payment Dates"])
def test_a_plural_is_the_same_term(spelling):
    """The half of the mismatch that costs answers.

    Retrieval and this guard were reading the same words with two different matchers.
    `select_definitions` matches inflection-tolerantly -- `_term_pattern` appends `\\w*` to the
    last word, which is how "premium" retrieves on a question saying "premiums" -- while this check
    compared for exact equality. So retrieval puts `total payment` in the packet BECAUSE the model
    said "total payments", and the guard then reports "total payments" ungrounded and refuses.

    A false deferral on a defined term, sourced in a lexical detail rather than in anything about
    meaning. It matters beyond the one answer: the withdrawal rule for this guard is a measured
    over-declaration rate, so a matcher bug spending that budget retires a guard for a reason that
    was never about the design.
    """
    assert ungrounded_terms([spelling], _defs()) == []


def test_a_longer_phrase_is_not_grounded_by_the_term_it_contains():
    """The limit on that tolerance, and the reason it is a full match rather than a search.

    "revenue per customer" is a DERIVATION over a defined term, not the term -- exactly the M35
    shape, where every part is certified and the composition is not. Inflection tolerance that
    matched a substring would ground it and silence the guard on its own finding.
    """
    assert ungrounded_terms(["revenue per customer"], _defs()) == ["revenue per customer"]
    assert ungrounded_terms(["lifetime value of revenue"], _defs()) == ["lifetime value of revenue"]


def test_the_two_matchers_agree_on_the_corpus_they_both_read():
    """Stated against `select_definitions` itself, not against a restatement of it.

    The defect was not that either matcher was wrong; it was that two were reading one corpus and
    nobody had compared them. Pinning the agreement is what stops them drifting apart again -- the
    next edit to either has to keep a term retrieval finds groundable here.
    """
    from mnemiq.authz.grants import GrantSet
    from mnemiq.semantic.glossary import select_definitions

    # Bound and granted, which is how a policy definition is actually visible: `public` is False
    # by default and an unbound non-public definition is visible to no one, so `_defs()` as written
    # retrieves nothing and the comparison would pass on an empty set.
    bound = [d.model_copy(update={"bound_objects": ["fs.payments"]}) for d in _defs()]
    grants = GrantSet(objects=frozenset({"fs.payments"}))
    for spelling in ("total payments", "revenues", "payment dates"):
        retrieved = select_definitions(spelling, bound, grants)
        assert retrieved, f"retrieval finds a definition for {spelling!r}"
        assert ungrounded_terms([spelling], retrieved) == [], (
            f"retrieval matched {spelling!r} but the guard called it ungrounded"
        )


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
    """In deep mode a declared term must defer AS an undefined term, not as a disagreement.

    What this changes is the REASON, and the original framing of the finding was wrong about that.
    It said deep mode "returns the invented derivation with no trace of the guard having fired". It
    does not: real deep mode sets `min_agreement=0.6`, and selective answering defers whenever
    `len(executed) < candidates` -- so two candidates deferring already made it 3 of 5 and the
    answer was withheld either way. I accepted that framing without checking it.

    The guard was never inert; it was ILLEGIBLE. The caller was told "the candidates disagreed too
    much to answer confidently", which is false -- they did not disagree, two of them found the
    question unanswerable -- and it points at the one repair that cannot work. Now it says which
    term has no definition.

    That is also the difference beacon can measure: their grader reads `deferred` as a boolean and
    both paths defer, but the reason is persisted, and today every deferral in that corpus carries
    the identical string. A DISAGREEMENT here would be indistinguishable from a genuine one.

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
            # The decider's proof step calls `execute`, not `execute_arrow`. Without it candidates
            # 3-5 came back `Deferred(invalid_query, "'_Adapter' object has no attribute
            # 'execute'")` -- candidates 1-2 never got that far, because the guard fires before
            # `decide` -- so `executed` stayed empty and the branch this test exists for never ran.
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
        # Deep mode's real setting. Omitting it skipped selective answering entirely, so the test
        # exercised a configuration that does not ship and its rationale described a vote that
        # could never have happened.
        min_agreement=0.6,
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

    assert answer.deferred is True
    assert answer.reason_code == DeferralReason.UNDEFINED_TERM
    assert "lifetime value" in answer.answer
    assert generator.calls == 5, "every candidate still runs; the guard decides after, not by short-circuit"
    assert answer.candidates_executed == 3, "three produced a table; the adapter must reach `execute`"
    assert "disagreed" not in answer.answer, (
        "selective answering would defer this as DISAGREEMENT -- true of the candidate set and "
        "false of the question, and pointing the caller at a repair that cannot work"
    )


def test_guided_mode_can_emit_the_declaration_at_all():
    """A guided reply is constrained to `_GUIDED_SQL_SCHEMA`, and grammar backends compile over the
    DECLARED properties only -- so a property missing from the schema is one the model cannot emit.

    Inert in a way no parser test could see: the parser handles `assumed_terms` correctly and would
    simply never receive it, on exactly the local-model deployments `MNEMIQ_GUIDED_SQL` exists for.
    Asserted against the schema the request actually carries, because that is the artefact that
    decides what can come back.
    """
    from mnemiq.generate.generator import _GUIDED_SQL_SCHEMA, _guided_extra_body

    assert "assumed_terms" in _GUIDED_SQL_SCHEMA["properties"], (
        "the model cannot declare a term the guided schema does not allow"
    )

    body = _guided_extra_body(True)
    schema = body["response_format"]["json_schema"]["schema"]
    assert "assumed_terms" in schema["properties"], "the sent schema must carry it, not just ours"
    assert "assumed_terms" not in schema.get("required", []), (
        "declaring nothing is the common case and must stay valid"
    )


def test_the_prompt_asks_for_the_field_the_parser_reads():
    """The prompt names the key and the parser reads it, and nothing ties the two together. Rename
    one and the guard goes silently inert -- the model sends a field nobody reads, or the parser
    waits for a field nobody asked for. The same hand-enumerated-allowlist shape that caused the
    finding this guard exists for."""
    import inspect

    from mnemiq.generate import generator, prompts

    assert "assumed_terms" in inspect.getsource(prompts), "the prompt must ask for it"
    assert "assumed_terms" in inspect.getsource(generator), "the parser must read the same key"
